"""Content moderation of what is sent to cloud models.

Calls an OpenAI-compatible ``POST /v1/moderations`` service (the local one of
local-multimodal-infra) set by the system setting ``content_moderation_url``;
an empty setting turns moderation off. Each platform instance chooses how its
messages are checked with its ``content_moderation`` setting:

- ``enabled``: text and images;
- ``text_only``: text only, images go unchecked;
- ``disabled``: nothing.

Only the categories of a blocked input are logged, never its text.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence

import aiohttp

from astrbot.core import astrbot_config, logger

MODE_ENABLED = "enabled"
MODE_TEXT_ONLY = "text_only"
MODE_DISABLED = "disabled"
#: Replied instead of an answer when the message was blocked, unless the
#: system setting content_moderation_blocked_reply says otherwise.
BLOCKED_REPLY = "这条内容无法处理。"

TEXT_TIMEOUT_S = 5.0
IMAGE_TIMEOUT_S = 15.0
#: Tool results can be long (a whole file, a web page): one deadline for all.
TOOL_RESULT_TIMEOUT_S = 15.0
#: Text per request, in UTF-8 bytes. The service checks one request at a
#: time and a check costs about its tokens: 16 KB takes at most ~1.3 s even
#: for token-dense text (digits, punctuation), so no request holds up the
#: others for long. Longer texts are checked in pieces of this size.
CHUNK_BYTES = 16 * 1024
#: Bytes a piece repeats from the end of the one before it.
OVERLAP_BYTES = 512
#: Text of one tool result that is checked (in pieces); the rest is cut off,
#: so padding a result cannot outlast its check.
MAX_TOOL_TEXT_BYTES = 64 * 1024
#: Image data per request, well under the service's 64 MB.
MAX_REQUEST_IMAGE_BYTES = 24 * 1024 * 1024
#: Image types the service reads.
SERVICE_IMAGE_TYPES = (
    "image/png",
    "image/jpeg",
    "image/bmp",
    "image/webp",
    "image/gif",
)
#: The service takes at most this many images per request.
MAX_IMAGES_PER_REQUEST = 16
#: History messages checked per request.
TEXT_BATCH = 32
#: Background checks (group messages as they arrive) sent at once, so a busy
#: group does not flood the service. Checks a request waits on bypass it.
MAX_IN_FLIGHT = 4
MAX_IMAGE_BYTES = 20 * 1024 * 1024
UNAVAILABLE = "moderation_unavailable"
INVALID_INPUT = "invalid_input"
#: A tool result part withheld because it could not be checked.
UNCHECKABLE = "uncheckable"
#: Event extra: images the user sent as file attachments, checked like images.
ATTACHMENT_IMAGES_EXTRA = "_moderation_attachment_images"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

_IMAGE_SIGNATURES = (
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"BM", "image/bmp"),
)

#: name -> (event loop, semaphore): see ``_semaphore``.
_semaphores: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Semaphore]] = {}


def _semaphore(name: str, size: int) -> asyncio.Semaphore:
    """A semaphore of the running event loop (tests run several loops)."""
    loop = asyncio.get_running_loop()
    held = _semaphores.get(name)
    if held is None or held[0] is not loop:
        held = (loop, asyncio.Semaphore(size))
        _semaphores[name] = held
    return held[1]


def _utf8_len(text: str) -> int:
    # surrogatepass: a lone surrogate (from JSON) must not raise.
    return len(text.encode("utf-8", "surrogatepass"))


def cut_to_bytes(text: str, max_bytes: int) -> str:
    """``text`` cut to at most ``max_bytes`` of UTF-8, on a character."""
    data = text.encode("utf-8", "surrogatepass")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", "ignore")


def _pieces(text: str, max_bytes: int) -> list[str]:
    """``text`` in pieces of at most ``max_bytes`` of UTF-8 (``[text]`` when
    it is short or empty). Each piece starts with the end of the one before
    (``OVERLAP_BYTES``, at most half a piece), so a phrase cut at a boundary
    is still whole in one of them."""
    overlap = min(OVERLAP_BYTES, max_bytes // 2)
    sizes = [_utf8_len(char) for char in text]
    pieces: list[str] = []
    start = used = 0
    for index, size in enumerate(sizes):
        if used + size > max_bytes and index > start:
            pieces.append(text[start:index])
            # Step back over the overlap, but always move forward.
            back, carried = index, 0
            while (
                back - 1 > start
                and carried + sizes[back - 1] <= overlap
                and carried + sizes[back - 1] + size <= max_bytes
            ):
                back -= 1
                carried += sizes[back]
            start, used = back, carried
        used += size
    pieces.append(text[start:])
    return pieces


def blocked_reply() -> str:
    """What a blocked message is answered with (the system setting, or
    ``BLOCKED_REPLY`` when it is empty)."""
    reply = str(astrbot_config.get("content_moderation_blocked_reply") or "").strip()
    return reply or BLOCKED_REPLY


def service_url() -> str:
    """The moderation service's ``/v1/moderations`` endpoint, or "" when off."""
    base = str(astrbot_config.get("content_moderation_url") or "").strip()
    if not base:
        return ""
    base = base.rstrip("/")
    return base if base.endswith("/v1/moderations") else f"{base}/v1/moderations"


def check_options() -> dict:
    """What the system settings tell the service about flagging.

    ``content_moderation_threshold``: a text is flagged when
    ``1 - p(safe)`` exceeds it (the service's default 0.9; lower is
    stricter). ``content_moderation_nsfw_threshold``: an image is flagged
    when its NSFW probability exceeds it (default 0.5; lower is stricter).
    A value the service would reject (outside 0 <= x < 1) is left out, so
    the service's default applies. ``content_moderation_nsfw_labels``: the
    image classifier's labels that count as NSFW (``low`` suggestive,
    ``medium``, ``high`` explicit); none selected keeps the service's.
    ``content_moderation_categories``: the text categories that flag (text
    judged unsafe only for others passes); none selected means all.

    Returns:
        The request fields to send.
    """
    fields: dict = {}
    for setting, field in (
        ("content_moderation_threshold", "threshold"),
        ("content_moderation_nsfw_threshold", "nsfw_threshold"),
    ):
        value = astrbot_config.get(setting)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = -1.0
        if 0.0 <= number < 1.0:
            fields[field] = number
        else:
            logger.warning(
                "Ignoring %s=%r: it must be at least 0 and below 1.", setting, value
            )
    for setting, field in (
        ("content_moderation_nsfw_labels", "nsfw_labels"),
        ("content_moderation_categories", "categories"),
    ):
        names = astrbot_config.get(setting)
        if isinstance(names, list):
            names = [str(name).strip() for name in names if str(name).strip()]
            if names:
                fields[field] = names
    return fields


def platform_mode(platform_id: str) -> str:
    """How messages of a platform instance are checked.

    Args:
        platform_id: Id of the platform instance (the first part of a UMO);
            empty for input not tied to a chat, e.g. a plugin's own model call.

    Returns:
        ``enabled``, ``text_only`` or ``disabled``. Platforms without the
        setting, and input of no known platform, are checked fully.
    """
    if not service_url():
        return MODE_DISABLED
    for platform in astrbot_config.get("platform") or []:
        if isinstance(platform, dict) and platform.get("id") == platform_id:
            mode = platform.get("content_moderation") or MODE_ENABLED
            return mode if mode in (MODE_TEXT_ONLY, MODE_DISABLED) else MODE_ENABLED
    return MODE_ENABLED


def _data_url(data: bytes, mime: str = "") -> str:
    mime = mime.split(";", 1)[0].strip().lower()
    if not mime.startswith("image/"):
        mime = next(
            (m for sig, m in _IMAGE_SIGNATURES if data.startswith(sig)),
            "image/webp" if data[8:12] == b"WEBP" else "image/png",
        )
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


async def image_data_url(ref: str, session: aiohttp.ClientSession) -> str:
    """An image as the base64 data URL the service takes.

    Args:
        ref: A data URL, an http(s) URL (downloaded: the service fetches
            nothing itself), or a local path (optionally a ``file://`` URL).
        session: Session for downloads.

    Returns:
        The data URL.

    Raises:
        Exception: The image could not be read or downloaded.
    """
    if ref.startswith("data:"):
        return ref
    if ref.startswith("base64://"):
        payload = ref[len("base64://") :]
        if len(payload) * 3 // 4 - payload.count("=") > MAX_IMAGE_BYTES:
            raise ValueError("image too large")
        return _data_url(base64.b64decode(payload))
    if ref.startswith(("http://", "https://")):
        async with session.get(ref) as resp:
            resp.raise_for_status()
            data = await resp.content.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("image too large")
            return _data_url(data, resp.headers.get("Content-Type", ""))
    if ref.startswith("file://"):
        ref = ref[len("file://") :]
        # file:///C:/x on Windows; file:///home/x stays absolute elsewhere.
        if len(ref) > 2 and ref[0] == "/" and ref[2] == ":":
            ref = ref[1:]

    # A file sent as an attachment can be gigabytes: never read it whole.
    def read() -> bytes:
        with open(ref, "rb") as file:
            return file.read(MAX_IMAGE_BYTES + 1)

    data = await asyncio.to_thread(read)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image too large")
    return _data_url(data)


async def _post(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    timeout: float,
    background: bool = False,
) -> list[dict] | int | None:
    """One moderation request.

    Args:
        session: Session to send it with.
        url: The service endpoint.
        payload: The request body.
        timeout: Seconds the request may take once sent.
        background: A check nothing waits on: it first queues for one of
            ``MAX_IN_FLIGHT`` slots, which ``timeout`` does not count.

    Returns:
        The results, one per input; 400 when the service rejects the input;
        None when it cannot answer (timeout, 5xx, misconfigured, malformed).
    """

    payload = {**payload, **check_options()}

    # A service with inference tokens configured wants one.
    token = str(astrbot_config.get("content_moderation_token") or "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else None

    async def send() -> tuple[int, object]:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                return resp.status, None
            return 200, await resp.json()

    try:
        if background:
            async with _semaphore("background", MAX_IN_FLIGHT):
                status, body = await asyncio.wait_for(send(), timeout)
        else:
            status, body = await asyncio.wait_for(send(), timeout)
    except Exception as e:  # noqa: BLE001 - timeout, connection refused, bad JSON
        detail = f": {e}" if str(e) else ""
        logger.warning(
            "Content moderation service unavailable: %s%s", type(e).__name__, detail
        )
        return None
    if status in (400, 413):
        return 400
    if status != 200:
        log = logger.error if status in (401, 403, 404, 405) else logger.warning
        log("Content moderation service answered HTTP %d (%s).", status, url)
        return None
    results = body.get("results") if isinstance(body, dict) else None
    if (
        not isinstance(results, list)
        or not results
        or not all(isinstance(r, dict) for r in results)
    ):
        logger.warning("Content moderation service gave an unexpected answer.")
        return None
    return results


def _flagged_categories(result: dict) -> list[str]:
    if not result.get("flagged"):
        return []
    categories = result.get("categories")
    if not isinstance(categories, dict):
        return ["flagged"]
    return [str(k) for k, v in categories.items() if v] or ["flagged"]


async def check_message(
    texts: Sequence[str], images: Sequence[str], label: str = ""
) -> list[str]:
    """Moderates one message: its text and images, judged together.

    The service being unavailable (timeout, 5xx) refuses a message with
    images and lets text through; an input it rejects (400) is refused. A
    message with images gets ``IMAGE_TIMEOUT_S`` in all, downloads included.

    Args:
        texts: Text parts of the message.
        images: Image references (see ``image_data_url``); pass none to check
            text only.
        label: Where the message is from, for the log.

    Returns:
        The categories that block the message; empty when it may go.
    """
    url = service_url()
    parts: list[dict] = [{"type": "text", "text": t} for t in texts if t.strip()]
    if not url or (not parts and not images):
        return []
    categories: list[str] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (IMAGE_TIMEOUT_S if images else TEXT_TIMEOUT_S)
    # Downloads go through the system proxy like the rest of AstrBot; the
    # moderation service is local.
    async with (
        aiohttp.ClientSession(trust_env=True) as downloads,
        aiohttp.ClientSession(trust_env=False) as session,
    ):
        try:
            urls = await asyncio.wait_for(
                asyncio.gather(*(image_data_url(ref, downloads) for ref in images)),
                IMAGE_TIMEOUT_S,
            )
        except Exception as e:  # noqa: BLE001
            # An image that cannot be checked is not sent.
            logger.warning(
                "Content moderation could not read an image: %s", type(e).__name__
            )
            categories = [UNAVAILABLE]
        else:
            # Text goes with the first images; more than one request's worth
            # of images is checked in further requests.
            batches: list[list[str]] = [[]]
            size = 0
            for data_url in urls:
                if batches[-1] and (
                    len(batches[-1]) >= MAX_IMAGES_PER_REQUEST
                    or size + len(data_url) > MAX_REQUEST_IMAGE_BYTES
                ):
                    batches.append([])
                    size = 0
                batches[-1].append(data_url)
                size += len(data_url)
            for index, batch in enumerate(batches):
                payload = (parts if index == 0 else []) + [
                    {"type": "image_url", "image_url": {"url": u}} for u in batch
                ]
                remaining = deadline - loop.time()
                result = (
                    await _post(session, url, {"input": payload}, remaining)
                    if remaining > 0
                    else None
                )
                if result == 400:
                    categories = [INVALID_INPUT]
                elif result is None:
                    categories = [UNAVAILABLE] if images else []
                else:
                    categories = _flagged_categories(result[0])
                if categories:
                    break
    if categories:
        logger.info(
            "Content moderation blocked a message%s: %s",
            f" ({label})" if label else "",
            ", ".join(categories),
        )
    return categories


async def check_request(
    prompt: str,
    extra_parts: Sequence,
    image_urls: Sequence[str],
    mode: str,
    label: str = "",
) -> list[str]:
    """Moderates what a user sends with a model request, before it is sent.

    The user's own content only: the prompt, the parts added with it (a
    quoted message, attachment notes) and its images. Context AstrBot adds
    for the model alone (retrieved knowledge: parts marked temporary) is not
    the user's message; group history is checked message by message where it
    is added.

    Args:
        prompt: The message text.
        extra_parts: ``ProviderRequest.extra_user_content_parts``.
        image_urls: Image references (see ``image_data_url``).
        mode: The platform's mode (see ``platform_mode``).
        label: Where the message is from, for the log.

    Returns:
        The categories that block the request; empty when it may go.
    """
    if mode == MODE_DISABLED:
        return []
    texts = [prompt or ""]
    images = list(image_urls) if mode == MODE_ENABLED else []
    for part in extra_parts:
        if getattr(part, "_no_save", False):
            continue
        if getattr(part, "type", None) == "text":
            texts.append(getattr(part, "text", "") or "")
        elif getattr(part, "type", None) == "image_url" and mode == MODE_ENABLED:
            url = getattr(getattr(part, "image_url", None), "url", "")
            if url:
                images.append(url)
    return await check_message(texts, images, label)


async def flagged_texts(
    texts: Sequence[str], label: str = "", background: bool = False
) -> list[bool | None]:
    """Moderates texts one by one (e.g. group history), each on its own.

    A long text is checked in pieces of ``CHUNK_BYTES`` and is flagged when
    any piece is. A text the service rejects (400) counts as flagged. Once
    the service turns out unavailable the rest is left unchecked rather than
    waiting out a timeout per request.

    Args:
        texts: The texts.
        label: Where they are from, for the log.
        background: Nothing waits on the result (see ``_post``).

    Returns:
        Whether each text is flagged, in order; None for a text the service
        could not check. Callers keep those: they are text only.
    """
    url = service_url()
    flags: list[bool | None] = [None] * len(texts)
    if not url:
        return flags
    # (text index, piece); the service rejects empty strings.
    units = [
        (owner, piece if piece.strip() else ".")
        for owner, text in enumerate(texts)
        for piece in _pieces(text, CHUNK_BYTES)
    ]
    # Requests of at most TEXT_BATCH pieces and CHUNK_BYTES of text.
    batches: list[list[tuple[int, str]]] = []
    size = 0
    for unit in units:
        length = _utf8_len(unit[1])
        if batches and len(batches[-1]) < TEXT_BATCH and size + length <= CHUNK_BYTES:
            batches[-1].append(unit)
            size += length
        else:
            batches.append([unit])
            size = length
    # Per text: whether a piece was flagged; pieces answered, and in all.
    hit = [False] * len(texts)
    seen = [0] * len(texts)
    total = [0] * len(texts)
    for owner, _ in units:
        total[owner] += 1
    async with aiohttp.ClientSession(trust_env=False) as session:
        for batch in batches:
            answers = await _post(
                session,
                url,
                {"input": [p for _, p in batch]},
                TEXT_TIMEOUT_S,
                background,
            )
            unavailable = answers is None
            if answers == 400:
                # One piece it rejects fails the whole batch: ask one by one.
                answers = []
                for _, piece in batch:
                    single = await _post(
                        session, url, {"input": [piece]}, TEXT_TIMEOUT_S, background
                    )
                    if single is None:
                        unavailable = True
                        break
                    answers.append(400 if single == 400 else single[0])
            # Pieces without an answer leave their text unchecked.
            for (owner, _), answer in zip(batch, answers or []):
                categories = (
                    [INVALID_INPUT] if answer == 400 else _flagged_categories(answer)
                )
                seen[owner] += 1
                if categories:
                    if not hit[owner]:
                        logger.info(
                            "Content moderation dropped a history message%s: %s",
                            f" ({label})" if label else "",
                            ", ".join(categories),
                        )
                    hit[owner] = True
            if unavailable:
                break
    for owner in range(len(texts)):
        if hit[owner]:
            flags[owner] = True
        elif seen[owner] == total[owner]:
            flags[owner] = False
    return flags


async def filter_tool_result(
    items: list[dict], mode: str, label: str = ""
) -> list[dict]:
    """Removes the parts of a tool result that may not go to the model.

    A tool result (a file read, an image, a web page, an MCP or sandbox
    output) reaches the cloud model like a message does. Its text is checked
    first, then its images, each in order and never in the same request (an
    image slow to decode must not hold up the text); the service names each part's
    verdict (``parts``), so only the flagged parts are removed, and a note at
    the end tells the model what was removed and why. A part that cannot be
    checked (an image while the service is unavailable, an image type it does
    not read, anything it rejects) is withheld; text goes through when the
    service is unavailable. Text past ``MAX_TOOL_TEXT_BYTES`` is cut off,
    the rest checked in pieces of ``CHUNK_BYTES``; an item is removed when
    any of its pieces is flagged. Tool results are checked one at a time, so
    one holds up the checks of messages by about a request (~1 s) at most.

    Args:
        items: Codex content items (``inputText`` / ``inputImage``).
        mode: The platform's mode (see ``platform_mode``); ``text_only``
            checks text and lets images through.
        label: Where the result is from, for the log.

    Returns:
        The items to give the model.
    """
    url = service_url()
    if not url or mode == MODE_DISABLED:
        return items
    items = list(items)
    budget, cut = MAX_TOOL_TEXT_BYTES, False
    for index, item in enumerate(items):
        if item.get("type") == "inputText":
            text = str(item.get("text") or "")
            short = cut_to_bytes(text, budget)
            if short != text:
                items[index] = {**item, "text": short}
                cut = True
            budget = max(budget - _utf8_len(short), 0)
    if cut:
        items = [i for i in items if i.get("type") != "inputText" or i.get("text")]

    # Why each removed item was removed: its categories, or UNCHECKABLE.
    removed: dict[int, list[str]] = {}
    loop = asyncio.get_running_loop()
    async with (
        _semaphore("tool_results", 1),
        aiohttp.ClientSession(trust_env=True) as downloads,
        aiohttp.ClientSession(trust_env=False) as session,
    ):
        # From when it is this result's turn: waiting for others is no cost.
        deadline = loop.time() + TOOL_RESULT_TIMEOUT_S
        # Index of each item sent, and its part (a piece, for long text).
        sent: list[tuple[int, dict]] = []
        for index, item in enumerate(items):
            text = str(item.get("text") or "")
            if item.get("type") == "inputText" and text.strip():
                sent.extend(
                    (index, {"type": "text", "text": piece})
                    for piece in _pieces(text, CHUNK_BYTES)
                    if piece.strip()
                )
            elif item.get("type") == "inputImage" and mode == MODE_ENABLED:
                try:
                    data_url = await asyncio.wait_for(
                        image_data_url(str(item.get("imageUrl") or ""), downloads),
                        max(deadline - loop.time(), 0.001),
                    )
                except Exception:  # noqa: BLE001 - unreadable
                    data_url = ""
                mime = data_url[len("data:") :].split(";", 1)[0].lower()
                if mime not in SERVICE_IMAGE_TYPES and data_url:
                    # Mislabelled (image/jpg, octet-stream): go by the bytes.
                    try:
                        data = base64.b64decode(data_url.split(",", 1)[1])
                    except Exception:  # noqa: BLE001
                        data = b""
                    known = next(
                        (m for sig, m in _IMAGE_SIGNATURES if data.startswith(sig)),
                        "image/webp" if data[8:12] == b"WEBP" else "",
                    )
                    if known:
                        mime = known
                        data_url = _data_url(data, known)
                if mime not in SERVICE_IMAGE_TYPES:
                    removed[index] = [UNCHECKABLE]
                    continue
                sent.append(
                    (index, {"type": "image_url", "image_url": {"url": data_url}})
                )

        def mark(index: int, reasons: list[str]) -> None:
            # Categories found for any piece outrank "could not be checked".
            old = removed.get(index, [])
            if reasons == [UNCHECKABLE] and old:
                return
            if old == [UNCHECKABLE]:
                old = []
            removed[index] = old + [r for r in reasons if r not in old]

        # Whether the service answered a request of this result: then it is
        # up, only busy, and text it did not get to is withheld rather than
        # let through (padding must not outlast the check).
        answered = False

        async def judge(parts: list[tuple[int, dict]]) -> None:
            nonlocal answered
            remaining = deadline - loop.time()
            result = (
                await _post(session, url, {"input": [p for _, p in parts]}, remaining)
                if remaining > 0
                else None
            )
            if result is None:
                for index, part in parts:
                    if part["type"] == "image_url" or answered:
                        mark(index, [UNCHECKABLE])
                return
            answered = True
            if result == 400:
                # One part it cannot read fails them all: ask one by one.
                if len(parts) > 1:
                    for part in parts:
                        await judge([part])
                else:
                    mark(parts[0][0], [UNCHECKABLE])
                return
            verdicts = result[0].get("parts")
            if not isinstance(verdicts, list) or len(verdicts) != len(parts):
                # A service without per-part verdicts: the parts as a whole.
                categories = _flagged_categories(result[0])
                if categories:
                    for index, _ in parts:
                        mark(index, categories)
                return
            for (index, _), verdict in zip(parts, verdicts):
                if isinstance(verdict, dict) and verdict.get("flagged"):
                    names = verdict.get("categories")
                    mark(index, [str(n) for n in names or []] or ["flagged"])

        # Text first and on its own: an image slow to decode must not make
        # the text share its timeout. Requests of at most CHUNK_BYTES of
        # text, MAX_IMAGES_PER_REQUEST images and MAX_REQUEST_IMAGE_BYTES of
        # image data each.
        sent.sort(key=lambda entry: entry[1]["type"] == "image_url")
        batch: list[tuple[int, dict]] = []
        images = image_bytes = text_bytes = 0
        for entry in sent:
            part = entry[1]
            if part["type"] == "image_url":
                length = len(part["image_url"]["url"])
                full = (
                    images >= MAX_IMAGES_PER_REQUEST
                    or image_bytes + length > MAX_REQUEST_IMAGE_BYTES
                )
            else:
                length = _utf8_len(part["text"])
                full = text_bytes + length > CHUNK_BYTES
            # Each part costs the service a few milliseconds of its own.
            if batch and (
                full or len(batch) >= TEXT_BATCH or batch[-1][1]["type"] != part["type"]
            ):
                await judge(batch)
                batch, images, image_bytes, text_bytes = [], 0, 0, 0
            if part["type"] == "image_url":
                images, image_bytes = images + 1, image_bytes + length
            else:
                text_bytes += length
            batch.append(entry)
        if batch:
            await judge(batch)
    if not removed and not cut:
        return items

    kept = [item for index, item in enumerate(items) if index not in removed]
    # One line per kind and reason, with a count: a result of thousands of
    # items must not turn into a note as long.
    groups: dict[tuple[bool, str], int] = {}
    for index, reasons in sorted(removed.items()):
        if reasons == [UNCHECKABLE]:
            why = "it could not be checked by content moderation"
        else:
            why = f"flagged by content moderation: {', '.join(reasons)}"
        key = (items[index].get("type") == "inputImage", why)
        if reasons != [UNCHECKABLE]:
            why = f"flagged by content moderation: {', '.join(sorted(reasons))}"
            key = (key[0], why)
        groups[key] = groups.get(key, 0) + 1
    notes = []
    for (image, why), count in groups.items():
        if count == 1:
            kind = "an image" if image else "a text part"
        else:
            kind = f"{count} images" if image else f"{count} text parts"
        notes.append(f"- {kind} ({why})")
    if cut:
        notes.append(
            f"- the text after its first {MAX_TOOL_TEXT_BYTES // 1024} KB "
            "(too long to check)"
        )
    if removed:
        reasons: dict[str, int] = {}
        for found in removed.values():
            for reason in found:
                reasons[reason] = reasons.get(reason, 0) + 1
        logger.info(
            "Content moderation removed %d part(s) of a tool result%s: %s",
            len(removed),
            f" ({label})" if label else "",
            ", ".join(f"{r} x{n}" for r, n in reasons.items()),
        )
    kept.append(
        {
            "type": "inputText",
            "text": "[Some of this tool result was removed before it reached you:\n"
            + "\n".join(notes)
            + "\nIt cannot be shown. Do not try to fetch or reproduce what was "
            "flagged; tell the user only if it matters for the answer.]",
        }
    )
    return kept


async def filter_tool_text(text: str, mode: str, label: str = "") -> str:
    """``filter_tool_result`` for a tool result that is one text.

    For results that reach the model inside a prompt (a background task's,
    a background command's output) or a plugin's own tool loop.

    Args:
        text: The result.
        mode: The platform's mode (see ``platform_mode``).
        label: Where the result is from, for the log.

    Returns:
        The text to give the model: the result, or what is left of it and the
        note on what was removed.
    """
    if not text.strip():
        return text
    kept = await filter_tool_result([{"type": "inputText", "text": text}], mode, label)
    return "\n".join(str(i.get("text") or "") for i in kept)
