"""Content moderation before messages go to the cloud model."""

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from astrbot.core import content_moderation as cm
from astrbot.core.agent.message import ImageURLPart, TextPart

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
CATEGORIES = ("violent", "illegal", "sexual", "political")


class FakeService:
    """A /v1/moderations double: flags texts containing "bomb" or any image
    whose data contains "NSFW"; answers with ``status`` when set."""

    def __init__(self):
        self.requests: list = []
        self.bodies: list = []
        self.auth: list = []
        self.status = 200
        #: Answered instead of results, when set.
        self.body = None
        #: Seconds each request takes.
        self.delay = 0.0
        #: Answer per-part verdicts for an array of parts, as the infra does.
        self.part_verdicts = True
        self.in_flight = 0
        self.most_in_flight = 0

    async def handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body["input"])
        self.bodies.append(body)
        self.auth.append(request.headers.get("Authorization"))
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        if self.status != 200:
            return web.json_response({"error": {}}, status=self.status)
        if self.body is not None:
            return web.json_response(self.body)
        items = body["input"]
        texts = [i if isinstance(i, str) else i.get("text", "") for i in items]
        images = [
            i["image_url"]["url"]
            for i in items
            if isinstance(i, dict) and i.get("type") == "image_url"
        ]
        if any("REJECT" in t for t in texts) or any(
            b"REJECT" in base64.b64decode(u.split(",", 1)[1]) for u in images
        ):
            return web.json_response({"error": {}}, status=400)
        if items and isinstance(items[0], str):
            groups = [[{"type": "text", "text": t}] for t in items]
        else:
            groups = [items]

        def verdict(part):
            if part["type"] == "text":
                return ["violent"] if "bomb" in part["text"] else []
            data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            return ["sexual"] if b"NSFW" in data else []

        results = []
        for group in groups:
            named = [verdict(p) for p in group]
            categories = dict.fromkeys(CATEGORIES, False)
            for names in named:
                categories.update(dict.fromkeys(names, True))
            result = {"flagged": any(named), "categories": categories}
            if self.part_verdicts and not isinstance(items[0], str):
                result["parts"] = [
                    {"type": p["type"], "flagged": bool(n), "categories": n}
                    for p, n in zip(group, named)
                ]
            results.append(result)
        return web.json_response({"results": results})


@pytest_asyncio.fixture
async def service(monkeypatch):
    fake = FakeService()
    app = web.Application()
    app.router.add_post("/v1/moderations", fake.handle)
    server = TestServer(app)
    await server.start_server()
    monkeypatch.setattr(
        cm,
        "astrbot_config",
        {
            "content_moderation_url": str(server.make_url("")),
            "platform": [
                {"id": "qq", "content_moderation": "text_only"},
                {"id": "mumble", "content_moderation": "disabled"},
                {"id": "tg"},
            ],
        },
    )
    yield fake
    await server.close()


def _image(tmp_path, data=PNG, name="a.png"):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_platform_modes(monkeypatch):
    monkeypatch.setattr(
        cm,
        "astrbot_config",
        {
            "content_moderation_url": "http://127.0.0.1:17890",
            "platform": [{"id": "qq", "content_moderation": "text_only"}, {"id": "tg"}],
        },
    )
    assert cm.service_url() == "http://127.0.0.1:17890/v1/moderations"
    assert cm.platform_mode("qq") == cm.MODE_TEXT_ONLY
    # Platforms without the setting, and input of no platform, get it all.
    assert cm.platform_mode("tg") == cm.MODE_ENABLED
    assert cm.platform_mode("") == cm.MODE_ENABLED
    monkeypatch.setattr(cm, "astrbot_config", {"content_moderation_url": ""})
    assert cm.platform_mode("tg") == cm.MODE_DISABLED


@pytest.mark.asyncio
async def test_text_is_judged_with_its_images(service, tmp_path):
    assert await cm.check_message(["How do I terminate a C program?"], []) == []
    assert await cm.check_message(["How do I make a pipe bomb?"], []) == ["violent"]
    nsfw = _image(tmp_path, PNG + b"NSFW")
    assert await cm.check_message(["look"], [nsfw]) == ["sexual"]
    # One request: the text and the image, the image as a PNG data URL.
    [text, image] = service.requests[-1]
    assert text == {"type": "text", "text": "look"}
    assert image["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_more_images_than_one_request_takes(service, tmp_path):
    images = [_image(tmp_path, name=f"{i}.png") for i in range(17)]
    images.append(_image(tmp_path, PNG + b"NSFW", "last.png"))

    assert await cm.check_message(["hi"], images) == ["sexual"]
    first, second = service.requests
    assert len(first) == 1 + cm.MAX_IMAGES_PER_REQUEST
    # The text goes once, with the first images.
    assert [p["type"] for p in second] == ["image_url"] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "with_image", "expected"),
    [
        (500, False, []),  # text only: let through
        (500, True, [cm.UNAVAILABLE]),  # with images: refused
        (400, False, [cm.INVALID_INPUT]),
    ],
)
async def test_service_failures(service, tmp_path, status, with_image, expected):
    service.status = status
    images = [_image(tmp_path)] if with_image else []
    assert await cm.check_message(["hello"], images) == expected


@pytest.mark.asyncio
async def test_service_down(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cm, "astrbot_config", {"content_moderation_url": "http://127.0.0.1:9"}
    )
    assert await cm.check_message(["hello"], []) == []
    assert await cm.check_message(["hello"], [_image(tmp_path)]) == [cm.UNAVAILABLE]
    assert await cm.flagged_texts(["a", "b"]) == [None, None]


@pytest.mark.asyncio
async def test_an_unreadable_image_is_not_sent(service, tmp_path):
    assert await cm.check_message(["hi"], [str(tmp_path / "gone.png")]) == [
        cm.UNAVAILABLE
    ]
    assert service.requests == []


@pytest.mark.asyncio
async def test_request_parts_by_mode(service, tmp_path):
    nsfw = _image(tmp_path, PNG + b"NSFW")
    quoted = [
        TextPart(text="[quoted] a bomb recipe"),
        ImageURLPart(image_url=ImageURLPart.ImageURL(url="data:image/png;base64,AA==")),
    ]

    # The quoted text counts as the user's message.
    assert await cm.check_request("what is this", quoted, [], cm.MODE_ENABLED) == [
        "violent"
    ]
    # Text only: the image is never sent to the service.
    assert await cm.check_request("hi", [], [nsfw], cm.MODE_TEXT_ONLY) == []
    assert all(p["type"] == "text" for p in service.requests[-1])
    assert await cm.check_request("hi", [], [nsfw], cm.MODE_ENABLED) == ["sexual"]
    count = len(service.requests)
    assert await cm.check_request("a bomb", [], [nsfw], cm.MODE_DISABLED) == []
    assert len(service.requests) == count


@pytest.mark.asyncio
async def test_history_messages_are_judged_one_by_one(service, monkeypatch):
    monkeypatch.setattr(cm, "TEXT_BATCH", 2)
    flags = await cm.flagged_texts(["hello", "a bomb", "", "fine"])

    assert flags == [False, True, False, False]
    # Batched, and the service gets no empty string.
    assert service.requests == [["hello", "a bomb"], [".", "fine"]]


# --------------------------------------------------------- group history


@pytest.mark.asyncio
async def test_a_flagged_history_message_is_dropped_not_the_request(
    service, monkeypatch
):
    from astrbot.builtin_stars.astrbot import group_chat_context as gcc

    monkeypatch.setattr(gcc, "content_moderation", cm)
    umo = "tg:GroupMessage:g1"
    ctx = gcc.GroupChatContext.__new__(gcc.GroupChatContext)
    ctx._locks = {}
    ctx.raw_records = gcc.defaultdict(gcc.deque)
    ctx._record_ids = gcc.defaultdict(gcc.deque)
    ctx._pending_triggers = gcc.defaultdict(dict)
    ctx._verdicts = gcc.defaultdict(dict)
    for line, rid in [("[a] hello", "r1"), ("[b] a bomb", "r2"), ("[c] @bot hi", "r3")]:
        ctx.raw_records[umo].append(line)
        ctx._record_ids[umo].append(rid)
    extras = {"_group_context_record_id": "r3"}
    event = SimpleNamespace(
        unified_msg_origin=umo,
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=extras.__setitem__,
    )
    units = []
    req = SimpleNamespace(
        add_persistent_context=lambda name, content, unit_id=None: units.append(
            (name, content, unit_id)
        )
    )

    await ctx.on_req_llm(event, req)

    [(name, block, unit_id)] = units
    assert name == "group_history"
    assert "[a] hello" in block
    assert "bomb" not in block
    assert unit_id == "r1"


def _group_context():
    from astrbot.builtin_stars.astrbot import group_chat_context as gcc

    ctx = gcc.GroupChatContext.__new__(gcc.GroupChatContext)
    ctx._locks = {}
    ctx.raw_records = gcc.defaultdict(gcc.deque)
    ctx._record_ids = gcc.defaultdict(gcc.deque)
    ctx._pending_triggers = gcc.defaultdict(dict)
    ctx._verdicts = gcc.defaultdict(dict)
    ctx._checks = set()
    ctx.cfg = lambda event: {"group_message_max_cnt": 100, "image_caption": False}
    return ctx


def _group_message(text, *, mention=False, umo="tg:GroupMessage:g1"):
    from astrbot.api.message_components import Plain
    from astrbot.api.platform import MessageType

    extras = {}
    return SimpleNamespace(
        unified_msg_origin=umo,
        is_at_or_wake_command=mention,
        message_obj=SimpleNamespace(
            sender=SimpleNamespace(nickname="n", user_id="1"), timestamp=0
        ),
        get_message_type=lambda: MessageType.GROUP_MESSAGE,
        get_messages=lambda: [Plain(text)],
        get_self_id=lambda: "999",
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=extras.__setitem__,
        extras=extras,
    )


def _history_request():
    units = []
    return units, SimpleNamespace(
        add_persistent_context=lambda name, content, unit_id=None: units.append(content)
    )


@pytest.mark.asyncio
async def test_history_is_checked_as_it_arrives_and_not_again(service):
    ctx = _group_context()
    for text in ["hello", "a bomb", "fine"]:
        await ctx.handle_message(_group_message(text))
    # Let the background checks finish, as they would before a reply.
    await asyncio.gather(*ctx._verdicts["tg:GroupMessage:g1"].values())
    checked_on_arrival = len(service.requests)
    trigger = _group_message("@bot hi", mention=True)
    await ctx.handle_message(trigger)

    units, req = _history_request()
    await ctx.on_req_llm(trigger, req)

    assert checked_on_arrival == 3  # one per message; none for the trigger
    assert len(service.requests) == 3  # the request asked nothing more
    [block] = units
    assert "hello" in block and "fine" in block
    assert "bomb" not in block
    # Given back unanswered: shown again, flagged one still gone, not rechecked.
    await trigger.extras["_group_context_restore"]()
    again = _group_message("@bot again", mention=True)
    await ctx.handle_message(again)
    units, req = _history_request()
    await ctx.on_req_llm(again, req)
    [block] = units
    assert "hello" in block and "bomb" not in block
    assert len(service.requests) == 3
    assert ctx._verdicts["tg:GroupMessage:g1"] == {}


@pytest.mark.asyncio
async def test_history_the_service_missed_is_checked_with_the_request(service):
    ctx = _group_context()
    service.status = 500
    for text in ["hello", "a bomb"]:
        await ctx.handle_message(_group_message(text))
    await asyncio.gather(*ctx._verdicts["tg:GroupMessage:g1"].values())
    service.status = 200
    trigger = _group_message("@bot hi", mention=True)
    await ctx.handle_message(trigger)

    units, req = _history_request()
    await ctx.on_req_llm(trigger, req)

    [block] = units
    assert "hello" in block and "bomb" not in block
    # Both rechecked in one batch with the request.
    last = service.requests[-1]
    assert len(last) == 2
    assert last[0].endswith("hello") and last[1].endswith("a bomb")


@pytest.mark.asyncio
async def test_a_platform_with_moderation_off_starts_no_checks(service):
    ctx = _group_context()
    await ctx.handle_message(_group_message("a bomb", umo="mumble:GroupMessage:g"))

    assert ctx._verdicts["mumble:GroupMessage:g"] == {}
    assert service.requests == []


# ------------------------------------------------------------ the stage


@pytest.mark.asyncio
async def test_a_blocked_message_reaches_no_turn(monkeypatch):
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import (
        third_party,
    )
    from tests.unit.test_third_party_agent_sub_stage import _process_with

    check = AsyncMock(return_value=["violent"])
    refund = AsyncMock()
    steer = AsyncMock(return_value=None)
    monkeypatch.setattr(third_party.content_moderation, "check_request", check)
    monkeypatch.setattr(third_party, "refund_rate_limit", refund)
    forget = AsyncMock()
    monkeypatch.setattr(third_party, "forget_group_message", forget)

    release, keep, event, runner, _ = await _process_with(monkeypatch, steer=steer)

    steer.assert_not_awaited()
    runner.reset.assert_not_awaited()
    release.assert_awaited_once_with(event)
    refund.assert_awaited_once_with(event)
    forget.assert_awaited_once_with(event)
    [result] = event.set_result.call_args.args
    assert result.get_plain_text() == cm.BLOCKED_REPLY
    assert check.await_args.args[0] == "hello"


@pytest.mark.asyncio
async def test_a_plugin_model_call_is_checked(monkeypatch):
    from astrbot.core.agent.runners.codex import provider_adapter

    monkeypatch.setattr(
        provider_adapter.content_moderation,
        "check_request",
        AsyncMock(return_value=["sexual"]),
    )
    provider = provider_adapter.make_codex_provider({})
    provider._start = AsyncMock()

    resp = await provider.text_chat(prompt="caption this", image_urls=["x.png"])

    assert resp.role == "err"
    assert resp.completion_text == cm.BLOCKED_REPLY
    provider._start.assert_not_awaited()


# ------------------------------------------------------- review round 1


@pytest.mark.asyncio
async def test_a_history_text_the_service_rejects_counts_as_flagged(service):
    flags = await cm.flagged_texts(["hello", "REJECT me", "a bomb"])

    assert flags == [False, True, True]
    # The batch failed as a whole, then each text was asked on its own.
    assert service.requests[1:] == [["hello"], ["REJECT me"], ["a bomb"]]


@pytest.mark.asyncio
async def test_an_unavailable_service_is_not_waited_out_batch_by_batch(
    service, monkeypatch
):
    monkeypatch.setattr(cm, "TEXT_BATCH", 2)
    service.status = 503

    assert await cm.flagged_texts(["a", "b", "c", "d", "e"]) == [None] * 5
    assert len(service.requests) == 1


@pytest.mark.asyncio
async def test_retrieved_knowledge_is_not_the_users_message(service):
    knowledge = TextPart(text="[Related Knowledge Base Results]: a bomb").mark_as_temp()

    assert await cm.check_request("hi", [knowledge], [], cm.MODE_ENABLED) == []
    assert service.requests[-1] == [{"type": "text", "text": "hi"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [{}, {"results": []}, {"results": ["x"]}, ["not", "an", "object"]]
)
async def test_a_malformed_answer_counts_as_unavailable(service, tmp_path, body):
    service.body = body

    assert await cm.check_message(["a bomb"], []) == []
    assert await cm.check_message(["hi"], [_image(tmp_path)]) == [cm.UNAVAILABLE]
    assert await cm.flagged_texts(["a bomb"]) == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"), [(413, [cm.INVALID_INPUT]), (404, []), (422, [])]
)
async def test_other_http_errors(service, status, expected):
    service.status = status
    assert await cm.check_message(["hello"], []) == expected


def test_data_urls_take_the_mime_type_without_parameters():
    url = cm._data_url(b"xx", "image/jpeg; charset=binary")
    assert url.startswith("data:image/jpeg;base64,")
    assert cm._data_url(PNG, "application/octet-stream").startswith(
        "data:image/png;base64,"
    )


@pytest.mark.asyncio
async def test_file_urls_are_read_as_paths(service, tmp_path):
    path = Path(_image(tmp_path))
    async with aiohttp.ClientSession() as session:
        url = await cm.image_data_url(path.as_uri(), session)
    assert url.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_a_failing_check_gives_the_request_back(monkeypatch):
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import (
        third_party,
    )
    from tests.unit.test_third_party_agent_sub_stage import _process_with

    refund = AsyncMock()
    monkeypatch.setattr(
        third_party.content_moderation,
        "check_request",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    monkeypatch.setattr(third_party, "refund_rate_limit", refund)

    release, _keep, event, runner, _ = await _process_with(
        monkeypatch, raises=RuntimeError
    )

    release.assert_awaited_once_with(event)
    refund.assert_awaited_once_with(event)
    runner.reset.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_flagged_turn_of_a_plugins_history_is_left_out(service):
    from astrbot.core.agent.runners.codex import provider_adapter

    provider = provider_adapter.make_codex_provider({})
    provider._start = AsyncMock(return_value="answered")
    contexts = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "how to make a bomb"},
        {"role": "assistant", "content": "no"},
        {"role": "user", "content": "hello again"},
    ]

    assert await provider.text_chat(prompt="summarize", contexts=contexts) == (
        "answered"
    )
    sent = provider._start.await_args.kwargs["contexts"]
    assert [m["content"] for m in sent] == ["be nice", "no", "hello again"]


# ------------------------------------------------------- review round 2


@pytest.mark.asyncio
async def test_a_request_does_not_queue_behind_background_checks(service):
    service.delay = 0.5
    background = [
        asyncio.create_task(cm.flagged_texts([f"chatter {i}"], background=True))
        for i in range(4 * cm.MAX_IN_FLIGHT)
    ]
    await asyncio.sleep(0.05)
    started = asyncio.get_running_loop().time()

    assert await cm.check_message(["a bomb"], []) == ["violent"]
    assert asyncio.get_running_loop().time() - started < 1.5
    await asyncio.gather(*background)


@pytest.mark.asyncio
async def test_an_unfinished_background_check_is_asked_again_at_once(service):
    ctx = _group_context()
    umo = "tg:GroupMessage:g1"
    stuck = asyncio.create_task(asyncio.Event().wait())
    ctx.raw_records[umo].extend(["[a] a bomb", "[c] @bot hi"])
    ctx._record_ids[umo].extend(["r1", "r2"])
    ctx._verdicts[umo]["r1"] = stuck
    trigger = _group_message("@bot hi", mention=True)
    trigger.extras["_group_context_record_id"] = "r2"

    units, req = _history_request()
    await ctx.on_req_llm(trigger, req)

    assert units == []  # the only history line was flagged when asked again
    await asyncio.sleep(0)
    assert stuck.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize(("umo", "blocked"), [("tg:G:g", True), ("qq:G:g", False)])
async def test_a_caption_image_is_checked_as_the_platform_says(
    service, tmp_path, umo, blocked
):
    from astrbot.core.provider.entities import LLMResponse
    from astrbot.core.provider.provider import Provider

    provider = MagicMock(spec=Provider)
    provider.text_chat = AsyncMock(
        return_value=LLMResponse(role="assistant", completion_text="a cat")
    )
    ctx = _group_context()
    ctx.context = SimpleNamespace(get_provider_by_id=lambda _id: provider)
    nsfw = _image(tmp_path, PNG + b"NSFW")

    if blocked:
        with pytest.raises(Exception, match=cm.BLOCKED_REPLY):
            await ctx.get_image_caption(nsfw, "caption", "describe", umo)
        provider.text_chat.assert_not_awaited()
    else:  # qq is text only: the image is not checked
        assert await ctx.get_image_caption(nsfw, "caption", "describe", umo) == (
            "a cat"
        )


@pytest.mark.asyncio
async def test_images_sent_as_files_are_checked(monkeypatch):
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import (
        third_party,
    )
    from tests.unit.test_third_party_agent_sub_stage import _process_with

    check = AsyncMock(return_value=[])
    monkeypatch.setattr(third_party.content_moderation, "check_request", check)

    async def prepare(event, req, *args, **kwargs):
        req.image_urls.append("/tmp/photo.png")
        event.set_extra(cm.ATTACHMENT_IMAGES_EXTRA, ["/tmp/sent-as-file.jpg"])

    await _process_with(monkeypatch, prepare=prepare)

    assert check.await_args.args[2] == ["/tmp/photo.png", "/tmp/sent-as-file.jpg"]


@pytest.mark.asyncio
async def test_a_short_answer_leaves_only_the_missing_texts_unchecked(
    service, monkeypatch
):
    monkeypatch.setattr(cm, "TEXT_BATCH", 2)
    service.body = {"results": [{"flagged": True, "categories": {"violent": True}}]}

    assert await cm.flagged_texts(["a", "b", "c", "d"]) == [True, None, True, None]
    assert len(service.requests) == 2


# ------------------------------------------------------------ tool results


def _data(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _text_item(text):
    return {"type": "inputText", "text": text}


def _image_item(data=PNG):
    return {"type": "inputImage", "imageUrl": _data(data)}


@pytest.mark.asyncio
async def test_only_the_flagged_parts_of_a_tool_result_are_removed(service):
    items = [
        _text_item("file a: hello"),
        _image_item(PNG + b"NSFW"),
        _text_item("file b: a bomb recipe"),
        _image_item(),
    ]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED, "read_file")

    assert kept[:2] == [items[0], items[3]]
    note = kept[2]["text"]
    assert "an image (flagged by content moderation: sexual)" in note
    assert "a text part (flagged by content moderation: violent)" in note
    assert "bomb" not in note  # the note never repeats what was removed
    # Text first, then the images, each in order.
    types = [[p["type"] for p in r] for r in service.requests]
    assert types == [["text", "text"], ["image_url", "image_url"]]


@pytest.mark.asyncio
async def test_a_clean_tool_result_is_untouched(service):
    items = [_text_item("fine"), _image_item()]
    assert await cm.filter_tool_result(items, cm.MODE_ENABLED) == items


@pytest.mark.asyncio
async def test_text_only_platforms_check_tool_text_alone(service):
    items = [_text_item("a bomb"), _image_item(PNG + b"NSFW")]

    kept = await cm.filter_tool_result(items, cm.MODE_TEXT_ONLY)

    assert kept[0] == items[1]
    assert all(p["type"] == "text" for p in service.requests[-1])
    assert await cm.filter_tool_result(items, cm.MODE_DISABLED) == items


@pytest.mark.asyncio
async def test_tool_images_are_withheld_when_the_service_is_down(service):
    service.status = 503
    items = [_text_item("a bomb"), _image_item()]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    assert kept[0] == items[0]  # text goes through, as for messages
    assert "could not be checked" in kept[1]["text"]
    assert len(kept) == 2


@pytest.mark.asyncio
async def test_a_service_without_part_verdicts_judges_the_whole_result(service):
    service.part_verdicts = False
    items = [_text_item("hello"), _text_item("a bomb")]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    assert len(kept) == 1
    assert "- 2 text parts" in kept[0]["text"]


@pytest.mark.asyncio
async def test_a_rejected_tool_result_is_removed(service):
    items = [_text_item("REJECT this"), _text_item("")]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    # The empty text was never sent, so it stays.
    assert kept[0] == items[1]
    assert "a text part (it could not be checked" in kept[1]["text"]


@pytest.mark.asyncio
async def test_more_tool_images_than_one_request_takes(service):
    items = [_image_item() for _ in range(17)] + [_image_item(PNG + b"NSFW")]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    assert len(service.requests) == 2
    assert kept[:17] == items[:17]
    assert "sexual" in kept[17]["text"]


@pytest.mark.asyncio
async def test_the_tool_bridge_filters_what_a_tool_returns(service):
    from astrbot.core.agent.hooks import BaseAgentRunHooks
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.runners.codex.tool_bridge import CodexToolBridge
    from astrbot.core.agent.tool import FunctionTool, ToolSet

    async def read(event, path: str = ""):
        return "a bomb recipe"

    tool = FunctionTool(
        name="read",
        description="reads",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=read,
    )
    event = SimpleNamespace(
        get_result=lambda: None,
        get_platform_id=lambda: "tg",
        unified_msg_origin="tg:GroupMessage:g",
    )
    ctx = ContextWrapper(context=SimpleNamespace(event=event), tool_call_timeout=5)

    result = await CodexToolBridge(ToolSet([tool])).call(
        {"namespace": "astrbot", "tool": "read", "arguments": {"path": "x"}},
        ctx,
        BaseAgentRunHooks(),
    )

    [note] = result["contentItems"]
    assert "violent" in note["text"] and "bomb" not in note["text"]


# ------------------------------------------------------- review round 3


@pytest.mark.asyncio
async def test_an_image_the_service_rejects_costs_only_that_image(service):
    items = [_text_item("notes: hello"), _image_item(PNG + b"REJECT"), _image_item()]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    assert kept[:2] == [items[0], items[2]]
    assert "an image (it could not be checked" in kept[2]["text"]
    assert "flagged" not in kept[2]["text"].split("Do not")[0]
    # The batch was rejected as a whole, then each part asked on its own.
    # Text alone, then the images: rejected together, then one by one.
    assert [len(r) for r in service.requests] == [1, 2, 1, 1]


@pytest.mark.asyncio
async def test_an_image_type_the_service_cannot_read_is_withheld(service):
    svg = {"type": "inputImage", "imageUrl": "data:image/svg+xml;base64,PHN2Zz4="}
    items = [_text_item("hello"), svg]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    assert kept[0] == items[0]
    assert "an image (it could not be checked" in kept[1]["text"]
    assert service.requests == [[{"type": "text", "text": "hello"}]]


@pytest.mark.asyncio
async def test_a_long_tool_result_is_cut_before_its_check(service, monkeypatch):
    monkeypatch.setattr(cm, "MAX_TOOL_TEXT_BYTES", 10)
    items = [_text_item("0123456789abcdef"), _text_item("more text")]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    assert kept[0] == _text_item("0123456789")
    assert "too long to check" in kept[1]["text"]
    assert service.requests == [[{"type": "text", "text": "0123456789"}]]


@pytest.mark.asyncio
async def test_a_text_result_keeps_what_passes_and_says_what_went(service):
    assert await cm.filter_tool_text("all fine", cm.MODE_ENABLED) == "all fine"
    filtered = await cm.filter_tool_text("a bomb recipe", cm.MODE_ENABLED)
    assert "bomb" not in filtered and "violent" in filtered


@pytest.mark.asyncio
async def test_base64_images_are_read(service):
    async with aiohttp.ClientSession() as session:
        url = await cm.image_data_url(
            "base64://" + base64.b64encode(PNG).decode(), session
        )
    assert url.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_a_background_commands_output_is_filtered(monkeypatch):
    from astrbot.core.agent.runners.codex import native, wake

    seen = {}

    async def filter_text(text, mode, label=""):
        seen["text"] = text
        return "[removed]"

    async def steer(umo, sender_id, turn_input, *, prompt="", scopes=None):
        seen["prompt"] = prompt
        return ""

    monkeypatch.setattr(wake.content_moderation, "filter_tool_text", filter_text)
    monkeypatch.setattr(native, "try_steer", steer)
    ctx = SimpleNamespace(
        get_config=lambda umo=None: {"agent_runner": {"runner_type": "codex"}}
    )

    await wake.run_background_exec_completion(
        ctx,
        session_str="tg:GroupMessage:g",
        sender_id="1",
        role="member",
        session_id="s1",
        exit_code=0,
        output="a bomb recipe",
    )

    assert seen["text"] == "a bomb recipe"
    assert "[removed]" in seen["prompt"] and "bomb" not in seen["prompt"]


@pytest.mark.asyncio
async def test_a_plugin_tool_loops_results_are_filtered(monkeypatch):
    from astrbot.core.agent.runners.codex import provider_adapter

    monkeypatch.setattr(
        provider_adapter.content_moderation,
        "filter_tool_text",
        AsyncMock(return_value="[removed]"),
    )
    provider = provider_adapter.make_codex_provider({})
    future = asyncio.get_running_loop().create_future()
    pending = SimpleNamespace(futures={"c1": future}, created=1e18)
    provider._pending["c1"] = pending
    provider._drive = AsyncMock(return_value="driven")
    monkeypatch.setattr(
        provider_adapter, "_tool_results", lambda *_: {"c1": "a bomb recipe"}
    )

    assert await provider.text_chat(prompt=None) == "driven"
    sent = future.result()["contentItems"][0]["text"]
    assert sent == "[removed]"


# ------------------------------------------------------- review round 4


def test_text_is_cut_on_a_character_within_its_bytes():
    assert cm.cut_to_bytes("abc", 10) == "abc"
    assert cm.cut_to_bytes("中文字", 7) == "中文"  # 3 bytes each


@pytest.mark.asyncio
async def test_tool_results_are_checked_one_at_a_time(service):
    service.delay = 0.2
    await asyncio.gather(
        *(
            cm.filter_tool_result([_text_item(f"page {i}")], cm.MODE_ENABLED)
            for i in range(3)
        )
    )
    assert service.most_in_flight == 1


@pytest.mark.asyncio
async def test_history_batches_stay_small_in_bytes(service, monkeypatch):
    monkeypatch.setattr(cm, "CHUNK_BYTES", 10)
    flags = await cm.flagged_texts(["aaaaaa", "bbbbbb", "cc", "a bomb"])

    assert flags == [False, False, False, True]
    assert service.requests == [["aaaaaa"], ["bbbbbb", "cc"], ["a bomb"]]


@pytest.mark.asyncio
async def test_a_mislabelled_image_is_checked_by_its_bytes(service):
    jpg = {
        "type": "inputImage",
        "imageUrl": "data:image/jpg;base64," + base64.b64encode(PNG + b"NSFW").decode(),
    }

    kept = await cm.filter_tool_result([jpg], cm.MODE_ENABLED)

    assert "sexual" in kept[0]["text"]
    assert service.requests[0][0]["image_url"]["url"].startswith("data:image/png")


@pytest.mark.asyncio
async def test_a_tool_loop_step_checks_only_its_own_results(monkeypatch):
    from astrbot.core.agent.runners.codex import provider_adapter

    check = AsyncMock(side_effect=lambda text, mode, label="": text)
    monkeypatch.setattr(provider_adapter.content_moderation, "filter_tool_text", check)
    provider = provider_adapter.make_codex_provider({})
    future = asyncio.get_running_loop().create_future()
    provider._pending["c2"] = SimpleNamespace(futures={"c2": future}, created=1e18)
    provider._drive = AsyncMock(return_value="driven")
    monkeypatch.setattr(
        provider_adapter,
        "_tool_results",
        lambda *_: {"c0": "old", "c1": "older", "c2": "new"},
    )

    await provider.text_chat(prompt=None)

    assert [c.args[0] for c in check.await_args_list] == ["new"]


@pytest.mark.asyncio
async def test_a_long_tool_turn_of_a_plugins_history_is_cut(service, monkeypatch):
    from astrbot.core.agent.runners.codex import provider_adapter

    monkeypatch.setattr(cm, "MAX_TOOL_TEXT_BYTES", 5)
    provider = provider_adapter.make_codex_provider({})
    provider._start = AsyncMock(return_value="answered")
    contexts = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "0123456789", "tool_call_id": "t1"},
    ]

    await provider.text_chat(prompt="go on", contexts=contexts)

    sent = provider._start.await_args.kwargs["contexts"]
    assert sent[1]["content"].startswith("01234\n[cut off")
    assert service.requests[-1] == ["hi", "01234"]


# ------------------------------------------------------- review round 5


def test_text_splits_into_pieces_on_characters():
    assert cm._pieces("", 4) == [""]
    # Each piece repeats up to half a piece from the one before.
    assert cm._pieces("abcdefghij", 4) == ["abcd", "cdef", "efgh", "ghij"]
    assert cm._pieces("中文字符", 7) == ["中文", "文字", "字符"]
    # A lone surrogate (from JSON) neither raises nor loops.
    assert cm._pieces("a\ud800b", 4) == ["a\ud800", "b"]
    assert cm.cut_to_bytes("a\ud800b", 4) == "a"


@pytest.mark.asyncio
async def test_a_long_history_text_is_checked_in_pieces(service, monkeypatch):
    monkeypatch.setattr(cm, "CHUNK_BYTES", 8)

    flags = await cm.flagged_texts(["hello there, a bomb here", "fine"])

    assert flags == [True, False]
    assert all(len(t.encode()) <= 8 for r in service.requests for t in r)


@pytest.mark.asyncio
async def test_a_long_tool_text_is_checked_in_pieces(service, monkeypatch):
    monkeypatch.setattr(cm, "CHUNK_BYTES", 8)
    items = [_text_item("safe text then a bomb"), _text_item("ok")]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    assert kept[0] == items[1]
    assert kept[1]["text"].count("- a text part") == 1
    assert len(service.requests) > 2
    assert all(
        sum(len(p["text"].encode()) for p in r if p["type"] == "text") <= 8
        for r in service.requests
    )


# ------------------------------------------------------- review round 6


def test_pieces_never_pass_their_limit():
    for limit in range(1, 9):
        for piece in cm._pieces("ab中c字d\ud800efg文", limit):
            assert len(piece.encode("utf-8", "surrogatepass")) <= max(limit, 4)


@pytest.mark.asyncio
async def test_text_the_busy_service_never_got_to_is_withheld(service, monkeypatch):
    monkeypatch.setattr(cm, "CHUNK_BYTES", 8)
    monkeypatch.setattr(cm, "TOOL_RESULT_TIMEOUT_S", 0.5)
    service.delay = 0.2
    items = [_text_item("aaaaaaa bbbbbbb ccccccc ddddddd a bomb")]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    # The first pieces were answered, the rest ran out the deadline.
    assert all(i["text"] != items[0]["text"] for i in kept)
    assert "could not be checked" in kept[-1]["text"]


@pytest.mark.asyncio
async def test_text_passes_when_the_service_is_down_from_the_start(service):
    service.status = 503
    items = [_text_item("hello")]
    assert await cm.filter_tool_result(items, cm.MODE_ENABLED) == items


@pytest.mark.asyncio
async def test_many_small_tool_items_go_in_small_requests(service):
    items = [_text_item(str(i)) for i in range(70)]

    assert await cm.filter_tool_result(items, cm.MODE_ENABLED) == items
    assert [len(r) for r in service.requests] == [32, 32, 6]


@pytest.mark.asyncio
async def test_a_text_item_without_text_is_kept(service):
    items = [{"type": "inputText", "text": None}, _text_item("hi")]
    assert await cm.filter_tool_result(items, cm.MODE_ENABLED) == items


# ------------------------------------------------------- review round 7


@pytest.mark.asyncio
async def test_a_huge_image_file_is_never_read(service, tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "MAX_IMAGE_BYTES", 64)
    big = _image(tmp_path, PNG + b"\0" * 100, "big.png")

    assert await cm.check_message(["hi"], [big]) == [cm.UNAVAILABLE]
    assert service.requests == []
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError):
            await cm.image_data_url("base64://" + "A" * 200, session)


@pytest.mark.asyncio
async def test_tool_text_never_shares_a_request_with_images(service):
    items = [_image_item(), _text_item("hello"), _image_item(), _text_item("bye")]

    assert await cm.filter_tool_result(items, cm.MODE_ENABLED) == items
    types = [[p["type"] for p in r] for r in service.requests]
    assert types == [["text", "text"], ["image_url", "image_url"]]


@pytest.mark.asyncio
async def test_notes_count_removed_parts_instead_of_listing_them(service):
    items = [_text_item(f"bomb {i}") for i in range(5)] + [_text_item("fine")]

    kept = await cm.filter_tool_result(items, cm.MODE_ENABLED)

    assert kept[0] == items[5]
    assert "- 5 text parts (flagged by content moderation: violent)" in kept[1]["text"]


# ------------------------------------------------------- review round 8


@pytest.mark.asyncio
async def test_big_message_images_go_in_requests_under_the_size_cap(
    service, tmp_path, monkeypatch
):
    monkeypatch.setattr(cm, "MAX_REQUEST_IMAGE_BYTES", 100)
    images = [_image(tmp_path, PNG + bytes([i]) * 30, f"{i}.png") for i in range(3)]

    assert await cm.check_message(["hi"], images) == []
    assert [len(r) for r in service.requests] == [2, 1, 1]


# --------------------------------------------- persisted group history tool


@pytest.mark.asyncio
async def test_the_history_tool_leaves_out_flagged_messages(service, tmp_path):
    from astrbot.core.db.sqlite import SQLiteDatabase
    from astrbot.core.message.components import Plain
    from astrbot.core.message.message_event_result import MessageChain
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform_message_history_mgr import (
        PlatformMessageHistoryManager,
    )
    from astrbot.core.tools.message_tools import GetGroupMessageHistoryTool

    db = SQLiteDatabase(str(tmp_path / "history.db"))
    manager = PlatformMessageHistoryManager(db)
    umo = "tg:GroupMessage:g1"
    try:
        for text in ("hello", "a bomb recipe", "fine"):
            await manager.insert_message_chain(
                platform_id="tg",
                user_id=umo,
                message_chain=MessageChain([Plain(text)]),
                role="user",
                sender_id="1",
                sender_name="Alice",
                max_messages=20,
            )
        event = SimpleNamespace(
            unified_msg_origin=umo,
            get_message_type=lambda: MessageType.GROUP_MESSAGE,
            get_platform_id=lambda: "tg",
            get_extra=lambda key, default=None: default,
        )
        context = SimpleNamespace(
            context=SimpleNamespace(
                event=event,
                context=SimpleNamespace(
                    get_config=lambda umo: {
                        "provider_ltm_settings": {"group_message_history_enable": True}
                    },
                    message_history_manager=manager,
                ),
            )
        )

        result = await GetGroupMessageHistoryTool().call(context, limit=10)
    finally:
        await db.engine.dispose()

    assert "hello" in result and "fine" in result
    assert "bomb" not in result


# ------------------------------------------------------------ thresholds


def test_thresholds_come_from_the_system_settings(monkeypatch):
    monkeypatch.setattr(
        cm,
        "astrbot_config",
        {
            "content_moderation_threshold": 0.5,
            "content_moderation_nsfw_threshold": "0.3",
        },
    )
    assert cm.check_options() == {"threshold": 0.5, "nsfw_threshold": 0.3}
    # Values the service would reject leave its default in place.
    for bad in (1, 1.5, -0.1, "x"):
        monkeypatch.setattr(cm, "astrbot_config", {"content_moderation_threshold": bad})
        assert cm.check_options() == {}
    monkeypatch.setattr(cm, "astrbot_config", {})
    assert cm.check_options() == {}
    # Image levels: sent as chosen; none chosen keeps the service's.
    monkeypatch.setattr(
        cm, "astrbot_config", {"content_moderation_nsfw_labels": ["medium", " high"]}
    )
    assert cm.check_options() == {"nsfw_labels": ["medium", "high"]}
    monkeypatch.setattr(cm, "astrbot_config", {"content_moderation_nsfw_labels": []})
    assert cm.check_options() == {}
    # Text categories likewise; none selected means all (sent as nothing).
    monkeypatch.setattr(
        cm, "astrbot_config", {"content_moderation_categories": ["violent", "pii"]}
    )
    assert cm.check_options() == {"categories": ["violent", "pii"]}
    monkeypatch.setattr(cm, "astrbot_config", {"content_moderation_categories": []})
    assert cm.check_options() == {}


@pytest.mark.asyncio
async def test_every_check_carries_the_thresholds(service, tmp_path):
    cm.astrbot_config["content_moderation_threshold"] = 0.7
    cm.astrbot_config["content_moderation_nsfw_threshold"] = 0.2

    await cm.check_message(["hi"], [_image(tmp_path)])
    await cm.flagged_texts(["a"])
    await cm.filter_tool_result([_text_item("b")], cm.MODE_ENABLED)

    assert len(service.bodies) == 3
    for body in service.bodies:
        assert body["threshold"] == 0.7 and body["nsfw_threshold"] == 0.2


@pytest.mark.asyncio
async def test_a_token_is_sent_when_set(service):
    await cm.check_message(["hi"], [])
    cm.astrbot_config["content_moderation_token"] = " s3cret "
    await cm.flagged_texts(["a"])

    assert service.auth == [None, "Bearer s3cret"]
