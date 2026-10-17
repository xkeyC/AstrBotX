"""A client of a local-multimodal-infra server.

Plugins get one with ``Context.get_local_infra()`` (or
``astrbot.api.local_infra.get_local_infra()``) instead of keeping their own
address, token and task code. The client talks to the server AstrBot is set
to (``local_infra``): its OpenAI-compatible routes where there are some
(embeddings, rerank, moderation), otherwise the generic task flow of its
JSON-RPC API (create a task, upload its files, start it and wait).

Example::

    infra = self.context.get_local_infra()
    if infra is not None:
        wav_path = await infra.synthesize("你好", reference_audio="ref.wav")
        text = (await infra.transcribe("voice.wav"))["text"]
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

import aiohttp

from .config import SERVICE_TASK_KINDS, http_base, local_infra_settings

#: A task file: a path, raw bytes, or ``(name, bytes, mime)``.
TaskFile = str | os.PathLike | bytes | tuple[str, bytes, str]


class LocalInfraError(RuntimeError):
    """The server refused or failed a request."""


@dataclass
class InfraModel:
    """A model the server knows."""

    id: str
    name: str = ""
    #: Task kinds it runs (``asr.transcribe``, ``tts.synthesize``, ...).
    task_kinds: list[str] = field(default_factory=list)
    #: Enabled on the server.
    enabled: bool = True
    #: Its files are present; ``None`` when the server does not say.
    downloaded: bool | None = None

    @property
    def usable(self) -> bool:
        """Enabled, and not known to miss its files."""
        return self.enabled and self.downloaded is not False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_kinds": list(self.task_kinds),
            "enabled": self.enabled,
            "downloaded": self.downloaded,
            "usable": self.usable,
        }


class LocalInfraClient:
    """Calls one local-multimodal-infra server.

    Args:
        base_url: The server's HTTP address, e.g. ``http://127.0.0.1:17890``.
        token: Inference token (``LOCAL_MCP_INFER_TOKENS``), empty when none.
        timeout: Seconds a request may take.
        models: Model chosen per service (``tts``, ``stt``, ...); a service
            without one uses the server's choice for its task kind.
    """

    def __init__(
        self,
        base_url: str,
        token: str = "",
        timeout: float = 120,
        models: dict[str, str] | None = None,
    ) -> None:
        self.base_url = http_base({"base_url": base_url})
        self.token = token.strip()
        self.timeout = float(timeout)
        self.models = {k: v for k, v in (models or {}).items() if v}

    @classmethod
    def from_settings(cls, settings: dict | None = None) -> LocalInfraClient:
        """A client of the server in AstrBot's ``local_infra`` settings, each
        service asking for the model it resolves to (chosen or automatic).

        A chosen model the server no longer serves is still asked for: the
        server's error is better than another model's results (embeddings of
        a knowledge base must come from one model)."""
        from .catalog import catalog

        settings = settings or local_infra_settings()
        return cls(
            settings["base_url"],
            settings["token"],
            settings["timeout"],
            models={
                name: state["model"] or state["chosen"]
                for name in settings["services"]
                for state in [catalog.resolve(name, settings)]
            },
        )

    # ----- plumbing -----------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update(extra)
        return headers

    def _timeout(self, seconds: float | None = None) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(total=seconds or self.timeout)

    def url(self, path_or_url: str) -> str:
        """An absolute URL for a server path or a URL the server returned."""
        if urlsplit(path_or_url).scheme in ("http", "https"):
            return path_or_url
        return urljoin(self.base_url + "/", path_or_url.lstrip("/"))

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        async with aiohttp.ClientSession(timeout=self._timeout(timeout)) as session:
            async with session.request(
                method,
                self.url(path),
                json=json,
                data=data,
                headers=self._headers(headers),
            ) as response:
                body = await response.read()
                if response.status >= 400:
                    # Without the query: upload URLs carry their signature.
                    raise LocalInfraError(
                        f"{method} {urlsplit(path).path}: HTTP {response.status}: "
                        f"{_error_text(body)}"
                    )
                if not body:
                    return None
                try:
                    return await response.json(content_type=None)
                except ValueError as exc:
                    raise LocalInfraError(
                        f"{method} {path}: not JSON: {body[:200]!r}"
                    ) from exc

    async def rpc(self, method: str, params: dict, timeout: float | None = None) -> Any:
        """Calls a JSON-RPC inference method (``/rpc/infer``)."""
        payload = await self._request(
            "POST",
            "/rpc/infer",
            json={
                "jsonrpc": "2.0",
                "id": f"astrbot-{uuid.uuid4().hex}",
                "method": method,
                "params": params,
            },
            timeout=timeout,
        )
        if not isinstance(payload, dict):
            raise LocalInfraError(f"{method}: unexpected response {payload!r}")
        if payload.get("error"):
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise LocalInfraError(f"{method}: {message}")
        return payload.get("result")

    def model_for(self, service: str, model: str | None = None) -> str | None:
        """The model to ask for: ``model``, else the service's chosen one."""
        return model or self.models.get(service) or None

    # ----- catalog -------------------------------------------------------

    async def list_models(self, timeout: float | None = 10) -> list[InfraModel]:
        """Every model the server knows (``GET /v1/models``).

        Servers older than the model metadata list ids only: their models
        come without task kinds.
        """
        payload = await self._request("GET", "/v1/models", timeout=timeout)
        models = []
        for item in (payload or {}).get("data") or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            downloaded = item.get("downloaded")
            models.append(
                InfraModel(
                    id=str(item["id"]),
                    name=str(item.get("name") or ""),
                    task_kinds=[str(k) for k in item.get("task_kinds") or []],
                    enabled=bool(item.get("enabled", True)),
                    downloaded=downloaded if isinstance(downloaded, bool) else None,
                )
            )
        return models

    async def models_for(self, task_kind: str) -> list[InfraModel]:
        """The usable models that run ``task_kind`` (or a service name)."""
        kind = SERVICE_TASK_KINDS.get(task_kind, task_kind)
        return [
            m for m in await self.list_models() if m.usable and kind in m.task_kinds
        ]

    # ----- generic tasks -------------------------------------------------

    async def run_task(
        self,
        task_kind: str,
        *,
        model: str | None = None,
        params: dict | None = None,
        files: dict[str, TaskFile] | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Runs one task and returns its output.

        Args:
            task_kind: ``asr.transcribe``, ``tts.synthesize``, ``ocr.lines``,
                ``object.detect``, ``chat.complete``, ... (the server's
                ``list_models`` tells which take what).
            model: Model id; none lets the server pick.
            params: The task's parameters.
            files: Files by role (``audio``, ``reference_audio``, ``image``).
            timeout: Seconds to wait for it.

        Returns:
            The task's ``output`` (its ``type`` names the shape). Files it
            produced carry a ``url`` (see ``download``).

        Raises:
            LocalInfraError: The server refused or the task failed.
        """
        timeout = timeout or self.timeout
        request: dict[str, Any] = {"task_kind": task_kind, "params": params or {}}
        if model:
            request["model"] = model
        if not files:
            result = await self.rpc("run_task", request, timeout=timeout + 10)
            return _task_output(task_kind, result)

        prepared = {role: _read_task_file(role, value) for role, value in files.items()}
        request["files"] = [
            {"name": name, "mime": mime, "role": role, "required": True}
            for role, (name, _, mime) in prepared.items()
        ]
        task = await self.rpc("create_task", request)
        if not isinstance(task, dict) or not task.get("task_id"):
            raise LocalInfraError(f"create_task: unexpected result {task!r}")
        uploads = {
            str(upload.get("role") or upload.get("slot")): upload
            for upload in task.get("uploads") or []
            if isinstance(upload, dict)
        }
        for role, (_, content, mime) in prepared.items():
            upload = uploads.get(role)
            if not upload or not upload.get("upload_url"):
                raise LocalInfraError(f"create_task: no upload slot for {role!r}")
            await self._request(
                "POST",
                str(upload["upload_url"]),
                data=content,
                headers={"Content-Type": mime},
                timeout=timeout,
            )
        result = await self.rpc(
            "start_task",
            {"task_id": task["task_id"], "wait": True, "timeout_sec": int(timeout)},
            timeout=timeout + 10,
        )
        return _task_output(task_kind, result)

    async def download(
        self,
        file_ref: dict | str,
        dest_dir: str | os.PathLike | None = None,
        timeout: float | None = None,
    ) -> str:
        """Saves a file a task produced and returns its local path.

        Args:
            file_ref: The file in a task output (a dict with ``url``) or its
                URL. A ``file://`` reference on this machine is returned as
                its path, not copied.
            dest_dir: Where to save it (default: AstrBot's temp directory).
            timeout: Seconds the download may take.
        """
        url = file_ref
        if isinstance(file_ref, dict):
            url = next(
                (
                    str(file_ref[key])
                    for key in ("url", "download_url", "uri", "path")
                    if file_ref.get(key)
                ),
                "",
            )
        url = str(url or "")
        if not url:
            raise LocalInfraError(f"no file location in {file_ref!r}")
        if url.startswith("file://"):
            path = unquote(urlsplit(url).path)
            if re.match(r"^/[A-Za-z]:/", path):
                path = path[1:]
            if Path(path).is_file():
                return str(Path(path))
            raise LocalInfraError(f"{url} is not a file on this machine")
        if not urlsplit(url).scheme and Path(url).is_file():
            return url

        if dest_dir is None:
            from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

            dest_dir = get_astrbot_temp_path()
        absolute = self.url(url)
        async with aiohttp.ClientSession(timeout=self._timeout(timeout)) as session:
            async with session.get(absolute, headers=self._headers()) as response:
                if response.status >= 400:
                    raise LocalInfraError(
                        f"download: HTTP {response.status}: "
                        f"{_error_text(await response.read())}"
                    )
                content = await response.read()
                mime = response.headers.get("Content-Type", "").split(";")[0]
        suffix = Path(urlsplit(absolute).path).suffix or (
            mimetypes.guess_extension(mime) or ""
        )
        target = Path(dest_dir) / f"local_infra_{uuid.uuid4().hex}{suffix}"
        await asyncio.to_thread(_write_file, target, content)
        return str(target)

    # ----- typed helpers -------------------------------------------------

    async def synthesize(
        self,
        text: str,
        *,
        model: str | None = None,
        reference_audio: TaskFile | None = None,
        reference_text: str | None = None,
        params: dict | None = None,
        dest_dir: str | os.PathLike | None = None,
    ) -> str:
        """Speech for ``text``; returns the path of a WAV file.

        Args:
            text: What to say.
            model: TTS model (default: the ``tts`` service's).
            reference_audio: A recording whose voice is cloned.
            reference_text: What the reference recording says.
            params: More model parameters (``emotion``, ...).
            dest_dir: Where to save the WAV.
        """
        task_params = {"text": text, **(params or {})}
        if reference_text:
            task_params["reference_text"] = reference_text
        output = await self.run_task(
            "tts.synthesize",
            model=self.model_for("tts", model),
            params=task_params,
            files={"reference_audio": reference_audio} if reference_audio else None,
        )
        audio = output.get("audio") or output.get("file")
        if not audio:
            raise LocalInfraError(f"tts.synthesize returned no audio: {output!r}")
        return await self.download(audio, dest_dir)

    async def transcribe(
        self,
        audio: TaskFile,
        *,
        model: str | None = None,
        params: dict | None = None,
    ) -> dict:
        """Speech recognition.

        Args:
            audio: WAV, FLAC, MP3, AAC/M4A or Ogg audio.
            model: ASR model (default: the ``stt`` service's).
            params: Model parameters (``timestamps``,
                ``speaker_diarization``, ...).

        Returns:
            The ``asr_transcription`` output: ``text``, ``segments``
            (``start_ms``, ``end_ms``, ``text``, ``speaker``, ...) and more.
        """
        return await self.run_task(
            "asr.transcribe",
            model=self.model_for("stt", model),
            params=params,
            files={"audio": audio},
        )

    async def embed(
        self,
        texts: str | list[str],
        *,
        model: str | None = None,
        input_type: str | None = None,
    ) -> list[list[float]]:
        """Embeddings, one per text.

        ``POST /v1/embeddings`` names its model; without one known the
        server picks it in a ``text.embed`` task. ``input_type`` is ``query``
        or ``passage`` for models that tell them apart (multilingual-e5).
        """
        body: dict[str, Any] = {"input": texts}
        if input_type:
            body["input_type"] = input_type
        chosen = self.model_for("embedding", model)
        if not chosen:
            output = await self.run_task("text.embed", params=body)
            return list(output.get("embeddings") or [])
        body["model"] = chosen
        payload = await self._request("POST", "/v1/embeddings", json=body)
        data = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        top_n: int | None = None,
    ) -> list[dict]:
        """Ranks ``documents`` for ``query`` (``POST /v1/rerank``; without a
        known model, a ``text.rerank`` task the server picks one for).

        Returns:
            ``[{"index", "relevance_score", ...}]``, best first.
        """
        body: dict[str, Any] = {"query": query, "documents": documents}
        if top_n:
            body["top_n"] = top_n
        chosen = self.model_for("rerank", model)
        if not chosen:
            output = await self.run_task("text.rerank", params=body)
            return list(output.get("results") or [])
        body["model"] = chosen
        payload = await self._request("POST", "/v1/rerank", json=body)
        return list(payload.get("results") or [])

    async def moderate(self, input: Any, **options: Any) -> dict:
        """Content moderation (``POST /v1/moderations``).

        Args:
            input: Text, a list of texts, or OpenAI-style parts
                (``{"type": "image_url", ...}``).
            **options: ``threshold``, ``nsfw_threshold``, ``categories``, ...
        """
        return await self._request(
            "POST", "/v1/moderations", json={"input": input, **options}
        )

    async def ocr_lines(self, image: TaskFile, *, model: str | None = None) -> dict:
        """Text lines of an image (``ocr_lines``: ``lines[{text, bbox}]``)."""
        return await self.run_task("ocr.lines", model=model, files={"image": image})

    async def detect_objects(
        self, image: TaskFile, *, model: str | None = None
    ) -> dict:
        """Objects in an image (``object_detections``)."""
        return await self.run_task("object.detect", model=model, files={"image": image})


def _error_text(body: bytes) -> str:
    text = body.decode("utf-8", "replace").strip()
    return text[:500] or "(no body)"


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _read_task_file(role: str, value: TaskFile) -> tuple[str, bytes, str]:
    """``(name, bytes, mime)`` of a task file."""
    if isinstance(value, tuple):
        name, content, mime = value
        return str(name), bytes(content), str(mime)
    if isinstance(value, bytes | bytearray):
        return role, bytes(value), "application/octet-stream"
    path = Path(value)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.name, path.read_bytes(), mime


def _task_output(task_kind: str, result: Any) -> dict:
    if not isinstance(result, dict):
        raise LocalInfraError(f"{task_kind}: unexpected result {result!r}")
    state = result.get("state")
    if state not in (None, "succeeded"):
        raise LocalInfraError(
            f"{task_kind} {state}: {result.get('error') or 'no error given'}"
        )
    output = result.get("output")
    if not isinstance(output, dict):
        raise LocalInfraError(f"{task_kind}: no output in {result!r}")
    return output
