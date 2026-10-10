"""Chat provider backed by ephemeral Codex threads (K6).

With Codex as the only agent, plugin LLM APIs (``llm_generate``,
``tool_loop_agent``, ``get_using_provider().text_chat``) keep their
signatures and run on short-lived, non-persisted Codex threads.

Function calling keeps real provider semantics: when Codex asks for a tool,
``text_chat`` returns the tool calls to the caller (e.g. the tool loop in
``tool_loop_agent``) and the thread waits; the next ``text_chat`` carrying the
tool results resumes it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from astrbot.core import content_moderation, logger
from astrbot.core.provider.entities import LLMResponse, ToolCallsResult
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import register_provider_adapter

from .native import TERMINAL_EVENTS, CodexEngine, JsonObject
from .tool_bridge import CodexToolBridge

PROVIDER_ID = "codex"
PROVIDER_TYPE = "codex_chat"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant. Answer concisely."
PENDING_TTL_S = 15 * 60
TOOL_BATCH_QUIET_S = 0.2


@dataclass
class _Pending:
    thread_id: str
    queue: asyncio.Queue[JsonObject]
    bridge: CodexToolBridge
    futures: dict[str, asyncio.Future[JsonObject]] = field(default_factory=dict)
    created: float = field(default_factory=time.monotonic)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
                elif part.get("type") in ("image_url", "input_image"):
                    parts.append("[image]")
            elif hasattr(part, "text"):
                parts.append(str(part.text))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _as_dict(message: Any) -> dict:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return {}


def render_transcript(contexts: list | None) -> tuple[str, str]:
    """Flatten OpenAI-style history into (system text, transcript text)."""
    system: list[str] = []
    lines: list[str] = []
    for raw in contexts or []:
        msg = _as_dict(raw)
        role = msg.get("role")
        text = _content_text(msg.get("content"))
        if role == "system":
            system.append(text)
        elif role == "user":
            lines.append(f"User: {text}")
        elif role == "assistant":
            calls = msg.get("tool_calls") or []
            if calls:
                names = ", ".join(
                    str((c.get("function") or {}).get("name") or c.get("name") or "")
                    for c in calls
                    if isinstance(c, dict)
                )
                lines.append(f"Assistant (called tools: {names})")
            if text:
                lines.append(f"Assistant: {text}")
        elif role == "tool":
            lines.append(f"Tool result: {text}")
    return "\n\n".join(system), "\n\n".join(lines)


def _tool_results(
    tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None,
    contexts: list | None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    items = (
        tool_calls_result
        if isinstance(tool_calls_result, list)
        else [tool_calls_result]
        if tool_calls_result
        else []
    )
    for item in items:
        for seg in item.tool_calls_result:
            out[str(seg.tool_call_id)] = _content_text(seg.content)
    for raw in contexts or []:
        msg = _as_dict(raw)
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            out.setdefault(str(msg["tool_call_id"]), _content_text(msg.get("content")))
    return out


@register_provider_adapter(
    PROVIDER_TYPE, "Codex (in-process) chat provider", provider_display_name="Codex"
)
class CodexChatProvider(Provider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        self.runner_config: dict = provider_config.get("runner_config") or {}
        self.set_model(str(self.runner_config.get("model") or "codex"))
        self._pending: dict[str, _Pending] = {}

    # ------------------------------------------------------------ provider API

    def get_current_key(self) -> str:
        return ""

    def set_key(self, key: str) -> None:
        return None

    async def get_models(self) -> list[str]:
        return [self.get_model()]

    async def text_chat(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        func_tool=None,
        contexts: list | None = None,
        system_prompt: str | None = None,
        tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None = None,
        model: str | None = None,
        extra_user_content_parts=None,
        tool_choice="auto",
        request_max_retries: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        await self._expire()
        results = _tool_results(tool_calls_result, contexts)
        pending = next(
            (self._pending[cid] for cid in results if cid in self._pending), None
        )
        if pending is not None:
            # The plugin's own tool loop (a handoff agent, tool_loop_agent):
            # its tools' results go to the model like any tool's.
            # Only this step's: the history repeats every earlier result.
            mode = content_moderation.platform_mode("")
            results = {
                cid: await content_moderation.filter_tool_text(
                    text, mode, "plugin tool loop"
                )
                for cid, text in results.items()
                if cid in pending.futures
            }
            for cid, fut in list(pending.futures.items()):
                self._pending.pop(cid, None)
                if not fut.done():
                    fut.set_result(
                        {
                            "contentItems": [
                                {
                                    "type": "inputText",
                                    "text": results.get(cid, "(no result)"),
                                }
                            ],
                            "success": True,
                        }
                    )
            pending.futures.clear()
            return await self._drive(pending)
        # A plugin's own model call (an image caption, a summary) sends what it
        # was given just as a chat would; it belongs to no platform, so it is
        # checked in full.
        mode = content_moderation.platform_mode("")
        if await content_moderation.check_request(
            prompt or "",
            extra_user_content_parts or [],
            image_urls or [],
            mode,
            label="plugin model call",
        ):
            return LLMResponse(
                role="err", completion_text=content_moderation.blocked_reply()
            )
        if contexts and mode != content_moderation.MODE_DISABLED:
            # The history a plugin passes goes along as a transcript. As with
            # group history, a flagged user or tool turn is left out, not the
            # call.
            contexts = list(contexts)
            turns: dict[int, str] = {}
            for i, msg in enumerate(map(_as_dict, contexts)):
                if msg.get("role") not in ("user", "tool"):
                    continue
                text = _content_text(msg.get("content"))
                if msg.get("role") == "tool":
                    # As any tool result: the rest of a long one is cut off.
                    short = content_moderation.cut_to_bytes(
                        text, content_moderation.MAX_TOOL_TEXT_BYTES
                    )
                    if short != text:
                        text = short
                        contexts[i] = {
                            **msg,
                            "content": f"{short}\n[cut off: too long to check]",
                        }
                turns[i] = text
            flags = await content_moderation.flagged_texts(
                list(turns.values()), "plugin model call"
            )
            dropped = {i for i, flagged in zip(turns, flags) if flagged}
            contexts = [raw for i, raw in enumerate(contexts) if i not in dropped]
        return await self._start(
            prompt=prompt,
            image_urls=image_urls or [],
            func_tool=func_tool,
            contexts=contexts,
            system_prompt=system_prompt,
            extra_user_content_parts=extra_user_content_parts or [],
            model=model,
        )

    async def text_chat_stream(self, *args, **kwargs):
        yield await self.text_chat(*args, **kwargs)

    # --------------------------------------------------------------- internals

    async def _engine(self) -> CodexEngine:
        from .codex_agent_runner import engine_options

        return await CodexEngine.get(engine_options(self.runner_config))

    async def _start(
        self,
        *,
        prompt,
        image_urls,
        func_tool,
        contexts,
        system_prompt,
        extra_user_content_parts,
        model,
    ) -> LLMResponse:
        from .codex_agent_runner import _default_cwd, _image_input

        engine = await self._engine()
        bridge = CodexToolBridge(func_tool, defer=False)
        history_system, transcript = render_transcript(contexts)
        instructions = "\n\n".join(
            t for t in (system_prompt or "", history_system) if t and t.strip()
        )
        params = {
            "cwd": _default_cwd("codex_provider"),
            "ephemeral": True,
            "no_environment": True,
            "base_instructions": instructions or DEFAULT_INSTRUCTIONS,
            "dynamic_tools": bridge.dynamic_tools(),
            # Provider semantics: one function call per tool call, returned to
            # the caller, so no code mode here.
            "config": {"model_tool_mode": "direct"},
        }
        info = json.loads(await engine.rt.start_thread(json.dumps(params)))
        thread_id = info["thread_id"]
        pump = engine.pump(thread_id)
        pending = _Pending(thread_id=thread_id, queue=asyncio.Queue(), bridge=bridge)

        async def on_tool_call(msg: JsonObject) -> JsonObject:
            call_id = str(msg.get("callId") or msg.get("call_id") or uuid.uuid4().hex)
            fut: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
            pending.futures[call_id] = fut
            self._pending[call_id] = pending
            pending.queue.put_nowait(
                {"type": "_tool_call", "call_id": call_id, "msg": msg}
            )
            return await fut

        route_queue = pump.open_turn(on_tool_call)
        pending.queue = route_queue

        items: list[JsonObject] = []
        if transcript:
            items.append(
                {
                    "type": "text",
                    "text": f"<conversation_so_far>\n{transcript}\n</conversation_so_far>",
                    "text_elements": [],
                }
            )
        for part in extra_user_content_parts:
            text = getattr(part, "text", None)
            if text:
                items.append({"type": "text", "text": text, "text_elements": []})
        if prompt:
            items.append({"type": "text", "text": prompt, "text_elements": []})
        items.extend(_image_input(ref) for ref in image_urls)
        if not items:
            items.append({"type": "text", "text": "(empty)", "text_elements": []})
        request: JsonObject = {"input": items, "mode": "start_if_idle"}
        if model and model != self.get_model():
            request["model"] = model
        sub = await engine.submit_turn(thread_id, request)
        if sub.get("status") != "started":
            await self._close(pending)
            return LLMResponse(role="err", completion_text=f"Codex: {sub}")
        return await self._drive(pending)

    async def _drive(self, pending: _Pending) -> LLMResponse:
        texts: list[str] = []
        calls: list[JsonObject] = []
        timeout = float(self.runner_config.get("turn_timeout") or 600)
        while True:
            wait = TOOL_BATCH_QUIET_S if calls else timeout
            try:
                msg = await asyncio.wait_for(pending.queue.get(), wait)
            except asyncio.TimeoutError:
                if calls:
                    return self._tool_call_response(pending, calls)
                await self._close(pending)
                return LLMResponse(
                    role="err", completion_text="Codex request timed out"
                )
            kind = msg.get("type")
            if kind == "_tool_call":
                calls.append(msg)
            elif kind == "agent_message" and msg.get("phase") in (None, "final_answer"):
                texts.append(msg.get("message") or "")
            elif kind in TERMINAL_EVENTS or kind == "_pump_closed":
                if calls:
                    return self._tool_call_response(pending, calls)
                await self._close(pending)
                return LLMResponse(
                    role="assistant", completion_text="\n\n".join(t for t in texts if t)
                )
            elif kind == "error":
                logger.warning("Codex provider error: %s", msg.get("message"))

    def _tool_call_response(
        self, pending: _Pending, calls: list[JsonObject]
    ) -> LLMResponse:
        names, args, ids = [], [], []
        for call in calls:
            msg = call["msg"]
            tool = pending.bridge.lookup(
                msg.get("namespace"), str(msg.get("tool") or "")
            )
            names.append(tool.name if tool else str(msg.get("tool") or ""))
            arguments = msg.get("arguments")
            args.append(arguments if isinstance(arguments, dict) else {})
            ids.append(call["call_id"])
        return LLMResponse(
            role="assistant",
            tools_call_args=args,
            tools_call_name=names,
            tools_call_ids=ids,
        )

    async def _close(self, pending: _Pending) -> None:
        for cid, fut in pending.futures.items():
            self._pending.pop(cid, None)
            if not fut.done():
                fut.set_result({"contentItems": [], "success": False})
        with contextlib.suppress(Exception):
            engine = await self._engine()
            pump = engine.pumps.get(pending.thread_id)
            if pump is not None:
                pump.close_turn()
            await engine.forget_thread(pending.thread_id)

    async def _expire(self) -> None:
        now = time.monotonic()
        stale = {
            id(p): p for p in self._pending.values() if now - p.created > PENDING_TTL_S
        }
        for pending in stale.values():
            logger.info(
                "Dropping abandoned Codex provider thread %s", pending.thread_id
            )
            await self._close(pending)


def make_codex_provider(runner_config: dict) -> CodexChatProvider:
    return CodexChatProvider(
        {
            "id": PROVIDER_ID,
            "type": PROVIDER_TYPE,
            "enable": True,
            "model": runner_config.get("model") or "codex",
            "runner_config": runner_config,
        },
        {},
    )
