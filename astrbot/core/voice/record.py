"""What a voice conversation leaves behind: a conversation record and stats.

A voice thread (kept per conversation key, from call to call) has a
conversation of its own under the paired chat's UMO, next to the chat's
conversations: it holds the transcript (what was heard and what was said)
and is what the thread's stats rows point at. It is never made the chat's
current conversation. On the local-multimodal-infra backend, "heard" is only
what the voice model was given (Codex's ``InputDelivered``: the lines an
input carried, each with its time); talk that wanted no reply and never
reached the model is left out.

Every turn of the voice thread (on the local-multimodal-infra backend: its
replies and compactions) is a stats row of agent type ``codex_voice`` with
the thread's model. The realtime backend's model reports no usage to Codex,
so its calls leave the transcript only.

Recording never breaks a call: every failure is logged and skipped.
"""

from __future__ import annotations

import asyncio
import time
import typing as T

from astrbot import logger
from astrbot.core import db_helper
from astrbot.core.agent.runners.codex.constants import CODEX_VOICE_STATS_TYPE
from astrbot.core.agent.runners.codex.native import TERMINAL_EVENTS
from astrbot.core.agent.runners.codex.usage import (
    FIRST_TOKEN_EVENTS,
    UsageMeter,
    thread_usage,
)

if T.TYPE_CHECKING:
    from astrbot.core.agent.runners.codex.native import CodexEngine

# Waits for the last transcript and stats writes when the call ends.
FLUSH_TIMEOUT = 10.0


async def voice_conversation(umo: str, cid: str | None, title: str) -> str | None:
    """The voice thread's conversation record, created when it has none (or
    it was deleted); the chat's current conversation stays as it is.

    Args:
        umo: The paired chat's UMO.
        cid: The record the thread had, if any.
        title: Title of a new record.

    Returns:
        The record's id, or None if it could not be made.
    """
    try:
        if cid and await db_helper.get_conversation_by_id(cid=cid):
            return cid
        conv = await db_helper.create_conversation(
            user_id=umo, platform_id=umo.split(":", 1)[0] or "unknown", title=title
        )
        return conv.conversation_id
    except Exception as exc:  # noqa: BLE001 - the call goes on without it
        logger.warning("voice conversation record for %s failed: %s", umo, exc)
        return None


class _Turn:
    def __init__(self, total: dict | None) -> None:
        self.start = time.time()
        self.ttft = 0.0
        self.failed = False
        self.meter = UsageMeter()
        self.meter.start({"total_token_usage": total})


class VoiceRecord:
    """Follows a voice thread's events: the transcript goes to its
    conversation record, each turn's usage to the stats."""

    def __init__(
        self,
        umo: str,
        cid: str | None,
        engine: CodexEngine,
        thread_id: str,
        label: str,
        delivered_only: bool = False,
    ) -> None:
        self.umo = umo
        # Heard lines come from InputDelivered, not every transcript.
        self._delivered_only = delivered_only
        self.cid = cid
        self._engine = engine
        self._thread_id = thread_id
        self._label = label
        # Transcript messages not written yet.
        self._lines: list[dict] = []
        # The thread's cumulative usage; None until known (then no stats).
        self._usage: dict | None = None
        self._total: dict | None = None
        self._turn: _Turn | None = None
        self._context_tokens = 0
        self._writes: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Takes the thread's usage so far, the baseline of its next turn."""
        self._usage = await thread_usage(self._engine, self._thread_id)
        if self._usage is not None:
            self._total = self._usage.get("total_token_usage")

    def event(self, msg: dict) -> None:
        """One event of the voice thread."""
        kind = msg.get("type")
        if kind == "realtime_conversation_realtime":
            payload = msg.get("payload")
            if not isinstance(payload, dict):
                return
            if (sent := payload.get("InputDelivered")) is not None:
                if self._delivered_only:
                    for line in sent.get("lines") or ():
                        self._line("user", {"text": line})
            elif (done := payload.get("InputTranscriptDone")) is not None:
                if not self._delivered_only:
                    self._line("user", done)
            elif (done := payload.get("OutputTranscriptDone")) is not None:
                self._line("assistant", done)
                self._write(self._flush())
        elif kind in ("task_started", "turn_started"):
            if self._usage is not None:
                self._turn = _Turn(self._total)
        elif kind == "token_count":
            info = msg.get("info") or {}
            if (total := info.get("total_token_usage")) is not None:
                self._total = total
                if self._turn is not None:
                    self._turn.meter.observe(total)
            last = info.get("last_token_usage") or {}
            if int(last.get("input_tokens") or 0) > 0:
                self._context_tokens = int(last.get("total_tokens") or 0)
        elif kind in FIRST_TOKEN_EVENTS:
            if self._turn is not None and not self._turn.ttft:
                self._turn.ttft = time.time() - self._turn.start
        elif kind == "error":
            if self._turn is not None:
                self._turn.failed = True
        elif kind in TERMINAL_EVENTS:
            turn, self._turn = self._turn, None
            if turn is None:
                return
            ttft_ms = msg.get("time_to_first_token_ms")
            if isinstance(ttft_ms, int | float) and ttft_ms > 0:
                turn.ttft = ttft_ms / 1000
            if kind == "turn_aborted":
                status = "aborted"
            elif turn.failed or msg.get("error"):
                status = "error"
            else:
                status = "completed"
            self._write(self._stat(turn, status, time.time()))

    def _line(self, role: str, done: dict) -> None:
        text = str(done.get("text") or "").strip()
        if text:
            self._lines.append({"role": role, "content": text})

    def _write(self, coro) -> None:
        task = asyncio.create_task(coro, name=f"{self._label}-voice-record")
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _flush(self) -> None:
        """Appends the transcript not written yet to the record."""
        async with self._lock:
            lines, self._lines = self._lines, []
            if not lines or not self.cid:
                return
            try:
                conv = await db_helper.get_conversation_by_id(cid=self.cid)
                if conv is None:
                    return
                await db_helper.update_conversation(
                    cid=self.cid,
                    content=[*(conv.content or []), *lines],
                    token_usage=self._context_tokens or None,
                )
            except Exception as exc:  # noqa: BLE001 - lost, the call goes on
                logger.warning(
                    "%s voice transcript not saved for %s: %s",
                    self._label,
                    self.umo,
                    exc,
                )

    async def _stat(self, turn: _Turn, status: str, end: float) -> None:
        """Stores one turn's stats row, with the model it ran on."""
        after = await thread_usage(self._engine, self._thread_id)
        if after is not None:
            self._usage = after
            turn.meter.observe(after.get("total_token_usage"))
        usage = self._usage or {}
        spent = turn.meter.spent
        try:
            await db_helper.insert_provider_stat(
                umo=self.umo,
                conversation_id=self.cid,
                provider_id=str(usage.get("model_provider") or CODEX_VOICE_STATS_TYPE),
                provider_model=str(usage["model"]) if usage.get("model") else None,
                status=status,
                stats={
                    "token_usage": {
                        "input_other": spent.input_other if spent else 0,
                        "input_cached": spent.input_cached if spent else 0,
                        "output": spent.output if spent else 0,
                    },
                    "start_time": turn.start,
                    "end_time": end,
                    "time_to_first_token": turn.ttft,
                },
                agent_type=CODEX_VOICE_STATS_TYPE,
            )
        except Exception as exc:  # noqa: BLE001 - stats never break a call
            logger.warning(
                "%s voice stats not saved for %s: %s", self._label, self.umo, exc
            )

    async def close(self) -> None:
        """Writes what is left (the transcript, a turn cut by the end)."""
        if (turn := self._turn) is not None:
            self._turn = None
            self._write(self._stat(turn, "aborted", time.time()))
        self._write(self._flush())
        if self._writes:
            await asyncio.wait(set(self._writes), timeout=FLUSH_TIMEOUT)
