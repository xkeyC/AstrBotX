"""Full-duplex voice through Codex realtime.

Which realtime backend carries it is a setting of the Codex runner
(``realtime_voice.backend``): this module's ``VoiceSession`` talks to an
OpenAI realtime model over WebRTC; ``infra.InfraVoiceSession`` has Codex
connect to a local-multimodal-infra server. ``new_voice_session`` picks it.

The realtime model listens and speaks. What it hands off runs as a turn of
the paired chat (see ``chat``): the chat's own thread, context, persona,
tools and memory, queued with the chat's text messages; the answer comes back
to be spoken. The realtime conversation itself is carried by a thread of its
own (Codex attaches realtime to a thread), persisted per conversation key; it
has no tools and runs no turns.

The platform supplies the audio as a ``VoiceMedia``: Mumble mixes Opus
streams, a phone bridge carries raw PCM (see ``pcm``).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)

from astrbot import logger
from astrbot.core import astrbot_config, sp
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .chat import TASK_BODY, VOICE_SESSIONS, VoiceChat, deliver
from .icetcp import IceTcpRelay, replace_candidates, tcp_candidates
from .record import VoiceRecord, voice_conversation

VOICE_THREAD_KEY = "voice_thread"
# Voices of realtime v1/v3, which subscription (WebRTC) sessions use.
REALTIME_VOICES = (
    "juniper",
    "maple",
    "spruce",
    "ember",
    "vale",
    "breeze",
    "arbor",
    "sol",
    "cove",
)
# Overrides of the thread carrying a realtime conversation, on top of the
# runner's engine config. It runs no turns: handoffs go to the paired chat.
VOICE_THREAD_CONFIG = {
    # Codex reports handoffs (HandoffRequested) and starts no turn for them.
    "realtime.host_routes_handoffs": True,
    "features.apps": False,
    "features.memories": False,
    "features.image_generation": False,
    "agents.enabled": False,
    "features.multi_agent_v2": False,
}
# Given to the realtime model as the backend's word on a request: while it
# waits for the chat's turn, when it ended without an answer, and when it
# could not be answered. The model tells the listener in its own words.
BUSY_SPEECH = "Still busy with an earlier request; this one is next, the answer follows as soon as it is done."
DONE_SPEECH = "Done; there is nothing more to say about it."
FAILED_SPEECH = "That request could not be completed."
# A result reaching the chat later (background work, a scheduled task).
NOTE_PROMPT = """(Note from the backend, not said by the listener: {text}
Tell the listener about it if it matters to them, briefly and in your own words; otherwise say nothing.)"""
SDP_TIMEOUT = 30.0
CONNECT_TIMEOUT = 15.0
# Pause between WebRTC connecting and handing the model the audio kept
# meanwhile. Core reports the conversation started before the SDP answer, so
# there is no later signal to wait for; sent right away, the start is lost.
READY_DELAY = 0.5
# How long a close waits for a start in progress to reach a safe point.
CLOSE_WAIT = 10.0

VOICE_THREAD_INSTRUCTIONS = """This thread only carries a realtime voice conversation; requests are handled elsewhere."""

# Appended to every realtime prompt: the realtime model has no clock of its own.
TIME_PROMPT = """Today is {date} ({weekday}), time zone {timezone}; the current time was {time} when this conversation started. Your own knowledge is older than that: anything about current events, news, prices, weather, schedules or other recent or changing information must be delegated to the backend, never answered from memory."""


def _patch_ice_candidates() -> None:
    """Skips host candidates that cannot carry WebRTC media.

    Proxy TUN adapters (fake-IP range 198.18.0.0/15) and link-local
    addresses get nominated first on machines that have them, and the DTLS
    handshake then stalls, so they are left out of ICE gathering.
    """
    import aioice.ice as ice

    if getattr(ice.get_host_addresses, "_astrbot_filtered", False):
        return
    original = ice.get_host_addresses
    fake_ip = ipaddress.ip_network("198.18.0.0/15")

    def filtered(use_ipv4: bool, use_ipv6: bool) -> list[str]:
        addresses = []
        for address in original(use_ipv4, use_ipv6):
            ip = ipaddress.ip_address(address)
            if not ip.is_link_local and ip not in fake_ip:
                addresses.append(address)
        return addresses

    filtered._astrbot_filtered = True  # type: ignore[attr-defined]
    ice.get_host_addresses = filtered


_patch_ice_candidates()


def _local_address() -> str:
    """A host address of ours that aioice also uses, for the relay candidate."""
    import aioice.ice as ice

    addresses = list(ice.get_host_addresses(True, False))
    return addresses[0] if addresses else "127.0.0.1"


# aioice logs every connectivity check at INFO.
logging.getLogger("aioice").setLevel(logging.WARNING)


def _disable_consent_expiry() -> None:
    """Keeps aioice from dropping a working call on a slow path.

    aioice checks consent (RFC 7675) every ~5 s with a single STUN request
    that waits about 0.5 s and is never retransmitted, and closes the ICE
    connection after 6 misses. On a path with a round trip above that (a
    proxy, a distant network) every check misses and a healthy call is torn
    down after ~30 s. Consent freshness guards browsers against being used to
    send traffic; here the end of a call is known from the peer closing DTLS,
    from Codex reporting the realtime session closed, and from standby.
    """
    import aioice.ice as ice

    ice.CONSENT_FAILURES = 1_000_000_000


_disable_consent_expiry()


class VoiceMedia(Protocol):
    """The platform side of a voice session's audio."""

    # What the model hears, served at real-time pace.
    track: MediaStreamTrack

    async def play(self, track: MediaStreamTrack) -> None:
        """Consumes the model's audio track until it ends."""

    def start(self) -> None:
        """The model listens now: hand over what was held back meanwhile."""

    def stop(self) -> None:
        """The session is closing: drop pending audio, end speech in progress."""

    def flush(self) -> None:
        """Drops the model's audio not played yet (its speech was cut)."""


WAKE_MODES = ("auto", "always", "off")


def wakes_on_name(others: int | None, mode: str = "auto") -> bool:
    """Whether a group conversation with ``others`` other people (None: not
    known) hears only what calls the bot by name (with what was said just
    before it). ``mode`` (``VoiceOptions.wake_mode``): ``auto``, with one
    other person (or none) it hears everything, with more (or not known)
    only that; ``always``, only that whoever is there; ``off``, everything
    (the voice model tells for itself what is for it)."""
    if mode == "off":
        return False
    if mode == "always":
        return True
    return others is None or others > 1


def group_rule(options: VoiceOptions, *, gated: bool) -> str:
    """When the voice model of a group conversation (a channel, a room)
    speaks. ``gated``: the voice server passes on only what calls the bot by
    name, everything when one person is left to talk with (the
    ``local_infra`` backend); else the model hears all the talk and must
    tell for itself."""
    aliases = [a for a in options.aliases if a and a != options.name]
    names = (
        f' ("{options.name}"' + "".join(f', "{a}"' for a in aliases) + ")"
        if aliases
        else f' "{options.name}"'
    )
    if gated:
        return (
            f"What you are given to answer is meant for you: with several people around, only what "
            f"someone says calling you by your name{names} reaches you (with what others said just "
            f"before it, as context: not to you); with just one other person, all they say does. "
            f"Answer it, unless it is not for you after all (your name only came up in their talk, "
            f"or they are talking to someone else, or on the phone): then produce no audio and no "
            f"text at all - complete silence."
        )
    return (
        f"The one rule that matters most: speak ONLY when the speaker says your name{names} to "
        f"you in that utterance, or is directly continuing an exchange with you from a few seconds "
        f"ago. In every other case produce no audio and no text at all - complete silence. Do not "
        f'acknowledge, do not react, do not say "mm", do not comment, do not delegate.'
    )


@dataclass
class VoiceOptions:
    name: str
    aliases: list[str]
    # Appended to the voice model's prompt unless the paired chat's persona
    # has a voice persona.
    extra_prompt: str = ""
    # Realtime media over the peer's ICE-TCP candidate even without a proxy:
    # no packet loss on lossy paths, at the cost of latency spikes (which a
    # playout buffer absorbs). With a proxy, media always takes TCP.
    media_tcp: bool = False
    # When a group conversation hears only what calls the bot by name
    # (``wakes_on_name``): ``auto`` (by how many people are there),
    # ``always`` or ``off``.
    wake_mode: str = "auto"


@dataclass
class VoiceTool:
    """A quick action of the platform the voice model calls itself (wave,
    jump, write a line), answered at once instead of handed to the chat.

    Only a voice thread that runs its own tools has them: the
    ``local_infra`` backend. Codex realtime hands everything off.

    Attributes:
        spec: The dynamic tool: ``name``, ``description``, ``inputSchema``.
        run: Does it with the call's arguments; returns the result for the
            model: a text, or Codex content items (``{"type": "inputText",
            "text"}``, ``{"type": "inputImage", "imageUrl"}`` with a data
            URL) to show it a picture. An exception fails the call.
        ends_turn: Whether a call ends the turn unless its ``end_turn``
            argument says otherwise (the model chooses per call when the
            tool offers that argument). A call that ends the turn is the
            whole answer: no model round follows it (when every call of the
            reply ends the turn), and its ``say`` argument, if any, is
            spoken as the reply. Not for tools whose result the model must
            tell.
    """

    spec: dict
    run: Callable[[dict], Awaitable[str | list[dict]]]
    ends_turn: bool = False


def _runner_config() -> dict:
    from astrbot.core.config.agent_runner import normalize_agent_runner

    return normalize_agent_runner(astrbot_config.get("agent_runner"))["config"]


def realtime_voice_config() -> dict:
    """The runner's realtime voice settings (``realtime_voice``)."""
    return _runner_config()["realtime_voice"]


def new_voice_session(**kwargs) -> VoiceSession:
    """A voice session on the configured realtime backend.

    Args:
        **kwargs: As for ``VoiceSession``.

    Returns:
        A ``VoiceSession`` (Codex realtime) or an ``InfraVoiceSession``
        (local-multimodal-infra), not started yet.
    """
    if realtime_voice_config()["backend"] == "local_infra":
        from .infra import InfraVoiceSession

        return InfraVoiceSession(**kwargs)
    return VoiceSession(**kwargs)


async def _codex_engine():
    """The Codex engine the chat runner uses: same runtime, same account."""
    from astrbot.core.agent.runners.codex.codex_agent_runner import engine_options
    from astrbot.core.agent.runners.codex.native import CodexEngine

    engine = await CodexEngine.get(engine_options(_runner_config()))
    if not hasattr(engine.rt, "realtime_start"):
        raise RuntimeError(
            "codex_astrbot binding has no realtime support; update codex-astrbot"
        )
    return engine


class _SessionClosed(Exception):
    """The session was closed while it was still starting."""


class VoiceSession:
    """One realtime conversation, its tasks run by the paired chat.

    Lifecycle: ``launch()`` starts it in the background; ``close()`` may be
    called at any time, from any path (standby, mute, disconnect, a failure),
    and runs once. A close during start lets the start stop at its next step
    and then releases whatever it had created, so nothing outlives the
    session (in particular no realtime call keeps running unowned).

    Another voice model replaces the transport (``_open_agent``,
    ``_connect``, ``_release_transport``, ``say``) and hands its tasks to
    the same chat.
    """

    def __init__(
        self,
        key: str,
        scope_id: str,
        prompt: str,
        options: VoiceOptions,
        media: VoiceMedia,
        on_closed: Callable[[VoiceSession], None],
        chat: VoiceChat,
        label: str = "voice",
        thread_key: str = VOICE_THREAD_KEY,
        tools: list[VoiceTool] | None = None,
        on_wake: Callable[..., Awaitable[None] | None] | None = None,
    ) -> None:
        """
        Args:
            key: Conversation key within the platform, e.g. ``server``.
            scope_id: Storage scope of the persisted voice thread.
            prompt: Instructions for the realtime model (who and where it
                is); the session adds the voice persona and the time.
            options: Voice settings of the platform.
            media: The platform's audio in and out.
            on_closed: Called once the session has ended, for any reason.
            chat: The paired chat, which runs what the voice model hands off.
            label: Names the session in logs and task names, e.g. ``Mumble``.
            thread_key: Storage key of the persisted voice thread.
            tools: The platform's quick actions for the voice model (see
                ``VoiceTool``); ignored by Codex realtime.
            on_wake: Called (or awaited) with ``source=`` when the bot is
                called by name, at most once every few seconds: ``"wake"``
                as soon as the voice server hears a wake word,
                ``"transcript"`` when only the utterance's transcript says
                so (later: it comes once the utterance ended). If it takes
                ``speaker=`` (or ``**kwargs``), that is who the server
                guesses said it (the transcript's speaker; None on a wake
                word, heard before the transcript). E.g. to turn toward the
                speaker. Only the ``local_infra`` backend calls it.
        """
        self.tools = list(tools or [])
        self._on_wake = on_wake
        self.key = key
        self.scope_id = scope_id
        self.prompt = prompt
        # The voice persona (or the platform's extra prompt) the start added
        # to the prompt.
        self.persona = ""
        self.options = options
        self.chat = chat
        # Requests handed to the chat and not answered yet.
        self._pending = 0
        self.last_answer_at = 0.0
        self.media = media
        self.label = label
        self.thread_key = thread_key
        self._on_closed = on_closed
        self._engine = None
        self._thread_id: str | None = None
        self._events_queue: asyncio.Queue | None = None
        self._realtime_requested = False
        self._thread_released = False
        # The platform's latest context for the voice model (``set_context``).
        self._context = ""
        # The context the model has (with the start, or given since); one
        # sender at a time, so the newest is what it ends with.
        self._context_given = ""
        self._context_lock = asyncio.Lock()
        # Hear only what calls the bot by name (``set_people``), and what
        # the voice server was last told.
        self._wake = wakes_on_name(None, options.wake_mode)
        self._wake_given: bool | None = None
        self._wake_lock = asyncio.Lock()
        # The transcript and stats of the thread (see record.py).
        self._record: VoiceRecord | None = None
        self._pc: RTCPeerConnection | None = None
        self._relay: IceTcpRelay | None = None
        self._tasks: list[asyncio.Task] = []
        self._start_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        self._closed_event = asyncio.Event()
        self._closed = False
        # Why the conversation ended while it was still starting (reported
        # as the start's failure).
        self._start_error: Exception | None = None
        self.created_at = time.monotonic()
        # Set once the model listens; standby only counts from then.
        self.ready = False
        self.started_at = 0.0
        self.last_transcript_at = 0.0

    @property
    def last_activity(self) -> float:
        """Start, latest speech recognised by the model or latest answer;
        now while a request waits for its answer. For standby and idle
        hang-up."""
        if self._pending:
            return time.monotonic()
        return max(self.started_at, self.last_transcript_at, self.last_answer_at)

    def launch(self, on_failed: Callable[[Exception], None]) -> None:
        """Starts the session in the background.

        Args:
            on_failed: Called when starting fails (not when it is closed).
        """

        async def run() -> None:
            try:
                await self._start()
            except _SessionClosed:
                if self._start_error is not None:
                    on_failed(self._start_error)
                return
            except Exception as exc:  # noqa: BLE001 - reported to the owner
                if self._closed and self._start_error is None:
                    return
                on_failed(self._start_error or exc)
                # Not awaited: the close waits for this very task to end.
                self._request_close(f"start failed: {exc}")

        self._start_task = asyncio.create_task(
            run(), name=f"{self.label}-voice-{self.key}-start"
        )

    def _phase(self, name: str) -> None:
        logger.debug(
            "%s voice %s: %s after %.1fs",
            self.label,
            self.key,
            name,
            time.monotonic() - self.created_at,
        )

    def _check_open(self) -> None:
        if self._closed:
            raise _SessionClosed

    async def _wait_open(self, awaitable, timeout: float):
        """Awaits ``awaitable``, giving up as soon as the session is closed.

        Raises:
            _SessionClosed: The session was closed first.
            asyncio.TimeoutError: ``timeout`` passed first.
        """
        waiter = asyncio.ensure_future(awaitable)
        closed = asyncio.ensure_future(self._closed_event.wait())
        try:
            done, _ = await asyncio.wait(
                {waiter, closed}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            closed.cancel()
        if waiter in done:
            return waiter.result()
        waiter.cancel()
        if self._closed:
            raise _SessionClosed
        raise asyncio.TimeoutError

    async def _start(self) -> None:
        # The voice persona of the paired chat's persona, or the platform's
        # extra prompt, completes the voice model's instructions.
        extra = await self.chat.voice_persona() or self.options.extra_prompt
        self._check_open()
        self.persona = extra
        if extra:
            self.prompt = f"{self.prompt}\n\n{extra}"
        await self._open_agent()
        await self._connect()
        self._check_open()
        # Results reaching the chat later are given to this conversation.
        VOICE_SESSIONS[self.chat.umo] = self

    async def _open_agent(self) -> None:
        """Opens (or resumes) the thread carrying the realtime conversation
        and routes its events to
        ``self._events_queue``."""
        engine = await _codex_engine()
        self._check_open()
        self._engine = engine
        state = await sp.get_async(
            scope="umo", scope_id=self.scope_id, key=self.thread_key, default={}
        )
        self._check_open()
        workspace = Path(get_astrbot_data_path()) / "voice"
        workspace.mkdir(parents=True, exist_ok=True)
        params = {"cwd": str(workspace), **self._thread_params()}
        # A resumed thread takes these params' instructions and tools (Codex
        # prefers the given base_instructions over the rollout's), so a
        # changed prompt reaches it with the next open; the rollout keeps
        # only the first session's instructions, so it is no evidence.
        # Opening and unloading this key's thread are serialised: a session
        # closed while its open was still running unloads the thread before
        # anyone else may open it, so it can never unload a newer session's
        # (the same thread id is resumed for the same key).
        async with engine.session_lock(self.scope_id):
            info, started_new = await engine.open_thread(state or None, params)
            self._thread_id = info["thread_id"]
            self._phase("thread opened")
            logger.info(
                "%s voice thread for %s: %s (%s)",
                self.label,
                self.key,
                self._thread_id,
                "new" if started_new else "resumed",
            )
            if self._closed:
                self._thread_released = True
                await engine.forget_thread(self._thread_id)
                raise _SessionClosed
            # Taken under the lock: an older session of this key releasing
            # the thread sees it is ours and leaves it (see _release). This
            # thread only ever carries the voice conversation, so its pump
            # route stays open for the whole session and sees every event.
            self._events_queue = engine.pump(self._thread_id).open_turn(
                self._tool_handler(), None
            )
        same_thread = state.get("thread_id") == self._thread_id
        cid = await voice_conversation(
            self.chat.umo,
            state.get("conversation_id") if same_thread else None,
            f"Voice: {self.label}",
        )
        if started_new or not same_thread or state.get("conversation_id") != cid:
            await sp.put_async(
                scope="umo",
                scope_id=self.scope_id,
                key=self.thread_key,
                value={
                    "thread_id": info["thread_id"],
                    "rollout_path": info.get("rollout_path"),
                    "conversation_id": cid,
                },
            )
        record = VoiceRecord(self.chat.umo, cid, engine, self._thread_id, self.label)
        await record.start()
        self._record = record
        self._check_open()

    def _thread_params(self) -> dict:
        """The parameters of the thread carrying the conversation (it runs
        no turns here: handoffs go to the paired chat)."""
        return {
            "base_instructions": VOICE_THREAD_INSTRUCTIONS,
            "dynamic_tools": [],
            "no_environment": True,
            "config": dict(VOICE_THREAD_CONFIG),
        }

    def _tool_handler(self):
        """Answers the thread's dynamic tool calls (none here)."""
        return None

    async def _connect(self) -> None:
        """Starts the realtime conversation over WebRTC on the agent thread."""
        engine, events = self._engine, self._events_queue
        # No STUN: the far end offers public host candidates and we connect
        # out to them. aiortc's default Google STUN server only adds a
        # multi-second wait while gathering.
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._pc = pc
        pc.addTrack(self.media.track)
        pc.createDataChannel("oai-events")

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                self._spawn(self.media.play(track), "outbound")

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            if pc.connectionState in ("failed", "closed"):
                self._request_close(f"webrtc {pc.connectionState}")

        await pc.setLocalDescription(await pc.createOffer())
        self._phase("offer ready")
        self._check_open()
        request: dict = {
            "transport": {"type": "webrtc", "sdp": pc.localDescription.sdp},
            # Subscription (AVAS) calls only accept the frameless protocol.
            "version": "v3",
            # The answers come from the paired chat and are spoken here.
            "client_managed_handoffs": True,
            "include_startup_context": False,
            # The realtime model has no clock of its own; the platform's
            # context so far (set_context) goes with the start.
            "prompt": "\n\n".join(
                part for part in (self.prompt, time_prompt(), self._context) if part
            ),
        }
        self._context_given = self._context
        settings = realtime_voice_config()
        if voice := str(settings["voice"]).strip():
            request["voice"] = voice
        if model := str(settings["model"]).strip():
            request["model"] = model
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._spawn(self._events(events, answer), "events")
        self._realtime_requested = True
        await engine.rt.realtime_start(self._thread_id, json.dumps(request))
        sdp = await self._wait_open(answer, SDP_TIMEOUT)
        self._phase("answer received")
        proxy = str(_runner_config().get("proxy") or "").strip()
        if proxy or self.options.media_tcp:
            # Media cannot take a proxy over UDP: go through the peer's
            # ICE-TCP candidates, one TCP connection (via the proxy, if set).
            candidates = tcp_candidates(sdp)
            if not candidates:
                raise RuntimeError(
                    "media over TCP, but the peer offered no ICE-TCP candidate"
                )
            self._relay = IceTcpRelay(
                proxy,
                candidates,
                on_lost=lambda: self._request_close("media connection lost"),
            )
            port = await self._wait_open(self._relay.start(), CONNECT_TIMEOUT)
            sdp = replace_candidates(sdp, _local_address(), port)
            self._phase("media relay over TCP ready")
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))
        deadline = time.monotonic() + CONNECT_TIMEOUT
        while pc.connectionState != "connected":
            self._check_open()
            if time.monotonic() > deadline or pc.connectionState in (
                "failed",
                "closed",
            ):
                raise RuntimeError(f"WebRTC did not connect ({pc.connectionState})")
            await asyncio.sleep(0.1)
        self._phase("webrtc connected")
        with contextlib.suppress(asyncio.TimeoutError):
            await self._wait_open(asyncio.sleep(READY_DELAY), READY_DELAY + 1)
        self.media.start()
        self.started_at = time.monotonic()
        self.ready = True
        await self._give_context()
        logger.info(
            "%s voice session %s started in %.1fs (thread %s)",
            self.label,
            self.key,
            self.started_at - self.created_at,
            self._thread_id,
        )

    async def say(self, text: str) -> None:
        """Gives the realtime model a text input, which it answers aloud.

        Args:
            text: What the model is told, e.g. why the bot placed a call.

        Raises:
            RuntimeError: The session is not ready.
        """
        if not self.ready or self._closed or self._engine is None:
            raise RuntimeError("voice session is not ready")
        await self._engine.rt.realtime_append_text(self._thread_id, text)

    def _spawn(self, coro, name: str) -> None:
        task = asyncio.create_task(coro, name=f"{self.label}-voice-{self.key}-{name}")
        self._tasks.append(task)

    async def _events(self, events: asyncio.Queue, answer: asyncio.Future) -> None:
        while True:
            msg = await events.get()
            if self._record is not None:
                self._record.event(msg)
            kind = msg.get("type")
            if kind == "realtime_conversation_sdp":
                if not answer.done():
                    answer.set_result(msg["sdp"])
            elif kind == "realtime_conversation_closed":
                self._conversation_ended(
                    answer, f"realtime closed: {msg.get('reason') or 'closed'}"
                )
                return
            elif kind == "realtime_conversation_realtime":
                payload = msg.get("payload")
                if isinstance(payload, dict) and "Error" in payload:
                    logger.warning(
                        "%s voice %s: realtime error: %s",
                        self.label,
                        self.key,
                        payload["Error"],
                    )
                    if not answer.done():
                        answer.set_exception(RuntimeError(str(payload["Error"])))
                elif isinstance(payload, dict) and "InputTranscriptDelta" in payload:
                    self.last_transcript_at = time.monotonic()
                elif isinstance(payload, dict) and "HandoffRequested" in payload:
                    self._spawn(self._handoff(payload["HandoffRequested"]), "handoff")
                elif isinstance(payload, dict) and "InputTranscriptDone" in payload:
                    self.last_transcript_at = time.monotonic()
                    logger.debug(
                        "%s voice %s heard: %s",
                        self.label,
                        self.key,
                        payload["InputTranscriptDone"].get("text"),
                    )
            elif kind == "_pump_closed":
                self._conversation_ended(answer, "voice thread closed")
                return

    def _conversation_ended(self, pending: asyncio.Future, reason: str) -> None:
        """The conversation ended: the session closes; one still starting
        fails its start with ``reason`` (reported by ``launch``).

        Args:
            pending: What the start waits for (the SDP answer, the server's
                start), failed with the reason if it is still pending.
            reason: Why it ended.
        """
        self._realtime_requested = False
        # A close the owner asked for is no failure.
        if not self.ready and not self._closed and self._start_error is None:
            if not pending.done():
                self._start_error = RuntimeError(reason)
                pending.set_exception(self._start_error)
            elif not pending.cancelled() and pending.exception() is not None:
                # Failed already (a server error): that is the reason.
                self._start_error = pending.exception()
            else:
                self._start_error = RuntimeError(reason)
        self._request_close(reason)

    async def _handoff(self, handoff: dict) -> None:
        """Has the paired chat answer a handoff, and the model speak it."""
        heard = next(
            (
                str(entry.get("text") or "")
                for entry in reversed(handoff.get("active_transcript") or [])
                if entry.get("role") == "user"
            ),
            "",
        )
        task = str(handoff.get("input_transcript") or heard)
        logger.info("%s voice %s: handoff %r", self.label, self.key, task)

        async def tell(answer: str | None) -> None:
            # A later handoff may be pending by now: say what this answers.
            if answer:
                answer = f'Answer to "{task}": {answer}'
            await self._tell(answer)

        if self._ask(TASK_BODY.format(heard=heard or task, task=task), tell):
            self._spawn(self._speak(BUSY_SPEECH), "busy")

    def _ask(
        self, body: str, tell=None, keep: bool = True, speaker: str | None = None
    ) -> bool:
        """Hands ``body`` to the paired chat (in order, outliving this
        session); the answer goes to ``tell`` (default ``_tell``).

        Args:
            body: The request.
            tell: Gets the answer while this conversation is on.
            keep: Deliver the answer elsewhere if the conversation ended
                (not for words only meant for it, like an opening).
            speaker: Who the platform guesses asked, in a group (see
                ``VoiceChat.ask``).

        Returns:
            Whether it waits behind other work.
        """
        self._pending += 1
        tell = tell or self._tell

        async def answered(answer: str | None) -> None:
            self._pending -= 1
            self.last_answer_at = time.monotonic()
            if self._closed:
                # The conversation ended meanwhile: the answer is not lost.
                if keep:
                    await deliver(self.chat.umo, answer or "")
                return
            await tell(answer)

        return self.chat.request(body, answered, speaker=speaker)

    async def _tell(self, answer: str | None) -> None:
        """Gives the voice model a request's answer (None: it failed)."""
        if answer is None:
            answer = FAILED_SPEECH
        await self._speak(answer or DONE_SPEECH)

    async def set_context(self, text: str) -> None:
        """Gives the voice model the platform's latest context (who is
        around, where it is): with the conversation's start, or later with
        its next input, a newer one replacing it. Not a turn of its own.
        """
        self._context = text
        if self.ready:
            # Else the start takes it along, or it is given once the
            # conversation is open.
            await self._give_context()

    async def set_people(self, others: int | None) -> None:
        """How many other people are in the conversation's room or channel
        (None: not known): with one, the voice model hears everything they
        say; with more, or not known, only what calls it by name (as
        ``VoiceOptions.wake_mode`` has it). Takes effect at once, also
        mid-conversation."""
        self._wake = wakes_on_name(others, self.options.wake_mode)
        if self.ready:
            await self._give_wake()

    async def _give_wake(self) -> None:
        """Tells the voice server to hear only what calls the bot by name, or
        everything (Codex realtime hears everything itself: nothing to
        tell)."""

    def label_speaker(
        self, name: str | None, start: int, end: int, final: bool, **info
    ) -> None:
        """Who the platform guesses spoke a stretch of its audio (see
        ``InfraVoiceSession.label_speaker``). Codex realtime takes no labels:
        nothing to do."""

    async def _give_context(self) -> None:
        """Gives the model the latest context if it has not got it yet (a
        newer one set meanwhile follows at once)."""
        async with self._context_lock:
            while (
                self._context != self._context_given
                and self._engine is not None
                and self._thread_id is not None
                and not self._closed
            ):
                text = self._context
                try:
                    await self._engine.rt.realtime_append_text(
                        self._thread_id, text, "context"
                    )
                except Exception as exc:  # noqa: BLE001 - the conversation goes on
                    logger.warning(
                        "%s voice %s: context not given: %s", self.label, self.key, exc
                    )
                    return
                self._context_given = text

    async def note(self, text: str) -> None:
        """Adds a note (a result that reached the chat) to the voice model's
        context; the model decides whether and how to tell the listener."""
        if self._engine is None or self._thread_id is None or self._closed:
            return
        try:
            await self._engine.rt.realtime_append_text(
                self._thread_id, NOTE_PROMPT.format(text=text), "developer"
            )
        except Exception as exc:  # noqa: BLE001 - the conversation goes on
            logger.warning("%s voice %s: note failed: %s", self.label, self.key, exc)

    async def _speak(self, text: str) -> None:
        """Gives the realtime model ``text`` as the backend's answer to the
        pending handoff (Codex hands it over as context; the model words
        it)."""
        if self._engine is None or self._thread_id is None or self._closed:
            return
        try:
            await self._engine.rt.realtime_append_speech(self._thread_id, text)
        except Exception as exc:  # noqa: BLE001 - the conversation goes on
            logger.warning("%s voice %s: speech failed: %s", self.label, self.key, exc)

    @property
    def closing(self) -> bool:
        return self._closed

    def _request_close(self, reason: str) -> asyncio.Task:
        """Starts closing (once) and returns the task doing it."""
        if self._close_task is None:
            self._closed = True
            self._closed_event.set()
            # Model audio still arriving is dropped, speech in progress ends.
            self.media.stop()
            self._close_task = asyncio.create_task(
                self._close(reason), name=f"{self.label}-voice-{self.key}-close"
            )
        return self._close_task

    async def close(self, reason: str = "") -> None:
        """Closes the session and waits until it is released.

        The release runs in its own task, so cancelling a caller does not stop
        it half way; every caller waits for the same release.
        """
        task = self._request_close(reason)
        if task is not asyncio.current_task():
            await asyncio.shield(task)

    async def _close(self, reason: str) -> None:
        try:
            start = self._start_task
            if start is not None and not start.done():
                # A start stops at its next step once closed; let it, so that
                # everything it created is known here and the realtime stop is
                # sent after its start.
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(start), CLOSE_WAIT)
                # A start still inside a slow call is covered: one opening the
                # thread unloads it itself on return, and the realtime start
                # is queued on the thread ahead of the stop sent below.
            logger.info("%s voice session %s closed: %s", self.label, self.key, reason)
            await self._release()
        finally:
            if VOICE_SESSIONS.get(self.chat.umo) is self:
                del VOICE_SESSIONS[self.chat.umo]
            self._on_closed(self)

    async def _release_transport(self) -> None:
        """Stops the realtime conversation and closes WebRTC (each once)."""
        engine, thread_id = self._engine, self._thread_id
        if self._realtime_requested and engine is not None and thread_id:
            self._realtime_requested = False
            with contextlib.suppress(Exception):
                await engine.rt.realtime_stop(thread_id)
        pc, self._pc = self._pc, None
        if pc is not None:
            with contextlib.suppress(Exception):
                await pc.close()
        relay, self._relay = self._relay, None
        if relay is not None:
            relay.close()

    async def _release(self) -> None:
        """Releases what exists now; each resource only once, so it can run
        again for what a late start created afterwards."""
        engine, thread_id = self._engine, self._thread_id
        await self._release_transport()
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        record, self._record = self._record, None
        if record is not None:
            await record.close()
        if engine is not None and thread_id is not None and not self._thread_released:
            self._thread_released = True
            # Under the lock a newer session of this key opens the thread with
            # (and takes its route right after): seen here, it is left to it.
            async with engine.session_lock(self.scope_id):
                pump = engine.pumps.get(thread_id)
                route = pump.route if pump is not None else None
                if route is not None and route.events is not self._events_queue:
                    return  # resumed by a quick call back: theirs now
                if route is not None:
                    pump.close_turn()
                # Unload the thread; the next session resumes it from its rollout.
                await engine.forget_thread(thread_id)


def time_prompt() -> str:
    """The current date for the realtime model, in AstrBot's configured zone."""
    import datetime
    import zoneinfo

    now = None
    if zone := astrbot_config.get("timezone"):
        with contextlib.suppress(Exception):
            now = datetime.datetime.now(zoneinfo.ZoneInfo(zone))
    if now is None:
        now = datetime.datetime.now().astimezone()
    return TIME_PROMPT.format(
        date=now.strftime("%Y-%m-%d"),
        weekday=now.strftime("%A"),
        timezone=now.strftime("%Z") or now.strftime("%z"),
        time=now.strftime("%H:%M"),
    )
