"""In-process Codex engine (``codex_astrbot`` pyo3 binding).

One engine per distinct option set. Each loaded Codex thread gets a pump task
that reads its events and routes them to the turn currently running on it;
dynamic tool calls are answered from their own tasks so tools run
concurrently with event streaming.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.core import logger

JsonObject = dict[str, Any]
ToolCallHandler = Callable[[JsonObject], Awaitable[JsonObject]]
# (kind "exec" | "patch", request) -> (approved, reason)
ApprovalHandler = Callable[[str, JsonObject], Awaitable[tuple[bool, str]]]
# (call id, saved path) -> text the model sees as the image tool's result
SavedImageHandler = Callable[[str, str], Awaitable[str | None]]

APPROVAL_EVENTS = {
    "exec_approval_request": "exec",
    "apply_patch_approval_request": "patch",
}

TERMINAL_EVENTS = ("task_complete", "turn_complete", "turn_aborted")


class CodexEngineError(RuntimeError):
    pass


def _import_binding():
    try:
        import codex_astrbot  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise CodexEngineError(
            "codex_astrbot is not installed. Build it with `maturin develop` in "
            "codex-rs/astrbot-py of the codex_for_astrbot fork."
        ) from e
    return codex_astrbot


def bundled_executable(name: str) -> str | None:
    """Helper executable shipped inside the codex_astrbot wheel, if any."""
    try:
        from codex_astrbot import bundled_executable as find  # type: ignore
    except ImportError:
        return None
    return find(name)


def find_codex_exe(explicit: str = "") -> str | None:
    """`codex` executable for sandboxed exec / memory consolidation."""
    if explicit:
        return explicit if Path(explicit).is_file() else None
    return bundled_executable("codex")


def find_code_mode_host(explicit: str = "") -> str | None:
    """Locate codex-code-mode-host: explicit path, the one bundled with the
    binding (matches its protocol), next to `codex` on PATH, or none."""
    if explicit:
        return explicit if Path(explicit).is_file() else None
    if bundled := bundled_executable("codex-code-mode-host"):
        return bundled
    exe = shutil.which("codex")
    if not exe:
        return None
    exe_path = Path(exe).resolve()
    names = ("codex-code-mode-host.exe", "codex-code-mode-host")
    candidates = [exe_path.parent / n for n in names]
    # scoop shims live in ~/scoop/shims; the real install is apps/codex/current/bin
    candidates += [
        exe_path.parent.parent / "apps" / "codex" / "current" / "bin" / n for n in names
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


@dataclass
class ActiveTurn:
    """A turn running for a session (UMO), used to steer same-sender follow-ups."""

    engine: CodexEngine
    thread_id: str
    turn_id: str
    sender_id: str
    message_id: str | None = None
    steered: int = 0
    aborted: bool = False
    steered_texts: list[str] = field(default_factory=list)
    #: Set once `turn_id` is known, or once the turn failed to start. Registered
    #: before the submit so a follow-up arriving during that round trip waits
    #: for the turn instead of being queued behind it as a separate reply.
    #: Cleared again while a continuation is being submitted.
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    #: Every turn started for it, continuations included.
    turn_ids: set[str] = field(default_factory=set)


# umo -> running turn
ACTIVE_TURNS: dict[str, ActiveTurn] = {}

#: How long a follow-up waits for a just-submitted turn to report its id.
STEER_READY_TIMEOUT_S = 10.0
# Added to each routed event: the id of the turn that emitted it.
TURN_ID_FIELD = "_turn_id"

#: umo -> turns holding or waiting for that chat's session lock.
_QUEUED_TURNS: dict[str, int] = {}


class SessionBusy(RuntimeError):
    """Too many turns are already queued for this chat."""

    def __init__(self, waiting: int) -> None:
        super().__init__(f"{waiting} turns already queued")
        self.waiting = waiting


@asynccontextmanager
async def session_slot(
    engine: CodexEngine, umo: str, max_queued: int = 0
) -> AsyncIterator[None]:
    """Runs one turn at a time per chat, refusing a caller when the line is long.

    One chat is one Codex thread, and a thread runs one turn at a time, so a
    message from another sender waits here. Without a cap a busy group can pile
    up an unbounded number of coroutines, each holding its event, all waiting
    out a turn that may run for minutes.

    Args:
        engine: Engine owning the per-chat locks.
        umo: Unified message origin identifying the chat.
        max_queued: Turns allowed to hold or wait for the lock; 0 means no cap.

    Raises:
        SessionBusy: The queue is already that deep.
    """
    waiting = _QUEUED_TURNS.get(umo, 0)
    if max_queued > 0 and waiting >= max_queued:
        raise SessionBusy(waiting)
    _QUEUED_TURNS[umo] = waiting + 1
    try:
        async with engine.session_lock(umo):
            yield
    finally:
        remaining = _QUEUED_TURNS.get(umo, 1) - 1
        if remaining > 0:
            _QUEUED_TURNS[umo] = remaining
        else:
            _QUEUED_TURNS.pop(umo, None)


async def try_steer(
    umo: str,
    sender_id: str,
    turn_input: list[JsonObject],
    *,
    prompt: str = "",
    scopes: list[str] | None = None,
) -> str | None:
    """Inject a follow-up from the same sender into the running turn.

    Returns the running turn's source message id (or "") when Codex accepted
    the input; the caller then produces no reply of its own. Other senders,
    stopped turns, or a turn that already finished return None and are handled
    as a new turn (queued). With ``scopes``, Codex also refuses to steer into
    a turn carrying other ones (the same sender under another role).
    """
    active = ACTIVE_TURNS.get(umo)
    if (
        active is None
        or active.aborted
        or not sender_id
        or active.sender_id != sender_id
    ):
        return None
    if not active.turn_id:
        # The turn is registered but its submit has not returned yet. Waiting
        # keeps this follow-up in the same reply instead of making it a second
        # turn, which is the whole point of steering.
        try:
            await asyncio.wait_for(active.ready.wait(), STEER_READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            return None
        if active.aborted or not active.turn_id:
            return None
    try:
        request: JsonObject = {
            "input": turn_input,
            "mode": "steer",
            "expected_turn_id": active.turn_id,
        }
        if scopes is not None:
            request["scopes"] = scopes
        result = await active.engine.submit_turn(active.thread_id, request)
    except Exception as e:  # noqa: BLE001
        logger.debug("codex steer failed for %s: %s", umo, e)
        return None
    if result.get("status") != "steered":
        return None
    active.steered += 1
    if prompt:
        active.steered_texts.append(prompt)
    logger.info(
        "Follow-up from %s steered into running Codex turn (umo=%s)", sender_id, umo
    )
    return active.message_id or ""


@dataclass
class _TurnRoute:
    events: asyncio.Queue[JsonObject]
    tool_handler: ToolCallHandler | None
    approval_handler: ApprovalHandler | None = None
    #: The turn the route serves; its tool handler runs as that turn's sender.
    turn: ActiveTurn | None = None


async def _calls_from_route_turn(route: _TurnRoute, msg: JsonObject) -> bool:
    """Whether a tool call comes from the turn ``route`` serves.

    A turn interrupted just before (timed out, cancelled, or another sender's
    left running) can still call a tool after the next turn opened its route;
    run there, it would act as that turn's sender.
    """
    turn = route.turn
    turn_of = msg.get(TURN_ID_FIELD) or msg.get("turnId") or msg.get("turn_id")
    if turn is None or not turn_of:
        return True
    if not turn.ready.is_set():
        # Ours may be the turn being submitted right now.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(turn.ready.wait(), STEER_READY_TIMEOUT_S)
    return turn_of in turn.turn_ids


@dataclass
class ThreadPump:
    engine: CodexEngine
    thread_id: str
    route: _TurnRoute | None = None
    task: asyncio.Task | None = None
    tool_tasks: set[asyncio.Task] = field(default_factory=set)
    closed: bool = False

    def start(self) -> None:
        self.task = asyncio.create_task(
            self._run(), name=f"codex-pump-{self.thread_id}"
        )

    def open_turn(
        self,
        tool_handler: ToolCallHandler | None,
        approval_handler: ApprovalHandler | None = None,
    ) -> asyncio.Queue[JsonObject]:
        queue: asyncio.Queue[JsonObject] = asyncio.Queue()
        self.route = _TurnRoute(queue, tool_handler, approval_handler)
        return queue

    def bind_turn(self, turn: ActiveTurn) -> None:
        """Tool calls through the open route must come from ``turn``."""
        if self.route is not None:
            self.route.turn = turn

    def close_turn(self) -> None:
        self.route = None

    async def _run(self) -> None:
        rt = self.engine.rt
        try:
            while True:
                raw = await rt.next_event(self.thread_id)
                if raw is None:  # thread terminated and drained
                    if self.route is not None:
                        self.route.events.put_nowait(
                            {"type": "_pump_closed", "message": "thread terminated"}
                        )
                    break
                event = json.loads(raw)
                msg = event.get("msg") or {}
                if msg.get("type") == "dynamic_tool_call_request":
                    msg[TURN_ID_FIELD] = str(event.get("id") or "")
                    # Bound now, like approvals: see _calls_from_route_turn.
                    task = asyncio.create_task(self._answer_tool(msg, self.route))
                    self.tool_tasks.add(task)
                    task.add_done_callback(self.tool_tasks.discard)
                    continue
                if (kind := APPROVAL_EVENTS.get(msg.get("type") or "")) is not None:
                    # Bind the request to the turn open now, not whichever
                    # turn is open when the task gets to run.
                    task = asyncio.create_task(
                        self._answer_approval(kind, msg, self.route)
                    )
                    self.tool_tasks.add(task)
                    task.add_done_callback(self.tool_tasks.discard)
                if self.route is not None:
                    # Which turn it belongs to: a turn interrupted just before
                    # this one still reports into the open route.
                    msg[TURN_ID_FIELD] = str(event.get("id") or "")
                    self.route.events.put_nowait(msg)
                if msg.get("type") == "shutdown_complete":
                    if self.route is not None:
                        # A turn still open would otherwise wait out its timeout.
                        self.route.events.put_nowait(
                            {"type": "_pump_closed", "message": "thread shut down"}
                        )
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - thread closed or engine failure
            logger.debug("codex pump %s stopped: %s", self.thread_id, e)
            if self.route is not None:
                self.route.events.put_nowait(
                    {"type": "_pump_closed", "message": str(e)}
                )
        finally:
            self.closed = True
            self.engine.pumps.pop(self.thread_id, None)

    async def _answer_approval(
        self, kind: str, msg: JsonObject, route: _TurnRoute | None
    ) -> None:
        """Decide a native exec / patch approval; unattended requests are denied."""
        approved, reason = False, "No AstrBot session is attached to this request."
        try:
            if route is not None and route.approval_handler is not None:
                approved, reason = await route.approval_handler(kind, msg)
        except Exception as e:  # noqa: BLE001
            logger.error("codex approval handler failed: %s", e, exc_info=True)
            approved, reason = False, f"approval failed: {e!s}"
        call_id = str(msg.get("call_id") or msg.get("callId") or "")
        request = {
            "kind": kind,
            "id": str(msg.get("approval_id") or msg.get("approvalId") or call_id),
            "turn_id": msg.get("turn_id") or msg.get("turnId") or None,
            "approved": approved,
            "reason": reason or None,
        }
        try:
            await self.engine.rt.review_decision(
                self.thread_id, json.dumps(request, ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("codex approval response for %s failed: %s", call_id, e)

    async def _answer_tool(self, msg: JsonObject, route: _TurnRoute | None) -> None:
        call_id = msg.get("callId") or msg.get("call_id") or ""
        try:
            if route is None or route.tool_handler is None:
                result = {
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": "No AstrBot session is attached to this call.",
                        }
                    ],
                    "success": False,
                }
            elif not await _calls_from_route_turn(route, msg):
                logger.warning(
                    "Codex tool call %s from turn %s refused: not the open turn's",
                    msg.get("tool"),
                    msg.get(TURN_ID_FIELD),
                )
                result = {
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": "error: this call's turn is no longer running.",
                        }
                    ],
                    "success": False,
                }
            else:
                result = await route.tool_handler(msg)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "codex tool call %s failed: %s", msg.get("tool"), e, exc_info=True
            )
            result = {
                "contentItems": [{"type": "inputText", "text": f"error: {e!s}"}],
                "success": False,
            }
        try:
            await self.engine.rt.dynamic_tool_response(
                self.thread_id, call_id, json.dumps(result, ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("codex tool response for %s failed: %s", call_id, e)
            # Codex waits for an answer: one it can read, else the turn hangs.
            with contextlib.suppress(Exception):
                await self.engine.rt.dynamic_tool_response(
                    self.thread_id,
                    call_id,
                    json.dumps(
                        {
                            "contentItems": [
                                {"type": "inputText", "text": "error: no result"}
                            ],
                            "success": False,
                        }
                    ),
                )


class CodexEngine:
    """Shared in-process Codex runtime for one option set."""

    _instances: dict[str, CodexEngine] = {}
    _lock: asyncio.Lock | None = None

    def __init__(self, rt: Any, codex_home: str = "") -> None:
        self.rt = rt
        self.codex_home = codex_home
        self.pumps: dict[str, ThreadPump] = {}
        # thread_id -> the additional context its inputs last carried.
        self._additional_context: dict[str, JsonObject] = {}
        self.session_locks: dict[str, asyncio.Lock] = {}
        #: thread id -> handler of images Codex saves during that thread's turn.
        self.saved_image_handlers: dict[str, SavedImageHandler] = {}

    async def _on_saved_image(
        self, thread_id: str, call_id: str, saved_path: str
    ) -> str | None:
        """Codex's saved-image hook: the returned text is the tool result.

        Routed to the turn running on that thread, which knows the chat and so
        where the image has to go. Never raises: a failure here only means
        Codex falls back to its own hint.
        """
        handler = self.saved_image_handlers.get(thread_id)
        if handler is None:
            return None
        try:
            return await handler(call_id, saved_path)
        except Exception as e:  # noqa: BLE001
            logger.error("Saved-image handler failed for %s: %s", saved_path, e)
            return None

    @classmethod
    async def get(cls, options: JsonObject) -> CodexEngine:
        key = json.dumps(options, sort_keys=True, ensure_ascii=False)
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        async with cls._lock:
            engine = cls._instances.get(key)
            if engine is not None:
                return engine
            binding = _import_binding()
            codex_home = str(options["codex_home"])
            Path(codex_home).mkdir(parents=True, exist_ok=True)
            # A settings change (another model, say) builds a different option
            # set. The engine it replaces still owns every chat's thread and
            # its rollout file, so resuming those chats would fail with
            # "already has an active writer" and lose their history: shut it
            # down first, and start again from one engine per codex home.
            for old_key, old in list(cls._instances.items()):
                if old.codex_home != codex_home:
                    continue
                cls._instances.pop(old_key, None)
                logger.info(
                    "Codex settings changed; restarting the engine (codex_home=%s).",
                    codex_home,
                )
                with contextlib.suppress(Exception):
                    await old.rt.shutdown()
            rt = await binding.Runtime.create(json.dumps(options, ensure_ascii=False))
            engine = cls(rt, codex_home)
            # Older bindings have no hook; images then arrive through the
            # event stream alone.
            if hasattr(rt, "set_saved_image_hook"):
                rt.set_saved_image_hook(engine._on_saved_image)
            cls._instances[key] = engine
            logger.info("Codex engine ready (codex_home=%s)", codex_home)
            return engine

    @classmethod
    async def shutdown_all(cls) -> None:
        engines = list(cls._instances.values())
        cls._instances.clear()
        for engine in engines:
            with contextlib.suppress(Exception):
                await engine.rt.shutdown()

    def session_lock(self, umo: str) -> asyncio.Lock:
        """One lock per chat, not per thread: it also has to cover opening the
        thread, which is what decides the thread id."""
        return self.session_locks.setdefault(umo, asyncio.Lock())

    def _pump(self, thread_id: str) -> ThreadPump:
        pump = self.pumps.get(thread_id)
        if pump is None or pump.closed:
            pump = ThreadPump(self, thread_id)
            self.pumps[thread_id] = pump
            pump.start()
        return pump

    async def open_thread(
        self, state: JsonObject | None, params: JsonObject
    ) -> tuple[JsonObject, bool]:
        """Reuse, resume or start a thread. Returns (info, started_new)."""
        thread_id = (state or {}).get("thread_id")
        if thread_id and await self.rt.is_loaded(thread_id):
            self._pump(thread_id)
            return {
                "thread_id": thread_id,
                "rollout_path": state.get("rollout_path"),
            }, False
        rollout = (state or {}).get("rollout_path")
        if thread_id and rollout and Path(rollout).exists():
            try:
                info = json.loads(
                    await self.rt.resume_thread(
                        json.dumps(
                            {**params, "rollout_path": rollout}, ensure_ascii=False
                        )
                    )
                )
                self._pump(info["thread_id"])
                logger.info("Codex thread resumed: %s", info["thread_id"])
                return info, False
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Resume codex thread %s failed, starting new: %s", thread_id, e
                )
        info = json.loads(
            await self.rt.start_thread(json.dumps(params, ensure_ascii=False))
        )
        self._pump(info["thread_id"])
        logger.info(
            "Codex thread started: %s model=%s", info["thread_id"], info.get("model")
        )
        return info, True

    def pump(self, thread_id: str) -> ThreadPump:
        return self._pump(thread_id)

    async def submit_turn(self, thread_id: str, request: JsonObject) -> JsonObject:
        # Codex reads a missing additional_context as "none left" and clears
        # what it keeps for the thread, so a steer or a continuation would
        # lose the persona: not put back after compaction, and sent again
        # whole on the next turn. Inputs without it carry the last one.
        if "additional_context" in request:
            self._additional_context[thread_id] = request["additional_context"]
        elif thread_id in self._additional_context:
            request = {
                **request,
                "additional_context": self._additional_context[thread_id],
            }
        return json.loads(
            await self.rt.submit_turn(
                thread_id, json.dumps(request, ensure_ascii=False)
            )
        )

    async def interrupt(self, thread_id: str) -> None:
        with contextlib.suppress(Exception):
            await self.rt.interrupt(thread_id)

    def drop_additional_context(self, thread_id: str) -> None:
        """Forget the context a replaced thread's inputs carried."""
        self._additional_context.pop(thread_id, None)

    @classmethod
    async def release_thread(cls, umo: str, thread_id: str) -> None:
        """Unload a chat's thread in every engine once the chat left it
        (/new, /reset): Codex keeps a loaded thread, its event pump and the
        context cache until told otherwise, however many new ones follow.

        Holds the chat's lock, so a turn still opening the thread finishes
        first; the pump ends on its own when the thread shuts down.
        """
        for engine in list(cls._instances.values()):
            async with engine.session_lock(umo):
                await engine.forget_thread(thread_id)
        logger.info("Codex thread %s released (umo=%s)", thread_id, umo)

    async def forget_thread(self, thread_id: str) -> None:
        self.drop_additional_context(thread_id)
        with contextlib.suppress(Exception):
            await self.rt.shutdown_thread(thread_id)
