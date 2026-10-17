"""Codex-style ``exec_command`` / ``write_stdin`` over AstrBot's runtimes.

Codex models are trained on this pair: a command runs in a session, the call
yields after ``yield_time_ms`` with whatever output arrived so far, and a
session id lets the model keep reading or send input. That beats one-shot
execution for anything interactive or slow (installers, REPLs, servers, test
watchers), so these replace the plain shell tools when Codex drives the turn.

The local runtime uses AstrBot's managed shell sessions. The Shipyard Neo SDK
only offers one-shot ``exec``, so a session is emulated inside the sandbox: the
command runs in a detached tmux pane with ``pipe-pane`` mirroring it into a log
file, and each poll reads the log from the last offset. Without tmux the command
runs detached instead, with stdin arriving through a FIFO held open by a sleeper
process and no TTY.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shlex
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import FunctionTool
from astrbot.core import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.computer.booters.local import LocalShellComponent
from astrbot.core.computer.computer_client import get_booter
from astrbot.core.message.message_event_result import MessageChain

from ..registry import builtin_tool
from .fs import _read_allowed_roots, _write_allowed_roots
from .util import (
    check_local_execution_permission,
    get_local_permission_policy,
    is_local_runtime,
    workspace_root_for_context,
)

_COMPUTER_RUNTIME_TOOL_CONFIG = {
    "provider_settings.computer_use_runtime": ("local", "sandbox"),
}

DEFAULT_YIELD_TIME_MS = 10_000
MAX_YIELD_TIME_MS = 30_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
# Codex counts tokens with a tokenizer; four bytes per token is close enough
# for a budget and never underestimates ASCII output.
CHARS_PER_TOKEN = 4
SANDBOX_SESSION_ROOT = "/tmp/astrbot-exec"
META_MARKER = "<<<ASTRBOT_EXEC_META"
TRUNCATION_MARKER = "[... omitted middle of output ...]"


# A PTY writes terminal control sequences into the stream: colours, cursor
# moves, title changes and a CR before every LF. The model reads the output as
# text, so these are dropped before it is shown. Byte offsets are unaffected:
# the sandbox reports them with `wc -c` over the untouched log.
_ANSI_SEQUENCES = re.compile(
    r"""
    \x1b\][^\x07\x1b]*(?:\x07|\x1b\\)   # OSC: window title and friends
    | \x1b[@-Z\\-_]                     # two-byte escapes
    | \x1b\[[0-?]*[ -/]*[@-~]           # CSI: colours, cursor moves
    | \x1b[PX^_][^\x1b]*\x1b\\          # DCS, SOS, PM, APC strings
    | [\x00\x07\x08\x0b\x0c\x0e\x0f]    # stray control bytes
    """,
    re.VERBOSE,
)


def clean_terminal_output(text: str) -> str:
    """Strip terminal control sequences from PTY output.

    Args:
        text: Raw bytes decoded from the session log.

    Returns:
        The same text as a person would read it on screen, with escape
        sequences removed and CRLF line endings normalised.
    """
    if not text:
        return text
    return _ANSI_SEQUENCES.sub("", text).replace("\r\n", "\n").replace("\r", "\n")


def approx_token_count(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN) if text else 0


def truncate_middle(text: str, max_tokens: int) -> tuple[str, bool]:
    """Keep the head and tail of ``text`` within the token budget."""
    budget = max(1, max_tokens) * CHARS_PER_TOKEN
    if len(text) <= budget:
        return text, False
    keep = max(1, (budget - len(TRUNCATION_MARKER) - 2) // 2)
    return f"{text[:keep]}\n{TRUNCATION_MARKER}\n{text[-keep:]}", True


def format_exec_response(
    *,
    output: str,
    wall_time_seconds: float,
    exit_code: int | None = None,
    session_id: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """Render one exec result the way Codex's own exec tools do."""
    output = clean_terminal_output(output)
    original_tokens = approx_token_count(output)
    text, truncated = truncate_middle(output, max_output_tokens)
    sections = []
    if truncated:
        sections.append(f"Chunk ID: {uuid.uuid4().hex[:6]}")
    sections.append(f"Wall time: {wall_time_seconds:.4f} seconds")
    if exit_code is not None:
        sections.append(f"Process exited with code {exit_code}")
    if session_id is not None:
        sections.append(f"Process running with session ID {session_id}")
    if truncated:
        sections.append(f"Original token count: {original_tokens}")
    sections.append("Output:")
    return "\n".join(sections) + "\n" + text


def clamp_yield_time(value: Any, default: int = DEFAULT_YIELD_TIME_MS) -> int:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(ms, MAX_YIELD_TIME_MS))


def clamp_output_tokens(value: Any) -> int:
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_OUTPUT_TOKENS
    return max(256, min(tokens, DEFAULT_MAX_OUTPUT_TOKENS * 4))


# --------------------------------------------------------------- sandbox sessions


@dataclass
class SandboxSession:
    session_id: str
    directory: str
    cursor: int = 0
    tmux: bool = False
    """Whether the command runs in a tmux pane, which gives it a real PTY."""


class SandboxSessions:
    """Sessions emulated inside a Shipyard Neo sandbox, keyed by chat."""

    _sessions: dict[str, dict[str, SandboxSession]] = {}

    @classmethod
    def get(cls, umo: str, session_id: str) -> SandboxSession | None:
        return cls._sessions.get(umo, {}).get(session_id)

    @classmethod
    def add(cls, umo: str, session: SandboxSession) -> None:
        cls._sessions.setdefault(umo, {})[session.session_id] = session

    @classmethod
    def drop(cls, umo: str, session_id: str) -> None:
        cls._sessions.get(umo, {}).pop(session_id, None)

    @classmethod
    def count(cls, umo: str) -> int:
        return len(cls._sessions.get(umo, {}))


# ---------------------------------------------------- background completion

#: Live sessions allowed per chat when nothing is configured. Each one is a
#: tmux pane plus a shell, so the real ceiling is the sandbox's own process
#: limit, which comes from the Bay profile and cannot be raised from here.
DEFAULT_MAX_LIVE_SESSIONS = 128

#: Raises the session's soft process limit to whatever the container's hard
#: limit allows. The hard limit and the container's pids cgroup come from the
#: Bay profile; this only stops a low soft default from biting first.
RAISE_NPROC = 'ulimit -u "$(ulimit -Hu)" 2>/dev/null || true; '

#: Grace period before a watcher starts polling, so it stays out of the way
#: while the agent is still working the session itself.
WATCH_GRACE_SECONDS = 60
#: How long a watcher keeps waiting for a command the agent walked away from.
WATCH_MAX_SECONDS = 30 * 60
#: How long each watcher poll waits.
WATCH_POLL_MS = 15_000
#: Output carried in the completion message.
COMPLETION_TAIL_CHARS = 1_200

#: One watcher per session, so repeated polls do not spawn more.
_watchers: dict[tuple[str, str], asyncio.Task] = {}
#: Sessions whose completion has been reported, by whoever got there first.
_claimed: OrderedDict[tuple[str, str], None] = OrderedDict()
_CLAIMED_LIMIT = 512


def max_live_sessions(context: ContextWrapper[AstrAgentContext]) -> int:
    """Live shell sessions one chat may hold.

    Raise it with `provider_settings.sandbox.max_shell_sessions`. The hard
    ceiling is the sandbox's process limit, which is part of the Bay profile:
    if commands start failing to fork, pick a profile with more headroom
    rather than raising this.
    """
    try:
        conf = context.context.context.get_config(
            umo=context.context.event.unified_msg_origin
        )
        sandbox = (conf.get("provider_settings") or {}).get("sandbox") or {}
        configured = int(sandbox.get("max_shell_sessions") or 0)
    except Exception:  # noqa: BLE001 - config shape is not guaranteed
        return DEFAULT_MAX_LIVE_SESSIONS
    return configured if configured > 0 else DEFAULT_MAX_LIVE_SESSIONS


def claim_completion(umo: str, session_id: str) -> bool:
    """Returns whether the caller is the one to report this session's exit.

    A session can be finished either by the agent polling it or by the watcher
    behind it. Both must not report the same completion.
    """
    key = (umo, session_id)
    if key in _claimed:
        return False
    _claimed[key] = None
    while len(_claimed) > _CLAIMED_LIMIT:
        _claimed.popitem(last=False)
    return True


@dataclass
class _WatchTarget:
    """What a watcher needs after the turn that started the command is gone."""

    plugin_context: Any
    umo: str
    session_id: str
    sandbox: SandboxSession | None
    #: The command's initiator. The completion is delivered as a message from
    #: them, so their permission rule and their place in the queue apply.
    sender_id: str
    is_admin: bool
    #: The group it was started in ("" in private), for group permission rules.
    group_id: str = ""


def watch_background_session(
    context: ContextWrapper[AstrAgentContext],
    umo: str,
    session_id: str,
    *,
    sandbox: SandboxSession | None = None,
) -> None:
    """Delivers a command's result to the chat once it finishes.

    `exec_command` yields after `yield_time_ms` and leaves the command running;
    the only way to see the rest is for the model to poll. When it answers the
    chat message instead, the turn ends and nothing ever collects the output.
    This watcher closes that loop: it waits the command out and posts the exit
    code and the tail of its output into the same chat.
    """
    key = (umo, session_id)
    if key in _watchers:
        return
    event = context.context.event
    target = _WatchTarget(
        plugin_context=context.context.context,
        umo=umo,
        session_id=session_id,
        sandbox=sandbox,
        sender_id=str(event.get_sender_id() or ""),
        is_admin=getattr(event, "role", "") == "admin",
        group_id=str(event.get_group_id() or ""),
    )
    task = asyncio.create_task(_watch(target), name=f"exec-watch-{session_id}")
    _watchers[key] = task
    task.add_done_callback(lambda _t, k=key: _watchers.pop(k, None))


async def _watch(target: _WatchTarget) -> None:
    plugin_context = target.plugin_context
    umo = target.umo
    session_id = target.session_id
    sandbox = target.sandbox
    await asyncio.sleep(WATCH_GRACE_SECONDS)
    deadline = time.monotonic() + WATCH_MAX_SECONDS
    cursor = sandbox.cursor if sandbox else 0
    try:
        booter = await get_booter(plugin_context, umo)
    except Exception as e:  # noqa: BLE001 - sandbox may be gone by now
        logger.debug("exec watcher could not reach the runtime for %s: %s", umo, e)
        return
    while time.monotonic() < deadline:
        if sandbox is not None and SandboxSessions.get(umo, session_id) is None:
            return  # the agent polled it through to the end itself
        try:
            if sandbox is not None:
                raw = await _sandbox_exec(
                    booter.shell,
                    _poll_script(
                        sandbox.directory,
                        cursor,
                        WATCH_POLL_MS,
                        sandbox.session_id if sandbox.tmux else None,
                    ),
                    timeout=WATCH_POLL_MS // 1000 + 15,
                )
                output, cursor, exit_code = _parse_poll(raw, cursor)
            else:
                result = await booter.shell.poll_session(
                    owner_id=umo,
                    requester_id=target.sender_id,
                    requester_is_admin=target.is_admin,
                    session_id=session_id,
                    yield_time_ms=WATCH_POLL_MS,
                    max_output_chars=COMPLETION_TAIL_CHARS,
                )
                output = str(result.get("stdout") or "")
                exit_code = result.get("exit_code")
        except Exception as e:  # noqa: BLE001 - session gone, or runtime down
            logger.debug("exec watcher stopped for session %s: %s", session_id, e)
            return
        if exit_code is None:
            continue
        if sandbox is not None:
            SandboxSessions.drop(umo, session_id)
        if claim_completion(umo, session_id):
            await _deliver(target, exit_code, output)
        return
    # Still running after the watch window. A pane nobody polls is a process
    # nobody will ever stop, so end it here rather than leave it in the sandbox.
    logger.info(
        "Background command %s is still running after %d minutes; stopping it.",
        session_id,
        WATCH_MAX_SECONDS // 60,
    )
    if sandbox is not None:
        SandboxSessions.drop(umo, session_id)
        if sandbox.tmux:
            with contextlib.suppress(Exception):
                await _sandbox_exec(
                    booter.shell,
                    f"tmux kill-session -t {shlex.quote(session_id)} 2>/dev/null; true",
                    timeout=30,
                )
    if claim_completion(umo, session_id):
        with contextlib.suppress(Exception):
            await plugin_context.send_message(
                umo,
                MessageChain().message(
                    f"后台命令（会话 {session_id}）运行超过 "
                    f"{WATCH_MAX_SECONDS // 60} 分钟仍未结束，已停止。"
                ),
            )


async def _deliver(target: _WatchTarget, exit_code: int, output: str) -> None:
    """Hands the result back to the agent as a message from whoever started it.

    A fixed notice would be cheaper, but it leaves the agent unaware: the
    result would not be in its history, so it could neither mention it in its
    own voice nor act on it. Routing it as a message also means the ordinary
    rules apply -- steered into the initiator's running turn, or queued behind
    whatever else the chat is doing.
    """
    from astrbot.core.agent.runners.codex.wake import run_background_exec_completion

    tail = clean_terminal_output(output).strip()
    if len(tail) > COMPLETION_TAIL_CHARS:
        tail = "…" + tail[-COMPLETION_TAIL_CHARS:]
    try:
        await run_background_exec_completion(
            target.plugin_context,
            session_str=target.umo,
            sender_id=target.sender_id,
            group_id=target.group_id,
            role="admin" if target.is_admin else "member",
            session_id=target.session_id,
            exit_code=exit_code,
            output=tail,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Could not report background session %s: %s", target.session_id, e)


def _poll_script(
    directory: str, cursor: int, yield_ms: int, tmux_session: str | None = None
) -> str:
    """Wait inside the sandbox for new output or exit, then report both.

    The wait happens in the sandbox so one poll costs one round trip. The
    trailing metadata line carries the log size and exit code.

    Args:
        directory: Session directory holding ``out.log`` and ``exit``.
        cursor: Byte offset already reported to the model.
        yield_ms: How long to wait for new output before giving up.
        tmux_session: tmux session name when the command runs in a pane. It
            adds a liveness probe, because a pane that died without writing an
            exit code -- killed, signalled with Ctrl-C, or lost with the tmux
            server -- would otherwise stay "running" for the rest of the chat.

    Returns:
        A shell script to run in the sandbox.
    """
    seconds = (max(0, yield_ms) + 999) // 1000
    probe = cleanup = ""
    if tmux_session:
        probe = (
            '[ -f "$d/exit" ] || tmux has-session -t '
            f"{tmux_session} 2>/dev/null || "
            '{ echo "[session ended without an exit code]" >> "$d/out.log"; '
            'echo -1 > "$d/exit"; }; '
        )
        cleanup = (
            f'; [ -f "$d/exit" ] && tmux kill-session -t {tmux_session} '
            ">/dev/null 2>&1; true"
        )
    return (
        f"d={shlex.quote(directory)}; c={cursor}; "
        f"end=$(( $(date +%s) + {seconds} )); "
        "while :; do "
        'size=$(wc -c < "$d/out.log" 2>/dev/null || echo 0); size=$((size + 0)); '
        '{ [ -f "$d/exit" ] || [ "$size" -gt "$c" ] || '
        '[ "$(date +%s)" -ge "$end" ]; } && break; '
        "sleep 0.2; done; " + probe + 'tail -c +$((c + 1)) "$d/out.log" 2>/dev/null; '
        f'printf "\\n{META_MARKER} %s %s>>>\\n" '
        '"$(( $(wc -c < "$d/out.log" 2>/dev/null || echo 0) + 0 ))" '
        '"$(cat "$d/exit" 2>/dev/null || echo none)"' + cleanup
    )


def _parse_poll(raw: str, cursor: int) -> tuple[str, int, int | None]:
    """Split a poll result into (output, new cursor, exit code)."""
    marker = raw.rfind(META_MARKER)
    end = raw.find(">>>", marker) if marker >= 0 else -1
    if marker < 0 or end < 0:
        return raw, cursor + len(raw.encode("utf-8", "replace")), None
    output = raw[:marker].rstrip("\n")
    meta = raw[marker + len(META_MARKER) : end].split()
    size = cursor + len(output.encode("utf-8", "replace"))
    exit_code: int | None = None
    if meta:
        try:
            size = int(meta[0])
        except ValueError:
            pass
    if len(meta) >= 2 and meta[1] != "none":
        try:
            exit_code = int(meta[1])
        except ValueError:
            exit_code = None
    return output, size, exit_code


async def _sandbox_exec(shell: Any, script: str, timeout: int) -> str:
    result = await shell.exec(script, timeout=timeout)
    if callable(getattr(result, "model_dump", None)):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = getattr(result, "__dict__", {})
    output = str(payload.get("output") or payload.get("stdout") or "")
    error = str(payload.get("error") or payload.get("stderr") or "")
    return output + error


async def _start_sandbox_session(
    shell: Any, cmd: str, workdir: str | None, shell_binary: str | None
) -> SandboxSession:
    """Start a detached command in the sandbox and return its session.

    With tmux available the command gets a real PTY, so programs that ask for
    confirmation, prompt for a password or run a REPL behave as they would for
    a person. `pipe-pane` mirrors the raw pane output into a log file, which
    keeps the byte-offset reads that `write_stdin` uses. Without tmux the
    command runs under a FIFO instead: plain pipes, no TTY.

    The pane waits for a `go` file before it runs anything, because
    `pipe-pane` only forwards what the pane prints after it is attached and
    refuses a pane that has already exited. Starting the command first would
    lose the output of short commands, and losing `pipe-pane` would send the
    whole call down the fallback path with the command already running -- so a
    command such as `pip install` would run a second time.

    Args:
        shell: Sandbox shell capability (one-shot exec).
        cmd: Command line to run.
        workdir: Working directory, or None for the sandbox default.
        shell_binary: Shell to launch the command with, or None for `sh`.

    Returns:
        The session, with `tmux` telling which backend started it.
    """
    session_id = uuid.uuid4().hex[:8]
    directory = f"{SANDBOX_SESSION_ROOT}/{session_id}"
    runner = shell_binary or "sh"
    quoted_dir = shlex.quote(directory)
    quoted_workdir = shlex.quote(workdir or ".")
    log = shlex.quote(directory + "/out.log")
    exit_file = shlex.quote(directory + "/exit")
    go_file = shlex.quote(directory + "/go")
    # The command keeps a shell of its own, so `echo $?` still runs when the
    # command ends in `;` or `&`, ends in a comment, or calls `exit` itself.
    # The wait gives up after five seconds in case the start call never got
    # far enough to write the go file.
    pane = (
        RAISE_NPROC + f"i=0; while [ ! -e {go_file} ]; do i=$((i + 1)); "
        '[ "$i" -gt 100 ] && exit 1; sleep 0.05; done; '
        f"{runner} -c {shlex.quote(cmd)}; echo $? > {exit_file}"
    )
    tmux_script = (
        f"cd {quoted_workdir} && mkdir -p {quoted_dir} && : > {log} && "
        f"tmux new-session -d -s {session_id} -x 120 -y 40 "
        f"-c {quoted_workdir} {shlex.quote(pane)} && "
        f"tmux pipe-pane -o -t {session_id} {shlex.quote(f'cat >> {log}')} && "
        f": > {go_file} && echo tmux-ok || "
        f"{{ tmux kill-session -t {session_id} >/dev/null 2>&1; false; }}"
    )
    if "tmux-ok" in await _sandbox_exec(shell, tmux_script, timeout=30):
        return SandboxSession(session_id=session_id, directory=directory, tmux=True)

    # The sleeper keeps a writer on the FIFO so the command does not see EOF
    # between write_stdin calls.
    fifo = shlex.quote(directory + "/stdin")
    script = (
        RAISE_NPROC + f"mkdir -p {quoted_dir} && cd {quoted_workdir} && "
        f"mkfifo {fifo} 2>/dev/null; : > {log}; "
        f"(sleep 86400 > {fifo} &) ; "
        f"( {runner} -c {shlex.quote(cmd)} < {fifo} > {log} 2>&1; "
        f"echo $? > {exit_file} ) >/dev/null 2>&1 & "
        f"echo $! > {shlex.quote(directory + '/pid')}"
    )
    await _sandbox_exec(shell, script, timeout=30)
    return SandboxSession(session_id=session_id, directory=directory)


def _write_stdin_script(session: SandboxSession, chars: str) -> str:
    """Build the command that delivers `chars` to a session's stdin.

    tmux takes the text literally through `send-keys -l`, with a trailing
    newline sent as Enter so line-based programs see a submitted line. Two
    tmux argument rules need care before the text reaches the pane: an
    argument that ends in `;` is a command separator unless the semicolon is
    escaped, and an argument that starts with `-` is read as a flag unless it
    comes after `--`.

    Args:
        session: Session to write to.
        chars: Text to deliver; one trailing newline becomes Enter.

    Returns:
        A shell command for the sandbox, or `true` when there is nothing to send.
    """
    if not session.tmux:
        return f"printf %s {shlex.quote(chars)} > {shlex.quote(session.directory + '/stdin')}"
    # Only the final newline is the submit key; the rest is part of the text.
    body = chars[:-1] if chars.endswith("\n") else chars
    parts = []
    if body:
        literal = f"{body[:-1]}\\;" if body.endswith(";") else body
        parts.append(
            f"tmux send-keys -t {session.session_id} -l -- {shlex.quote(literal)}"
        )
    if chars.endswith("\n"):
        parts.append(f"tmux send-keys -t {session.session_id} Enter")
    return " && ".join(parts) or "true"


# -------------------------------------------------------------------- the tools


@builtin_tool(config=_COMPUTER_RUNTIME_TOOL_CONFIG)
@dataclass
class ExecCommandTool(FunctionTool):
    name: str = "exec_command"
    description: str = (
        "Runs a command in a shell session, returning output or a session ID for "
        "ongoing interaction. Use write_stdin to keep reading or to answer prompts. "
        "A session ID means the command is still running. Poll it with write_stdin "
        "when it should be done in seconds. For anything slower, do not hold up the "
        "conversation: say what you started, end your turn, and the result comes "
        "back to you as a message when the command finishes. Never idle with sleep "
        "or repeated empty polls to keep a turn alive."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute."},
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command. Defaults to the workspace root.",
                },
                "yield_time_ms": {
                    "type": "number",
                    "description": (
                        "Wait before yielding output. Defaults to 10000 ms; "
                        "effective range is 250-30000 ms. Commands that finish "
                        "sooner return immediately."
                    ),
                },
                "max_output_tokens": {
                    "type": "number",
                    "description": "Output token budget. Defaults to 10000 tokens.",
                },
                "shell": {
                    "type": "string",
                    "description": "Shell binary to launch. Defaults to the runtime's default shell.",
                },
            },
            "required": ["cmd"],
            "additionalProperties": False,
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        cmd: str,
        workdir: str | None = None,
        yield_time_ms: Any = DEFAULT_YIELD_TIME_MS,
        max_output_tokens: Any = DEFAULT_MAX_OUTPUT_TOKENS,
        shell: str | None = None,
    ) -> ToolExecResult:
        # The Local permission policy of the caller's role, as the shell tool
        # applies it (outside Local: the administrator gate).
        local_policy, permission_error = check_local_execution_permission(
            context, "Shell execution"
        )
        if permission_error:
            return permission_error
        event = context.context.event
        umo = event.unified_msg_origin
        wait_ms = clamp_yield_time(yield_time_ms)
        budget = clamp_output_tokens(max_output_tokens)
        booter = await get_booter(context.context.context, umo)
        started = time.monotonic()

        if is_local_runtime(context):
            if not isinstance(booter.shell, LocalShellComponent):
                return "Error: the local shell component is unavailable."
            creator_id = event.get_sender_id()
            if not creator_id:
                return "Error: sender identity is unavailable."
            root = await workspace_root_for_context(context)
            root.mkdir(parents=True, exist_ok=True)
            creator_is_admin = event.role == "admin"
            sandboxed = bool(local_policy and local_policy.requires_sandbox)
            # A relative workdir is in the workspace. A run without host
            # access keeps it there: the isolation mounts the working
            # directory writable (host access mounts everything anyway).
            cwd_path = Path(workdir).expanduser() if workdir else root
            if not cwd_path.is_absolute():
                cwd_path = root / cwd_path
            cwd_path = cwd_path.resolve()
            scope = local_policy.filesystem_scope if local_policy else "host"
            if scope != "host" and not cwd_path.is_relative_to(root.resolve()):
                return (
                    f"Error: workdir must be inside the workspace ({root}) "
                    "for this role."
                )
            cwd = str(cwd_path)
            sandbox_roots = {}
            if local_policy and local_policy.filesystem_scope == "workspace":
                sandbox_roots = {
                    "readable_roots": _read_allowed_roots(umo, root),
                    "writable_roots": _write_allowed_roots(
                        umo, root, include_installed_skills=creator_is_admin
                    ),
                }
            try:
                result = await booter.shell.exec_managed(
                    cmd,
                    owner_id=umo,
                    creator_id=creator_id,
                    creator_is_admin=creator_is_admin,
                    sandboxed=sandboxed,
                    permission_check=lambda: (
                        is_local_runtime(context)
                        and get_local_permission_policy(context) == local_policy
                        # The original event role does not reflect admin removal.
                        and (
                            not creator_is_admin
                            or str(creator_id)
                            in context.context.context.get_config(umo=umo).get(
                                "admins_id", []
                            )
                        )
                    ),
                    allow_network=(
                        local_policy.allow_network if local_policy else True
                    ),
                    filesystem_scope=(
                        local_policy.filesystem_scope if local_policy else "host"
                    ),
                    cwd=cwd,
                    env={},
                    # Isolated processes are capped like the shell tool's.
                    timeout=300 if sandboxed else None,
                    yield_time_ms=wait_ms,
                    max_output_chars=budget * CHARS_PER_TOKEN,
                    **sandbox_roots,
                )
            except (PermissionError, ValueError) as e:
                return f"Error: {e}"
            running = result.get("exit_code") is None
            if running:
                watch_background_session(context, umo, str(result["session_id"]))
            return format_exec_response(
                output=str(result.get("stdout") or ""),
                wall_time_seconds=time.monotonic() - started,
                exit_code=result.get("exit_code"),
                session_id=str(result["session_id"]) if running else None,
                max_output_tokens=budget,
            )

        max_sessions = max_live_sessions(context)
        if SandboxSessions.count(umo) >= max_sessions:
            # Every live session is a tmux pane and a shell inside the sandbox,
            # which has its own process limits. Refusing here is better than
            # failing somewhere deeper once those run out.
            return (
                f"Error: {max_sessions} shell sessions are already "
                "running for this chat. Poll them with write_stdin until they "
                "report an exit code before starting another."
            )
        try:
            session = await _start_sandbox_session(booter.shell, cmd, workdir, shell)
        except Exception as e:  # noqa: BLE001 - sandbox or transport failure
            logger.error("exec_command failed to start a sandbox session: %s", e)
            return f"Error starting command: {e}"
        raw = await _sandbox_exec(
            booter.shell,
            _poll_script(
                session.directory,
                0,
                wait_ms,
                session.session_id if session.tmux else None,
            ),
            timeout=max(30, wait_ms // 1000 + 15),
        )
        output, cursor, exit_code = _parse_poll(raw, 0)
        session.cursor = cursor
        if exit_code is None:
            SandboxSessions.add(umo, session)
            watch_background_session(context, umo, session.session_id, sandbox=session)
        else:
            claim_completion(umo, session.session_id)
        return format_exec_response(
            output=output,
            wall_time_seconds=time.monotonic() - started,
            exit_code=exit_code,
            session_id=session.session_id if exit_code is None else None,
            max_output_tokens=budget,
        )


@builtin_tool(config=_COMPUTER_RUNTIME_TOOL_CONFIG)
@dataclass
class WriteStdinTool(FunctionTool):
    name: str = "write_stdin"
    description: str = (
        "Writes characters to an existing exec_command session and returns recent "
        "output. Pass empty chars to poll a running command without writing."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Identifier of the running session, as reported by exec_command.",
                },
                "chars": {
                    "type": "string",
                    "description": (
                        "Bytes to write to stdin. Defaults to empty, which polls "
                        "without writing. Append a newline to submit a line."
                    ),
                },
                "yield_time_ms": {
                    "type": "number",
                    "description": "Wait before yielding output. Defaults to 5000 ms, capped at 30000 ms.",
                },
                "max_output_tokens": {
                    "type": "number",
                    "description": "Output token budget. Defaults to 10000 tokens.",
                },
            },
            "required": ["session_id"],
            "additionalProperties": False,
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        session_id: str,
        chars: str = "",
        yield_time_ms: Any = 5_000,
        max_output_tokens: Any = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ToolExecResult:
        _, permission_error = check_local_execution_permission(
            context, "Shell execution"
        )
        if permission_error:
            return permission_error
        event = context.context.event
        umo = event.unified_msg_origin
        wait_ms = clamp_yield_time(yield_time_ms, default=5_000)
        budget = clamp_output_tokens(max_output_tokens)
        booter = await get_booter(context.context.context, umo)
        started = time.monotonic()

        if is_local_runtime(context):
            if not isinstance(booter.shell, LocalShellComponent):
                return "Error: the local shell component is unavailable."
            requester_id = event.get_sender_id()
            is_admin = event.role == "admin"
            try:
                if chars:
                    await booter.shell.write_session(
                        owner_id=umo,
                        requester_id=requester_id,
                        requester_is_admin=is_admin,
                        session_id=session_id,
                        chars=chars,
                    )
                result = await booter.shell.poll_session(
                    owner_id=umo,
                    requester_id=requester_id,
                    requester_is_admin=is_admin,
                    session_id=session_id,
                    yield_time_ms=wait_ms,
                    max_output_chars=budget * CHARS_PER_TOKEN,
                )
            except ValueError as e:
                return f"Error: {e}"
            running = result.get("exit_code") is None
            if not running:
                claim_completion(umo, session_id)
            return format_exec_response(
                output=str(result.get("stdout") or ""),
                wall_time_seconds=time.monotonic() - started,
                exit_code=result.get("exit_code"),
                session_id=session_id if running else None,
                max_output_tokens=budget,
            )

        session = SandboxSessions.get(umo, session_id)
        if session is None:
            return f"Error: unknown session {session_id}."
        if chars:
            await _sandbox_exec(
                booter.shell, _write_stdin_script(session, chars), timeout=30
            )
        raw = await _sandbox_exec(
            booter.shell,
            _poll_script(
                session.directory,
                session.cursor,
                wait_ms,
                session.session_id if session.tmux else None,
            ),
            timeout=max(30, wait_ms // 1000 + 15),
        )
        output, cursor, exit_code = _parse_poll(raw, session.cursor)
        session.cursor = cursor
        if exit_code is not None:
            # The poll script already killed the pane on its way out.
            SandboxSessions.drop(umo, session_id)
            # Reported here, so the watcher behind this session says nothing.
            claim_completion(umo, session_id)
        return format_exec_response(
            output=output,
            wall_time_seconds=time.monotonic() - started,
            exit_code=exit_code,
            session_id=session_id if exit_code is None else None,
            max_output_tokens=budget,
        )
