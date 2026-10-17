import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrbot.core.tools.computer_tools import codex_exec
from astrbot.core.tools.computer_tools.codex_exec import (
    ExecCommandTool,
    SandboxSessions,
    WriteStdinTool,
    _parse_poll,
    _poll_script,
    format_exec_response,
    truncate_middle,
)


def _context(monkeypatch, booter, *, local: bool, role: str = "admin"):
    event = SimpleNamespace(
        unified_msg_origin="qq:FriendMessage:1",
        get_sender_id=lambda: "1",
        get_group_id=lambda: "",
        role=role,
    )
    ctx = SimpleNamespace(context=SimpleNamespace(event=event, context=None))

    async def fake_get_booter(_context, _umo):
        return booter

    monkeypatch.setattr(codex_exec, "get_booter", fake_get_booter)
    monkeypatch.setattr(
        codex_exec, "check_local_execution_permission", lambda *_: (None, None)
    )
    monkeypatch.setattr(codex_exec, "is_local_runtime", lambda _: local)

    async def fake_workspace_root(_context):
        return Path(tempfile.gettempdir())

    monkeypatch.setattr(codex_exec, "workspace_root_for_context", fake_workspace_root)
    return ctx


def _policy_context(role: str, policy: dict, admins: list[str]):
    """A Local-runtime tool context whose role policy is ``policy``."""
    config = {
        "admins_id": admins,
        "provider_settings": {
            "computer_use_runtime": "local",
            "computer_use_local_permissions": {role: policy},
        },
    }
    event = SimpleNamespace(
        unified_msg_origin="qq:GroupMessage:g",
        get_sender_id=lambda: "1",
        get_group_id=lambda: "g",
        role=role,
    )
    return SimpleNamespace(
        context=SimpleNamespace(
            event=event, context=SimpleNamespace(get_config=lambda umo: config)
        )
    )


class RecordingShell(codex_exec.LocalShellComponent):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def exec_managed(self, command, **kwargs):
        self.calls.append(kwargs)
        return {"session_id": "s-1", "stdout": "ok", "exit_code": 0}


def _patch_local(monkeypatch, shell, tmp_path):
    from astrbot.core.tools.computer_tools import util

    # Isolation is available (Linux bubblewrap), whatever the test host has.
    monkeypatch.setattr(
        util, "get_process_sandbox_status", lambda: {"status": "detected"}
    )

    async def fake_get_booter(_context, _umo):
        return SimpleNamespace(shell=shell)

    async def fake_workspace_root(_context):
        return tmp_path

    monkeypatch.setattr(codex_exec, "get_booter", fake_get_booter)
    monkeypatch.setattr(codex_exec, "workspace_root_for_context", fake_workspace_root)


def test_local_exec_follows_the_roles_local_policy(monkeypatch, tmp_path):
    shell = RecordingShell()
    _patch_local(monkeypatch, shell, tmp_path)

    # A member whose role may not execute is refused before anything starts.
    ctx = _policy_context(
        "member",
        {
            "allow_execution": False,
            "allow_network": False,
            "filesystem_scope": "workspace",
        },
        admins=[],
    )
    out = asyncio.run(ExecCommandTool().call(ctx, "ls"))
    assert "Permission denied" in out and not shell.calls
    out = asyncio.run(WriteStdinTool().call(ctx, "s-1", chars="y"))
    assert "Permission denied" in out

    # An admin with host access runs unisolated, with the policy's network,
    # and the run stays allowed while the policy and admin status hold.
    ctx = _policy_context(
        "admin",
        {"allow_execution": True, "allow_network": True, "filesystem_scope": "host"},
        admins=["1"],
    )
    out = asyncio.run(ExecCommandTool().call(ctx, "ls"))
    assert "Process exited with code 0" in out
    call = shell.calls[-1]
    assert call["sandboxed"] is False
    assert call["allow_network"] is True
    assert call["filesystem_scope"] == "host"
    assert call["cwd"] == str(tmp_path)
    assert call["permission_check"]() is True
    ctx.context.context.get_config(umo="")["admins_id"] = []
    assert call["permission_check"]() is False


def test_response_format_matches_codex_shape():
    text = format_exec_response(output="hello", wall_time_seconds=1.25, exit_code=0)
    assert text.splitlines()[0] == "Wall time: 1.2500 seconds"
    assert "Process exited with code 0" in text
    assert text.endswith("Output:\nhello")

    running = format_exec_response(
        output="", wall_time_seconds=0.5, session_id="abc123"
    )
    assert "Process running with session ID abc123" in running
    assert "Process exited" not in running


def test_long_output_is_truncated_in_the_middle():
    body = "x" * 10_000
    kept, truncated = truncate_middle(body, 100)
    assert truncated and len(kept) < len(body)
    assert kept.startswith("x") and kept.endswith("x")

    text = format_exec_response(
        output=body, wall_time_seconds=0.1, exit_code=0, max_output_tokens=100
    )
    assert "Chunk ID: " in text
    assert "Original token count: 2500" in text
    assert "omitted middle of output" in text


def test_local_runtime_reports_session_while_running(monkeypatch):
    calls = {}

    class Shell(codex_exec.LocalShellComponent):
        async def exec_managed(self, command, **kwargs):
            calls["command"] = command
            calls["yield_time_ms"] = kwargs["yield_time_ms"]
            return {
                "session_id": "s-1",
                "stdout": "building...",
                "exit_code": None,
            }

        async def poll_session(self, **kwargs):
            calls["poll"] = kwargs
            return {"session_id": "s-1", "stdout": "done", "exit_code": 0}

        async def write_session(self, **kwargs):
            calls["write"] = kwargs["chars"]
            return {"status": "running"}

    booter = SimpleNamespace(shell=Shell())
    ctx = _context(monkeypatch, booter, local=True)

    out = asyncio.run(ExecCommandTool().call(ctx, "make", workdir="/w"))
    assert calls["command"] == "make" and calls["yield_time_ms"] == 10_000
    assert "Process running with session ID s-1" in out
    assert out.endswith("building...")

    out = asyncio.run(WriteStdinTool().call(ctx, "s-1", chars="y\n"))
    assert calls["write"] == "y\n"
    assert "Process exited with code 0" in out


def test_local_runtime_surfaces_blocked_commands(monkeypatch):
    class Shell(codex_exec.LocalShellComponent):
        async def exec_managed(self, command, **kwargs):
            raise PermissionError("Blocked unsafe shell command.")

    ctx = _context(monkeypatch, SimpleNamespace(shell=Shell()), local=True)
    assert "Blocked unsafe shell command." in asyncio.run(
        ExecCommandTool().call(ctx, "rm -rf /")
    )


class FakeSandboxShell:
    """Minimal stand-in for the Shipyard Neo one-shot shell capability.

    ``has_tmux`` decides whether the image looks like it has tmux, which is
    what the session start script probes for.
    """

    def __init__(self, polls, has_tmux=True):
        self.polls = list(polls)
        self.has_tmux = has_tmux
        self.scripts = []

    async def exec(self, command, *, timeout=30, **_kwargs):
        self.scripts.append(command)
        if "out.log" in command and "tail -c" in command:
            return SimpleNamespace(
                output=self.polls.pop(0), error=None, model_dump=None
            )
        if "tmux new-session" in command:
            return {"output": "tmux-ok" if self.has_tmux else "", "error": ""}
        return {"output": "", "error": None}


def test_sandbox_session_lifecycle(monkeypatch):
    marker = codex_exec.META_MARKER
    shell = FakeSandboxShell(
        [
            f"waiting for input\n{marker} 18 none>>>\n",
            f"got it\n{marker} 25 0>>>\n",
        ]
    )
    ctx = _context(monkeypatch, SimpleNamespace(shell=shell), local=False)

    out = asyncio.run(ExecCommandTool().call(ctx, "python repl.py", workdir="/w"))
    assert "Process running with session ID" in out
    assert out.endswith("waiting for input")
    start_script = shell.scripts[0]
    # tmux gives the command a real PTY and mirrors the pane into the log.
    assert "tmux new-session" in start_script and "pipe-pane" in start_script
    session_id = out.split("session ID ")[1].splitlines()[0]
    session = SandboxSessions.get("qq:FriendMessage:1", session_id)
    assert session is not None and session.cursor == 18

    out = asyncio.run(WriteStdinTool().call(ctx, session_id, chars="hi\n"))
    assert any(
        "send-keys -t" in script and " -l " in script for script in shell.scripts
    )
    assert any(script.endswith("Enter") for script in shell.scripts)
    assert "Process exited with code 0" in out
    # A finished session is forgotten.
    assert SandboxSessions.get("qq:FriendMessage:1", session_id) is None

    assert "unknown session" in asyncio.run(
        WriteStdinTool().call(ctx, "does-not-exist")
    )


@pytest.mark.parametrize(
    ("raw", "cursor", "expected"),
    [
        (f"out\n{codex_exec.META_MARKER} 12 0>>>\n", 0, ("out", 12, 0)),
        (f"{codex_exec.META_MARKER} 3 none>>>\n", 3, ("", 3, None)),
        ("no marker", 5, ("no marker", 14, None)),
    ],
)
def test_parse_poll(raw, cursor, expected):
    assert _parse_poll(raw, cursor) == expected


def test_poll_script_waits_inside_the_sandbox():
    script = _poll_script("/tmp/astrbot-exec/abc", 42, 3_000)
    assert "c=42" in script and "+ 3 ))" in script
    assert "tail -c +$((c + 1))" in script
    assert codex_exec.META_MARKER in script
    # Without a tmux session there is nothing to probe or clean up.
    assert "tmux" not in script


def test_poll_script_probes_a_tmux_pane_that_died_without_an_exit_code():
    script = _poll_script("/tmp/astrbot-exec/abc", 0, 1_000, "abc")
    # A pane killed from outside, or signalled, never runs `echo $? > exit`;
    # without this probe the session would stay "running" forever.
    assert "tmux has-session -t abc" in script
    assert 'echo -1 > "$d/exit"' in script
    assert script.endswith(
        '; [ -f "$d/exit" ] && tmux kill-session -t abc >/dev/null 2>&1; true'
    )


def test_the_sandbox_runtime_keeps_skills_in_the_sandbox_and_native_exec_off(
    tmp_path,
):
    from astrbot.core.agent.runners.codex.codex_agent_runner import engine_options
    from astrbot.core.agent.runners.codex.skills import build_codex_skills_prompt
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.codex_request import (
        _build_config,
    )
    from astrbot.core.skills.skill_manager import SkillInfo

    plugin_context = SimpleNamespace(get_config=lambda: {"timezone": "UTC"})
    sandbox_config = {"provider_settings": {"computer_use_runtime": "sandbox"}}
    config = _build_config(sandbox_config, {}, plugin_context)
    assert config.computer_use_runtime == "sandbox"
    assert config.provider_settings["_codex_skills"] == "sandbox"
    assert config.provider_settings["computer_use_runtime"] == "sandbox"

    for runtime in ("none", "local"):
        plain = _build_config(
            {"provider_settings": {"computer_use_runtime": runtime}},
            # A leftover shipyard_mode key no longer means anything: the
            # runtime is the one switch.
            {"shipyard_mode": True},
            plugin_context,
        )
        assert plain.computer_use_runtime == runtime
        assert plain.provider_settings["_codex_skills"] is True
        assert plain.provider_settings["computer_use_runtime"] == "local"

    opts = engine_options(
        {
            "codex_home": str(tmp_path),
            "tool_mode": "direct",
            "native_exec_tools": True,
        }
    )
    assert opts["config"]["features.shell_tool"] is False
    assert "approve_every_command" not in opts

    skills = [SkillInfo(name="demo", description="d", path="x/SKILL.md", active=True)]
    sandbox_prompt = build_codex_skills_prompt(skills, in_sandbox=True)
    assert "/workspace/skills/demo/SKILL.md" in sandbox_prompt
    # One reader in every runtime, sandbox included.
    assert "astrbot__astrbot_read_skill" in sandbox_prompt
    assert "astrbot__astrbot_read_skill" in build_codex_skills_prompt(skills)


def test_sandbox_falls_back_to_a_fifo_without_tmux(monkeypatch):
    marker = codex_exec.META_MARKER
    shell = FakeSandboxShell([f"ready\n{marker} 6 none>>>\n"], has_tmux=False)
    ctx = _context(monkeypatch, SimpleNamespace(shell=shell), local=False)

    out = asyncio.run(ExecCommandTool().call(ctx, "top", workdir="/w"))
    session_id = out.split("session ID ")[1].splitlines()[0]
    session = SandboxSessions.get("qq:FriendMessage:1", session_id)
    assert session is not None and session.tmux is False
    assert any(
        "mkfifo" in script and "sleep 86400" in script for script in shell.scripts
    )

    shell.polls.append(f"done\n{marker} 11 0>>>\n")
    asyncio.run(WriteStdinTool().call(ctx, session_id, chars="q"))
    assert any(script.startswith("printf %s") for script in shell.scripts)


def test_write_stdin_script_shapes():
    tmux = codex_exec.SandboxSession(session_id="s1", directory="/d", tmux=True)
    fifo = codex_exec.SandboxSession(session_id="s2", directory="/d", tmux=False)

    line = codex_exec._write_stdin_script(tmux, "yes\n")
    assert "send-keys -t s1 -l -- yes" in line
    assert line.endswith("send-keys -t s1 Enter")
    # A bare control byte (Ctrl-C) is sent literally, with no Enter.
    assert codex_exec._write_stdin_script(tmux, "\x03").endswith("-l -- '\x03'")
    assert codex_exec._write_stdin_script(tmux, "\x04").endswith("-l -- '\x04'")
    assert codex_exec._write_stdin_script(tmux, "\n") == "tmux send-keys -t s1 Enter"
    # Empty input writes nothing; the caller polls instead.
    assert codex_exec._write_stdin_script(tmux, "") == "true"
    # `-l` keeps tmux key names as text, and `--` keeps a leading dash as text.
    assert codex_exec._write_stdin_script(tmux, "Enter").endswith("-l -- Enter")
    assert codex_exec._write_stdin_script(tmux, "C-c").endswith("-l -- C-c")
    assert codex_exec._write_stdin_script(tmux, "-l").endswith("-l -- -l")
    # tmux eats a trailing semicolon as a command separator unless it is
    # escaped, so `cd /x; ls;` would otherwise arrive as `cd /x; ls`.
    assert codex_exec._write_stdin_script(tmux, "cd /x; ls;\n").startswith(
        "tmux send-keys -t s1 -l -- 'cd /x; ls\\;' && "
    )
    # Only the submit newline becomes Enter; the rest stays part of the text.
    assert codex_exec._write_stdin_script(tmux, "a\n\n").startswith(
        "tmux send-keys -t s1 -l -- 'a\n'"
    )
    assert codex_exec._write_stdin_script(fifo, "q").startswith(
        "printf %s q > /d/stdin"
    )


def test_sandbox_start_script_guards_the_pane(monkeypatch):
    marker = codex_exec.META_MARKER
    shell = FakeSandboxShell([f"ok\n{marker} 3 none>>>\n"])
    ctx = _context(monkeypatch, SimpleNamespace(shell=shell), local=False)

    asyncio.run(ExecCommandTool().call(ctx, "echo semi;", workdir="/w"))
    start = shell.scripts[0]
    # The command keeps its own shell, so a command ending in `;`, in a
    # comment, or calling `exit` still records an exit code. The pane command
    # reaches the script single-quoted, hence the escaping here.
    assert (
        "sh -c 'echo semi;'; echo $? > /tmp/astrbot-exec/".replace("'", "'\"'\"'")
        in start
    )
    # Nothing runs until pipe-pane is attached: otherwise short commands lose
    # their output, and a failed pipe-pane would leave the command running in
    # an orphan pane while the fallback path starts it a second time.
    assert "while [ ! -e /tmp/astrbot-exec/" in start and "/go ]" in start
    assert start.index("pipe-pane") < start.index("/go && echo tmux-ok")
    # A partial start (pipe-pane refused, for instance) takes the fallback
    # path, so the half-built tmux session has to go with it.
    pane_id = start.split("new-session -d -s ")[1].split()[0]
    assert start.endswith(
        f"&& echo tmux-ok || {{ tmux kill-session -t {pane_id} >/dev/null 2>&1; false; }}"
    )


def test_pty_control_sequences_are_stripped_for_the_model():
    from astrbot.core.tools.computer_tools.codex_exec import clean_terminal_output

    # Colour, a cursor move, a window title and CRLF line endings.
    raw = (
        "\x1b]0;bash\x07\x1b[32mok\x1b[0m\r\n"
        "\x1b[2Kprogress 100%\r\n"
        "\x1b[1;31mfailed\x1b[m\r\n"
    )
    assert clean_terminal_output(raw) == "ok\nprogress 100%\nfailed\n"
    assert clean_terminal_output("") == ""
    # Plain output is untouched.
    assert clean_terminal_output("a\nb\n") == "a\nb\n"

    text = format_exec_response(
        output="\x1b[32mdone\x1b[0m\r\n", wall_time_seconds=0.1, exit_code=0
    )
    assert text.endswith("Output:\ndone\n")
    assert "\x1b" not in text and "\r" not in text


@pytest.mark.parametrize("workdir", ["/", "..", "../outside", str(Path.home())])
def test_a_workspace_scoped_run_cannot_leave_the_workspace(
    monkeypatch, tmp_path, workdir
):
    shell = RecordingShell()
    _patch_local(monkeypatch, shell, tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    ctx = _policy_context(
        "member",
        {
            "allow_execution": True,
            "allow_network": False,
            "filesystem_scope": "workspace",
        },
        admins=[],
    )
    out = asyncio.run(ExecCommandTool().call(ctx, "ls", workdir=workdir))
    assert out.startswith("Error: workdir must be inside the workspace")
    assert not shell.calls


def test_a_relative_workdir_is_in_the_workspace(monkeypatch, tmp_path):
    shell = RecordingShell()
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    _patch_local(monkeypatch, shell, ws)
    ctx = _policy_context(
        "member",
        {
            "allow_execution": True,
            "allow_network": False,
            "filesystem_scope": "workspace",
        },
        admins=[],
    )
    asyncio.run(ExecCommandTool().call(ctx, "ls", workdir="src"))
    assert shell.calls[-1]["cwd"] == str((ws / "src").resolve())

    # Host access may run anywhere.
    ctx = _policy_context(
        "admin",
        {"allow_execution": True, "allow_network": True, "filesystem_scope": "host"},
        admins=["1"],
    )
    asyncio.run(ExecCommandTool().call(ctx, "ls", workdir=str(tmp_path)))
    assert shell.calls[-1]["cwd"] == str(tmp_path.resolve())
