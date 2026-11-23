"""后台 shell 会话的闭环：命令跑完要能回到群里，跑不完要被收掉。

背景：`exec_command` 到 yield 时间就返回一个 session ID，后续输出只能靠
模型自己用 `write_stdin` 轮询——没有任何推送。模型一旦直接回答就结束了
回合，那条命令的结果永远没人收，tmux 面板也一直留在沙箱里。
"""

import asyncio

import pytest

from astrbot.core.tools.computer_tools import codex_exec
from astrbot.core.tools.computer_tools.codex_exec import (
    DEFAULT_MAX_LIVE_SESSIONS,
    SandboxSession,
    SandboxSessions,
    claim_completion,
)

UMO = "aiocqhttp:GroupMessage:30003"
_SESSION = SandboxSession(session_id="sess", directory="/tmp/astrbot-exec/sess")


@pytest.fixture(autouse=True)
def _clean():
    SandboxSessions._sessions.clear()
    codex_exec._claimed.clear()
    codex_exec._watchers.clear()
    yield
    SandboxSessions._sessions.clear()
    codex_exec._claimed.clear()
    codex_exec._watchers.clear()


def _ctx(sender_id: str = "20017", role: str = "member"):
    class _Ctx:
        class context:  # noqa: N801 - mirrors ContextWrapper's shape
            context = object()

            class event:  # noqa: N801
                @staticmethod
                def get_sender_id():
                    return sender_id

                @staticmethod
                def get_group_id():
                    return "30003"

    _Ctx.context.event.role = role
    return _Ctx()


def test_only_one_side_reports_a_completion():
    """agent 轮询到退出和 watcher 观察到退出，只能有一个人播报。"""
    assert claim_completion(UMO, "sess-1") is True
    assert claim_completion(UMO, "sess-1") is False
    # 不同会话互不影响
    assert claim_completion(UMO, "sess-2") is True


def test_claim_registry_stays_bounded():
    """长期运行的 bot 不能因为这个表无限增长。"""
    for i in range(codex_exec._CLAIMED_LIMIT + 50):
        claim_completion(UMO, f"sess-{i}")

    assert len(codex_exec._claimed) <= codex_exec._CLAIMED_LIMIT
    # 最早的已被淘汰，可以再次被认领——只是不会重复播报同一批
    assert claim_completion(UMO, "sess-0") is True


def test_session_count_is_per_chat():
    SandboxSessions.add(UMO, SandboxSession(session_id="a", directory="/d/a"))
    SandboxSessions.add(UMO, SandboxSession(session_id="b", directory="/d/b"))
    SandboxSessions.add("other", SandboxSession(session_id="c", directory="/d/c"))

    assert SandboxSessions.count(UMO) == 2
    assert SandboxSessions.count("other") == 1
    assert SandboxSessions.count("nobody") == 0

    SandboxSessions.drop(UMO, "a")
    assert SandboxSessions.count(UMO) == 1


def test_the_live_session_cap_is_configurable_with_a_usable_default():
    """默认值要够用，但仍然先于沙箱的进程限制拦下来。"""
    assert DEFAULT_MAX_LIVE_SESSIONS >= 128

    class _Ctx:
        class context:  # noqa: N801 - mirrors ContextWrapper's shape
            class context:  # noqa: N801
                @staticmethod
                def get_config(umo=None):
                    return {
                        "provider_settings": {"sandbox": {"max_shell_sessions": 64}}
                    }

            class event:  # noqa: N801
                unified_msg_origin = UMO

    assert codex_exec.max_live_sessions(_Ctx()) == 64


def test_the_cap_falls_back_when_config_is_unreadable():
    class _Broken:
        class context:  # noqa: N801
            context = None
            event = None

    assert codex_exec.max_live_sessions(_Broken()) == DEFAULT_MAX_LIVE_SESSIONS


@pytest.mark.asyncio
async def test_watch_background_session_starts_one_task_per_session(monkeypatch):
    started: list[str] = []

    async def _fake_watch(target):
        started.append(target.session_id)
        await asyncio.sleep(0)

    monkeypatch.setattr(codex_exec, "_watch", _fake_watch)

    codex_exec.watch_background_session(_ctx(), UMO, "sess-1", sandbox=_SESSION)
    codex_exec.watch_background_session(_ctx(), UMO, "sess-1", sandbox=_SESSION)
    codex_exec.watch_background_session(_ctx(), UMO, "sess-2", sandbox=_SESSION)
    await asyncio.gather(*list(codex_exec._watchers.values()))
    await asyncio.sleep(0)

    assert started == ["sess-1", "sess-2"], "同一个会话不该被重复监视"
    # 任务结束后要把自己从表里摘掉，否则同一个会话再也不会被监视
    assert codex_exec._watchers == {}


@pytest.mark.asyncio
async def test_the_watcher_remembers_who_started_the_command(monkeypatch):
    """结果要以发起人的身份送回去——插队和排队都按这个人算。"""
    captured: list[codex_exec._WatchTarget] = []

    async def _fake_watch(target):
        captured.append(target)

    monkeypatch.setattr(codex_exec, "_watch", _fake_watch)

    codex_exec.watch_background_session(
        _ctx(sender_id="20017", role="admin"), UMO, "sess-9", sandbox=_SESSION
    )
    await asyncio.gather(*list(codex_exec._watchers.values()))

    assert captured[0].sender_id == "20017"
    assert captured[0].is_admin is True
    # Group rules match as in the message that started it.
    assert captured[0].group_id == "30003"
    assert captured[0].umo == UMO


@pytest.mark.asyncio
async def test_a_completion_is_delivered_as_that_sender_s_message(monkeypatch):
    """闭环走的是一条消息，而不是一句固定播报——否则 agent 根本不知道结果。"""
    calls: list[dict] = []

    async def _fake_completion(ctx, **kwargs):
        calls.append(kwargs)

    import astrbot.core.agent.runners.codex.wake as wake

    monkeypatch.setattr(wake, "run_background_exec_completion", _fake_completion)
    target = codex_exec._WatchTarget(
        plugin_context=object(),
        umo=UMO,
        session_id="sess-3",
        sandbox=_SESSION,
        sender_id="20017",
        is_admin=False,
    )

    await codex_exec._deliver(target, 0, "BACKGROUND_DONE_7741\n")

    assert calls[0]["sender_id"] == "20017"
    assert calls[0]["role"] == "member"
    assert calls[0]["exit_code"] == 0
    assert "BACKGROUND_DONE_7741" in calls[0]["output"]


@pytest.mark.asyncio
async def test_a_failing_delivery_does_not_escape_the_watcher(monkeypatch):
    import astrbot.core.agent.runners.codex.wake as wake

    async def _boom(ctx, **kwargs):
        raise RuntimeError("pipeline down")

    monkeypatch.setattr(wake, "run_background_exec_completion", _boom)
    target = codex_exec._WatchTarget(
        plugin_context=object(),
        umo=UMO,
        session_id="sess-4",
        sandbox=_SESSION,
        sender_id="20017",
        is_admin=False,
    )

    await codex_exec._deliver(target, 1, "boom")
