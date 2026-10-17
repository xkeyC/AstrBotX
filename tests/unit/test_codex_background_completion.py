"""后台命令跑完后，结果以发起人的身份作为一条消息回到会话。

固定播报的问题是 agent 根本不知道结果——不进历史、不能用自己的语气说、
也不能就结果继续动作。所以走真正的一轮：发起人的回合还在跑就插进去，
否则按原有规则排队。
"""

import pytest

from astrbot.core.agent.runners.codex import native, wake
from astrbot.core.agent.runners.codex.native import ACTIVE_TURNS, ActiveTurn

UMO = "aiocqhttp:GroupMessage:30003"


class _Ctx:
    @staticmethod
    def get_config(umo=None):
        return {"agent_runner": {"runner_type": "codex"}}

    @staticmethod
    async def send_message(session, chain):
        return True


@pytest.fixture(autouse=True)
def _clean():
    ACTIVE_TURNS.clear()
    yield
    ACTIVE_TURNS.clear()


@pytest.fixture
def _no_new_turn(monkeypatch):
    """记录是否退而起了新的一轮。"""
    calls: list[str] = []

    async def _run(ctx, event, cfg, prompt, delivery):
        calls.append(event.get_sender_id())
        return True

    monkeypatch.setattr(wake, "run_in_session_thread", _run)
    return calls


@pytest.mark.asyncio
async def test_the_initiator_s_running_turn_absorbs_the_result(
    monkeypatch, _no_new_turn
):
    steered: list[tuple] = []

    async def _steer(umo, sender_id, turn_input, *, prompt="", scopes=None):
        steered.append((umo, sender_id, scopes))
        return ""

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Ctx(),
        session_str=UMO,
        sender_id="20017",
        role="member",
        session_id="sess-1",
        exit_code=0,
        output="DONE",
    )

    # Only into a turn of the same person in the same role.
    [(umo, sender, scopes)] = steered
    assert (umo, sender) == (UMO, "20017")
    assert scopes == [f"principal:{UMO.split(':')[0]}:20017:member"]
    assert _no_new_turn == [], "已经插进那一轮了，不该再起一轮"


@pytest.mark.asyncio
async def test_someone_else_s_turn_makes_it_queue(monkeypatch, _no_new_turn):
    """别人的回合在跑：插不进去，退回成新的一轮，由会话锁排队。"""

    async def _steer(umo, sender_id, turn_input, *, prompt="", scopes=None):
        return None  # try_steer 对不同发送者就是这样返回的

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Ctx(),
        session_str=UMO,
        sender_id="20017",
        role="member",
        session_id="sess-2",
        exit_code=0,
        output="DONE",
    )

    # 新的一轮仍然以发起人的身份运行，权限规则才对得上
    assert _no_new_turn == ["20017"]


@pytest.mark.asyncio
async def test_the_result_reaches_the_model(monkeypatch):
    seen: list[str] = []

    async def _run(ctx, event, cfg, prompt, delivery):
        seen.append(prompt)
        return True

    monkeypatch.setattr(wake, "run_in_session_thread", _run)

    async def _steer(*a, **k):
        return None

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Ctx(),
        session_str=UMO,
        sender_id="20017",
        role="member",
        session_id="sess-3",
        exit_code=7,
        output="BACKGROUND_DONE_7741",
    )

    prompt = seen[0]
    assert "BACKGROUND_DONE_7741" in prompt
    assert 'exit_code="7"' in prompt
    assert 'session="sess-3"' in prompt
    # 别让它以为该重跑一遍
    assert "Do not start the command again" in prompt


@pytest.mark.asyncio
async def test_a_silent_turn_still_reports_the_exit_code(monkeypatch):
    """那一轮没产出回复时，至少要让用户知道命令结束了。"""
    sent: list[str] = []

    async def _run(ctx, event, cfg, prompt, delivery):
        return False

    class _Recording(_Ctx):
        @staticmethod
        async def send_message(session, chain):
            sent.append(chain.get_plain_text())
            return True

    monkeypatch.setattr(wake, "run_in_session_thread", _run)

    async def _steer(*a, **k):
        return None

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Recording(),
        session_str=UMO,
        sender_id="20017",
        role="member",
        session_id="sess-4",
        exit_code=1,
        output="",
    )

    assert sent and "sess-4" in sent[0] and "1" in sent[0]


@pytest.mark.asyncio
async def test_an_unknown_sender_skips_straight_to_a_new_turn(
    monkeypatch, _no_new_turn
):
    """没有发起人身份时不该去插队——那会插进别人的回合。"""
    ACTIVE_TURNS[UMO] = ActiveTurn(object(), "thread-1", "turn-1", "20019")
    ACTIVE_TURNS[UMO].ready.set()
    steered = []

    async def _steer(*a, **k):
        steered.append(a)
        return ""

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Ctx(),
        session_str=UMO,
        sender_id="",
        role="member",
        session_id="sess-5",
        exit_code=0,
        output="",
    )

    assert steered == [], "没有身份就去插队，等于插进别人的回合"
    assert len(_no_new_turn) == 1, "仍然要起一轮把结果说出来"


class _RulesCtx(_Ctx):
    """A group rule grants what the catch-all rule does not."""

    @staticmethod
    def get_config(umo=None):
        return {
            "agent_runner": {"runner_type": "codex"},
            "permission_rules": [
                {"match": ["g_30003"], "global_memory": True},
                {"match": ["*"], "global_memory": False},
            ],
        }


@pytest.mark.asyncio
async def test_the_result_is_judged_by_the_group_it_was_started_in(monkeypatch):
    from astrbot.core.permission_rules import EVENT_EXTRA_KEY

    steered, queued = [], []

    async def _steer(umo, sender_id, turn_input, *, prompt="", scopes=None):
        steered.append(scopes)

    async def _run(ctx, event, cfg, prompt, delivery):
        queued.append(event)
        return True

    monkeypatch.setattr(native, "try_steer", _steer)
    monkeypatch.setattr(wake, "run_in_session_thread", _run)

    await wake.run_background_exec_completion(
        _RulesCtx(),
        session_str=UMO,
        sender_id="20017",
        group_id="30003",
        role="member",
        session_id="sess-1",
        exit_code=0,
        output="DONE",
    )

    # As in the sender's own messages there: the group rule applies.
    assert steered[0][:2] == ["memory.write_global", "memory.delete"]
    [event] = queued
    assert event.get_group_id() == "30003"
    assert event.get_extra(EVENT_EXTRA_KEY).global_memory is True


@pytest.mark.asyncio
async def test_a_scheduled_task_runs_as_in_the_group_it_was_created_in(monkeypatch):
    seen = []

    async def _run(ctx, event, cfg, prompt, delivery):
        seen.append((event.get_sender_id(), event.get_group_id()))
        return True

    monkeypatch.setattr(wake, "run_in_session_thread", _run)

    await wake.run_codex_cron_job(
        _RulesCtx(),
        message="tick",
        session_str=UMO,
        extras={
            "cron_payload": {"sender_id": "20017", "group_id": "30003"},
            "cron_job": {"id": "job-1"},
        },
    )

    assert seen == [("20017", "30003")]


@pytest.mark.asyncio
async def test_a_wake_turn_can_be_stopped_from_the_chat(monkeypatch):
    from astrbot.core.agent.runners.codex import codex_agent_runner
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import (
        codex_request,
    )
    from astrbot.core.utils.active_event_registry import active_event_registry

    seen = {}

    class Runner:
        async def reset(self, **kwargs):
            pass

        def request_stop(self):
            seen["stopped"] = True

        def was_aborted(self):
            return seen.get("stopped", False)

        async def step_until_done(self):
            # While it runs, /stop in the chat finds and stops it.
            seen["count"] = active_event_registry.request_agent_stop_all(UMO)
            if False:
                yield None

        def get_final_llm_resp(self):
            return None

    async def prepare(event, req, ctx, cfg, runner_cfg):
        pass

    import astrbot.core.astr_agent_context as agent_context

    monkeypatch.setattr(codex_agent_runner, "CodexAgentRunner", Runner)
    monkeypatch.setattr(agent_context, "AstrAgentContext", lambda **kw: kw)
    monkeypatch.setattr(agent_context, "AgentContextWrapper", lambda **kw: kw)
    monkeypatch.setattr(codex_request, "prepare_codex_request", prepare)
    from astrbot.core.cron.events import CronMessageEvent
    from astrbot.core.platform.message_session import MessageSession

    session = MessageSession.from_str(UMO)
    event = CronMessageEvent(
        context=_Ctx(),
        session=session,
        message="tick",
        message_type=session.message_type,
    )

    ok = await wake.run_in_session_thread(_Ctx(), event, _Ctx.get_config(), "tick", "")

    assert seen == {"count": 1, "stopped": True}
    # Stopped: nothing delivered, so a background command reports plainly.
    assert ok is False
    # Gone once it ended.
    assert active_event_registry.request_agent_stop_all(UMO) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plugin_set", "expected"), [(["*"], None), (["only_this"], ["only_this"])]
)
async def test_a_scheduled_task_has_only_the_sessions_plugins(
    monkeypatch, plugin_set, expected
):
    seen = []

    class Ctx(_RulesCtx):
        @staticmethod
        def get_config(umo=None):
            return {**_RulesCtx.get_config(umo), "plugin_set": plugin_set}

    async def _run(ctx, event, cfg, prompt, delivery):
        seen.append(event.plugins_name)
        return True

    monkeypatch.setattr(wake, "run_in_session_thread", _run)

    await wake.run_codex_cron_job(
        Ctx(),
        message="tick",
        session_str=UMO,
        extras={"cron_payload": {}, "cron_job": {"id": "job-1"}},
    )

    assert seen == [expected]
