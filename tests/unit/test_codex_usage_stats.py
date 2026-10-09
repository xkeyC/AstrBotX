"""Codex runs feed AstrBot's stats: per-turn usage, TTFT and the stats rows."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlmodel import select

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.runners.codex import codex_agent_runner as runner_mod
from astrbot.core.agent.runners.codex import native
from astrbot.core.agent.runners.codex.codex_agent_runner import CodexAgentRunner
from astrbot.core.agent.runners.codex.usage import UsageMeter, to_token_usage
from astrbot.core.db.po import ProviderStat
from astrbot.core.provider.entities import ProviderRequest, TokenUsage

UMO = "qq:GroupMessage:g1"


def _total(input_tokens, cached, output):
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output,
    }


# ------------------------------------------------------------ usage maths


def test_codex_input_includes_cached_tokens():
    usage = to_token_usage(_total(1000, 800, 50))

    assert usage == TokenUsage(input_other=200, input_cached=800, output=50)


def _meter(before, *totals):
    meter = UsageMeter()
    meter.start(before)
    for total in totals:
        meter.observe(total)
    return meter.spent


def test_turn_usage_is_the_growth_of_the_thread_total():
    spent = _meter(
        {"total_token_usage": _total(1000, 800, 50)}, _total(3500, 3000, 170)
    )

    assert spent == TokenUsage(input_other=300, input_cached=2200, output=120)


def test_a_new_thread_counts_from_zero():
    spent = _meter({"total_token_usage": None}, _total(900, 0, 40))

    assert spent == TokenUsage(input_other=900, output=40)


def test_an_unknown_total_changes_nothing():
    assert _meter({"total_token_usage": None}, None) == TokenUsage()


def test_a_reset_total_keeps_what_was_spent_before_it():
    # Context overflow replaces the total with a zeroed placeholder.
    meter = UsageMeter()
    meter.start({"total_token_usage": _total(1000, 800, 50)})
    meter.observe(_total(3000, 2500, 150))
    meter.observe(_total(0, 0, 0))
    meter.observe(_total(500, 0, 20))

    assert meter.spent == TokenUsage(input_other=800, input_cached=1700, output=120)


def test_repeated_totals_are_counted_once():
    meter = UsageMeter()
    meter.start({"total_token_usage": None})
    for _ in range(3):
        meter.observe(_total(900, 100, 40))

    assert meter.spent == TokenUsage(input_other=800, input_cached=100, output=40)


def test_no_baseline_means_no_meter():
    meter = UsageMeter()
    meter.start(None)
    meter.observe(_total(900, 100, 40))

    assert meter.spent is None


# ------------------------------------------------------------ a whole turn


class _Rt:
    def __init__(self, totals):
        self.totals = list(totals)

    async def thread_usage(self, thread_id):
        return json.dumps(
            {
                "model": "gpt-5.5",
                "model_provider": "openai",
                "total_token_usage": self.totals.pop(0),
            }
        )


class _Pump:
    def __init__(self, events):
        self.events = events

    def open_turn(self, on_tool, on_approval):
        queue = asyncio.Queue()
        for event in self.events:
            queue.put_nowait(event)
        return queue

    def bind_turn(self, turn):
        pass

    def close_turn(self):
        pass


class _Engine:
    def __init__(self, events, totals):
        self.rt = _Rt(totals)
        self._pump = _Pump(events)
        self.saved_image_handlers = {}
        self.lock = asyncio.Lock()

    def session_lock(self, umo):
        return self.lock

    def pump(self, thread_id):
        return self._pump

    async def submit_turn(self, thread_id, request):
        return {"status": "started", "turn_id": "turn-1"}


async def _run(monkeypatch, temp_db, events, totals, cfg=None, streaming=False):
    engine = _Engine(events, totals)

    async def get(options):
        return engine

    async def open_thread(self, engine):
        return "thread-1", None

    async def no_history(self, text):
        pass

    monkeypatch.setattr(runner_mod.CodexEngine, "get", staticmethod(get))
    monkeypatch.setattr(CodexAgentRunner, "_open_thread", open_thread)
    monkeypatch.setattr(CodexAgentRunner, "_sync_history", no_history)
    monkeypatch.setattr(runner_mod, "db_helper", temp_db)

    class _Ctx:
        event = None

    class _Wrapper:
        context = _Ctx()

    req = ProviderRequest(prompt="hi", session_id=UMO)
    runner = CodexAgentRunner()
    await runner.reset(
        request=req,
        run_context=_Wrapper(),
        agent_hooks=BaseAgentRunHooks(),
        # A short timeout: a test that runs out of events fails, not hangs.
        provider_config={"turn_timeout": 10, **(cfg or {})},
        streaming=streaming,
    )
    responses = [r async for r in runner.step_until_done()]
    async with temp_db.get_db() as session:
        rows = (await session.execute(select(ProviderStat))).scalars().all()
    return runner, responses, rows


@pytest.mark.asyncio
async def test_a_turn_is_recorded_for_the_stats_page(monkeypatch, temp_db):
    # token_count repeats itself (rate-limit refreshes); only the thread total
    # tells what the turn spent.
    count = {
        "type": "token_count",
        "info": {
            "total_token_usage": _total(3500, 3000, 170),
            "last_token_usage": _total(2000, 1900, 70),
        },
    }
    events = [
        {"type": "agent_message_content_delta", "item_id": "m", "delta": "hi"},
        count,
        count,
        {"type": "agent_message", "message": "hello", "phase": "final_answer"},
        {"type": "task_complete", "time_to_first_token_ms": 850},
    ]
    runner, responses, rows = await _run(
        monkeypatch,
        temp_db,
        events,
        [_total(1000, 800, 50), _total(3500, 3000, 170)],
    )

    assert [r.type for r in responses] == ["agent_stats", "llm_result"]
    [row] = rows
    assert row.agent_type == "codex"
    assert row.status == "completed"
    assert row.umo == UMO
    assert row.provider_id == "openai"
    assert row.provider_model == "gpt-5.5"
    assert (row.token_input_other, row.token_input_cached, row.token_output) == (
        300,
        2200,
        120,
    )
    # Codex's own measurement wins over the host-side one.
    assert row.time_to_first_token == pytest.approx(0.85)
    assert row.end_time >= row.start_time > 0

    stats = responses[0].data["chain"].chain[0].data
    assert stats["token_usage"] == {
        "input_other": 300,
        "input_cached": 2200,
        "output": 120,
    }
    assert stats["current_context_tokens"] == 2000


@pytest.mark.asyncio
async def test_without_codex_ttft_the_first_output_is_timed(monkeypatch, temp_db):
    events = [
        {"type": "agent_message", "message": "hello", "phase": "final_answer"},
        {"type": "task_complete"},
    ]
    _, _, rows = await _run(monkeypatch, temp_db, events, [None, _total(100, 0, 10)])

    [row] = rows
    assert 0 < row.time_to_first_token < 5
    assert (row.token_input_other, row.token_output) == (100, 10)


@pytest.mark.asyncio
async def test_an_aborted_turn_is_recorded_as_aborted(monkeypatch, temp_db):
    events = [{"type": "turn_aborted"}]
    _, _, rows = await _run(
        monkeypatch, temp_db, events, [_total(10, 0, 1), _total(10, 0, 1)]
    )

    [row] = rows
    assert row.status == "aborted"


@pytest.mark.asyncio
async def test_a_failed_turn_is_recorded_once_as_an_error(monkeypatch, temp_db):
    events = [
        {
            "type": "token_count",
            "info": {
                "total_token_usage": _total(300, 0, 20),
                "last_token_usage": _total(300, 0, 20),
            },
        },
        {"type": "_pump_closed", "message": "thread closed"},
    ]
    runner, responses, rows = await _run(
        monkeypatch, temp_db, events, [None, _total(300, 0, 20)]
    )

    assert [r.type for r in responses] == ["err"]
    [row] = rows
    assert row.status == "error"
    assert (row.token_input_other, row.token_output) == (300, 20)


@pytest.mark.asyncio
async def test_a_partial_answer_with_an_error_counts_as_an_error(monkeypatch, temp_db):
    events = [
        {"type": "agent_message", "message": "half", "phase": "final_answer"},
        {"type": "error", "message": "stream disconnected"},
        {"type": "task_complete"},
    ]
    _, responses, rows = await _run(
        monkeypatch, temp_db, events, [None, _total(10, 0, 1)]
    )

    assert responses[-1].type == "llm_result"
    [row] = rows
    assert row.status == "error"


@pytest.mark.asyncio
async def test_a_busy_chat_records_nothing(monkeypatch, temp_db):
    monkeypatch.setitem(native._QUEUED_TURNS, UMO, 1)

    _, responses, rows = await _run(
        monkeypatch, temp_db, [], [], cfg={"max_queued_turns": 1}
    )

    assert [r.type for r in responses] == ["llm_result"]
    assert rows == []


@pytest.mark.asyncio
async def test_an_estimate_does_not_replace_the_context_size(monkeypatch, temp_db):
    # After compaction Codex re-sends an estimate with no input count.
    real = {
        "type": "token_count",
        "info": {
            "total_token_usage": _total(1200, 0, 30),
            "last_token_usage": _total(1200, 0, 30),
        },
    }
    estimate = {
        "type": "token_count",
        "info": {
            "total_token_usage": _total(1200, 0, 30),
            "last_token_usage": {"input_tokens": 0, "total_tokens": 400},
        },
    }
    events = [
        real,
        estimate,
        {"type": "agent_message", "message": "ok", "phase": "final_answer"},
        {"type": "task_complete"},
    ]
    runner, responses, _ = await _run(
        monkeypatch, temp_db, events, [None, _total(1200, 0, 30)]
    )

    assert runner.stats.current_context_tokens == 1200
    assert runner.final_llm_resp.usage.total == 1230


def _count(total):
    return {
        "type": "token_count",
        "info": {"total_token_usage": total, "last_token_usage": total},
    }


@pytest.mark.asyncio
async def test_usage_survives_a_context_overflow_reset(monkeypatch, temp_db):
    events = [
        _count(_total(3000, 2500, 150)),
        # Overflow: Codex swaps the total for a zeroed placeholder.
        _count(_total(0, 0, 0)),
        _count(_total(500, 0, 20)),
        {"type": "agent_message", "message": "ok", "phase": "final_answer"},
        {"type": "task_complete"},
    ]
    _, _, rows = await _run(
        monkeypatch,
        temp_db,
        events,
        [_total(1000, 800, 50), _total(500, 0, 20)],
    )

    [row] = rows
    assert (row.token_input_other, row.token_input_cached, row.token_output) == (
        800,
        1700,
        120,
    )


@pytest.mark.asyncio
async def test_the_closing_total_is_read_before_the_chat_is_released(
    monkeypatch, temp_db
):
    seen_locked = []

    async def thread_usage(engine, thread_id):
        seen_locked.append(engine.lock.locked())
        return {"model": "m", "total_token_usage": _total(10, 0, 1)}

    monkeypatch.setattr(runner_mod, "thread_usage", thread_usage)
    events = [
        {"type": "agent_message", "message": "ok", "phase": "final_answer"},
        {"type": "task_complete"},
    ]
    await _run(monkeypatch, temp_db, events, [])

    # Baseline and closing total, both while this turn held the chat.
    assert seen_locked == [True, True]


# ------------------------------------------------------------ continuations


def _steered_engine(monkeypatch, fail_continuation=False, refuse=False):
    """The first submit marks a follow-up as steered in, forcing a continuation."""
    calls = []

    async def submit_turn(self, thread_id, request):
        calls.append(request)
        if len(calls) == 1:
            native.ACTIVE_TURNS[UMO].steered = 1
        elif fail_continuation:
            raise RuntimeError("thread gone")
        elif refuse:
            return {"status": "not_submitted", "reason": "busy"}
        return {"status": "started", "turn_id": f"turn-{len(calls)}"}

    monkeypatch.setattr(_Engine, "submit_turn", submit_turn)
    # One continuation: a silent one would otherwise be continued again.
    monkeypatch.setattr(runner_mod, "MAX_CONTINUATIONS", 1)
    return calls


@pytest.mark.asyncio
async def test_a_continuation_without_an_answer_keeps_the_error(monkeypatch, temp_db):
    calls = _steered_engine(monkeypatch)
    events = [
        {"type": "user_message", "message": "and also"},
        {"type": "error", "message": "stream disconnected"},
        {"type": "task_complete"},
        {"type": "task_complete"},  # the continuation, silent
    ]
    _, responses, rows = await _run(
        monkeypatch, temp_db, events, [None, _total(10, 0, 1)]
    )

    assert len(calls) == 2
    assert [r.type for r in responses] == ["err"]
    assert "stream disconnected" in responses[0].data["chain"].get_plain_text()
    [row] = rows
    assert row.status == "error"


@pytest.mark.asyncio
async def test_a_continuation_that_answers_clears_the_error(monkeypatch, temp_db):
    _steered_engine(monkeypatch)
    events = [
        {"type": "user_message", "message": "and also"},
        {"type": "error", "message": "stream disconnected"},
        {"type": "task_complete"},
        {"type": "agent_message", "message": "done", "phase": "final_answer"},
        {"type": "task_complete"},
    ]
    _, responses, rows = await _run(
        monkeypatch, temp_db, events, [None, _total(10, 0, 1)]
    )

    assert responses[-1].type == "llm_result"
    [row] = rows
    assert row.status == "completed"


@pytest.mark.asyncio
async def test_a_failed_continuation_keeps_the_first_answer(monkeypatch, temp_db):
    _steered_engine(monkeypatch, fail_continuation=True)
    events = [
        {"type": "agent_message", "message": "first", "phase": "final_answer"},
        {"type": "user_message", "message": "and also"},
        {"type": "task_complete"},
    ]
    _, responses, rows = await _run(
        monkeypatch, temp_db, events, [None, _total(10, 0, 1)]
    )

    assert responses[-1].type == "llm_result"
    assert responses[-1].data["chain"].get_plain_text().startswith("first")
    [row] = rows
    assert row.status == "completed"


@pytest.mark.asyncio
async def test_an_answer_then_an_error_and_a_silent_continuation_is_an_error(
    monkeypatch, temp_db
):
    _steered_engine(monkeypatch)
    events = [
        {"type": "agent_message", "message": "part one", "phase": "final_answer"},
        {"type": "user_message", "message": "and also"},
        {"type": "error", "message": "stream disconnected"},
        {"type": "task_complete"},
        {"type": "task_complete"},  # the continuation, silent
    ]
    _, responses, rows = await _run(
        monkeypatch, temp_db, events, [None, _total(10, 0, 1)]
    )

    # The partial answer still goes out, but the run counts as failed.
    assert responses[-1].type == "llm_result"
    [row] = rows
    assert row.status == "error"


@pytest.mark.asyncio
async def test_a_blank_continuation_answer_does_not_hide_the_error(
    monkeypatch, temp_db
):
    _steered_engine(monkeypatch)
    events = [
        {"type": "user_message", "message": "and also"},
        {"type": "error", "message": "stream disconnected"},
        {"type": "task_complete"},
        {"type": "agent_message", "message": "  ", "phase": "final_answer"},
        {"type": "task_complete"},
    ]
    _, responses, rows = await _run(
        monkeypatch, temp_db, events, [None, _total(10, 0, 1)]
    )

    assert [r.type for r in responses] == ["err"]
    [row] = rows
    assert row.status == "error"


@pytest.mark.asyncio
async def test_a_dropped_follow_up_is_not_passed_off_as_answered(monkeypatch, temp_db):
    _steered_engine(monkeypatch, fail_continuation=True)
    events = [
        {"type": "agent_message", "message": "first", "phase": "final_answer"},
        {"type": "user_message", "message": "and also"},
        {"type": "task_complete"},
    ]
    _, responses, _ = await _run(monkeypatch, temp_db, events, [None, _total(10, 0, 1)])

    text = responses[-1].data["chain"].get_plain_text()
    assert text.startswith("first")
    assert text.endswith(runner_mod.FOLLOW_UP_DROPPED_NOTE)


@pytest.mark.asyncio
async def test_a_streamed_reply_gets_the_dropped_follow_up_note(monkeypatch, temp_db):
    _steered_engine(monkeypatch, refuse=True)
    events = [
        {"type": "agent_message", "message": "first", "phase": "final_answer"},
        {"type": "user_message", "message": "and also"},
        {"type": "task_complete"},
    ]
    _, responses, rows = await _run(
        monkeypatch, temp_db, events, [None, _total(10, 0, 1)], streaming=True
    )

    deltas = [
        r.data["chain"].get_plain_text()
        for r in responses
        if r.type == "streaming_delta"
    ]
    # The streamed answer is already out, so the note follows as a delta.
    assert deltas[-1] == "\n\n" + runner_mod.FOLLOW_UP_DROPPED_NOTE
    [row] = rows
    assert row.status == "completed"


@pytest.mark.asyncio
async def test_a_follow_up_past_the_last_continuation_is_not_passed_off(
    monkeypatch, temp_db
):
    _steered_engine(monkeypatch)
    monkeypatch.setattr(runner_mod, "MAX_CONTINUATIONS", 0)
    events = [
        {"type": "agent_message", "message": "first", "phase": "final_answer"},
        {"type": "user_message", "message": "and also"},
        {"type": "task_complete"},
    ]
    _, responses, _ = await _run(monkeypatch, temp_db, events, [None, _total(10, 0, 1)])

    text = responses[-1].data["chain"].get_plain_text()
    assert text == "first\n\n" + runner_mod.FOLLOW_UP_DROPPED_NOTE


@pytest.mark.asyncio
async def test_nothing_answered_but_the_note_is_an_error(monkeypatch, temp_db):
    _steered_engine(monkeypatch, fail_continuation=True)
    events = [
        {"type": "user_message", "message": "and also"},
        {"type": "task_complete"},
    ]
    _, responses, rows = await _run(
        monkeypatch, temp_db, events, [None, _total(10, 0, 1)]
    )

    assert responses[-1].data["chain"].get_plain_text() == (
        runner_mod.FOLLOW_UP_DROPPED_NOTE
    )
    [row] = rows
    assert row.status == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("granted", [True, False])
async def test_each_turn_carries_its_senders_permission_scopes(
    monkeypatch, temp_db, granted
):
    from astrbot.core.permission_rules import EVENT_EXTRA_KEY, PermissionPolicy

    requests = []

    class Engine(_Engine):
        async def submit_turn(self, thread_id, request):
            requests.append(request)
            return await super().submit_turn(thread_id, request)

    policy = PermissionPolicy(global_memory=True) if granted else PermissionPolicy()
    extras = {EVENT_EXTRA_KEY: policy}
    event = SimpleNamespace(
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=extras.__setitem__,
        get_group_id=lambda: "group-1",
        get_sender_id=lambda: "42",
        get_sender_name=lambda: "Ann",
        get_platform_id=lambda: "qq",
        role="member",
        unified_msg_origin=UMO,
    )
    engine = Engine([{"type": "task_complete"}], [None, None])

    async def get(options):
        return engine

    monkeypatch.setattr(runner_mod.CodexEngine, "get", staticmethod(get))

    async def open_thread(self, engine):
        return "thread-1", None

    async def no_history(self, text):
        pass

    monkeypatch.setattr(CodexAgentRunner, "_open_thread", open_thread)
    monkeypatch.setattr(CodexAgentRunner, "_sync_history", no_history)
    monkeypatch.setattr(runner_mod, "db_helper", temp_db)
    runner = CodexAgentRunner()
    await runner.reset(
        request=ProviderRequest(prompt="remember this", session_id=UMO),
        run_context=SimpleNamespace(context=SimpleNamespace(event=event)),
        agent_hooks=BaseAgentRunHooks(),
        provider_config={"turn_timeout": 10, "memory_enabled": True},
        streaming=False,
    )
    [_ async for _ in runner.step_until_done()]

    # A group chat too: the sender's rule decides, turn by turn.
    [request] = requests
    granted_scopes = ["memory.write_global", "memory.delete"] if granted else []
    # Plus who they are, so nobody else's input or code cell acts as them.
    assert request["scopes"] == [*granted_scopes, "principal:qq:42:member"]


@pytest.mark.asyncio
async def test_a_continuation_keeps_its_senders_scopes(monkeypatch, temp_db):
    from astrbot.core.permission_rules import PermissionPolicy

    calls = _steered_engine(monkeypatch)
    monkeypatch.setattr(
        runner_mod, "event_policy", lambda event: PermissionPolicy(global_memory=True)
    )
    events = [
        {"type": "user_message", "message": "also save that to shared memory"},
        {"type": "task_complete"},
        {"type": "task_complete"},  # the continuation
    ]
    await _run(monkeypatch, temp_db, events, [None, _total(10, 0, 1)])

    # The follow-up is the same sender's: it acts with their rights too.
    assert [c["mode"] for c in calls] == ["start_or_steer", "start_if_idle"]
    assert calls[0]["scopes"][:2] == ["memory.write_global", "memory.delete"]
    assert calls[1]["scopes"] == calls[0]["scopes"]


# ------------------------------------------------------------ orphaned turns


@pytest.mark.asyncio
async def test_a_turn_left_running_by_another_sender_is_interrupted_not_joined(
    monkeypatch, temp_db
):
    calls, interrupts = [], []

    async def submit_turn(self, thread_id, request):
        calls.append(request)
        if len(calls) == 1:
            return {"status": "not_submitted", "reason": "ActiveTurnScopesMismatch"}
        return {"status": "started", "turn_id": "turn-2"}

    async def interrupt(self, thread_id):
        interrupts.append(thread_id)

    monkeypatch.setattr(_Engine, "submit_turn", submit_turn)
    monkeypatch.setattr(_Engine, "interrupt", interrupt, raising=False)
    events = [
        {"type": "agent_message", "message": "hi", "phase": "final_answer"},
        {"type": "task_complete"},
    ]
    _, responses, _ = await _run(monkeypatch, temp_db, events, [None, None])

    # Stopped, then started afresh with the same request.
    assert interrupts == ["thread-1"]
    assert len(calls) == 2 and calls[0] == calls[1]
    assert responses[-1].type == "llm_result"


@pytest.mark.asyncio
async def test_leaving_mid_turn_interrupts_it(monkeypatch, temp_db):
    interrupts = []

    async def interrupt(self, thread_id):
        interrupts.append(thread_id)

    monkeypatch.setattr(_Engine, "interrupt", interrupt, raising=False)
    # No terminal event: the turn is still running when the runner is cancelled.
    task = asyncio.create_task(_run(monkeypatch, temp_db, [], [None, None]))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if native.ACTIVE_TURNS.get(UMO) is not None:
            break
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Otherwise it runs on unread and the chat's next turn runs into it.
    assert interrupts == ["thread-1"]


@pytest.mark.asyncio
async def test_a_steer_carries_the_senders_scopes(monkeypatch):
    requests = []

    class Engine:
        async def submit_turn(self, thread_id, request):
            requests.append(request)
            return {"status": "steered", "turn_id": "turn-1"}

    active = native.ActiveTurn(Engine(), "thread-1", "turn-1", "42")
    monkeypatch.setitem(native.ACTIVE_TURNS, UMO, active)

    steered = await native.try_steer(
        UMO, "42", [{"type": "text", "text": "and"}], scopes=["principal:qq:42:member"]
    )

    assert steered is not None
    # Codex refuses it if the running turn is the same person under another role.
    assert requests[0]["scopes"] == ["principal:qq:42:member"]


@pytest.mark.asyncio
async def test_another_turns_end_and_words_are_not_taken_for_ours(monkeypatch, temp_db):
    # The turn interrupted just before ours still reports into our route.
    events = [
        {"type": "agent_message", "message": "stale", "_turn_id": "old"},
        {"type": "turn_aborted", "_turn_id": "old"},
        {"type": "agent_message", "message": "fresh", "_turn_id": "turn-1"},
        {"type": "task_complete", "_turn_id": "turn-1"},
    ]
    _, responses, _ = await _run(monkeypatch, temp_db, events, [None, None])

    [answer] = [r for r in responses if r.type == "llm_result"]
    assert answer.data["chain"].get_plain_text() == "fresh"


@pytest.mark.asyncio
async def test_a_stop_during_the_submit_interrupts_the_turn(monkeypatch, temp_db):
    interrupts = []
    runners = []

    async def submit_turn(self, thread_id, request):
        runners[0].request_stop()  # the user presses stop meanwhile
        return {"status": "started", "turn_id": "turn-1"}

    async def interrupt(self, thread_id):
        interrupts.append(thread_id)

    monkeypatch.setattr(_Engine, "submit_turn", submit_turn)
    monkeypatch.setattr(_Engine, "interrupt", interrupt, raising=False)
    real_init = CodexAgentRunner.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        runners.append(self)

    monkeypatch.setattr(CodexAgentRunner, "__init__", init)
    events = [{"type": "turn_aborted", "_turn_id": "turn-1"}]
    await _run(monkeypatch, temp_db, events, [None, None])

    # Stopping could not interrupt a turn that did not exist yet; it does now.
    assert interrupts == ["thread-1"]


@pytest.mark.asyncio
async def test_a_turn_stopped_while_queued_never_reaches_codex(monkeypatch, temp_db):
    calls = []

    async def submit_turn(self, thread_id, request):
        calls.append(request)
        return {"status": "started", "turn_id": "turn-1"}

    monkeypatch.setattr(_Engine, "submit_turn", submit_turn)
    real_reset = CodexAgentRunner.reset

    async def reset(self, *args, **kwargs):
        await real_reset(self, *args, **kwargs)
        self.request_stop()  # /stop while it waited for the chat

    monkeypatch.setattr(CodexAgentRunner, "reset", reset)
    runner, responses, _ = await _run(monkeypatch, temp_db, [], [None, None])

    assert calls == []
    assert [r.data["chain"].get_plain_text() for r in responses] == ["（已中断）"]


@pytest.mark.asyncio
async def test_a_thread_shut_down_mid_turn_ends_the_turn(monkeypatch):
    class Rt:
        def __init__(self):
            self.raw = [json.dumps({"id": "", "msg": {"type": "shutdown_complete"}})]

        async def next_event(self, thread_id):
            return self.raw.pop(0) if self.raw else None

    engine = SimpleNamespace(rt=Rt(), pumps={})
    pump = native.ThreadPump(engine, "thread-1")
    queue = pump.open_turn(None)
    await pump._run()

    # Not left waiting for its timeout.
    kinds = [queue.get_nowait()["type"] for _ in range(queue.qsize())]
    assert kinds == ["shutdown_complete", "_pump_closed"]


@pytest.mark.asyncio
async def test_a_streamed_turn_stopped_while_queued_says_so(monkeypatch, temp_db):
    real_reset = CodexAgentRunner.reset

    async def reset(self, *args, **kwargs):
        await real_reset(self, *args, **kwargs)
        self.request_stop()

    monkeypatch.setattr(CodexAgentRunner, "reset", reset)
    _, responses, _ = await _run(monkeypatch, temp_db, [], [None, None], streaming=True)

    # A streamed reply's final result is not sent again: the note is a delta.
    assert responses[0].type == "streaming_delta"
    assert responses[0].data["chain"].get_plain_text() == "（已中断）"


@pytest.mark.asyncio
async def test_a_shutdown_after_an_answer_keeps_the_answer(monkeypatch, temp_db):
    events = [
        {"type": "agent_message", "message": "done", "phase": "final_answer"},
        {"type": "_pump_closed", "message": "thread shut down"},
    ]
    _, responses, rows = await _run(monkeypatch, temp_db, events, [None, None])

    assert responses[-1].type == "llm_result"
    assert responses[-1].data["chain"].get_plain_text().startswith("done")


@pytest.mark.asyncio
async def test_a_follow_up_cut_short_by_a_shutdown_is_not_passed_off_as_answered(
    monkeypatch, temp_db
):
    _steered_engine(monkeypatch)
    events = [
        {"type": "agent_message", "message": "first", "phase": "final_answer"},
        {"type": "user_message", "message": "and also"},  # steered in at the end
        {"type": "task_complete"},
        # The continuation for "and also" never finishes: the thread shuts down.
        {"type": "_pump_closed", "message": "thread shut down"},
    ]
    _, responses, _ = await _run(monkeypatch, temp_db, events, [None, None])

    text = responses[-1].data["chain"].get_plain_text()
    assert text.startswith("first")
    assert runner_mod.FOLLOW_UP_DROPPED_NOTE in text


class _SteeredIn(dict):
    """A user_message whose arrival marks a follow-up steered into the turn."""

    def get(self, key, default=None):
        if key == "type" and not self.get_counted():
            native.ACTIVE_TURNS[UMO].steered += 1
            self["_counted"] = True
        return super().get(key, default)

    def get_counted(self):
        return super().get("_counted", False)


@pytest.mark.asyncio
async def test_a_follow_up_steered_into_a_continuation_counts_as_dropped(
    monkeypatch, temp_db
):
    calls = []

    async def submit_turn(self, thread_id, request):
        calls.append(request)
        return {"status": "started", "turn_id": f"turn-{len(calls)}"}

    async def place(self, path):
        return f"image at {path}"

    async def no_steer(self, engine, thread_id, active, note):
        return False

    monkeypatch.setattr(_Engine, "submit_turn", submit_turn)
    monkeypatch.setattr(runner_mod, "generated_image_path", lambda item: "img.png")
    monkeypatch.setattr(CodexAgentRunner, "_place_generated_image", place)
    monkeypatch.setattr(CodexAgentRunner, "_steer_note", no_steer)
    events = [
        {"type": "agent_message", "message": "first", "phase": "final_answer"},
        # An image note that missed the turn: a notes-only continuation.
        {"type": "item_completed", "item": {}},
        {"type": "task_complete"},
        # During it, the same sender's follow-up is steered in; then shutdown.
        _SteeredIn(type="user_message", message="and also"),
        {"type": "_pump_closed", "message": "thread shut down"},
    ]
    _, responses, _ = await _run(monkeypatch, temp_db, events, [None, None])

    assert len(calls) == 2  # the continuation was started
    text = responses[-1].data["chain"].get_plain_text()
    assert text.startswith("first")
    assert runner_mod.FOLLOW_UP_DROPPED_NOTE in text


@pytest.mark.asyncio
async def test_leaked_citation_markup_is_stripped(monkeypatch, temp_db):
    ref = chr(0xE200) + "cite" + chr(0xE202) + "turn0search6" + chr(0xE201)
    events = [
        {"type": "item_started", "item": {"type": "AgentMessage", "id": "m"}},
        {"type": "agent_message_content_delta", "item_id": "m", "delta": "答案。"},
        {"type": "agent_message_content_delta", "item_id": "m", "delta": ref[:5]},
        {"type": "agent_message_content_delta", "item_id": "m", "delta": ref[5:]},
        {"type": "agent_message_content_delta", "item_id": "m", "delta": " citeturn"},
        {"type": "agent_message_content_delta", "item_id": "m", "delta": "1search2"},
        {"type": "item_completed", "item": {"type": "AgentMessage", "id": "m"}},
        {
            "type": "agent_message",
            "message": "答案。" + ref + " citeturn1search2",
            "phase": "final_answer",
        },
        {"type": "task_complete"},
    ]
    _, responses, _ = await _run(
        monkeypatch, temp_db, events, [None, _total(10, 0, 1)], streaming=True
    )

    streamed = "".join(
        r.data["chain"].get_plain_text()
        for r in responses
        if r.type == "streaming_delta"
    )
    assert streamed == "答案。 "
    assert responses[-1].data["chain"].get_plain_text() == "答案。"
