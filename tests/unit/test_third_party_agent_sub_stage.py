import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.permission_rate_limit import keep_rate_limit_use
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import third_party
from astrbot.core.provider.entities import LLMResponse


def _stage_ctx(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        astrbot_config=config,
        plugin_manager=SimpleNamespace(
            context=SimpleNamespace(
                conversation_manager=MagicMock(),
                persona_manager=MagicMock(),
            )
        ),
    )


def _provider_settings() -> dict:
    return {
        "streaming_response": False,
        "unsupported_streaming_strategy": "turn_off",
        "third_party_stream_consumption_close_timeout_sec": 30,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner_type", ["local", "dify", "coze", "dashscope", "deerflow", "unknown"]
)
async def test_third_party_stage_rejects_non_codex_runner(runner_type: str):
    stage = third_party.ThirdPartyAgentSubStage()
    config = {
        "agent_runner": {"runner_type": runner_type, "config": {}},
        "provider_settings": _provider_settings(),
    }

    with pytest.raises(ValueError, match="Unsupported third party agent runner"):
        await stage.initialize(_stage_ctx(config))


@pytest.mark.asyncio
async def test_codex_runner_receives_inline_profile_config(
    monkeypatch: pytest.MonkeyPatch,
):
    inline_config = {"model": "inline-model", "tool_call_timeout": 60}
    runner = MagicMock()
    runner.reset = AsyncMock()
    runner.get_final_llm_resp.return_value = LLMResponse(
        role="assistant",
        result_chain=MessageChain().message("done"),
    )
    runner.close = AsyncMock()

    async def step_until_done(max_step: int = 30):
        _ = max_step
        if False:
            yield None

    runner.step_until_done = step_until_done
    runner_factory_calls = []

    class RunnerFactory:
        @classmethod
        def __class_getitem__(cls, item):
            _ = item
            return cls

        def __new__(cls):
            runner_factory_calls.append(True)
            return runner

    monkeypatch.setattr(third_party, "CodexAgentRunner", RunnerFactory)
    monkeypatch.setattr(third_party, "prepare_codex_request", AsyncMock())
    monkeypatch.setattr(third_party, "try_steer", AsyncMock(return_value=None))
    monkeypatch.setattr(third_party, "build_turn_input", MagicMock(return_value=[]))
    registry = MagicMock()
    monkeypatch.setattr(third_party, "active_event_registry", registry)
    monkeypatch.setattr(
        third_party, "AstrAgentContext", MagicMock(return_value=object())
    )
    monkeypatch.setattr(
        third_party, "AgentContextWrapper", MagicMock(return_value=object())
    )
    monkeypatch.setattr(third_party, "call_event_hook", AsyncMock(return_value=False))
    monkeypatch.setattr(third_party.Metric, "upload", AsyncMock(return_value=None))

    config = {
        "agent_runner": {"runner_type": "codex", "config": inline_config},
        "provider_settings": _provider_settings(),
    }
    stage = third_party.ThirdPartyAgentSubStage()
    await stage.initialize(_stage_ctx(config))
    stage._resolve_persona_custom_error_message = AsyncMock(return_value=None)
    event = MagicMock()
    event.message_str = "hello"
    event.unified_msg_origin = "webchat:FriendMessage:test"
    event.message_obj.message = []
    event.platform_meta.support_streaming_message = True
    event.get_extra.return_value = None

    results = [item async for item in stage.process(event, "")]

    assert results == [None]
    assert runner.reset.await_args.kwargs["provider_config"] is inline_config
    assert runner_factory_calls == [True]
    registry.register_agent_stop_callback.assert_called_once()
    registry.unregister_agent_stop_callback.assert_called_once_with(event)


class _StatsRunner:
    """Yields a stats chunk, then the reply."""

    def __init__(self):
        self.stats_chain = MessageChain(type="agent_stats")
        self.reply = MessageChain().message("hi")

    async def step_until_done(self, max_step=30):
        from astrbot.core.agent.response import AgentResponse

        yield AgentResponse(type="agent_stats", data={"chain": self.stats_chain})
        yield AgentResponse(type="llm_result", data={"chain": self.reply})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "sent"), [("webchat", True), ("aiocqhttp", False)]
)
async def test_run_stats_reach_webchat_only(platform: str, sent: bool):
    runner = _StatsRunner()
    event = MagicMock()
    event.get_platform_name.return_value = platform
    event.send = AsyncMock()

    chunks = [
        chain
        async for chain, _ in third_party.run_third_party_agent(
            runner, stream_to_general=True, event=event
        )
    ]

    # The stats never become part of the reply.
    assert chunks == [runner.reply]
    if sent:
        event.send.assert_awaited_once_with(runner.stats_chain)
    else:
        event.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_stats_send_does_not_cost_the_reply():
    runner = _StatsRunner()
    event = MagicMock()
    event.get_platform_name.return_value = "webchat"
    event.send = AsyncMock(side_effect=ConnectionError("client gone"))

    results = [
        pair
        async for pair in third_party.run_third_party_agent(
            runner, stream_to_general=True, event=event
        )
    ]

    assert results == [(runner.reply, False)]


def _group_request():
    from astrbot.core.provider.entities import ProviderRequest

    req = ProviderRequest(prompt="hi")
    req.add_persistent_context("group_history", "<group_history>x</group_history>")
    req.add_persistent_context("message_meta", "Sender: Bob (ID: 2)")
    return req


async def _process_with(
    monkeypatch,
    *,
    hook_stops=False,
    steer=None,
    reset=None,
    streaming=False,
    raises=None,
    mock_history=True,
    watchdog_s=None,
    restore_gate=None,
    prepare=None,
):
    """Runs the stage once.

    Returns (release, keep, event, runner, stage). With mock_history off the
    real release/keep run, against a restore callback recorded on the event.
    """
    runner = MagicMock()
    runner.req = _group_request()
    runner.reset = reset or AsyncMock()
    runner.get_final_llm_resp.return_value = LLMResponse(
        role="assistant", result_chain=MessageChain().message("done")
    )
    runner.close = AsyncMock()

    async def step_until_done(max_step: int = 30):
        # An accepted turn keeps its history and its rate-limit use, as the
        # Codex runner does.
        third_party.keep_group_history(event)
        keep_rate_limit_use(event)
        if False:
            yield None

    runner.step_until_done = step_until_done

    class RunnerFactory:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

        def __new__(cls):
            return runner

    release, keep = AsyncMock(), MagicMock()
    if mock_history:
        monkeypatch.setattr(third_party, "release_group_history", release)
        monkeypatch.setattr(third_party, "keep_group_history", keep)
    monkeypatch.setattr(third_party, "CodexAgentRunner", RunnerFactory)
    monkeypatch.setattr(third_party, "prepare_codex_request", prepare or AsyncMock())
    monkeypatch.setattr(third_party, "try_steer", steer or AsyncMock(return_value=None))
    monkeypatch.setattr(third_party, "build_turn_input", MagicMock(return_value=[]))
    monkeypatch.setattr(third_party, "active_event_registry", MagicMock())
    monkeypatch.setattr(
        third_party, "AstrAgentContext", MagicMock(return_value=object())
    )
    monkeypatch.setattr(
        third_party, "AgentContextWrapper", MagicMock(return_value=object())
    )
    monkeypatch.setattr(
        third_party, "call_event_hook", AsyncMock(return_value=hook_stops)
    )
    monkeypatch.setattr(third_party.Metric, "upload", AsyncMock(return_value=None))

    settings = {**_provider_settings(), "streaming_response": streaming}
    config = {
        "agent_runner": {"runner_type": "codex", "config": {}},
        "provider_settings": settings,
    }
    stage = third_party.ThirdPartyAgentSubStage()
    await stage.initialize(_stage_ctx(config))
    if watchdog_s is not None:
        stage.stream_consumption_close_timeout_sec = watchdog_s
    stage._resolve_persona_custom_error_message = AsyncMock(return_value=None)
    extras = {}
    event = MagicMock()
    event.message_str = "hello"
    event.unified_msg_origin = "qq:GroupMessage:g"
    event.message_obj.message = []
    event.platform_meta.support_streaming_message = True
    event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)
    event.set_extra.side_effect = extras.__setitem__
    event.restored = []

    async def restore():
        if restore_gate is not None:
            await restore_gate.wait()  # e.g. waiting on the history lock
        event.restored.append(True)

    extras[third_party_restore_key()] = restore
    if raises is not None:
        with pytest.raises(raises):
            [item async for item in stage.process(event, "")]
    else:
        [item async for item in stage.process(event, "")]
    return release, keep, event, runner, stage


def third_party_restore_key():
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        GROUP_HISTORY_RESTORE_KEY,
    )

    return GROUP_HISTORY_RESTORE_KEY


@pytest.mark.asyncio
async def test_a_request_stopped_by_a_plugin_gives_group_history_back(monkeypatch):
    release, keep, event, *_ = await _process_with(monkeypatch, hook_stops=True)

    release.assert_awaited_once_with(event)
    keep.assert_not_called()


@pytest.mark.asyncio
async def test_a_steered_request_keeps_group_history(monkeypatch):
    release, keep, event, *_ = await _process_with(
        monkeypatch, steer=AsyncMock(return_value="run-1")
    )

    keep.assert_called_once_with(event)
    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_steer_gives_group_history_back_and_still_fails(monkeypatch):
    release, _, event, *_ = await _process_with(
        monkeypatch,
        steer=AsyncMock(side_effect=RuntimeError("boom")),
        raises=RuntimeError,
    )

    release.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_a_streamed_run_that_never_started_gives_group_history_back(
    monkeypatch,
):
    release, _, event, *_ = await _process_with(
        monkeypatch,
        streaming=True,
        reset=AsyncMock(side_effect=RuntimeError("bad config")),
        raises=RuntimeError,
    )

    release.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_an_accepted_run_does_not_give_history_back(monkeypatch):
    # Real release/keep: the finally's release must be a no-op after keep.
    _, _, event, *_ = await _process_with(monkeypatch, mock_history=False)

    assert event.restored == []


@pytest.mark.asyncio
async def test_an_unconsumed_stream_gives_history_back_and_a_late_run_drops_it(
    monkeypatch,
):
    _, _, event, runner, _ = await _process_with(
        monkeypatch, streaming=True, mock_history=False, watchdog_s=0
    )
    await asyncio.sleep(0.05)  # the watchdog fires: nobody consumed the stream

    assert event.restored == [True]
    # The respond stage turns up late after all: the run goes ahead, without
    # the history it gave back.
    result = event.set_result.call_args_list[0].args[0]
    [_ async for _ in result.async_stream]
    units = [p.text for p in runner.req.persistent_user_context_parts]
    assert not any("group_history" in u for u in units)
    assert any("message_meta" in u for u in units)
    assert event.restored == [True]


@pytest.mark.asyncio
async def test_a_consumer_arriving_mid_give_back_does_not_lose_the_history(
    monkeypatch,
):
    gate = asyncio.Event()
    _, _, event, runner, _ = await _process_with(
        monkeypatch,
        streaming=True,
        mock_history=False,
        watchdog_s=0,
        restore_gate=gate,
    )
    await asyncio.sleep(0.05)  # the watchdog is now giving the history back
    assert event.restored == []

    # The consumer turns up and cancels the watchdog mid-way.
    result = event.set_result.call_args_list[0].args[0]
    [_ async for _ in result.async_stream]
    gate.set()
    await asyncio.sleep(0.05)

    # Given back all the same, and not shown by the late run.
    assert event.restored == [True]
    units = [p.text for p in runner.req.persistent_user_context_parts]
    assert not any("group_history" in u for u in units)
    # The late run was closed again when it ended, so a live turn would be
    # interrupted rather than left running.
    assert runner.close.await_count == 2


@pytest.mark.asyncio
async def test_a_consumer_right_after_the_watchdog_check_sees_the_runner_closed(
    monkeypatch,
):
    """No gap between the watchdog deciding to close and the runner being closed."""
    consumed_flags = []
    real_start = third_party._start_stream_watchdog

    checked = asyncio.Event()

    def start(*, timeout_sec, is_stream_consumed, claim_close):
        def check():
            consumed = is_stream_consumed()
            consumed_flags.append(consumed)
            checked.set()
            return consumed

        return real_start(
            timeout_sec=timeout_sec,
            is_stream_consumed=check,
            claim_close=claim_close,
        )

    monkeypatch.setattr(third_party, "_start_stream_watchdog", start)
    _, _, event, runner, _ = await _process_with(
        monkeypatch, streaming=True, mock_history=False, watchdog_s=0
    )
    # Let the watchdog run its check, then consume in the very next step,
    # before the shielded close has had a turn.
    await checked.wait()
    result = event.set_result.call_args_list[0].args[0]
    [_ async for _ in result.async_stream]
    await asyncio.sleep(0.05)

    assert event.restored == [True]
    units = [p.text for p in runner.req.persistent_user_context_parts]
    assert not any("group_history" in u for u in units)
    assert runner.close.await_count == 2
    assert consumed_flags == [False]  # the watchdog did decide to close


@pytest.mark.asyncio
@pytest.mark.parametrize(("refusal", "replied"), [("too fast", True), ("", False)])
async def test_a_rate_limited_request_never_reaches_the_agent(
    monkeypatch, refusal, replied
):
    prepare = AsyncMock()
    forget = AsyncMock()
    monkeypatch.setattr(
        third_party, "check_rate_limit", AsyncMock(return_value=refusal)
    )
    monkeypatch.setattr(third_party, "forget_group_message", forget)
    monkeypatch.setattr(third_party, "prepare_codex_request", prepare)
    config = {
        "agent_runner": {"runner_type": "codex", "config": {}},
        "provider_settings": _provider_settings(),
    }
    stage = third_party.ThirdPartyAgentSubStage()
    await stage.initialize(_stage_ctx(config))
    event = MagicMock()
    event.message_str = "hello"
    event.message_obj.message = []

    [item async for item in stage.process(event, "")]

    prepare.assert_not_awaited()
    forget.assert_awaited_once_with(event)
    assert event.set_result.called is replied
    if replied:
        [result] = event.set_result.call_args.args
        assert result.get_plain_text() == "too fast"


@pytest.mark.asyncio
async def test_an_allowed_request_goes_on(monkeypatch):
    monkeypatch.setattr(third_party, "check_rate_limit", AsyncMock(return_value=None))
    forget = AsyncMock()
    monkeypatch.setattr(third_party, "forget_group_message", forget)

    *_, runner, _stage = await _process_with(monkeypatch)

    forget.assert_not_awaited()
    runner.reset.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_request_a_plugin_stops_gives_its_use_back(monkeypatch):
    refund = AsyncMock()
    monkeypatch.setattr(third_party, "refund_rate_limit", refund)

    *_, event, _runner, _stage = await _process_with(monkeypatch, hook_stops=True)

    refund.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_a_failed_steer_gives_its_use_back(monkeypatch):
    refund = AsyncMock()
    monkeypatch.setattr(third_party, "refund_rate_limit", refund)

    await _process_with(
        monkeypatch, steer=AsyncMock(side_effect=RuntimeError("x")), raises=RuntimeError
    )

    refund.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reset", "given_back"),
    [(None, False), (AsyncMock(side_effect=RuntimeError("bad config")), True)],
)
async def test_only_a_run_codex_never_accepted_gives_its_use_back(
    monkeypatch, reset, given_back
):
    from astrbot.core.permission_rate_limit import USE_EXTRA_KEY

    limiter = MagicMock()
    limiter.refund = AsyncMock()
    real_check = AsyncMock(return_value=None)

    async def check(event, conf):
        event.set_extra(USE_EXTRA_KEY, (limiter, 7))
        return await real_check()

    monkeypatch.setattr(third_party, "check_rate_limit", check)

    await _process_with(
        monkeypatch, reset=reset, raises=RuntimeError if reset else None
    )

    assert limiter.refund.await_count == (1 if given_back else 0)


@pytest.mark.asyncio
async def test_an_unconsumed_stream_keeps_its_use_for_a_late_run(monkeypatch):
    refund = AsyncMock()
    monkeypatch.setattr(third_party, "refund_rate_limit", refund)
    _, _, event, _runner, _ = await _process_with(
        monkeypatch, streaming=True, mock_history=False, watchdog_s=0
    )
    await asyncio.sleep(0.05)  # the watchdog closes the unconsumed stream

    # A late consumer may still run it: nothing is given back yet.
    refund.assert_not_awaited()
    result = event.set_result.call_args_list[0].args[0]
    [_ async for _ in result.async_stream]
    # The late run's own close settles it (a no-op once Codex accepted it).
    refund.assert_awaited_with(event)


@pytest.mark.asyncio
async def test_a_consumer_arriving_mid_close_keeps_the_accepted_use(monkeypatch):
    from astrbot.core.permission_rate_limit import USE_EXTRA_KEY

    limiter = MagicMock()
    limiter.refund = AsyncMock()

    async def check(event, conf):
        event.set_extra(USE_EXTRA_KEY, (limiter, 7))

    monkeypatch.setattr(third_party, "check_rate_limit", check)
    gate = asyncio.Event()
    _, _, event, _runner, _ = await _process_with(
        monkeypatch,
        streaming=True,
        mock_history=False,
        watchdog_s=0,
        restore_gate=gate,
    )
    await asyncio.sleep(0.05)  # the watchdog's close waits on the history

    # A consumer turns up meanwhile; Codex accepts the late run only after
    # the watchdog's close has finished.
    accepted = asyncio.Event()

    async def step_until_done(max_step: int = 30):
        await accepted.wait()
        keep_rate_limit_use(event)
        if False:
            yield None

    _runner.step_until_done = step_until_done
    result = event.set_result.call_args_list[0].args[0]
    consumer = asyncio.create_task(_drain(result.async_stream))
    await asyncio.sleep(0.05)
    gate.set()
    await asyncio.sleep(0.05)
    accepted.set()
    await consumer

    limiter.refund.assert_not_awaited()


async def _drain(stream):
    return [_ async for _ in stream]


@pytest.mark.asyncio
async def test_a_message_stopped_on_its_way_never_runs(monkeypatch):
    check = AsyncMock(return_value=None)
    prepare = AsyncMock()
    monkeypatch.setattr(third_party, "check_rate_limit", check)
    monkeypatch.setattr(third_party, "prepare_codex_request", prepare)
    config = {
        "agent_runner": {"runner_type": "codex", "config": {}},
        "provider_settings": _provider_settings(),
    }
    stage = third_party.ThirdPartyAgentSubStage()
    await stage.initialize(_stage_ctx(config))
    extras = {"agent_stop_requested": True}  # /stop came before it got here
    event = MagicMock()
    event.message_str = "hello"
    event.message_obj.message = []
    event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)

    [item async for item in stage.process(event, "")]

    # Nothing counted against its quota, nothing prepared or run.
    check.assert_not_awaited()
    prepare.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_message_stopped_on_its_way_leaves_the_group_history(monkeypatch):
    forget = AsyncMock()
    monkeypatch.setattr(third_party, "forget_group_message", forget)
    config = {
        "agent_runner": {"runner_type": "codex", "config": {}},
        "provider_settings": _provider_settings(),
    }
    stage = third_party.ThirdPartyAgentSubStage()
    await stage.initialize(_stage_ctx(config))
    extras = {"agent_stop_requested": True}
    event = MagicMock()
    event.message_str = "hello"
    event.message_obj.message = []
    event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)

    [item async for item in stage.process(event, "")]

    # No later request answers the cancelled one from the history.
    forget.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_a_message_stopped_while_prepared_is_not_steered_in(monkeypatch):
    async def prepare(event, *args, **kwargs):
        event.set_extra("agent_stop_requested", True)

    steer = AsyncMock(return_value="run-1")
    refund = AsyncMock()
    monkeypatch.setattr(third_party, "check_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(third_party, "refund_rate_limit", refund)
    monkeypatch.setattr(third_party, "forget_group_message", AsyncMock())
    release, keep, event, runner, _ = await _process_with(
        monkeypatch, prepare=prepare, steer=steer
    )

    steer.assert_not_awaited()
    release.assert_awaited_once_with(event)
    refund.assert_awaited_once_with(event)
    runner.reset.assert_not_awaited()


async def _prepared_request(monkeypatch, wake_prefix: str, extras: dict):
    """Runs the Codex stage up to the request hook; the prepare call or None."""
    prepare = AsyncMock()
    monkeypatch.setattr(third_party, "prepare_codex_request", prepare)
    monkeypatch.setattr(third_party, "check_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(third_party, "refund_rate_limit", AsyncMock())
    monkeypatch.setattr(third_party, "release_group_history", AsyncMock())
    # Stops right after preparing: only the request matters here.
    monkeypatch.setattr(third_party, "call_event_hook", AsyncMock(return_value=True))
    config = {
        "agent_runner": {"runner_type": "codex", "config": {}},
        "provider_settings": _provider_settings(),
    }
    stage = third_party.ThirdPartyAgentSubStage()
    await stage.initialize(_stage_ctx(config))
    stage._resolve_persona_custom_error_message = AsyncMock(return_value=None)
    event = MagicMock()
    event.message_str = "the weather is nice today"
    event.unified_msg_origin = "aiocqhttp:GroupMessage:456"
    event.get_platform_name.return_value = "aiocqhttp"
    event.message_obj.message = []
    event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)
    event.set_extra.side_effect = extras.__setitem__

    [item async for item in stage.process(event, wake_prefix)]
    return prepare.await_args


@pytest.mark.asyncio
async def test_an_active_reply_runs_despite_the_llm_wake_prefix(monkeypatch):
    """群聊上下文感知的主动回复：设置了 LLM 唤醒前缀时也要触发（旧 bug：被前缀拦下）。"""
    from astrbot.core.provider.entities import ProviderRequest

    active_reply = ProviderRequest(
        prompt="the weather is nice today",
        image_urls=["/tmp/cat.png"],
        system_prompt="Chime in briefly.",
    )

    call = await _prepared_request(
        monkeypatch, "ai", {"provider_request": active_reply}
    )

    assert call is not None, "the active reply never reached Codex"
    req = call.args[1]
    assert req.prompt == "the weather is nice today"
    assert req.image_urls == ["/tmp/cat.png"]
    # The plugin's instructions go with this request only: as the request's
    # system prompt they would stay in the Codex thread for later messages.
    assert req.system_prompt == ""
    [note] = [part.text for part in req.dynamic_user_context_parts]
    assert note.startswith('<request_context name="plugin_instructions">')
    assert "Chime in briefly." in note
    assert req.session_id == "aiocqhttp:GroupMessage:456"
    # Its own attachments, not the message's.
    assert call.kwargs["collect_media"] is False
    # The plugin's request object is not changed.
    assert active_reply.session_id == ""


@pytest.mark.asyncio
async def test_a_plain_message_without_the_wake_prefix_still_does_not_run(
    monkeypatch,
):
    assert await _prepared_request(monkeypatch, "ai", {}) is None


@pytest.mark.asyncio
async def test_a_plugin_request_replaces_the_message_prompt(monkeypatch):
    from astrbot.core.provider.entities import ProviderRequest

    call = await _prepared_request(
        monkeypatch, "", {"provider_request": ProviderRequest(prompt="summarize")}
    )

    assert call.args[1].prompt == "summarize"
    assert call.kwargs["collect_media"] is False


@pytest.mark.asyncio
async def test_a_plugin_request_is_answered_once(monkeypatch):
    # A steered request leaves no result, and the process stage runs the
    # agent again for the event: the plugin's request must not go twice.
    from astrbot.core.provider.entities import ProviderRequest

    extras = {"provider_request": ProviderRequest(prompt="summarize")}
    assert await _prepared_request(monkeypatch, "", extras) is not None
    assert extras["provider_request"] is None
