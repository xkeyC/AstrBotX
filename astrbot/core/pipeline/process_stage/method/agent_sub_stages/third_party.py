import asyncio
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING

import astrbot.core.provider.provider as provider_core
from astrbot.core import content_moderation, logger
from astrbot.core.agent.runners.codex.codex_agent_runner import (
    CodexAgentRunner,
    build_turn_input,
    drop_group_history,
    forget_group_message,
    give_back,
    keep_group_history,
    release_group_history,
    take_group_history,
    turn_scopes,
)
from astrbot.core.agent.runners.codex.constants import CODEX_RUNNER_TYPE
from astrbot.core.agent.runners.codex.native import try_steer
from astrbot.core.astr_agent_hooks import MAIN_AGENT_HOOKS
from astrbot.core.message.components import Image, Record
from astrbot.core.message.message_event_result import (
    MessageChain,
    MessageEventResult,
    ResultContentType,
)
from astrbot.core.permission_rate_limit import check_rate_limit, refund_rate_limit
from astrbot.core.persona_error_reply import (
    resolve_event_conversation_persona_id,
    resolve_persona_custom_error_message,
    set_persona_custom_error_message_on_event,
)

if TYPE_CHECKING:
    from astrbot.core.agent.runners.base import BaseAgentRunner
    from astrbot.core.provider.entities import LLMResponse
from astrbot.core.pipeline.stage import Stage
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.provider.entities import (
    ProviderRequest,
)
from astrbot.core.star.star_handler import EventType
from astrbot.core.utils.active_event_registry import active_event_registry
from astrbot.core.utils.config_number import coerce_int_config
from astrbot.core.utils.metrics import Metric

from .....astr_agent_context import AgentContextWrapper, AstrAgentContext
from ....context import PipelineContext, call_event_hook
from .codex_request import prepare_codex_request

THIRD_PARTY_RUNNER_ERROR_EXTRA_KEY = "_third_party_runner_error"
# Set on every active event of a chat by /stop (active_event_registry).
AGENT_STOP_REQUESTED = "agent_stop_requested"
STREAM_CONSUMPTION_CLOSE_TIMEOUT_SEC = 30
RUNNER_NO_RESULT_FALLBACK_MESSAGE = "Agent Runner did not return any result."
RUNNER_NO_FINAL_RESPONSE_LOG = (
    "Agent Runner returned no final response, fallback to streamed error/result chain."
)
RUNNER_NO_RESULT_LOG = "Agent Runner did not return final result."


def _resolve_third_party_streaming_mode(
    streaming_response: bool,
    stream_to_general: bool,
) -> tuple[bool, bool]:
    all_streaming = provider_core.ENABLE_ALL_STREAMING_MODE
    streaming_used = streaming_response and not stream_to_general
    runner_streaming = streaming_response or all_streaming
    suppress_streaming_deltas = stream_to_general or (
        all_streaming and not streaming_used
    )
    return runner_streaming, suppress_streaming_deltas


async def run_third_party_agent(
    runner: "BaseAgentRunner",
    stream_to_general: bool = False,
    custom_error_message: str | None = None,
    event: AstrMessageEvent | None = None,
) -> AsyncGenerator[tuple[MessageChain, bool], None]:
    """
    运行第三方 agent runner 并转换响应格式
    类似于 run_agent 函数，但专门处理第三方 agent runner
    """
    try:
        async for resp in runner.step_until_done(max_step=30):  # type: ignore[misc]
            if resp.type == "streaming_delta":
                if stream_to_general:
                    continue
                yield resp.data["chain"], False
            elif resp.type == "llm_result":
                if stream_to_general:
                    yield resp.data["chain"], False
            elif resp.type == "err":
                yield resp.data["chain"], True
            elif resp.type == "agent_stats":
                # WebChat shows per-reply usage; other platforms have no use
                # for it (the stats page reads the database).
                if event is not None and event.get_platform_name() == "webchat":
                    try:
                        await event.send(resp.data["chain"])
                    except Exception as e:  # noqa: BLE001 - never costs the reply
                        logger.warning("Sending agent stats failed: %s", e)
    except Exception as e:
        logger.error(f"Third party agent runner error: {e}")
        err_msg = custom_error_message
        if not err_msg:
            err_msg = (
                f"Error occurred during AI execution.\n"
                f"Error Type: {type(e).__name__} (3rd party)\n"
                f"Error Message: {str(e)}"
            )
        yield MessageChain().message(err_msg), True


class _RunnerResultAggregator:
    def __init__(self) -> None:
        self.merged_chain = MessageChain()
        self.has_error = False
        self.has_metadata = False

    def add_chunk(self, chain: MessageChain, is_error: bool) -> None:
        if not self.has_metadata:
            self.merged_chain.inherit_metadata(chain)
            self.has_metadata = True
        elif chain.disable_segment_reply:
            self.merged_chain.disable_segment_reply = True
        if chain.use_t2i_ is not None and self.merged_chain.use_t2i_ is None:
            self.merged_chain.use_t2i_ = chain.use_t2i_
        if chain.use_markdown_ is not None and self.merged_chain.use_markdown_ is None:
            self.merged_chain.use_markdown_ = chain.use_markdown_
        if chain.type is not None and self.merged_chain.type is None:
            self.merged_chain.type = chain.type
        self.merged_chain.chain.extend(chain.chain or [])
        if is_error:
            self.has_error = True

    def finalize(
        self,
        final_resp: "LLMResponse | None",
    ) -> tuple[MessageChain, bool]:
        if not final_resp or not final_resp.result_chain:
            if self.merged_chain.chain:
                logger.warning(RUNNER_NO_FINAL_RESPONSE_LOG)
                return self.merged_chain, self.has_error

            logger.warning(RUNNER_NO_RESULT_LOG)
            fallback_error_chain = MessageChain().message(
                RUNNER_NO_RESULT_FALLBACK_MESSAGE,
            )
            return fallback_error_chain, True

        final_chain = final_resp.result_chain
        if self.has_metadata:
            if final_chain.use_t2i_ is None:
                final_chain.use_t2i_ = self.merged_chain.use_t2i_
            if final_chain.use_markdown_ is None:
                final_chain.use_markdown_ = self.merged_chain.use_markdown_
            if final_chain.type is None:
                final_chain.type = self.merged_chain.type
            final_chain.disable_segment_reply = (
                final_chain.disable_segment_reply
                or self.merged_chain.disable_segment_reply
            )
        is_runner_error = self.has_error or final_resp.role == "err"
        return final_chain, is_runner_error


def _start_stream_watchdog(
    *,
    timeout_sec: int,
    is_stream_consumed: Callable[[], bool],
    claim_close: Callable[[], Callable[[], Awaitable[None]] | None],
) -> asyncio.Task[None]:
    async def _watchdog() -> None:
        try:
            await asyncio.sleep(timeout_sec)
        except asyncio.CancelledError:
            return
        if not is_stream_consumed():
            logger.warning(
                "Third-party runner stream was never consumed in %ss; closing runner to avoid resource leak.",
                timeout_sec,
            )
            # The close is claimed in the same step as the check, so a consumer
            # arriving next already sees the runner closed; the rest is
            # shielded, as that consumer cancels the watchdog, which must not
            # cut short giving the group history back.
            finish_close = claim_close()
            if finish_close is None:
                return
            try:
                await asyncio.shield(finish_close())
            except Exception:
                logger.warning(
                    "Exception while closing third-party runner from stream watchdog.",
                    exc_info=True,
                )

    return asyncio.create_task(_watchdog())


async def _close_runner_if_supported(runner: "BaseAgentRunner") -> None:
    close_callable = getattr(runner, "close", None)
    if not callable(close_callable):
        return

    try:
        close_result = close_callable()
        if inspect.isawaitable(close_result):
            await close_result
    except Exception as e:
        logger.warning(f"Failed to close third-party runner cleanly: {e}")


class ThirdPartyAgentSubStage(Stage):
    async def initialize(self, ctx: PipelineContext) -> None:
        self.ctx = ctx
        self.conf = ctx.astrbot_config
        agent_runner = self.conf["agent_runner"]
        self.runner_type = agent_runner["runner_type"]
        if self.runner_type != CODEX_RUNNER_TYPE:
            raise ValueError(
                f"Unsupported third party agent runner type: {self.runner_type!r}. "
                f"Only {CODEX_RUNNER_TYPE!r} is supported; Dify, Coze, DashScope "
                "and DeerFlow runners have been removed."
            )
        self.runner_config = agent_runner["config"]
        settings = ctx.astrbot_config["provider_settings"]
        self.streaming_response: bool = settings["streaming_response"]
        self.unsupported_streaming_strategy: str = settings[
            "unsupported_streaming_strategy"
        ]
        self.stream_consumption_close_timeout_sec: int = coerce_int_config(
            settings.get(
                "third_party_stream_consumption_close_timeout_sec",
                STREAM_CONSUMPTION_CLOSE_TIMEOUT_SEC,
            ),
            default=STREAM_CONSUMPTION_CLOSE_TIMEOUT_SEC,
            min_value=1,
            field_name="third_party_stream_consumption_close_timeout_sec",
            source="Third-party runner config",
        )

    async def _resolve_persona_custom_error_message(
        self, event: AstrMessageEvent
    ) -> str | None:
        try:
            conversation_persona_id = await resolve_event_conversation_persona_id(
                event,
                self.ctx.plugin_manager.context.conversation_manager,
            )
            return await resolve_persona_custom_error_message(
                event=event,
                persona_manager=self.ctx.plugin_manager.context.persona_manager,
                provider_settings={"default_personality": "default"},
                conversation_persona_id=conversation_persona_id,
            )
        except Exception as e:
            logger.debug("Failed to resolve persona custom error message: %s", e)
            return None

    async def _handle_streaming_response(
        self,
        *,
        runner: "BaseAgentRunner",
        event: AstrMessageEvent,
        custom_error_message: str | None,
        close_runner_once: Callable[[], Awaitable[None]],
        mark_stream_consumed: Callable[[], bool],
    ) -> AsyncGenerator[None, None]:
        aggregator = _RunnerResultAggregator()

        async def _stream_runner_chain() -> AsyncGenerator[MessageChain, None]:
            if mark_stream_consumed():
                # The watchdog closed the runner first and gave the group
                # history back; the late run goes ahead without it, so the
                # next request is the one that shows it.
                drop_group_history(getattr(runner, "req", None))
            try:
                async for chain, is_error in run_third_party_agent(
                    runner,
                    stream_to_general=False,
                    custom_error_message=custom_error_message,
                    event=event,
                ):
                    aggregator.add_chunk(chain, is_error)
                    if is_error:
                        event.set_extra(THIRD_PARTY_RUNNER_ERROR_EXTRA_KEY, True)
                    yield chain
            finally:
                # Streaming runner cleanup must happen after consumer
                # finishes iterating to avoid tearing down active streams.
                await close_runner_once()

        event.set_result(
            MessageEventResult()
            .set_result_content_type(ResultContentType.STREAMING_RESULT)
            .set_async_stream(_stream_runner_chain()),
        )
        yield

        if runner.done():
            final_chain, is_runner_error = aggregator.finalize(
                runner.get_final_llm_resp()
            )
            event.set_extra(THIRD_PARTY_RUNNER_ERROR_EXTRA_KEY, is_runner_error)
            event.set_result(
                MessageEventResult.from_chain(
                    final_chain,
                    result_content_type=ResultContentType.STREAMING_FINISH,
                ),
            )

    async def _handle_non_streaming_response(
        self,
        *,
        runner: "BaseAgentRunner",
        event: AstrMessageEvent,
        stream_to_general: bool,
        custom_error_message: str | None,
    ) -> AsyncGenerator[None, None]:
        aggregator = _RunnerResultAggregator()
        async for chain, is_error in run_third_party_agent(
            runner,
            stream_to_general=stream_to_general,
            custom_error_message=custom_error_message,
            event=event,
        ):
            aggregator.add_chunk(chain, is_error)
            if is_error:
                event.set_extra(THIRD_PARTY_RUNNER_ERROR_EXTRA_KEY, True)
            yield

        final_chain, is_runner_error = aggregator.finalize(runner.get_final_llm_resp())
        event.set_extra(THIRD_PARTY_RUNNER_ERROR_EXTRA_KEY, is_runner_error)
        result_content_type = (
            ResultContentType.AGENT_RUNNER_ERROR
            if is_runner_error
            else ResultContentType.LLM_RESULT
        )
        event.set_result(
            MessageEventResult.from_chain(
                final_chain,
                result_content_type=result_content_type,
            ),
        )
        # Second yield keeps scheduler progress consistent after final result update.
        yield

    async def process(
        self, event: AstrMessageEvent, provider_wake_prefix: str
    ) -> AsyncGenerator[None, None]:
        req: ProviderRequest | None = None

        if provider_wake_prefix and not event.message_str.startswith(
            provider_wake_prefix
        ):
            return

        # make provider request
        req = ProviderRequest()
        req.session_id = event.unified_msg_origin
        req.prompt = event.message_str[len(provider_wake_prefix) :]
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                image_path = await comp.convert_to_base64()
                req.image_urls.append(image_path)
            elif isinstance(comp, Record):
                audio_path = await comp.convert_to_file_path()
                req.audio_urls.append(audio_path)

        if not req.prompt and not req.image_urls and not req.audio_urls:
            return
        if event.get_extra(AGENT_STOP_REQUESTED) is True:
            # /stop arrived while this message was still on its way here. No
            # later request may answer it from the group history either.
            await forget_group_message(event)
            return

        # The sender's permission group may cap their requests; a refused one
        # never reaches the agent, and no later request answers it either.
        refusal = await check_rate_limit(event, self.conf)
        if refusal is not None:
            await forget_group_message(event)
            if refusal:
                event.set_result(MessageEventResult().message(refusal))
                yield
            return

        try:
            await prepare_codex_request(
                event,
                req,
                self.ctx.plugin_manager.context,
                self.conf,
                self.runner_config,
            )
        except BaseException:
            await refund_rate_limit(event)
            raise

        custom_error_message = await self._resolve_persona_custom_error_message(event)
        set_persona_custom_error_message_on_event(event, custom_error_message)

        # call event hook
        if await call_event_hook(event, EventType.OnLLMRequestEvent, req):
            # Stopped before the model: give back the group history it took,
            # and the use it was counted as.
            await release_group_history(event)
            await refund_rate_limit(event)
            return

        if event.get_extra(AGENT_STOP_REQUESTED) is True:
            # Stopped while being prepared: not even into a turn that started
            # after the /stop.
            await release_group_history(event)
            await refund_rate_limit(event)
            await forget_group_message(event)
            return

        # Content moderation, before anything of the message goes to Codex:
        # a blocked message starts no turn and is steered into none.
        try:
            blocked = await content_moderation.check_request(
                req.prompt or "",
                req.extra_user_content_parts,
                [
                    *req.image_urls,
                    *(
                        event.get_extra(content_moderation.ATTACHMENT_IMAGES_EXTRA)
                        or []
                    ),
                ],
                content_moderation.platform_mode(event.get_platform_id()),
                label=event.unified_msg_origin,
            )
        except BaseException:
            await release_group_history(event)
            await refund_rate_limit(event)
            raise
        if blocked:
            # Its history goes back for the next request; the message itself
            # never becomes history another request would send.
            await release_group_history(event)
            await refund_rate_limit(event)
            await forget_group_message(event)
            event.set_result(
                MessageEventResult().message(content_moderation.blocked_reply())
            )
            yield
            return

        # Same sender while a Codex turn runs: steer into that turn (after the
        # request hooks, so moderation plugins still apply); the running turn
        # answers. Other senders queue behind it.
        try:
            target = await try_steer(
                event.unified_msg_origin,
                str(event.get_sender_id() or ""),
                build_turn_input(req),
                prompt=req.prompt or "",
                scopes=turn_scopes(event),
            )
        except BaseException:
            await release_group_history(event)
            await refund_rate_limit(event)
            raise
        if target is not None:
            keep_group_history(event)
            event.set_extra("_follow_up_captured", {"target_run_id": target})
            return

        runner = CodexAgentRunner[AstrAgentContext]()
        active_event_registry.register_agent_stop_callback(event, runner.request_stop)

        astr_agent_ctx = AstrAgentContext(
            context=self.ctx.plugin_manager.context,
            event=event,
        )

        streaming_response = self.streaming_response
        if (enable_streaming := event.get_extra("enable_streaming")) is not None:
            streaming_response = bool(enable_streaming)

        stream_to_general = (
            self.unsupported_streaming_strategy == "turn_off"
            and not event.platform_meta.support_streaming_message
        )
        streaming_used = streaming_response and not stream_to_general
        runner_streaming, suppress_streaming_deltas = (
            _resolve_third_party_streaming_mode(streaming_response, stream_to_general)
        )

        runner_closed = False
        stream_consumed = False
        stream_watchdog_task: asyncio.Task[None] | None = None

        def claim_close() -> Callable[[], Awaitable[None]] | None:
            """Takes the one close synchronously, with the group history's
            give-back; None if the close was already taken."""
            nonlocal runner_closed
            if runner_closed:
                return None
            runner_closed = True
            # A streamed run closed before it was ever consumed never started;
            # after a started run there is nothing left to give back.
            restore = take_group_history(event)
            # A stream closed before it was consumed may still be run by a
            # late consumer, so its use stays until that run's close. Decided
            # now: a consumer arriving during this close starts a live run.
            refund = not streaming_used or stream_consumed

            async def finish_close() -> None:
                await _close_runner_if_supported(runner)
                await give_back(restore)
                # A no-op once Codex accepted the turn (the runner keeps it).
                if refund:
                    await refund_rate_limit(event)

            return finish_close

        async def close_runner_once() -> None:
            if finish_close := claim_close():
                await finish_close()

        def mark_stream_consumed() -> bool:
            """Marks the stream consumed; True if the runner was closed first."""
            nonlocal stream_consumed, runner_closed
            stream_consumed = True
            if stream_watchdog_task and not stream_watchdog_task.done():
                stream_watchdog_task.cancel()
            closed_first = runner_closed
            # The late run is live again: closing it later must interrupt it.
            runner_closed = False
            return closed_first

        try:
            await runner.reset(
                request=req,
                run_context=AgentContextWrapper(
                    context=astr_agent_ctx,
                    tool_call_timeout=coerce_int_config(
                        self.runner_config.get("tool_call_timeout", 120),
                        default=120,
                        min_value=1,
                        field_name="tool_call_timeout",
                        source="Agent Runner config",
                    ),
                ),
                agent_hooks=MAIN_AGENT_HOOKS,
                provider_config=self.runner_config,
                streaming=runner_streaming,
            )
            if event.get_extra(AGENT_STOP_REQUESTED) is True:
                # /stop came while the request was being prepared, before this
                # runner could hear it (reset clears its own flag).
                runner.request_stop()

            if streaming_used:
                stream_watchdog_task = _start_stream_watchdog(
                    timeout_sec=self.stream_consumption_close_timeout_sec,
                    is_stream_consumed=lambda: stream_consumed,
                    claim_close=claim_close,
                )
                async for _ in self._handle_streaming_response(
                    runner=runner,
                    event=event,
                    custom_error_message=custom_error_message,
                    close_runner_once=close_runner_once,
                    mark_stream_consumed=mark_stream_consumed,
                ):
                    yield
            else:
                async for _ in self._handle_non_streaming_response(
                    runner=runner,
                    event=event,
                    stream_to_general=suppress_streaming_deltas,
                    custom_error_message=custom_error_message,
                ):
                    yield
        finally:
            if (
                stream_watchdog_task
                and not stream_watchdog_task.done()
                and (stream_consumed or runner_closed)
            ):
                stream_watchdog_task.cancel()
            if not streaming_used:
                await close_runner_once()
            elif stream_watchdog_task is None:
                # Failed before the stream was set up (runner.reset): the run
                # never starts, so its group history (and its use) goes back.
                await release_group_history(event)
                await refund_rate_limit(event)
            active_event_registry.unregister_agent_stop_callback(event)

        asyncio.create_task(
            Metric.upload(
                llm_tick=1,
                model_name=self.runner_type,
                provider_type=self.runner_type,
            ),
        )
