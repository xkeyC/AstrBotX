from sqlalchemy import case, func, select
from sqlmodel import col

from astrbot.api import sp, star
from astrbot.api.event import AstrMessageEvent, MessageEventResult
from astrbot.core.agent.runners.codex.constants import (
    CODEX_RUNNER_TYPE,
    CODEX_THREAD_STATE_KEY,
    CODEX_VOICE_STATS_TYPE,
)
from astrbot.core.db.po import ProviderStat
from astrbot.core.utils.active_event_registry import active_event_registry

THIRD_PARTY_AGENT_RUNNER_KEY = {
    CODEX_RUNNER_TYPE: CODEX_THREAD_STATE_KEY,
}
THIRD_PARTY_AGENT_RUNNER_STR = ", ".join(THIRD_PARTY_AGENT_RUNNER_KEY.keys())


async def _clear_third_party_agent_runner_state(
    context: star.Context,
    umo: str,
    agent_runner_type: str,
) -> None:
    session_key = THIRD_PARTY_AGENT_RUNNER_KEY.get(agent_runner_type)
    if not session_key:
        return

    if agent_runner_type == CODEX_RUNNER_TYPE:
        from astrbot.core.agent.runners.codex.native import CodexEngine

        state = await sp.get_async(
            scope="umo", scope_id=umo, key=session_key, default={}
        )
        if isinstance(state, dict) and (thread_id := state.get("thread_id")):
            await CodexEngine.release_thread(umo, str(thread_id))
    await sp.remove_async(
        scope="umo",
        scope_id=umo,
        key=session_key,
    )


class ConversationCommands:
    def __init__(self, context: star.Context) -> None:
        self.context = context

    async def _get_current_persona_id(self, session_id):
        curr = await self.context.conversation_manager.get_curr_conversation_id(
            session_id,
        )
        if not curr:
            return None
        conv = await self.context.conversation_manager.get_conversation(
            session_id,
            curr,
        )
        if not conv:
            return None
        return conv.persona_id

    async def reset(self, message: AstrMessageEvent) -> None:
        """Clear the context of the current conversation.

        Args:
            message: Command event identifying the session and sender.
        """
        umo = message.unified_msg_origin
        cfg = self.context.get_config(umo=umo)
        agent_runner_type = cfg["agent_runner"]["runner_type"]

        active_event_registry.stop_all(umo, exclude=message)
        cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
        if agent_runner_type in THIRD_PARTY_AGENT_RUNNER_KEY:
            await _clear_third_party_agent_runner_state(
                self.context,
                umo,
                agent_runner_type,
            )
        else:
            if cid:
                await self.context.conversation_manager.update_conversation(
                    umo,
                    cid,
                    history=[],
                )

        message.set_extra("_clean_group_context_session", True)
        message.set_result(
            MessageEventResult().message(
                "✅ The current conversation context has been cleared."
            )
        )

    async def stop(self, message: AstrMessageEvent) -> None:
        """停止当前会话正在运行的 Agent"""
        umo = message.unified_msg_origin
        # Asks each running agent to stop, as the WebUI's stop button does: a
        # Codex turn is interrupted (and says so), and one still queued for the
        # chat ends before it reaches Codex, giving back what it took.
        stopped_count = active_event_registry.request_agent_stop_all(
            umo,
            exclude=message,
        )

        if stopped_count > 0:
            message.set_result(
                MessageEventResult().message(
                    f"✅ Requested to stop {stopped_count} running tasks."
                )
            )
            return

        message.set_result(
            MessageEventResult().message("✅ No running tasks in the current session.")
        )

    async def new_conv(self, message: AstrMessageEvent) -> None:
        """Start a new conversation without clearing the previous history.

        Args:
            message: Command event identifying the session and sender.
        """
        cfg = self.context.get_config(umo=message.unified_msg_origin)
        agent_runner_type = cfg["agent_runner"]["runner_type"]
        active_event_registry.stop_all(message.unified_msg_origin, exclude=message)
        if agent_runner_type in THIRD_PARTY_AGENT_RUNNER_KEY:
            await _clear_third_party_agent_runner_state(
                self.context,
                message.unified_msg_origin,
                agent_runner_type,
            )

        cpersona = await self._get_current_persona_id(message.unified_msg_origin)
        cid = await self.context.conversation_manager.new_conversation(
            message.unified_msg_origin,
            message.get_platform_id(),
            persona_id=cpersona,
        )

        message.set_extra("_clean_group_context_session", True)

        message.set_result(
            MessageEventResult().message(
                f"✅ Switched to new conversation: {cid[:4]}."
            ),
        )

    async def stats(self, message: AstrMessageEvent) -> None:
        """Show token usage statistics for the current conversation."""
        umo = message.unified_msg_origin
        cid = await self.context.conversation_manager.get_curr_conversation_id(umo)

        if not cid:
            message.set_result(
                MessageEventResult().message(
                    "❌ You are not in a conversation. Use /new to create one."
                ),
            )
            return

        db = self.context.get_db()
        async with db.get_db() as session:
            result = await session.execute(
                select(
                    func.count(case((col(ProviderStat.id).is_not(None), 1))).label(
                        "record_count",
                    ),
                    func.coalesce(func.sum(ProviderStat.token_input_other), 0).label(
                        "total_input_other",
                    ),
                    func.coalesce(func.sum(ProviderStat.token_input_cached), 0).label(
                        "total_input_cached",
                    ),
                    func.coalesce(func.sum(ProviderStat.token_output), 0).label(
                        "total_output",
                    ),
                ).where(
                    col(ProviderStat.agent_type).in_(
                        (CODEX_RUNNER_TYPE, CODEX_VOICE_STATS_TYPE)
                    ),
                    col(ProviderStat.conversation_id) == cid,
                )
            )
            stats = result.one()

        if stats.record_count == 0:
            message.set_result(
                MessageEventResult().message(
                    "📊 No stats available for this conversation yet."
                ),
            )
            return

        total_input_other = stats.total_input_other
        total_input_cached = stats.total_input_cached
        total_output = stats.total_output
        total_tokens = total_input_other + total_input_cached + total_output

        ret = (
            f"📊 Conversation Token usage (ID: {cid[:8]}...)\n"
            f"Total:          {total_tokens:,}\n"
            f"Input (cached): {total_input_cached:,}\n"
            f"Input (other):  {total_input_other:,}\n"
            f"Output:         {total_output:,}\n"
        )

        message.set_result(MessageEventResult().message(ret))
