"""The chat a voice conversation shares its agent with.

A voice conversation has no agent of its own: what the voice model hands off
runs as a turn of the paired chat's Codex thread, through the chat runner,
like a message of that chat. So voice and text share one context, persona,
tool set, memory and approvals, and queue behind each other on the chat's
session lock. The answer goes back to the voice model, which tells it in its
own words; nothing is posted, unless the conversation has ended by then (the
answer is then posted as text, not lost).

Results reaching the chat later (background work, scheduled tasks) are also
given to the chat's open voice conversation, if any (``announce``).

Identity: in a private conversation (a call, a whisper) the turn runs as the
person talking, with their own permissions. In a group (a voice channel) the
speakers cannot be told apart for sure (a platform may guess who spoke: the
request names the guess), so it runs as one fixed voice user, locked to the
member role.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from astrbot import logger
from astrbot.core.message.message_event_result import MessageChain

if TYPE_CHECKING:
    from .session import VoiceSession

# The sender of group voice turns (never an admin, whatever is configured).
VOICE_SENDER_ID = "voice"
VOICE_SENDER_NAME = "Voice"
# Marks a voice turn's event: its turn scopes differ from the same person's
# text turns, so their text is queued rather than joined into a voice turn
# (whose answer is only spoken), and it keeps a speaker label.
VOICE_TURN_EXTRA = "voice_turn"

REQUEST_PROMPT = """<voice_request via="{via}" speaker="{speaker}">
{body}
</voice_request>
<system>This task comes from a live voice conversation: the speaker is waiting on the line. Finish it as fast as you can. If it needs long or asynchronous work (a background command, a scheduled task, anything you cannot finish right away), start that work and reply at once with its status (what was started, roughly how long it takes, how the result will reach them) instead of waiting for it. Your final reply is given to the voice model, which tells it to the speaker; it is not posted to the chat: answer in plain spoken language, briefly, without Markdown, lists, links or code, in the speaker's language. A voice task's result goes back to the voice model by default; to deliver text content, send it to the user with send_message_to_user.</system>"""

TASK_BODY = """What was said: {heard}
Task: {task}"""


# Voice conversations open now, by the UMO of their paired chat.
VOICE_SESSIONS: dict[str, VoiceSession] = {}
# Requests outliving the voice session that made them.
_REQUESTS: set[asyncio.Task] = set()


def _attribute(text: str) -> str:
    """``text`` made safe inside a double-quoted tag attribute."""
    return text.replace('"', "'").replace("<", "(").replace(">", ")")


async def announce(umo: str, text: str) -> bool:
    """Gives ``text`` (a result that reached the chat) to the chat's open
    voice conversation, whose model decides whether and how to tell it.

    Returns:
        Whether a voice conversation took it.
    """
    session = VOICE_SESSIONS.get(umo)
    if session is None or session.closing or not session.ready or not text:
        return False
    await session.note(text)
    return True


@dataclass
class VoiceChat:
    """The paired chat of a voice conversation, and who talks in it.

    Attributes:
        umo: Unified message origin of the chat.
        private: One person with the bot (a call, a whisper): turns run as
            ``sender_id`` with their permissions. Otherwise (a channel) they
            run as the voice user, a member.
        sender_id: The person talking, for a private chat.
        sender_name: Their display name.
        via: What the conversation is, for the agent, e.g. ``QQ voice call``.
    """

    umo: str
    private: bool
    sender_id: str = VOICE_SENDER_ID
    sender_name: str = VOICE_SENDER_NAME
    via: str = "voice"
    # Requests of this conversation run one after the other, in order.
    _order: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False, compare=False
    )
    # Requests made and not answered yet (counted from the moment they are
    # made, before they reach the lock).
    _open: int = field(default=0, init=False, repr=False, compare=False)

    def busy(self) -> bool:
        """Whether a request would wait: a turn runs or waits in the chat, or
        an earlier request of this conversation is not done."""
        from astrbot.core.agent.runners.codex.native import _QUEUED_TURNS

        return _QUEUED_TURNS.get(self.umo, 0) > 0 or self._open > 0

    def request(
        self,
        body: str,
        on_answer: Callable[[str | None], Awaitable[Any]],
        speaker: str | None = None,
    ) -> bool:
        """Runs ``body`` as a turn of the chat, after this conversation's
        earlier requests, and hands the answer to ``on_answer``. The request
        outlives the voice session (a hang-up does not cancel it).
        ``speaker``: who the platform guesses asked (see ``ask``).

        Returns:
            Whether it has to wait (see ``busy``), decided before it queues.
        """
        busy = self.busy()
        self._open += 1

        async def run() -> None:
            answer = None
            try:
                async with self._order:
                    answer = await self.ask(body, speaker)
            except Exception as exc:  # noqa: BLE001 - answered as failed
                logger.warning("Voice: request to %s failed: %s", self.umo, exc)
            finally:
                self._open -= 1
            await on_answer(answer)

        task = asyncio.create_task(run(), name=f"voice-request-{self.umo}")
        _REQUESTS.add(task)
        task.add_done_callback(_REQUESTS.discard)
        return busy

    def _event(self, ctx, message: str):
        """A synthetic event of the chat, from the person talking."""
        from astrbot.core.cron.events import CronMessageEvent
        from astrbot.core.platform.message_session import MessageSession

        session = MessageSession.from_str(self.umo)
        sender_id = self.sender_id if self.private else VOICE_SENDER_ID
        sender_name = self.sender_name if self.private else VOICE_SENDER_NAME
        # The sender is set on the message, not passed in: the constructor
        # would make them the bot (self_id) too.
        event = CronMessageEvent(
            context=ctx,
            session=session,
            message=message,
            message_type=session.message_type,
        )
        event.message_obj.sender.user_id = sender_id
        event.message_obj.sender.nickname = sender_name
        if not self.private:
            event.message_obj.group_id = session.session_id
        cfg = ctx.get_config(umo=self.umo)
        admins = {str(a) for a in cfg.get("admins_id", [])}
        event.role = "admin" if self.private and sender_id in admins else "member"
        event.set_extra(VOICE_TURN_EXTRA, True)
        return event

    def memory_config(self) -> dict:
        """Codex memory settings for a voice thread that runs its own turns:
        those of the chat's thread (its config profile, its store, whether
        its consolidation may reach global memory), so the voice model reads
        and writes the same memories as the chat. Voice turns carry no
        permission scopes, so what scopes grant a text turn (writing global
        memory, deleting) is refused to them: they fail closed. The voice
        thread runs on the global runner's engine, so its memories live in
        that engine's ``codex_home``: a profile with a ``codex_home`` of its
        own keeps its chat's memories apart from its voice's.

        Returns:
            Codex config overrides; empty when the runner keeps no memories
            or they cannot be worked out.
        """
        try:
            return self._memory_config()
        except Exception as exc:  # noqa: BLE001 - no memories rather than no voice
            logger.warning("Voice: memories of %s left out: %s", self.umo, exc)
            return {}

    def _memory_config(self) -> dict:
        from astrbot.core.agent.runners.codex.codex_agent_runner import (
            memory_thread_config,
        )
        from astrbot.core.config.agent_runner import normalize_agent_runner
        from astrbot.core.permission_rules import (
            CONFIG_KEY,
            EVENT_EXTRA_KEY,
            policy_for_event,
        )
        from astrbot.core.star.context import current_context

        ctx = current_context()
        if ctx is None:
            return {}
        # The chat's config profile, as its turns read it.
        chat_cfg = ctx.get_config(umo=self.umo)
        cfg = normalize_agent_runner(chat_cfg.get("agent_runner"))["config"]
        if not cfg.get("memory_enabled"):
            return {}
        # The speaker's rules, as the chat's turns read them.
        event = self._event(ctx, "")
        event.set_extra(
            EVENT_EXTRA_KEY,
            policy_for_event(event, chat_cfg.get(CONFIG_KEY) or []),
        )
        return memory_thread_config(cfg, self.umo, event)

    async def voice_persona(self) -> str:
        """The voice persona of the chat's active persona: short instructions
        for the voice model, or empty when there are none.

        The persona is resolved as for the chat's turns: a persona the
        permission rules pick for the speaker, else one forced on the session,
        else the conversation's, else the configured default.
        """
        from astrbot.core.event_llm_overrides import get_event_selected_persona_id
        from astrbot.core.permission_rules import CONFIG_KEY, policy_for_event
        from astrbot.core.star.context import current_context

        ctx = current_context()
        if ctx is None:
            return ""
        try:
            event = self._event(ctx, "")
            # The rules as the chat's turns read them (codex_request).
            policy = policy_for_event(
                event, ctx.get_config(umo=self.umo).get(CONFIG_KEY) or []
            )
            if policy.persona_id:
                event.set_selected_persona(policy.persona_id)
            conversation_id = await ctx.conversation_manager.get_curr_conversation_id(
                self.umo
            )
            conversation = (
                await ctx.conversation_manager.get_conversation(
                    self.umo, conversation_id
                )
                if conversation_id
                else None
            )
            _, persona, _, _ = await ctx.persona_manager.resolve_selected_persona(
                umo=self.umo,
                conversation_persona_id=conversation.persona_id
                if conversation
                else None,
                platform_name=self.umo.split(":", 1)[0],
                provider_settings=ctx.get_config(umo=self.umo),
                selected_persona_id=get_event_selected_persona_id(event),
            )
        except Exception as exc:  # noqa: BLE001 - the platform default applies
            logger.warning("Voice: persona of %s not resolved: %s", self.umo, exc)
            return ""
        return str((persona or {}).get("voice_prompt") or "").strip()

    async def ask(self, body: str, speaker: str | None = None) -> str | None:
        """Runs ``body`` as a turn of the chat and returns the answer.

        Args:
            body: The request (see TASK_BODY).
            speaker: In a group, who the platform guesses said it (from
                where the voice came from, say): named in the request for
                the agent, nothing more.

        Returns:
            The answer (empty when the turn ended without one), or None when
            the turn was stopped, refused (a full queue), failed or the core
            is not up.
        """
        from astrbot.core.agent.runners.codex.codex_agent_runner import BUSY_NOTE
        from astrbot.core.agent.runners.codex.wake import run_turn_in_session
        from astrbot.core.star.context import current_context

        ctx = current_context()
        if ctx is None:
            logger.warning("Voice: no core context, a request is not answered")
            return None
        sender_name = self.sender_name if self.private else VOICE_SENDER_NAME
        if speaker and not self.private:
            # Only a guess, which can be wrong: the turn still runs as the
            # voice user, a member. A guessed name grants nothing.
            sender_name = f"{sender_name}, maybe {speaker} (a guess)"
        text = body
        while "</voice_request>" in text:
            text = text.replace("</voice_request>", "")
        prompt = REQUEST_PROMPT.format(
            via=_attribute(self.via), speaker=_attribute(sender_name), body=text
        )
        try:
            # The event carries what was asked (a task created now quotes
            # it); the turn gets it wrapped.
            event = self._event(ctx, body)
            answer = await run_turn_in_session(
                ctx, event, ctx.get_config(umo=self.umo), prompt
            )
        except Exception as exc:  # noqa: BLE001 - reported; the voice goes on
            logger.warning("Voice: request to %s failed: %s", self.umo, exc)
            return None
        return None if answer == BUSY_NOTE else answer


async def deliver(umo: str, answer: str) -> None:
    """Gets a voice request's answer to its chat after its voice conversation
    ended: to another open voice conversation of the chat, else as text."""
    if not answer or await announce(umo, answer):
        return
    from astrbot.core.star.context import current_context

    if (ctx := current_context()) is not None:
        await ctx.send_message(umo, MessageChain().message(answer))
