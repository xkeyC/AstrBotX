"""Verify AstrBot's prompt-cache structure across a multi-turn conversation.

A group conversation is driven through the real request path with a fake
model: build_main_agent, the builtin group_icl hook, plugin relocation, the
tool loop runner and the pipeline's history saver, turn after turn. The run
checks that:

* every request reuses the previous request's stored prefix byte for byte;
* each stored message keeps its sender and time;
* anchors (persona, tool rules, safety) are sent once, resent only when they
  change or after truncation dropped them, and revoked when withdrawn;
* group messages are stored exactly once, temporary context never.

Run from the repository root with:

    python scripts/verify_prompt_cache_structure.py
"""

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
from astrbot.core import astr_main_agent as ama
from astrbot.core.agent.context.persistent_context import (
    REVOKED_ANCHOR_HASH,
    anchor_hash,
    stored_anchor_hashes,
)
from astrbot.core.agent.context.token_counter import EstimateTokenCounter
from astrbot.core.agent.message import Message
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.db.po import Conversation
from astrbot.core.message.components import Plain
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse, TokenUsage
from astrbot.core.provider.provider import Provider
from astrbot.core.star.context import Context

UMO = "fake:GroupMessage:group-1"
PERSONA_TUTOR = {
    "name": "tutor",
    "prompt": "You are a patient tutor who answers in short steps.",
    "_begin_dialogs_processed": [
        {"role": "user", "content": "Teach me something."},
        {"role": "assistant", "content": "Sure, pick a topic."},
    ],
    "tools": None,
    "skills": None,
}
PERSONA_REVIEWER = {
    "name": "reviewer",
    "prompt": "You are a terse code reviewer.",
    "_begin_dialogs_processed": [],
    "tools": None,
    "skills": None,
}
CONFIG = {
    "timezone": "UTC",
    "provider_settings": {
        "prompt_prefix": "",
        "identifier": False,
        "group_name_display": True,
        "datetime_system_prompt": True,
        "image_caption_prompt": "Describe the image.",
        "computer_use_runtime": "none",
    },
    "provider_ltm_settings": {
        "group_icl_enable": True,
        "group_message_max_cnt": 100,
        "image_caption": False,
        "image_caption_provider_id": "",
        "group_message_history_enable": False,
        "active_reply": {
            "enable": False,
            "method": "possibility_reply",
            "possibility_reply": 0.0,
            "prompt": "",
            "whitelist": [],
        },
    },
}
PLUGIN_NOTE = "Plugin note: the canteen closes early today."


class FakeModel(Provider):
    """Fake chat model that records every request it receives."""

    def __init__(self) -> None:
        super().__init__(
            {
                "id": "fake-model",
                "model": "fake-model",
                "modalities": ["text", "tool_use"],
                "max_context_tokens": 0,
            },
            {},
        )
        self.set_model("fake-model")
        self.requests: list[list[Message]] = []

    def get_current_key(self) -> str:
        return "fake"

    def set_key(self, key: str) -> None:
        del key

    async def get_models(self) -> list[str]:
        return ["fake-model"]

    async def text_chat(self, **kwargs: Any) -> LLMResponse:
        self.requests.append(
            [
                Message.model_validate(message.model_dump())
                if isinstance(message, Message)
                else Message.model_validate(message)
                for message in kwargs.get("contexts") or []
            ]
        )
        return LLMResponse(
            role="assistant",
            completion_text=f"Reply {len(self.requests)}.",
            usage=TokenUsage(input_other=1, output=1),
        )


class InMemoryConversations:
    """Conversation manager keeping one group conversation in memory."""

    def __init__(self) -> None:
        self.conversation = Conversation(
            platform_id="fake",
            user_id=UMO,
            cid="conv-1",
            history="[]",
        )

    async def get_curr_conversation_id(self, umo: str) -> str:
        del umo
        return self.conversation.cid

    async def new_conversation(self, umo: str, *args: Any, **kwargs: Any) -> str:
        del umo, args, kwargs
        return self.conversation.cid

    async def get_conversation(self, umo: str, cid: str, **kwargs: Any):
        del umo, cid, kwargs
        return self.conversation

    async def update_conversation(
        self,
        umo: str,
        cid: str,
        history: list[dict] | None = None,
        token_usage: int | None = None,
        **kwargs: Any,
    ) -> None:
        del umo, cid, kwargs
        if history is not None:
            self.conversation.history = json.dumps(history, ensure_ascii=False)
        if token_usage is not None:
            self.conversation.token_usage = token_usage

    def history(self) -> list[dict]:
        return json.loads(self.conversation.history)


class FakePersonas:
    """Persona manager whose selected persona the scenario can switch."""

    personas_v3: list = []

    def __init__(self) -> None:
        self.persona = PERSONA_TUTOR

    async def resolve_selected_persona(self, **kwargs: Any):
        del kwargs
        return self.persona["name"], self.persona, None, False

    def get_persona_v3_by_id(self, persona_id: str):
        del persona_id
        return None


def make_event(
    user_id: str, nickname: str, text: str, trigger: bool = True
) -> AstrMessageEvent:
    """Build a group message event with working extras."""
    platform_meta = PlatformMetadata(id="fake", name="fake", description="fake")
    platform_meta.support_proactive_message = False
    message_obj = SimpleNamespace(
        message=[Plain(text=text)],
        message_str=text,
        sender=SimpleNamespace(user_id=user_id, nickname=nickname),
        group_id="group-1",
        group=SimpleNamespace(group_name="Study Group"),
        self_id="bot",
        type=MessageType.GROUP_MESSAGE,
    )
    extras: dict[str, Any] = {}
    event = MagicMock(spec=AstrMessageEvent)
    event.message_str = text
    event.message_obj = message_obj
    event.platform_meta = platform_meta
    event.unified_msg_origin = UMO
    event.session_id = "group-1"
    event.role = "member"
    event.plugins_name = None
    # Chatter does not wake the bot; only the message being answered does.
    event.is_at_or_wake_command = trigger
    event.trace = MagicMock()
    event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)
    event.set_extra.side_effect = extras.__setitem__
    event.get_platform_name.return_value = "fake"
    event.get_platform_id.return_value = "fake"
    event.get_message_type.return_value = MessageType.GROUP_MESSAGE
    event.get_group_id.return_value = "group-1"
    event.get_sender_name.return_value = nickname
    event.get_sender_id.return_value = user_id
    event.get_self_id.return_value = "bot"
    event.get_messages.return_value = message_obj.message
    event.is_stopped.return_value = False
    return event


def make_plugin_context(conversations, personas) -> MagicMock:
    tool_manager = MagicMock()
    tool_manager.get_full_tool_set.side_effect = lambda: ToolSet(
        [
            FunctionTool(
                name="lookup_fact",
                description="Look up a fact",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        ]
    )
    tool_manager.get_builtin_tool.side_effect = lambda cls, **kwargs: cls(**kwargs)
    tool_manager.get_func.return_value = None
    context = MagicMock(spec=Context)
    context.get_config.return_value = CONFIG
    context.conversation_manager = conversations
    context.persona_manager = personas
    context.get_llm_tool_manager.return_value = tool_manager
    context.get_provider_by_id.return_value = None
    context._db = None
    return context


@dataclass
class Turn:
    """One user message and what the scenario expects it to send."""

    sender: tuple[str, str, str]
    chatter: list[tuple[str, str, str]] = field(default_factory=list)
    persona: dict | None = None
    plugin_note: str | None = None
    max_context_length: int = -1


SCENARIO = [
    Turn(sender=("u-alice", "Alice", "Hi, I am Alice. Can you help us study?")),
    Turn(
        chatter=[
            ("u-bob", "Bob", "Anyone here?"),
            ("u-carol", "Carol", "Lunch at noon."),
        ],
        sender=("u-bob", "Bob", "What did Carol just say?"),
        plugin_note=PLUGIN_NOTE,
    ),
    Turn(
        sender=("u-alice", "Alice", "Switch to reviewing my code."),
        persona=PERSONA_REVIEWER,
    ),
    Turn(sender=("u-carol", "Carol", "Is a for loop fine here?")),
    Turn(
        sender=("u-bob", "Bob", "Summarize the last answer."),
        max_context_length=1,
    ),
    Turn(sender=("u-carol", "Carol", "Thanks, that helps.")),
]


def _dump(message: Message) -> str:
    return json.dumps(message.model_dump(), ensure_ascii=False, sort_keys=True)


def _common_prefix(previous: list[Message], current: list[Message]) -> int:
    count = 0
    for before, after in zip(previous, current):
        if _dump(before) != _dump(after):
            break
        count += 1
    return count


def _anchor_changes(message: Message | None) -> list[str]:
    if message is None:
        return []
    return [
        f"{name}={'revoked' if version == REVOKED_ANCHOR_HASH else 'set'}"
        for name, version in stored_anchor_hashes([message]).items()
    ]


def _unit_names(message: Message | None) -> list[str]:
    if message is None or not isinstance(message.content, list):
        return []
    return [
        part.text.split('"')[1]
        for part in message.content
        if getattr(part, "text", "").startswith("<context_unit")
    ]


async def run_scenario() -> list[dict[str, Any]]:
    """Run every turn and collect what each request sent and reused."""
    conversations = InMemoryConversations()
    personas = FakePersonas()
    context = make_plugin_context(conversations, personas)
    group_context = GroupChatContext(MagicMock(), context)
    model = FakeModel()
    stage = InternalAgentSubStage()
    stage.conv_manager = conversations
    counter = EstimateTokenCounter()
    reports: list[dict[str, Any]] = []

    for number, turn in enumerate(SCENARIO, start=1):
        if turn.persona is not None:
            personas.persona = turn.persona
        for user_id, nickname, text in turn.chatter:
            await group_context.handle_message(
                make_event(user_id, nickname, text, trigger=False)
            )
        user_id, nickname, text = turn.sender
        event = make_event(user_id, nickname, text)
        await group_context.handle_message(event)

        config = ama.MainAgentBuildConfig(
            tool_call_timeout=60,
            streaming_response=False,
            provider_settings=CONFIG["provider_settings"],
            computer_use_runtime="none",
            add_cron_tools=False,
            llm_safety_mode=True,
            max_context_length=turn.max_context_length,
            dequeue_context_length=1,
        )
        result = await ama.build_main_agent(
            event=event,
            plugin_context=context,
            config=config,
            provider=model,
            apply_reset=False,
        )
        assert result is not None, f"turn {number}: no agent was built"
        request = result.provider_request
        # Same order as the pipeline: OnLLMRequest hooks, then relocation.
        baseline = ama.snapshot_plugin_context_baseline(request)
        await group_context.on_req_llm(event, request)
        if turn.plugin_note:
            request.system_prompt += turn.plugin_note
        ama.relocate_plugin_injected_context(request, baseline)
        await result.reset_coro

        runner = result.agent_runner
        async for _ in runner.step_until_done(3):
            pass
        await stage._save_to_history(
            event,
            request,
            runner.get_final_llm_resp(),
            runner.run_context.messages,
            runner.stats,
        )

        payload = model.requests[-1]
        persisted = runner._persistent_context_message
        tail = payload[payload.index(persisted) + 1 :] if persisted in payload else []
        reports.append(
            {
                "turn": number,
                "sender": f"{nickname} (ID: {user_id})",
                "payload": payload,
                "anchors": dict(request.context_anchors),
                "anchor_changes": _anchor_changes(persisted),
                "units": _unit_names(persisted),
                "tail_messages": len(tail),
                "temporary": any(
                    getattr(part, "text", "").startswith("<request_context")
                    for message in tail
                    if isinstance(message.content, list)
                    for part in message.content
                ),
                "tokens": counter.count_tokens(payload),
                "history": conversations.history(),
                "truncated": turn.max_context_length > 0,
            }
        )

    previous: dict[str, Any] | None = None
    for report in reports:
        payload = report["payload"]
        if previous is None:
            report["reused_messages"] = 0
            report["reused_tokens"] = 0
        else:
            reused = _common_prefix(previous["payload"], payload)
            report["reused_messages"] = reused
            report["reused_tokens"] = counter.count_tokens(payload[:reused])
            # Temporary context is never stored, so the previous request is
            # reused up to its temporary message, or entirely without one.
            report["expected_reuse"] = len(previous["payload"]) - (
                previous["tail_messages"] if previous["temporary"] else 0
            )
        report["reuse_ratio"] = round(report["reused_tokens"] / report["tokens"], 3)
        previous = report
    return reports


def check(reports: list[dict[str, Any]]) -> None:
    """Assert the multi-turn cache and context invariants."""
    by_turn = {report["turn"]: report for report in reports}
    final_history = json.dumps(reports[-1]["history"], ensure_ascii=False)

    for report in reports:
        turn = report["turn"]
        # The model always sees exactly the declared anchors, newest version.
        effective = stored_anchor_hashes(report["payload"])
        for name, content in report["anchors"].items():
            assert effective.get(name) == anchor_hash(content), (turn, name)
        for name, version in effective.items():
            if name not in report["anchors"]:
                assert version == REVOKED_ANCHOR_HASH, (turn, name)
        # Every stored turn keeps its sender; temporary context is never stored.
        history_text = json.dumps(report["history"], ensure_ascii=False)
        assert f"Sender: {report['sender']}" in history_text, turn
        # Group senders are named with their id even with identifier off:
        # nicknames repeat, and the id keeps two people apart.
        assert "<request_context" not in history_text, turn
        # Apart from the truncating turn, each request extends the previous one.
        if turn > 1 and not report["truncated"]:
            assert report["reused_messages"] == report["expected_reuse"], (
                turn,
                report["reused_messages"],
                report["expected_reuse"],
            )

    assert {"persona", "persona_examples", "safety_mode", "tool_runtime"} <= {
        change.split("=")[0] for change in by_turn[1]["anchor_changes"]
    }, by_turn[1]["anchor_changes"]
    assert by_turn[2]["anchor_changes"] == [], by_turn[2]["anchor_changes"]
    assert sorted(by_turn[3]["anchor_changes"]) == [
        "persona=set",
        "persona_examples=revoked",
    ], by_turn[3]["anchor_changes"]
    assert by_turn[4]["anchor_changes"] == [], by_turn[4]["anchor_changes"]
    assert "persona=set" in by_turn[5]["anchor_changes"], by_turn[5]["anchor_changes"]

    assert by_turn[2]["units"] == ["group_history", "message_meta"], by_turn[2]
    # Group messages are stored exactly once; turn 5's truncation drops them.
    stored_before_truncation = json.dumps(by_turn[4]["history"], ensure_ascii=False)
    for line in ("Anyone here?", "Lunch at noon."):
        assert stored_before_truncation.count(line) == 1, line
    # Anchors resent after the truncation were stored, so turn 6 sends none.
    assert by_turn[6]["anchor_changes"] == [], by_turn[6]["anchor_changes"]
    payload_text = json.dumps(
        [message.model_dump() for message in by_turn[2]["payload"]],
        ensure_ascii=False,
    )
    assert PLUGIN_NOTE in payload_text
    assert PLUGIN_NOTE not in final_history


def _check_tool_order() -> None:
    tools = [
        FunctionTool(name=name, description=name, parameters={"type": "object"})
        for name in ("web_search", "read_file")
    ]
    if ToolSet(tools).openai_schema() != ToolSet(tools[::-1]).openai_schema():
        raise AssertionError("Tool registration order changed the tool schema.")


async def verify_structure() -> list[dict[str, Any]]:
    """Run the scenario under isolated side effects and check it."""
    _check_tool_order()
    with (
        patch.object(ama.SkillManager, "list_skills", return_value=[]),
        patch.object(ama, "retrieve_knowledge_base", AsyncMock(return_value=None)),
    ):
        reports = await run_scenario()
    check(reports)
    return reports


async def main() -> None:
    reports = await verify_structure()
    print("turn sender  tokens  reused  ratio  anchors sent / units stored")
    for report in reports:
        note = " (truncated)" if report["truncated"] else ""
        print(
            f"{report['turn']:>4} {report['sender']:<6} {report['tokens']:>6}  "
            f"{report['reused_tokens']:>6}  {report['reuse_ratio']:>5.0%}  "
            f"{', '.join(report['anchor_changes']) or '-'} / "
            f"{', '.join(report['units']) or '-'}{note}"
        )
    print("PASS: multi-turn prompt-cache and context structure holds.")


if __name__ == "__main__":
    asyncio.run(main())
