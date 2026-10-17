from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
    qq_emoji_id,
)


def _event(message_type: MessageType) -> AiocqhttpMessageEvent:
    event = AiocqhttpMessageEvent.__new__(AiocqhttpMessageEvent)
    event.message_obj = SimpleNamespace(message_id="12345", type=message_type)
    event.bot = SimpleNamespace(call_action=AsyncMock(return_value={"result": True}))
    return event


def test_emoji_ids():
    # QQ face ids as they are; Unicode emoji as their code point.
    assert qq_emoji_id("76") == "76"
    assert qq_emoji_id(" 124 ") == "124"
    assert qq_emoji_id("👀") == "128064"
    # A variation selector is not part of the emoji.
    assert qq_emoji_id("❤️") == "10084"
    # NapCat would read these as face ids, or cannot take a sequence.
    for bad in (" ", "#️⃣", "1️⃣", "©", "🇨🇳", "👍🏻", "👨‍👩‍👧", "１２"):
        with pytest.raises(ValueError):
            qq_emoji_id(bad)


@pytest.mark.asyncio
async def test_a_group_message_gets_a_napcat_reaction():
    event = _event(MessageType.GROUP_MESSAGE)
    await event.react("🤔")
    event.bot.call_action.assert_awaited_once_with(
        "set_msg_emoji_like", message_id="12345", emoji_id="129300", set=True
    )


@pytest.mark.asyncio
async def test_a_private_message_gets_none():
    # QQ shows reactions in groups only; never a fallback text message.
    event = _event(MessageType.FRIEND_MESSAGE)
    event.send = AsyncMock()
    await event.react("76")
    event.bot.call_action.assert_not_awaited()
    event.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_preprocess_stage_reacts_on_aiocqhttp():
    from astrbot.core.pipeline.preprocess_stage.stage import PreProcessStage

    stage = PreProcessStage.__new__(PreProcessStage)
    stage.config = {
        "platform_specific": {
            "aiocqhttp": {"pre_ack_emoji": {"enable": True, "emojis": ["76"]}}
        }
    }
    stage.platform_settings = {}
    stage.stt_settings = {"enable": False}
    event = SimpleNamespace(
        get_platform_name=lambda: "aiocqhttp",
        is_at_or_wake_command=True,
        react=AsyncMock(),
        get_messages=lambda: [],
    )
    result = stage.process(event)
    if hasattr(result, "__anext__"):
        async for _ in result:
            break
    else:
        await result
    event.react.assert_awaited_once_with("76")
