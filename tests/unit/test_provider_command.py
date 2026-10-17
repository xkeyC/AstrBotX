"""/provider lists and switches TTS / STT only; the chat model is Codex's."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.builtin_stars.builtin_commands.commands.provider import ProviderCommands
from astrbot.core.provider.entities import ProviderType


def _provider(id_: str):
    return SimpleNamespace(meta=lambda: SimpleNamespace(id=id_))


def _commands():
    tts = [_provider("tts-a"), _provider("tts-b")]
    context = MagicMock()
    context.get_all_providers.return_value = [_provider("chat-1")]
    context.get_all_tts_providers.return_value = tts
    context.get_all_stt_providers.return_value = []
    context.get_using_tts_provider_async = AsyncMock(return_value=tts[1])
    context.get_using_stt_provider_async = AsyncMock(return_value=None)
    context.provider_manager.set_provider = AsyncMock()
    return ProviderCommands(context), context


def _event():
    event = MagicMock()
    event.unified_msg_origin = "aiocqhttp:FriendMessage:1"
    return event


def _text(event) -> str:
    return event.set_result.call_args.args[0].get_plain_text()


@pytest.mark.asyncio
async def test_the_list_has_no_chat_providers():
    commands, context = _commands()
    event = _event()

    await commands.provider(event)

    text = _text(event)
    assert "chat-1" not in text and "LLM" not in text
    assert "1. tts-a\n" in text and "2. tts-b 👈" in text
    assert "/provider tts <idx>" in text and "STT" not in text
    context.get_all_providers.assert_not_called()


@pytest.mark.asyncio
async def test_switching_a_chat_provider_by_index_is_gone():
    commands, context = _commands()
    event = _event()

    await commands.provider(event, "1")

    assert "Invalid parameter" in _text(event)
    context.provider_manager.set_provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_switching_tts_still_works():
    commands, context = _commands()
    event = _event()

    await commands.provider(event, "tts", 1)

    context.provider_manager.set_provider.assert_awaited_once_with(
        provider_id="tts-a",
        provider_type=ProviderType.TEXT_TO_SPEECH,
        umo=event.unified_msg_origin,
    )
