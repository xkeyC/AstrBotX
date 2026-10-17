import asyncio
from unittest.mock import AsyncMock

import pytest

import astrbot.api  # noqa: F401  # Initialize API before platform adapters.
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import (
    AiocqhttpAdapter,
)


def make_adapter(**config):
    adapter = AiocqhttpAdapter(
        {
            "id": "napcat",
            "type": "aiocqhttp",
            "ws_reverse_host": "127.0.0.1",
            "ws_reverse_port": 6199,
            **config,
        },
        {},
        asyncio.Queue(),
    )
    adapter.bot.call_action = AsyncMock(return_value=[{"user_id": 100}])
    adapter.bot.send = AsyncMock()
    adapter.handle_msg = AsyncMock()
    return adapter


def message(user_id, message_type="private"):
    return {
        "post_type": "message",
        "message_type": message_type,
        "sub_type": "friend" if message_type == "private" else "normal",
        "self_id": 1,
        "user_id": user_id,
        "group_id": 500,
        "sender": {"user_id": user_id, "nickname": "someone"},
        "message_id": 7,
        "message": [{"type": "text", "data": {"text": "hi"}}],
    }


def friend_list_calls(adapter):
    return [
        c
        for c in adapter.bot.call_action.await_args_list
        if c.kwargs.get("action") == "get_friend_list"
    ]


@pytest.mark.asyncio
async def test_switch_off_by_default_drops_strangers():
    adapter = make_adapter()
    await adapter.bot._handle_event(message(200))
    adapter.handle_msg.assert_not_awaited()


@pytest.mark.asyncio
async def test_switch_on_lets_strangers_through():
    adapter = make_adapter(allow_non_friend_private_message=True)
    await adapter.bot._handle_event(message(200))
    adapter.handle_msg.assert_awaited_once()
    assert friend_list_calls(adapter) == []


@pytest.mark.asyncio
async def test_switch_off_drops_strangers_and_keeps_friends():
    adapter = make_adapter()
    await adapter.bot._handle_event(message(100))
    await adapter.bot._handle_event(message(100))
    await adapter.bot._handle_event(message(200))
    assert adapter.handle_msg.await_count == 2
    # One fetch serves the friend; the stranger falls within the 30 s window.
    assert len(friend_list_calls(adapter)) == 1
    assert friend_list_calls(adapter)[0].kwargs == {
        "action": "get_friend_list",
        "self_id": 1,
    }
    adapter.bot.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_messages_are_not_filtered():
    adapter = make_adapter()
    await adapter.bot._handle_event(message(200, "group"))
    adapter.handle_msg.assert_awaited_once()
    assert friend_list_calls(adapter) == []


@pytest.mark.asyncio
async def test_stale_list_is_refetched_for_a_missing_sender():
    adapter = make_adapter()
    await adapter.bot._handle_event(message(200))
    adapter.bot.call_action.return_value = [{"user_id": 100}, {"user_id": 200}]
    fetched_at, friends = adapter._friend_cache["1"]
    adapter._friend_cache["1"] = (fetched_at - 31, friends)
    await adapter.bot._handle_event(message(200))
    adapter.handle_msg.assert_awaited_once()
    assert len(friend_list_calls(adapter)) == 2


@pytest.mark.asyncio
async def test_friend_add_notice_updates_the_cache():
    adapter = make_adapter()
    await adapter.bot._handle_event(message(100))
    await adapter.bot._handle_event(
        {
            "post_type": "notice",
            "notice_type": "friend_add",
            "self_id": 1,
            "user_id": 200,
        }
    )
    adapter.handle_msg.reset_mock()
    await adapter.bot._handle_event(message(200))
    adapter.handle_msg.assert_awaited_once()
    assert len(friend_list_calls(adapter)) == 1


@pytest.mark.asyncio
async def test_failed_fetch_drops_and_waits_before_retrying():
    adapter = make_adapter()
    adapter.bot.call_action.side_effect = RuntimeError("napcat down")
    await adapter.bot._handle_event(message(100))
    await adapter.bot._handle_event(message(100))
    adapter.handle_msg.assert_not_awaited()
    assert len(friend_list_calls(adapter)) == 1


@pytest.mark.asyncio
async def test_reply_hint_goes_to_each_stranger_once_per_window():
    adapter = make_adapter(non_friend_private_reply="  请先添加好友  ")
    await adapter.bot._handle_event(message(200))
    await adapter.bot._handle_event(message(200))
    await adapter.bot._handle_event(message(300))
    adapter.handle_msg.assert_not_awaited()
    assert [
        (c.args[0]["user_id"], c.args[1]) for c in adapter.bot.send.await_args_list
    ] == [
        (200, "请先添加好友"),
        (300, "请先添加好友"),
    ]
    assert adapter.bot.send.await_args_list[0].kwargs == {"self_id": 1}
    adapter._non_friend_replied_at["1:200"] -= 600
    await adapter.bot._handle_event(message(200))
    assert adapter.bot.send.await_count == 3
