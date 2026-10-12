import asyncio
from types import SimpleNamespace

import pytest

from astrbot.core.platform.sources.mumble.client import Channel, User
from astrbot.core.platform.sources.mumble.mumble_adapter import (
    SERVER_SESSION,
    MumblePlatformAdapter,
)


class FakeSession:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.people: list[int | None] = []

    async def event(self, text: str) -> None:
        self.events.append(text)

    async def set_people(self, others: int | None) -> None:
        self.people.append(others)


def adapter_with(*users: User) -> MumblePlatformAdapter:
    """An adapter whose client knows ``users`` (the first is the bot) and
    the channels Lobby (1) and Games (2), with no voice session yet."""
    adapter = object.__new__(MumblePlatformAdapter)
    adapter.client = SimpleNamespace(
        session=users[0].session,
        users={user.session: user for user in users},
        channels={1: Channel(1, "Lobby"), 2: Channel(2, "Games")},
        me=users[0],
    )
    adapter.voice_sessions = {}
    adapter._tasks = set()
    adapter._present = (None, {})
    return adapter


async def settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_comings_and_goings_of_the_bots_channel_are_told():
    bot = User(1, "Jarvis", channel_id=1)
    alice = User(2, "Alice", channel_id=1)
    adapter = adapter_with(bot, alice, User(3, "Carol", channel_id=2))
    adapter._people_changed()  # the base, as after connecting
    assert adapter._here() == '"Lobby" (here: Alice)'
    session = FakeSession()
    adapter.voice_sessions[SERVER_SESSION] = session

    bob = User(4, "Bob", channel_id=1)
    adapter.client.users[4] = bob
    adapter._people_changed()
    alice.channel_id = 2
    adapter._people_changed()
    # Someone elsewhere changing channels is nothing to tell.
    adapter.client.users[3].channel_id = 0
    adapter._people_changed()
    del adapter.client.users[4]
    adapter._people_changed()
    await settle()
    assert session.events == [
        "Bob joined the voice channel.",
        "Alice left the voice channel.",
        "Bob left the voice channel.",
    ]


@pytest.mark.asyncio
async def test_the_bot_moving_is_one_event_not_everyone_coming_and_going():
    bot = User(1, "Jarvis", channel_id=1)
    adapter = adapter_with(
        bot, User(2, "Alice", channel_id=1), User(3, "Carol", channel_id=2)
    )
    adapter._people_changed()
    session = FakeSession()
    adapter.voice_sessions[SERVER_SESSION] = session
    bot.channel_id = 2
    adapter._people_changed()
    await settle()
    assert session.events == ['You moved to the voice channel "Games" (here: Carol).']


@pytest.mark.asyncio
async def test_nothing_is_told_before_there_is_a_base():
    adapter = adapter_with(
        User(1, "Jarvis", channel_id=1), User(2, "Alice", channel_id=1)
    )
    session = FakeSession()
    adapter.voice_sessions[SERVER_SESSION] = session
    adapter._people_changed()
    await settle()
    assert session.events == []
    assert session.people == [1]


def test_an_empty_channel_says_so():
    adapter = adapter_with(User(1, "Jarvis", channel_id=2))
    assert adapter._here() == '"Games" (nobody else is here)'
