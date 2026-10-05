import asyncio
import base64
import ssl
from collections import deque
from types import SimpleNamespace

import pytest

from astrbot.api.event import MessageChain
from astrbot.api.message_components import At, Image, Plain
from astrbot.core.platform.sources.mumble.client import TextMessage, User, VoicePacket
from astrbot.core.platform.sources.mumble.messages import AudioContext
from astrbot.core.platform.sources.mumble.mumble_adapter import (
    SERVER_SESSION,
    MumblePlatformAdapter,
    ensure_certificate,
    user_key,
)
from astrbot.core.platform.sources.mumble.mumble_event import chain_to_html

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 32


@pytest.fixture
def adapter(tmp_path):
    cert, key = ensure_certificate(tmp_path, "bot")
    queue: asyncio.Queue = asyncio.Queue()
    adapter = MumblePlatformAdapter(
        {
            "id": "mumble_test",
            "mumble_host": "localhost",
            "mumble_username": "Jarvis",
            "mumble_text_wake_prefix": "!",
            "mumble_certfile": cert,
            "mumble_keyfile": key,
        },
        {},
        queue,
    )
    client = adapter.client
    client.session = 1
    client.users = {
        1: User(1, name="Jarvis"),
        2: User(2, name="alice", hash="abc123"),
        3: User(3, name="guest"),
    }
    sent: list[tuple[str, dict]] = []
    client.send_text = lambda message, **target: sent.append((message, target))
    client.set_self_state = lambda **state: sent.append(("state", state))
    client._writer = object()  # type: ignore[assignment]

    async def drain():
        return None

    client.drain = drain
    adapter.sent = sent
    adapter.queue = queue
    return adapter


def text(actor: int, message: str, *, private: bool = False) -> TextMessage:
    return TextMessage(
        actor=actor,
        message=message,
        sessions=[1] if private else [],
        channel_ids=[] if private else [0],
        tree_ids=[],
    )


def test_certificate_is_created_once_and_loads(tmp_path):
    cert, key = ensure_certificate(tmp_path, "bot")
    assert ensure_certificate(tmp_path, "other") == (cert, key)
    ssl.create_default_context().load_cert_chain(cert, key)


def test_user_key_prefers_certificate_hash():
    assert user_key(User(2, name="alice", hash="abc", user_id=4)) == "abc"
    assert user_key(User(2, name="alice", user_id=4)) == "uid:4"
    # Without a certificate or registration a name is not an identity.
    assert user_key(User(3, name="guest")) == "session:3:guest"


@pytest.mark.asyncio
async def test_channel_text_wakes_only_with_prefix(adapter):
    adapter._on_text(text(2, "<p>!what&nbsp;time is it</p>"))
    adapter._on_text(text(3, "just chatting"))
    adapter._on_text(text(1, "!my own echo"))  # the bot itself
    adapter._on_text(text(2, "!"))  # prefix alone
    woken = adapter.queue.get_nowait()
    chat = adapter.queue.get_nowait()
    assert adapter.queue.empty()
    assert isinstance(woken.message_obj.message[0], At)
    assert str(woken.message_obj.message[0].qq) == woken.message_obj.self_id == "1"
    assert woken.message_str == "what time is it"
    assert woken.message_obj.sender.user_id == "abc123"
    assert (woken.session_id, chat.session_id) == (SERVER_SESSION, SERVER_SESSION)
    assert woken.message_obj.group.group_id == SERVER_SESSION
    assert [type(c) for c in chat.message_obj.message] == [Plain]


@pytest.mark.asyncio
async def test_private_text_and_inline_images(adapter):
    encoded = base64.b64encode(PNG).decode()
    adapter._on_text(
        text(3, f'look <img src="data:image/png;base64,{encoded}"/>', private=True)
    )
    event = adapter.queue.get_nowait()
    assert event.message_obj.type.name == "FRIEND_MESSAGE"
    assert event.session_id == "session:3:guest"
    kinds = [type(c) for c in event.message_obj.message]
    assert kinds == [Plain, Image]


@pytest.mark.asyncio
async def test_mute_commands(adapter):
    adapter._on_text(text(2, "!闭麦"))
    await asyncio.sleep(0)
    assert adapter.muted
    assert ("state", {"self_mute": True, "self_deaf": True}) in adapter.sent
    adapter._on_text(text(2, "/unmute", private=True))
    await asyncio.sleep(0)
    assert not adapter.muted
    adapter._on_text(text(2, "闭麦"))  # channel text without prefix is chat
    await asyncio.sleep(0)
    assert not adapter.muted
    assert adapter.queue.qsize() == 1


@pytest.mark.asyncio
async def test_send_chain_targets(adapter):
    await adapter.send_chain(SERVER_SESSION, MessageChain([Plain("**hi**")]))
    await adapter.send_chain("abc123", MessageChain([Plain("psst")]))
    await adapter.send_chain("offline", MessageChain([Plain("lost")]))
    assert adapter.sent == [
        ("<b>hi</b>", {"channel_ids": [0]}),
        ("psst", {"sessions": [2]}),
    ]


@pytest.mark.asyncio
async def test_channel_reply_goes_where_the_message_was_sent(adapter):
    from astrbot.core.platform.sources.mumble.client import Channel

    adapter.client.channels = {0: Channel(0), 5: Channel(5, parent=0)}
    origin = TextMessage(
        actor=2, message="x", sessions=[], channel_ids=[0, 9], tree_ids=[5]
    )
    await adapter.send_chain(SERVER_SESSION, MessageChain([Plain("ok")]), origin=origin)
    # Channel 9 no longer exists: listing it would make the server drop all.
    assert adapter.sent == [("ok", {"channel_ids": [0], "tree_ids": [5]})]
    adapter.sent.clear()
    gone = TextMessage(actor=2, message="x", sessions=[], channel_ids=[9], tree_ids=[])
    await adapter.send_chain(SERVER_SESSION, MessageChain([Plain("ok")]), origin=gone)
    assert adapter.sent == [("ok", {"channel_ids": [0]})]  # the bot's channel


@pytest.mark.asyncio
async def test_long_replies_are_paced_after_the_burst(adapter, monkeypatch):
    from astrbot.core.platform.sources.mumble import mumble_adapter

    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(mumble_adapter.asyncio, "sleep", fake_sleep)
    adapter.client.server_config["message_length"] = 20
    await adapter.send_chain(SERVER_SESSION, MessageChain([Plain("0123456789\n" * 8)]))
    assert len(adapter.sent) == 8
    assert sleeps == [mumble_adapter.TEXT_INTERVAL] * (8 - mumble_adapter.TEXT_BURST)


@pytest.mark.asyncio
async def test_zero_length_limit_means_unlimited(adapter):
    adapter.client.server_config["message_length"] = 0
    await adapter.send_chain(SERVER_SESSION, MessageChain([Plain("x" * 9000)]))
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_chain_to_html_splits_and_embeds_images(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(PNG)
    chain = MessageChain(
        [Plain("line\n" * 30), Image.fromFileSystem(str(image)), Plain("after")]
    )
    messages = await chain_to_html(chain, message_length=60, image_length=10000)
    assert all(len(m) <= 60 for m in messages if not m.startswith("<img"))
    assert sum(m.count("line") for m in messages) == 30
    assert any(m.startswith('<img src="data:image/png;base64,') for m in messages)
    assert messages[-1] == "after"
    too_small = await chain_to_html(chain, message_length=5000, image_length=10)
    assert any("too large" in m for m in too_small)


def test_voice_routing_in_standby(adapter, monkeypatch):
    started: list[str] = []

    class FakeSession:
        def __init__(self, key):
            self.key = key
            self.fed = []

            class Mixer:
                def feed(inner, *args):
                    self.fed.append(args)

            class Media:
                mixer = Mixer()

            self.media = Media()

    def start(key, user):
        started.append(key)
        session = FakeSession(key)
        adapter.voice_sessions[key] = session
        return session

    monkeypatch.setattr(adapter, "_start_voice", start)
    speech = {"n": 0}

    def detector(speaker, data, terminator):
        speech["n"] += 1
        return speech["n"] >= 3

    monkeypatch.setattr(adapter._detector, "feed", detector)

    def packet(context, frame):
        return VoicePacket(2, context, frame, b"x", False)

    adapter._on_voice(packet(AudioContext.NORMAL, 0))
    adapter._on_voice(packet(AudioContext.NORMAL, 1))
    assert started == []
    adapter._on_voice(packet(AudioContext.NORMAL, 2))
    assert started == [SERVER_SESSION]
    # The pre-roll that woke it is replayed into the new session.
    assert len(adapter.voice_sessions[SERVER_SESSION].fed) == 3
    adapter._on_voice(packet(AudioContext.NORMAL, 3))
    assert len(adapter.voice_sessions[SERVER_SESSION].fed) == 4
    adapter._on_voice(packet(AudioContext.LISTEN, 4))  # not our conversation
    assert len(adapter.voice_sessions[SERVER_SESSION].fed) == 4
    adapter._on_voice(packet(AudioContext.WHISPER, 5))
    assert started == [SERVER_SESSION, "whisper:abc123"]
    adapter.muted = True
    adapter._on_voice(packet(AudioContext.NORMAL, 6))
    assert len(adapter.voice_sessions[SERVER_SESSION].fed) == 4


def test_whisper_targets_are_released_and_bounded(adapter, monkeypatch):
    targets: list[tuple[int, list[int]]] = []
    adapter.client.set_voice_target = lambda t, sessions: targets.append((t, sessions))
    users = [User(100 + i, name=f"u{i}", hash=f"h{i}") for i in range(31)]
    got = [adapter._whisper_target(u) for u in users]
    assert got[:30] == list(range(1, 31))
    assert got[30] is None  # all 30 in use: no whisper session, no crash

    class Closed:
        key = "whisper:h0"

    adapter._voice_closed(Closed())
    assert adapter._whisper_target(users[30]) == 1


def test_departed_speaker_state_is_dropped(adapter):
    adapter._preroll[SERVER_SESSION] = deque(
        [(0.0, 2, b"a", False), (0.0, 3, b"b", False)]
    )
    adapter._preroll["whisper:abc123"] = deque([(0.0, 2, b"a", False)])
    adapter._on_user_removed(adapter.client.users[2], {})
    assert list(adapter._preroll[SERVER_SESSION]) == [(0.0, 3, b"b", False)]
    assert "whisper:abc123" not in adapter._preroll


@pytest.mark.asyncio
async def test_the_channel_voice_session_follows_who_is_in_the_channel(adapter):
    told: list[int | None] = []

    class Session:
        media = SimpleNamespace(mixer=SimpleNamespace(forget=lambda session: None))

        async def set_people(self, others):
            told.append(others)

    adapter.voice_sessions[SERVER_SESSION] = Session()
    users = adapter.client.users
    # Everyone in the root channel: two others.
    assert adapter._channel_others() == 2
    # One leaves for another channel, then the other goes away.
    users[2].channel_id = 5
    adapter._on_user_changed(users[2], {"channel_id"})
    adapter._on_user_removed(users.pop(3), {})
    await asyncio.sleep(0)
    assert told == [1, 0]
