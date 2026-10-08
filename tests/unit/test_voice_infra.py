import asyncio
import base64
import json
import wave
from types import SimpleNamespace

import av
import pytest
import pytest_asyncio
from aiortc.mediastreams import MediaStreamError
from sqlmodel import select

from astrbot.core.config.agent_runner import get_agent_runner_config_default
from astrbot.core.db.po import ProviderStat
from astrbot.core.voice import chat as chat_module
from astrbot.core.voice import infra as infra_module
from astrbot.core.voice import record as record_module
from astrbot.core.voice import session as voice
from astrbot.core.voice.chat import VOICE_SESSIONS, VoiceChat
from astrbot.core.voice.infra import (
    INFRA_INSTRUCTIONS,
    InfraVoiceSession,
    SpeechTrack,
    infra_settings_error,
)
from astrbot.core.voice.session import VoiceOptions, VoiceSession, new_voice_session


class FakeChat(VoiceChat):
    """The real ordering and busy logic; the chat's turns are faked."""

    def __init__(self, private: bool) -> None:
        super().__init__(
            umo="test:FriendMessage:1" if private else "test:GroupMessage:server",
            private=private,
            sender_name="Alice" if private else "Voice",
        )
        self.asked: list[str] = []
        self.speakers: list[str | None] = []
        self.answer: str | None = "It is three."
        self.persona = "Speak like a pirate."
        self.memory: dict = {}

    async def voice_persona(self) -> str:
        return self.persona

    def memory_config(self) -> dict:
        return self.memory

    async def ask(self, body: str, speaker: str | None = None) -> str | None:
        self.asked.append(body)
        self.speakers.append(speaker)
        return self.answer


class Input:
    """Platform audio: one 20 ms frame of 48 kHz stereo, then nothing."""

    def __init__(self) -> None:
        self.sent = False

    async def recv(self):
        if self.sent:
            await asyncio.Event().wait()
        self.sent = True
        frame = av.AudioFrame(format="s16", layout="stereo", samples=960)
        frame.planes[0].update(bytes(960 * 4))
        frame.sample_rate = 48000
        frame.pts = 0
        return frame


class FakeMedia:
    def __init__(self) -> None:
        self.flushed = 0
        self.played: list = []
        self.track = Input()

    async def play(self, track) -> None:
        self.played.append(track)
        await asyncio.Event().wait()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def flush(self) -> None:
        self.flushed += 1


class FakeRuntime:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.audio: list[dict] = []
        self.texts: list[tuple[str, str]] = []
        self.stopped = 0
        # The voice thread's model and cumulative usage (thread_usage).
        self.usage: dict = {
            "model": "deepseek-v4.1-flash",
            "model_provider": "deepseek",
            "total_token_usage": None,
        }

    async def realtime_start(self, thread_id, request):
        self.started.append(json.loads(request))

    async def realtime_append_audio(self, thread_id, frame):
        self.audio.append(json.loads(frame))

    async def realtime_append_text(self, thread_id, text, role="user"):
        self.texts.append((text, role))

    async def realtime_stop(self, thread_id):
        self.stopped += 1

    async def thread_usage(self, thread_id):
        return json.dumps(self.usage)


class FakePump:
    def __init__(self) -> None:
        self.route = None
        self.tool_handler = None
        self.queue: asyncio.Queue | None = None

    def open_turn(self, tool_handler, approval_handler):
        self.tool_handler = tool_handler
        self.queue = asyncio.Queue()
        queue = self.queue

        class Route:
            events = queue

        self.route = Route()
        return queue

    def close_turn(self):
        self.route = None


class FakeEngine:
    def __init__(self) -> None:
        self.rt = FakeRuntime()
        self.pumps: dict[str, FakePump] = {}
        self.params: dict | None = None
        self.forgotten: list[str] = []
        self.locks: dict[str, asyncio.Lock] = {}

    def session_lock(self, key):
        return self.locks.setdefault(key, asyncio.Lock())

    async def open_thread(self, state, params):
        self.params = params
        return {"thread_id": "t1", "rollout_path": None}, True

    def pump(self, thread_id):
        return self.pumps.setdefault(thread_id, FakePump())

    async def forget_thread(self, thread_id):
        self.forgotten.append(thread_id)


class FakeSp:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.values: dict[str, dict] = {}

    async def get_async(self, **kwargs):
        return self.values.get(kwargs["key"], {})

    async def put_async(self, **kwargs):
        self.keys.append(kwargs["key"])
        self.values[kwargs["key"]] = kwargs["value"]


def runner_config(**voice_settings) -> dict:
    config = get_agent_runner_config_default("codex")
    config["realtime_voice"].update(
        {
            "backend": "local_infra",
            "infra_url": "ws://127.0.0.1:17890/v1/realtime",
            "infra_token": "secret",
            "emotion": "happy",
            "emotion_strength": 0.5,
            "text_model_provider": "deepseek",
            "text_model": "deepseek-v4.1-flash",
            "text_reasoning_effort": "none",
            **voice_settings,
        }
    )
    return config


@pytest_asyncio.fixture
async def voice_db(monkeypatch, temp_db):
    """The voice records' database, ready (its first use is slow)."""
    async with temp_db.get_db():
        pass
    monkeypatch.setattr(record_module, "db_helper", temp_db)
    return temp_db


@pytest.fixture
def engine(monkeypatch, voice_db):
    engine = FakeEngine()

    async def codex_engine():
        return engine

    monkeypatch.setattr(voice, "_codex_engine", codex_engine)
    monkeypatch.setattr(voice, "sp", FakeSp())
    monkeypatch.setattr(voice, "_runner_config", runner_config)
    return engine


def told(engine, role: str) -> list[str]:
    """The texts given to the realtime conversation in ``role``."""
    return [text for text, r in engine.rt.texts if r == role]


async def eventually(check) -> None:
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0.01)
    assert check()


async def open_session(
    engine,
    private: bool = True,
    started: bool = True,
    memory: dict | None = None,
    media=None,
    **kwargs,
):
    t = SimpleNamespace(
        media=media or FakeMedia(), chat=FakeChat(private), closed=[], failures=[]
    )
    t.chat.memory = memory or {}
    t.session = new_voice_session(
        key="server",
        scope_id="test:voice:server",
        prompt="You are Jarvis, on a call.",
        options=VoiceOptions(name="Jarvis", aliases=["Jar"]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
        thread_key="mumble_voice_thread",
        **kwargs,
    )
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    t.pump = engine.pumps.get("t1")
    if started and t.pump is not None:
        await t.pump.queue.put({"type": "realtime_conversation_started"})
        await eventually(lambda: t.session.ready)
    return t


async def event(t, payload: dict) -> None:
    await t.pump.queue.put(
        {"type": "realtime_conversation_realtime", "payload": payload}
    )


def test_the_configured_backend_picks_the_session(monkeypatch):
    monkeypatch.setattr(voice, "_runner_config", runner_config)
    kwargs = {
        "key": "k",
        "scope_id": "s",
        "prompt": "p",
        "options": VoiceOptions(name="J", aliases=[]),
        "media": FakeMedia(),
        "on_closed": lambda s: None,
        "chat": FakeChat(True),
    }
    assert type(new_voice_session(**kwargs)) is InfraVoiceSession
    monkeypatch.setattr(
        voice, "_runner_config", lambda: runner_config(backend="builtin")
    )
    assert type(new_voice_session(**kwargs)) is VoiceSession


@pytest.mark.asyncio
async def test_the_platforms_context_goes_with_the_start_then_as_context(engine):
    t = SimpleNamespace(media=FakeMedia(), chat=FakeChat(False), closed=[], failures=[])
    t.session = new_voice_session(
        key="room",
        scope_id="test:voice:room",
        prompt="You are Jarvis, in a room.",
        options=VoiceOptions(name="Jarvis", aliases=[]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
    )
    # Before the start: it goes with the start.
    await t.session.set_context("(Room: Home; here: Alice)")
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    assert engine.rt.started[0]["realtime_start_instructions"].endswith(
        "(Room: Home; here: Alice)"
    )
    await engine.pumps["t1"].queue.put({"type": "realtime_conversation_started"})
    await eventually(lambda: t.session.ready)
    assert told(engine, "context") == []
    # Later: as context, for the next input.
    await t.session.set_context("(Room: Home; here: Alice, Bob)")
    assert told(engine, "context") == ["(Room: Home; here: Alice, Bob)"]
    await t.session.close("done")


@pytest.mark.asyncio
async def test_the_voice_thread_keeps_the_chats_memories(engine):
    memory = {
        "features.memories": True,
        "memories.scope_key": "test:GroupMessage:server",
        "memories.may_write_global": False,
    }
    await open_session(engine, private=False, memory=memory)
    config = engine.params["config"]
    # The chat's memory settings win over the voice thread's default (off).
    assert {k: config[k] for k in memory} == memory


@pytest.mark.asyncio
async def test_the_voice_thread_talks_on_the_chosen_model(engine):
    t = await open_session(engine, private=False)
    params = engine.params
    assert params["base_instructions"].startswith(INFRA_INSTRUCTIONS)
    # The platform's prompt and the voice persona, but not the time: the
    # prefix stays the same from call to call.
    assert "You are Jarvis, on a call." in params["base_instructions"]
    assert params["base_instructions"].endswith("Speak like a pirate.")
    assert "Today is" not in params["base_instructions"]
    assert [tool["name"] for tool in params["dynamic_tools"]] == [
        "backend_task",
        "stay_silent",
    ]
    config = params["config"]
    assert config["realtime.backend"] == "local_multimodal_infra"
    assert config["realtime.local_infra.url"] == "ws://127.0.0.1:17890/v1/realtime"
    assert config["realtime.local_infra.token"] == "secret"
    assert config["realtime.local_infra.session"] == {
        "name": "Jarvis",
        "aliases": ["Jar"],
        "group": True,
        # A room of unknown size: only what calls the bot by name.
        "wake": True,
        "tts_emotion": "happy",
        "tts_emotion_strength": 0.5,
        "tts_stream_text": True,
    }
    assert "realtime.local_infra.ref_text" not in config
    assert config["model_provider"] == "deepseek"
    assert config["model"] == "deepseek-v4.1-flash"
    assert config["model_reasoning_effort"] == "none"
    # No memories when the runner keeps none.
    assert config["features.memories"] is False
    assert config["model_tool_mode"] == "direct"
    assert "realtime.host_routes_handoffs" not in config
    # Its own thread, apart from the realtime one.
    assert voice.sp.keys == ["mumble_voice_thread_infra"]
    request = engine.rt.started[0]
    assert request["transport"] == {"type": "websocket"}
    assert "Today is" in request["realtime_start_instructions"]
    # Codex fills in the time of the hang-up.
    assert "{now}" in request["realtime_end_instructions"]
    assert VOICE_SESSIONS[t.chat.umo] is t.session
    await t.session.close("done")


@pytest.mark.asyncio
async def test_audio_goes_both_ways(engine):
    t = await open_session(engine)
    # Platform audio reaches Codex as 16 kHz mono.
    await eventually(lambda: engine.rt.audio)
    frame = engine.rt.audio[0]
    assert frame["sample_rate"] == 16000 and frame["num_channels"] == 1
    assert len(base64.b64decode(frame["data"])) == frame["samples_per_channel"] * 2
    # The server's speech is played.
    track = t.media.played[0]
    assert isinstance(track, SpeechTrack)
    pcm = bytes(range(10)) * 48
    await event(
        t,
        {
            "AudioOut": {
                "data": base64.b64encode(pcm).decode(),
                "sample_rate": 24000,
                "num_channels": 1,
            }
        },
    )
    played = await asyncio.wait_for(track.recv(), 2)
    assert bytes(played.planes[0])[: len(pcm)] == pcm
    # Talked over: what is buffered goes.
    await event(t, {"ResponseCancelled": {"response_id": "m1"}})
    await eventually(lambda: t.media.flushed == 1)
    await t.session.close("done")
    assert engine.rt.stopped == 1
    with pytest.raises(MediaStreamError):
        await track.recv()


@pytest.mark.asyncio
async def test_a_backend_task_runs_in_the_chat_and_its_answer_is_told(engine):
    t = await open_session(engine)
    await event(t, {"InputTranscriptDone": {"text": "what time is it"}})
    await eventually(lambda: t.session._heard)
    result = await t.pump.tool_handler(
        {"tool": "backend_task", "callId": "c1", "arguments": {"task": "Tell the time"}}
    )
    assert result["success"] is True
    assert "later" in result["contentItems"][0]["text"]
    await eventually(lambda: engine.rt.texts)
    assert t.chat.asked == [
        chat_module.TASK_BODY.format(heard="what time is it", task="Tell the time")
    ]
    text, role = engine.rt.texts[0]
    assert text == infra_module.RESULT_PROMPT.format(
        task="Tell the time", answer="It is three."
    )
    assert role == "developer"
    unknown = await t.pump.tool_handler({"tool": "shell", "arguments": {}})
    assert unknown["success"] is False
    await t.session.close("done")


@pytest.mark.asyncio
async def test_a_platform_tool_is_done_at_once(engine):
    calls: list[dict] = []

    async def wave(arguments: dict) -> str:
        calls.append(arguments)
        return "Waved."

    async def broken(arguments: dict) -> str:
        raise RuntimeError("no avatar")

    spec = {
        "type": "function",
        "name": "wave",
        "description": "Wave.",
        "inputSchema": {"type": "object", "properties": {"end_turn": {}, "say": {}}},
    }
    t = SimpleNamespace(media=FakeMedia(), chat=FakeChat(False), closed=[], failures=[])
    t.session = new_voice_session(
        key="room",
        scope_id="test:voice:room",
        prompt="You are Jarvis, in a room.",
        options=VoiceOptions(name="Jarvis", aliases=[]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
        tools=[
            voice.VoiceTool(spec=spec, run=wave),
            voice.VoiceTool(spec={**spec, "name": "broken"}, run=broken),
            voice.VoiceTool(spec={**spec, "name": "jump"}, run=wave, ends_turn=True),
            voice.VoiceTool(spec={**spec, "name": "walk", "inputSchema": {}}, run=wave),
        ],
    )
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    pump = engine.pumps["t1"]
    assert [tool["name"] for tool in engine.params["dynamic_tools"]] == [
        "backend_task",
        "stay_silent",
        "wave",
        "broken",
        "jump",
        "walk",
    ]
    result = await pump.tool_handler({"tool": "wave", "arguments": {"hand": "left"}})
    assert result == {
        "contentItems": [{"type": "inputText", "text": "Waved."}],
        "success": True,
    }
    assert calls == [{"hand": "left"}]
    # Not handed to the chat.
    assert t.chat.asked == []
    failed = await pump.tool_handler({"tool": "broken", "arguments": {}})
    assert failed["success"] is False
    assert "no avatar" in failed["contentItems"][0]["text"]
    # The action is the whole answer: the turn ends, saying its say.
    assert await pump.tool_handler(
        {"tool": "jump", "arguments": {"say": " Watch this! "}}
    ) == {
        "contentItems": [{"type": "inputText", "text": "Waved."}],
        "success": True,
        "endTurn": True,
        "speak": "Watch this!",
    }
    # The model may keep the turn going.
    assert await pump.tool_handler(
        {"tool": "jump", "arguments": {"say": "Hop.", "end_turn": False}}
    ) == {"contentItems": [{"type": "inputText", "text": "Waved."}], "success": True}
    # ... or end it with a tool that does not by default.
    assert (await pump.tool_handler({"tool": "wave", "arguments": {"end_turn": True}}))[
        "endTurn"
    ] is True
    # A tool that does not offer end_turn is not ended by it: its result is
    # for the model to see.
    assert "endTurn" not in await pump.tool_handler(
        {"tool": "walk", "arguments": {"end_turn": True}}
    )
    assert await pump.tool_handler({"tool": "stay_silent", "arguments": {}}) == {
        "contentItems": [{"type": "inputText", "text": "Silent."}],
        "success": True,
        "endTurn": True,
    }
    assert t.chat.asked == []
    await t.session.close("done")


@pytest.mark.asyncio
async def test_a_platform_tool_may_answer_with_a_picture(engine):
    picture = [
        {"type": "inputText", "text": "The view now."},
        {"type": "inputImage", "imageUrl": "data:image/jpeg;base64,AAAA"},
    ]

    async def look(arguments: dict) -> list[dict]:
        return picture

    spec = {
        "type": "function",
        "name": "look",
        "description": "Look.",
        "inputSchema": {},
    }
    t = SimpleNamespace(media=FakeMedia(), chat=FakeChat(False), closed=[], failures=[])
    t.session = new_voice_session(
        key="room",
        scope_id="test:voice:room",
        prompt="You are Jarvis, in a room.",
        options=VoiceOptions(name="Jarvis", aliases=[]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
        tools=[voice.VoiceTool(spec=spec, run=look)],
    )
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    assert await engine.pumps["t1"].tool_handler({"tool": "look", "arguments": {}}) == {
        "contentItems": picture,
        "success": True,
    }
    await t.session.close("done")


@pytest.mark.asyncio
async def test_the_voice_and_its_transcript_go_to_the_server(
    engine, monkeypatch, tmp_path
):
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF")
    monkeypatch.setattr(
        voice,
        "_runner_config",
        lambda: runner_config(
            ref_audio=str(ref), ref_text=" 你好，我是小乐。 ", stream_text=False
        ),
    )
    t = await open_session(engine)
    config = engine.params["config"]
    assert config["realtime.local_infra.ref_audio_path"] == str(ref)
    assert config["realtime.local_infra.ref_text"] == "你好，我是小乐。"
    assert config["realtime.local_infra.session"]["tts_stream_text"] is False
    await t.session.close("done")


@pytest.mark.asyncio
async def test_a_reference_text_without_its_audio_is_left_out(engine, monkeypatch):
    monkeypatch.setattr(
        voice, "_runner_config", lambda: runner_config(ref_text="你好。")
    )
    t = await open_session(engine)
    assert "realtime.local_infra.ref_text" not in engine.params["config"]
    await t.session.close("done")


def test_a_transcribed_reference_must_be_short(tmp_path):
    def silence(seconds: int) -> str:
        path = tmp_path / f"{seconds}s.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(bytes(2 * 8000 * seconds))
        return str(path)

    long = runner_config(ref_audio=silence(16), ref_text="你好。")["realtime_voice"]
    assert "at most 15 s" in infra_settings_error(long)
    long["ref_text"] = ""
    assert infra_settings_error(long) is None
    short = runner_config(ref_audio=silence(5), ref_text="你好。")["realtime_voice"]
    assert infra_settings_error(short) is None


@pytest.mark.asyncio
async def test_bad_settings_fail_the_start(engine, monkeypatch):
    monkeypatch.setattr(
        voice, "_runner_config", lambda: runner_config(infra_url="http://x")
    )
    t = await open_session(engine, started=False)
    await eventually(lambda: t.failures)
    assert "ws://" in str(t.failures[0])
    await eventually(lambda: t.closed)


@pytest.mark.asyncio
async def test_a_failed_server_start_fails_the_session(engine):
    t = await open_session(engine, started=False)
    await event(t, {"Error": "cannot connect to ws://127.0.0.1:17890/v1/realtime"})
    await eventually(lambda: t.failures)
    assert "cannot connect" in str(t.failures[0])


@pytest.mark.asyncio
async def test_a_conversation_closed_while_starting_fails_the_start(engine):
    t = await open_session(engine, started=False)
    await t.pump.queue.put(
        {"type": "realtime_conversation_closed", "reason": "transport_closed"}
    )
    await eventually(lambda: t.failures and t.closed)
    assert len(t.failures) == 1
    assert "transport_closed" in str(t.failures[0])


@pytest.mark.asyncio
async def test_the_session_ends_with_the_conversation(engine):
    t = await open_session(engine)
    await t.pump.queue.put(
        {"type": "realtime_conversation_closed", "reason": "transport_closed"}
    )
    await eventually(lambda: t.closed)
    assert t.session.closing


@pytest.mark.asyncio
async def test_a_provider_without_its_model_fails_the_start(engine, monkeypatch):
    monkeypatch.setattr(voice, "_runner_config", lambda: runner_config(text_model=""))
    t = await open_session(engine, started=False)
    await eventually(lambda: t.failures)
    assert "voice text model" in str(t.failures[0])


@pytest.mark.asyncio
async def test_a_server_that_fails_before_starting_is_reported(engine):
    t = await open_session(engine, started=False)
    # Codex reports the failure and ends the conversation at once.
    await event(t, {"Error": "the voice server closed the connection"})
    await t.pump.queue.put({"type": "realtime_conversation_closed", "reason": "error"})
    await eventually(lambda: t.failures)
    assert "closed the connection" in str(t.failures[0])
    await eventually(lambda: t.closed)


@pytest.mark.asyncio
async def test_a_bad_event_does_not_end_the_conversation(engine):
    t = await open_session(engine)
    await event(t, {"AudioOut": {"data": "!!not base64!!", "sample_rate": 24000}})
    await t.pump.queue.put(
        {"type": "realtime_conversation_closed", "reason": "requested"}
    )
    await eventually(lambda: t.closed)


def _total(input_tokens, cached, output):
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output,
    }


async def _stat_rows(db):
    async with db.get_db() as session:
        result = await session.execute(select(ProviderStat))
        return result.scalars().all()


@pytest.mark.asyncio
async def test_a_call_leaves_its_transcript_and_stats(engine, voice_db):
    engine.rt.usage["total_token_usage"] = _total(1000, 800, 50)
    t = await open_session(engine, private=False)
    await event(t, {"InputTranscriptDone": {"text": "what time is it"}})
    await t.pump.queue.put({"type": "task_started", "turn_id": "u1"})
    await t.pump.queue.put(
        {"type": "token_count", "info": {"total_token_usage": _total(3000, 2500, 90)}}
    )
    await t.pump.queue.put(
        {"type": "task_complete", "turn_id": "u1", "time_to_first_token_ms": 400}
    )
    await event(t, {"OutputTranscriptDone": {"text": "It is three."}})
    await event(t, {"InputTranscriptDone": {"text": "thanks"}})
    await t.session.close("hung up")

    # Its own conversation under the chat, which stays the chat's choice.
    state = voice.sp.values["mumble_voice_thread_infra"]
    conv = await voice_db.get_conversation_by_id(cid=state["conversation_id"])
    assert (conv.user_id, conv.title) == ("test:GroupMessage:server", "Voice: voice")
    assert conv.content == [
        {"role": "user", "content": "what time is it"},
        {"role": "assistant", "content": "It is three."},
        {"role": "user", "content": "thanks"},
    ]
    [row] = await _stat_rows(voice_db)
    assert (
        row.agent_type,
        row.umo,
        row.conversation_id,
        row.provider_id,
        row.provider_model,
        row.status,
        row.token_input_other,
        row.token_input_cached,
        row.token_output,
        row.time_to_first_token,
    ) == (
        "codex_voice",
        "test:GroupMessage:server",
        conv.conversation_id,
        "deepseek",
        "deepseek-v4.1-flash",
        "completed",
        300,
        1700,
        40,
        0.4,
    )


@pytest.mark.asyncio
async def test_the_next_call_keeps_the_conversation_and_a_cut_turn_counts(
    engine, voice_db
):
    t = await open_session(engine)
    await event(t, {"InputTranscriptDone": {"text": "hello"}})
    await t.session.close("hung up")
    cid = voice.sp.values["mumble_voice_thread_infra"]["conversation_id"]

    engine.rt.started.clear()
    t = await open_session(engine)
    await t.pump.queue.put({"type": "task_started", "turn_id": "u2"})
    await t.pump.queue.put(
        {"type": "token_count", "info": {"total_token_usage": _total(500, 0, 20)}}
    )
    await eventually(lambda: t.pump.queue.empty())
    await t.session.close("hung up")

    assert voice.sp.values["mumble_voice_thread_infra"]["conversation_id"] == cid
    conv = await voice_db.get_conversation_by_id(cid=cid)
    assert conv.content == [{"role": "user", "content": "hello"}]
    [row] = await _stat_rows(voice_db)
    assert (row.status, row.token_input_other, row.token_output) == ("aborted", 500, 20)


@pytest.mark.asyncio
async def test_new_instructions_reach_the_resumed_thread(engine, voice_db, monkeypatch):
    # A changed prompt keeps the thread (and the conversation): the resume is
    # given the new instructions, which Codex prefers over the rollout's.
    opened: list = []

    async def open_thread(state, params):
        opened.append((state, params["base_instructions"]))
        thread = state["thread_id"] if state else f"t{len(opened)}"
        return {"thread_id": thread, "rollout_path": None}, state is None

    monkeypatch.setattr(engine, "open_thread", open_thread)
    monkeypatch.setattr(
        engine, "pump", lambda thread_id: engine.pumps.setdefault("t1", FakePump())
    )

    async def call(prompt: str) -> dict:
        t = SimpleNamespace(
            media=FakeMedia(), chat=FakeChat(True), closed=[], failures=[]
        )
        t.session = new_voice_session(
            key="server",
            scope_id="test:voice:server",
            prompt=prompt,
            options=VoiceOptions(name="Jarvis", aliases=[]),
            media=t.media,
            on_closed=t.closed.append,
            chat=t.chat,
            thread_key="mumble_voice_thread",
        )
        await t.session._open_agent()
        await t.session.close("done")
        return dict(voice.sp.values["mumble_voice_thread_infra"])

    first = await call("You are Jarvis, on a call.")
    changed = await call("You are Jarvis, on a call. Lines start with who spoke.")
    assert opened[1][0]["thread_id"] == first["thread_id"]
    assert "Lines start with who spoke." in opened[1][1]
    assert changed["thread_id"] == first["thread_id"]
    assert changed["conversation_id"] == first["conversation_id"]


@pytest.mark.asyncio
async def test_a_speaker_labels_candidates_go_with_it(engine, monkeypatch):
    monkeypatch.setattr(infra_module, "LABEL_INTERVAL", 0.01)
    media = FakeMedia()
    media.track = PlacedInput([1000, 1960])
    t = await open_session(engine, private=False, media=media)
    await eventually(lambda: len(t.session._positions) == 2)
    session = t.session
    session.label_speaker(
        "Bob",
        1000,
        2920,
        False,
        candidates=[
            {"name": "Bob", "p": 0.55},
            {"name": None, "p": 0.4},
            {"bad": 1},
            {"name": " Ann ", "p": 0.12},
            {"name": "Cid", "p": 0.11},
        ],
    )
    await eventually(lambda: speaker_labels(engine))
    assert speaker_labels(engine) == [
        {
            "name": "Bob",
            "start": 0,
            "end": 640,
            "final": False,
            "candidates": [
                {"name": "Bob", "p": 0.55},
                {"name": None, "p": 0.4},
                {"name": "Ann", "p": 0.12},
            ],
        }
    ]
    # Only the likelihoods changed: told again.
    session.label_speaker(
        "Bob", 1000, 2920, True, candidates=[{"name": "Bob", "p": 0.9}]
    )
    await eventually(lambda: len(speaker_labels(engine)) == 2)
    assert speaker_labels(engine)[1]["candidates"] == [{"name": "Bob", "p": 0.9}]
    await eventually(lambda: not session._labels)
    await t.session.close("done")


@pytest.mark.asyncio
async def test_any_tool_result_reaches_codex_in_time(engine, monkeypatch):
    monkeypatch.setattr(infra_module, "VOICE_TOOL_TIMEOUT", 0.05)

    async def nothing(arguments: dict):
        return None

    async def mapping(arguments: dict):
        return {"ok": True}

    async def slow(arguments: dict) -> str:
        await asyncio.sleep(5)
        return "late"

    spec = {
        "type": "function",
        "name": "nothing",
        "description": ".",
        "inputSchema": {},
    }
    t = SimpleNamespace(media=FakeMedia(), chat=FakeChat(False), closed=[], failures=[])
    t.session = new_voice_session(
        key="room",
        scope_id="test:voice:room",
        prompt="You are Jarvis, in a room.",
        options=VoiceOptions(name="Jarvis", aliases=[]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
        tools=[
            voice.VoiceTool(spec=spec, run=nothing),
            voice.VoiceTool(spec={**spec, "name": "mapping"}, run=mapping),
            voice.VoiceTool(spec={**spec, "name": "slow"}, run=slow),
            # A built-in's name: left out.
            voice.VoiceTool(spec={**spec, "name": "stay_silent"}, run=slow),
        ],
    )
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    names = [tool["name"] for tool in engine.params["dynamic_tools"]]
    assert names == ["backend_task", "stay_silent", "nothing", "mapping", "slow"]
    pump = engine.pumps["t1"]
    assert (await pump.tool_handler({"tool": "nothing", "arguments": {}}))[
        "contentItems"
    ] == [{"type": "inputText", "text": "Done."}]
    assert (await pump.tool_handler({"tool": "mapping", "arguments": {}}))[
        "contentItems"
    ] == [{"type": "inputText", "text": "{'ok': True}"}]
    late = await pump.tool_handler({"tool": "slow", "arguments": {}})
    assert late["success"] is False
    # The built-in, not the platform's tool of that name.
    assert (await pump.tool_handler({"tool": "stay_silent", "arguments": {}}))[
        "endTurn"
    ] is True
    await t.session.close("done")


def test_malformed_content_items_become_text():
    assert infra_module.tool_content([{"type": "inputImage", "image_url": "x"}]) == [
        {"type": "inputText", "text": "[{'type': 'inputImage', 'image_url': 'x'}]"}
    ]
    assert infra_module.tool_content([]) == [{"type": "inputText", "text": "[]"}]
    good = [{"type": "inputText", "text": "a", "extra": 1}]
    assert infra_module.tool_content(good) == [{"type": "inputText", "text": "a"}]


@pytest.mark.asyncio
async def test_context_set_while_starting_is_given_once_open(engine):
    t = SimpleNamespace(media=FakeMedia(), chat=FakeChat(False), closed=[], failures=[])
    t.session = new_voice_session(
        key="room",
        scope_id="test:voice:room",
        prompt="You are Jarvis, in a room.",
        options=VoiceOptions(name="Jarvis", aliases=[]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
    )
    await t.session.set_context("(Room: Home; here: Alice)")
    start = engine.rt.realtime_start

    async def realtime_start(thread_id, request):
        # Changed after the start was asked, before it is open.
        await t.session.set_context("(Room: Home; here: Alice, Bob)")
        await start(thread_id, request)

    engine.rt.realtime_start = realtime_start
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    assert engine.rt.started[0]["realtime_start_instructions"].endswith(
        "(Room: Home; here: Alice)"
    )
    assert engine.rt.texts == []
    await engine.pumps["t1"].queue.put({"type": "realtime_conversation_started"})
    await eventually(lambda: told(engine, "context"))
    assert told(engine, "context") == ["(Room: Home; here: Alice, Bob)"]
    # The same again: not repeated.
    await t.session.set_context("(Room: Home; here: Alice, Bob)")
    assert len(told(engine, "context")) == 1
    await t.session.close("done")


@pytest.mark.asyncio
async def test_the_room_size_sets_what_the_voice_server_passes_on(engine):
    t = SimpleNamespace(media=FakeMedia(), chat=FakeChat(False), closed=[], failures=[])
    t.session = new_voice_session(
        key="room",
        scope_id="test:voice:room",
        prompt="You are Jarvis, in a room.",
        options=VoiceOptions(name="Jarvis", aliases=[]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
    )
    # Three others: only what calls the bot, from the start.
    await t.session.set_people(3)
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    assert engine.params["config"]["realtime.local_infra.session"]["wake"] is True
    await engine.pumps["t1"].queue.put({"type": "realtime_conversation_started"})
    await eventually(lambda: t.session.ready)
    # Given again once up (the thread's settings may be older).
    assert engine.rt.texts == [(json.dumps({"wake": True}), "voice_session")]
    # One left: everything, at once; the same again is not repeated.
    await t.session.set_people(1)
    await t.session.set_people(1)
    # Not known: only what calls the bot again.
    await t.session.set_people(None)
    assert engine.rt.texts == [
        (json.dumps({"wake": True}), "voice_session"),
        (json.dumps({"wake": False}), "voice_session"),
        (json.dumps({"wake": True}), "voice_session"),
    ]
    await t.session.close("done")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "people", "wake"),
    [
        ("always", [1, 0, None], [True, True, True]),
        ("off", [3, None, 1], [False, False, False]),
    ],
)
async def test_wake_mode_overrides_the_headcount(engine, mode, people, wake):
    from astrbot.core.voice.session import wakes_on_name

    assert [wakes_on_name(n, mode) for n in people] == wake
    t = SimpleNamespace(media=FakeMedia(), chat=FakeChat(False), closed=[], failures=[])
    t.session = new_voice_session(
        key="room",
        scope_id="test:voice:room",
        prompt="You are Jarvis, in a room.",
        options=VoiceOptions(name="Jarvis", aliases=[], wake_mode=mode),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
    )
    await t.session.set_people(people[0])
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    assert engine.params["config"]["realtime.local_infra.session"]["wake"] is wake[0]
    await engine.pumps["t1"].queue.put({"type": "realtime_conversation_started"})
    await eventually(lambda: t.session.ready)
    for n in people[1:]:
        await t.session.set_people(n)
    # The same setting all along: given once, once the server is up.
    assert engine.rt.texts == [(json.dumps({"wake": wake[0]}), "voice_session")]
    await t.session.close("done")


@pytest.mark.asyncio
async def test_people_changing_while_it_starts_reach_the_voice_server(engine):
    t = SimpleNamespace(media=FakeMedia(), chat=FakeChat(False), closed=[], failures=[])
    t.session = new_voice_session(
        key="room",
        scope_id="test:voice:room",
        prompt="You are Jarvis, in a room.",
        options=VoiceOptions(name="Jarvis", aliases=[]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
    )
    await t.session.set_people(3)
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    assert engine.params["config"]["realtime.local_infra.session"]["wake"] is True
    # Two left before the server was up.
    await t.session.set_people(1)
    await engine.pumps["t1"].queue.put({"type": "realtime_conversation_started"})
    await eventually(lambda: engine.rt.texts)
    assert engine.rt.texts == [(json.dumps({"wake": False}), "voice_session")]
    await t.session.close("done")


class PlacedInput:
    """Platform audio whose frames say where they came from (``pos``, as
    ``PcmMedia``'s track does; None for silence), then nothing."""

    def __init__(self, positions: list[int | None]) -> None:
        self.positions = list(positions)
        self.pos: int | None = None

    async def recv(self):
        if not self.positions:
            await asyncio.Event().wait()
        self.pos = self.positions.pop(0)
        frame = av.AudioFrame(format="s16", layout="mono", samples=960)
        frame.planes[0].update(bytes(960 * 2))
        frame.sample_rate = 48000
        return frame


def speaker_labels(engine) -> list[dict]:
    """The speaker labels given to the voice server."""
    return [
        json.loads(text)["speaker"]
        for text, role in engine.rt.texts
        if role == "voice_session" and "speaker" in json.loads(text)
    ]


@pytest.mark.asyncio
async def test_speaker_labels_reach_the_server_on_its_clock(engine, monkeypatch):
    monkeypatch.setattr(infra_module, "LABEL_INTERVAL", 0.01)
    media = FakeMedia()
    # 20 ms frames (320 samples at the server): one of silence the media
    # put in, then audio it dropped (2920-3880).
    media.track = PlacedInput([1000, 1960, None, 3880, 4840])
    t = await open_session(engine, private=False, media=media)
    await eventually(lambda: len(t.session._positions) == 4)
    session = t.session
    # Audio before what reached the server, and audio dropped on the way:
    # nothing to tell.
    session.label_speaker("Bob", 500, 900, True)
    session.label_speaker("Bob", 2920, 3400, True)
    # Audio not handed over yet: it waits for it.
    session.label_speaker("Carol", 7000, 8000, True, bearing_deg=-30.0)
    session.label_speaker("Alice", 1480, 5800, True, user_id="usr_1")
    session.label_speaker(None, 5000, 9000, False)
    await eventually(lambda: len(speaker_labels(engine)) == 2)
    assert speaker_labels(engine) == [
        {"name": "Alice", "start": 160, "end": 1600, "final": True},
        # Only what reached the server so far; not final yet.
        {"name": None, "start": 1333, "end": 1600, "final": False},
    ]
    # The same span again, ended: its start stays where it was.
    session.label_speaker(None, 5000, 5500, True)
    await eventually(lambda: len(speaker_labels(engine)) == 3)
    assert speaker_labels(engine)[-1] == {
        "name": None,
        "start": 1333,
        "end": 1500,
        "final": True,
    }
    assert set(session._labels) == {7000}
    await t.session.close("done")


class GatedInput:
    """Platform audio whose frames come when the test puts their positions
    in ``queue`` (``pos``, as ``PcmMedia``'s track says)."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[int] = asyncio.Queue()
        self.pos: int | None = None

    async def recv(self):
        self.pos = await self.queue.get()
        frame = av.AudioFrame(format="s16", layout="mono", samples=960)
        frame.planes[0].update(bytes(960 * 2))
        frame.sample_rate = 48000
        return frame


@pytest.mark.asyncio
async def test_a_speaker_label_not_given_is_tried_again(engine, monkeypatch):
    monkeypatch.setattr(infra_module, "LABEL_INTERVAL", 0.01)
    media = FakeMedia()
    media.track = PlacedInput([1000, 1960])
    t = await open_session(engine, private=False, media=media)
    await eventually(lambda: len(t.session._positions) == 2)
    append = engine.rt.realtime_append_text
    failures = []

    async def flaky(thread_id, text, role="user"):
        if "speaker" in json.loads(text) and not failures:
            failures.append(text)
            raise RuntimeError("busy")
        await append(thread_id, text, role)

    monkeypatch.setattr(engine.rt, "realtime_append_text", flaky)
    session = t.session
    session.label_speaker("Bob", 1000, 2920, True)
    await eventually(lambda: speaker_labels(engine))
    assert failures
    assert speaker_labels(engine) == [
        {"name": "Bob", "start": 0, "end": 640, "final": True}
    ]
    await eventually(lambda: not session._labels)
    # The same span, its name changed: told again.
    session.label_speaker("Bob", 1960, 2920, False)
    await eventually(lambda: len(speaker_labels(engine)) == 2)
    session.label_speaker("Carol", 1960, 2920, False)
    await eventually(lambda: len(speaker_labels(engine)) == 3)
    assert speaker_labels(engine)[1:] == [
        {"name": "Bob", "start": 320, "end": 640, "final": False},
        {"name": "Carol", "start": 320, "end": 640, "final": False},
    ]
    await t.session.close("done")


@pytest.mark.asyncio
async def test_a_new_platform_clock_keeps_its_own_labels(engine, monkeypatch):
    monkeypatch.setattr(infra_module, "LABEL_INTERVAL", 0.5)
    media = FakeMedia()
    media.track = GatedInput()
    t = await open_session(engine, private=False, media=media)
    session = t.session
    for pos in (1000, 1960):
        media.track.queue.put_nowait(pos)
    await eventually(lambda: len(session._positions) == 2)
    session.label_speaker("Old", 1000, 5000, False)
    await eventually(lambda: speaker_labels(engine))
    # Between two rounds: a label on the new clock (the stream came back),
    # then its audio starts there.
    session.label_speaker("New", 0, 1920, True)
    media.track.queue.put_nowait(0)
    await eventually(lambda: len(session._positions) == 1)
    assert set(session._labels) == {0}
    media.track.queue.put_nowait(960)
    await eventually(lambda: len(speaker_labels(engine)) == 2)
    assert speaker_labels(engine) == [
        {"name": "Old", "start": 0, "end": 640, "final": False},
        {"name": "New", "start": 640, "end": 1280, "final": True},
    ]
    await t.session.close("done")


class Calls:
    """An ``on_wake`` that records its calls (async, as a plugin's may be)."""

    def __init__(self, fail: bool = False) -> None:
        self.sources: list[str] = []
        self.speakers: list[str | None] = []
        self.fail = fail

    async def __call__(self, source: str, speaker: str | None = None) -> None:
        self.sources.append(source)
        self.speakers.append(speaker)
        if self.fail:
            raise RuntimeError("bridge down")


def later(t) -> None:
    """As if the last turn to a caller was a while ago."""
    t.session._woke_at = None


def done(text: str, respond: bool, called: bool, speaker: str | None = None):
    return {
        "InputTranscriptDone": {
            "text": text,
            "respond": respond,
            "called": called,
            "speaker": speaker,
        }
    }


@pytest.mark.asyncio
async def test_being_called_by_name_calls_on_wake_once_per_call(engine):
    calls = Calls()
    t = await open_session(engine, private=False, on_wake=calls)
    # A wake word, as soon as it is heard (twice in the utterance); the
    # utterance's transcript then wants a reply: the same call.
    await event(t, {"InputWake": {"word": "Jarvis", "score": 0.9}})
    await event(t, {"InputWake": {"word": "Jarvis", "score": 0.8}})
    await event(t, done("Bob: Jarvis, hi", True, True, "Bob"))
    await eventually(lambda: t.session._heard == "Bob: Jarvis, hi")
    assert calls.sources == ["wake"]
    assert calls.speakers == [None]
    # Even long after the wake word (a long request): its transcript is
    # still the wake word's call.
    later(t)
    await event(t, {"InputWake": {"word": "Jarvis", "score": 0.9}})
    await eventually(lambda: len(calls.sources) == 2)
    later(t)
    await event(t, done("Jarvis, a long one", True, True))
    await eventually(lambda: t.session._heard == "Jarvis, a long one")
    assert calls.sources == ["wake", "wake"]
    # No wake word spotted, but the transcript names the bot (it comes
    # later): who said it goes along.
    later(t)
    await event(t, done("Ann: jarvis come", True, True, "Ann"))
    await eventually(lambda: len(calls.sources) == 3)
    assert (calls.sources[-1], calls.speakers[-1]) == ("transcript", "Ann")
    # Talk not for the bot.
    later(t)
    await event(t, done("hi Bob", False, False))
    # One other person: everything wants a reply, nobody calls the bot.
    await t.session.set_people(1)
    await event(t, done("come", True, False))
    await eventually(lambda: t.session._heard == "come")
    assert len(calls.sources) == 3
    await t.session.close("done")


@pytest.mark.asyncio
async def test_the_next_wake_word_before_this_transcript_is_one_turn(engine):
    calls = Calls()
    t = await open_session(engine, private=False, on_wake=calls)
    # Utterance 1 called by name in its text; utterance 2's wake word is
    # heard before utterance 1's transcript comes: one turn, to the newer.
    await event(t, {"InputWake": {"word": "Jarvis", "score": 0.9}})
    await event(t, done("jarvis look", True, True, "Ann"))
    await event(t, done("Jarvis, here", True, True, "Bob"))
    await eventually(lambda: t.session._heard == "Jarvis, here")
    assert calls.sources == ["wake"]
    # Called again within a few seconds (by its transcript): no new turn.
    await event(t, done("jarvis, and", True, True))
    await eventually(lambda: t.session._heard == "jarvis, and")
    assert calls.sources == ["wake"]
    # Going on with the exchange (a reply wanted, not called): a turn,
    # once a while has passed.
    later(t)
    await event(t, done("Bob: and then?", True, False, "Bob"))
    await eventually(lambda: len(calls.sources) == 2)
    assert (calls.sources[-1], calls.speakers[-1]) == ("transcript", "Bob")
    await t.session.close("done")


@pytest.mark.asyncio
async def test_an_on_wake_without_speaker_still_gets_called(engine):
    sources: list[str] = []

    def older(source: str) -> None:  # a platform's from before ``speaker``
        sources.append(source)

    t = await open_session(engine, private=False, on_wake=older)
    await event(t, done("Ann: jarvis come", True, True, "Ann"))
    await eventually(lambda: sources == ["transcript"])
    await t.session.close("done")


@pytest.mark.asyncio
async def test_a_failing_on_wake_does_not_end_the_conversation(engine):
    calls = Calls(fail=True)
    t = await open_session(engine, private=False, on_wake=calls)
    await event(t, {"InputWake": {"word": "Jarvis", "score": 0.9}})
    await eventually(lambda: calls.sources == ["wake"])

    def broken(source: str) -> None:
        raise RuntimeError("no")

    t.session._on_wake = broken
    later(t)
    await event(t, {"InputTranscriptDone": {"text": "x", "respond": True}})
    later(t)
    await event(t, {"InputWake": {"word": "Jarvis", "score": 0.9}})
    await event(t, {"InputTranscriptDone": {"text": "still here"}})
    await eventually(lambda: t.session._heard == "still here")
    assert not t.session.closing
    await t.session.close("done")


@pytest.mark.asyncio
async def test_a_task_names_the_guessed_speaker(engine):
    t = await open_session(engine, private=False)
    await event(
        t,
        {
            "InputTranscriptDone": {
                "text": "Alice: Jarvis, what time is it",
                "respond": True,
                "called": True,
                "speaker": "Alice",
            }
        },
    )
    await eventually(lambda: t.session._speaker == "Alice")
    await t.pump.tool_handler(
        {"tool": "backend_task", "callId": "c1", "arguments": {"task": "Tell the time"}}
    )
    await eventually(lambda: t.chat.asked)
    assert t.chat.speakers == ["Alice"]
    assert "Alice: Jarvis, what time is it" in t.chat.asked[0]
    await t.session.close("done")


def test_codex_realtime_takes_no_speaker_labels(monkeypatch):
    monkeypatch.setattr(
        voice, "_runner_config", lambda: runner_config(backend="builtin")
    )
    session = new_voice_session(
        key="k",
        scope_id="s",
        prompt="p",
        options=VoiceOptions(name="J", aliases=[]),
        media=FakeMedia(),
        on_closed=lambda s: None,
        chat=FakeChat(False),
        on_wake=lambda source: None,
    )
    assert session.label_speaker("Alice", 0, 960, True, bearing_deg=3.0) is None
