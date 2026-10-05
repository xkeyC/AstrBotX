import asyncio

import pytest
import pytest_asyncio

from astrbot.core.config.agent_runner import get_agent_runner_config_default
from astrbot.core.platform.sources.mumble import voice as mumble_voice
from astrbot.core.platform.sources.mumble.audio import MumbleMedia
from astrbot.core.voice import chat as chat_module
from astrbot.core.voice import record as record_module
from astrbot.core.voice import session as voice
from astrbot.core.voice.chat import VoiceChat
from astrbot.core.voice.session import VoiceOptions, VoiceSession


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.roles: list[str] = []

    async def realtime_start(self, thread_id, request):
        self.calls.append("start")

    async def realtime_append_text(self, thread_id, text, role="user"):
        self.calls.append(f"text:{text}")
        self.roles.append(role)

    async def realtime_append_speech(self, thread_id, text):
        self.calls.append(f"speech:{text}")

    async def realtime_stop(self, thread_id):
        self.calls.append("stop")


class FakePump:
    def __init__(self) -> None:
        self.route = None

    def open_turn(self, tool_handler, approval_handler):
        queue: asyncio.Queue = asyncio.Queue()

        class Route:
            events = queue

        self.route = Route()
        return queue

    def close_turn(self):
        self.route = None


class FakeEngine:
    def __init__(self, gate: asyncio.Event) -> None:
        self.rt = FakeRuntime()
        self.gate = gate
        self.pumps: dict[str, FakePump] = {}
        self.forgotten: list[str] = []
        self.locks: dict[str, asyncio.Lock] = {}

    def session_lock(self, key):
        return self.locks.setdefault(key, asyncio.Lock())

    async def open_thread(self, state, params):
        await self.gate.wait()
        return {"thread_id": "t1", "rollout_path": None}, True

    def pump(self, thread_id):
        return self.pumps.setdefault(thread_id, FakePump())

    async def forget_thread(self, thread_id):
        self.forgotten.append(thread_id)


class FakeChat(VoiceChat):
    """The real ordering and busy logic; the chat's turns are faked."""

    def __init__(self) -> None:
        super().__init__(umo="test:FriendMessage:1", private=True)
        self.asked: list[str] = []
        self.answer: str | None = "It is three."
        self.is_busy = False
        self.persona = ""
        self.release = asyncio.Event()
        self.release.set()

    def busy(self) -> bool:
        return self.is_busy or super().busy()

    async def voice_persona(self) -> str:
        return self.persona

    async def ask(self, body: str) -> str | None:
        self.asked.append(body)
        await self.release.wait()
        return self.answer


async def settle() -> None:
    """Lets requests handed to the chat finish."""
    for _ in range(5):
        await asyncio.gather(*chat_module._REQUESTS)
        await asyncio.sleep(0)


class FakeSp:
    async def get_async(self, **_kwargs):
        return {}

    async def put_async(self, **_kwargs):
        return None


@pytest_asyncio.fixture
async def voice_db(monkeypatch, temp_db):
    """The voice records' database, ready (its first use is slow)."""
    async with temp_db.get_db():
        pass
    monkeypatch.setattr(record_module, "db_helper", temp_db)
    return temp_db


@pytest.fixture
def engine(monkeypatch, voice_db):
    gate = asyncio.Event()
    engine = FakeEngine(gate)

    async def codex_engine():
        return engine

    monkeypatch.setattr(voice, "_codex_engine", codex_engine)
    monkeypatch.setattr(voice, "sp", FakeSp())
    monkeypatch.setattr(
        voice,
        "_runner_config",
        lambda: get_agent_runner_config_default("codex"),
    )
    return engine


def make_session(closed: list) -> VoiceSession:
    return VoiceSession(
        key="server",
        scope_id="test:voice:server",
        prompt="p",
        options=VoiceOptions(name="Jarvis", aliases=[]),
        media=MumbleMedia(lambda frame, end: None),
        on_closed=closed.append,
        chat=FakeChat(),
    )


@pytest.mark.asyncio
async def test_close_during_start_leaves_nothing_running(engine):
    closed: list = []
    failures: list = []
    session = make_session(closed)
    session.launch(failures.append)
    await asyncio.sleep(0)  # the start is now waiting in open_thread

    closing = asyncio.create_task(session.close("disconnected"))
    await asyncio.sleep(0)
    assert not closing.done()  # waits for the start to reach a safe point
    engine.gate.set()
    await asyncio.wait_for(closing, 5)

    assert engine.rt.calls == []  # no realtime call was ever started
    assert engine.forgotten == ["t1"]  # the thread it opened is released
    assert closed == [session]
    assert failures == []
    await session.close("again")
    assert closed == [session]


@pytest.mark.asyncio
async def test_start_failure_is_reported_once(engine, monkeypatch):
    async def broken(realtime=True):
        raise RuntimeError("binding has no realtime support")

    monkeypatch.setattr(voice, "_codex_engine", broken)
    closed: list = []
    failures: list = []
    session = make_session(closed)
    session.launch(failures.append)
    for _ in range(10):
        await asyncio.sleep(0)
    assert [str(e) for e in failures] == ["binding has no realtime support"]
    assert closed == [session]


@pytest.mark.asyncio
async def test_the_voice_thread_only_carries_the_conversation(engine):
    captured = {}

    async def open_thread(state, params):
        captured.update(params)
        return {"thread_id": "t1", "rollout_path": None}, True

    engine.open_thread = open_thread
    session = make_session([])
    await session._open_agent()
    assert captured["dynamic_tools"] == []
    assert captured["no_environment"] is True
    config = captured["config"]
    assert config["realtime.host_routes_handoffs"] is True
    assert config["features.memories"] is False


def handoff(text: str) -> dict:
    return {
        "handoff_id": "h1",
        "item_id": "i1",
        "input_transcript": text,
        "active_transcript": [
            {"role": "user", "text": "what time is it"},
            {"role": "assistant", "text": "let me check"},
        ],
    }


@pytest.mark.asyncio
async def test_a_handoff_is_answered_by_the_chat_and_spoken(engine):
    session = make_session([])
    session._engine, session._thread_id = engine, "t1"
    await session._handoff(handoff("look up the time"))
    await settle()
    (body,) = session.chat.asked
    assert "look up the time" in body and "what time is it" in body
    assert engine.rt.calls == ['speech:Answer to "look up the time": It is three.']


@pytest.mark.asyncio
async def test_a_busy_chat_is_announced_and_requests_keep_their_order(engine):
    session = make_session([])
    session._engine, session._thread_id = engine, "t1"
    session.chat.is_busy = True
    session.chat.release.clear()
    await session._handoff(handoff("first"))
    session.chat.is_busy = False
    # The first request still waits: the second is told it is busy too.
    await session._handoff(handoff("second"))
    await asyncio.sleep(0.05)
    assert engine.rt.calls == [f"speech:{voice.BUSY_SPEECH}"] * 2
    session.chat.release.set()
    await settle()
    assert [b.split("Task: ")[1] for b in session.chat.asked] == ["first", "second"]
    assert engine.rt.calls[2:] == [
        'speech:Answer to "first": It is three.',
        'speech:Answer to "second": It is three.',
    ]


@pytest.mark.asyncio
async def test_failed_and_empty_answers_are_told_as_such(engine):
    session = make_session([])
    session._engine, session._thread_id = engine, "t1"
    session.chat.answer = None
    await session._handoff(handoff("x"))
    await settle()
    session.chat.answer = ""
    await session._handoff(handoff("y"))
    await settle()
    assert engine.rt.calls == [
        f"speech:{voice.FAILED_SPEECH}",
        f"speech:{voice.DONE_SPEECH}",
    ]


@pytest.mark.asyncio
async def test_a_waited_answer_counts_as_activity(engine):
    session = make_session([])
    session._engine, session._thread_id = engine, "t1"
    session.chat.release.clear()
    await session._handoff(handoff("x"))
    await asyncio.sleep(0.05)
    before = session.last_activity
    await asyncio.sleep(0.05)
    assert session.last_activity > before  # still waiting: active now
    session.chat.release.set()
    await settle()
    assert session.last_answer_at > 0 and session._pending == 0


@pytest.mark.asyncio
async def test_an_answer_after_the_session_closed_is_posted(engine, monkeypatch):
    posted = []

    async def fake_deliver(umo, answer):
        posted.append((umo, answer))

    monkeypatch.setattr(voice, "deliver", fake_deliver)
    session = make_session([])
    session._engine, session._thread_id = engine, "t1"
    session.chat.release.clear()
    await session._handoff(handoff("x"))
    await session.close("hung up")
    session.chat.release.set()  # the request was not cancelled by the close
    await settle()
    assert posted == [("test:FriendMessage:1", "It is three.")]
    assert "speech:It is three." not in engine.rt.calls


@pytest.mark.asyncio
async def test_a_note_goes_to_the_model_as_developer_context(engine):
    session = make_session([])
    session._engine, session._thread_id = engine, "t1"
    await session.note("The backup finished.")
    (call,) = engine.rt.calls
    assert call.startswith("text:") and "The backup finished." in call
    assert engine.rt.roles == ["developer"]


@pytest.mark.asyncio
async def test_close_while_waiting_for_answer_is_prompt(engine):
    engine.gate.set()
    closed: list = []
    session = make_session(closed)
    session.launch(lambda exc: None)
    for _ in range(1000):  # wait until the realtime start was requested
        if engine.rt.calls:
            break
        await asyncio.sleep(0.01)
    assert engine.rt.calls == ["start"]
    loop = asyncio.get_running_loop()
    began = loop.time()
    await asyncio.wait_for(session.close("standby"), 5)
    assert loop.time() - began < 2  # not the 30 s answer timeout
    assert engine.rt.calls == ["start", "stop"]
    assert engine.forgotten == ["t1"]
    assert closed == [session]


@pytest.mark.asyncio
async def test_cancelled_close_still_releases(engine):
    engine.gate.set()
    closed: list = []
    session = make_session(closed)
    session.launch(lambda exc: None)
    for _ in range(1000):
        if engine.rt.calls:
            break
        await asyncio.sleep(0.01)
    caller = asyncio.create_task(session.close("disconnected"))
    await asyncio.sleep(0)
    caller.cancel()
    for _ in range(300):
        if closed:
            break
        await asyncio.sleep(0.01)
    assert closed == [session]
    assert engine.rt.calls == ["start", "stop"]
    await session.close("again")  # already released: returns at once


@pytest.mark.asyncio
async def test_start_slower_than_close_wait_is_released_later(engine, monkeypatch):
    monkeypatch.setattr(voice, "CLOSE_WAIT", 0.05)
    closed: list = []
    session = make_session(closed)
    session.launch(lambda exc: None)
    await asyncio.sleep(0)  # the start is stuck opening the thread
    await asyncio.wait_for(session.close("standby"), 5)
    assert closed == [session]
    assert engine.forgotten == []  # no thread yet
    engine.gate.set()  # the thread opens after the close gave up waiting
    for _ in range(100):
        if engine.forgotten:
            break
        await asyncio.sleep(0.01)
    assert engine.forgotten == ["t1"]  # released once the start returned
    assert engine.rt.calls == []


@pytest.mark.asyncio
async def test_late_start_cannot_unload_a_newer_sessions_thread(engine, monkeypatch):
    monkeypatch.setattr(voice, "CLOSE_WAIT", 0.05)
    old_closed: list = []
    old = make_session(old_closed)
    old.launch(lambda exc: None)
    await asyncio.sleep(0)  # stuck opening thread t1
    await asyncio.wait_for(old.close("muted"), 5)
    new_closed: list = []
    new = make_session(new_closed)  # same key, same persisted thread
    new.launch(lambda exc: None)
    await asyncio.sleep(0)
    engine.gate.set()
    for _ in range(1000):
        if engine.rt.calls:
            break
        await asyncio.sleep(0.01)
    # The old start unloaded t1 before the new one could open it; the new
    # session then opened (resumed) it and kept it.
    assert engine.forgotten == ["t1"]
    assert engine.rt.calls == ["start"]
    assert new_closed == []
    await new.close("done")
    assert engine.forgotten == ["t1", "t1"]


def test_consent_expiry_is_disabled():
    import aioice.ice

    assert aioice.ice.CONSENT_FAILURES >= 1_000_000


def test_realtime_prompts_name_the_bot_and_the_session_adds_the_date():
    import datetime

    options = voice.VoiceOptions(name="Jarvis", aliases=["贾维斯"])
    today = datetime.datetime.now().astimezone().strftime("%Y-%m-%d")
    assert '("Jarvis", "贾维斯")' in mumble_voice.channel_prompt(options, gated=False)
    assert '("Jarvis", "贾维斯")' in mumble_voice.channel_prompt(options, gated=True)
    assert "alice" in mumble_voice.whisper_prompt(options, "alice")
    # The date goes with each conversation's start (not in the prompt kept
    # by the thread).
    assert f"Today is {today}" in voice.time_prompt()
    assert "must be delegated to the backend" in voice.time_prompt()


@pytest.mark.asyncio
async def test_say_needs_a_ready_session(engine):
    session = make_session([])
    with pytest.raises(RuntimeError):
        await session.say("hello")
    session.ready = True
    session._engine = engine
    session._thread_id = "t1"
    await session.say("hello")
    assert engine.rt.calls == ["text:hello"]


def test_close_stops_the_media(engine):
    class Media:
        track = None
        stopped = 0

        async def play(self, track):
            return None

        def start(self):
            return None

        def stop(self):
            self.stopped += 1

    media = Media()
    session = VoiceSession(
        key="k",
        scope_id="s",
        prompt="p",
        options=VoiceOptions(name="n", aliases=[]),
        media=media,
        on_closed=lambda s: None,
        chat=FakeChat(),
    )

    async def run():
        await session.close("a")
        await session.close("b")

    asyncio.run(run())
    assert media.stopped == 1


@pytest.mark.asyncio
async def test_the_voice_persona_completes_the_prompt(engine, monkeypatch):
    for persona, extra, expected in (
        ("Speak like a pirate.", "platform extra", "p\n\nSpeak like a pirate."),
        ("", "platform extra", "p\n\nplatform extra"),
        ("", "", "p"),
    ):
        session = VoiceSession(
            key="k",
            scope_id="s",
            prompt="p",
            options=VoiceOptions(name="n", aliases=[], extra_prompt=extra),
            media=MumbleMedia(lambda frame, end: None),
            on_closed=lambda s: None,
            chat=FakeChat(),
        )
        session.chat.persona = persona

        async def stop_here():
            raise RuntimeError("stop")

        monkeypatch.setattr(session, "_open_agent", stop_here)
        with pytest.raises(RuntimeError):
            await session._start()
        assert session.prompt == expected


@pytest.mark.asyncio
async def test_a_quick_call_back_keeps_the_thread(engine, monkeypatch):
    class SlowSp(FakeSp):
        async def put_async(self, **_kwargs):
            await asyncio.sleep(0)  # the database write yields

    monkeypatch.setattr(voice, "sp", SlowSp())
    old = make_session([])
    old._engine, old._thread_id = engine, "t1"
    old._events_queue = engine.pump("t1").open_turn(None, None)
    new = make_session([])
    opening = asyncio.create_task(new._open_agent())
    await asyncio.sleep(0)  # the new session holds the lock in open_thread
    releasing = asyncio.create_task(old._release())
    await asyncio.sleep(0)
    engine.gate.set()
    await asyncio.wait_for(asyncio.gather(opening, releasing), 5)

    assert engine.forgotten == []  # the new session's thread stays loaded
    assert engine.pumps["t1"].route.events is new._events_queue


@pytest.mark.asyncio
async def test_the_model_ends_with_the_newest_context(engine):
    session = make_session([])
    session._engine, session._thread_id, session.ready = engine, "t1", True
    session._context = session._context_given = "(Room: A)"
    sent = []

    async def slow_append(thread_id, text, role="user"):
        await asyncio.sleep(0.01)
        sent.append((text, role))

    engine.rt.realtime_append_text = slow_append
    # Bob comes and goes while the first change is still on its way.
    await asyncio.gather(
        session.set_context("(Room: A, Bob)"), session.set_context("(Room: A)")
    )
    assert sent[-1] == ("(Room: A)", "context")
    assert session._context_given == "(Room: A)"


@pytest.mark.asyncio
async def test_context_set_before_the_start_waits_for_it(engine):
    session = make_session([])
    await session.set_context("(Room: A)")
    assert engine.rt.calls == []  # not yet: the start takes it along
    assert session._context == "(Room: A)"
