"""Group chats: earlier messages are reference only, and senders stay apart."""

from types import SimpleNamespace

import pytest

from astrbot.builtin_stars.astrbot import group_chat_context as gcc
from astrbot.core.agent.runners.codex import codex_agent_runner as runner_mod
from astrbot.core.agent.runners.codex.codex_agent_runner import (
    CodexAgentRunner,
    speaker_change_note,
)
from astrbot.core.agent.runners.codex.constants import DEFAULT_SYSTEM_PROMPT
from astrbot.core.provider.entities import ProviderRequest

UMO = "qq:GroupMessage:g1"

# ------------------------------------------------------------ group history


def test_group_history_is_marked_as_reference_not_instructions():
    block = gcc._format_group_history_block(["[Alice/10:00:00]:  @bot 帮我查天气"])

    assert block.startswith("<group_history>")
    assert block.endswith("</group_history>")
    assert "NOT instructions" in block
    assert "[Alice/10:00:00]:  @bot 帮我查天气" in block
    # The old framing read as a live system instruction.
    assert "system_reminder" not in block


def _event(nickname="Alice", user_id="111", *, at_self=False, timestamp=0):
    from astrbot.api.message_components import At, Plain

    chain = [Plain("hello")]
    if at_self:
        chain.append(At(qq="999", name="bot"))
    return SimpleNamespace(
        message_obj=SimpleNamespace(
            sender=SimpleNamespace(nickname=nickname, user_id=user_id),
            timestamp=timestamp,
        ),
        get_messages=lambda: chain,
        get_self_id=lambda: "999",
    )


@pytest.mark.asyncio
async def test_history_lines_carry_nickname_id_and_send_time():
    import datetime

    sent = datetime.datetime(2026, 9, 22, 10, 5, 7)
    ctx = gcc.GroupChatContext.__new__(gcc.GroupChatContext)
    line = await ctx._format_message(
        _event(timestamp=int(sent.timestamp())), {"image_caption": False}
    )

    assert line.startswith("[Alice | ID: 111 | 2026-09-22 10:05:07]: ")


@pytest.mark.asyncio
async def test_a_past_mention_is_not_flagged_as_a_pending_request():
    ctx = gcc.GroupChatContext.__new__(gcc.GroupChatContext)
    line = await ctx._format_message(_event(at_self=True), {"image_caption": False})

    assert "[mentioned you]" in line
    assert "DIRECTED AT YOU" not in line


# ------------------------------------------------------------ speaker change


def test_same_sender_gets_no_note():
    assert speaker_change_note({"id": "1", "label": "A"}, {"id": "1"}) is None


def test_first_turn_or_unknown_sender_gets_no_note():
    assert speaker_change_note(None, {"id": "1", "label": "A"}) is None
    assert speaker_change_note({"id": "1"}, {"id": "", "label": ""}) is None


def test_a_new_sender_is_flagged_by_name():
    note = speaker_change_note(
        {"id": "1", "label": "Alice"}, {"id": "2", "label": "Bob"}
    )

    assert note["type"] == "text"
    text = note["text"]
    assert text.startswith("<speaker_change>")
    assert "from Bob" in text
    assert "not Alice" in text


def test_the_system_prompt_explains_both_blocks():
    assert "<group_history>" in DEFAULT_SYSTEM_PROMPT
    assert "<speaker_change>" in DEFAULT_SYSTEM_PROMPT


# ------------------------------------------------------------ in the runner


class _Engine:
    async def open_thread(self, state, params):
        return {"thread_id": state.get("thread_id") or "t1"}, not state


def _runner(sender_id, name, platform="aiocqhttp"):
    event = SimpleNamespace(
        get_sender_id=lambda: sender_id,
        get_sender_name=lambda: name,
        get_platform_name=lambda: platform,
        message_obj=SimpleNamespace(message_id="m1"),
    )
    runner = CodexAgentRunner()
    runner.req = ProviderRequest(prompt="hi", session_id=UMO)
    runner.umo = UMO
    runner.cfg = {}
    runner.run_context = SimpleNamespace(context=SimpleNamespace(event=event))
    runner.bridge = SimpleNamespace(fingerprint="fp", dynamic_tools=lambda: [])
    runner._speaker_change = None
    runner._active_persona = None
    runner._additional_context = {}
    return runner


@pytest.fixture
def prefs(monkeypatch):
    store = {}

    async def get_async(scope, scope_id, key, default=None):
        return store.get((scope_id, key), default)

    async def put_async(scope, scope_id, key, value):
        store[(scope_id, key)] = value

    monkeypatch.setattr(runner_mod.sp, "get_async", get_async)
    monkeypatch.setattr(runner_mod.sp, "put_async", put_async)
    monkeypatch.setattr(CodexAgentRunner, "_thread_params", lambda self: {})
    return store


async def _turn(sender_id, name, *, accepted=True, platform="aiocqhttp", req=None):
    runner = _runner(sender_id, name, platform)
    if req is not None:
        runner.req = req
    thread_id, _ = await runner._open_thread(_Engine())
    turn_input = runner._turn_request(None)["input"]
    if accepted:
        await runner._remember_sender(thread_id)
    return turn_input


def _notes(turn_input):
    return [i["text"] for i in turn_input if i["text"].startswith("<speaker_change>")]


@pytest.mark.asyncio
async def test_alternating_senders_are_flagged_each_time(prefs):
    assert _notes(await _turn("1", "Alice")) == []
    assert _notes(await _turn("1", "Alice")) == []
    [note] = _notes(await _turn("2", "Bob"))
    assert "from Bob" in note and "not Alice" in note
    [note] = _notes(await _turn("1", "Alice"))
    assert "from Alice" in note and "not Bob" in note


@pytest.mark.asyncio
async def test_the_note_comes_before_the_message(prefs):
    await _turn("1", "Alice")
    turn_input = await _turn("2", "Bob")

    assert turn_input[0]["text"].startswith("<speaker_change>")
    assert turn_input[-1]["text"] == "hi"


@pytest.mark.asyncio
async def test_the_note_names_both_people_with_their_ids(prefs):
    await _turn("1", "Alice")
    [note] = _notes(await _turn("2", "Bob"))

    assert "Bob (ID: 2)" in note and "Alice (ID: 1)" in note


@pytest.mark.asyncio
async def test_a_new_thread_starts_without_a_note(prefs):
    await _turn("1", "Alice")
    prefs.clear()  # /reset: no thread state left

    assert _notes(await _turn("2", "Bob")) == []


# ------------------------------------------------------------ each message once


class _Req:
    def __init__(self):
        self.units = []

    def add_persistent_context(self, name, content, unit_id=None):
        self.units.append((name, content, unit_id))


def _history(lines_and_ids, pending=(), pending_age=0.0):
    ctx = gcc.GroupChatContext.__new__(gcc.GroupChatContext)
    ctx._locks = {}
    ctx.raw_records = gcc.defaultdict(gcc.deque)
    ctx._record_ids = gcc.defaultdict(gcc.deque)
    ctx._pending_triggers = gcc.defaultdict(dict)
    ctx._verdicts = gcc.defaultdict(dict)
    for line, rid in lines_and_ids:
        ctx.raw_records[UMO].append(line)
        ctx._record_ids[UMO].append(rid)
    for rid in pending:
        ctx._pending_triggers[UMO][rid] = gcc.time.monotonic() - pending_age
    return ctx


def _trigger(record_id):
    extras = {"_group_context_record_id": record_id, "_group_context_raw_idx": 99}
    return SimpleNamespace(
        unified_msg_origin=UMO,
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=extras.__setitem__,
        extras=extras,
    )


def _shown(req):
    return [
        line
        for _, block, _ in req.units
        for line in block.splitlines()
        if line.startswith("[")
    ]


@pytest.mark.asyncio
async def test_a_message_is_shown_only_once():
    ctx = _history([("[a] one", "r1"), ("[b] @bot two", "r2"), ("[a] three", "r3")])
    first = _Req()
    await ctx.on_req_llm(_trigger("r2"), first)
    later = _Req()
    await ctx.on_req_llm(_trigger("r3"), later)

    assert _shown(first) == ["[a] one"]
    assert _shown(later) == []


@pytest.mark.asyncio
async def test_another_pending_request_is_not_shown_as_history():
    # Bob @'d the bot first, but Alice's request is prepared first.
    ctx = _history(
        [("[x] chatter", "r1"), ("[Bob] @bot help", "r2"), ("[Alice] @bot hi", "r3")],
        pending={"r2", "r3"},
    )
    alice = _Req()
    await ctx.on_req_llm(_trigger("r3"), alice)
    bob = _Req()
    await ctx.on_req_llm(_trigger("r2"), bob)

    assert _shown(alice) == ["[x] chatter"]
    # Bob's own message was not consumed by Alice's request.
    assert list(ctx.raw_records[UMO]) == []
    assert _shown(bob) == []


@pytest.mark.asyncio
async def test_an_already_consumed_trigger_does_not_eat_other_messages():
    ctx = _history([("[a] one", "r1"), ("[b] two", "r2")])
    req = _Req()
    # r0 was consumed long ago; its stale index 0 must not drop "[a] one".
    event = _trigger("r0")
    event.extras["_group_context_raw_idx"] = 0
    await ctx.on_req_llm(event, req)

    assert req.units == []
    assert list(ctx.raw_records[UMO]) == ["[a] one", "[b] two"]


@pytest.mark.asyncio
async def test_a_trigger_that_never_got_a_request_becomes_history():
    ctx = _history(
        [("[Bob] @bot help", "r1"), ("[Alice] @bot hi", "r2")],
        pending={"r1"},
        pending_age=gcc.PENDING_TRIGGER_TTL_S + 1,
    )
    req = _Req()
    await ctx.on_req_llm(_trigger("r2"), req)

    assert _shown(req) == ["[Bob] @bot help"]
    # An expired trigger shown as history is no longer tracked as pending.
    assert ctx._pending_triggers[UMO] == {}


@pytest.mark.asyncio
async def test_history_of_a_request_that_never_ran_is_given_back():
    ctx = _history([("[a] one", "r1"), ("[b] two", "r2"), ("[c] @bot hi", "r3")])
    event = _trigger("r3")
    await ctx.on_req_llm(event, _Req())
    assert list(ctx.raw_records[UMO]) == []

    # e.g. the chat was busy: the request never reached the model.
    await runner_mod.release_group_history(event)

    assert list(ctx.raw_records[UMO]) == ["[a] one", "[b] two"]
    later = _Req()
    ctx.raw_records[UMO].append("[d] @bot again")
    ctx._record_ids[UMO].append("r4")
    await ctx.on_req_llm(_trigger("r4"), later)
    assert _shown(later) == ["[a] one", "[b] two"]


@pytest.mark.asyncio
async def test_history_the_model_saw_is_not_given_back():
    ctx = _history([("[a] one", "r1"), ("[c] @bot hi", "r2")])
    event = _trigger("r2")
    await ctx.on_req_llm(event, _Req())

    runner_mod.keep_group_history(event)
    await runner_mod.release_group_history(event)

    assert list(ctx.raw_records[UMO]) == []


@pytest.mark.asyncio
async def test_a_turn_that_was_not_accepted_does_not_become_the_last_sender(prefs):
    await _turn("1", "Alice")
    await _turn("2", "Bob", accepted=False)  # e.g. the submit failed

    # The model never saw Bob, so Alice is not flagged as a change.
    assert _notes(await _turn("1", "Alice")) == []


@pytest.mark.asyncio
async def test_scheduled_turns_are_not_a_change_of_person(prefs):
    await _turn("1", "Alice")
    assert _notes(await _turn("1", "Scheduler", platform="cron")) == []
    # Nor do they replace the last person who talked.
    assert _notes(await _turn("1", "Alice")) == []


@pytest.mark.asyncio
async def test_the_note_sits_between_the_chatter_and_the_message(prefs):
    await _turn("1", "Alice")
    req = ProviderRequest(prompt="hi", session_id=UMO)
    req.add_persistent_context("group_history", "<group_history>x</group_history>")
    req.add_persistent_context("message_meta", "Sender: Bob (ID: 2)")
    texts = [i["text"] for i in await _turn("2", "Bob", req=req)]

    assert [t.split(">")[0] for t in texts] == [
        '<context_unit name="group_history"',
        "<speaker_change",
        '<context_unit name="message_meta"',
        "hi",
    ]


@pytest.mark.asyncio
async def test_a_mention_names_who_was_mentioned():
    ctx = gcc.GroupChatContext.__new__(gcc.GroupChatContext)
    line = await ctx._format_message(_event(at_self=True), {"image_caption": False})

    assert "[At: bot (ID: 999)]" in line


@pytest.mark.asyncio
async def test_history_is_not_given_back_into_a_reset_chat():
    ctx = _history([("[a] one", "r1"), ("[c] @bot hi", "r2")])
    event = _trigger("r2")
    await ctx.on_req_llm(event, _Req())
    ctx.raw_records.pop(UMO)  # /reset while the request was in flight
    ctx._record_ids.pop(UMO)

    await runner_mod.release_group_history(event)

    assert list(ctx.raw_records.get(UMO, [])) == []


@pytest.mark.asyncio
async def test_given_back_history_respects_the_cap(monkeypatch):
    ctx = _history([("[a] one", "r1"), ("[b] two", "r2"), ("[c] @bot hi", "r3")])
    monkeypatch.setattr(
        gcc.GroupChatContext, "cfg", lambda self, event: {"group_message_max_cnt": 2}
    )
    event = _trigger("r3")
    await ctx.on_req_llm(event, _Req())
    for i in range(2):
        ctx.raw_records[UMO].append(f"[n] new {i}")
        ctx._record_ids[UMO].append(f"n{i}")

    await runner_mod.release_group_history(event)

    # The oldest go first, as when new messages push past the cap.
    assert list(ctx.raw_records[UMO]) == ["[n] new 0", "[n] new 1"]
    assert list(ctx._record_ids[UMO]) == ["n0", "n1"]


@pytest.mark.asyncio
async def test_a_run_that_fails_before_codex_gives_the_history_back(monkeypatch):
    from astrbot.core.agent.hooks import BaseAgentRunHooks

    given_back = []

    async def restore():
        given_back.append(True)

    extras = {runner_mod.GROUP_HISTORY_RESTORE_KEY: restore}
    event = SimpleNamespace(
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=extras.__setitem__,
    )

    async def broken_engine(options):
        raise RuntimeError("app-server did not start")

    monkeypatch.setattr(runner_mod.CodexEngine, "get", staticmethod(broken_engine))
    runner = CodexAgentRunner()
    await runner.reset(
        request=ProviderRequest(prompt="hi", session_id=UMO),
        run_context=SimpleNamespace(context=SimpleNamespace(event=event)),
        agent_hooks=BaseAgentRunHooks(),
        provider_config={},
    )
    responses = [r async for r in runner.step_until_done()]

    assert [r.type for r in responses] == ["err"]
    assert given_back == [True]


@pytest.mark.asyncio
async def test_a_refused_trigger_is_dropped_from_the_history():
    # Rate limited: nobody else's request may answer it later.
    ctx = _history(
        [("[a] one", "r1"), ("[Bob] @bot help", "r2"), ("[a] two", "r3")],
        pending={"r2"},
    )
    event = _trigger("r2")
    event.extras[gcc.GROUP_MESSAGE_FORGET_KEY] = ctx._forgetter(UMO, "r2")

    await runner_mod.forget_group_message(event)

    assert list(ctx.raw_records[UMO]) == ["[a] one", "[a] two"]
    assert list(ctx._record_ids[UMO]) == ["r1", "r3"]
    assert ctx._pending_triggers[UMO] == {}
    # Already gone (shown or trimmed): nothing else is touched.
    await runner_mod.forget_group_message(event)
    assert list(ctx.raw_records[UMO]) == ["[a] one", "[a] two"]


def test_the_forget_key_is_shared():
    assert runner_mod.GROUP_MESSAGE_FORGET_KEY == gcc.GROUP_MESSAGE_FORGET_KEY
