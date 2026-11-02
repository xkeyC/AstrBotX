"""生图之后：宿主直接把图发到会话，再复制进会话工作区供模型继续处理。

背景：Codex 把图存在 CODEX_HOME（宿主机上），模型的工具都够不着这个目录。
所以图一保存就由宿主发给用户，同时复制到工作区（沙箱运行时复制进沙箱），
并告诉模型"已发送"以及副本的位置。
"""

import asyncio
import json

import pytest

from astrbot.core.agent.runners.codex import codex_agent_runner as runner_mod
from astrbot.core.agent.runners.codex.codex_agent_runner import (
    GENERATED_IMAGE_DIR,
    CodexAgentRunner,
    generated_image_failure_note,
    generated_image_note,
)

UMO = "aiocqhttp:GroupMessage:30003"


def _runner(runtime: str = "sandbox", send=True):
    """`send`: what the platform's send_message returns, or an exception to raise."""
    runner = CodexAgentRunner.__new__(CodexAgentRunner)
    runner.cfg = {}
    runner.umo = UMO
    runner.sent = []
    runner._aborted = False
    runner._images_running = set()

    class _PluginCtx:
        @staticmethod
        def get_config(umo=None):
            return {"provider_settings": {"computer_use_runtime": runtime}}

        @staticmethod
        async def send_message(session, chain):
            if isinstance(send, Exception):
                raise send
            runner.sent.append((session, chain))
            return send

    class _Event:
        unified_msg_origin = UMO
        _has_send_oper = False

    class _AgentCtx:
        context = _PluginCtx()
        event = _Event()

    class _Wrapper:
        context = _AgentCtx()

    runner.run_context = _Wrapper()
    return runner


class _Engine:
    def __init__(self, status):
        self.status = status
        self.requests = []

    async def submit_turn(self, thread_id, request):
        self.requests.append(request)
        return {"status": self.status}


async def _no_steer(*args, **kwargs):
    return None


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "codex_home" / "generated_images" / "sess" / "call-1.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x89PNG fake")
    return path


# ------------------------------------------------------------------ 提示文案


def test_note_says_the_image_was_sent_and_where_the_copy_is():
    note = generated_image_note(
        "generated_images/call-1.png", "`generated_images/call-1.png`"
    )

    assert "generated_images/call-1.png" in note
    assert "already been sent" in note
    assert "generate it again" in note
    # 已经发过了，不要再引导模型重复发送
    assert "send_message_to_user" not in note


def test_note_after_a_failed_send_tells_the_model_how_to_send_it():
    note = generated_image_note(
        "generated_images/call-1.png", "`generated_images/call-1.png`", "boom"
    )

    assert "could NOT be sent" in note
    assert "boom" in note
    # 路径要能原样喂给 send_message_to_user
    assert '"path": "generated_images/call-1.png"' in note


def test_failure_note_after_a_send_only_says_the_copy_is_missing():
    note = generated_image_failure_note("sandbox unreachable")

    assert "sandbox unreachable" in note
    assert "already been sent" in note
    assert "could not be delivered" not in note


def test_failure_note_when_nothing_worked_says_it_cannot_be_delivered():
    note = generated_image_failure_note("sandbox unreachable", "no platform")

    assert "sandbox unreachable" in note
    assert "could not be delivered" in note


# ------------------------------------------------------------------ 放置


@pytest.mark.asyncio
async def test_sandbox_runtime_uploads_into_the_sandbox_workspace(monkeypatch, image):
    uploads: list[tuple[str, str]] = []

    class _Booter:
        async def upload_file(self, path, name):
            uploads.append((path, name))
            return {"success": True, "file_path": name}

    async def _get_booter(ctx, umo):
        return _Booter()

    import astrbot.core.computer.computer_client as client

    monkeypatch.setattr(client, "get_booter", _get_booter)

    relative, where = await _runner("sandbox")._place_generated_image(str(image))

    assert uploads == [(str(image), f"{GENERATED_IMAGE_DIR}/call-1.png")]
    assert relative == f"{GENERATED_IMAGE_DIR}/call-1.png"
    assert "/workspace/generated_images/call-1.png" in where


@pytest.mark.asyncio
async def test_an_upload_the_sandbox_reports_as_failed_is_a_failure(monkeypatch, image):
    """booter 用返回值报告失败（不抛异常）时，不能当成已放进沙箱。"""

    class _Booter:
        async def upload_file(self, path, name):
            return {"success": False, "error": "disk full"}

    async def _get_booter(ctx, umo):
        return _Booter()

    import astrbot.core.computer.computer_client as client

    monkeypatch.setattr(client, "get_booter", _get_booter)

    with pytest.raises(RuntimeError, match="disk full"):
        await _runner("sandbox")._place_generated_image(str(image))


@pytest.mark.asyncio
async def test_a_leftover_shipyard_key_does_not_override_the_runtime(
    monkeypatch, image, tmp_path
):
    """运行环境是唯一开关：残留的 shipyard_mode 不再把图片送进沙箱。"""
    workspace = tmp_path / "workspace"
    import astrbot.core.tools.computer_tools.util as util

    monkeypatch.setattr(util, "workspace_root", lambda umo: workspace)
    runner = _runner("none")
    runner.cfg = {"shipyard_mode": True}

    assert runner._computer_runtime() == "none"


@pytest.mark.asyncio
async def test_without_a_sandbox_the_image_lands_in_the_host_workspace(
    monkeypatch, image, tmp_path
):
    workspace = tmp_path / "workspace"
    import astrbot.core.tools.computer_tools.util as util

    monkeypatch.setattr(util, "workspace_root", lambda umo: workspace)

    relative, _ = await _runner("none")._place_generated_image(str(image))

    copied = workspace / GENERATED_IMAGE_DIR / "call-1.png"
    assert copied.read_bytes() == image.read_bytes()
    assert relative == f"{GENERATED_IMAGE_DIR}/call-1.png"


# ------------------------------------------------------------------ 发送 + 放置


@pytest.mark.asyncio
async def test_the_image_is_sent_to_the_chat_and_copied(monkeypatch, image, tmp_path):
    import astrbot.core.tools.computer_tools.util as util

    monkeypatch.setattr(util, "workspace_root", lambda umo: tmp_path / "ws")
    runner = _runner("none")

    note = await runner._deliver_generated_image(str(image))

    [(session, chain)] = runner.sent
    assert session == UMO
    [component] = chain.chain
    assert component.type == "Image"
    assert component.path == str(image)
    assert runner.run_context.context.event._has_send_oper is True
    assert (tmp_path / "ws" / GENERATED_IMAGE_DIR / "call-1.png").is_file()
    assert "already been sent" in note
    assert f"{GENERATED_IMAGE_DIR}/call-1.png" in note


@pytest.mark.asyncio
async def test_a_failed_send_still_places_the_copy(monkeypatch, image, tmp_path):
    import astrbot.core.tools.computer_tools.util as util

    monkeypatch.setattr(util, "workspace_root", lambda umo: tmp_path / "ws")
    runner = _runner("none", send=RuntimeError("napcat offline"))

    note = await runner._deliver_generated_image(str(image))

    assert (tmp_path / "ws" / GENERATED_IMAGE_DIR / "call-1.png").is_file()
    assert runner.run_context.context.event._has_send_oper is False
    assert "napcat offline" in note
    assert '"path": "generated_images/call-1.png"' in note


@pytest.mark.asyncio
async def test_a_chat_without_a_platform_counts_as_not_sent(
    monkeypatch, image, tmp_path
):
    import astrbot.core.tools.computer_tools.util as util

    monkeypatch.setattr(util, "workspace_root", lambda umo: tmp_path / "ws")
    runner = _runner("none", send=False)

    note = await runner._deliver_generated_image(str(image))

    assert "could NOT be sent" in note
    assert "send_message_to_user" in note


@pytest.mark.asyncio
async def test_a_failed_copy_still_sends_and_produces_a_note(monkeypatch, image):
    """复制失败时图照样发出，模型也必须收到说明，否则它会一直以为图在路上。"""

    async def _get_booter(ctx, umo):
        raise RuntimeError("sandbox unreachable")

    import astrbot.core.computer.computer_client as client

    monkeypatch.setattr(client, "get_booter", _get_booter)
    runner = _runner("sandbox")

    note = await runner._deliver_generated_image(str(image))

    assert len(runner.sent) == 1
    assert "sandbox unreachable" in note
    assert "already been sent" in note


@pytest.mark.asyncio
async def test_a_slow_sandbox_answers_now_and_steers_the_path_in_later(
    monkeypatch, image
):
    """沙箱冷启动可能超过 Codex 给回调的 60 秒：先回答，放好后再插进回合。"""
    release = asyncio.Event()

    class _Booter:
        async def upload_file(self, path, name):
            await release.wait()
            return {"success": True, "file_path": name}

    async def _get_booter(ctx, umo):
        return _Booter()

    import astrbot.core.computer.computer_client as client

    monkeypatch.setattr(client, "get_booter", _get_booter)
    monkeypatch.setattr(runner_mod, "PLACE_WAIT_SECONDS", 0.01)
    runner = _runner("sandbox")
    engine = _Engine("steered")
    runner._engine = engine
    runner._thread_id = "thread-1"
    runner._active = runner_mod.ActiveTurn(engine, "thread-1", "turn-1", "20017")

    note = await runner._deliver_generated_image(str(image))

    assert "already been sent" in note
    assert "still being placed" in note
    assert engine.requests == []

    release.set()
    for _ in range(50):
        if engine.requests:
            break
        await asyncio.sleep(0.01)

    steered = engine.requests[0]["input"][0]["text"]
    assert steered.startswith('<request_context name="generated_image">')
    assert "/workspace/generated_images/call-1.png" in steered


# ------------------------------------------------------------------ 插入回合


@pytest.mark.asyncio
async def test_the_note_is_steered_into_the_running_turn():
    engine = _Engine("steered")
    active = runner_mod.ActiveTurn(engine, "thread-1", "turn-1", "20017")

    ok = await _runner()._steer_note(engine, "thread-1", active, "NOTE")

    assert ok is True
    request = engine.requests[0]
    assert request["mode"] == "steer"
    assert request["expected_turn_id"] == "turn-1"
    assert request["input"][0]["text"] == "NOTE"


@pytest.mark.asyncio
async def test_a_turn_that_already_ended_reports_the_note_as_undelivered():
    """steer 被拒时返回 False，调用方会把提示留给回合结束后的续轮。"""
    engine = _Engine("not_steered")
    active = runner_mod.ActiveTurn(engine, "thread-1", "turn-1", "20017")

    assert await _runner()._steer_note(engine, "thread-1", active, "NOTE") is False


@pytest.mark.asyncio
async def test_no_turn_id_yet_means_no_steer():
    engine = _Engine("steered")
    active = runner_mod.ActiveTurn(engine, "thread-1", "", "20017")

    assert await _runner()._steer_note(engine, "thread-1", active, "NOTE") is False
    assert engine.requests == []


# ------------------------------------------------------------------ 单回调


@pytest.mark.asyncio
async def test_the_engine_hook_routes_to_the_thread_s_turn():
    """Codex 保存图片后回调引擎，引擎按 thread 交给正在跑的那一轮。"""
    from astrbot.core.agent.runners.codex.native import CodexEngine

    engine = CodexEngine(rt=object())
    calls = []

    async def _handler(call_id, saved_path):
        calls.append((call_id, saved_path))
        return "copied to generated_images/call-1.png"

    engine.saved_image_handlers["thread-1"] = _handler

    text = await engine._on_saved_image("thread-1", "call-1", "/codex/call-1.png")

    assert text == "copied to generated_images/call-1.png"
    assert calls == [("call-1", "/codex/call-1.png")]


@pytest.mark.asyncio
async def test_no_running_turn_keeps_codex_s_own_hint():
    from astrbot.core.agent.runners.codex.native import CodexEngine

    engine = CodexEngine(rt=object())

    assert await engine._on_saved_image("thread-9", "call-1", "/x.png") is None


@pytest.mark.asyncio
async def test_a_failing_handler_never_reaches_codex():
    """回调里抛异常只意味着退回 Codex 自带提示，不能把工具调用带崩。"""
    from astrbot.core.agent.runners.codex.native import CodexEngine

    engine = CodexEngine(rt=object())

    async def _boom(call_id, saved_path):
        raise RuntimeError("sandbox down")

    engine.saved_image_handlers["thread-1"] = _boom

    assert await engine._on_saved_image("thread-1", "call-1", "/x.png") is None


@pytest.mark.asyncio
async def test_the_runner_handler_places_the_image_and_marks_it_handled(
    monkeypatch, image, tmp_path
):
    """经回调处理过的图，事件流那条兜底路径不会再处理第二次。"""
    import astrbot.core.tools.computer_tools.util as util

    monkeypatch.setattr(util, "workspace_root", lambda umo: tmp_path / "ws")
    runner = _runner("none")
    runner._hooked_images = set()

    text = await runner._on_saved_image("call-1", str(image))

    assert str(image) in runner._hooked_images
    assert len(runner.sent) == 1
    assert "already been sent" in text
    assert f"{GENERATED_IMAGE_DIR}/call-1.png" in text


@pytest.mark.asyncio
async def test_the_hook_does_not_steer_a_running_exec(monkeypatch, image, tmp_path):
    """插入回合算用户输入，Codex 会让还在跑的 exec 提前返回，模型就看不到图了。"""
    import astrbot.core.tools.computer_tools.util as util

    monkeypatch.setattr(util, "workspace_root", lambda umo: tmp_path / "ws")
    engine = _Engine("steered")
    runner = _runner("none")
    runner._hooked_images = set()
    runner._engine = engine
    runner._thread_id = "thread-1"
    runner._active = runner_mod.ActiveTurn(engine, "thread-1", "turn-1", "20017")

    text = await runner._on_saved_image("call-1", str(image))
    await asyncio.sleep(0)

    assert "already been sent" in text
    assert engine.requests == []


def _late_runner(monkeypatch, tmp_path):
    """A finished turn's runner whose report only records what it would say."""
    import astrbot.core.tools.computer_tools.util as util

    from astrbot.core.agent.runners.codex import wake

    monkeypatch.setattr(util, "workspace_root", lambda umo: tmp_path / "ws")
    reports = asyncio.Queue()

    async def report(ctx, *, note, **kwargs):
        reports.put_nowait({**kwargs, "note": await note})

    monkeypatch.setattr(wake, "run_late_image_completion", report)
    runner = _runner("none")
    runner.run_context.context.event.role = "admin"
    runner.run_context.context.event.get_sender_id = lambda: "alice"
    runner.run_context.context.event.get_group_id = lambda: "30003"
    return runner, reports


@pytest.mark.asyncio
async def test_a_late_image_is_sent_and_reported_as_who_started_it(
    monkeypatch, image, tmp_path
):
    """回合结束后才生成好的图：照常发送，再以发起人的身份用事件告诉模型。"""
    runner, reports = _late_runner(monkeypatch, tmp_path)

    await runner.on_late_saved_image("call-1", str(image))
    report = await asyncio.wait_for(reports.get(), 2)

    assert len(runner.sent) == 1
    assert (tmp_path / "ws" / GENERATED_IMAGE_DIR / "call-1.png").is_file()
    assert report["session_str"] == UMO
    assert (report["sender_id"], report["role"]) == ("alice", "admin")
    assert report["group_id"] == "30003"
    assert report["note"].startswith("An image you started generating earlier has")
    assert f"{GENERATED_IMAGE_DIR}/call-1.png" in report["note"]
    # 副本的路径就在通知里，不用再等下一条
    assert "you will be told its path" not in report["note"]


@pytest.mark.asyncio
async def test_a_slow_copy_of_a_late_image_is_reported_when_done(
    monkeypatch, image, tmp_path
):
    """沙箱还在启动时先把图发出去，副本就绪后才通知模型，不受回调时限影响。"""
    runner, reports = _late_runner(monkeypatch, tmp_path)
    release = asyncio.Event()
    place = runner._place_generated_image

    async def slow_place(path):
        await release.wait()
        return await place(path)

    runner._place_generated_image = slow_place

    await asyncio.wait_for(runner.on_late_saved_image("call-1", str(image)), 2)

    assert len(runner.sent) == 1
    assert reports.empty()
    release.set()
    report = await asyncio.wait_for(reports.get(), 2)
    assert f"{GENERATED_IMAGE_DIR}/call-1.png" in report["note"]


@pytest.mark.asyncio
async def test_a_late_image_whose_copy_failed_says_so(monkeypatch, image, tmp_path):
    runner, reports = _late_runner(monkeypatch, tmp_path)

    async def no_place(path):
        raise RuntimeError("sandbox gone")

    runner._place_generated_image = no_place

    await runner.on_late_saved_image("call-1", str(image))
    report = await asyncio.wait_for(reports.get(), 2)

    assert len(runner.sent) == 1
    assert "sandbox gone" in report["note"]
    assert "already been sent" in report["note"]


@pytest.mark.asyncio
async def test_an_image_being_delivered_as_its_turn_ends_is_not_a_failure(
    monkeypatch, image, tmp_path
):
    """回调还在发图时回合结束：不能留下记录，否则随后的完成事件会被当成失败。"""
    import astrbot.core.tools.computer_tools.util as util

    monkeypatch.setattr(util, "workspace_root", lambda umo: tmp_path / "ws")
    runner = _runner("none")
    runner._hooked_images = set()
    runner._images_running = {"call-1", "call-2"}

    await runner._on_saved_image("call-1", str(image))

    assert len(runner.sent) == 1
    assert runner._images_running == {"call-2"}


@pytest.mark.asyncio
async def test_a_stopped_turn_s_late_image_is_dropped(monkeypatch, image, tmp_path):
    runner, reports = _late_runner(monkeypatch, tmp_path)
    runner._aborted = True

    assert await runner.on_late_saved_image("call-1", str(image)) is None
    runner.on_late_image_failed()
    await asyncio.sleep(0)

    assert runner.sent == []
    assert reports.empty()


@pytest.mark.asyncio
async def test_a_late_image_goes_to_the_turn_that_started_it():
    """归属按调用记录：Alice 的图不会算到后来结束的 Bob 那一轮头上。"""
    from astrbot.core.agent.runners.codex.native import CodexEngine

    class _Owner:
        def __init__(self):
            self.calls = []

        async def on_late_saved_image(self, call_id, saved_path):
            self.calls.append(call_id)
            return "late"

    engine = CodexEngine(rt=object())
    alice, bob = _Owner(), _Owner()
    engine.late_images["thread-1"] = {"call-a": alice, "call-b": bob}

    async def open_turn(call_id, saved_path):
        return "open turn"

    engine.saved_image_handlers["thread-1"] = open_turn

    assert await engine._on_saved_image("thread-1", "call-a", "/a.png") == "late"
    assert (alice.calls, bob.calls) == (["call-a"], [])
    # Each fires once; a call nobody owns goes to the open turn.
    assert await engine._on_saved_image("thread-1", "call-a", "/a.png") == "open turn"
    assert await engine._on_saved_image("thread-1", "call-c", "/c.png") == "open turn"


@pytest.mark.asyncio
async def test_a_late_image_that_failed_is_reported(monkeypatch, tmp_path):
    runner, reports = _late_runner(monkeypatch, tmp_path)

    runner.on_late_image_failed()
    report = await asyncio.wait_for(reports.get(), 2)

    assert report["sender_id"] == "alice"
    assert "failed" in report["note"]
    assert runner.sent == []


@pytest.mark.asyncio
async def test_the_pump_reports_a_late_image_that_was_never_saved():
    """保存成功时回调先取走了记录，这里还在就说明没存下来。"""
    from astrbot.core.agent.runners.codex.native import CodexEngine, ThreadPump

    class _Owner:
        failed = 0

        def on_late_image_failed(self):
            self.failed += 1

    events = [
        {"msg": {"type": "item_completed", "item": {"id": "other", "status": "x"}}},
        {
            "msg": {
                "type": "item_completed",
                "item": {"id": "call-1", "status": "failed"},
            }
        },
        None,
    ]

    class _Rt:
        @staticmethod
        async def next_event(thread_id):
            event = events.pop(0)
            return json.dumps(event) if event else None

    engine = CodexEngine(rt=_Rt())
    owner = _Owner()
    engine.late_images["thread-1"] = {"call-1": owner}

    await ThreadPump(engine, "thread-1")._run()

    assert owner.failed == 1
    assert engine.late_images["thread-1"] == {}


@pytest.mark.asyncio
async def test_an_interrupt_ends_the_late_images_of_its_thread():
    """中断会结束所有代码单元，里面的图不会再有结果，记录不必留着。"""
    from astrbot.core.agent.runners.codex.native import CodexEngine, ThreadPump

    events = [
        {"msg": {"type": "turn_aborted", "reason": "replaced"}},
        {"msg": {"type": "turn_aborted", "reason": "interrupted"}},
        None,
    ]
    seen = []

    class _Rt:
        @staticmethod
        async def next_event(thread_id):
            seen.append(dict(engine.late_images))
            event = events.pop(0)
            return json.dumps(event) if event else None

    engine = CodexEngine(rt=_Rt())
    engine.late_images["thread-1"] = {"call-1": object()}

    await ThreadPump(engine, "thread-1")._run()

    assert seen[1] == seen[0]  # another reason keeps them
    assert engine.late_images == {}


@pytest.mark.asyncio
async def test_a_late_image_reaches_the_chat_s_thread_as_an_event(monkeypatch):
    """走后台命令同一条路：回合在跑就插入，否则排队开新回合，回复发到会话。"""
    from astrbot.core.agent.runners.codex import native, wake

    turns = []

    async def turn(ctx, event, cfg, prompt, delivery):
        turns.append((event, prompt, delivery))
        return True

    monkeypatch.setattr(native, "try_steer", _no_steer)
    monkeypatch.setattr(wake, "run_in_session_thread", turn)

    class _Ctx:
        @staticmethod
        def get_config(umo=None):
            return {"plugin_set": ["only-this"]}

    async def note():
        return "THE NOTE"

    await wake.run_late_image_completion(
        _Ctx(),
        session_str=UMO,
        sender_id="user-1",
        role="member",
        group_id="30003",
        note=note(),
    )

    ((event, prompt, delivery),) = turns
    assert (event.get_sender_id(), event.role, delivery) == ("user-1", "member", UMO)
    assert event.plugins_name == ["only-this"]
    assert "THE NOTE" in prompt


@pytest.mark.asyncio
async def test_a_chat_event_with_a_full_queue_posts_no_busy_note(monkeypatch):
    """队列满时不要让用户"再发一次"：这条事件不是用户发的。"""
    from astrbot.core.agent.runners.codex import wake

    async def busy(ctx, event, cfg, prompt):
        return runner_mod.BUSY_NOTE

    monkeypatch.setattr(wake, "run_turn_in_session", busy)
    sent = []

    class _Ctx:
        @staticmethod
        async def send_message(session, chain):
            sent.append(chain)

    from astrbot.core.cron.events import CronMessageEvent
    from astrbot.core.platform.message_session import MessageSession

    session = MessageSession.from_str(UMO)
    event = CronMessageEvent(
        context=_Ctx(),
        session=session,
        message="x",
        message_type=session.message_type,
    )

    assert await wake.run_in_session_thread(_Ctx(), event, {}, "x", UMO) is False
    assert sent == []


@pytest.mark.asyncio
async def test_a_chat_event_for_a_bad_session_asks_for_no_plain_report():
    from astrbot.core.agent.runners.codex import wake

    assert await wake.deliver_chat_event(
        None, session_str="bad", sender_id="", role="", group_id="", prompt="x"
    )


@pytest.mark.asyncio
async def test_a_steered_chat_event_asks_for_no_plain_report(monkeypatch):
    from astrbot.core.agent.runners.codex import native, wake

    async def steered(*args, **kwargs):
        return 1

    monkeypatch.setattr(native, "try_steer", steered)

    class _Ctx:
        @staticmethod
        def get_config(umo=None):
            return {}

    assert await wake.deliver_chat_event(
        _Ctx(), session_str=UMO, sender_id="u", role="member", group_id="", prompt="x"
    )


@pytest.mark.asyncio
async def test_forgetting_a_thread_drops_its_image_handler():
    from astrbot.core.agent.runners.codex.native import CodexEngine

    class _Rt:
        @staticmethod
        async def shutdown_thread(thread_id):
            pass

    engine = CodexEngine(rt=_Rt())
    engine.saved_image_handlers["thread-1"] = object()
    engine.late_images["thread-1"] = {"call-1": object()}

    await engine.forget_thread("thread-1")

    assert engine.saved_image_handlers == {}
    assert engine.late_images == {}


def test_the_prompt_tells_the_model_how_to_read_an_image_result():
    """系统提示要说明生图结果没有 content，否则模型遍历不到内容，以为没生成。"""
    prompt = runner_mod.DEFAULT_SYSTEM_PROMPT
    assert "`generatedImage(result)`" in prompt
    assert '"yield_time_ms": 120000' in prompt
    assert "`wait`" in prompt
    assert "do not generate it again" in prompt
    assert "later steps of that script may not run" in " ".join(prompt.split())


@pytest.mark.asyncio
async def test_a_path_that_is_already_gone_is_left_to_codex():
    runner = _runner("none")
    runner._hooked_images = set()

    assert await runner._on_saved_image("call-1", "/nowhere/call-1.png") is None
    assert runner._hooked_images == set()


def test_fallback_notes_are_wrapped_as_request_context():
    """兜底路径是作为输入插进回合的，要和工具结果区分开。"""
    wrapped = runner_mod.request_context("generated_image", "BODY")

    assert wrapped.startswith('<request_context name="generated_image">')
    assert "BODY" in wrapped
    assert wrapped.endswith("</request_context>")
