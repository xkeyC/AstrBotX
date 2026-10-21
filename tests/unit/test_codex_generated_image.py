"""生图之后：宿主直接把图发到会话，再复制进会话工作区供模型继续处理。

背景：Codex 把图存在 CODEX_HOME（宿主机上），模型的工具都够不着这个目录。
所以图一保存就由宿主发给用户，同时复制到工作区（沙箱运行时复制进沙箱），
并告诉模型"已发送"以及副本的位置。
"""

import asyncio

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
