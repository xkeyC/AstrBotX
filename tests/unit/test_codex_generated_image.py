"""生图之后：复制进会话工作区，告诉模型位置，由它决定发送或继续使用。

背景：Codex 把图存在 CODEX_HOME（宿主机上），消息发送工具链明确拒绝这个
目录，所以之前"自动附在回复里"实际发不出去。现在改成复制到工作区（沙箱
运行时复制进沙箱），再用一条 request_context 告诉模型路径，不再自动发送。
"""

import pytest

from astrbot.core.agent.runners.codex import codex_agent_runner as runner_mod
from astrbot.core.agent.runners.codex.codex_agent_runner import (
    GENERATED_IMAGE_DIR,
    CodexAgentRunner,
    generated_image_failure_note,
    generated_image_note,
)

UMO = "aiocqhttp:GroupMessage:30003"


def _runner(runtime: str = "sandbox"):
    runner = CodexAgentRunner.__new__(CodexAgentRunner)
    runner.cfg = {}
    runner.umo = UMO

    class _PluginCtx:
        @staticmethod
        def get_config(umo=None):
            return {"provider_settings": {"computer_use_runtime": runtime}}

    class _AgentCtx:
        context = _PluginCtx()

        class event:  # noqa: N801
            unified_msg_origin = UMO

    class _Wrapper:
        context = _AgentCtx()

    runner.run_context = _Wrapper()
    return runner


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "codex_home" / "generated_images" / "sess" / "call-1.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x89PNG fake")
    return path


# ------------------------------------------------------------------ 提示文案


def test_note_tells_the_model_where_the_image_is_and_that_it_was_not_sent():
    note = generated_image_note(
        "generated_images/call-1.png", "`generated_images/call-1.png`"
    )

    assert "generated_images/call-1.png" in note
    assert "NOT been sent" in note
    assert "send_message_to_user" in note
    # 路径要能原样喂给 send_message_to_user
    assert '"path": "generated_images/call-1.png"' in note


def test_failure_note_says_the_image_cannot_be_delivered():
    note = generated_image_failure_note("sandbox unreachable")

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

    note = await _runner("sandbox")._place_generated_image(str(image))

    assert uploads == [(str(image), f"{GENERATED_IMAGE_DIR}/call-1.png")]
    assert "/workspace/generated_images/call-1.png" in note
    assert "NOT been sent" in note


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

    note = await _runner("none")._place_generated_image(str(image))

    copied = workspace / GENERATED_IMAGE_DIR / "call-1.png"
    assert copied.read_bytes() == image.read_bytes()
    assert f"{GENERATED_IMAGE_DIR}/call-1.png" in note


@pytest.mark.asyncio
async def test_a_failed_copy_still_produces_a_note(monkeypatch, image):
    """复制失败时模型也必须收到说明，否则它会一直以为图在路上。"""

    async def _get_booter(ctx, umo):
        raise RuntimeError("sandbox unreachable")

    import astrbot.core.computer.computer_client as client

    monkeypatch.setattr(client, "get_booter", _get_booter)

    note = await _runner("sandbox")._place_generated_image(str(image))

    assert "sandbox unreachable" in note
    assert "could not be delivered" in note


# ------------------------------------------------------------------ 插入回合


class _Engine:
    def __init__(self, status):
        self.status = status
        self.requests = []

    async def submit_turn(self, thread_id, request):
        self.requests.append(request)
        return {"status": self.status}


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
    assert "NOT been sent" in text
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
