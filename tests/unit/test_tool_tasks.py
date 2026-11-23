"""A tool outliving the tool call timeout keeps running as a tool task; the
model is reminded and waits for it, checks on it, or cancels it."""

import asyncio
from types import SimpleNamespace

import pytest

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.tools import tool_tasks
from astrbot.core.tools.tool_tasks import ToolTaskTool


class _Event:
    def __init__(self, umo: str = "qq:GroupMessage:1") -> None:
        self.unified_msg_origin = umo
        self.extras: dict[str, object] = {}
        self.role = "member"

    def get_extra(self, key):
        return self.extras.get(key)

    def set_extra(self, key, value) -> None:
        self.extras[key] = value

    def get_result(self):
        return None


def _ctx(event: _Event, timeout: float, *, tool_tasks: bool = True) -> ContextWrapper:
    return ContextWrapper(
        context=SimpleNamespace(event=event, context=SimpleNamespace()),
        tool_call_timeout=timeout,
        tool_tasks=tool_tasks,
    )


def _tool(handler) -> FunctionTool:
    return FunctionTool(
        name="research",
        description="slow research",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def _text(result) -> str:
    return " ".join(c.text for c in result.content if getattr(c, "text", None))


async def _run(tool, ctx):
    return [r async for r in FunctionToolExecutor.execute(tool, ctx)]


@pytest.fixture(autouse=True)
def _clear_tasks():
    tool_tasks._tasks.clear()
    yield
    for entry in list(tool_tasks._tasks.values()):
        entry.task.cancel()
    tool_tasks._tasks.clear()


@pytest.mark.asyncio
async def test_a_tool_within_the_timeout_returns_its_result():
    async def handler(event):
        return "quick result"

    out = await _run(_tool(handler), _ctx(_Event(), timeout=5))
    assert [_text(r) for r in out] == ["quick result"]
    assert not tool_tasks._tasks


@pytest.mark.asyncio
async def test_a_slow_tool_keeps_running_and_its_result_is_waited_for():
    release = asyncio.Event()

    async def handler(event):
        await release.wait()
        return "research done"

    event = _Event()
    out = await _run(_tool(handler), _ctx(event, timeout=1))
    reminder = _text(out[0])
    assert "still running after 1 s" in reminder and "not stopped" in reminder
    (entry,) = tool_tasks._tasks.values()
    assert entry.id in reminder and not entry.done

    task_tool = ToolTaskTool()
    status = await task_tool.call(_ctx(event, 10), task_id=entry.id, action="status")
    assert "still running" in _text(status)

    release.set()
    result = await task_tool.call(
        _ctx(event, 10), task_id=entry.id, action="wait", wait_seconds=3
    )
    assert _text(result) == "research done" and not result.isError
    # Collected once.
    again = await task_tool.call(_ctx(event, 10), task_id=entry.id, action="wait")
    assert again.isError and "no tool task" in _text(again)


@pytest.mark.asyncio
async def test_a_wait_that_runs_out_says_so_and_the_task_stays():
    async def handler(event):
        await asyncio.sleep(30)

    event = _Event()
    await _run(_tool(handler), _ctx(event, timeout=1))
    (entry,) = tool_tasks._tasks.values()
    # The wait is capped below the call timeout (here 6 - 5 = 1 s).
    result = await ToolTaskTool().call(
        _ctx(event, 6), task_id=entry.id, action="wait", wait_seconds=60
    )
    assert "still running" in _text(result) and not entry.done


@pytest.mark.asyncio
async def test_cancel_stops_the_tool():
    stopped = asyncio.Event()

    async def handler(event):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            stopped.set()
            raise

    event = _Event()
    await _run(_tool(handler), _ctx(event, timeout=1))
    (entry,) = tool_tasks._tasks.values()
    result = await ToolTaskTool().call(
        _ctx(event, 10), task_id=entry.id, action="cancel"
    )
    assert "cancelled" in _text(result)
    assert stopped.is_set() and not tool_tasks._tasks


@pytest.mark.asyncio
async def test_a_failing_task_reports_its_error():
    async def handler(event):
        await asyncio.sleep(1.5)
        raise RuntimeError("upstream 503")

    event = _Event()
    await _run(_tool(handler), _ctx(event, timeout=1))
    (entry,) = tool_tasks._tasks.values()
    result = await ToolTaskTool().call(
        _ctx(event, 10), task_id=entry.id, action="wait", wait_seconds=3
    )
    assert result.isError and "upstream 503" in _text(result)


@pytest.mark.asyncio
async def test_another_chat_cannot_reach_the_task():
    async def handler(event):
        await asyncio.sleep(30)

    await _run(_tool(handler), _ctx(_Event("qq:GroupMessage:1"), timeout=1))
    (entry,) = tool_tasks._tasks.values()
    other = _Event("qq:GroupMessage:2")
    result = await ToolTaskTool().call(
        _ctx(other, 10), task_id=entry.id, action="cancel"
    )
    assert result.isError and not entry.done


@pytest.mark.asyncio
async def test_a_caller_cancelled_before_the_timeout_stops_the_tool():
    stopped = asyncio.Event()
    started = asyncio.Event()

    async def handler(event):
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            stopped.set()
            raise

    call = asyncio.create_task(_run(_tool(handler), _ctx(_Event(), timeout=10)))
    await asyncio.wait_for(started.wait(), 2)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    await asyncio.wait_for(stopped.wait(), 2)
    assert not tool_tasks._tasks


@pytest.mark.asyncio
async def test_without_tool_tasks_the_timeout_still_stops_the_tool():
    """A run that does not offer astrbot_tool_task (a plugin's tool loop, say)
    keeps the old behaviour."""
    stopped = asyncio.Event()

    async def handler(event):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            stopped.set()
            raise

    with pytest.raises(Exception, match="execution timeout"):
        await _run(_tool(handler), _ctx(_Event(), timeout=1, tool_tasks=False))
    assert stopped.is_set() and not tool_tasks._tasks


@pytest.mark.asyncio
async def test_cancelling_a_finished_task_gives_its_result():
    release = asyncio.Event()

    async def handler(event):
        await release.wait()
        return "done anyway"

    event = _Event()
    await _run(_tool(handler), _ctx(event, timeout=1))
    (entry,) = tool_tasks._tasks.values()
    release.set()
    await asyncio.wait({entry.task}, timeout=2)
    result = await ToolTaskTool().call(
        _ctx(event, 10), task_id=entry.id, action="cancel"
    )
    assert _text(result) == "done anyway" and not tool_tasks._tasks


@pytest.mark.asyncio
async def test_waits_stay_under_a_small_timeout():
    async def handler(event):
        await asyncio.sleep(30)

    event = _Event()
    await _run(_tool(handler), _ctx(event, timeout=1))
    (entry,) = tool_tasks._tasks.values()
    loop = asyncio.get_running_loop()
    start = loop.time()
    result = await ToolTaskTool().call(
        _ctx(event, 2), task_id=entry.id, action="wait", wait_seconds=60
    )
    assert loop.time() - start < 2 and "still running" in _text(result)


@pytest.mark.asyncio
async def test_a_step_finishing_at_the_timeout_is_not_lost(monkeypatch):
    """The result queued as the timeout fires is delivered, not handed over."""
    import astrbot.core.astr_agent_tool_exec as exec_mod

    async def handler(event):
        return "made it"

    real_wait = asyncio.wait

    async def late_wait(aws, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
        # Let the tool finish and queue its result, then report a timeout.
        for _ in range(20):
            await asyncio.sleep(0)
        if timeout is not None:
            return set(), set(aws)
        return await real_wait(aws, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(exec_mod.asyncio, "wait", late_wait)
    out = await _run(_tool(handler), _ctx(_Event(), timeout=1))
    monkeypatch.setattr(exec_mod.asyncio, "wait", real_wait)
    assert [_text(r) for r in out] == ["made it"]
    assert not tool_tasks._tasks


@pytest.mark.asyncio
async def test_a_cancel_during_timeout_cleanup_is_kept():
    """/stop arriving while a timed-out tool cleans up stays a cancellation."""

    async def handler(event):
        try:
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(1)

    call = asyncio.create_task(
        _run(_tool(handler), _ctx(_Event(), timeout=1, tool_tasks=False))
    )
    await asyncio.sleep(1.4)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.mark.asyncio
async def test_a_direct_result_is_sent_outside_the_timeout():
    sent = []

    async def send(chain):
        await asyncio.sleep(1.2)
        sent.append(chain)

    event = _Event()
    result = SimpleNamespace(chain=["image"], derive=lambda chain: SimpleNamespace())
    event.get_result = lambda: result
    event.send = send

    async def handler(event):
        return None

    out = await _run(_tool(handler), _ctx(event, timeout=1, tool_tasks=False))
    assert out == [None] and len(sent) == 1


@pytest.mark.asyncio
async def test_another_sender_needs_the_tools_permission():
    from astrbot.core.permission_rules import EVENT_EXTRA_KEY, PermissionPolicy

    async def handler(event):
        await asyncio.sleep(30)

    starter = _Event()
    starter.get_sender_id = lambda: "admin"
    await _run(_tool(handler), _ctx(starter, timeout=1))
    (entry,) = tool_tasks._tasks.values()
    assert entry.sender_id == "admin"

    member = _Event()
    member.get_sender_id = lambda: "member"
    member.set_extra(
        EVENT_EXTRA_KEY, PermissionPolicy(rule_name="r", tools_deny=("research",))
    )
    denied = await ToolTaskTool().call(
        _ctx(member, 10), task_id=entry.id, action="cancel"
    )
    assert denied.isError and "permission denied" in _text(denied) and not entry.done

    starter.set_extra(
        EVENT_EXTRA_KEY, PermissionPolicy(rule_name="r", tools_deny=("research",))
    )
    own = await ToolTaskTool().call(
        _ctx(starter, 10), task_id=entry.id, action="status"
    )
    assert "still running" in _text(own)


@pytest.mark.asyncio
async def test_waiting_zero_seconds_only_checks():
    async def handler(event):
        await asyncio.sleep(30)

    event = _Event()
    await _run(_tool(handler), _ctx(event, timeout=1))
    (entry,) = tool_tasks._tasks.values()
    loop = asyncio.get_running_loop()
    start = loop.time()
    result = await ToolTaskTool().call(
        _ctx(event, 120), task_id=entry.id, action="wait", wait_seconds=0
    )
    assert loop.time() - start < 0.5 and "still running" in _text(result)
