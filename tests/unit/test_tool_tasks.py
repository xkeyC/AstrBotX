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


def _ctx(event: _Event, timeout: int) -> ContextWrapper:
    return ContextWrapper(
        context=SimpleNamespace(event=event, context=SimpleNamespace()),
        tool_call_timeout=timeout,
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
