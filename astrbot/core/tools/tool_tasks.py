"""Tool calls that outlive the tool call timeout.

A tool still running when ``tool_call_timeout`` passes is not stopped: it keeps
running as a *tool task*, and the model gets a reminder instead of a result.
The model then decides, through :class:`ToolTaskTool`, whether to wait for the
result, check on it, or cancel it. Tasks belong to the chat (``umo``) that
started them and are dropped an hour after they finish if nobody collects them.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

import mcp
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.permission_rules import EVENT_EXTRA_KEY as POLICY_EXTRA_KEY
from astrbot.core.permission_rules import PermissionPolicy

from .registry import builtin_tool

TOOL_TASK_TOOL_NAME = "astrbot_tool_task"
# Finished tasks nobody collected are forgotten after this long.
FINISHED_TASK_TTL_SECONDS = 3600
# A wait leaves this much of the tool call timeout for the call itself.
WAIT_MARGIN_SECONDS = 5
NO_RETURN_VALUE_TEXT = (
    "The tool has no return value, or has sent the result directly to the user."
)


@dataclass
class ToolTask:
    """A tool call still running after its timeout."""

    id: str
    tool_name: str
    umo: str
    started_at: float
    task: asyncio.Task
    # What the tool produced (CallToolResult, or None for "nothing to return").
    outputs: list[Any]
    # Outputs before this index were already given to the model.
    delivered: int = 0
    finished_at: float | None = None
    # Who started it, and the started tool's MCP server: another sender of
    # the chat needs permission for that tool to collect or cancel it.
    sender_id: str = ""
    mcp_server: str | None = None

    @property
    def done(self) -> bool:
        return self.task.done()

    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at


_tasks: dict[str, ToolTask] = {}


def _purge() -> None:
    now = time.time()
    for task_id, t in list(_tasks.items()):
        if (
            t.finished_at is not None
            and now - t.finished_at > FINISHED_TASK_TTL_SECONDS
        ):
            _tasks.pop(task_id, None)


def register_tool_task(
    *,
    tool_name: str,
    umo: str,
    started_at: float,
    task: asyncio.Task,
    outputs: list[Any],
    delivered: int,
    sender_id: str = "",
    mcp_server: str | None = None,
) -> ToolTask:
    """Keep a still-running tool call as a task the model can come back to.

    Args:
        tool_name: Name of the tool that outlived its timeout.
        umo: The chat that started it; only that chat can reach the task.
        started_at: When the tool call started (epoch seconds).
        task: The asyncio task still running the tool.
        outputs: The list the task keeps appending the tool's outputs to.
        delivered: How many of ``outputs`` the model has already seen.
        sender_id: Who started it; another sender needs the tool's permission.
        mcp_server: The MCP server of the tool, for that permission check.

    Returns:
        The registered task.
    """
    _purge()
    entry = ToolTask(
        id=uuid.uuid4().hex[:12],
        tool_name=tool_name,
        umo=umo,
        started_at=started_at,
        task=task,
        outputs=outputs,
        delivered=delivered,
        sender_id=sender_id,
        mcp_server=mcp_server,
    )

    def _finished(t: asyncio.Task) -> None:
        entry.finished_at = time.time()

    task.add_done_callback(_finished)
    _tasks[entry.id] = entry
    return entry


def get_tool_task(task_id: str, umo: str) -> ToolTask | None:
    """Find a tool task of one chat.

    Args:
        task_id: The id the timeout reminder named.
        umo: The chat asking.

    Returns:
        The task, or None if it is unknown, collected, or another chat's.
    """
    _purge()
    entry = _tasks.get(task_id)
    if entry is None or entry.umo != umo:
        return None
    return entry


def timeout_reminder(entry: ToolTask, timeout: float) -> str:
    """What the model gets instead of a result when a tool outlives its timeout.

    Args:
        entry: The task the tool now runs as.
        timeout: The tool call timeout it outlived, in seconds.

    Returns:
        The reminder text, naming the task and how to manage it.
    """
    return (
        f"Tool `{entry.tool_name}` is still running after {timeout:g} s. It was "
        f"not stopped: it keeps running in the background as task {entry.id}. "
        f"Decide what to do with `{TOOL_TASK_TOOL_NAME}` "
        f'(task_id="{entry.id}"): action "wait" to wait for its result, '
        '"status" to check on it, or "cancel" to stop it. If it may take a '
        "while, tell the user before waiting."
    )


def _collect(entry: ToolTask) -> mcp.types.CallToolResult:
    """The outputs the model has not seen yet, as one result."""
    content: list[Any] = []
    is_error = False
    for out in entry.outputs[entry.delivered :]:
        if isinstance(out, mcp.types.CallToolResult):
            content.extend(out.content)
            is_error = is_error or bool(out.isError)
        elif out is None:
            content.append(
                mcp.types.TextContent(type="text", text=NO_RETURN_VALUE_TEXT)
            )
    entry.delivered = len(entry.outputs)
    if not content:
        content.append(mcp.types.TextContent(type="text", text=NO_RETURN_VALUE_TEXT))
    return mcp.types.CallToolResult(content=content, isError=is_error)


def _text(text: str, *, error: bool = False) -> mcp.types.CallToolResult:
    return mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type="text", text=text)], isError=error
    )


def _finished_result(entry: ToolTask) -> mcp.types.CallToolResult:
    _tasks.pop(entry.id, None)
    # Read from the task itself: its done callback may not have run yet.
    if entry.task.cancelled():
        return _text(f"Task {entry.id} ({entry.tool_name}) was cancelled.", error=True)
    if (error := entry.task.exception()) is not None:
        return _text(
            f"Task {entry.id} ({entry.tool_name}) failed after "
            f"{entry.elapsed():.0f} s: {error}",
            error=True,
        )
    return _collect(entry)


@builtin_tool
@pydantic_dataclass
class ToolTaskTool(FunctionTool[AstrAgentContext]):
    name: str = TOOL_TASK_TOOL_NAME
    description: str = (
        "Manage a tool call that is still running after its timeout (the "
        "timeout reminder names its task_id). action 'wait' waits up to "
        "wait_seconds for its result and returns it; 'status' says whether it "
        "is still running; 'cancel' stops it."
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task id from the timeout reminder.",
                },
                "action": {
                    "type": "string",
                    "enum": ["wait", "status", "cancel"],
                    "description": "wait for the result, check its status, or cancel it.",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "How long 'wait' waits at most. Defaults to "
                    "just under the tool call timeout, which also caps it.",
                },
            },
            "required": ["task_id", "action"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        """Wait for, check on, or cancel a tool task of the current chat.

        Args:
            context: Current agent execution context.
            **kwargs: task_id, action ("wait", "status" or "cancel") and
                optional wait_seconds.

        Returns:
            The task's result, its status, or what cancelling it did.
        """
        event = context.context.event
        task_id = str(kwargs.get("task_id") or "").strip()
        action = str(kwargs.get("action") or "").strip()
        entry = get_tool_task(task_id, event.unified_msg_origin)
        if entry is None:
            return _text(
                f"error: no tool task {task_id!r} in this chat (it may have "
                "finished and been collected already).",
                error=True,
            )
        # In a shared chat, another sender may only manage a tool they may use.
        get_extra = getattr(event, "get_extra", None)
        policy = get_extra(POLICY_EXTRA_KEY) if callable(get_extra) else None
        get_sender_id = getattr(event, "get_sender_id", None)
        sender_id = str(get_sender_id() or "") if callable(get_sender_id) else ""
        if (
            isinstance(policy, PermissionPolicy)
            and sender_id != entry.sender_id
            and not policy.allows_tool(entry.tool_name, entry.mcp_server)
        ):
            return _text(
                f"error: permission denied — this user may not use "
                f"{entry.tool_name}, so cannot manage its task.",
                error=True,
            )
        if action == "status":
            if entry.done:
                return _text(
                    f"Task {entry.id} ({entry.tool_name}) has finished after "
                    f"{entry.elapsed():.0f} s; use action 'wait' to get its result."
                )
            return _text(
                f"Task {entry.id} ({entry.tool_name}) is still running "
                f"({entry.elapsed():.0f} s so far)."
            )
        # Every wait here stays under this call's own timeout.
        timeout = float(context.tool_call_timeout)
        cap = max(0.5, timeout - min(WAIT_MARGIN_SECONDS, timeout / 2))
        if action == "cancel":
            if entry.done:
                # Too late: it finished, so the result is what it did.
                return _finished_result(entry)
            entry.task.cancel()
            await asyncio.wait({entry.task}, timeout=min(WAIT_MARGIN_SECONDS, cap))
            if not entry.done:
                return _text(
                    f"Cancellation of task {entry.id} ({entry.tool_name}) was "
                    "requested, but it has not stopped yet; check its status again."
                )
            _tasks.pop(entry.id, None)
            return _text(f"Task {entry.id} ({entry.tool_name}) was cancelled.")
        if action != "wait":
            return _text(f"error: unknown action {action!r}.", error=True)

        try:
            raw = kwargs.get("wait_seconds")
            wait_seconds = cap if raw is None else float(raw)
        except (TypeError, ValueError):
            wait_seconds = cap
        # 0 checks without waiting.
        wait_seconds = max(0.0, min(wait_seconds, cap))
        if not entry.done and wait_seconds > 0:
            await asyncio.wait({entry.task}, timeout=wait_seconds)
        if entry.done:
            return _finished_result(entry)
        return _text(
            f"Task {entry.id} ({entry.tool_name}) is still running after "
            f"{entry.elapsed():.0f} s in total. Wait again, or cancel it."
        )
