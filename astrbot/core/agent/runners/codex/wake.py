"""Scheduled-task wake-ups for the Codex runner (K12).

A cron run is delivered into the chat's own Codex thread (sharing its
context and prompt cache) as one message: a quote of the user's original
request plus the task to execute now. The wording makes clear this is one run
of an existing task, so the agent performs it instead of scheduling another.
"""

from __future__ import annotations

from typing import Any

from astrbot.core import content_moderation, logger
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.provider.entities import ProviderRequest


def build_cron_prompt(cron_job: dict, payload: dict) -> str:
    name = str(cron_job.get("name") or cron_job.get("id") or "task")
    started = str(cron_job.get("run_started_at") or "")
    note = str(
        payload.get("note") or cron_job.get("note") or cron_job.get("description") or ""
    )
    origin = str(payload.get("origin_message") or "").strip()
    lines = [
        f'<scheduled_task name="{name}" triggered_at="{started}">',
        "This is one scheduled run of a task the user set up earlier. Carry out the "
        "task now. A task's run cannot create or change scheduled tasks; do not "
        "cancel one unless the task itself asks for it. Your final reply is sent "
        "to the chat; do not greet.",
    ]
    if origin:
        quoted = "\n".join(f"> {line}" for line in origin.splitlines())
        lines += ["User's original request (when the task was created):", quoted]
    lines += ["Task to execute:", note, "</scheduled_task>"]
    return "\n".join(lines)


async def run_codex_cron_job(
    ctx: Any,
    *,
    message: str,
    session_str: str,
    extras: dict,
    delivery_session_str: str = "",
) -> None:
    from astrbot.core.cron.events import CronMessageEvent

    try:
        session = MessageSession.from_str(session_str)
    except Exception as e:  # noqa: BLE001
        logger.error("Invalid session for cron job: %s", e)
        return

    extras = extras or {}
    payload = extras.get("cron_payload", {}) or {}
    cron_job = extras.get("cron_job", {}) or {}
    event = CronMessageEvent(
        context=ctx,
        session=session,
        message=message,
        extras=extras,
        message_type=session.message_type,
    )
    cfg = ctx.get_config(umo=event.unified_msg_origin)
    # Run with the permissions of the user who created the task, matched as
    # in the group they created it in.
    if sender_id := str(payload.get("sender_id") or ""):
        event.message_obj.sender.user_id = sender_id
    event.message_obj.group_id = str(payload.get("group_id") or "")
    admin_ids = [str(a) for a in cfg.get("admins_id", [])]
    event.role = (
        "admin"
        if payload.get("origin") == "api" or sender_id in admin_ids
        else "member"
    )

    ok = await run_in_session_thread(
        ctx, event, cfg, build_cron_prompt(cron_job, payload), delivery_session_str
    )
    if not ok:
        logger.warning("Codex cron job %s produced no reply", cron_job.get("id"))


async def run_in_session_thread(
    ctx: Any, event: Any, cfg: dict, prompt: str, delivery_session_str: str
) -> bool:
    """Run one Codex turn with ``prompt`` in the event session's thread."""
    text = await run_turn_in_session(ctx, event, cfg, prompt)
    if text is None:
        return False
    if text and delivery_session_str:
        await ctx.send_message(delivery_session_str, MessageChain().message(text))
        # A voice conversation of that chat hears of it too (its model
        # decides whether to tell it).
        from astrbot.core.voice.chat import announce

        await announce(delivery_session_str, text)
    return True


async def run_turn_in_session(
    ctx: Any, event: Any, cfg: dict, prompt: str
) -> str | None:
    """Runs one Codex turn with ``prompt`` in the event session's thread, as
    a message of that chat would (same session lock: it queues behind turns
    running there, and they behind it).

    Returns:
        The final answer (possibly empty), or None when the turn was stopped
        or produced no assistant answer.
    """
    from astrbot.core.agent.runners.codex.codex_agent_runner import CodexAgentRunner
    from astrbot.core.astr_agent_context import AgentContextWrapper, AstrAgentContext
    from astrbot.core.astr_agent_hooks import MAIN_AGENT_HOOKS
    from astrbot.core.config.agent_runner import normalize_agent_runner
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.codex_request import (
        prepare_codex_request,
    )
    from astrbot.core.utils.active_event_registry import active_event_registry

    runner_cfg = normalize_agent_runner(cfg.get("agent_runner"))["config"]
    req = ProviderRequest()
    req.session_id = event.unified_msg_origin
    req.prompt = prompt
    # Active in the chat like any message's turn, from the start, so /stop
    # reaches it (and counts it) at any point.
    active_event_registry.register(event)
    try:
        await prepare_codex_request(event, req, ctx, cfg, runner_cfg)
        req.prompt = prompt
        runner = CodexAgentRunner()
        await runner.reset(
            request=req,
            run_context=AgentContextWrapper(
                context=AstrAgentContext(context=ctx, event=event),
                tool_call_timeout=int(runner_cfg.get("tool_call_timeout") or 120),
            ),
            agent_hooks=MAIN_AGENT_HOOKS,
            provider_config=runner_cfg,
            streaming=False,
        )
        active_event_registry.register_agent_stop_callback(event, runner.request_stop)
        if event.get_extra("agent_stop_requested") is True:
            # Stopped while it was being prepared.
            runner.request_stop()
        async for _ in runner.step_until_done():
            pass
    finally:
        active_event_registry.unregister(event)
    if runner.was_aborted():
        # Stopped: nothing to deliver. A background command's caller then
        # reports its result plainly, so the result itself is not lost.
        return None
    resp = runner.get_final_llm_resp()
    if resp is None or resp.role != "assistant":
        return None
    return (resp.completion_text or "").strip()


def build_background_prompt(task_result: dict, original_message: str) -> str:
    lines = [
        f'<background_task_result tool="{task_result.get("tool_name", "")}" '
        f'task_id="{task_result.get("task_id", "")}">',
        "A background task you started earlier has finished. Use its result to "
        "continue what the user asked. Your final reply is sent to the chat; if "
        "nothing needs to be said, reply with an empty message.",
    ]
    if original_message.strip():
        quoted = "\n".join(f"> {line}" for line in original_message.splitlines())
        lines += ["The request it belongs to:", quoted]
    lines += [
        "Result:",
        str(task_result.get("result") or ""),
        "</background_task_result>",
    ]
    return "\n".join(lines)


async def run_codex_background_wake(
    ctx: Any, origin_event: Any, task_result: dict
) -> None:
    from astrbot.core.cron.events import CronMessageEvent

    session = MessageSession.from_str(origin_event.unified_msg_origin)
    event = CronMessageEvent(
        context=ctx,
        session=session,
        message="background task result",
        extras={"background_task_result": task_result},
        message_type=session.message_type,
    )
    event.message_obj.sender.user_id = str(origin_event.get_sender_id() or "")
    event.message_obj.group_id = str(origin_event.get_group_id() or "")
    event.role = origin_event.role
    cfg = ctx.get_config(umo=origin_event.unified_msg_origin) or {}
    # The result reaches the model inside the prompt, past the tool bridge
    # that checks the results of tools run in a turn.
    result = await content_moderation.filter_tool_text(
        str(task_result.get("result") or ""),
        content_moderation.platform_mode(session.platform_id),
        label=f"background {task_result.get('tool_name', '')}",
    )
    task_result = {**task_result, "result": result}
    prompt = build_background_prompt(task_result, origin_event.message_str or "")
    await run_in_session_thread(
        ctx, event, cfg, prompt, origin_event.unified_msg_origin
    )


def build_background_exec_prompt(session_id: str, exit_code: int, output: str) -> str:
    """Prompt announcing a background command that finished after its turn."""
    lines = [
        f'<background_command session="{session_id}" exit_code="{exit_code}">',
        "A command you started earlier has finished. Its turn was already over, so "
        "this is how the result reaches you. Tell the user what happened, in your "
        "own voice, and act on it if the situation calls for it. Do not start the "
        "command again.",
        "Output:",
        output.strip() or "(no output)",
        "</background_command>",
    ]
    return "\n".join(lines)


async def run_background_exec_completion(
    ctx: Any,
    *,
    session_str: str,
    sender_id: str,
    role: str,
    group_id: str = "",
    session_id: str,
    exit_code: int,
    output: str,
) -> None:
    """Delivers a finished background command into its own chat.

    It arrives as a message from whoever started the command, so it follows the
    same rules any message of theirs would: if their own turn is still running
    it is steered into that turn and answered in the same reply, and otherwise
    it queues behind whatever else that chat is doing.
    """
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        build_turn_input,
        turn_scopes,
    )
    from astrbot.core.agent.runners.codex.native import try_steer
    from astrbot.core.cron.events import CronMessageEvent
    from astrbot.core.permission_rules import CONFIG_KEY, policy_for_event

    # As a tool's result: the output reaches the model inside the prompt.
    output = await content_moderation.filter_tool_text(
        output,
        content_moderation.platform_mode(session_str.split(":", 1)[0]),
        label=f"background command {session_id}",
    )
    prompt = build_background_exec_prompt(session_id, exit_code, output)
    try:
        session = MessageSession.from_str(session_str)
    except Exception as e:  # noqa: BLE001
        logger.error("Invalid session for background command %s: %s", session_id, e)
        return
    event = CronMessageEvent(
        context=ctx,
        session=session,
        message=prompt,
        message_type=session.message_type,
    )
    if sender_id:
        event.message_obj.sender.user_id = sender_id
    # Group rules match as for the message that started the command.
    event.message_obj.group_id = group_id
    event.role = role or "member"
    cfg = ctx.get_config(umo=event.unified_msg_origin)
    if sender_id:
        # Steered only into a turn of the same person in the same role.
        policy_for_event(event, cfg.get(CONFIG_KEY) or [])
        req = ProviderRequest()
        req.session_id = session_str
        req.prompt = prompt
        steered = await try_steer(
            session_str,
            sender_id,
            build_turn_input(req),
            prompt=prompt,
            scopes=turn_scopes(event),
        )
        if steered is not None:
            return

    if not await run_in_session_thread(ctx, event, cfg, prompt, session_str):
        logger.warning(
            "Background command %s produced no reply; reporting it plainly.",
            session_id,
        )
        await ctx.send_message(
            session_str,
            MessageChain().message(
                f"后台命令已结束（会话 {session_id}，退出码 {exit_code}）"
            ),
        )
