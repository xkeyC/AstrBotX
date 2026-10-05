"""Build a full ProviderRequest for the Codex runner.

Third-party runners normally get a bare prompt. Codex runs AstrBot's plugin
tools itself, so it gets what the local agent would: conversation, persona,
message metadata, quoted messages, knowledge base results and the per-request
tool set. Only the provider-bound steps (model selection, context compression,
computer-use tools) are left out; Codex owns those.
"""

from __future__ import annotations

import os

from astrbot.core import logger
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.tool import ToolSet
from astrbot.core.astr_main_agent import (
    PERSONA_ALLOWED_TOOLS_EXTRA_KEY,
    MainAgentBuildConfig,
    _apply_kb,
    _apply_llm_safety_mode,
    _apply_local_env_tools,
    _apply_sandbox_tools,
    _decorate_llm_request,
    _filter_tools_by_persona_scope,
    _get_quoted_message_parser_settings,
    _get_session_conv,
    _plugin_tool_fix,
    _proactive_cron_job_tools,
    _process_quote_message,
)
from astrbot.core.message.components import File, Image, Record, Reply
from astrbot.core.permission_rules import CONFIG_KEY as PERMISSION_RULES_KEY
from astrbot.core.permission_rules import policy_for_event
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.message_type import MessageType
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.star.context import Context
from astrbot.core.tools.message_tools import (
    GetGroupMessageHistoryTool,
    SendMessageToUserTool,
)


def _build_config(
    astrbot_config: dict, runner_config: dict, plugin_context: Context
) -> MainAgentBuildConfig:
    original = astrbot_config.get("provider_settings", {})
    settings = dict(original)
    # Codex reads images natively; do not caption them with another provider.
    settings["default_image_caption_provider_id"] = ""
    # Shipyard mode keeps every agent file operation inside the sandbox: Codex
    # is only the orchestrator, and even skills are read from the sandbox copy.
    shipyard_mode = bool(runner_config.get("shipyard_mode"))
    runtime = (
        "sandbox" if shipyard_mode else original.get("computer_use_runtime", "none")
    )
    # Skills are read from this host through astrbot_read_skill (not a shell),
    # unless shipyard mode moves that into the sandbox as well.
    settings["_codex_skills"] = "sandbox" if shipyard_mode else True
    settings["computer_use_runtime"] = "sandbox" if shipyard_mode else "local"
    proactive_cfg = settings.get("proactive_capability", {}) or {}
    return MainAgentBuildConfig(
        tool_call_timeout=int(runner_config.get("tool_call_timeout") or 120),
        provider_settings=settings,
        kb_agentic_mode=astrbot_config.get("kb_agentic_mode", False),
        llm_safety_mode=bool(runner_config.get("safety_mode", False)),
        add_cron_tools=proactive_cfg.get("add_cron_tools", True),
        timezone=plugin_context.get_config().get("timezone"),
        max_quoted_fallback_images=settings.get("max_quoted_fallback_images", 20),
        # Execution environment as configured by the user (sandbox / local /
        # none), or the sandbox alone in shipyard mode.
        computer_use_runtime=runtime,
        sandbox_cfg=original.get("sandbox", {}) or {},
    )


async def _collect_media(event: AstrMessageEvent, req: ProviderRequest) -> None:
    """Replace the base64 images of the generic path with local file paths."""
    req.image_urls = []
    req.audio_urls = []
    for comp in event.message_obj.message:
        chains = [comp]
        if isinstance(comp, Reply) and comp.chain:
            chains = list(comp.chain)
        for c in chains:
            try:
                if isinstance(c, Image):
                    req.image_urls.append(await c.convert_to_file_path())
                elif isinstance(c, Record):
                    req.audio_urls.append(await c.convert_to_file_path())
                elif isinstance(c, File):
                    path = await c.get_file()
                    name = c.name or os.path.basename(path)
                    req.extra_user_content_parts.append(
                        TextPart(text=f"[File Attachment: name {name}, path {path}]")
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("Codex: failed to resolve attachment %s: %s", c, e)


async def prepare_codex_request(
    event: AstrMessageEvent,
    req: ProviderRequest,
    plugin_context: Context,
    astrbot_config: dict,
    runner_config: dict,
) -> None:
    config = _build_config(astrbot_config, runner_config, plugin_context)
    policy = policy_for_event(event, astrbot_config.get(PERMISSION_RULES_KEY) or [])
    if policy.persona_id and not event.get_selected_persona():
        event.set_selected_persona(policy.persona_id)
    if policy.model and not req.model:
        req.model = policy.model
    await _collect_media(event, req)

    req.conversation = await _get_session_conv(event, plugin_context)
    await _decorate_llm_request(event, req, plugin_context, config, provider=None)
    # Upstream appends the quoted message inside build_main_agent, which this
    # path replaces. Codex reads images natively, so no caption is needed.
    await _process_quote_message(
        event,
        req,
        img_cap_prov_id="",
        plugin_context=plugin_context,
        quoted_message_settings=_get_quoted_message_parser_settings(
            config.provider_settings
        ),
        main_provider_supports_image=True,
        skip_quote_image_caption=True,
    )
    await _apply_kb(event, req, plugin_context, config)
    _plugin_tool_fix(event, req)

    if config.llm_safety_mode:
        _apply_llm_safety_mode(config, req)
    # AstrBot's execution tools (shipyard-neo sandbox or local host). They are
    # deferred dynamic tools under code mode, so they cost no prompt tokens.
    if config.computer_use_runtime == "sandbox":
        _apply_sandbox_tools(config, req, req.session_id or event.unified_msg_origin)
    elif config.computer_use_runtime == "local":
        _apply_local_env_tools(req, plugin_context)
    if config.computer_use_runtime in ("sandbox", "local") and req.func_tool:
        from astrbot.core.tools.computer_tools.apply_patch import ApplyPatchTool
        from astrbot.core.tools.computer_tools.codex_exec import (
            ExecCommandTool,
            WriteStdinTool,
        )

        # Codex models are trained on this patch format and on the
        # exec_command / write_stdin session pair, so they replace AstrBot's
        # one-shot shell tools here.
        req.func_tool.add_tool(ApplyPatchTool())
        req.func_tool.remove_tool("astrbot_execute_shell")
        req.func_tool.remove_tool("astrbot_shell_session")
        req.func_tool.add_tool(ExecCommandTool())
        req.func_tool.add_tool(WriteStdinTool())
        # A persona that lists the shell or edit tools keeps them under their
        # Codex names, so its tool list means the same with either runner.
        allowed = event.get_extra(PERSONA_ALLOWED_TOOLS_EXTRA_KEY)
        if allowed is not None:
            allowed = set(allowed)
            if allowed & {"astrbot_execute_shell", "astrbot_shell_session"}:
                allowed |= {"exec_command", "write_stdin"}
            if allowed & {"astrbot_file_edit_tool", "astrbot_file_write_tool"}:
                allowed.add("apply_patch")
            event.set_extra(PERSONA_ALLOWED_TOOLS_EXTRA_KEY, allowed)
    if config.add_cron_tools:
        _proactive_cron_job_tools(req, plugin_context)

    tmgr = plugin_context.get_llm_tool_manager()
    if req.func_tool is None:
        req.func_tool = ToolSet()
    if event.platform_meta.support_proactive_message:
        req.func_tool.add_tool(tmgr.get_builtin_tool(SendMessageToUserTool))
    ltm = plugin_context.get_config(umo=event.unified_msg_origin).get(
        "provider_ltm_settings", {}
    )
    if event.get_message_type() == MessageType.GROUP_MESSAGE and ltm.get(
        "group_message_history_enable", False
    ):
        req.func_tool.add_tool(tmgr.get_builtin_tool(GetGroupMessageHistoryTool))

    _filter_tools_by_persona_scope(event, req)
    # One reader for every runtime: it opens the host copy, or the sandbox copy
    # in shipyard mode, so the model never has to locate skill files itself.
    # Added after the persona's tool list is applied: the persona's skill list
    # already decides which skills this request carries.
    if event.get_extra("_codex_skills"):
        from astrbot.core.agent.runners.codex.skills import ReadSkillTool

        req.func_tool.add_tool(ReadSkillTool())
    if not policy.is_default:
        # Name the execution tools this request actually carries, so a sender
        # barred from the host does not conclude that nothing can run.
        offered = {tool.name for tool in (req.func_tool.tools if req.func_tool else [])}
        sandbox_tools = tuple(
            name for name in ("exec_command", "write_stdin") if name in offered
        )
        summary = policy.summary(
            host_exec=bool(runner_config.get("native_exec_tools")),
            sandbox_tools=sandbox_tools,
        )
        if summary:
            # Per-message and append-only: the tool set and cached prefix stay the same.
            req.add_persistent_context(
                "sender_permissions",
                f"Permissions of this sender: {summary}. Do not attempt restricted "
                "actions for them; if asked, say politely that they are not allowed.",
            )
    if runner_config.get("memory_enabled") and policy.global_memory is True:
        # Codex checks the turn's scopes when the memory tools run; this tells
        # the model who holds them, per message, without touching the prefix.
        req.add_persistent_context(
            "memory_permission",
            "This sender may manage shared memories: when they ask, you may "
            "save impersonal knowledge to the shared memory folder, and delete "
            "memories, in this chat whether it is a group or not.",
        )
    req.context_anchors_complete = True
    if not req.prompt and (req.image_urls or req.extra_user_content_parts):
        req.prompt = "<attachment>"
