"""Agent runner that hands orchestration and execution to Codex.

AstrBot keeps message intake, persona, plugin hooks and tools; Codex (driven
in-process through the ``codex_astrbot`` binding) runs the agent loop. Plugin
tools are exposed as Codex dynamic tools — deferred and called from code mode
by default — and executed back in AstrBot with the triggering event.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import time
import typing as T
from pathlib import Path

from astrbot.core import db_helper, logger, sp
from astrbot.core.message.components import Json
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.permission_rate_limit import keep_rate_limit_use
from astrbot.core.permission_rules import DEFAULT_POLICY, PermissionPolicy
from astrbot.core.permission_rules import EVENT_EXTRA_KEY as POLICY_EXTRA_KEY
from astrbot.core.platform.message_type import MessageType
from astrbot.core.provider.entities import LLMResponse, ProviderRequest, TokenUsage
from astrbot.core.utils.model_markup import ModelMarkupStream, strip_model_markup

from ...hooks import BaseAgentRunHooks
from ...response import AgentResponseData, AgentStats
from ...run_context import ContextWrapper, TContext
from ..base import AgentResponse, AgentState, BaseAgentRunner
from .constants import (
    CODEX_RUNNER_TYPE,
    CODEX_THREAD_STATE_KEY,
    DEFAULT_SYSTEM_PROMPT,
    NATIVE_EXEC_SESSION_KEY,
)
from .native import (
    ACTIVE_TURNS,
    TERMINAL_EVENTS,
    TURN_ID_FIELD,
    ActiveTurn,
    CodexEngine,
    JsonObject,
    SessionBusy,
    find_code_mode_host,
    find_codex_exe,
    session_slot,
)
from .tool_bridge import CodexToolBridge
from .usage import FIRST_TOKEN_EVENTS, UsageMeter, thread_usage

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

_FINAL_PHASES = (None, "final_answer")
# A steered follow-up that reached Codex only as its turn ended is answered by
# a continuation turn carrying this note (B13).
CONTINUE_NOTE = (
    '<request_context name="follow_up">\n'
    "The user sent the message(s) above while you were finishing your previous "
    "reply. Reply to them now.\n"
    "</request_context>"
)
MAX_CONTINUATIONS = 2
# Shown instead of an answer when a chat already has too many turns waiting.
BUSY_NOTE = "我这边还在处理前面的消息，稍后再发一次吧。"
# Appended when a follow-up was steered in but the turn to answer it could not
# be started: without it the follow-up would look answered.
FOLLOW_UP_DROPPED_NOTE = "（后面补充的消息没能处理，请再发一次。）"
CODE_MODES = ("code_mode", "code_mode_only")
STOPPED_NOTE = "（已中断）"


class _StoppedWhileQueued(Exception):
    """Stopped before the chat was free: the turn never reaches Codex."""


def _data_path() -> Path:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path

    return Path(get_astrbot_data_path())


def _default_cwd(umo: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in umo)
    path = _data_path() / "codex_workspaces" / safe
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def engine_options(cfg: dict) -> JsonObject:
    """Process-wide Codex options derived from the runner config."""
    codex_home = str(cfg.get("codex_home") or "") or str(_data_path() / "codex_home")
    tool_mode = cfg.get("tool_mode") or "code_mode_only"
    # Shipyard mode keeps every file operation inside the sandbox, so Codex
    # never gets its own shell on this host, whatever native_exec_tools says.
    native_exec = bool(cfg.get("native_exec_tools")) and not cfg.get("shipyard_mode")
    config: JsonObject = {
        # Chat-bot defaults: no coding-assistant scaffolding in the prompt.
        "include_permissions_instructions": False,
        "include_environment_context": False,
        "include_apps_instructions": False,
        "include_collaboration_mode_instructions": False,
        "skills.include_instructions": False,
        "project_doc_max_bytes": 0,
        "agents.enabled": False,
        "tools.experimental_request_user_input.enabled": False,
        "features.shell_tool": native_exec,
        "web_search": "live" if cfg.get("web_search") else "disabled",
        "model_tool_mode": tool_mode,
        "features.code_mode.structured_dynamic_tool_results": True,
        "features.code_mode.compact_exec_description": True,
        # Deferred tools stay out of the prompt prefix; this names them, with
        # short descriptions, in history and appends loads and unloads.
        "features.code_mode.tool_catalog": True,
        # The persona and other standing context are sent once and then kept
        # by history; compaction drops them, so Codex puts them back. A long
        # persona must not lose its middle either. The budget counts 4 bytes
        # a token, so Chinese text runs about 1.2x over it; 6000 keeps one
        # value under Codex's 10K-token limit for a single context item.
        "additional_context.reinject_after_compaction": True,
        "additional_context.max_value_tokens": 6000,
        "features.code_mode.exec_as_function_tool": bool(
            cfg.get("exec_as_function_tool")
        ),
        "approval_policy": cfg.get("approval_policy") or "never",
        "sandbox_mode": cfg.get("sandbox") or "read-only",
        # Codex reports skill invocations (by name), MCP tool calls and thread
        # metadata to chatgpt.com whenever an account is signed in. A chat bot
        # runs other people's conversations, so this is off unless the operator
        # turns it back on through thread_config.
        "analytics.enabled": False,
        # Inert today (the binding builds no OTEL provider, so every metric is
        # a no-op) but the built-in default is a Statsig exporter, so pin it.
        "otel.metrics_exporter": "none",
    }
    if model := cfg.get("model"):
        config["model"] = model
    if provider := cfg.get("model_provider"):
        config["model_provider"] = provider
    if effort := cfg.get("reasoning_effort"):
        config["model_reasoning_effort"] = effort
    if proxy := str(cfg.get("proxy") or "").strip():
        # Only Codex's own clients use it; the rest of AstrBot is unaffected.
        config["outbound_proxy"] = proxy
    config.update(model_provider_overrides(cfg.get("model_providers")))
    config.update(dict(cfg.get("thread_config") or {}))
    # ChatGPT apps (connectors) would hand every chat the signed-in account's
    # connected data (mail, drive, ...). Off for every thread, and set after
    # thread_config so it cannot be turned back on.
    config["features.apps"] = False
    options: JsonObject = {"codex_home": codex_home, "config": config}
    if tool_mode in CODE_MODES:
        host = find_code_mode_host(str(cfg.get("code_mode_host") or ""))
        if host:
            options["code_mode_host"] = host
        else:
            logger.warning(
                "codex-code-mode-host not found; set code_mode_host or install Codex CLI. "
                "code_mode_only turns will fail."
            )
    # The executable is what gives Codex a local execution environment, which
    # only native execution needs. Memory consolidation used to need it too;
    # it now maintains its files through the memories extension's file tools,
    # which run in-process.
    if native_exec:
        if exe := find_codex_exe(str(cfg.get("codex_self_exe") or "")):
            options["codex_self_exe"] = exe
        else:
            logger.warning(
                "No codex executable found, so native execution cannot run: "
                "without it Codex has no local execution environment. Chat and "
                "memory consolidation are unaffected. Set codex_self_exe, or "
                "reinstall the binding with CODEX_ASTRBOT_WITH_CODEX=1."
            )
    if native_exec and (cfg.get("approval_policy") or "never") == "never":
        # Every native command asks for approval; AstrBot answers it from the
        # sender's permission rule (native_exec_decision).
        options["approve_every_command"] = True
        config.pop("approval_policy", None)
    return options


def _without_nulls(value: T.Any) -> T.Any:
    """`value` with nested nulls left out (TOML has none)."""
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_nulls(v) for v in value if v is not None]
    return value


# How long a provider keeps the images uploaded to its Files API.
FILES_API_EXPIRES_SECONDS = 86400


def model_provider_overrides(providers: T.Any) -> JsonObject:
    """Map WebUI-managed providers to Codex overrides: `model_providers.<id>`
    (endpoint, key, extra request headers) and `model_provider_options.<id>`
    (compaction mode, wire, extra chat request body, images through the Files
    API, metadata of the provider's models). The chat wire is a Codex fork option: the provider's
    own `wire_api` stays Responses (upstream rejects "chat")."""
    out: JsonObject = {}
    for p in providers if isinstance(providers, list) else []:
        if not isinstance(p, dict):
            continue
        pid = re.sub(r"[^A-Za-z0-9_-]", "_", str(p.get("id") or "").strip())
        base_url = str(p.get("base_url") or "").strip()
        if not pid or not base_url:
            continue
        prefix = f"model_providers.{pid}"
        out[f"{prefix}.name"] = str(p.get("name") or pid)
        out[f"{prefix}.base_url"] = base_url
        out[f"{prefix}.wire_api"] = "responses"
        # Codex's turn metadata header is an OpenAI convention; DeepSeek takes
        # a request carrying it as Codex's and thinks whatever the effort.
        out[f"model_provider_options.{pid}.omit_turn_metadata"] = True
        if str(p.get("wire_api") or "") == "chat":
            out[f"model_provider_options.{pid}.wire"] = "chat"
            extra_body = p.get("extra_body")
            if isinstance(extra_body, dict):
                # A null removes a field Codex sends; TOML has no null, so
                # Codex takes those as a list.
                if fields := {
                    k: _without_nulls(v) for k, v in extra_body.items() if v is not None
                }:
                    out[f"model_provider_options.{pid}.extra_body"] = fields
                if removed := [k for k, v in extra_body.items() if v is None]:
                    out[f"model_provider_options.{pid}.extra_body_remove"] = removed
        if key := str(p.get("api_key") or "").strip():
            out[f"{prefix}.experimental_bearer_token"] = key
        headers = p.get("headers")
        if isinstance(headers, dict) and (
            headers := {
                str(name).strip(): str(value)
                for name, value in headers.items()
                if str(name).strip()
            }
        ):
            out[f"{prefix}.http_headers"] = headers
        # Images uploaded once to the provider's Files API (DeepSeek's,
        # OpenAI's), not sent again inline with every request.
        if p.get("files_api") is True:
            out[f"model_provider_options.{pid}.files_api"] = {
                "expires_seconds": FILES_API_EXPIRES_SECONDS
            }
        compaction = str(p.get("compaction") or "auto")
        if compaction in ("local", "remote"):
            out[f"model_provider_options.{pid}.compaction"] = compaction
        models: JsonObject = {}
        for m in p.get("models") if isinstance(p.get("models"), list) else []:
            slug = str(m.get("slug") or "").strip() if isinstance(m, dict) else ""
            if not slug:
                continue
            info: JsonObject = {}
            for field in ("context_window", "auto_compact_token_limit"):
                with contextlib.suppress(TypeError, ValueError):
                    if (value := int(m.get(field) or 0)) > 0:
                        info[field] = value
            efforts = m.get("reasoning_efforts")
            efforts = (
                [str(e) for e in efforts if e] if isinstance(efforts, list) else []
            )
            if efforts:
                info["supported_reasoning_levels"] = [
                    {"effort": e, "description": e} for e in efforts
                ]
            default = str(m.get("default_reasoning_effort") or "")
            # A default the model does not list is left out.
            if default and (not efforts or default in efforts):
                info["default_reasoning_level"] = default
            # Unset keeps Codex's default (text and images).
            if isinstance(m.get("image_input"), bool):
                info["input_modalities"] = (
                    ["text", "image"] if m["image_input"] else ["text"]
                )
            # Any other model info fields, over the form's.
            try:
                extra = json.loads(str(m.get("metadata_json") or "").strip() or "{}")
            except json.JSONDecodeError as exc:
                logger.warning("Codex model %s: metadata JSON ignored: %s", slug, exc)
                extra = {}
            if isinstance(extra, dict):
                info.update(extra)
            models[slug] = info
        if models:
            out[f"model_provider_options.{pid}.models"] = models
    return out


def event_policy(event: T.Any) -> PermissionPolicy:
    """The permission policy resolved for the event's sender."""
    get_extra = getattr(event, "get_extra", None)
    policy = get_extra(POLICY_EXTRA_KEY) if callable(get_extra) else None
    return policy if isinstance(policy, PermissionPolicy) else DEFAULT_POLICY


def turn_scopes(event: T.Any) -> list[str]:
    """Scopes a Codex turn of this event's sender carries.

    Their permission scopes, plus one naming who they are (platform, sender
    and role). Codex compares a turn's scopes before letting input join it or
    letting a code cell from an earlier turn call tools in it, so the marker
    keeps two people -- or one person under another role, like a task created
    through the API -- from acting with each other's rights, even when their
    permissions happen to match.
    """
    platform = sender = role = ""
    voice = False
    with contextlib.suppress(Exception):
        platform = str(event.get_platform_id() or "")
        sender = str(event.get_sender_id() or "")
        role = str(getattr(event, "role", "") or "member")
        voice = bool(event.get_extra("voice_turn"))
    scopes = [*event_policy(event).scopes, f"principal:{platform}:{sender}:{role}"]
    # A voice turn's answer is only spoken: the same person's text must not
    # join it, but wait for its own turn.
    return [*scopes, "via:voice"] if voice else scopes


def memory_thread_config(cfg: dict, umo: str, event: T.Any) -> JsonObject:
    """Per-thread Codex memory settings (R16–R18).

    Each chat gets its own local store (scope = UMO). Writing a shared memory
    or deleting one is up to the sender of each turn (``turn_scopes``): each
    turn carries its sender's permission scopes, so whoever has a rule
    granting ``global_memory`` may do it in a group as well as in private,
    e.g. to correct the bot. The thread's tools and prompt stay the same for
    everyone, so switching senders costs no cache.

    Automatic consolidation still promotes to the global store only from a
    private chat of such a sender: a group's transcript mixes many people's
    details. That flag is fixed when the thread starts (Codex keeps it
    sticky-false), so a rule change applies to it from the next thread.

    Args:
        cfg: Runner config.
        umo: The chat, which is also its memory scope.
        event: The event starting the thread, for its sender's rule.

    Returns:
        Codex config overrides for the thread.
    """
    # By the chat's type, not the event's group id, which a scheduled task
    # created before tasks recorded their group (or through the API) lacks.
    is_private = False
    with contextlib.suppress(Exception):
        is_private = event.get_message_type() == MessageType.FRIEND_MESSAGE
    trusted = event_policy(event).global_memory is True
    return {
        "features.memories": True,
        "memories.dedicated_tools": True,
        "memories.extra_session_sources": ["astrbot"],
        "memories.scope_key": umo,
        "memories.may_write_global": bool(is_private and trusted),
        # Deleting is decided per turn, by its sender's scopes.
        "memories.may_delete": False,
        "memories.turn_scopes": True,
        "memories.auto_consolidate": bool(cfg.get("memory_auto_consolidate", True)),
    }


def approvals_disabled(cfg: dict) -> bool:
    """An explicit non-"never" approval policy with auto_approve off denies
    every approval request, as the setting documents. With the default
    "never" policy and native exec on, approve_every_command is used and the
    permission rules decide instead."""
    policy = str(cfg.get("approval_policy") or "never")
    return policy != "never" and not cfg.get("auto_approve")


def native_exec_decision(
    event: T.Any, session_enabled: bool | None = None
) -> tuple[bool, str]:
    """Approve native execution unless this chat turned it off (K3) or the
    sender's rule sets native_exec: false. Deciding per command keeps the
    thread's tool set, history and prompt cache unchanged when toggled."""
    if session_enabled is False:
        return False, "Native command execution is turned off in this chat."
    get_extra = getattr(event, "get_extra", None)
    policy = get_extra(POLICY_EXTRA_KEY) if callable(get_extra) else None
    if isinstance(policy, PermissionPolicy) and policy.native_exec is False:
        logger.info("Codex native execution denied by rule %r", policy.rule_name)
        return False, "Native command execution is not permitted for this user."
    return True, ""


def system_prompt(cfg: dict) -> str:
    custom = str(cfg.get("base_instructions") or "").strip()
    return custom or DEFAULT_SYSTEM_PROMPT


# Codex saves a generated image under CODEX_HOME, on the host and outside every
# tool the model has -- the message tools refuse CODEX_HOME outright. So the
# image is copied into the chat's workspace and the model is told where, and it
# decides whether to send it or keep working on it. Both the hosted Responses
# item ("ImageGeneration") and the standalone extension item
# ("image_gen.generation") carry the same `saved_path`.
_IMAGE_ITEM_TYPES = ("ImageGeneration",)
_IMAGE_ITEM_KIND = "image_gen.generation"
#: Workspace-relative directory generated images are copied into. The same
#: relative path resolves for send_message_to_user in every runtime: the host
#: workspace is tried first, then the sandbox.
GENERATED_IMAGE_DIR = "generated_images"
#: Where Shipyard Neo mounts the sandbox workspace.
SANDBOX_WORKSPACE = "/workspace"


def generated_image_note(relative_path: str, where: str) -> str:
    """Tells the model where its generated image went and what to do with it."""
    return (
        f"The image was copied to {where}. It has NOT been sent to the user, and "
        "nothing will send it for you. To show it, call send_message_to_user with "
        f'{{"type": "image", "path": "{relative_path}"}}. You can also keep working '
        "with the file where it is."
    )


def generated_image_failure_note(error: str) -> str:
    """Tells the model its generated image could not be made reachable."""
    return (
        f"The image could not be copied into your workspace ({error}), so you "
        "cannot send or use it. Tell the user it could not be delivered."
    )


def request_context(name: str, text: str) -> str:
    """Wraps host text given to the model as input rather than as a tool result."""
    return f'<request_context name="{name}">\n{text}\n</request_context>'


def generated_image_path(item: JsonObject) -> str | None:
    """Path of a completed generated image, if this item is one."""
    is_image = (
        item.get("type") in _IMAGE_ITEM_TYPES or item.get("kind") == _IMAGE_ITEM_KIND
    )
    if not is_image or item.get("status") != "completed":
        return None
    path = item.get("saved_path")
    if not isinstance(path, str) or not path:
        return None
    return path if os.path.isfile(path) else None


def _part_to_input(part: T.Any) -> JsonObject | None:
    data = (
        part.model_dump_for_context()
        if hasattr(part, "model_dump_for_context")
        else part
    )
    if not isinstance(data, dict):
        return None
    if data.get("type") == "text" and data.get("text"):
        return {"type": "text", "text": data["text"], "text_elements": []}
    if data.get("type") == "image_url":
        url = (data.get("image_url") or {}).get("url")
        if url:
            return {"type": "image", "image_url": url}
    return None


def _image_input(ref: str) -> JsonObject:
    if ref.startswith(("http://", "https://", "data:")):
        return {"type": "image", "image_url": ref}
    if ref.startswith("file:///"):
        ref = ref[len("file:///") :]
    return {"type": "local_image", "path": ref}


def build_turn_input(req: ProviderRequest) -> list[JsonObject]:
    """Flatten a ProviderRequest's per-message content into Codex user input."""
    items: list[JsonObject] = []
    for part in [*req.dynamic_user_context_parts, *req.persistent_user_context_parts]:
        if item := _part_to_input(part):
            items.append(item)
    prompt = (req.prompt or "").strip()
    if prompt:
        items.append({"type": "text", "text": prompt, "text_elements": []})
    for part in req.extra_user_content_parts:
        if item := _part_to_input(part):
            items.append(item)
    for ref in req.image_urls:
        items.append(_image_input(ref))
    for ref in req.audio_urls:
        if ref.startswith(("http://", "https://", "data:")):
            items.append({"type": "audio", "audio_url": ref})
        elif os.path.exists(ref):
            items.append({"type": "local_audio", "path": ref})
    if not items:
        items.append({"type": "text", "text": "<empty message>", "text_elements": []})
    return items


def build_additional_context(req: ProviderRequest) -> dict[str, JsonObject]:
    """Standing instructions: re-sent by Codex only when their value changes."""
    ctx: dict[str, JsonObject] = {}
    if req.system_prompt and req.system_prompt.strip():
        ctx["astrbot_system_prompt"] = {
            "value": req.system_prompt.strip(),
            "kind": "application",
        }
    for name, content in req.context_anchors.items():
        ctx[f"astrbot_{name}"] = {"value": content, "kind": "application"}
    return ctx


GROUP_HISTORY_RESTORE_KEY = "_group_context_restore"
GROUP_MESSAGE_FORGET_KEY = "_group_context_forget"
_META_PREFIX = '<context_unit name="message_meta"'
_GROUP_HISTORY_PREFIX = '<context_unit name="group_history"'


async def release_group_history(event: T.Any) -> None:
    """Gives back the group history a request took, when it never ran.

    Group chat context hands each earlier message to exactly one request; a
    request refused or failed before the model saw it must return them, or
    they are lost.
    """
    await give_back(take_group_history(event))


async def forget_group_message(event: T.Any) -> None:
    """Removes a refused triggering message from the group history, so no
    later request answers it on the sender's behalf."""
    get_extra = getattr(event, "get_extra", None)
    forget = get_extra(GROUP_MESSAGE_FORGET_KEY) if callable(get_extra) else None
    if not callable(forget):
        return
    try:
        await forget()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not drop a refused message from group history: %s", e)


def take_group_history(event: T.Any) -> T.Any:
    """Claims the give-back of a request's group history, synchronously.

    Returns the callback that gives it back, or None when there is nothing to
    give back (none taken, already kept or already claimed). Claiming in the
    same step as the decision leaves no gap for a run to keep it meanwhile.
    """
    get_extra = getattr(event, "get_extra", None)
    if not callable(get_extra):
        return None
    restore = get_extra(GROUP_HISTORY_RESTORE_KEY)
    if not callable(restore):
        return None
    keep_group_history(event)
    return restore


async def give_back(restore: T.Any) -> None:
    """Runs a callback from take_group_history; None is a no-op."""
    if restore is None:
        return
    try:
        await restore()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not give back group history: %s", e)


def keep_group_history(event: T.Any) -> None:
    """The model has the history now; it must not be given back later."""
    set_extra = getattr(event, "set_extra", None)
    if callable(set_extra):
        set_extra(GROUP_HISTORY_RESTORE_KEY, None)


def drop_group_history(req: ProviderRequest | None) -> None:
    """Removes the group history a request carries, after it was given back."""
    if req is None:
        return
    req.persistent_user_context_parts = [
        part
        for part in req.persistent_user_context_parts
        if not str(getattr(part, "text", "")).startswith(_GROUP_HISTORY_PREFIX)
    ]


def _message_start(turn_input: list[JsonObject]) -> int:
    """Index where the message being answered starts: its metadata, or the
    first item after the leading context units."""
    for index, item in enumerate(turn_input):
        if str(item.get("text") or "").startswith(_META_PREFIX):
            return index
    for index, item in enumerate(turn_input):
        text = str(item.get("text") or "")
        if not text.startswith(("<context_unit", "<request_context")):
            return index
    return len(turn_input)


def speaker_change_note(
    previous: JsonObject | None, current: JsonObject
) -> JsonObject | None:
    """Input item flagging that this turn comes from a different person.

    One thread serves the whole group, so consecutive turns can come from
    different senders; the model otherwise tends to answer them as one person
    and carry one sender's request over into the other's reply.
    """
    prev_id = str((previous or {}).get("id") or "")
    cur_id = str(current.get("id") or "")
    if not prev_id or not cur_id or prev_id == cur_id:
        return None
    prev_label = str((previous or {}).get("label") or prev_id)
    cur_label = str(current.get("label") or cur_id)
    text = (
        "<speaker_change>\n"
        f"The person talking to you changed: this message is from {cur_label}, "
        f"not {prev_label} who sent the previous request. Treat it as a new "
        "request from a different person; do not merge it with the previous "
        "sender's requests or attribute either person's words to the other.\n"
        "</speaker_change>"
    )
    return {"type": "text", "text": text, "text_elements": []}


PERSONA_ANCHORS = ("persona", "persona_examples", "default_persona")


def persona_context(
    req: ProviderRequest, catalog: dict[str, str]
) -> tuple[dict[str, str], dict[str, JsonObject], JsonObject | None]:
    """Cache-friendly personas when one thread serves several (B16).

    With a single persona per thread the anchors are sent as before. Once a
    second persona shows up (per-user persona rules in a group), all personas
    seen in the thread go out as one catalog that only changes when a new one
    appears, and each turn just names the active one, instead of re-sending
    the full persona text every time speakers alternate.

    Returns (updated catalog, additional_context, per-turn input item or None).
    """
    context = build_additional_context(req)
    block = "\n\n".join(
        req.context_anchors[name]
        for name in PERSONA_ANCHORS
        if req.context_anchors.get(name)
    )
    key = hashlib.sha1(block.encode("utf-8")).hexdigest()[:8] if block else ""
    updated = dict(catalog)
    if key:
        updated.setdefault(key, block)
    if len(updated) <= 1:
        return updated, context, None
    for name in PERSONA_ANCHORS:
        context.pop(f"astrbot_{name}", None)
    catalog_text = "\n\n".join(
        f'<persona id="{k}">\n{v}\n</persona>' for k, v in updated.items()
    )
    context["astrbot_personas"] = {
        "value": "Several personas are used in this chat; each user message names "
        "the one to use for that reply.\n\n" + catalog_text,
        "kind": "application",
    }
    active = (
        f'<active_persona id="{key}"/>'
        if key
        else "<active_persona>none: reply as the plain assistant</active_persona>"
    )
    return updated, context, {"type": "text", "text": active, "text_elements": []}


class CodexAgentRunner(BaseAgentRunner[TContext]):
    """In-process Codex agent runner."""

    @override
    async def reset(
        self,
        request: ProviderRequest,
        run_context: ContextWrapper[TContext],
        agent_hooks: BaseAgentRunHooks[TContext],
        provider_config: dict,
        **kwargs: T.Any,
    ) -> None:
        self.req = request
        self._additional_context: dict[str, JsonObject] = {}
        self._active_persona: JsonObject | None = None
        self.run_context = run_context
        self.agent_hooks = agent_hooks
        self.cfg = provider_config
        self.streaming = bool(kwargs.get("streaming", False))
        self.final_llm_resp: LLMResponse | None = None
        self._state = AgentState.IDLE
        self.umo = request.session_id or ""
        tool_mode = self.cfg.get("tool_mode") or "code_mode_only"
        event = getattr(getattr(run_context, "context", None), "event", None)
        get_extra = getattr(event, "get_extra", None)
        policy = get_extra(POLICY_EXTRA_KEY) if callable(get_extra) else None
        self.bridge = CodexToolBridge(
            request.func_tool,
            defer=tool_mode in CODE_MODES,
            policy=policy if isinstance(policy, PermissionPolicy) else None,
        )
        self._engine: CodexEngine | None = None
        self._thread_id: str | None = None
        self._speaker_change: JsonObject | None = None
        self._turn_running = False
        self._aborted = False
        self._active: ActiveTurn | None = None
        # Usage of this run, in the shape the stats page and WebChat read.
        self.stats = AgentStats()
        self._usage = UsageMeter()
        # The thread's model and provider, read when the turn closes.
        self._usage_after: JsonObject | None = None
        self._stats_recorded = False

    # ------------------------------------------------------------------ public

    @override
    async def step(self) -> T.AsyncGenerator[AgentResponse, None]:
        if self._state == AgentState.IDLE:
            try:
                await self.agent_hooks.on_agent_begin(self.run_context)
            except Exception as e:  # noqa: BLE001
                logger.error("Error in on_agent_begin hook: %s", e, exc_info=True)
        self._transition_state(AgentState.RUNNING)
        try:
            async for resp in self._run_turn():
                yield resp
        except Exception as e:  # noqa: BLE001
            logger.error("Codex runner failed: %s", e, exc_info=True)
            msg = f"Codex 请求失败：{e!s}"
            self._transition_state(AgentState.ERROR)
            self.final_llm_resp = LLMResponse(role="err", completion_text=msg)
            await self._record_stats("error")
            yield AgentResponse(
                type="err", data=AgentResponseData(chain=MessageChain().message(msg))
            )
        finally:
            # Anything that ended the run before Codex accepted the turn (a
            # busy chat, an engine or thread failure, a failed submit,
            # cancellation) gives the group history back; an accepted turn
            # has already kept it, which makes this a no-op.
            await release_group_history(self._event())

    @override
    async def step_until_done(
        self, max_step: int = 30
    ) -> T.AsyncGenerator[AgentResponse, None]:
        while not self.done():
            async for resp in self.step():
                yield resp

    @override
    def done(self) -> bool:
        return self._state in (AgentState.DONE, AgentState.ERROR)

    @override
    def get_final_llm_resp(self) -> LLMResponse | None:
        return self.final_llm_resp

    def request_stop(self) -> None:
        self._aborted = True
        if self._active is not None:
            self._active.aborted = True
        if self._turn_running and self._engine and self._thread_id:
            asyncio.ensure_future(self._engine.interrupt(self._thread_id))

    def was_aborted(self) -> bool:
        return self._aborted

    async def close(self) -> None:
        if self._turn_running and self._engine and self._thread_id:
            await self._engine.interrupt(self._thread_id)

    def _event(self) -> T.Any:
        return getattr(getattr(self.run_context, "context", None), "event", None)

    def _message_id(self) -> str | None:
        event = getattr(getattr(self.run_context, "context", None), "event", None)
        message_obj = getattr(event, "message_obj", None)
        mid = getattr(message_obj, "message_id", None)
        return str(mid) if mid else None

    def _sender_id(self) -> str:
        event = getattr(getattr(self.run_context, "context", None), "event", None)
        try:
            return str(event.get_sender_id()) if event is not None else ""
        except Exception:  # noqa: BLE001
            return ""

    def _sender(self) -> JsonObject:
        """Who triggered this turn: id (for comparing) and a display label.

        Empty for scheduled and background-completion turns: those are the
        system reporting back, not a person talking.
        """
        event = getattr(getattr(self.run_context, "context", None), "event", None)
        try:
            # Voice requests come as synthetic events too, but from a person.
            if (
                event is not None
                and event.get_platform_name() == "cron"
                and not event.get_extra("voice_turn")
            ):
                return {"id": "", "label": ""}
        except Exception:  # noqa: BLE001
            pass
        sender_id = self._sender_id()
        try:
            name = str(event.get_sender_name() or "") if event is not None else ""
        except Exception:  # noqa: BLE001
            name = ""
        # With the id, as group metadata and history show it: nicknames repeat.
        label = f"{name} (ID: {sender_id})" if name and sender_id else name or sender_id
        return {"id": sender_id, "label": label}

    # ---------------------------------------------------------------- threads

    def _thread_params(self) -> JsonObject:
        cwd = str(self.cfg.get("cwd") or "") or _default_cwd(self.umo)
        params: JsonObject = {
            "cwd": cwd,
            "base_instructions": system_prompt(self.cfg),
            "dynamic_tools": self.bridge.dynamic_tools(),
            "no_environment": not self.cfg.get("native_exec_tools"),
        }
        if extra := str(self.cfg.get("developer_instructions") or "").strip():
            params["developer_instructions"] = extra
        if self.cfg.get("memory_enabled"):
            event = getattr(getattr(self.run_context, "context", None), "event", None)
            params["config"] = memory_thread_config(self.cfg, self.umo, event)
        return params

    async def _open_thread(self, engine: CodexEngine) -> tuple[str, list | None]:
        """Return the thread id and the tool set to send if it changed."""
        state = await sp.get_async(
            scope="umo", scope_id=self.umo, key=CODEX_THREAD_STATE_KEY, default={}
        )
        state = state if isinstance(state, dict) else {}
        info, started_new = await engine.open_thread(state, self._thread_params())
        self._turn_sender = self._sender()
        previous = None if started_new else state.get("last_sender")
        previous = previous if isinstance(previous, dict) else None
        self._speaker_change = speaker_change_note(previous, self._turn_sender)
        catalog = {} if started_new else state.get("personas") or {}
        catalog, self._additional_context, self._active_persona = persona_context(
            self.req, catalog if isinstance(catalog, dict) else {}
        )
        tools_update = None
        if not started_new and state.get("tools_fp") != self.bridge.fingerprint:
            # Replace the tool set in place; history and cache prefix are kept.
            tools_update = self.bridge.dynamic_tools()
            logger.info(
                "AstrBot tool set changed for umo=%s; updating thread tools.", self.umo
            )
        await sp.put_async(
            scope="umo",
            scope_id=self.umo,
            key=CODEX_THREAD_STATE_KEY,
            value={
                "thread_id": info["thread_id"],
                "rollout_path": info.get("rollout_path") or state.get("rollout_path"),
                "tools_fp": self.bridge.fingerprint,
                "personas": catalog,
                # Who triggered the latest turn the model saw; updated once
                # this turn is accepted (_remember_sender).
                "last_sender": previous,
            },
        )
        return info["thread_id"], tools_update

    async def _remember_sender(self, thread_id: str) -> None:
        """Records this turn's sender once Codex accepted the turn."""
        sender = getattr(self, "_turn_sender", None)
        if not sender or not sender.get("id"):
            return
        try:
            state = await sp.get_async(
                scope="umo", scope_id=self.umo, key=CODEX_THREAD_STATE_KEY, default={}
            )
            if not isinstance(state, dict) or state.get("thread_id") != thread_id:
                return
            await sp.put_async(
                scope="umo",
                scope_id=self.umo,
                key=CODEX_THREAD_STATE_KEY,
                value={**state, "last_sender": sender},
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not record the turn's sender: %s", e)

    # ------------------------------------------------------------------- turn

    async def _handle_tool_call(self, msg: JsonObject) -> JsonObject:
        return await self.bridge.call(msg, self.run_context, self.agent_hooks)

    async def _handle_approval(self, kind: str, msg: JsonObject) -> tuple[bool, str]:
        """Native exec / patch approvals follow the sender's permission rule (B15)."""
        if approvals_disabled(self.cfg):
            return False, "Approval requests are disabled (auto_approve is off)."
        session_setting = await sp.get_async(
            scope="umo", scope_id=self.umo, key=NATIVE_EXEC_SESSION_KEY, default=None
        )
        return native_exec_decision(
            getattr(getattr(self.run_context, "context", None), "event", None),
            session_enabled=session_setting
            if isinstance(session_setting, bool)
            else None,
        )

    def _turn_request(self, tools_update: list | None) -> JsonObject:
        turn_input = build_turn_input(self.req)
        if self._speaker_change is not None:
            # Right before the message it describes: after the earlier group
            # chatter, in front of the sender metadata.
            turn_input.insert(_message_start(turn_input), self._speaker_change)
        if self._active_persona is not None:
            turn_input.insert(0, self._active_persona)
        request: JsonObject = {
            "input": turn_input,
            "mode": "start_or_steer",
            "additional_context": self._additional_context,
        }
        if tools_update is not None:
            request["dynamic_tools"] = tools_update
        if self.req.model:
            request["model"] = self.req.model
        # The sender's capabilities travel with their turn (scheduled and
        # background tasks run as whoever created them); Codex's tools check
        # them when they run.
        request["scopes"] = turn_scopes(self._event())
        return request

    def _computer_runtime(self) -> str:
        """Execution environment this chat's tools run in, as codex_request sets it."""
        if self.cfg.get("shipyard_mode"):
            return "sandbox"
        try:
            ctx = self.run_context.context.context  # type: ignore[attr-defined]
            settings = ctx.get_config(umo=self.umo).get("provider_settings") or {}
        except Exception:  # noqa: BLE001 - no config means no execution runtime
            return "none"
        return str(settings.get("computer_use_runtime") or "none")

    async def _place_generated_image(self, path: str) -> str:
        """Copies a generated image into the chat's workspace.

        The sandbox workspace when tools run there, the host workspace
        otherwise -- the same place a relative path given to
        send_message_to_user resolves to.

        Returns:
            The note to give the model: where the image is, or why it is not
            anywhere it can reach.
        """
        from astrbot.core.computer.computer_client import get_booter
        from astrbot.core.tools.computer_tools.util import (
            workspace_root,
            workspace_root_for_context,
        )

        relative = f"{GENERATED_IMAGE_DIR}/{os.path.basename(path)}"
        runtime = self._computer_runtime()
        try:
            if runtime == "sandbox":
                ctx = self.run_context.context.context  # type: ignore[attr-defined]
                booter = await get_booter(ctx, self.umo)
                await booter.upload_file(path, relative)
                where = (
                    f"`{relative}` in your sandbox workspace "
                    f"(`{SANDBOX_WORKSPACE}/{relative}`)"
                )
            else:
                root = (
                    await workspace_root_for_context(self.run_context)  # type: ignore[arg-type]
                    if runtime == "local"
                    else workspace_root(self.umo)
                )
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(shutil.copyfile, path, target)
                where = f"`{relative}` in your workspace"
        except Exception as e:  # noqa: BLE001 - sandbox or filesystem failure
            logger.warning("Could not place generated image %s: %s", path, e)
            return generated_image_failure_note(str(e))
        logger.info("Generated image placed at %s (%s runtime)", relative, runtime)
        return generated_image_note(relative, where)

    async def _on_saved_image(self, call_id: str, saved_path: str) -> str | None:
        """Places an image Codex just saved; the result is the tool's output."""
        if not os.path.isfile(saved_path):
            return None
        self._hooked_images.add(saved_path)
        return await self._place_generated_image(saved_path)

    async def _steer_note(
        self, engine: CodexEngine, thread_id: str, active: ActiveTurn | None, note: str
    ) -> bool:
        """Adds a note to the running turn; False when the turn is already over."""
        if active is None or not active.turn_id:
            return False
        try:
            result = await engine.submit_turn(
                thread_id,
                {
                    "input": [{"type": "text", "text": note, "text_elements": []}],
                    "mode": "steer",
                    "expected_turn_id": active.turn_id,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not steer a note into %s: %s", thread_id, e)
            return False
        return result.get("status") == "steered"

    async def _run_turn(self) -> T.AsyncGenerator[AgentResponse, None]:
        engine = await CodexEngine.get(engine_options(self.cfg))
        self._engine = engine
        show_commentary = bool(self.cfg.get("show_commentary"))
        timeout = float(self.cfg.get("turn_timeout") or 600)
        max_queued = int(self.cfg.get("max_queued_turns") or 0)

        try:
            async with session_slot(engine, self.umo, max_queued):
                if self._aborted:
                    raise _StoppedWhileQueued
                # Opening the thread belongs in the critical section: it
                # is what decides the thread id, so two senders arriving
                # together on a new chat would otherwise each make one.
                thread_id, tools_update = await self._open_thread(engine)
                self._thread_id = thread_id
                # Images Codex saves during this turn are placed through the
                # engine's hook, so the model reads where the image went in the
                # tool result itself. The event-stream path below only covers
                # an image the hook did not handle.
                self._hooked_images = set()
                engine.saved_image_handlers[thread_id] = self._on_saved_image
                pump = engine.pump(thread_id)
                queue = pump.open_turn(self._handle_tool_call, self._handle_approval)
                phases: dict[str, str | None] = {}
                # Per agent-message item: strips citation markup the model
                # leaks, which may be split across streamed deltas.
                markup_streams: dict[str, ModelMarkupStream] = {}
                final_texts: list[str] = []
                commentary: list[str] = []
                reasoning: list[str] = []
                # Generated images already handled, and notes about them that
                # could not be steered in because the turn was ending.
                placed_images: set[str] = set()
                pending_notes: list[str] = []
                usage: JsonObject | None = None
                error_msg: str | None = None
                end = "task_complete"
                started = time.monotonic()
                active: ActiveTurn | None = None
                event_seq = last_user_seq = last_agent_seq = 0
                first_end = True
                continuations = 0
                # The agent-message count when a continuation answering a
                # steered follow-up was started; None when none was.
                follow_up_since: int | None = None
                # Follow-ups steered in before the latest continuation started.
                steered_before = 0
                # An error from before a continuation, and how many answers
                # existed then: it stands unless the continuation answers.
                carried_error: str | None = None
                answers_before = 0
                follow_up_dropped = False
                try:
                    # Registered before the submit, not after: a same-sender
                    # follow-up arriving during that round trip should wait for
                    # this turn and be steered into it, not become a second one.
                    active = ActiveTurn(
                        engine,
                        thread_id,
                        "",
                        self._sender_id(),
                        message_id=self._message_id(),
                    )
                    self._active = active
                    ACTIVE_TURNS[self.umo] = active
                    self._usage.start(await thread_usage(engine, thread_id))
                    self.stats.start_time = time.time()
                    try:
                        request = self._turn_request(tools_update)
                        sub = await engine.submit_turn(thread_id, request)
                        if sub.get("reason") == "ActiveTurnScopesMismatch":
                            # This chat's slot is ours, so a turn still running
                            # was left behind by a runner that stopped early.
                            # Stop it and start ours instead of joining it.
                            logger.warning(
                                "Codex thread %s still ran another sender's turn; "
                                "interrupting it.",
                                thread_id,
                            )
                            await engine.interrupt(thread_id)
                            sub = await engine.submit_turn(thread_id, request)
                        if sub.get("status") == "not_submitted":
                            raise RuntimeError(
                                f"Codex did not accept the turn: {sub.get('reason')}"
                            )
                        active.turn_id = str(sub.get("turn_id") or "")
                        # Running from here on: stopping or leaving must now
                        # interrupt it, even before the loop below starts.
                        self._turn_running = True
                        if self._aborted:
                            # Stopped during the submit, before stopping could
                            # interrupt anything.
                            await engine.interrupt(thread_id)
                    except BaseException as e:
                        active.aborted = True
                        if isinstance(e, asyncio.CancelledError):
                            # Codex may have started the turn before the
                            # cancellation reached us; do not leave it running.
                            with contextlib.suppress(Exception):
                                await asyncio.shield(engine.interrupt(thread_id))
                        raise
                    finally:
                        # Release a follow-up waiting on the turn id, including
                        # when the submit failed.
                        active.ready.set()
                    # The model has this turn's input now, group history too,
                    # so the request's rate-limit use is spent.
                    keep_group_history(self._event())
                    keep_rate_limit_use(self._event())
                    await self._remember_sender(thread_id)
                    while True:
                        remaining = timeout - (time.monotonic() - started)
                        if remaining <= 0:
                            await engine.interrupt(thread_id)
                            error_msg = f"Codex turn timed out after {timeout:.0f}s"
                            break
                        try:
                            msg = await asyncio.wait_for(queue.get(), remaining)
                        except asyncio.TimeoutError:
                            continue
                        kind = msg.get("type")
                        turn_of = msg.get(TURN_ID_FIELD)
                        if turn_of and active.turn_id and turn_of != active.turn_id:
                            # Another turn's (its end, words, images, usage): one
                            # interrupted just before ours still reports here.
                            continue
                        event_seq += 1
                        if (
                            not self.stats.time_to_first_token
                            and kind in FIRST_TOKEN_EVENTS
                        ):
                            # Floored: 0 means "not measured" to the stats.
                            self.stats.time_to_first_token = max(
                                time.time() - self.stats.start_time, 0.001
                            )
                        if kind == "user_message":
                            last_user_seq = event_seq
                        if kind == "item_started":
                            item = msg.get("item") or {}
                            if item.get("type") == "AgentMessage":
                                phases[item.get("id", "")] = item.get("phase")
                        elif kind == "item_completed":
                            item = msg.get("item") or {}
                            # Only streamed items have one; flush what it held.
                            markup = markup_streams.pop(item.get("id", ""), None)
                            tail = markup.finish() if markup else ""
                            if tail:
                                yield AgentResponse(
                                    type="streaming_delta",
                                    data=AgentResponseData(
                                        chain=MessageChain().message(tail)
                                    ),
                                )
                            path = generated_image_path(item)
                            if (
                                path
                                and path not in placed_images
                                and path not in self._hooked_images
                            ):
                                placed_images.add(path)
                                note = request_context(
                                    "generated_image",
                                    await self._place_generated_image(path),
                                )
                                if not await self._steer_note(
                                    engine, thread_id, active, note
                                ):
                                    pending_notes.append(note)
                        elif kind == "agent_message_content_delta":
                            item_id = msg.get("item_id", "")
                            phase = phases.get(item_id)
                            if self.streaming and (
                                phase in _FINAL_PHASES or show_commentary
                            ):
                                delta = markup_streams.setdefault(
                                    item_id, ModelMarkupStream()
                                ).push(msg.get("delta", ""))
                                if delta:
                                    yield AgentResponse(
                                        type="streaming_delta",
                                        data=AgentResponseData(
                                            chain=MessageChain().message(delta)
                                        ),
                                    )
                        elif kind == "agent_message":
                            last_agent_seq = event_seq
                            text = strip_model_markup(msg.get("message") or "")
                            if msg.get("phase") in _FINAL_PHASES:
                                final_texts.append(text)
                            else:
                                commentary.append(text)
                                if self.streaming and show_commentary:
                                    yield AgentResponse(
                                        type="streaming_delta",
                                        data=AgentResponseData(
                                            chain=MessageChain().message("\n\n")
                                        ),
                                    )
                        elif kind == "agent_reasoning":
                            reasoning.append(msg.get("text") or "")
                        elif kind == "token_count":
                            info = msg.get("info") or {}
                            self._usage.observe(info.get("total_token_usage"))
                            last = info.get("last_token_usage") or {}
                            # A real request, not an estimate after compaction
                            # or overflow (those carry no input count): it is
                            # what last filled the context.
                            if int(last.get("input_tokens") or 0) > 0:
                                usage = last
                                self.stats.current_context_tokens = int(
                                    last["input_tokens"]
                                )
                        elif kind == "error":
                            error_msg = msg.get("message") or str(msg)
                        elif kind in TERMINAL_EVENTS:
                            end = kind
                            # Codex's own measure, from the first turn only: a
                            # continuation times from its own start, and
                            # would understate what the user waited.
                            ttft_ms = msg.get("time_to_first_token_ms")
                            if (
                                first_end
                                and isinstance(ttft_ms, int | float)
                                and ttft_ms > 0
                            ):
                                self.stats.time_to_first_token = ttft_ms / 1000
                            first_end = False
                            # A follow-up steered in as the turn ended was recorded
                            # but never answered (B13): answer it in one more turn.
                            unanswered = (
                                active.steered > 0 and last_user_seq > last_agent_seq
                            )
                            # An image note that missed the turn gets the same
                            # treatment: without it the model never learns
                            # where its image went.
                            notes = [CONTINUE_NOTE] if unanswered else []
                            notes += pending_notes
                            pending_notes.clear()
                            if (
                                kind != "turn_aborted"
                                and notes
                                and not self._aborted
                                and continuations < MAX_CONTINUATIONS
                            ):
                                continuations += 1
                                try:
                                    again = await engine.submit_turn(
                                        thread_id,
                                        {
                                            "input": [
                                                {
                                                    "type": "text",
                                                    "text": note,
                                                    "text_elements": [],
                                                }
                                                for note in notes
                                            ],
                                            "mode": "start_if_idle",
                                            # The same sender's follow-up: their rights.
                                            "scopes": turn_scopes(self._event()),
                                        },
                                    )
                                except Exception as e:  # noqa: BLE001
                                    # Keep the answer this turn already has.
                                    logger.warning(
                                        "Codex continuation not submitted: %s", e
                                    )
                                    again = {}
                                if again.get("status") == "started":
                                    active.turn_id = str(again.get("turn_id") or "")
                                    follow_up_since = (
                                        last_agent_seq if unanswered else None
                                    )
                                    steered_before = active.steered
                                    if self._aborted:
                                        # Stopped while it was being submitted.
                                        await engine.interrupt(thread_id)
                                    if error_msg:
                                        carried_error = error_msg
                                        answers_before = _answer_count(final_texts)
                                        error_msg = None
                                    continue
                            # Also when out of continuations: a follow-up that
                            # no turn will answer must not look answered.
                            follow_up_dropped = (
                                unanswered
                                and kind != "turn_aborted"
                                and not self._aborted
                            )
                            break
                        elif kind == "_pump_closed":
                            reason = msg.get("message") or "Codex thread closed"
                            if not _answer_count(final_texts):
                                raise RuntimeError(reason)
                            # Keep what was already answered (e.g. before a
                            # continuation the shutdown cut short), but not
                            # as if a follow-up still waiting had been too.
                            error_msg = reason
                            if continuations:
                                # A follow-up the continuation carried and has
                                # not answered, or one steered into it since.
                                follow_up_dropped = (
                                    follow_up_since is not None
                                    and last_agent_seq <= follow_up_since
                                ) or (
                                    active is not None
                                    and active.steered > steered_before
                                    and last_user_seq > last_agent_seq
                                )
                            else:
                                follow_up_dropped = (
                                    active is not None
                                    and active.steered > 0
                                    and last_user_seq > last_agent_seq
                                )
                            break
                except BaseException:
                    if self._turn_running:
                        # Left mid-turn (cancelled, closed early): stop the turn,
                        # or it runs on with nobody reading it and the chat's
                        # next turn runs into it.
                        with contextlib.suppress(Exception):
                            await asyncio.shield(engine.interrupt(thread_id))
                    raise
                finally:
                    self._turn_running = False
                    if (
                        engine.saved_image_handlers.get(thread_id)
                        == self._on_saved_image
                    ):
                        engine.saved_image_handlers.pop(thread_id, None)
                    if active is not None and ACTIVE_TURNS.get(self.umo) is active:
                        ACTIVE_TURNS.pop(self.umo, None)
                    pump.close_turn()
                    # Still holding the chat's slot: once it is released the
                    # next queued turn may run on this thread, and its usage
                    # would land in this turn's closing total too.
                    if self.stats.start_time:
                        self._usage_after = await thread_usage(engine, thread_id)
                        self._usage.observe(
                            (self._usage_after or {}).get("total_token_usage")
                        )

        except SessionBusy as busy:
            logger.info(
                "Codex session %s is busy; %d turns already queued.",
                self.umo,
                busy.waiting,
            )
            chain = MessageChain().message(BUSY_NOTE)
            self.final_llm_resp = LLMResponse(role="assistant", result_chain=chain)
            self._transition_state(AgentState.DONE)
            if self.streaming:
                # A streamed reply's final result is not sent again.
                yield AgentResponse(
                    type="streaming_delta", data=AgentResponseData(chain=chain)
                )
            yield AgentResponse(type="llm_result", data=AgentResponseData(chain=chain))
            return
        except _StoppedWhileQueued:
            # Nothing was kept, so closing the run gives back its group
            # history and its rate-limit use.
            logger.info("Codex turn for %s stopped while queued.", self.umo)
            chain = MessageChain().message(STOPPED_NOTE)
            self.final_llm_resp = LLMResponse(role="assistant", result_chain=chain)
            self._transition_state(AgentState.DONE)
            if self.streaming:
                yield AgentResponse(
                    type="streaming_delta", data=AgentResponseData(chain=chain)
                )
            yield AgentResponse(type="llm_result", data=AgentResponseData(chain=chain))
            return

        text = "\n\n".join(t for t in final_texts if t.strip())
        if show_commentary and commentary:
            text = "\n\n".join([*commentary, text]) if text else "\n\n".join(commentary)
        if not text and (commentary or final_texts):
            text = (commentary or final_texts)[-1]
        # A continuation that produced no answer of its own does not clear
        # the error before it.
        if (
            carried_error
            and not error_msg
            and _answer_count(final_texts) == answers_before
        ):
            error_msg = carried_error
        if error_msg and not text.strip():
            raise RuntimeError(error_msg)
        answered = bool(text.strip())
        # Host text after the model's answer. A streamed reply is already out
        # and its final result is not sent again, so it goes out as a delta.
        tail = ""
        if follow_up_dropped:
            tail = FOLLOW_UP_DROPPED_NOTE
        elif end == "turn_aborted" and not text and self._aborted:
            tail = STOPPED_NOTE
        if tail:
            if answered:
                tail = f"\n\n{tail}"
            text = f"{text}{tail}" if answered else tail.lstrip()
            if self.streaming:
                yield AgentResponse(
                    type="streaming_delta",
                    data=AgentResponseData(chain=MessageChain().message(tail)),
                )

        chain = MessageChain().message(text)
        self.final_llm_resp = LLMResponse(
            role="assistant",
            result_chain=chain,
            reasoning_content="\n".join(r for r in reasoning if r) or None,
            usage=_token_usage(usage),
        )
        self._transition_state(AgentState.DONE)
        if error_msg or (follow_up_dropped and not answered):
            # Timed out with a partial answer, or nothing was answered at all.
            status = "error"
        elif end == "turn_aborted":
            status = "aborted"
        else:
            status = "completed"
        await self._record_stats(status)
        await self._sync_history(text)
        yield AgentResponse(
            type="agent_stats",
            data=AgentResponseData(
                chain=MessageChain(
                    type="agent_stats", chain=[Json(data=self.stats.to_dict())]
                )
            ),
        )
        try:
            await self.agent_hooks.on_agent_done(self.run_context, self.final_llm_resp)
        except Exception as e:  # noqa: BLE001
            logger.error("Error in on_agent_done hook: %s", e, exc_info=True)
        yield AgentResponse(type="llm_result", data=AgentResponseData(chain=chain))

    async def _record_stats(self, status: str) -> None:
        """Closes this run's stats and stores them for the stats page.

        Once per run, and never raising: stats must not break a reply. Rows are
        written with agent type ``codex``, which is what the stats page reads.
        Recorded from the moment the turn is about to be submitted, so a
        submit that fails shows up as an error; a refused (busy) chat, or a
        failure opening the thread, reached no model and is not recorded.
        """
        if self._stats_recorded or not self.stats.start_time:
            return
        self._stats_recorded = True
        self.stats.end_time = time.time()
        after = self._usage_after
        spent = self._usage.spent
        if spent is not None:
            self.stats.token_usage = spent
        elif self.final_llm_resp and self.final_llm_resp.usage:
            # Older binding: the last request's usage is the best there is.
            self.stats.token_usage = self.final_llm_resp.usage
        model = (after or {}).get("model") or self.req.model or self.cfg.get("model")
        provider = (after or {}).get("model_provider") or self.cfg.get("model_provider")
        try:
            conv = self.req.conversation
            await db_helper.insert_provider_stat(
                umo=self.umo,
                conversation_id=conv.cid if conv else None,
                provider_id=str(provider or CODEX_RUNNER_TYPE),
                provider_model=str(model) if model else None,
                status=status,
                stats=self.stats.to_dict(),
                agent_type=CODEX_RUNNER_TYPE,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Persist codex stats failed: %s", e, exc_info=True)

    async def _sync_history(self, text: str) -> None:
        """Mirror the exchange into AstrBot's conversation for the WebUI."""
        conv = self.req.conversation
        if not conv or not self.cfg.get("sync_history", True):
            return
        try:
            ctx = self.run_context.context.context  # type: ignore[attr-defined]
            steered = self._active.steered_texts if self._active else []
            user_text = chr(10).join(t for t in [self.req.prompt or "", *steered] if t)
            await ctx.conversation_manager.add_message_pair(
                conv.cid,
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": text},
            )
            # The conversation list's context size: the last request's input
            # plus output, as the built-in runner records it.
            last = self.final_llm_resp.usage if self.final_llm_resp else None
            if last is not None and last.total:
                await ctx.conversation_manager.update_conversation(
                    self.umo, conv.cid, token_usage=last.total
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to mirror codex exchange into history: %s", e)


def _answer_count(final_texts: list[str]) -> int:
    """Final answers with any text; a blank one answers nothing."""
    return sum(1 for t in final_texts if t.strip())


def _token_usage(usage: JsonObject | None) -> TokenUsage | None:
    if not isinstance(usage, dict):
        return None
    try:
        cached = int(usage.get("cached_input_tokens") or 0)
        return TokenUsage(
            input_other=int(usage.get("input_tokens") or 0) - cached,
            input_cached=cached,
            output=int(usage.get("output_tokens") or 0),
        )
    except Exception:  # noqa: BLE001
        return None
