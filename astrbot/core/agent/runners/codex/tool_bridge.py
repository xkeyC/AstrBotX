"""Expose an AstrBot ToolSet to Codex as dynamic tools and run Codex tool calls."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from astrbot.core import content_moderation, logger
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.permission_rules import EVENT_EXTRA_KEY as POLICY_EXTRA_KEY
from astrbot.core.permission_rules import PermissionPolicy, tool_mcp_server

from .constants import CODEX_TOOL_NAMESPACE

_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_TOOL_NAME = 128
# Source keys never contain "__", which would read as one more level.
_SOURCE_KEY_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
_MAX_SOURCE_KEY = 40
_MAX_SOURCE_DESCRIPTION = 200
_BASE_DESCRIPTION = "AstrBot plugin tools for the current chat session."
_DEFERRED_BASE_DESCRIPTION = "AstrBot tools for the current chat session."
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

JsonObject = dict[str, Any]


def _codex_name(name: str, taken: set[str], *, js_safe: bool = False) -> str:
    """A Codex tool name for ``name``, unique within ``taken``.

    ``js_safe`` also turns ``-`` into ``_``, drops leading ``_`` and folds
    ``__``: code mode calls tools as JS identifiers, where ``a-b`` and ``a_b``
    are one name, and joins namespace and name with ``__`` (a leading ``_``
    or an inner ``__`` could make it another namespace's tool).
    """
    base = _INVALID_NAME_CHARS.sub("_", name)
    if js_safe:
        base = re.sub(r"_{2,}", "_", base.replace("-", "_")).lstrip("_")
    if not base.strip("_-"):
        base = "tool"
    # "mcp" and "mcp__*" are reserved by Codex.
    if base == "mcp" or base.startswith("mcp__"):
        base = f"ext_{base}"
    base = base[:_MAX_TOOL_NAME]
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        suffix = f"_{n}"
        candidate = base[: _MAX_TOOL_NAME - len(suffix)] + suffix
    taken.add(candidate)
    return candidate


def _source_key(text: str) -> str:
    return _SOURCE_KEY_SPLIT.sub("_", text).strip("_")[:_MAX_SOURCE_KEY].strip("_")


def tool_source(tool: FunctionTool) -> tuple[str, str, str] | None:
    """``(key, description, identity)`` of the plugin or MCP server a tool
    comes from; the key may be shared by several sources.

    None for AstrBot's own tools, which belong to no plugin.
    """
    if server := tool_mcp_server(tool):
        key = f"mcp_{_source_key(server) or 'server'}"
        return key, f"MCP server {server}.", f"mcp:{server}"
    module = getattr(tool, "handler_module_path", None)
    if not module:
        return None
    from astrbot.core.star.star import star_map

    plugin = star_map.get(module)
    if plugin is None or not plugin.name:
        return None
    short = re.sub(r"^astrbot_plugin_", "", plugin.name, flags=re.IGNORECASE)
    key = _source_key(short) or _source_key(plugin.root_dir_name or "") or "plugin"
    if key.lower().startswith("mcp"):
        key = f"plugin_{key}"  # MCP servers own the mcp_ keys
    about = (plugin.short_desc or plugin.desc or "").strip()
    label = plugin.display_name or plugin.name
    description = f"Plugin {label}: {about}" if about else f"Plugin {label}."
    return key, description[:_MAX_SOURCE_DESCRIPTION], f"plugin:{plugin.name}"


def _source_namespaces(tools: list[FunctionTool]) -> dict[int, tuple[str, str]]:
    """``id(tool) -> (namespace, description)`` for tools from a plugin or an
    MCP server; sources whose keys collide get a short hash each."""
    sources = {id(tool): source for tool in tools if (source := tool_source(tool))}
    identities: dict[str, set[str]] = {}
    for key, _, identity in sources.values():
        identities.setdefault(key, set()).add(identity)
    namespaces = {}
    for tool_id, (key, description, identity) in sources.items():
        if len(identities[key]) > 1:
            digest = hashlib.sha1(identity.encode()).hexdigest()[:6]
            key = f"{key}_{digest}"
        namespaces[tool_id] = (f"{CODEX_TOOL_NAMESPACE}__{key}", description)
    return namespaces


def _input_schema(tool: FunctionTool) -> JsonObject:
    params = tool.parameters
    if not isinstance(params, dict) or not params:
        return dict(_EMPTY_SCHEMA)
    schema = dict(params)
    schema.setdefault("type", "object")
    if schema.get("type") == "object":
        schema.setdefault("properties", {})
    return schema


class CodexToolBridge:
    """Name mapping between an AstrBot ToolSet and one Codex thread."""

    def __init__(
        self,
        tool_set: ToolSet | None,
        *,
        defer: bool = False,
        policy: PermissionPolicy | None = None,
    ) -> None:
        """Map an AstrBot ToolSet onto one Codex thread.

        Args:
            tool_set: Tools offered for this request.
            defer: Whether the tools load on demand, which is what code mode
                does. Deferred tools never enter the prompt prefix.
            policy: Sender's permission rule. Denied tools are left out when
                the tools are deferred, so the model cannot even see them;
                without deferral, dropping them per sender would change the
                prompt prefix and cost the cache, so they stay listed and are
                refused when called.

        Deferred tools from a plugin or an MCP server get a namespace of their
        own, ``astrbot__<source>``, so Codex's tool catalog lists them under
        that source; AstrBot's own tools stay in ``astrbot``. Without
        deferral everything stays in ``astrbot``, keeping names short.
        """
        self.defer = defer
        # (namespace, Codex name) -> tool
        self.tools: dict[tuple[str, str], FunctionTool] = {}
        self.specs: list[JsonObject] = []
        self._namespaces: dict[str, JsonObject] = {}
        taken: dict[str, set[str]] = {}
        active = [
            tool
            for tool in sorted(
                (tool_set.tools if tool_set else []), key=lambda t: t.name
            )
            if getattr(tool, "active", True)
        ]
        if defer:
            # Tools whose names need no folding claim them first, so a
            # folded look-alike never takes a name prompts refer to.
            active.sort(
                key=lambda t: (
                    _codex_name(t.name, set(), js_safe=True) != t.name,
                    t.name,
                )
            )
        sources = _source_namespaces(active) if defer else {}
        for tool in active:
            namespace, description = sources.get(
                id(tool),
                (
                    CODEX_TOOL_NAMESPACE,
                    _DEFERRED_BASE_DESCRIPTION if defer else _BASE_DESCRIPTION,
                ),
            )
            # Names are given before the sender's rule filters anything, so a
            # name means the same tool whoever speaks.
            name = _codex_name(
                tool.name, taken.setdefault(namespace, set()), js_safe=defer
            )
            if (
                defer
                and policy is not None
                and not policy.allows_tool(tool.name, tool_mcp_server(tool))
            ):
                continue
            self._namespaces.setdefault(
                namespace,
                {
                    "type": "namespace",
                    "name": namespace,
                    "description": description,
                    "tools": [],
                },
            )
            self.tools[(namespace, name)] = tool
            spec = {
                "type": "function",
                "name": name,
                "description": (tool.description or tool.name)[:4000],
                "inputSchema": _input_schema(tool),
            }
            if defer:
                # Deferred tools never enter the prompt prefix; code mode
                # discovers them through ALL_TOOLS.
                spec["deferLoading"] = True
            self.specs.append(spec)
            self._namespaces[namespace]["tools"].append(spec)
        self.fingerprint = hashlib.sha256(
            json.dumps(
                self.dynamic_tools(), sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest()[:16]

    def dynamic_tools(self) -> list[JsonObject]:
        return [
            namespace
            for _, namespace in sorted(self._namespaces.items())
            if namespace["tools"]
        ]

    def lookup(self, namespace: str | None, name: str) -> FunctionTool | None:
        """The tool Codex calls ``namespace.name``."""
        return self.tools.get((namespace or CODEX_TOOL_NAMESPACE, name))

    async def call(
        self,
        params: JsonObject,
        run_context: ContextWrapper,
        agent_hooks: BaseAgentRunHooks,
    ) -> JsonObject:
        """Answer an ``item/tool/call`` request."""
        from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor

        name = params.get("tool", "")
        raw_args = params.get("arguments")
        args: JsonObject = raw_args if isinstance(raw_args, dict) else {}
        tool = self.lookup(params.get("namespace"), name)
        if tool is None:
            return _text_result(
                f"error: tool {params.get('namespace')}.{name} is not available.",
                success=False,
            )

        if tool.handler and tool.parameters and tool.parameters.get("properties"):
            expected = set(tool.parameters["properties"].keys())
            ignored = set(args) - expected
            if ignored:
                logger.warning(
                    "Codex tool %s: ignoring unexpected args %s", name, ignored
                )
            args = {k: v for k, v in args.items() if k in expected}

        event = getattr(getattr(run_context, "context", None), "event", None)
        get_extra = getattr(event, "get_extra", None)
        policy = get_extra(POLICY_EXTRA_KEY) if callable(get_extra) else None
        if isinstance(policy, PermissionPolicy) and not policy.allows_tool(
            tool.name, tool_mcp_server(tool)
        ):
            logger.info("Codex tool %s denied by rule %r", tool.name, policy.rule_name)
            return _text_result(
                f"error: permission denied — this user may not use {tool.name}.",
                success=False,
            )

        logger.info("Codex -> AstrBot tool %s(%s)", tool.name, args)
        try:
            await agent_hooks.on_tool_start(run_context, tool, args)
        except Exception as e:  # noqa: BLE001
            logger.error("Error in on_tool_start hook: %s", e, exc_info=True)

        items: list[JsonObject] = []
        final: CallToolResult | None = None
        success = True
        try:
            async for resp in FunctionToolExecutor.execute(
                tool=tool, run_context=run_context, **args
            ):
                if isinstance(resp, CallToolResult):
                    final = resp
                    items.extend(_content_items(resp))
                    if resp.isError:
                        success = False
                elif resp is None:
                    items.append(
                        {
                            "type": "inputText",
                            "text": "The tool has no return value, or has sent "
                            "the result directly to the user.",
                        }
                    )
                else:
                    items.append({"type": "inputText", "text": str(resp)})
        except Exception as e:  # noqa: BLE001
            logger.warning("Codex tool %s failed: %s", name, e, exc_info=True)
            items = [{"type": "inputText", "text": f"error: {e!s}"}]
            success = False
        finally:
            try:
                await agent_hooks.on_tool_end(run_context, tool, args, final)
            except Exception as e:  # noqa: BLE001
                logger.error("Error in on_tool_end hook: %s", e, exc_info=True)

        if not items:
            items = [{"type": "inputText", "text": "The tool returned no content."}]
        # What a tool read (a file, an image, a page) goes to the cloud model
        # like a message: flagged parts are removed and the model is told.
        get_platform_id = getattr(event, "get_platform_id", None)
        platform_id = get_platform_id() if callable(get_platform_id) else ""
        items = await content_moderation.filter_tool_result(
            items,
            content_moderation.platform_mode(str(platform_id or "")),
            label=f"{tool.name}, {getattr(event, 'unified_msg_origin', '')}",
        )
        return {"contentItems": items, "success": success}


def _text_result(text: str, *, success: bool) -> JsonObject:
    return {"contentItems": [{"type": "inputText", "text": text}], "success": success}


def _content_items(result: CallToolResult) -> list[JsonObject]:
    items: list[JsonObject] = []
    for content in result.content or []:
        if isinstance(content, TextContent):
            items.append({"type": "inputText", "text": content.text})
        elif isinstance(content, ImageContent):
            mime = content.mimeType or "image/png"
            items.append(
                {"type": "inputImage", "imageUrl": f"data:{mime};base64,{content.data}"}
            )
        elif isinstance(content, EmbeddedResource):
            res = content.resource
            if isinstance(res, TextResourceContents):
                items.append({"type": "inputText", "text": res.text})
            elif (
                isinstance(res, BlobResourceContents)
                and res.mimeType
                and res.mimeType.startswith("image/")
            ):
                items.append(
                    {
                        "type": "inputImage",
                        "imageUrl": f"data:{res.mimeType};base64,{res.blob}",
                    }
                )
            else:
                items.append(
                    {"type": "inputText", "text": "[unsupported embedded resource]"}
                )
        else:
            items.append(
                {"type": "inputText", "text": f"[unsupported content: {content.type}]"}
            )
    return items
