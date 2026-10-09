import asyncio
from types import SimpleNamespace

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.codex.codex_agent_runner import (
    build_additional_context,
    build_turn_input,
    engine_options,
    generated_image_path,
)
from astrbot.core.agent.runners.codex.tool_bridge import CodexToolBridge
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.provider.entities import ProviderRequest


async def _echo(event, text: str = ""):
    return f"echo:{text}"


def _tool(name: str, params: dict | None = None) -> FunctionTool:
    return FunctionTool(
        name=name,
        description=f"{name} tool",
        parameters=params
        if params is not None
        else {"type": "object", "properties": {"text": {"type": "string"}}},
        handler=_echo,
    )


def test_bridge_sanitizes_and_dedupes_names():
    bridge = CodexToolBridge(
        ToolSet([_tool("a.b"), _tool("a_b"), _tool("mcp__x"), _tool("ok", {})])
    )
    names = [spec["name"] for spec in bridge.specs]
    assert names == ["a_b", "a_b_2", "ext_mcp__x", "ok"]
    ok_spec = next(s for s in bridge.specs if s["name"] == "ok")
    assert ok_spec["inputSchema"] == {"type": "object", "properties": {}}
    [namespace] = bridge.dynamic_tools()
    assert namespace["type"] == "namespace"
    assert namespace["name"] == "astrbot"


def test_bridge_fingerprint_tracks_tool_set():
    a = CodexToolBridge(ToolSet([_tool("x")]))
    b = CodexToolBridge(ToolSet([_tool("x")]))
    c = CodexToolBridge(ToolSet([_tool("x"), _tool("y")]))
    assert a.fingerprint == b.fingerprint != c.fingerprint
    assert CodexToolBridge(None).dynamic_tools() == []


def test_bridge_call_runs_tool_and_hooks():
    bridge = CodexToolBridge(ToolSet([_tool("echo")]))
    event = SimpleNamespace(get_result=lambda: None)
    ctx = ContextWrapper(context=SimpleNamespace(event=event), tool_call_timeout=5)
    seen = []

    class Hooks(BaseAgentRunHooks):
        async def on_tool_start(self, run_context, tool, tool_args):
            seen.append(("start", tool.name, tool_args))

        async def on_tool_end(self, run_context, tool, tool_args, tool_result):
            seen.append(("end", tool.name))

    result = asyncio.run(
        bridge.call(
            {
                "namespace": "astrbot",
                "tool": "echo",
                "arguments": {"text": "hi", "junk": 1},
            },
            ctx,
            Hooks(),
        )
    )
    assert result == {
        "contentItems": [{"type": "inputText", "text": "echo:hi"}],
        "success": True,
    }
    assert seen == [("start", "echo", {"text": "hi"}), ("end", "echo")]

    missing = asyncio.run(
        bridge.call({"namespace": "astrbot", "tool": "nope"}, ctx, Hooks())
    )
    assert missing["success"] is False


def test_turn_input_orders_context_prompt_and_media():
    req = ProviderRequest(prompt="hello")
    req.add_temporary_context("kb", "retrieved facts")
    req.add_persistent_context("message_meta", "Sender: alice")
    req.image_urls = ["https://example.com/a.png", "C:/tmp/b.png"]
    items = build_turn_input(req)
    texts = [i.get("text", "") for i in items if i["type"] == "text"]
    assert "retrieved facts" in texts[0]
    assert "Sender: alice" in texts[1]
    assert texts[2] == "hello"
    assert items[-2] == {"type": "image", "image_url": "https://example.com/a.png"}
    assert items[-1] == {"type": "local_image", "path": "C:/tmp/b.png"}


def test_additional_context_maps_anchors_and_system_prompt():
    req = ProviderRequest(prompt="x", system_prompt=" be nice ")
    req.set_context_anchor("persona", "cat girl")
    assert build_additional_context(req) == {
        "astrbot_system_prompt": {"value": "be nice", "kind": "application"},
        "astrbot_persona": {"value": "cat girl", "kind": "application"},
    }


def test_bridge_defers_tools_in_code_mode():
    bridge = CodexToolBridge(ToolSet([_tool("x")]), defer=True)
    assert bridge.specs[0]["deferLoading"] is True
    assert "deferLoading" not in CodexToolBridge(ToolSet([_tool("x")])).specs[0]


def test_engine_options_default_to_lean_code_mode(tmp_path):
    opts = engine_options({"codex_home": str(tmp_path), "code_mode_host": ""})
    cfg = opts["config"]
    assert cfg["model_tool_mode"] == "code_mode_only"
    assert cfg["features.shell_tool"] is False
    assert cfg["web_search"] == "disabled"
    assert cfg["include_permissions_instructions"] is False
    assert cfg["features.code_mode.structured_dynamic_tool_results"] is True
    assert cfg["features.code_mode.tool_catalog"] is True
    assert cfg["additional_context.reinject_after_compaction"] is True
    assert cfg["additional_context.max_value_tokens"] == 6000
    overridden = engine_options(
        {
            "codex_home": str(tmp_path),
            "tool_mode": "direct",
            "thread_config": {"web_search": "live"},
        }
    )
    assert overridden["config"]["model_tool_mode"] == "direct"
    assert overridden["config"]["web_search"] == "live"


def test_read_skill_file_stays_inside_skill(tmp_path):
    from astrbot.core.agent.runners.codex.skills import (
        build_codex_skills_prompt,
        read_skill_file,
    )
    from astrbot.core.skills.skill_manager import SkillInfo

    skill_dir = tmp_path / "demo"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: d\n---\nBODY", encoding="utf-8"
    )
    (skill_dir / "scripts" / "run.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    skills = [
        SkillInfo(
            name="demo", description="d", path=str(skill_dir / "SKILL.md"), active=True
        )
    ]

    assert "BODY" in read_skill_file(skills, "demo")
    assert read_skill_file(skills, "demo", "scripts/run.py") == "print(1)"
    assert "run.py" in read_skill_file(skills, "demo", "scripts")
    assert read_skill_file(skills, "demo", "../secret.txt").startswith("error")
    assert read_skill_file(skills, "missing").startswith("error: unknown skill")
    assert "astrbot__astrbot_read_skill" in build_codex_skills_prompt(skills)


def test_read_skill_rejects_absolute_and_unc(tmp_path):
    from astrbot.core.agent.runners.codex.skills import read_skill_file
    from astrbot.core.skills.skill_manager import SkillInfo

    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "SKILL.md").write_text("x", encoding="utf-8")
    skills = [
        SkillInfo(
            name="s", description="", path=str(tmp_path / "s" / "SKILL.md"), active=True
        )
    ]
    for bad in (
        r"\\attacker\share\x",
        r"C:\Windows\win.ini",
        "/etc/passwd",
        "../x",
        "C:x",
    ):
        assert read_skill_file(skills, "s", bad).startswith("error"), bad


def test_model_providers_map_to_codex_overrides(tmp_path):
    opts = engine_options(
        {
            "codex_home": str(tmp_path),
            "tool_mode": "direct",
            "model_provider": "my.relay",
            "model_providers": [
                {"id": "my.relay", "base_url": "https://r/v1", "api_key": "sk-1"},
                {"id": "no-url"},
                "junk",
            ],
        }
    )
    cfg = opts["config"]
    assert cfg["model_providers.my_relay.base_url"] == "https://r/v1"
    assert cfg["model_providers.my_relay.experimental_bearer_token"] == "sk-1"
    assert cfg["model_providers.my_relay.wire_api"] == "responses"
    assert not any(k.startswith("model_providers.no-url") for k in cfg)


def test_provider_headers_compaction_and_model_metadata(tmp_path):
    opts = engine_options(
        {
            "codex_home": str(tmp_path),
            "tool_mode": "direct",
            "model_providers": [
                {
                    "id": "go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "headers": {"x-opencode-session": "abc", " ": "dropped"},
                    "compaction": "local",
                    "models": [
                        {
                            "slug": "deepseek-v4.1-flash",
                            "context_window": 128000,
                            "auto_compact_token_limit": "100000",
                            "reasoning_efforts": ["none", "high"],
                            "default_reasoning_effort": "none",
                            "image_input": True,
                            "metadata_json": '{"support_verbosity": false}',
                        },
                        {
                            "slug": "bad-json",
                            "metadata_json": "{nope",
                            "reasoning_efforts": "high",
                        },
                        {"context_window": 1},
                    ],
                },
                {"id": "plain", "base_url": "https://p/v1", "compaction": "auto"},
            ],
        }
    )
    cfg = opts["config"]
    assert cfg["model_providers.go.http_headers"] == {"x-opencode-session": "abc"}
    assert cfg["model_provider_options.go.compaction"] == "local"
    assert cfg["model_provider_options.go.models"] == {
        "deepseek-v4.1-flash": {
            "context_window": 128000,
            "auto_compact_token_limit": 100000,
            "supported_reasoning_levels": [
                {"effort": "none", "description": "none"},
                {"effort": "high", "description": "high"},
            ],
            "default_reasoning_level": "none",
            "input_modalities": ["text", "image"],
            "support_verbosity": False,
        },
        # Unset image input keeps Codex's default.
        "bad-json": {},
    }
    # Nothing else is said for a provider left as it is; none of them gets
    # Codex's turn metadata header (DeepSeek thinks on it).
    assert [k for k in cfg if k.startswith("model_provider_options.plain")] == [
        "model_provider_options.plain.omit_turn_metadata"
    ]
    assert cfg["model_provider_options.plain.omit_turn_metadata"] is True
    assert "model_provider_options.go.wire" not in cfg


def test_a_chat_provider_goes_over_the_chat_wire(tmp_path):
    opts = engine_options(
        {
            "codex_home": str(tmp_path),
            "tool_mode": "direct",
            "model_providers": [
                {
                    "id": "go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "wire_api": "chat",
                    "extra_body": {
                        "thinking": {"type": "disabled", "budget": None},
                        "stop": ["END", None],
                        "reasoning_effort": None,
                    },
                },
                # An extra body without the chat wire says nothing.
                {"id": "r", "base_url": "https://r/v1", "extra_body": {"x": 1}},
                {"id": "ds", "base_url": "https://api.deepseek.com", "files_api": True},
            ],
        }
    )
    cfg = opts["config"]
    # Codex's own wire_api stays Responses; the chat wire is a fork option.
    assert cfg["model_providers.go.wire_api"] == "responses"
    assert cfg["model_provider_options.go.wire"] == "chat"
    assert cfg["model_provider_options.go.extra_body"] == {
        "thinking": {"type": "disabled"},
        "stop": ["END"],
    }
    assert cfg["model_provider_options.go.extra_body_remove"] == ["reasoning_effort"]
    assert [k for k in cfg if k.startswith("model_provider_options.r.")] == [
        "model_provider_options.r.omit_turn_metadata"
    ]
    assert "model_providers.plain.http_headers" not in cfg
    assert cfg["model_provider_options.ds.files_api"] == {"expires_seconds": 86400}
    assert "model_provider_options.go.files_api" not in cfg


def test_native_exec_approvals_follow_permission_rules(tmp_path):
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        native_exec_decision,
    )
    from astrbot.core.permission_rules import EVENT_EXTRA_KEY, PermissionPolicy

    opts = engine_options(
        {"codex_home": str(tmp_path), "tool_mode": "direct", "native_exec_tools": True}
    )
    assert opts["approve_every_command"] is True
    assert "approval_policy" not in opts["config"]
    plain = engine_options({"codex_home": str(tmp_path), "tool_mode": "direct"})
    assert "approve_every_command" not in plain
    assert plain["config"]["approval_policy"] == "never"

    def event(policy):
        return SimpleNamespace(
            get_extra=lambda key: policy if key == EVENT_EXTRA_KEY else None
        )

    assert native_exec_decision(event(None)) == (True, "")
    assert native_exec_decision(event(PermissionPolicy(native_exec=True)))[0] is True
    denied = native_exec_decision(event(PermissionPolicy(native_exec=False)))
    assert denied[0] is False and "not permitted" in denied[1]
    off = native_exec_decision(event(PermissionPolicy(native_exec=True)), False)
    assert off[0] is False and "turned off" in off[1]
    assert native_exec_decision(event(None), True) == (True, "")


def test_memory_thread_config_leaves_shared_writes_to_the_sender(tmp_path):
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        memory_thread_config,
    )
    from astrbot.core.permission_rules import EVENT_EXTRA_KEY, PermissionPolicy
    from astrbot.core.platform.message_type import MessageType

    def event(policy, group=""):
        kind = MessageType.GROUP_MESSAGE if group else MessageType.FRIEND_MESSAGE
        return SimpleNamespace(
            get_extra=lambda key: policy if key == EVENT_EXTRA_KEY else None,
            get_group_id=lambda: group,
            get_message_type=lambda: kind,
        )

    allowed = PermissionPolicy(global_memory=True)
    cfg = {"memory_auto_consolidate": False}
    conf = memory_thread_config(cfg, "qq:FriendMessage:1", event(allowed))
    assert conf["memories.scope_key"] == "qq:FriendMessage:1"
    # Shared writes and deletions follow each turn's scopes, in any chat.
    assert conf["memories.turn_scopes"] is True
    # Automatic consolidation reaches the shared store from their private chat.
    assert conf["memories.may_write_global"] is True
    assert conf["memories.auto_consolidate"] is False
    assert conf["memories.extra_session_sources"] == ["astrbot"]
    in_group = memory_thread_config(cfg, "u", event(allowed, group="9"))
    assert in_group["memories.turn_scopes"] is True
    # A group's transcript mixes many people: it is never consolidated globally.
    assert in_group["memories.may_write_global"] is False
    # A group task created before tasks recorded their group: still a group.
    legacy = event(allowed, group="9")
    legacy.get_group_id = lambda: ""
    assert memory_thread_config(cfg, "u", legacy)["memories.may_write_global"] is False
    without_rule = memory_thread_config(cfg, "u", event(None))
    assert without_rule["memories.may_write_global"] is False

    # The scopes a turn carries: a rule granting global_memory, in a group too.
    assert allowed.scopes == ["memory.write_global", "memory.delete"]
    assert PermissionPolicy().scopes == []
    assert PermissionPolicy(global_memory=False).scopes == []
    exe = tmp_path / "codex.exe"
    exe.write_bytes(b"")
    base = {
        "codex_home": str(tmp_path),
        "tool_mode": "direct",
        "codex_self_exe": str(exe),
    }
    # 记忆整理通过 memories 扩展的文件工具完成，进程内运行，不需要二进制。
    assert "codex_self_exe" not in engine_options({**base, "memory_enabled": True})
    # 原生执行仍然需要它：没有它 Codex 就没有本机执行环境。
    assert engine_options({**base, "native_exec_tools": True})["codex_self_exe"] == str(
        exe
    )


def test_auto_approve_off_denies_explicit_approval_policies():
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        approvals_disabled,
    )

    assert approvals_disabled({}) is False
    assert approvals_disabled({"approval_policy": "never"}) is False
    assert approvals_disabled({"approval_policy": "on-request"}) is True
    assert (
        approvals_disabled({"approval_policy": "on-request", "auto_approve": True})
        is False
    )


def test_approval_is_answered_for_the_turn_it_arrived_in():
    from astrbot.core.agent.runners.codex.native import ThreadPump, _TurnRoute

    decisions: list[str] = []

    class Rt:
        async def review_decision(self, thread_id, request):
            decisions.append(request)

    async def approve(kind, msg):
        return True, ""

    async def run():
        import json

        pump = ThreadPump(SimpleNamespace(rt=Rt(), pumps={}), "t1")
        route = _TurnRoute(asyncio.Queue(), None, approve)
        msg = {"type": "exec_approval_request", "call_id": "c1", "turn_id": "u1"}
        await pump._answer_approval("exec", msg, route)
        await pump._answer_approval("exec", msg, None)
        first, second = (json.loads(d) for d in decisions)
        assert first["approved"] is True and first["id"] == "c1"
        assert second["approved"] is False and second["reason"]

    asyncio.run(run())


def test_tool_call_runs_only_for_the_turn_it_came_from():
    from astrbot.core.agent.runners.codex.native import (
        TURN_ID_FIELD,
        ActiveTurn,
        ThreadPump,
        _TurnRoute,
    )

    answers: dict[str, dict] = {}
    handled: list[str] = []

    class Rt:
        async def dynamic_tool_response(self, thread_id, call_id, result):
            import json

            answers[call_id] = json.loads(result)

    async def handler(msg):
        handled.append(msg["callId"])
        return {"contentItems": [], "success": True}

    async def run():
        pump = ThreadPump(SimpleNamespace(rt=Rt(), pumps={}), "t1")
        turn = ActiveTurn(SimpleNamespace(), "t1", "", "B")
        route = _TurnRoute(asyncio.Queue(), handler, None, turn)

        def call(call_id, turn_id):
            return {"callId": call_id, "tool": "x", TURN_ID_FIELD: turn_id}

        # A call while B's turn is being submitted waits for its id.
        mine = asyncio.create_task(pump._answer_tool(call("c1", "b1"), route))
        stale = asyncio.create_task(pump._answer_tool(call("c2", "a1"), route))
        await asyncio.sleep(0)
        assert not answers
        turn.turn_id = "b1"
        turn.turn_ids.add("b1")
        turn.ready.set()
        await asyncio.gather(mine, stale)
        assert handled == ["c1"]
        assert answers["c1"]["success"] is True
        # The interrupted turn A's call never runs as sender B.
        assert answers["c2"]["success"] is False

        # A continuation's call made before its submit returns.
        turn.ready.clear()
        cont = asyncio.create_task(pump._answer_tool(call("c3", "b2"), route))
        await asyncio.sleep(0)
        turn.turn_id = "b2"
        turn.turn_ids.add("b2")
        turn.ready.set()
        await cont
        assert handled == ["c1", "c3"]

        # Unbound routes (voice, provider adapter) keep answering every call.
        await pump._answer_tool(call("c4", "zz"), _TurnRoute(asyncio.Queue(), handler))
        assert handled[-1] == "c4"

    asyncio.run(run())


def test_persona_catalog_only_when_personas_alternate():
    from astrbot.core.agent.runners.codex.codex_agent_runner import persona_context

    def req(persona):
        r = ProviderRequest(prompt="hi")
        if persona:
            r.set_context_anchor("persona", persona)
        return r

    catalog, ctx, active = persona_context(req("cat"), {})
    assert active is None and ctx["astrbot_persona"]["value"] == "cat"
    same, ctx, active = persona_context(req("cat"), catalog)
    assert same == catalog and active is None

    both, ctx_dog, active = persona_context(req("dog"), catalog)
    assert len(both) == 2 and "astrbot_persona" not in ctx_dog
    assert "cat" in ctx_dog["astrbot_personas"]["value"]
    assert active["text"].startswith("<active_persona id=")
    # Alternating back does not change the standing context (cache-safe).
    again, ctx_cat, active_cat = persona_context(req("cat"), both)
    assert again == both and ctx_cat == ctx_dog and active_cat != active
    _, _, none_active = persona_context(req(""), both)
    assert "none" in none_active["text"]


def test_code_mode_hides_denied_tools_but_direct_mode_lists_them():
    from astrbot.core.permission_rules import PermissionPolicy

    tools = ToolSet([_tool("weather"), _tool("calc")])
    denied = PermissionPolicy(rule_name="r", tools_deny=("weather",))

    hidden = CodexToolBridge(tools, defer=True, policy=denied)
    assert [spec["name"] for spec in hidden.specs] == ["calc"]
    assert hidden.lookup("astrbot", "weather") is None

    # Without deferral the tool set must stay identical for every sender, or
    # the prompt prefix (and its cache) would differ per user.
    listed = CodexToolBridge(tools, defer=False, policy=denied)
    assert [spec["name"] for spec in listed.specs] == ["calc", "weather"]

    # No policy: nothing is filtered in either mode.
    assert len(CodexToolBridge(tools, defer=True).specs) == 2


def test_code_mode_groups_plugin_and_mcp_tools_by_source(monkeypatch):
    from astrbot.core.star.star import StarMetadata, star_map

    monkeypatch.setitem(
        star_map,
        "data.plugins.weather.main",
        StarMetadata(name="astrbot_plugin_weather", short_desc="Weather lookups"),
    )
    plugin_tool = _tool("get_weather")
    plugin_tool.handler_module_path = "data.plugins.weather.main"
    mcp_tool = _tool("now")
    mcp_tool.mcp_server_name = "time.srv"
    tools = ToolSet([_tool("send_message_to_user"), plugin_tool, mcp_tool])

    grouped = CodexToolBridge(tools, defer=True)
    assert [
        (ns["name"], ns["description"], [spec["name"] for spec in ns["tools"]])
        for ns in grouped.dynamic_tools()
    ] == [
        (
            "astrbot",
            "AstrBot tools for the current chat session.",
            ["send_message_to_user"],
        ),
        ("astrbot__mcp_time_srv", "MCP server time.srv.", ["now"]),
        (
            "astrbot__weather",
            "Plugin astrbot_plugin_weather: Weather lookups",
            ["get_weather"],
        ),
    ]
    assert grouped.lookup("astrbot__weather", "get_weather") is plugin_tool
    assert grouped.lookup("astrbot", "get_weather") is None

    # Without deferral the names stay short and in one namespace.
    [flat] = CodexToolBridge(tools, defer=False).dynamic_tools()
    assert flat["name"] == "astrbot"
    assert len(flat["tools"]) == 3


def test_hidden_tools_change_the_fingerprint_per_sender():
    from astrbot.core.permission_rules import PermissionPolicy

    tools = ToolSet([_tool("weather"), _tool("calc")])
    allowed = CodexToolBridge(tools, defer=True, policy=PermissionPolicy())
    denied = CodexToolBridge(
        tools, defer=True, policy=PermissionPolicy(tools_deny=("weather",))
    )
    # A different tool set means the runner sends a tools update for that turn.
    assert allowed.fingerprint != denied.fingerprint


# ============================================================
# 遥测与 codex 可执行文件告警
# ============================================================


def test_engine_options_disable_codex_telemetry():
    """Codex 会把 skill 名、MCP 调用等上报到 chatgpt.com，默认必须关掉。"""
    config = engine_options({})["config"]

    assert config["analytics.enabled"] is False
    assert config["otel.metrics_exporter"] == "none"


def test_thread_config_can_re_enable_analytics():
    """运营者仍可通过 thread_config 覆盖回来。"""
    config = engine_options({"thread_config": {"analytics.enabled": True}})["config"]

    assert config["analytics.enabled"] is True


def _captured_warnings(monkeypatch, cfg):
    """AstrBot 用 loguru，日志不走 stdlib handler，caplog 抓不到，所以直接换掉 logger。"""
    monkeypatch.setattr(
        "astrbot.core.agent.runners.codex.codex_agent_runner.find_codex_exe",
        lambda _: "",
    )
    messages = []
    monkeypatch.setattr(
        "astrbot.core.agent.runners.codex.codex_agent_runner.logger",
        SimpleNamespace(
            warning=lambda msg, *args: messages.append(msg % args if args else msg)
        ),
    )
    engine_options(cfg)
    return "\n".join(messages)


def test_no_codex_exe_warning_only_fires_for_native_execution(monkeypatch):
    """记忆整理已不依赖二进制，只有原生执行还需要。"""
    message = _captured_warnings(monkeypatch, {"native_exec_tools": True})

    assert "native execution" in message
    assert "memory consolidation are unaffected" in message


def test_memory_alone_does_not_warn(monkeypatch):
    """开着记忆但没有二进制是完全正常的配置，不该报警。"""
    assert "No codex executable found" not in _captured_warnings(
        monkeypatch, {"memory_enabled": True}
    )


def test_no_codex_exe_warning_is_silent_when_nothing_needs_it(monkeypatch):
    """聊天本来就不需要这个二进制。"""
    assert "No codex executable found" not in _captured_warnings(monkeypatch, {})


# ============================================================
# 生成的图片自动发到 IM
# ============================================================


def _image_item(path, **overrides):
    item = {"type": "ImageGeneration", "status": "completed", "saved_path": str(path)}
    item.update(overrides)
    return item


def test_generated_image_path_accepts_hosted_item(tmp_path):
    png = tmp_path / "shot.png"
    png.write_bytes(b"png")

    assert generated_image_path(_image_item(png)) == str(png)


def test_generated_image_path_accepts_extension_item(tmp_path):
    """独立图片生成走扩展 item，tag 是 kind 而不是 type。"""
    png = tmp_path / "shot.png"
    png.write_bytes(b"png")
    item = {
        "type": "Extension",
        "kind": "image_gen.generation",
        "status": "completed",
        "saved_path": str(png),
    }

    assert generated_image_path(item) == str(png)


def test_generated_image_path_ignores_unfinished_and_missing(tmp_path):
    png = tmp_path / "shot.png"
    png.write_bytes(b"png")

    # 还在生成中
    assert generated_image_path(_image_item(png, status="in_progress")) is None
    # 失败时没有 saved_path
    assert (
        generated_image_path({"type": "ImageGeneration", "status": "completed"}) is None
    )
    # 路径已不存在：宁可不发，也不要抛到发送链路上
    assert generated_image_path(_image_item(tmp_path / "gone.png")) is None
    # 完全无关的 item
    assert generated_image_path({"type": "AgentMessage", "status": "completed"}) is None


def test_chatgpt_apps_are_always_off():
    """ChatGPT 连接器会把账号里连接的数据交给所有会话，始终关闭。"""
    assert engine_options({})["config"]["features.apps"] is False
    forced = engine_options({"thread_config": {"features.apps": True}})["config"]
    assert forced["features.apps"] is False


def test_proxy_is_passed_to_codex_only():
    """代理只交给 Codex（outbound_proxy），不写进进程环境变量。"""
    import os

    before = {k: os.environ.get(k) for k in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY")}
    config = engine_options({"proxy": " socks5://127.0.0.1:7890 "})["config"]
    assert config["outbound_proxy"] == "socks5://127.0.0.1:7890"
    assert "outbound_proxy" not in engine_options({"proxy": ""})["config"]
    assert {k: os.environ.get(k) for k in before} == before


def test_colliding_sources_get_their_own_namespaces(monkeypatch):
    from astrbot.core.star.star import StarMetadata, star_map

    monkeypatch.setitem(star_map, "p.x", StarMetadata(name="x"))
    monkeypatch.setitem(star_map, "p.ax", StarMetadata(name="astrbot_plugin_x"))
    monkeypatch.setitem(star_map, "p.mcp", StarMetadata(name="mcp_time"))
    tools = []
    for name, module in (("one", "p.x"), ("two", "p.ax"), ("three", "p.mcp")):
        tool = _tool(name)
        tool.handler_module_path = module
        tools.append(tool)
    server_tool = _tool("now")
    server_tool.mcp_server_name = "time"
    tools.append(server_tool)

    names = [
        ns["name"] for ns in CodexToolBridge(ToolSet(tools), defer=True).dynamic_tools()
    ]
    assert "astrbot__mcp_time" in names  # the MCP server
    assert "astrbot__plugin_mcp_time" in names  # the plugin named mcp_time
    x_namespaces = [n for n in names if n.startswith("astrbot__x_")]
    assert len(x_namespaces) == 2 and "astrbot__x" not in names


def test_code_mode_names_do_not_depend_on_the_sender():
    from astrbot.core.permission_rules import PermissionPolicy

    tools = ToolSet([_tool("a-b"), _tool("a_b")])
    everyone = CodexToolBridge(tools, defer=True)
    assert sorted(spec["name"] for spec in everyone.specs) == ["a_b", "a_b_2"]
    denied = CodexToolBridge(
        tools, defer=True, policy=PermissionPolicy(tools_deny=("a-b",))
    )
    # a_b keeps its own name; the folded a-b gets the suffix, whoever speaks.
    assert everyone.lookup("astrbot", "a_b").name == "a_b"
    assert [spec["name"] for spec in denied.specs] == ["a_b"]
    only_folded = CodexToolBridge(
        tools, defer=True, policy=PermissionPolicy(tools_deny=("a_b",))
    )
    assert [spec["name"] for spec in only_folded.specs] == ["a_b_2"]


def test_code_mode_names_cannot_reach_into_another_namespace():
    bridge = CodexToolBridge(
        ToolSet([_tool("__foo"), _tool("x__y"), _tool("-z")]), defer=True
    )
    assert sorted(spec["name"] for spec in bridge.specs) == ["foo", "x_y", "z"]


def test_inputs_without_additional_context_keep_the_threads_last_one():
    """Codex reads a missing additional_context as "none": a steer or a
    continuation must not clear the persona it keeps for the thread."""
    import json

    from astrbot.core.agent.runners.codex.native import CodexEngine

    sent = []

    class FakeRuntime:
        async def submit_turn(self, thread_id, request_json):
            sent.append((thread_id, json.loads(request_json)))
            return json.dumps({"status": "started"})

    engine = CodexEngine(FakeRuntime())
    persona = {"astrbot_persona": {"value": "be a cat", "kind": "application"}}

    async def run():
        await engine.submit_turn("t1", {"input": [], "additional_context": persona})
        await engine.submit_turn("t1", {"input": [], "mode": "steer"})
        await engine.submit_turn("t2", {"input": [], "mode": "steer"})
        await engine.submit_turn("t1", {"input": [], "additional_context": {}})
        await engine.submit_turn("t1", {"input": [], "mode": "start_if_idle"})

    asyncio.run(run())
    assert [request.get("additional_context") for _, request in sent] == [
        persona,
        persona,
        None,  # another thread has none of its own yet
        {},
        {},  # an explicit empty context is kept too
    ]


def test_releasing_a_thread_unloads_it_and_forgets_its_context():
    """/new and /reset leave the old thread: it is shut down in Codex and
    its cached context is dropped."""
    import json

    from astrbot.core.agent.runners.codex.native import CodexEngine

    sent, shut = [], []

    class FakeRuntime:
        async def submit_turn(self, thread_id, request_json):
            sent.append(json.loads(request_json))
            return json.dumps({"status": "started"})

        async def shutdown_thread(self, thread_id):
            shut.append(thread_id)

    engine = CodexEngine(FakeRuntime())
    persona = {"astrbot_persona": {"value": "be a cat", "kind": "application"}}

    async def run():
        await engine.submit_turn("old", {"input": [], "additional_context": persona})
        CodexEngine._instances["test"] = engine
        try:
            await CodexEngine.release_thread("chat", "old")
        finally:
            CodexEngine._instances.pop("test", None)
        await engine.submit_turn("old", {"input": [], "mode": "steer"})

    asyncio.run(run())
    assert shut == ["old"]
    assert "additional_context" not in sent[-1]
