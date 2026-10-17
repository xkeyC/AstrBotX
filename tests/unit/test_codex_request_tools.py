"""The tool set prepare_codex_request hands to Codex under a persona's tool list."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_main_agent import PERSONA_ALLOWED_TOOLS_EXTRA_KEY
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import codex_request
from astrbot.core.provider.entities import ProviderRequest


@dataclass
class _Tool(FunctionTool):
    parameters: dict | None = None


class _Event:
    unified_msg_origin = "aiocqhttp:GroupMessage:1"
    plugins_name = None
    platform_meta = SimpleNamespace(support_proactive_message=False)
    role = "member"

    def __init__(self) -> None:
        self._extras: dict = {}

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_selected_persona(self):
        return None

    def get_message_type(self):
        return None

    def get_sender_id(self):
        return "42"

    def get_group_id(self):
        return "1"


async def _prepared_tools(monkeypatch, allowed, *, skills=None) -> set[str]:
    event = _Event()

    async def decorate(event, req, *_args, **_kwargs):
        event.set_extra(PERSONA_ALLOWED_TOOLS_EXTRA_KEY, allowed)
        if skills:
            event.set_extra("_codex_skills", skills)

    def sandbox_tools(_config, req, _session_id):
        req.func_tool = req.func_tool or ToolSet()
        for name in ("astrbot_execute_shell", "astrbot_file_read_tool"):
            req.func_tool.add_tool(_Tool(name=name, description=name))

    for name in ("_collect_media", "_get_session_conv", "_process_quote_message"):
        monkeypatch.setattr(codex_request, name, AsyncMock(return_value=None))
    monkeypatch.setattr(codex_request, "_apply_kb", AsyncMock())
    monkeypatch.setattr(codex_request, "_decorate_llm_request", decorate)
    monkeypatch.setattr(codex_request, "_apply_sandbox_tools", sandbox_tools)
    monkeypatch.setattr(codex_request, "_proactive_cron_job_tools", MagicMock())
    plugin_context = MagicMock()
    plugin_context.get_config.return_value = {}

    req = ProviderRequest(prompt="hi")
    await codex_request.prepare_codex_request(
        event,
        req,
        plugin_context,
        {"provider_settings": {"computer_use_runtime": "sandbox"}},
        {},
    )
    return set(req.func_tool.names())


@pytest.mark.asyncio
async def test_a_persona_listing_the_shell_keeps_codex_exec(monkeypatch):
    names = await _prepared_tools(
        monkeypatch, {"astrbot_execute_shell", "astrbot_file_read_tool"}
    )

    assert names == {
        "exec_command",
        "write_stdin",
        "astrbot_file_read_tool",
    }


@pytest.mark.asyncio
async def test_a_persona_without_the_shell_gets_no_codex_exec(monkeypatch):
    names = await _prepared_tools(monkeypatch, {"astrbot_file_read_tool"})

    assert names == {"astrbot_file_read_tool"}


@pytest.mark.asyncio
async def test_a_persona_listing_the_codex_names_keeps_them(monkeypatch):
    names = await _prepared_tools(monkeypatch, {"exec_command", "apply_patch"})

    assert names == {"exec_command", "apply_patch"}


@pytest.mark.asyncio
async def test_a_persona_without_a_tool_list_gets_every_tool(monkeypatch):
    names = await _prepared_tools(monkeypatch, None)

    assert {"exec_command", "write_stdin", "apply_patch"} <= names
    assert "astrbot_execute_shell" not in names


@pytest.mark.asyncio
async def test_skills_stay_readable_under_a_persona_tool_list(monkeypatch):
    names = await _prepared_tools(
        monkeypatch, {"astrbot_file_read_tool"}, skills=[object()]
    )

    assert "astrbot_read_skill" in names


def test_codex_patch_and_exec_tools_are_listed_as_builtin():
    from astrbot.core.provider.func_tool_manager import FunctionToolManager

    mgr = FunctionToolManager()
    names = {tool.name for tool in mgr.iter_builtin_tools()}

    assert {"exec_command", "write_stdin", "apply_patch"} <= names


@pytest.mark.asyncio
async def test_the_local_runtime_note_names_the_codex_shell_tools(monkeypatch):
    def local_tools(req, _plugin_context):
        req.func_tool = req.func_tool or ToolSet()
        for name in ("astrbot_execute_shell", "astrbot_shell_session"):
            req.func_tool.add_tool(_Tool(name=name, description=name))
        req.set_context_anchor(
            "local_runtime", codex_request._build_local_mode_prompt()
        )

    for name in ("_collect_media", "_get_session_conv", "_process_quote_message"):
        monkeypatch.setattr(codex_request, name, AsyncMock(return_value=None))
    monkeypatch.setattr(codex_request, "_apply_kb", AsyncMock())
    monkeypatch.setattr(codex_request, "_decorate_llm_request", AsyncMock())
    monkeypatch.setattr(codex_request, "_apply_local_env_tools", local_tools)
    monkeypatch.setattr(codex_request, "_proactive_cron_job_tools", MagicMock())
    plugin_context = MagicMock()
    plugin_context.get_config.return_value = {}

    req = ProviderRequest(prompt="hi")
    await codex_request.prepare_codex_request(
        _Event(),
        req,
        plugin_context,
        {"provider_settings": {"computer_use_runtime": "local"}},
        {},
    )

    note = req.context_anchors["local_runtime"]
    assert "write_stdin" in note and "exec_command" in note
    assert "astrbot_shell_session" not in note
    assert "astrbot_shell_session" not in req.func_tool.names()


@pytest.mark.asyncio
async def test_a_plugin_request_keeps_its_own_images(monkeypatch, tmp_path):
    event = _Event()
    tracked: list[str] = []
    event.track_temporary_local_file = tracked.append
    collect = AsyncMock()
    monkeypatch.setattr(codex_request, "_collect_media", collect)
    for name in ("_get_session_conv", "_process_quote_message", "_apply_kb"):
        monkeypatch.setattr(codex_request, name, AsyncMock(return_value=None))
    monkeypatch.setattr(codex_request, "_decorate_llm_request", AsyncMock())
    monkeypatch.setattr(codex_request, "_proactive_cron_job_tools", MagicMock())
    resolved = str(tmp_path / "plugin.png")

    class _Resolver:
        def __init__(self, ref, media_type):
            assert ref == "base64://AAAA" and media_type == "image"

        async def to_path(self):
            return resolved

    monkeypatch.setattr(codex_request, "MediaResolver", _Resolver)
    plugin_context = MagicMock()
    plugin_context.get_config.return_value = {}
    req = ProviderRequest(
        prompt="look",
        image_urls=["https://example.com/a.png", "/tmp/b.png", "base64://AAAA"],
    )

    await codex_request.prepare_codex_request(
        event, req, plugin_context, {"provider_settings": {}}, {}, collect_media=False
    )

    # The message's attachments are not collected over the plugin's.
    collect.assert_not_awaited()
    # Codex reads URLs and local paths; a base64:// ref becomes a temp file.
    assert req.image_urls == ["https://example.com/a.png", "/tmp/b.png", resolved]
    assert tracked == [resolved]
