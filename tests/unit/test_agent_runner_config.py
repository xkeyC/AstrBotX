import copy
import json
import logging
from types import SimpleNamespace

import pytest

from astrbot.core.config import agent_runner as agent_runner_module
from astrbot.core.config.agent_runner import (
    AGENT_RUNNER_CONFIG_DEFAULTS,
    AGENT_RUNNER_TYPES,
    DEFAULT_AGENT_RUNNER_TYPE,
    SELECTABLE_AGENT_RUNNER_TYPES,
    get_agent_runner_config_default,
    normalize_agent_runner,
)
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.config.default import CONFIG_METADATA_3, DEFAULT_CONFIG
from astrbot.core.pipeline.process_stage.stage import AgentRequestSubStage
from astrbot.core.utils.migra_helper import (
    _migrate_agent_runner_config,
    _migrate_local_infra,
    _migrate_web_search_settings,
    finalize_config_migrations,
)

LEGACY_RUNNER_TYPES = ["local", "dify", "coze", "dashscope", "deerflow"]


def _codex_default_runner() -> dict:
    return {
        "runner_type": "codex",
        "config": get_agent_runner_config_default("codex"),
    }


def test_codex_is_the_default_and_only_selectable_runner():
    assert DEFAULT_AGENT_RUNNER_TYPE == "codex"
    assert AGENT_RUNNER_TYPES == ("codex", "local")
    assert SELECTABLE_AGENT_RUNNER_TYPES == ("codex",)
    assert set(AGENT_RUNNER_CONFIG_DEFAULTS) == {"codex", "local"}
    assert DEFAULT_CONFIG["agent_runner"] == _codex_default_runner()

    selector = CONFIG_METADATA_3["ai_group"]["metadata"]["agent_runner"]["items"][
        "agent_runner.runner_type"
    ]
    assert selector["options"] == ["codex"]
    assert selector["labels"] == ["Codex"]
    assert set(selector["runner_defaults"]) == {"codex"}
    for removed in ("dify_runner", "coze_runner", "dashscope_runner"):
        assert removed not in CONFIG_METADATA_3["ai_group"]["metadata"]
    assert "deerflow_runner" not in CONFIG_METADATA_3["ai_group"]["metadata"]


@pytest.mark.parametrize("runner_type", ["codex", "local"])
def test_agent_runner_defaults_are_isolated(runner_type: str):
    first = get_agent_runner_config_default(runner_type)
    second = get_agent_runner_config_default(runner_type)

    first["test_mutation"] = True

    assert second == AGENT_RUNNER_CONFIG_DEFAULTS[runner_type]


@pytest.mark.parametrize("runner_type", ["dify", "coze", "dashscope", "deerflow"])
def test_removed_runner_defaults_are_unavailable(runner_type: str):
    with pytest.raises(ValueError, match="Unsupported Agent Runner type"):
        get_agent_runner_config_default(runner_type)


def test_codex_configuration_is_normalized_and_drops_unknown_fields():
    normalized = normalize_agent_runner(
        {
            "runner_type": "codex",
            "config": {
                "model": "gpt-test",
                "turn_timeout": "30",
                "dify_api_key": "secret",
                "thread_config": {"custom": {"value": 1}},
            },
        }
    )

    assert normalized == {
        "runner_type": "codex",
        "config": {
            **get_agent_runner_config_default("codex"),
            "model": "gpt-test",
            "turn_timeout": 30,
            "thread_config": {"custom": {"value": 1}},
        },
    }


@pytest.mark.parametrize(
    "agent_runner",
    [
        *({"runner_type": t, "config": {"x": 1}} for t in LEGACY_RUNNER_TYPES),
        {"runner_type": "unknown"},
        {"config": {}},
        {},
        None,
        "codex",
    ],
)
def test_legacy_or_unknown_runner_migrates_to_codex(agent_runner):
    assert normalize_agent_runner(agent_runner) == _codex_default_runner()


def test_runner_migration_warning_is_logged_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent_runner_module, "_warned_migrated_runner_types", set())
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    agent_runner_module.logger.addHandler(handler)
    try:
        for _ in range(3):
            normalize_agent_runner({"runner_type": "dify"})
        normalize_agent_runner({"runner_type": "coze"})
    finally:
        agent_runner_module.logger.removeHandler(handler)

    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "'dify'" in warnings[0].getMessage()
    assert "'coze'" in warnings[1].getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_type", LEGACY_RUNNER_TYPES)
async def test_agent_request_migrates_legacy_runner_to_codex(runner_type: str):
    config = {
        "wake_prefix": [],
        "provider_settings": {
            "wake_prefix": "",
            "streaming_response": True,
            "unsupported_streaming_strategy": "aggregate",
        },
        "agent_runner": {
            "runner_type": runner_type,
            "config": {"dify_api_key": "saved-key"},
        },
    }
    stage = AgentRequestSubStage()

    await stage.initialize(SimpleNamespace(astrbot_config=config))

    assert config["agent_runner"] == _codex_default_runner()
    assert stage.agent_sub_stage.runner_type == "codex"
    assert stage.agent_sub_stage.runner_config == get_agent_runner_config_default(
        "codex"
    )


def test_codex_configuration_round_trips(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    expected = {
        "runner_type": "codex",
        "config": {
            **get_agent_runner_config_default("codex"),
            "model": "gpt-test",
            "thread_config": {"nested": {"value": 1}},
        },
    }
    config["agent_runner"] = expected
    config_path = tmp_path / "codex.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = AstrBotConfig(config_path=str(config_path))
    loaded.save_config()
    reloaded = AstrBotConfig(config_path=str(config_path))

    assert reloaded["agent_runner"] == expected


@pytest.mark.parametrize("runner_type", LEGACY_RUNNER_TYPES)
def test_legacy_agent_runner_root_is_migrated_to_codex_on_load(
    tmp_path, runner_type: str
):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["agent_runner"] = {"runner_type": runner_type, "config": {"a": 1}}
    config_path = tmp_path / f"{runner_type}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = AstrBotConfig(config_path=str(config_path))

    assert loaded["agent_runner"] == _codex_default_runner()
    on_disk = json.loads(config_path.read_text(encoding="utf-8-sig"))
    assert on_disk["agent_runner"] == _codex_default_runner()


def test_legacy_provider_settings_are_dropped_and_codex_is_selected():
    config = {
        "config_version": 2,
        "provider": [],
        "provider_settings": {
            "agent_runner_type": "local",
            "default_provider_id": "chat-main",
            "fallback_chat_models": ["chat-backup"],
            "default_personality": "developer",
            "max_agent_step": 42,
            "dify_agent_runner_provider_id": "dify-provider",
            "streaming_response": True,
        },
    }

    assert _migrate_agent_runner_config(config)
    assert config["config_version"] == 4
    expected = _codex_default_runner()
    # The old default persona carries over.
    expected["config"]["persona_id"] = "developer"
    assert config["agent_runner"] == expected
    assert config["provider_settings"] == {"streaming_response": True}
    assert not _migrate_agent_runner_config(config)


@pytest.mark.parametrize("runner_type", ["dify", "coze", "dashscope", "deerflow"])
def test_legacy_third_party_runner_profile_migrates_to_codex(runner_type: str):
    config = {
        "config_version": 2,
        "provider": [],
        "provider_settings": {
            "agent_runner_type": runner_type,
            f"{runner_type}_agent_runner_provider_id": f"{runner_type}-provider",
        },
    }

    assert _migrate_agent_runner_config(config)
    assert config["agent_runner"] == _codex_default_runner()
    assert config["provider_settings"] == {}


def test_every_runner_setting_in_the_webui_exists_for_codex():
    def runner_keys(metadata: dict) -> list[str]:
        keys = []
        for key, meta in metadata.items():
            if key.startswith("agent_runner.config."):
                keys.append(key.removeprefix("agent_runner.config."))
            if isinstance(meta, dict) and isinstance(meta.get("items"), dict):
                keys += runner_keys(meta["items"])
        return keys

    codex_default = get_agent_runner_config_default("codex")
    keys = runner_keys(CONFIG_METADATA_3["ai_group"]["metadata"])
    assert "persona_id" in keys
    for key in keys:
        node = codex_default
        for part in key.split("."):
            assert isinstance(node, dict) and part in node, key
            node = node[part]


def test_old_config_drops_removed_settings_and_keeps_default_persona(tmp_path):
    from unittest.mock import MagicMock

    from astrbot.core.persona_mgr import PersonaManager

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["provider_settings"].update(
        {
            "default_image_caption_provider_id": "caption",
            "provider_pool": ["*"],
            "persona_pool": ["*"],
            "show_tool_use_status": True,
            "show_tool_call_result": True,
            "buffer_intermediate_messages": True,
            "max_quoted_fallback_images": 20,
            "file_extract": {"enable": True, "provider": "moonshotai"},
            "image_compress_options": {"max_size": 1280},
            "image_caption_prompt": "Describe it.",
        }
    )
    config["default_kb_collection"] = "old-kb"
    config["persona"] = [{"name": "v3", "prompt": "kept for the v4 migration"}]
    config["agent_runner"] = {
        "runner_type": "codex",
        "config": {
            **get_agent_runner_config_default("codex"),
            "persona_id": "developer",
            "model": {"provider_id": "chat", "fallback_provider_ids": []},
            "persona": {"persona_id": "other", "safety_mode": True},
            "compression": {"max_turns": 10},
            "misc": {"max_steps": 30, "tool_schema_mode": "full"},
        },
    }
    config_path = tmp_path / "old.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = AstrBotConfig(config_path=str(config_path))
    on_disk = json.loads(config_path.read_text(encoding="utf-8-sig"))

    removed = (
        "default_image_caption_provider_id",
        "provider_pool",
        "persona_pool",
        "show_tool_use_status",
        "show_tool_call_result",
        "buffer_intermediate_messages",
        "max_quoted_fallback_images",
        "file_extract",
        "image_compress_options",
    )
    for data in (loaded, on_disk):
        for key in removed:
            assert key not in data["provider_settings"], key
        assert data["provider_settings"]["image_caption_prompt"] == "Describe it."
        assert "default_kb_collection" not in data
        assert data["persona"] == config["persona"]

    # Saving from the WebUI and running a request both normalize the runner.
    runner = normalize_agent_runner(loaded["agent_runner"])
    assert runner == {
        "runner_type": "codex",
        "config": {
            **get_agent_runner_config_default("codex"),
            "persona_id": "developer",
        },
    }
    acm = MagicMock()
    acm.default_conf = {"agent_runner": runner}
    assert PersonaManager(MagicMock(), acm).default_persona == "developer"


def test_local_runner_default_persona_carries_over_to_codex(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["agent_runner"] = {
        "runner_type": "local",
        "config": {
            "persona": {"persona_id": "developer", "safety_mode": True},
            "compression": {"max_turns": 10},
        },
    }
    config_path = tmp_path / "local.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = AstrBotConfig(config_path=str(config_path))

    assert loaded["agent_runner"] == {
        "runner_type": "codex",
        "config": {
            **get_agent_runner_config_default("codex"),
            "persona_id": "developer",
            # The local persona's safety mode carries over too.
            "safety_mode": True,
        },
    }


def test_codex_agent_runner_root_is_left_untouched():
    runner = {
        "runner_type": "codex",
        "config": {
            **get_agent_runner_config_default("codex"),
            "model": "gpt-test",
        },
    }
    config = {
        "config_version": 4,
        "provider_settings": {},
        "agent_runner": copy.deepcopy(runner),
    }

    assert not _migrate_agent_runner_config(config)
    assert config["agent_runner"] == runner


def test_finalize_drops_legacy_agent_runner_providers_idempotently():
    global_config = {
        "provider_sources": [
            {
                "id": "dify-source",
                "type": "dify",
                "provider_type": "agent_runner",
                "dify_api_key": "source-key",
            }
        ],
        "provider": [
            {"id": "chat-model", "provider_type": "chat_completion"},
            {"id": "dify-provider", "provider_source_id": "dify-source"},
            {
                "id": "shared-deerflow",
                "type": "deerflow",
                "provider_type": "agent_runner",
                "deerflow_api_key": "shared-key",
            },
            {"id": "legacy-coze", "type": "coze", "coze_api_key": "unused"},
            {
                "id": "unused-custom-runner",
                "type": "custom-runner",
                "provider_type": "agent_runner",
            },
        ],
    }

    assert finalize_config_migrations([global_config])
    assert global_config["provider"] == [
        {"id": "chat-model", "provider_type": "chat_completion"}
    ]
    assert not finalize_config_migrations([global_config])


def test_new_agent_runner_config_is_authoritative_on_reload(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["config_version"] = 2
    config["provider_settings"]["agent_runner_type"] = "coze"
    config["agent_runner"] = {
        "runner_type": "codex",
        "config": {
            **get_agent_runner_config_default("codex"),
            "model": "saved-model",
            "thread_config": {"nested": {"value": 1}},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = AstrBotConfig(config_path=str(config_path))

    assert loaded["agent_runner"]["runner_type"] == "codex"
    assert loaded["agent_runner"]["config"]["model"] == "saved-model"
    assert loaded["agent_runner"]["config"]["thread_config"] == {"nested": {"value": 1}}
    assert "agent_runner_type" not in loaded["provider_settings"]


def test_load_migration_backs_up_replaced_runner_once(tmp_path):
    from astrbot.core.utils.migra_helper import migrate_config_on_load

    path = tmp_path / "cmd_config.json"
    legacy = {"agent_runner": {"runner_type": "local", "config": {"x": 1}}}
    assert migrate_config_on_load(dict(legacy), path) is True
    backup = tmp_path / "cmd_config.pre-codex.json"
    assert json.loads(backup.read_text(encoding="utf-8-sig")) == legacy
    migrate_config_on_load({"agent_runner": {"runner_type": "dify"}}, path)
    assert json.loads(backup.read_text(encoding="utf-8-sig")) == legacy


def test_mumble_voice_settings_move_to_the_runner():
    from pathlib import Path

    from astrbot.core.utils.migra_helper import migrate_config_on_load

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["platform"] = [
        {
            "id": "m1",
            "type": "mumble",
            "mumble_voice_name": "Mon3tr",
            "mumble_voice_voice": "cove",
            "mumble_voice_backend": "local_cascade",
            "mumble_cascade_url": "ws://10.0.0.2:17890/v1/realtime",
            "mumble_cascade_token": "tok",
            "mumble_cascade_ref_audio": "voice.wav",
            "mumble_cascade_tool_filler": "OK",
            "mumble_cascade_emotion": "sad",
            "mumble_cascade_emotion_strength": 0.4,
        },
        {"id": "qq", "type": "aiocqhttp"},
    ]

    assert migrate_config_on_load(config, Path("cmd_config.json"))

    platform = config["platform"][0]
    assert platform == {"id": "m1", "type": "mumble", "mumble_voice_name": "Mon3tr"}
    voice = normalize_agent_runner(config["agent_runner"])["config"]["realtime_voice"]
    assert voice["backend"] == "local_infra"
    # The server itself is the local-multimodal-infra setting's now.
    assert "infra_url" not in voice and "infra_token" not in voice
    infra = config["local_infra"]
    assert infra["enable"] is True
    assert infra["base_url"] == "http://10.0.0.2:17890"
    assert infra["token"] == "tok"
    # Moderation was not in use (no address): it stays off.
    assert infra["services"]["moderation"]["enable"] is False
    assert voice["ref_audio"] == "voice.wav"
    assert voice["emotion"] == "sad"
    assert voice["emotion_strength"] == 0.4
    assert voice["voice"] == "cove"
    # Once only.
    assert not migrate_config_on_load(config, Path("cmd_config.json"))


def test_voice_migration_survives_a_missing_runner_config():
    from pathlib import Path

    from astrbot.core.utils.migra_helper import _migrate_voice_settings

    config = {
        "agent_runner": {"runner_type": "codex", "config": None},
        "platform": [{"type": "mumble", "mumble_voice_model": "gpt-realtime"}],
    }
    assert _migrate_voice_settings(config)
    assert config["agent_runner"]["config"]["realtime_voice"] == {
        "model": "gpt-realtime"
    }
    assert Path("cmd_config.json").name == "cmd_config.json"


def test_the_qq_call_plugin_voice_backend_moves_to_the_runner(tmp_path):
    from astrbot.core.utils.migra_helper import migrate_config_on_load

    (tmp_path / "config").mkdir()
    plugin = tmp_path / "config" / "astrbot_plugin_qq_voice_call_config.json"
    plugin.write_text(
        json.dumps(
            {
                "voice_backend": "local_cascade",
                "cascade_url": "ws://infra:17890/v1/realtime",
                "cascade_ref_audio": "voice.wav",
                "cascade_tts_emotion": "happy",
                "voice": "cove",
            }
        ),
        encoding="utf-8",
    )
    config = copy.deepcopy(DEFAULT_CONFIG)
    path = tmp_path / "cmd_config.json"

    assert migrate_config_on_load(config, path)
    voice = normalize_agent_runner(config["agent_runner"])["config"]["realtime_voice"]
    assert voice["backend"] == "local_infra"
    assert config["local_infra"]["base_url"] == "http://infra:17890"
    assert voice["ref_audio"] == "voice.wav"
    assert voice["emotion"] == "happy"
    assert voice["voice"] == "cove"
    # Once only, and the plugin's file is left alone.
    assert not migrate_config_on_load(config, path)
    assert (
        json.loads(plugin.read_text(encoding="utf-8"))["voice_backend"]
        == "local_cascade"
    )


def test_moderation_and_voice_server_settings_move_to_local_infra():
    config = {
        "content_moderation_url": "http://10.0.0.3:17890/v1/moderations",
        "content_moderation_token": "t0k",
        "content_moderation_threshold": 0.7,
        "content_moderation_categories": ["pii"],
        "agent_runner": {
            "runner_type": "codex",
            "config": {
                "realtime_voice": {
                    "backend": "local_infra",
                    "infra_url": "ws://other:1/v1/realtime",
                    "infra_token": "",
                }
            },
        },
    }

    assert _migrate_local_infra(config)
    infra = config["local_infra"]
    assert infra["enable"] is True
    # The moderation address wins; the voice server was the same machine.
    assert infra["base_url"] == "http://10.0.0.3:17890"
    assert infra["token"] == "t0k"
    moderation = infra["services"]["moderation"]
    assert moderation["enable"] is True
    assert moderation["threshold"] == 0.7
    assert moderation["categories"] == ["pii"]
    assert not any(key.startswith("content_moderation") for key in config)
    assert config["agent_runner"]["config"]["realtime_voice"] == {
        "backend": "local_infra"
    }
    # Once done, nothing more to do.
    assert not _migrate_local_infra(config)


def test_a_fresh_config_gets_the_local_infra_switched_off():
    config = {}
    assert _migrate_local_infra(config)
    assert config["local_infra"]["enable"] is False
    assert config["local_infra"]["services"]["moderation"]["enable"] is True


def _legacy_web_search_settings() -> dict:
    return {
        "web_search": True,
        "websearch_provider": "tavily",
        "websearch_tavily_key": ["tvly-a", " ", "tvly-b"],
        "websearch_bocha_key": ["b"],
        "websearch_brave_key": [],
        "websearch_baidu_app_builder_key": "",
        "websearch_firecrawl_key": [],
        "websearch_exa_key": [],
        "websearch_anysearch_key": [],
        "web_search_link": True,
    }


def test_web_search_settings_are_dropped_and_tavily_keys_go_to_the_kb():
    config = {
        "provider_settings": {"enable": True, **_legacy_web_search_settings()},
    }

    assert _migrate_web_search_settings(config)
    assert config["provider_settings"] == {"enable": True}
    assert config["kb_tavily_key"] == ["tvly-a", "tvly-b"]
    # Once done, nothing more to do.
    assert not _migrate_web_search_settings(config)


def test_web_search_migration_keeps_an_existing_kb_tavily_key():
    config = {
        "kb_tavily_key": ["tvly-kb"],
        "provider_settings": {"websearch_tavily_key": "tvly-old"},
    }

    assert _migrate_web_search_settings(config)
    assert config["kb_tavily_key"] == ["tvly-kb"]
    assert config["provider_settings"] == {}


def test_web_search_settings_are_gone_after_loading(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["provider_settings"].update(_legacy_web_search_settings())
    config_path = tmp_path / "web_search.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = AstrBotConfig(config_path=str(config_path))

    on_disk = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for conf in (loaded, on_disk):
        assert not any(
            key.startswith(("web_search", "websearch"))
            for key in conf["provider_settings"]
        )
        assert conf["kb_tavily_key"] == ["tvly-a", "tvly-b"]


def test_moderation_switched_off_by_an_empty_address_stays_off():
    # Moderation was off (its address empty); turning the local server on
    # later for speech must not turn it on for every platform.
    config = {"content_moderation_url": "", "content_moderation_threshold": 0.8}
    assert _migrate_local_infra(config)
    moderation = config["local_infra"]["services"]["moderation"]
    assert moderation["enable"] is False
    assert moderation["threshold"] == 0.8


def test_shipyard_mode_becomes_the_sandbox_runtime(monkeypatch):
    from pathlib import Path

    from astrbot.core.utils import migra_helper
    from astrbot.core.utils.migra_helper import migrate_config_on_load

    warnings: list[str] = []
    monkeypatch.setattr(
        migra_helper.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(msg % args if args else msg),
    )

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["provider_settings"]["computer_use_runtime"] = "local"
    config["agent_runner"]["config"]["shipyard_mode"] = True

    assert migrate_config_on_load(config, Path("cmd_config.json"))

    # One switch: shipyard mode used to leave commands on this host when the
    # runtime said local.
    assert config["provider_settings"]["computer_use_runtime"] == "sandbox"
    assert "shipyard_mode" not in config["agent_runner"]["config"]
    assert any("Shipyard mode was removed" in w for w in warnings)
    # Once only.
    assert not migrate_config_on_load(config, Path("cmd_config.json"))


def test_shipyard_mode_off_is_just_dropped():
    from pathlib import Path

    from astrbot.core.utils.migra_helper import migrate_config_on_load

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["provider_settings"]["computer_use_runtime"] = "local"
    config["agent_runner"]["config"]["shipyard_mode"] = False

    assert migrate_config_on_load(config, Path("cmd_config.json"))

    assert config["provider_settings"]["computer_use_runtime"] == "local"
    assert "shipyard_mode" not in config["agent_runner"]["config"]
    assert "shipyard_mode" not in AGENT_RUNNER_CONFIG_DEFAULTS["codex"]


def test_old_default_persona_and_safety_mode_carry_over():
    config = {
        "provider_settings": {"default_personality": "dev", "llm_safety_mode": True},
        "agent_runner": {"runner_type": "codex", "config": {}},
    }
    assert _migrate_agent_runner_config(config)
    runner_config = config["agent_runner"]["config"]
    assert runner_config["persona_id"] == "dev"
    assert runner_config["safety_mode"] is True
    assert "default_personality" not in config["provider_settings"]

    # A local runner's persona safety mode carries over too.
    migrated = normalize_agent_runner(
        {"runner_type": "local", "config": {"persona": {"safety_mode": True}}}
    )
    assert migrated["config"]["safety_mode"] is True
