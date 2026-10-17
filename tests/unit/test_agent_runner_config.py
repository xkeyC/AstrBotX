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
    assert config["agent_runner"] == _codex_default_runner()
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
    assert voice["infra_url"] == "ws://10.0.0.2:17890/v1/realtime"
    assert voice["infra_token"] == "tok"
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
    assert voice["infra_url"] == "ws://infra:17890/v1/realtime"
    assert voice["ref_audio"] == "voice.wav"
    assert voice["emotion"] == "happy"
    assert voice["voice"] == "cove"
    # Once only, and the plugin's file is left alone.
    assert not migrate_config_on_load(config, path)
    assert (
        json.loads(plugin.read_text(encoding="utf-8"))["voice_backend"]
        == "local_cascade"
    )
