from __future__ import annotations

import copy
import json
import logging
import platform
import traceback
from pathlib import Path
from typing import Any

from astrbot.core.config.agent_runner import (
    DEFAULT_AGENT_RUNNER_TYPE,
    LEGACY_AGENT_RUNNER_TYPES,
    get_agent_runner_config_default,
    normalize_agent_runner,
)

logger = logging.getLogger("astrbot")

_LEGACY_AGENT_RUNNER_PROVIDER_ID_KEYS = {
    "dify": "dify_agent_runner_provider_id",
    "coze": "coze_agent_runner_provider_id",
    "dashscope": "dashscope_agent_runner_provider_id",
    "deerflow": "deerflow_agent_runner_provider_id",
}
_LEGACY_AGENT_RUNNER_SETTING_KEYS = (
    "agent_runner_type",
    *_LEGACY_AGENT_RUNNER_PROVIDER_ID_KEYS.values(),
    "default_provider_id",
    "fallback_chat_models",
    "request_max_retries",
    "default_personality",
    "llm_safety_mode",
    "safety_mode_strategy",
    "max_agent_step",
    "tool_schema_mode",
    "tool_call_timeout",
    "sanitize_context_by_modalities",
    "context_limit_reached_strategy",
    "llm_compress_instruction",
    "llm_compress_keep_recent_ratio",
    "llm_compress_provider_id",
    "max_context_length",
    "dequeue_context_length",
    "fallback_max_context_tokens",
)


def _get_effective_provider_map(config: object) -> dict[str, dict[str, Any]]:
    """Build providers with their Provider Source fields merged in.

    Args:
        config: Configuration containing provider and provider_sources lists.

    Returns:
        Effective providers indexed by provider ID.
    """
    if not isinstance(config, dict):
        return {}
    provider_sources = config.get("provider_sources", [])
    source_map = {
        source.get("id"): source
        for source in provider_sources
        if isinstance(source, dict) and source.get("id")
    }
    provider_map: dict[str, dict[str, Any]] = {}
    for provider in config.get("provider", []):
        if not isinstance(provider, dict) or not provider.get("id"):
            continue
        effective_provider = copy.deepcopy(
            source_map.get(provider.get("provider_source_id"), {})
        )
        effective_provider.update(copy.deepcopy(provider))
        provider_map[provider["id"]] = effective_provider
    return provider_map


def _get_provider_runner_type(provider: object) -> str | None:
    """Return the removed third-party runner type represented by a provider.

    Legacy Dify/Coze/DashScope/DeerFlow Agent Runner providers are detected so
    they can be dropped from the provider list; their settings are not migrated
    because Codex is the only supported Agent Runner.

    Args:
        provider: Effective provider configuration.

    Returns:
        Runner type when the provider is a legacy Agent Runner, otherwise None.
    """
    if not isinstance(provider, dict):
        return None
    provider_type = provider.get("provider_type")
    runner_type = provider.get("type") or provider.get("provider")
    if provider_type == "agent_runner" and runner_type in LEGACY_AGENT_RUNNER_TYPES:
        return runner_type
    expected_field = {
        "dify": "dify_api_key",
        "coze": "coze_api_key",
        "dashscope": "dashscope_app_id",
        "deerflow": "deerflow_api_base",
    }
    if (
        runner_type in LEGACY_AGENT_RUNNER_TYPES
        and expected_field[runner_type] in provider
    ):
        return runner_type
    return None


def _default_agent_runner() -> dict[str, Any]:
    return {
        "runner_type": DEFAULT_AGENT_RUNNER_TYPE,
        "config": get_agent_runner_config_default(DEFAULT_AGENT_RUNNER_TYPE),
    }


def _migrate_agent_runner_config(config: dict[str, Any]) -> bool:
    """Migrate legacy Agent Runner fields in one core configuration.

    Codex is the only supported Agent Runner. Legacy ``provider_settings``
    runner fields are dropped and any non-Codex ``agent_runner`` root (legacy
    ``local``/Dify/Coze/DashScope/DeerFlow or unknown) is replaced with the
    Codex defaults.

    Args:
        config: Mutable AstrBot configuration loaded from disk.

    Returns:
        Whether the configuration changed.
    """
    changed = False
    provider_settings = config.get("provider_settings")
    if not isinstance(provider_settings, dict):
        provider_settings = {}
        config["provider_settings"] = provider_settings
        changed = True

    for key in _LEGACY_AGENT_RUNNER_SETTING_KEYS:
        if key in provider_settings:
            provider_settings.pop(key)
            changed = True

    existing_agent_runner = config.get("agent_runner")
    if not isinstance(existing_agent_runner, dict):
        config["agent_runner"] = _default_agent_runner()
        changed = True
    elif existing_agent_runner.get("runner_type") != DEFAULT_AGENT_RUNNER_TYPE:
        config["agent_runner"] = normalize_agent_runner(existing_agent_runner)
        changed = True

    config_version = config.get("config_version")
    if not isinstance(config_version, int) or config_version < 4:
        config["config_version"] = 4
        changed = True
    return changed


def migrate_config_on_load(config: dict[str, Any], config_path: Path) -> bool:
    """Run core configuration migrations before integrity cleanup.

    Args:
        config: Mutable AstrBot configuration loaded from disk.
        config_path: Path of the configuration being loaded; a one-time
            ``*.pre-codex.json`` backup is written next to it when the runner changes.

    Returns:
        Whether the configuration changed.
    """
    original = copy.deepcopy(config)
    changed = _migrate_agent_runner_config(config)
    if changed and original.get("agent_runner") != config.get("agent_runner"):
        _backup_pre_codex_config(original, config_path)
    changed = _migrate_dynamic_persona_bindings(config, config_path) or changed
    changed = _migrate_voice_settings(config) or changed
    changed = _migrate_qq_call_voice(config, config_path) or changed
    changed = _migrate_windows_member_scope(config) or changed
    return changed


def _migrate_windows_member_scope(config: dict[str, Any]) -> bool:
    """Upstream v4.29.1 (config_version 5): on Windows the members' old
    default Local policy (no execution, no network, no files) becomes
    workspace file access, once; code execution stays off.

    Args:
        config: Mutable AstrBot configuration loaded from disk.

    Returns:
        Whether the configuration changed.
    """
    config_version = config.get("config_version")
    if isinstance(config_version, int) and config_version >= 5:
        return False
    provider_settings = config.get("provider_settings", {})
    permissions = (
        provider_settings.get("computer_use_local_permissions", {})
        if isinstance(provider_settings, dict)
        else {}
    )
    if (
        platform.system() == "Windows"
        and isinstance(permissions, dict)
        and permissions.get("member")
        == {
            "allow_execution": False,
            "allow_network": False,
            "filesystem_scope": "none",
        }
    ):
        # The old default and an explicit disabled policy are indistinguishable.
        permissions["member"]["filesystem_scope"] = "workspace"
        logger.info(
            "Migrated Windows member Local file access from disabled to workspace; "
            "code execution and execution network access remain disabled."
        )
    config["config_version"] = 5
    return True


# Mumble's voice settings that are now the Codex runner's (realtime_voice).
_MUMBLE_VOICE_KEYS = (
    "mumble_voice_voice",
    "mumble_voice_model",
    "mumble_voice_backend",
    "mumble_cascade_url",
    "mumble_cascade_token",
    "mumble_cascade_ref_audio",
    "mumble_cascade_tool_filler",
    "mumble_cascade_emotion",
    "mumble_cascade_emotion_strength",
)


def _migrate_voice_settings(config: dict[str, Any]) -> bool:
    """Move Mumble's voice backend settings into the Codex runner.

    The realtime voice backend (Codex realtime or a local-multimodal-infra
    server), its voice and model are one setting of the runner now
    (``agent_runner.config.realtime_voice``). The first Mumble platform using
    the local server hands its server settings over; set voices and models
    fill empty ones. The Mumble keys are removed.

    Args:
        config: Mutable AstrBot configuration loaded from disk.

    Returns:
        Whether the configuration changed.
    """
    platforms = config.get("platform")
    runner = config.get("agent_runner")
    if not isinstance(platforms, list) or not isinstance(runner, dict):
        return False
    if not isinstance(runner.get("config"), dict):
        runner["config"] = {}
    runner_config = runner["config"]
    changed = False
    for platform_conf in platforms:
        if not isinstance(platform_conf, dict) or not any(
            key in platform_conf for key in _MUMBLE_VOICE_KEYS
        ):
            continue
        voice = runner_config.get("realtime_voice")
        if not isinstance(voice, dict):
            voice = runner_config["realtime_voice"] = {}
        if (
            platform_conf.get("mumble_voice_backend") == "local_cascade"
            and voice.get("backend") != "local_infra"
        ):
            voice["backend"] = "local_infra"
            for old, new in (
                ("mumble_cascade_url", "infra_url"),
                ("mumble_cascade_token", "infra_token"),
                ("mumble_cascade_ref_audio", "ref_audio"),
                ("mumble_cascade_emotion", "emotion"),
                ("mumble_cascade_emotion_strength", "emotion_strength"),
            ):
                if platform_conf.get(old) not in (None, ""):
                    voice[new] = platform_conf[old]
        for old, new in (
            ("mumble_voice_voice", "voice"),
            ("mumble_voice_model", "model"),
        ):
            if platform_conf.get(old) and not voice.get(new):
                voice[new] = platform_conf[old]
        for key in _MUMBLE_VOICE_KEYS:
            platform_conf.pop(key, None)
        changed = True
    if changed:
        logger.warning(
            "Mumble voice backend settings moved to the Codex runner's realtime voice settings."
        )
    return changed


# The QQ call plugin's config, whose voice backend settings moved too.
QQ_CALL_PLUGIN_CONFIG = "astrbot_plugin_qq_voice_call_config.json"


def _migrate_qq_call_voice(config: dict[str, Any], config_path: Path) -> bool:
    """Take over the QQ call plugin's voice backend, once.

    The plugin's own ``voice_backend`` / ``cascade_*`` / ``voice`` /
    ``voice_model`` settings are the runner's ``realtime_voice`` now. A plugin
    on the local server hands its server settings over (unless the runner is
    already set to one); the plugin's file is left as it is (it ignores them).

    Args:
        config: Mutable AstrBot configuration loaded from disk.
        config_path: Path of the configuration being loaded.

    Returns:
        Whether the configuration changed.
    """
    if config_path.name != "cmd_config.json":
        return False
    plugin_path = config_path.parent / "config" / QQ_CALL_PLUGIN_CONFIG
    marker = plugin_path.with_name(f"{plugin_path.name}.voice-migrated")
    runner = config.get("agent_runner")
    if not plugin_path.is_file() or marker.exists() or not isinstance(runner, dict):
        return False
    try:
        plugin = json.loads(plugin_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read QQ call plugin config %s: %s", plugin_path, exc)
        return False
    if not isinstance(runner.get("config"), dict):
        runner["config"] = {}
    voice = runner["config"].get("realtime_voice")
    if not isinstance(voice, dict):
        voice = runner["config"]["realtime_voice"] = {}
    changed = False
    if (
        plugin.get("voice_backend") == "local_cascade"
        and voice.get("backend") != "local_infra"
    ):
        voice["backend"] = "local_infra"
        for old, new in (
            ("cascade_url", "infra_url"),
            ("cascade_token", "infra_token"),
            ("cascade_ref_audio", "ref_audio"),
            ("cascade_tts_emotion", "emotion"),
            ("cascade_tts_emotion_strength", "emotion_strength"),
        ):
            if plugin.get(old) not in (None, ""):
                voice[new] = plugin[old]
        changed = True
    for old, new in (("voice", "voice"), ("voice_model", "model")):
        if plugin.get(old) and not voice.get(new):
            voice[new] = plugin[old]
            changed = True
    try:
        marker.write_text(
            "The QQ call plugin's voice backend settings were taken over by realtime_voice.\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Cannot write QQ call migration marker %s: %s", marker, exc)
    if changed:
        logger.warning(
            "The QQ call plugin's voice backend settings moved to the Codex runner's realtime voice settings."
        )
    return changed


def _migrate_dynamic_persona_bindings(
    config: dict[str, Any], config_path: Path
) -> bool:
    """Import DynamicPersona plugin bindings as permission rules, once."""
    from astrbot.core.permission_rules import (
        CONFIG_KEY,
        DYNAMIC_PERSONA_CONFIG,
        rules_from_dynamic_persona,
    )

    if config.get(CONFIG_KEY) or config_path.name != "cmd_config.json":
        return False
    plugin_path = config_path.parent / "config" / DYNAMIC_PERSONA_CONFIG
    # Marks a completed import so rules the user later clears stay cleared.
    marker = plugin_path.with_name(f"{plugin_path.name}.imported")
    if not plugin_path.is_file() or marker.exists():
        return False
    try:
        plugin_conf = json.loads(plugin_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read DynamicPersona config %s: %s", plugin_path, exc)
        return False
    rules = rules_from_dynamic_persona(plugin_conf)
    if not rules:
        return False
    config[CONFIG_KEY] = rules
    try:
        marker.write_text(
            "DynamicPersona bindings were imported into permission_rules.\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Cannot write DynamicPersona import marker %s: %s", marker, exc)
    logger.warning(
        "Imported %d DynamicPersona binding(s) into permission_rules; "
        "disable the DynamicPersona plugin.",
        len(rules),
    )
    return True


def _backup_pre_codex_config(original: dict[str, Any], config_path: Path) -> None:
    """Keep the pre-migration file once, so replaced runner settings stay recoverable."""
    backup = config_path.with_name(f"{config_path.stem}.pre-codex{config_path.suffix}")
    if backup.exists():
        return
    try:
        backup.write_text(
            json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8-sig"
        )
        logger.warning(
            "Agent runner migrated to Codex; previous config saved to %s", backup
        )
    except OSError as exc:
        logger.warning("Failed to back up config before Codex migration: %s", exc)


def finalize_config_migrations(configs: list[dict[str, Any]]) -> bool:
    """Clean legacy shared data after every profile has been migrated.

    Args:
        configs: Loaded configurations with the default configuration first.

    Returns:
        Whether the default configuration changed.
    """
    if not configs:
        return False
    default_config = configs[0]
    providers = default_config.get("provider", [])
    if not isinstance(providers, list):
        return False
    effective_provider_map = _get_effective_provider_map(default_config)
    filtered_providers = [
        provider
        for provider in providers
        if not (
            isinstance(provider, dict)
            and (
                provider.get("provider_type") == "agent_runner"
                or effective_provider_map.get(provider.get("id"), {}).get(
                    "provider_type"
                )
                == "agent_runner"
                or _get_provider_runner_type(
                    effective_provider_map.get(provider.get("id"), provider)
                )
                is not None
            )
        )
    ]
    if len(filtered_providers) == len(providers):
        return False
    default_config["provider"] = filtered_providers
    return True


def _migra_provider_to_source_structure(conf: Any) -> None:
    """Migrate old providers to the provider-source structure.

    Args:
        conf: Mutable default configuration with a save_config method.
    """
    providers = conf.get("provider", [])
    provider_sources = conf.get("provider_sources", [])
    migrated = False
    provider_only_fields = {
        "id",
        "provider_source_id",
        "model",
        "modalities",
        "custom_extra_body",
        "enable",
    }
    source_exclude_fields = provider_only_fields | {"model_config"}

    for provider in providers:
        if provider.get("provider_source_id"):
            continue
        provider_type = provider.get("provider_type", "")
        if provider_type != "chat_completion":
            old_type = provider.get("type", "")
            if "chat_completion" not in old_type:
                continue

        migrated = True
        logger.info("Migrating provider %s to new structure", provider.get("id"))
        source_fields = {
            key: value
            for key, value in list(provider.items())
            if key not in source_exclude_fields
        }
        source_id = provider.get("id", "") + "_source"
        new_source = {"id": source_id, **source_fields}
        provider["provider_source_id"] = source_id

        if "model_config" in provider and isinstance(provider["model_config"], dict):
            model_config = provider["model_config"]
            provider["model"] = model_config.get("model", "")
            extra_body_fields = {k: v for k, v in model_config.items() if k != "model"}
            if extra_body_fields:
                if "custom_extra_body" not in provider:
                    provider["custom_extra_body"] = {}
                provider["custom_extra_body"].update(extra_body_fields)

        if "modalities" not in provider:
            provider["modalities"] = []
        if "custom_extra_body" not in provider:
            provider["custom_extra_body"] = {}
        keys_to_remove = [key for key in provider if key not in provider_only_fields]
        for key in keys_to_remove:
            del provider[key]
        provider_sources.append(new_source)

    if migrated:
        conf["provider_sources"] = provider_sources
        conf.save_config()
        logger.info("Provider-source structure migration completed")


def _migrate_legacy_openai_responses_sources(conf: Any) -> None:
    """Move the fork's legacy Responses mode to the dedicated provider type."""
    migrated = False
    for source in conf.get("provider_sources", []):
        if not isinstance(source, dict) or "api_mode" not in source:
            continue

        api_mode = source.pop("api_mode")
        migrated = True
        if source.get("type") == "openai_chat_completion" and api_mode == "responses":
            source["type"] = "openai_responses"
            logger.info(
                "Migrated legacy Responses provider source %s to openai_responses",
                source.get("id", "<unknown>"),
            )

    if migrated:
        conf.save_config()


async def migra(
    db: Any, astrbot_config_mgr: Any, umop_config_router: Any, acm: Any
) -> None:
    """Run migrations that require initialized configuration or database state.

    Args:
        db: Initialized AstrBot database.
        astrbot_config_mgr: Configuration manager used by legacy migrations.
        umop_config_router: Initialized UMOP configuration router.
        acm: Initialized AstrBot configuration manager.
    """
    from astrbot.core.db.migration.migra_45_to_46 import migrate_45_to_46
    from astrbot.core.db.migration.migra_token_usage import migrate_token_usage
    from astrbot.core.db.migration.migra_webchat_session import (
        migrate_webchat_session,
    )

    try:
        await migrate_45_to_46(astrbot_config_mgr, umop_config_router)
    except Exception as exc:
        logger.error("Migration from version 4.5 to 4.6 failed: %s", exc)
        logger.error(traceback.format_exc())

    try:
        await migrate_webchat_session(db)
    except Exception as exc:
        logger.error("Migration for webchat session failed: %s", exc)
        logger.error(traceback.format_exc())

    try:
        await migrate_token_usage(db)
    except Exception as exc:
        logger.error("Migration for token_usage column failed: %s", exc)
        logger.error(traceback.format_exc())

    configs = list(acm.confs.values())
    try:
        if finalize_config_migrations(configs):
            configs[0].save_config()
            logger.info("Agent Runner configuration migration completed")
    except Exception as exc:
        logger.error("Agent Runner configuration migration failed: %s", exc)
        logger.error(traceback.format_exc())

    try:
        _migra_provider_to_source_structure(acm.default_conf)
    except Exception as exc:
        logger.error("Migration for provider-source structure failed: %s", exc)
        logger.error(traceback.format_exc())

    # Replace the fork's old api_mode patch with the upstream dedicated adapter.
    try:
        _migrate_legacy_openai_responses_sources(acm.default_conf)
    except Exception as e:
        logger.error(f"Migration for legacy Responses providers failed: {e!s}")
        logger.error(traceback.format_exc())
