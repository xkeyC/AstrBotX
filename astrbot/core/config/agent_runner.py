from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger("astrbot")

DEFAULT_AGENT_RUNNER_TYPE = "codex"
# "local" is kept only so internal code paths (tool_loop_agent, persona helpers)
# and their tests keep working; it is no longer selectable and
# ``normalize_agent_runner`` migrates it to Codex.
AGENT_RUNNER_TYPES = ("codex", "local")
SELECTABLE_AGENT_RUNNER_TYPES = ("codex",)
THIRD_PARTY_AGENT_RUNNER_TYPES = ("codex",)
# Runner types removed from this build; configs using them migrate to Codex.
LEGACY_AGENT_RUNNER_TYPES = ("dify", "coze", "dashscope", "deerflow")

_warned_migrated_runner_types: set[str] = set()

AGENT_RUNNER_CONFIG_DEFAULTS: dict[str, dict[str, Any]] = {
    "local": {
        "model": {
            "provider_id": "",
            "fallback_provider_ids": [],
            "request_max_retries": 5,
        },
        "persona": {
            "persona_id": "default",
            "safety_mode": True,
            "safety_mode_strategy": "system_prompt",
        },
        "compression": {
            "max_turns": -1,
            "trim_turns": 10,
            "overflow_strategy": "llm_compress",
            "instruction": "",
            "keep_recent_ratio": 0.15,
            "provider_id": "",
            "fallback_max_tokens": 128000,
        },
        "misc": {
            "max_steps": 30,
            "tool_schema_mode": "full",
            "tool_call_timeout": 120,
            "sanitize_context_by_modalities": False,
        },
    },
    "codex": {
        "codex_home": "",
        "tool_mode": "code_mode_only",
        "code_mode_host": "",
        "exec_as_function_tool": False,
        # Every agent file operation happens in the Shipyard Neo sandbox;
        # Codex only orchestrates and never touches this host's files.
        "shipyard_mode": False,
        "web_search": False,
        # Proxy for all of Codex's traffic, including realtime voice:
        # socks5://, socks5h:// or http:// (credentials may be embedded).
        "proxy": "",
        "model": "",
        "model_provider": "",
        # Custom endpoints: [{id, name, base_url, api_key, wire_api:
        # responses|chat, extra_body: {...} (chat: merged over each request
        # body), headers: {name: value}, compaction: auto|local|remote,
        # files_api: true (images uploaded once to the provider's Files API),
        # models: [{slug, context_window, auto_compact_token_limit,
        # reasoning_efforts: [...], default_reasoning_effort, image_input
        # (true/false; unset: Codex's default), metadata_json}]}] (see
        # model_provider_overrides).
        "model_providers": [],
        "reasoning_effort": "",
        "cwd": "",
        "developer_instructions": "",
        "base_instructions": "",
        "thread_config": {},
        # Codex native memories: per-chat local store. Senders whose permission
        # rule sets global_memory may write shared memories and delete memories
        # in any chat; automatic consolidation reaches the shared store only
        # from their private chats.
        "memory_enabled": False,
        "memory_auto_consolidate": True,
        "show_commentary": False,
        "safety_mode": False,
        "sync_history": True,
        "tool_call_timeout": 120,
        "turn_timeout": 600,
        # One chat runs one Codex turn at a time; messages from other senders
        # wait. This caps how many may wait before the rest are told to retry.
        # 0 removes the cap.
        "max_queued_turns": 5,
        # Realtime voice (Mumble, QQ calls). "builtin": Codex realtime, an
        # OpenAI realtime model (a ChatGPT subscription with voice).
        # "local_infra": a local-multimodal-infra server listens and speaks,
        # and a model of the providers above (text_*) does the talking.
        "realtime_voice": {
            "backend": "builtin",
            # builtin: voice and realtime model (empty: Codex's defaults).
            "voice": "",
            "model": "",
            # local_infra: the server's realtime WebSocket and token.
            "infra_url": "ws://127.0.0.1:17890/v1/realtime",
            "infra_token": "",
            # A WAV file the bot speaks like (relative to the data directory;
            # empty: the server's default voice), what it says (Qwen3-TTS
            # then clones the voice in context), and the emotion it speaks
            # with (IndexTTS-2.5) at a strength of 0 to 1.
            "ref_audio": "",
            "ref_text": "",
            "emotion": "calm",
            "emotion_strength": 0.8,
            # Speak a reply's first clause while the model is still writing
            # it (a TTS model that takes streamed text: Qwen3-TTS).
            "stream_text": True,
            # The model that talks (empty: the runner's provider and model).
            "text_model_provider": "",
            "text_model": "",
            "text_reasoning_effort": "",
            # Compact the voice history while idle once its prompt passes
            # this percentage of the context window (0: never).
            "idle_compact_percent": 70,
        },
    },
}


def get_agent_runner_config_default(runner_type: str) -> dict[str, Any]:
    """Return an isolated default configuration for an Agent Runner type.

    Args:
        runner_type: Short runner type name.

    Returns:
        A deep copy of the runner configuration defaults.

    Raises:
        ValueError: If the runner type is unsupported.
    """
    if runner_type not in AGENT_RUNNER_CONFIG_DEFAULTS:
        raise ValueError(f"Unsupported Agent Runner type: {runner_type}")
    return copy.deepcopy(AGENT_RUNNER_CONFIG_DEFAULTS[runner_type])


def resolve_context_compression_config(
    compression_config: dict[str, Any],
) -> dict[str, Any]:
    """Map session compression settings to main agent build arguments.

    Args:
        compression_config: The ``agent_runner.config.compression`` section.
            An empty mapping preserves the ordinary chat parser's defaults.

    Returns:
        Build arguments shared by chat, Cron and background-result wakeups.
        The input mapping is not modified.
    """
    max_turns = compression_config.get("max_turns", -1)
    trim_turns = compression_config.get("trim_turns", 1)
    dequeue_turns = min(
        max(1, trim_turns), max_turns - 1 if max_turns > 0 else trim_turns
    )
    return {
        "context_limit_reached_strategy": compression_config.get(
            "overflow_strategy", "truncate_by_turns"
        ),
        "llm_compress_instruction": compression_config.get("instruction", ""),
        "llm_compress_keep_recent_ratio": compression_config.get(
            "keep_recent_ratio", 0.15
        ),
        "llm_compress_provider_id": compression_config.get("provider_id", ""),
        "max_context_length": max_turns,
        "dequeue_context_length": max(1, dequeue_turns),
        "fallback_max_context_tokens": compression_config.get(
            "fallback_max_tokens", 128000
        ),
    }


def _normalize_value(value: Any, default: Any) -> Any:
    if isinstance(default, dict):
        if not isinstance(value, dict):
            return copy.deepcopy(default)
        if not default:
            return copy.deepcopy(value)
        return {
            key: _normalize_value(value.get(key), child_default)
            for key, child_default in default.items()
        }
    if isinstance(default, list):
        return (
            copy.deepcopy(value) if isinstance(value, list) else copy.deepcopy(default)
        )
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, int):
        if isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        if isinstance(value, bool):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    return copy.deepcopy(value) if value is not None else copy.deepcopy(default)


def _warn_runner_migration(runner_type: object) -> None:
    key = repr(runner_type)
    if key in _warned_migrated_runner_types:
        return
    _warned_migrated_runner_types.add(key)
    logger.warning(
        "Agent Runner type %r is no longer supported; migrating to %r with "
        "default configuration.",
        runner_type,
        DEFAULT_AGENT_RUNNER_TYPE,
    )


def normalize_agent_runner(agent_runner: object) -> dict[str, Any]:
    """Validate and normalize a complete Agent Runner configuration.

    Codex is the only supported Agent Runner. Any other value (legacy runner
    types such as ``local``, ``dify``, ``coze``, ``dashscope`` or ``deerflow``,
    unknown or missing types, or a non-object root) is migrated to Codex with
    its default configuration, logging a warning once per type.

    Args:
        agent_runner: Untrusted root Agent Runner configuration.

    Returns:
        A normalized configuration containing only fields for the Codex runner.
    """
    if not isinstance(agent_runner, dict):
        agent_runner = {}
    runner_type = agent_runner.get("runner_type")
    if runner_type not in SELECTABLE_AGENT_RUNNER_TYPES:
        _warn_runner_migration(runner_type)
        return {
            "runner_type": DEFAULT_AGENT_RUNNER_TYPE,
            "config": get_agent_runner_config_default(DEFAULT_AGENT_RUNNER_TYPE),
        }
    config = agent_runner.get("config", {})
    default = AGENT_RUNNER_CONFIG_DEFAULTS[runner_type]
    normalized = _normalize_value(config, default)
    return {"runner_type": runner_type, "config": normalized}
