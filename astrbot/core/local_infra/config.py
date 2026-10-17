"""Settings of the local-multimodal-infra service (``local_infra``).

One system setting holds the server (address, inference token) and what
AstrBot uses it for. Every service is on by default and takes the server's
first enabled model of its task kind unless a model is chosen; a service can
be switched off on its own:

- ``stt`` / ``tts`` / ``embedding`` / ``rerank``: AstrBot providers
  (``local_infra/<service>``) that speech, knowledge bases and plugins pick
  like any other provider;
- ``moderation``: the content moderation of what goes to cloud models
  (``POST /v1/moderations``) and its options;
- ``realtime_voice``: the server of the ``local_infra`` realtime voice
  backend (``/v1/realtime``), and the ASR model its sessions hear with
  (``asr_model``, empty: the server's realtime default).
"""

from __future__ import annotations

import copy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

CONFIG_KEY = "local_infra"

#: AstrBot's services and the task kind a model needs to serve each.
SERVICE_TASK_KINDS: dict[str, str] = {
    "stt": "asr.transcribe",
    "tts": "tts.synthesize",
    "embedding": "text.embed",
    "rerank": "text.rerank",
    "moderation": "text.moderate",
    "realtime_voice": "voice.realtime",
}
SERVICES = tuple(SERVICE_TASK_KINDS)
#: Services that are AstrBot providers, with the provider type of each.
PROVIDER_SERVICES: dict[str, str] = {
    "stt": "speech_to_text",
    "tts": "text_to_speech",
    "embedding": "embedding",
    "rerank": "rerank",
}
#: Id prefix of the providers the infra settings create.
PROVIDER_ID_PREFIX = "local_infra/"

MODERATION_CATEGORIES = [
    "violent",
    "illegal",
    "sexual",
    "pii",
    "self-harm",
    "unethical",
    "political",
    "copyright",
    "jailbreak",
]
NSFW_LABELS = ["low", "medium", "high"]

DEFAULT_LOCAL_INFRA: dict[str, Any] = {
    "enable": False,
    # The server's HTTP address (its OpenAI-compatible API is under /v1).
    "base_url": "http://127.0.0.1:17890",
    # An inference token (LOCAL_MCP_INFER_TOKENS), empty when it has none.
    "token": "",
    # Seconds one request may take (speech synthesis of a long text).
    "timeout": 120,
    "services": {
        "stt": {"enable": True, "model": ""},
        "tts": {
            "enable": True,
            "model": "",
            # A WAV the voice is cloned from (relative to the data directory;
            # empty: the model's default voice) and what it says.
            "reference_audio": "",
            "reference_text": "",
        },
        "embedding": {"enable": True, "model": ""},
        "rerank": {"enable": True, "model": ""},
        "moderation": {
            "enable": True,
            # Replied instead of an answer when a message was blocked.
            "blocked_reply": "这条内容无法处理。",
            # A text is flagged when 1 - p(safe) exceeds this (lower is
            # stricter); an image when its NSFW probability exceeds the
            # other one.
            "threshold": 0.9,
            "nsfw_threshold": 0.5,
            # Image labels that count as NSFW, text categories that flag
            # (none selected: the server's defaults).
            "nsfw_labels": list(NSFW_LABELS),
            "categories": list(MODERATION_CATEGORIES),
        },
        "realtime_voice": {"enable": True, "asr_model": ""},
    },
}


def default_local_infra() -> dict[str, Any]:
    """A fresh copy of the default settings."""
    return copy.deepcopy(DEFAULT_LOCAL_INFRA)


def normalize_local_infra(raw: Any) -> dict[str, Any]:
    """The settings with defaults filled in and wrong types replaced.

    Args:
        raw: The stored ``local_infra`` value (anything).

    Returns:
        A new dict shaped like ``DEFAULT_LOCAL_INFRA``.
    """
    settings = default_local_infra()
    if not isinstance(raw, dict):
        return settings
    _merge(settings, raw)
    settings["base_url"] = str(settings["base_url"] or "").strip().rstrip("/")
    settings["token"] = str(settings["token"] or "").strip()
    try:
        settings["timeout"] = max(5, int(settings["timeout"]))
    except (TypeError, ValueError):
        settings["timeout"] = DEFAULT_LOCAL_INFRA["timeout"]
    for service in settings["services"].values():
        for key in ("model", "asr_model"):
            if key in service:
                service[key] = str(service[key] or "").strip()
    return settings


def _merge(into: dict, raw: dict) -> None:
    """Copy the keys of ``raw`` that ``into`` knows, keeping their types."""
    for key, default in into.items():
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(default, dict):
            if isinstance(value, dict):
                _merge(default, value)
        elif isinstance(default, bool):
            if isinstance(value, bool):
                into[key] = value
        elif isinstance(default, int | float):
            if isinstance(value, int | float) and not isinstance(value, bool):
                into[key] = value
            elif isinstance(value, str):
                try:
                    into[key] = type(default)(value)
                except ValueError:
                    pass
        elif isinstance(default, list):
            if isinstance(value, list):
                into[key] = [str(item).strip() for item in value if str(item).strip()]
        elif value is not None:
            into[key] = str(value)


def local_infra_settings(config: Any = None) -> dict[str, Any]:
    """The current settings (of ``config``, or AstrBot's system config)."""
    if config is None:
        from astrbot.core import astrbot_config

        config = astrbot_config
    try:
        raw = config.get(CONFIG_KEY)
    except AttributeError:
        raw = None
    return normalize_local_infra(raw)


def service_settings(service: str, settings: dict | None = None) -> dict[str, Any]:
    """One service's settings."""
    settings = settings or local_infra_settings()
    return settings["services"][service]


def service_switched_on(service: str, settings: dict | None = None) -> bool:
    """Whether the server is on and the service not switched off.

    Whether the server also has a model for it is ``catalog``'s business.
    """
    settings = settings or local_infra_settings()
    return bool(
        settings["enable"]
        and settings["base_url"]
        and settings["services"][service]["enable"]
    )


def http_base(settings: dict | None = None) -> str:
    """The server's HTTP address without a trailing ``/`` or ``/v1``."""
    settings = settings or local_infra_settings()
    base = settings["base_url"].rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def realtime_url(settings: dict | None = None) -> str:
    """The realtime voice WebSocket (``ws(s)://.../v1/realtime``)."""
    parts = urlsplit(http_base(settings))
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)
    path = parts.path.rstrip("/") + "/v1/realtime"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


def base_from_url(url: str) -> str:
    """The server's HTTP address behind one of its endpoint URLs.

    ``ws://host:17890/v1/realtime`` and ``http://host:17890/v1/moderations``
    both give ``http://host:17890``.
    """
    url = str(url or "").strip()
    if not url:
        return ""
    parts = urlsplit(url)
    scheme = {"wss": "https", "ws": "http"}.get(parts.scheme, parts.scheme)
    path = parts.path.rstrip("/")
    for suffix in ("/v1/realtime", "/v1/moderations", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((scheme or "http", parts.netloc, path, "", "")).rstrip("/")
