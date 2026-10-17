"""The local-multimodal-infra settings page of the WebUI (模型提供商 → 本地多模态服务)."""

from __future__ import annotations

from typing import Any

from astrbot.core.core_lifecycle import AstrBotCoreLifecycle
from astrbot.core.local_infra.catalog import catalog
from astrbot.core.local_infra.client import LocalInfraClient
from astrbot.core.local_infra.config import (
    CONFIG_KEY,
    PROVIDER_ID_PREFIX,
    PROVIDER_SERVICES,
    SERVICES,
    local_infra_settings,
    normalize_local_infra,
    realtime_url,
)


class LocalInfraServiceError(Exception):
    pass


class LocalInfraService:
    def __init__(self, core_lifecycle: AstrBotCoreLifecycle) -> None:
        self.core_lifecycle = core_lifecycle

    @property
    def _config(self):
        return self.core_lifecycle.astrbot_config

    def state(self) -> dict[str, Any]:
        """The settings, the server's models and how each service stands."""
        settings = local_infra_settings(self._config)
        provider_manager = self.core_lifecycle.provider_manager
        services = {}
        for service in SERVICES:
            state = catalog.resolve(service, settings)
            if service in PROVIDER_SERVICES:
                provider_id = f"{PROVIDER_ID_PREFIX}{service}"
                state["provider_id"] = provider_id
                state["provider_loaded"] = provider_id in provider_manager.inst_map
            services[service] = state
        return {
            "settings": settings,
            "services": services,
            "realtime_url": realtime_url(settings),
            "catalog": {
                "known": catalog.known,
                "models": (
                    [model.to_dict() for model in catalog.models]
                    if catalog.models is not None
                    else None
                ),
                "error": catalog.error,
                "fetched_at": catalog.fetched_at,
            },
        }

    async def refresh(self) -> dict[str, Any]:
        """Asks the server for its models again."""
        await catalog.refresh(local_infra_settings(self._config))
        await self.core_lifecycle.provider_manager.sync_local_infra()
        return self.state()

    async def save(self, raw: Any) -> dict[str, Any]:
        """Stores new settings, then refreshes the models and providers."""
        if not isinstance(raw, dict):
            raise LocalInfraServiceError("设置格式错误")
        settings = normalize_local_infra(raw)
        # The embedding model, once chosen (also automatically), is kept: a
        # page opened before it was pinned sends "" and must not unpin it.
        stored = local_infra_settings(self._config)["services"]["embedding"]
        embedding = settings["services"]["embedding"]
        if not embedding["model"] and stored["model"]:
            embedding["model"] = stored["model"]
        if settings["enable"] and not settings["base_url"].startswith(
            ("http://", "https://")
        ):
            raise LocalInfraServiceError("服务地址必须以 http:// 或 https:// 开头")
        self._config[CONFIG_KEY] = settings
        self._config.save_config()
        return await self.refresh()

    async def probe(self, raw: Any) -> dict[str, Any]:
        """The model list of a server that is not saved yet."""
        settings = normalize_local_infra(raw if isinstance(raw, dict) else {})
        if not settings["base_url"]:
            raise LocalInfraServiceError("请填写服务地址")
        client = LocalInfraClient.from_settings(settings)
        try:
            models = await client.list_models()
        except Exception as exc:
            raise LocalInfraServiceError(f"无法连接本地多模态服务：{exc}") from exc
        return {
            "models": [model.to_dict() for model in models],
            "known": any(model.task_kinds for model in models),
        }
