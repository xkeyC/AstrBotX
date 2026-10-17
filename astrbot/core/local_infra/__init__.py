"""local-multimodal-infra as AstrBot's first-tier local service.

- ``config``: the ``local_infra`` system setting (server, services);
- ``client``: ``LocalInfraClient``, the API plugins call;
- ``catalog``: the server's models, refreshed in the background, and the
  model each service resolves to (``catalog.catalog``, the shared one).

``config`` imports nothing of AstrBot, so the config defaults can use it;
the rest is loaded on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import (
    CONFIG_KEY,
    PROVIDER_ID_PREFIX,
    PROVIDER_SERVICES,
    SERVICE_TASK_KINDS,
    SERVICES,
    local_infra_settings,
    realtime_url,
    service_settings,
)

if TYPE_CHECKING:
    from .catalog import InfraCatalog, resolved_services, service_available
    from .client import InfraModel, LocalInfraClient, LocalInfraError

_LAZY = {
    "InfraCatalog": "catalog",
    "resolved_services": "catalog",
    "service_available": "catalog",
    "InfraModel": "client",
    "LocalInfraClient": "client",
    "LocalInfraError": "client",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_local_infra() -> LocalInfraClient | None:
    """A client of AstrBot's local-multimodal-infra server, or ``None``
    when it is switched off (WebUI: 模型提供商 → 本地多模态服务)."""
    from .client import LocalInfraClient

    settings = local_infra_settings()
    if not settings["enable"] or not settings["base_url"]:
        return None
    return LocalInfraClient.from_settings(settings)


__all__ = [
    "CONFIG_KEY",
    "PROVIDER_ID_PREFIX",
    "PROVIDER_SERVICES",
    "SERVICES",
    "SERVICE_TASK_KINDS",
    "InfraCatalog",
    "InfraModel",
    "LocalInfraClient",
    "LocalInfraError",
    "get_local_infra",
    "local_infra_settings",
    "realtime_url",
    "resolved_services",
    "service_available",
    "service_settings",
]
