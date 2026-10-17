"""The server's models and the model each AstrBot service uses.

The catalog is fetched in the background (at start, after the settings
change, and every few minutes while the server is on) so that sync callers,
such as the moderation check, never wait on it. Each service resolves to:

- its chosen model, when the server has it enabled;
- otherwise the server's first enabled model of the service's task kind;
- nothing (unavailable) when the server has none, or the service is off.

Until the catalog is known (server unreachable, or one too old to list task
kinds), a switched-on service counts as available and lets the server pick.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from astrbot.core import logger

from .client import InfraModel, LocalInfraClient
from .config import (
    SERVICE_TASK_KINDS,
    SERVICES,
    local_infra_settings,
    service_switched_on,
)

#: Seconds between background refreshes.
REFRESH_INTERVAL_S = 300

Listener = Callable[[], Awaitable[None]]


class InfraCatalog:
    """The last model list fetched from the server."""

    def __init__(self) -> None:
        self.models: list[InfraModel] | None = None
        #: The base URL the models came from.
        self.base_url = ""
        self.error = ""
        self.fetched_at = 0.0
        self._listeners: list[Listener] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    @property
    def known(self) -> bool:
        """The server answered and lists the task kinds of its models."""
        return self.models is not None and any(m.task_kinds for m in self.models)

    def add_listener(self, listener: Listener) -> None:
        """Calls ``listener`` after a refresh that changed the catalog."""
        self._listeners.append(listener)

    async def refresh(self, settings: dict | None = None) -> None:
        """Fetches the model list; keeps the old one if the server fails."""
        settings = settings or local_infra_settings()
        async with self._lock:
            before = self._signature()
            if not settings["enable"] or not settings["base_url"]:
                self.models, self.base_url, self.error = None, "", ""
            else:
                client = LocalInfraClient.from_settings(settings)
                try:
                    self.models = await client.list_models()
                    self.base_url = client.base_url
                    self.error = ""
                    self.fetched_at = time.time()
                except Exception as exc:
                    if self.base_url != client.base_url:
                        self.models = None
                    self.error = str(exc) or type(exc).__name__
                    logger.warning(
                        "local-multimodal-infra at %s: cannot list models: %s",
                        client.base_url,
                        self.error,
                    )
            changed = self._signature() != before
        if changed:
            for listener in list(self._listeners):
                try:
                    await listener()
                except Exception:
                    logger.error("local-infra catalog listener failed", exc_info=True)

    def _signature(self) -> Any:
        if self.models is None:
            return (self.base_url, None)
        return (
            self.base_url,
            tuple(
                (m.id, tuple(m.task_kinds), m.enabled, m.downloaded)
                for m in self.models
            ),
        )

    def start(self) -> None:
        """Starts refreshing every ``REFRESH_INTERVAL_S`` (idempotent)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="local-infra:catalog")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(REFRESH_INTERVAL_S)
            try:
                await self.refresh()
            except Exception:
                logger.error("local-infra catalog refresh failed", exc_info=True)

    def resolve(self, service: str, settings: dict | None = None) -> dict[str, Any]:
        """How one service stands.

        Returns:
            ``{"service", "task_kind", "switched_on", "available", "model",
            "chosen", "candidates", "reason"}``; ``model`` is the model to
            ask for ("" lets the server pick), ``reason`` says why a
            switched-on service is unavailable.
        """
        settings = settings or local_infra_settings()
        kind = SERVICE_TASK_KINDS[service]
        chosen = str(settings["services"][service].get("model", "") or "")
        state: dict[str, Any] = {
            "service": service,
            "task_kind": kind,
            "switched_on": service_switched_on(service, settings),
            "available": False,
            "model": "",
            "chosen": chosen,
            "candidates": [],
            "reason": "",
        }
        if not state["switched_on"]:
            state["reason"] = "off"
            return state
        if not self.known:
            state["available"] = True
            state["model"] = chosen
            state["reason"] = "unknown"
            return state
        candidates = [
            m.id for m in self.models or [] if m.usable and kind in m.task_kinds
        ]
        state["candidates"] = candidates
        if chosen:
            if chosen in candidates:
                state["available"], state["model"] = True, chosen
            else:
                state["reason"] = "model_unavailable"
        elif candidates:
            state["available"], state["model"] = True, candidates[0]
        else:
            state["reason"] = "no_model"
        return state


catalog = InfraCatalog()


def resolved_services(settings: dict | None = None) -> dict[str, dict[str, Any]]:
    """``catalog.resolve`` of every service."""
    settings = settings or local_infra_settings()
    return {service: catalog.resolve(service, settings) for service in SERVICES}


def service_available(service: str, settings: dict | None = None) -> bool:
    """Whether AstrBot may use the server for ``service`` now."""
    return bool(catalog.resolve(service, settings)["available"])
