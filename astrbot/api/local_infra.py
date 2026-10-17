"""The local-multimodal-infra server for plugins.

``get_local_infra()`` (also ``Context.get_local_infra()``) returns a
``LocalInfraClient`` of the server AstrBot is set to, or ``None`` while it is
switched off; plugins use it instead of keeping their own address and token::

    from astrbot.api.local_infra import get_local_infra

    infra = get_local_infra()
    if infra is not None:
        wav = await infra.synthesize("你好")
        text = (await infra.transcribe("voice.wav"))["text"]
        vectors = await infra.embed(["a", "b"])
        output = await infra.run_task("ocr.lines", files={"image": "page.png"})

``service_model(name)`` tells the model a service (``stt``, ``tts``,
``embedding``, ``rerank``, ``moderation``, ``realtime_voice``) resolves to,
and whether it is available.
"""

from astrbot.core.local_infra import (
    InfraModel,
    LocalInfraClient,
    LocalInfraError,
    get_local_infra,
    local_infra_settings,
    realtime_url,
)


def service_model(service: str) -> dict:
    """``{"available", "model", "candidates", "reason", ...}`` of a service."""
    from astrbot.core.local_infra.catalog import catalog

    return catalog.resolve(service)


__all__ = [
    "InfraModel",
    "LocalInfraClient",
    "LocalInfraError",
    "get_local_infra",
    "local_infra_settings",
    "realtime_url",
    "service_model",
]
