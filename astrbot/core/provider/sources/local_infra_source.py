"""AstrBot providers served by the local-multimodal-infra server.

They are not configured one by one: the ``local_infra`` system setting
creates one of each available service (``local_infra/stt``,
``local_infra/tts``, ``local_infra/embedding``, ``local_infra/rerank``). The
server address, token and the model each service resolves to (see
``astrbot.core.local_infra.catalog``) are read on every call, so changing
them needs no reload and holders of a provider (knowledge bases) keep a
working one.
"""

from __future__ import annotations

import asyncio
import os

from astrbot import logger
from astrbot.core.local_infra import LocalInfraClient, local_infra_settings
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.core.utils.media_utils import MediaResolver

from ..entities import ProviderType, RerankResult
from ..provider import EmbeddingProvider, RerankProvider, STTProvider, TTSProvider
from ..register import register_provider_adapter

#: Seconds the embedding size probe may take while AstrBot starts.
DIM_PROBE_TIMEOUT_S = 10


def _client() -> LocalInfraClient:
    """A client asking each service for the model it resolves to now."""
    return LocalInfraClient.from_settings(local_infra_settings())


@register_provider_adapter(
    "local_infra_stt",
    "local-multimodal-infra 语音识别",
    provider_type=ProviderType.SPEECH_TO_TEXT,
)
class LocalInfraSTTProvider(STTProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

    async def get_text(self, audio_url: str) -> str:
        async with MediaResolver(
            audio_url,
            media_type="audio",
            default_suffix=".wav",
        ).as_path(target_format="wav") as audio:
            output = await _client().transcribe(audio.path)
        return str(output.get("text") or "").strip()


@register_provider_adapter(
    "local_infra_tts",
    "local-multimodal-infra 语音合成",
    provider_type=ProviderType.TEXT_TO_SPEECH,
)
class LocalInfraTTSProvider(TTSProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

    async def get_audio(self, text: str) -> str:
        tts = local_infra_settings()["services"]["tts"]
        reference = None
        if ref := str(tts.get("reference_audio") or "").strip():
            path = (
                ref
                if os.path.isabs(ref)
                else os.path.join(get_astrbot_data_path(), ref)
            )
            if os.path.isfile(path):
                reference = path
            else:
                logger.warning("local-infra TTS reference audio %s is missing", path)
        return await _client().synthesize(
            text,
            reference_audio=reference,
            reference_text=str(tts.get("reference_text") or "").strip() or None,
        )


@register_provider_adapter(
    "local_infra_embedding",
    "local-multimodal-infra 嵌入",
    provider_type=ProviderType.EMBEDDING,
)
class LocalInfraEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        self.dim = int(provider_config.get("embedding_dimensions") or 0)

    async def initialize(self) -> None:
        if self.dim:
            return
        try:
            await asyncio.wait_for(self.get_embedding("dimension"), DIM_PROBE_TIMEOUT_S)
        except Exception as exc:
            logger.warning(
                "local-infra embedding %s: cannot learn its size yet: %s",
                self.provider_config.get("id"),
                exc,
            )

    async def get_embedding(self, text: str) -> list[float]:
        return (await self.get_embeddings([text]))[0]

    async def get_embeddings(self, text: list[str]) -> list[list[float]]:
        vectors = await _client().embed(text)
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return vectors

    def get_dim(self) -> int:
        return self.dim


@register_provider_adapter(
    "local_infra_rerank",
    "local-multimodal-infra 重排序",
    provider_type=ProviderType.RERANK,
)
class LocalInfraRerankProvider(RerankProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        results = await _client().rerank(query, documents, top_n=top_n)
        return [
            RerankResult(
                index=int(item["index"]),
                relevance_score=float(item.get("relevance_score", 0.0)),
            )
            for item in results
        ]
