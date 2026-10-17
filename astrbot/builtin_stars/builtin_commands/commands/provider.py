from __future__ import annotations

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, MessageEventResult
from astrbot.core.provider.entities import ProviderType


class ProviderCommands:
    def __init__(self, context: star.Context) -> None:
        self.context = context

    async def provider(
        self,
        event: AstrMessageEvent,
        idx: str | int | None = None,
        idx2: int | None = None,
    ) -> None:
        """View or switch this session's TTS / STT provider.

        The chat model is Codex's (the Codex page or a permission rule), so
        chat providers are neither listed nor switched here.

        Args:
            event: Command event.
            idx: ``tts`` or ``stt`` to switch, or None to list.
            idx2: 1-based index of the provider to switch to.
        """
        umo = event.unified_msg_origin
        kinds = {
            "tts": (
                "TTS",
                self.context.get_all_tts_providers(),
                ProviderType.TEXT_TO_SPEECH,
            ),
            "stt": (
                "STT",
                self.context.get_all_stt_providers(),
                ProviderType.SPEECH_TO_TEXT,
            ),
        }

        if idx is None:
            using = {
                "tts": await self.context.get_using_tts_provider_async(umo=umo),
                "stt": await self.context.get_using_stt_provider_async(umo=umo),
            }
            parts = []
            for kind, (label, providers, _) in kinds.items():
                if not providers:
                    continue
                parts.append(f"## {label} Providers\n")
                for i, provider in enumerate(providers):
                    id_ = provider.meta().id
                    line = f"{i + 1}. {id_}"
                    if using[kind] and using[kind].meta().id == id_:
                        line += " 👈"
                    parts.append(line + "\n")
                parts.append(
                    f"Use /provider {kind} <idx> to switch {label} providers.\n\n"
                )
            if not parts:
                parts.append("No TTS or STT providers are configured.\n\n")
            parts.append(
                "The chat model is set on the Codex page of the WebUI "
                "(or by a permission rule)."
            )
            event.set_result(MessageEventResult().message("".join(parts)))
            return

        if idx not in kinds:
            event.set_result(
                MessageEventResult().message(
                    "❌ Invalid parameter. Use /provider, /provider tts <idx> "
                    "or /provider stt <idx>."
                )
            )
            return
        _, providers, provider_type = kinds[idx]
        if idx2 is None:
            event.set_result(MessageEventResult().message("Please enter the index."))
            return
        if idx2 > len(providers) or idx2 < 1:
            event.set_result(MessageEventResult().message("❌ Invalid provider index."))
            return
        id_ = providers[idx2 - 1].meta().id
        await self.context.provider_manager.set_provider(
            provider_id=id_,
            provider_type=provider_type,
            umo=umo,
        )
        event.set_result(
            MessageEventResult().message(f"✅ Successfully switched to {id_}.")
        )
