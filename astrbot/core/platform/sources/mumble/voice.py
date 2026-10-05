"""Realtime prompts of Mumble voice conversations (see ``astrbot.core.voice``)."""

from __future__ import annotations

from astrbot.core.voice.session import VoiceOptions, group_rule

CHANNEL_PROMPT = """Your name is {name}.

You are listening to a voice chat room where several people talk with each other. Almost everything said there is people talking to each other, not to you.

{rule}

When you are addressed, answer briefly in the speaker's language. Delegate real tasks (anything needing facts, lookups or work) to the backend and tell the speaker the result briefly."""

WHISPER_PROMPT = """Your name is {name}. You are talking privately, one to one, with {speaker} in a Mumble voice chat. Everything you hear is meant for you.

Answer briefly in the speaker's language. Delegate real tasks (anything needing facts, lookups or work) to the backend and tell the speaker the result briefly."""


def channel_prompt(options: VoiceOptions, *, gated: bool) -> str:
    """The channel's prompt; ``gated``: the voice server passes on only what
    calls the bot (``group_rule``)."""
    # The session adds the voice persona (or the platform's extra prompt)
    # and the time.
    return CHANNEL_PROMPT.format(
        name=options.name, rule=group_rule(options, gated=gated)
    )


def whisper_prompt(options: VoiceOptions, speaker: str) -> str:
    return WHISPER_PROMPT.format(name=options.name, speaker=speaker)
