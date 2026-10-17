"""The per-session chat provider choices are removed once at startup."""

from types import SimpleNamespace

import pytest

import astrbot.core as core
from astrbot.core.utils.migra_helper import _drop_session_chat_providers


class _FakePreferences:
    def __init__(self, rows: dict[tuple[str, str, str], object]) -> None:
        self.rows = rows

    async def global_get(self, key, default=None):
        return self.rows.get(("global", "global", key), default)

    async def global_put(self, key, value):
        self.rows[("global", "global", key)] = value

    async def range_get_async(self, scope, scope_id=None, key=None):
        return [
            SimpleNamespace(scope=s, scope_id=sid, key=k)
            for (s, sid, k) in self.rows
            if s == scope
            and (scope_id is None or sid == scope_id)
            and (key is None or k == key)
        ]

    async def remove_async(self, scope, scope_id, key):
        self.rows.pop((scope, scope_id, key), None)


@pytest.mark.asyncio
async def test_session_chat_providers_are_removed_once(monkeypatch):
    prefs = _FakePreferences(
        {
            ("umo", "qq:GroupMessage:1", "provider_perf_chat_completion"): "gpt",
            ("umo", "qq:GroupMessage:1", "provider_perf_text_to_speech"): "tts",
            ("umo", "qq:FriendMessage:2", "provider_perf_chat_completion"): "gpt",
        }
    )
    monkeypatch.setattr(core, "sp", prefs)

    await _drop_session_chat_providers()

    # TTS / STT choices stay; the chat ones are gone.
    assert set(prefs.rows) == {
        ("umo", "qq:GroupMessage:1", "provider_perf_text_to_speech"),
        ("global", "global", "migration_done_drop_session_chat_provider"),
    }

    # Once only: a plugin may set one again afterwards.
    prefs.rows[("umo", "qq:GroupMessage:1", "provider_perf_chat_completion")] = "x"
    await _drop_session_chat_providers()
    assert ("umo", "qq:GroupMessage:1", "provider_perf_chat_completion") in prefs.rows
