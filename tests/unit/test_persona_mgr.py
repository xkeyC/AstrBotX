import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text

import astrbot.api  # noqa: F401  # import first to avoid a circular import
from astrbot.core import persona_mgr as pm
from astrbot.core.db.po import Persona
from astrbot.core.db.sqlite import SQLiteDatabase
from astrbot.core.persona_mgr import PersonaManager


def make_acm(persona_id: str = "default") -> MagicMock:
    """Build a mocked AstrBotConfigManager with the given default persona."""
    conf = {
        "agent_runner": {
            "runner_type": "local",
            "config": {"persona": {"persona_id": persona_id}},
        },
        "provider_settings": {},
    }
    acm = MagicMock()
    acm.default_conf = conf
    acm.get_conf.return_value = conf
    return acm


def make_persona(persona_id: str, prompt: str) -> Persona:
    return Persona(
        persona_id=persona_id,
        system_prompt=prompt,
        begin_dialogs=[],
        tools=None,
        skills=None,
        custom_error_message=None,
        folder_id=None,
        sort_order=0,
    )


def make_manager(
    personas: list[Persona], persona_id: str = "default"
) -> pm.PersonaManager:
    manager = pm.PersonaManager(db_helper=MagicMock(), acm=make_acm(persona_id))
    manager.personas = personas
    manager.get_v3_persona_data()
    return manager


def make_seeding_db(personas: list[Persona]) -> MagicMock:
    db = MagicMock()
    db.get_personas = AsyncMock(return_value=personas)
    db.insert_persona = AsyncMock(
        side_effect=lambda **kwargs: make_persona(
            kwargs["persona_id"], kwargs["system_prompt"]
        )
    )
    return db


@pytest.mark.asyncio
async def test_initialize_seeds_default_when_absent():
    db = make_seeding_db([make_persona("alice", "ALICE")])
    manager = pm.PersonaManager(db_helper=db, acm=make_acm())

    await manager.initialize()

    db.insert_persona.assert_awaited_once()
    assert {persona.persona_id for persona in manager.personas} == {
        "alice",
        "default",
    }
    seeded = manager.get_persona_v3_by_id("default")
    assert seeded is not None
    assert seeded["prompt"] == pm.DEFAULT_PERSONALITY["prompt"]


@pytest.mark.asyncio
async def test_initialize_adopts_existing_default():
    db = make_seeding_db([make_persona("default", "USER DEFAULT")])
    manager = pm.PersonaManager(db_helper=db, acm=make_acm())

    await manager.initialize()

    db.insert_persona.assert_not_awaited()
    adopted = manager.get_persona_v3_by_id("default")
    assert adopted is not None
    assert adopted["prompt"] == "USER DEFAULT"


@pytest.mark.asyncio
async def test_delete_default_persona_is_rejected():
    db = make_seeding_db([make_persona("default", "USER DEFAULT")])
    manager = pm.PersonaManager(db_helper=db, acm=make_acm())
    await manager.initialize()

    with pytest.raises(ValueError):
        await manager.delete_persona("default")


@pytest.mark.asyncio
async def test_create_default_persona_is_rejected():
    manager = pm.PersonaManager(db_helper=MagicMock(), acm=make_acm())

    with pytest.raises(ValueError):
        await manager.create_persona("default", "PROMPT")


def test_get_persona_v3_by_id_prefers_persona_named_default():
    manager = make_manager([make_persona("default", "USER DEFAULT")])

    resolved = manager.get_persona_v3_by_id("default")

    assert resolved is not None
    assert resolved["prompt"] == "USER DEFAULT"


def test_get_persona_v3_by_id_falls_back_to_builtin_default():
    manager = make_manager([make_persona("alice", "ALICE")])

    assert manager.get_persona_v3_by_id("default") == pm.DEFAULT_PERSONALITY


@pytest.mark.asyncio
async def test_get_default_persona_v3_uses_db_default():
    manager = make_manager([make_persona("default", "USER DEFAULT")])

    resolved = await manager.get_default_persona_v3(umo="test:umo")

    assert resolved["prompt"] == "USER DEFAULT"


@pytest.mark.asyncio
async def test_resolve_selected_persona_matches_db_default():
    manager = make_manager([make_persona("default", "USER DEFAULT")])

    with patch.object(pm.sp, "get_async", new=AsyncMock(return_value={})):
        persona_id, persona, _, _ = await manager.resolve_selected_persona(
            umo="test:umo",
            conversation_persona_id="default",
            platform_name="webchat",
            provider_settings={},
        )

    assert persona_id == "default"
    assert persona is not None
    assert persona["prompt"] == "USER DEFAULT"


@pytest.mark.asyncio
async def test_resolve_selected_persona_webchat_implicit_uses_chatui_default():
    manager = make_manager([make_persona("default", "USER DEFAULT")])

    with patch.object(pm.sp, "get_async", new=AsyncMock(return_value={})):
        (
            persona_id,
            persona,
            _,
            use_webchat_special_default,
        ) = await manager.resolve_selected_persona(
            umo="test:umo",
            conversation_persona_id=None,
            platform_name="webchat",
            provider_settings={},
        )

    assert persona_id == "_chatui_default_"
    assert persona is None
    assert use_webchat_special_default is True


@pytest.mark.asyncio
async def test_resolve_selected_persona_non_webchat_implicit_uses_db_default():
    manager = make_manager([make_persona("default", "USER DEFAULT")])

    with patch.object(pm.sp, "get_async", new=AsyncMock(return_value={})):
        (
            persona_id,
            persona,
            _,
            use_webchat_special_default,
        ) = await manager.resolve_selected_persona(
            umo="test:umo",
            conversation_persona_id=None,
            platform_name="telegram",
            provider_settings={},
        )

    assert persona_id == "default"
    assert persona is not None
    assert persona["prompt"] == "USER DEFAULT"
    assert use_webchat_special_default is False


@pytest.mark.asyncio
async def test_resolve_selected_persona_explicit_default_ignores_chatui_default():
    manager = make_manager([make_persona("default", "USER DEFAULT")])

    with patch.object(pm.sp, "get_async", new=AsyncMock(return_value={})):
        (
            persona_id,
            persona,
            _,
            use_webchat_special_default,
        ) = await manager.resolve_selected_persona(
            umo="test:umo",
            conversation_persona_id="default",
            platform_name="webchat",
            provider_settings={},
        )

    assert persona_id == "default"
    assert persona is not None
    assert persona["prompt"] == "USER DEFAULT"
    assert use_webchat_special_default is False


@pytest.mark.asyncio
async def test_resolve_selected_persona_session_rule_default_uses_db_default():
    manager = make_manager([make_persona("default", "USER DEFAULT")])

    with patch.object(
        pm.sp, "get_async", new=AsyncMock(return_value={"persona_id": "default"})
    ):
        (
            persona_id,
            persona,
            force_applied_persona_id,
            use_webchat_special_default,
        ) = await manager.resolve_selected_persona(
            umo="test:umo",
            conversation_persona_id=None,
            platform_name="webchat",
            provider_settings={},
        )

    assert persona_id == "default"
    assert persona is not None
    assert persona["prompt"] == "USER DEFAULT"
    assert force_applied_persona_id == "default"
    assert use_webchat_special_default is False


@pytest.mark.asyncio
async def test_resolve_selected_persona_prefers_forced_session_persona():
    acm = MagicMock()
    acm.default_conf = {"provider_settings": {}}
    manager = PersonaManager(MagicMock(), acm)
    manager.personas_v3 = [
        {"name": "event-persona", "prompt": "event"},
        {"name": "session-persona", "prompt": "session"},
    ]

    with patch(
        "astrbot.core.persona_mgr.sp.get_async",
        AsyncMock(return_value={"persona_id": "session-persona"}),
    ):
        (
            persona_id,
            persona,
            force_applied_persona_id,
            use_webchat_default,
        ) = await manager.resolve_selected_persona(
            umo="platform:FriendMessage:user",
            conversation_persona_id="conversation-persona",
            platform_name="aiocqhttp",
            provider_settings={"default_personality": "default-persona"},
            selected_persona_id="event-persona",
        )

    assert persona_id == "session-persona"
    assert persona == {"name": "session-persona", "prompt": "session"}
    assert force_applied_persona_id == "session-persona"
    assert use_webchat_default is False


@pytest.mark.asyncio
async def test_persona_voice_prompt_migration_and_round_trip(tmp_path):
    """A legacy personas table gains voice_prompt, and the manager round-trips it."""
    db_path = tmp_path / "legacy-personas.db"
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            "CREATE TABLE personas ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "persona_id VARCHAR(255) NOT NULL UNIQUE, "
            "system_prompt TEXT NOT NULL, "
            "begin_dialogs JSON, "
            "tools JSON, "
            "created_at DATETIME, "
            "updated_at DATETIME)"
        )
        conn.execute(
            "INSERT INTO personas (persona_id, system_prompt, begin_dialogs) "
            "VALUES ('legacy', 'Legacy persona.', '[]')"
        )
    conn.close()

    db = SQLiteDatabase(str(db_path))
    try:
        await db.initialize()
        async with db.engine.connect() as conn:
            result = await conn.execute(text("PRAGMA table_info(personas)"))
            columns = {row[1] for row in result.fetchall()}
        assert {"custom_error_message", "voice_prompt"} <= columns
        legacy = await db.get_persona_by_id("legacy")
        assert legacy.voice_prompt is None

        acm = MagicMock()
        acm.default_conf = {}
        manager = PersonaManager(db, acm)
        await manager.initialize()

        created = await manager.create_persona(
            persona_id="voice",
            system_prompt="Full persona.",
            voice_prompt="Speak softly.",
        )
        assert created.voice_prompt == "Speak softly."
        assert (await db.get_persona_by_id("voice")).voice_prompt == "Speak softly."
        v3 = manager.get_persona_v3_by_id("voice")
        assert v3 is not None
        assert v3["voice_prompt"] == "Speak softly."

        clone = await manager.clone_persona("voice", "voice-clone")
        assert clone.voice_prompt == "Speak softly."

        # Updating other fields leaves voice_prompt unchanged.
        await manager.update_persona("voice", system_prompt="Updated persona.")
        assert (await db.get_persona_by_id("voice")).voice_prompt == "Speak softly."

        await manager.update_persona("voice", voice_prompt="Be brief.")
        assert (await db.get_persona_by_id("voice")).voice_prompt == "Be brief."

        await manager.update_persona("voice", voice_prompt=None)
        assert (await db.get_persona_by_id("voice")).voice_prompt is None
    finally:
        await db.engine.dispose()
