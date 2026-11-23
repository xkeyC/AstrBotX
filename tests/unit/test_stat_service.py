import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from astrbot.dashboard.services.stat_service import StatService


def _make_service(db) -> StatService:
    """Build a StatService with a real DB and a mocked core lifecycle."""
    core_lifecycle = MagicMock()
    core_lifecycle.star_context.get_all_stars.return_value = []
    core_lifecycle.platform_manager.get_insts.return_value = []
    core_lifecycle.start_time = int(time.time()) - 100
    return StatService(db_helper=db, core_lifecycle=core_lifecycle, config={})


@pytest.mark.asyncio
async def test_get_stat_aggregates_platform_stats(temp_db):
    """Seeded rows must aggregate into windowed platform sums and a global total."""
    now = datetime.now()
    seed = [
        ("aiocqhttp", 3, now - timedelta(hours=1)),
        ("aiocqhttp", 5, now - timedelta(hours=1, minutes=30)),
        ("qqofficial", 2, now - timedelta(hours=2)),
        ("webchat", 7, now - timedelta(minutes=10)),
        # Outside the 24h window: counted in the total but not in window stats.
        ("aiocqhttp", 4, now - timedelta(hours=26)),
    ]
    for platform_id, count, ts in seed:
        await temp_db.insert_platform_stats(platform_id, platform_id, count, ts)

    result = await _make_service(temp_db).get_stat(86400)

    # Global total counts every row, including the one outside the window.
    assert result["message_count"] == 21

    # Windowed per-platform sums, serialized with the legacy response keys.
    platform = {entry["name"]: entry["count"] for entry in result["platform"]}
    assert platform == {"aiocqhttp": 8, "qqofficial": 2, "webchat": 7}
    for entry in result["platform"]:
        assert set(entry) == {"name", "count", "timestamp"}

    # Hourly buckets cover [now - offset, now) in ascending order.
    series = result["message_time_series"]
    assert len(series) == 24
    bucket_ends = [bucket_end for bucket_end, _ in series]
    assert bucket_ends == sorted(bucket_ends)
    assert all(count >= 0 for _, count in series)
    # Rows within the current partial hour are not bucketed yet, so the
    # series sum never exceeds the windowed total of 17.
    assert sum(count for _, count in series) <= 17

    assert set(result) == {
        "platform",
        "message_count",
        "platform_count",
        "plugin_count",
        "plugins",
        "message_time_series",
        "running",
        "memory",
        "cpu_percent",
        "thread_count",
        "start_time",
    }


@pytest.mark.asyncio
async def test_get_stat_empty_window(temp_db):
    """A window with no rows yields empty platform stats but keeps the total."""
    old_ts = datetime.now() - timedelta(hours=2)
    await temp_db.insert_platform_stats("aiocqhttp", "aiocqhttp", 4, old_ts)

    result = await _make_service(temp_db).get_stat(1)

    assert result["platform"] == []
    assert result["message_count"] == 4
    assert all(count == 0 for _, count in result["message_time_series"])


@pytest.mark.asyncio
async def test_provider_token_ranking_includes_umo_display_names(temp_db):
    """UMO token rankings should prefer aliases and fall back to raw identifiers."""
    aliased_umo = "qq:GroupMessage:group-1"
    raw_umo = "webchat:FriendMessage:session-2"
    await temp_db.insert_provider_stat(
        umo=aliased_umo,
        provider_id="provider-1",
        stats={"token_usage": {"input_other": 3, "input_cached": 4, "output": 5}},
        agent_type="codex",
    )
    await temp_db.insert_provider_stat(
        umo=raw_umo,
        provider_id="provider-1",
        stats={"token_usage": {"input_other": 1, "input_cached": 1, "output": 1}},
        agent_type="codex",
    )
    await temp_db.upsert_umo_alias(
        umo=aliased_umo,
        creator_sender_id="creator-1",
        auto_name="研发群",
        user_alias="产品讨论群",
    )

    service = _make_service(temp_db)
    service.config = {
        "platform": [{"id": "qq", "type": "qq_official"}],
    }
    result = await service.get_provider_token_stats(1)

    assert result["range_by_umo"] == [
        {
            "umo": aliased_umo,
            "display_name": "产品讨论群",
            "platform_type": "qq_official",
            "tokens": 12,
        },
        {
            "umo": raw_umo,
            "display_name": raw_umo,
            "platform_type": "webchat",
            "tokens": 3,
        },
    ]


@pytest.mark.asyncio
async def test_provider_token_stats_read_codex_runs(temp_db):
    """The page shows Codex runs, ranked by model, with the cache hit rate."""
    now = time.time()
    for model, other, cached, ttft in (
        ("gpt-5.5", 200, 800, 1.5),
        ("gpt-5.5", 100, 900, 0.5),
        ("gpt-5.4-mini", 500, 500, 0.0),
    ):
        await temp_db.insert_provider_stat(
            umo="qq:GroupMessage:g",
            provider_id="openai",
            provider_model=model,
            stats={
                "token_usage": {
                    "input_other": other,
                    "input_cached": cached,
                    "output": 50,
                },
                "start_time": now - 10,
                "end_time": now,
                "time_to_first_token": ttft,
            },
            agent_type="codex",
        )
    # A row from the replaced internal runner is not part of the page.
    await temp_db.insert_provider_stat(
        umo="qq:GroupMessage:g",
        provider_id="legacy",
        stats={"token_usage": {"input_other": 9999, "input_cached": 0, "output": 0}},
    )

    result = await _make_service(temp_db).get_provider_token_stats(1)

    assert result["range_total_calls"] == 3
    assert result["range_total_tokens"] == 3150
    assert result["range_input_tokens"] == 3000
    assert result["range_cached_input_tokens"] == 2200
    assert result["range_cache_hit_rate"] == pytest.approx(2200 / 3000)
    # Only runs that measured it count toward the average.
    assert result["range_avg_ttft_ms"] == pytest.approx(1000)
    assert result["range_avg_duration_ms"] == pytest.approx(10000, rel=0.01)
    assert [r["provider_id"] for r in result["range_by_provider"]] == [
        "gpt-5.5",
        "gpt-5.4-mini",
    ]
    assert [s["name"] for s in result["trend"]["series"]] == [
        "gpt-5.5",
        "gpt-5.4-mini",
    ]


@pytest.mark.asyncio
async def test_voice_threads_show_their_model_apart(temp_db):
    """A voice thread's rows count, under its model marked as the voice's."""
    for agent_type in ("codex", "codex_voice"):
        await temp_db.insert_provider_stat(
            umo="qq:GroupMessage:g",
            provider_id="deepseek",
            provider_model="deepseek-v4.1-flash",
            stats={
                "token_usage": {"input_other": 100, "input_cached": 0, "output": 10}
            },
            agent_type=agent_type,
        )

    result = await _make_service(temp_db).get_provider_token_stats(1)

    assert result["range_total_calls"] == 2
    assert sorted(r["provider_id"] for r in result["range_by_provider"]) == [
        "deepseek-v4.1-flash",
        "deepseek-v4.1-flash (voice)",
    ]
    assert sorted(r["provider_model"] for r in result["today_by_model"]) == [
        "deepseek-v4.1-flash",
        "deepseek-v4.1-flash (voice)",
    ]
    assert [r["tokens"] for r in result["range_by_umo"]] == [220]
