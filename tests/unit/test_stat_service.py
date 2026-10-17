import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

import astrbot.dashboard.services.stat_service as stat_service
from astrbot.core.computer import process_sandbox
from astrbot.core.computer.process_sandbox import SandboxRunResult, SandboxTimeoutError
from astrbot.dashboard.services.stat_service import StatService


@pytest.fixture(autouse=True)
def reset_sandbox_probe():
    """Keep cached sandbox capability checks independent across tests."""
    process_sandbox.get_process_sandbox_status.cache_clear()
    yield
    process_sandbox.get_process_sandbox_status.cache_clear()


def _make_service(db) -> StatService:
    """Build a StatService with a real DB and a mocked core lifecycle."""
    core_lifecycle = MagicMock()
    core_lifecycle.star_context.get_all_stars.return_value = []
    core_lifecycle.platform_manager.get_insts.return_value = []
    core_lifecycle.start_time = int(time.time()) - 100
    return StatService(db_helper=db, core_lifecycle=core_lifecycle, config={})


@pytest.mark.parametrize(
    ("system", "arch", "executable", "backend", "status"),
    [
        ("Linux", "x86_64", "/usr/bin/bwrap", "bubblewrap", "detected"),
        ("Linux", "aarch64", None, "bubblewrap", "missing"),
        ("Darwin", "arm64", "/usr/bin/sandbox-exec", "seatbelt", "detected"),
        ("Darwin", "x86_64", None, "seatbelt", "missing"),
        ("Darwin", "arm64", "/opt/bin/sandbox-exec", "seatbelt", "missing"),
        ("Windows", "AMD64", None, None, "unsupported"),
        ("Windows", "ARM64", None, None, "unsupported"),
        ("FreeBSD", "", None, None, "unsupported"),
    ],
)
def test_runtime_detects_platform_dependencies(
    monkeypatch, tmp_path, system, arch, executable, backend, status
):
    """Only report a detected sandbox after its launch probe succeeds."""
    monkeypatch.setattr(
        stat_service,
        "platform",
        SimpleNamespace(system=lambda: system, machine=lambda: arch),
    )
    monkeypatch.setattr(process_sandbox, "platform", stat_service.platform)
    which = Mock(return_value=executable)
    monkeypatch.setattr(process_sandbox, "shutil", SimpleNamespace(which=which))
    sandbox = Mock()
    sandbox.run.return_value = SandboxRunResult(returncode=0)
    factory = Mock(return_value=sandbox)
    monkeypatch.setattr(process_sandbox, "create_process_sandbox", factory)
    monkeypatch.setattr(process_sandbox, "get_astrbot_temp_path", lambda: str(tmp_path))

    service = _make_service(MagicMock())

    assert service.runtime == {
        "os": system.lower(),
        "arch": arch,
        "sandbox": {"backend": backend, "status": status},
    }
    assert process_sandbox.get_process_sandbox_status() == service.runtime["sandbox"]
    if system == "Linux":
        which.assert_called_once_with("bwrap")
    elif system == "Darwin":
        which.assert_called_once_with("sandbox-exec", path="/usr/bin")
    else:
        which.assert_not_called()
    if status == "detected":
        factory.assert_called_once_with()
        sandbox.run.assert_called_once()
        args, kwargs = sandbox.run.call_args
        assert args[0] == ["/bin/sh", "-c", ":"]
        assert args[1].filesystem_scope == "workspace"
        assert args[1].allow_network is False
        assert not args[1].workspace.exists()
        assert kwargs["timeout"] == 5
    else:
        factory.assert_not_called()


@pytest.mark.parametrize("system", ["Linux", "Darwin"])
@pytest.mark.parametrize(
    ("outcome", "error"),
    [
        (
            SandboxRunResult(
                returncode=1,
                stderr=b"bwrap: Creating new namespace failed: Operation not permitted\n",
            ),
            "bwrap: Creating new namespace failed: Operation not permitted",
        ),
        (
            SandboxRunResult(
                returncode=1, stderr=b"bwrap: setting up uid map: Permission denied\n"
            ),
            "bwrap: setting up uid map: Permission denied",
        ),
        (
            SandboxRunResult(
                returncode=1,
                stderr=b"bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
            ),
            "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted",
        ),
        (SandboxRunResult(returncode=2), "Sandbox probe exited with code 2."),
        (PermissionError("Operation not permitted"), "Operation not permitted"),
        (SandboxTimeoutError("Sandbox probe timed out."), "Sandbox probe timed out."),
        (SandboxTimeoutError(), "SandboxTimeoutError"),
        (TimeoutError("Sandbox probe timed out."), "Sandbox probe timed out."),
        (TimeoutError(), "TimeoutError"),
        (RuntimeError("Sandbox launch failed."), "Sandbox launch failed."),
    ],
)
def test_runtime_reports_sandbox_startup_failure(
    monkeypatch, tmp_path, system, outcome, error
):
    """Distinguish installed but unusable sandboxes without preventing startup."""
    monkeypatch.setattr(stat_service.platform, "system", lambda: system)
    monkeypatch.setattr(
        process_sandbox,
        "shutil",
        SimpleNamespace(which=lambda name, **kwargs: f"/usr/bin/{name}"),
    )
    sandbox = Mock()
    if isinstance(outcome, Exception):
        sandbox.run.side_effect = outcome
    else:
        sandbox.run.return_value = outcome
    monkeypatch.setattr(process_sandbox, "create_process_sandbox", lambda: sandbox)
    monkeypatch.setattr(process_sandbox, "get_astrbot_temp_path", lambda: str(tmp_path))

    service = _make_service(MagicMock())

    assert service.runtime["sandbox"] == {
        "backend": "bubblewrap" if system == "Linux" else "seatbelt",
        "status": "unavailable",
        "error": error,
    }
    assert process_sandbox.get_process_sandbox_status() == service.runtime["sandbox"]
    sandbox.run.assert_called_once()
    assert not list(tmp_path.iterdir())


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
