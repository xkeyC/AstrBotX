"""Tests for ShipyardNeoBooter — readiness gate, shutdown cleanup, and rebuild recovery."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════
# _wait_until_ready
# ═══════════════════════════════════════════════════════════════


def _make_sandbox_mock(statuses: list[str], *, delete_side_effect=None):
    """Build a sandbox mock that returns *statuses* in order on refresh().

    After the list is exhausted subsequent refresh() calls return the last status.
    """
    call_count = 0

    async def _refresh():
        nonlocal call_count
        idx = min(call_count, len(statuses) - 1)
        call_count += 1
        s = statuses[idx]
        sandbox.status = SimpleNamespace(value=s)

    sandbox = SimpleNamespace(
        id="sandbox-test-1",
        profile="python-default",
        status=SimpleNamespace(value=statuses[0]),
        refresh=_refresh,
        delete=AsyncMock(side_effect=delete_side_effect),
    )
    return sandbox


class TestWaitUntilReady:
    def _make_booter(self):
        from astrbot.core.computer.booters.shipyard_neo import ShipyardNeoBooter

        return ShipyardNeoBooter(
            endpoint_url="http://localhost:8114",
            access_token="sk-bay-test",
        )

    @pytest.mark.asyncio
    async def test_already_ready_returns_immediately(self):
        """Sandbox is READY on first poll → instant return (warm hit)."""
        booter = self._make_booter()
        sandbox = _make_sandbox_mock(["ready"])

        await booter._wait_until_ready(sandbox)

        sandbox.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_starting_then_ready(self):
        """Sandbox transitions STARTING → READY within timeout."""
        booter = self._make_booter()
        sandbox = _make_sandbox_mock(["starting", "starting", "ready"])

        await booter._wait_until_ready(sandbox)

        sandbox.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_deletes_and_raises(self):
        """Sandbox reaches FAILED → delete called → RuntimeError raised."""
        booter = self._make_booter()
        sandbox = _make_sandbox_mock(["starting", "failed"])

        with pytest.raises(RuntimeError, match="terminal state"):
            await booter._wait_until_ready(sandbox)

        sandbox.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_expired_deletes_and_raises(self):
        """Sandbox reaches EXPIRED → delete called → RuntimeError raised."""
        booter = self._make_booter()
        sandbox = _make_sandbox_mock(["starting", "expired"])

        with pytest.raises(RuntimeError, match="terminal state"):
            await booter._wait_until_ready(sandbox)

        sandbox.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_timeout_deletes_and_raises(self):
        """Sandbox never reaches READY → delete called → TimeoutError raised."""
        booter = self._make_booter()
        # Return 'idle' every time to simulate a stuck sandbox
        sandbox = _make_sandbox_mock(["idle"])

        # Override the deadline so we don't actually sleep 180s
        original_time = asyncio.get_running_loop().time

        call_idx = 0

        def _fake_time():
            nonlocal call_idx
            # After one tick, jump past the deadline
            if call_idx == 0:
                call_idx += 1
                return original_time()
            # Return a value beyond the 180s timeout
            return original_time() + 200

        with patch(
            "astrbot.core.computer.booters.shipyard_neo.asyncio.get_running_loop"
        ) as mock_loop:
            mock_loop.return_value.time = _fake_time

            with pytest.raises(TimeoutError, match="did not become ready"):
                await booter._wait_until_ready(sandbox)

        sandbox.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_failure_during_cleanup_is_safe(self):
        """If sandbox.delete() itself throws, the original error is still raised."""
        booter = self._make_booter()
        sandbox = _make_sandbox_mock(
            ["failed"],
            delete_side_effect=RuntimeError("Bay unreachable"),
        )

        with pytest.raises(RuntimeError, match="terminal state"):
            await booter._wait_until_ready(sandbox)

        sandbox.delete.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════
# shutdown
# ═══════════════════════════════════════════════════════════════


class TestShutdown:
    def _make_booter(self):
        from astrbot.core.computer.booters.shipyard_neo import ShipyardNeoBooter

        return ShipyardNeoBooter(
            endpoint_url="http://localhost:8114",
            access_token="sk-bay-test",
        )

    @pytest.mark.asyncio
    async def test_delete_sandbox_true_calls_delete(self):
        """delete_sandbox=True → sandbox.delete() called, then client closed."""
        booter = self._make_booter()
        sandbox = SimpleNamespace(
            id="sandbox-test-1",
            delete=AsyncMock(),
        )
        client = SimpleNamespace(
            __aexit__=AsyncMock(),
        )
        booter._sandbox = sandbox  # type: ignore[assignment]
        booter._client = client  # type: ignore[assignment]

        await booter.shutdown(delete_sandbox=True)

        sandbox.delete.assert_awaited_once()
        client.__aexit__.assert_awaited_once()
        assert booter._client is None
        assert booter._sandbox is None

    @pytest.mark.asyncio
    async def test_delete_sandbox_false_does_not_call_delete(self):
        """delete_sandbox=False (default) → sandbox.delete() NOT called."""
        booter = self._make_booter()
        sandbox = SimpleNamespace(
            id="sandbox-test-1",
            delete=AsyncMock(),
        )
        client = SimpleNamespace(
            __aexit__=AsyncMock(),
        )
        booter._sandbox = sandbox  # type: ignore[assignment]
        booter._client = client  # type: ignore[assignment]

        await booter.shutdown()  # default delete_sandbox=False

        sandbox.delete.assert_not_called()
        client.__aexit__.assert_awaited_once()
        assert booter._client is None
        assert booter._sandbox is None

    @pytest.mark.asyncio
    async def test_delete_failure_still_closes_client(self):
        """If sandbox.delete() throws, HTTP client is still torn down."""
        booter = self._make_booter()
        sandbox = SimpleNamespace(
            id="sandbox-test-1",
            delete=AsyncMock(side_effect=RuntimeError("Bay gone")),
        )
        client = SimpleNamespace(
            __aexit__=AsyncMock(),
        )
        booter._sandbox = sandbox  # type: ignore[assignment]
        booter._client = client  # type: ignore[assignment]

        # Should not raise — delete failure is logged but swallowed
        await booter.shutdown(delete_sandbox=True)

        sandbox.delete.assert_awaited_once()
        client.__aexit__.assert_awaited_once()
        assert booter._client is None
        assert booter._sandbox is None

    @pytest.mark.asyncio
    async def test_no_client_is_noop(self):
        """shutdown() on an uninitialised booter is a no-op."""
        booter = self._make_booter()
        # _client is None by default
        await booter.shutdown(delete_sandbox=True)
        # No exception → ok


# ═══════════════════════════════════════════════════════════════
# get_booter rebuild path
# ═══════════════════════════════════════════════════════════════


class TestGetBooterRebuild:
    """Verify that stale ShipyardNeoBooter instances are cleaned up on rebuild."""

    def _make_fake_context(self, booter_type: str = "shipyard_neo"):
        """Build a context-like object for get_booter()."""
        _cfg = {
            "provider_settings": {
                "computer_use_runtime": "sandbox",
                "sandbox": {
                    "booter": booter_type,
                    "shipyard_neo_endpoint": "http://bay:8114",
                    "shipyard_neo_access_token": "sk-test",
                    "shipyard_neo_ttl": 3600,
                    "shipyard_neo_profile": "python-default",
                },
            }
        }
        return SimpleNamespace(
            get_config=lambda umo=None: _cfg,
        )

    @pytest.mark.asyncio
    async def test_stale_neo_booter_calls_shutdown_with_delete(self, monkeypatch):
        """A stale ShipyardNeoBooter gets shutdown(delete_sandbox=True) on eviction."""
        from astrbot.core.computer import computer_client
        from astrbot.core.computer.booters.shipyard_neo import ShipyardNeoBooter

        ctx = self._make_fake_context()

        stale = ShipyardNeoBooter(
            endpoint_url="http://bay:8114", access_token="sk-test"
        )
        stale._sandbox = SimpleNamespace(id="stale-sandbox")  # type: ignore[assignment]
        stale._client = SimpleNamespace(__aexit__=AsyncMock())  # type: ignore[assignment]
        stale._sandbox.refresh = AsyncMock(side_effect=RuntimeError("sandbox gone"))  # type: ignore[union-attr]
        # available() will return False because refresh() throws
        stale.shutdown = AsyncMock()

        monkeypatch.setitem(computer_client.session_booter, "session-1", stale)

        from astrbot.core.computer.computer_client import get_booter

        # get_booter should evict stale and rebuild.
        # We need to mock the entire rebuild path so it doesn't actually
        # try to connect to Bay.
        async def _fake_boot(_self, _sid):
            _self._sandbox = SimpleNamespace(  # type: ignore[assignment]
                id="new-sandbox",
                refresh=AsyncMock(),
                status=SimpleNamespace(value="ready"),
                capabilities=["python", "shell", "filesystem"],
            )
            _self._client = SimpleNamespace()  # type: ignore[assignment]
            _self._shell = SimpleNamespace()  # type: ignore[assignment]
            _self._fs = SimpleNamespace()  # type: ignore[assignment]
            _self._python = SimpleNamespace()  # type: ignore[assignment]

        with (
            patch.object(ShipyardNeoBooter, "boot", _fake_boot),
            patch(
                "astrbot.core.computer.computer_client._sync_skills_to_sandbox",
                AsyncMock(),
            ),
        ):
            await get_booter(ctx, "session-1")

        stale.shutdown.assert_awaited_once_with(delete_sandbox=True)
        # Old entry should be replaced
        new_booter = computer_client.session_booter.get("session-1")
        assert new_booter is not None
        assert new_booter is not stale

    @pytest.mark.asyncio
    async def test_stale_non_neo_booter_calls_plain_shutdown(self, monkeypatch):
        """Non-neo booter (e.g. shipyard) → plain shutdown() without delete_sandbox."""
        from astrbot.core.computer import computer_client

        ctx = self._make_fake_context(booter_type="shipyard")

        stale = SimpleNamespace(shutdown=AsyncMock())
        stale.available = AsyncMock(return_value=False)

        monkeypatch.setitem(computer_client.session_booter, "session-1", stale)

        # Patch ShipyardBooter entirely to skip its __init__ validation
        class _FakeShipyardBooter:
            def __init__(self, **kwargs):
                pass

            async def boot(self, _sid):
                self._sandbox = SimpleNamespace(  # type: ignore[assignment]
                    refresh=AsyncMock(),
                    status=SimpleNamespace(value="ready"),
                )
                self._shell = SimpleNamespace()  # type: ignore[assignment]
                self._fs = SimpleNamespace()  # type: ignore[assignment]
                self._python = SimpleNamespace()  # type: ignore[assignment]

            async def shutdown(self, **kwargs):
                pass

        with (
            patch(
                "astrbot.core.computer.booters.shipyard.ShipyardBooter",
                _FakeShipyardBooter,
            ),
            patch(
                "astrbot.core.computer.computer_client._sync_skills_to_sandbox",
                AsyncMock(),
            ),
        ):
            from astrbot.core.computer.computer_client import get_booter

            await get_booter(ctx, "session-1")

        stale.shutdown.assert_awaited_once()
        # No delete_sandbox kwarg for non-neo booters
        call_kwargs = stale.shutdown.call_args.kwargs
        assert call_kwargs == {}


class TestSandboxReuse:
    def _booter(self, reuse_id: str, client):
        from astrbot.core.computer.booters.shipyard_neo import ShipyardNeoBooter

        booter = ShipyardNeoBooter(
            endpoint_url="http://localhost:8114",
            access_token="sk-bay-test",
            reuse_sandbox_id=reuse_id,
        )
        booter._client = client
        return booter

    @pytest.mark.asyncio
    async def test_reuses_live_sandbox(self):
        sandbox = _make_sandbox_mock(["idle"])
        client = SimpleNamespace(get_sandbox=AsyncMock(return_value=sandbox))
        assert await self._booter("sandbox-test-1", client)._reuse_sandbox() is sandbox

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["failed", "expired"])
    async def test_terminal_sandbox_is_not_reused(self, status):
        client = SimpleNamespace(
            get_sandbox=AsyncMock(return_value=_make_sandbox_mock([status]))
        )
        assert await self._booter("sandbox-test-1", client)._reuse_sandbox() is None

    @pytest.mark.asyncio
    async def test_missing_sandbox_or_no_binding(self):
        client = SimpleNamespace(get_sandbox=AsyncMock(side_effect=RuntimeError("404")))
        assert await self._booter("gone", client)._reuse_sandbox() is None
        assert await self._booter("", client)._reuse_sandbox() is None

    @pytest.mark.asyncio
    async def test_profile_mismatch_is_not_reused(self):
        sandbox = _make_sandbox_mock(["idle"])
        client = SimpleNamespace(get_sandbox=AsyncMock(return_value=sandbox))
        booter = self._booter("sandbox-test-1", client)
        assert await booter._reuse_sandbox("browser-python") is None
        assert await booter._reuse_sandbox("python-default") is sandbox

    @pytest.mark.asyncio
    async def test_failed_reused_sandbox_falls_back_to_new_one(self, monkeypatch):
        from astrbot.core.computer import computer_client
        from astrbot.core.computer.booters.shipyard_neo import ShipyardNeoBooter

        monkeypatch.delitem(computer_client.session_booter, "s-reuse", raising=False)
        ctx = TestGetBooterRebuild()._make_fake_context()

        async def _fake_boot(_self, _sid):
            sid = _self._reuse_sandbox_id or "fresh-sandbox"
            _self._sandbox = SimpleNamespace(id=sid)  # type: ignore[assignment]

        async def _sync(booter):
            if booter.sandbox_id == "old-sandbox":
                raise RuntimeError("session start failed")

        shutdown = AsyncMock()
        save = AsyncMock()
        with (
            patch.object(ShipyardNeoBooter, "boot", _fake_boot),
            patch.object(ShipyardNeoBooter, "shutdown", shutdown),
            patch(
                "astrbot.core.computer.computer_client._load_sandbox_binding",
                AsyncMock(return_value="old-sandbox"),
            ),
            patch("astrbot.core.computer.computer_client._save_sandbox_binding", save),
            patch(
                "astrbot.core.computer.computer_client._sync_skills_to_sandbox", _sync
            ),
        ):
            booter = await computer_client.get_booter(ctx, "s-reuse")

        assert booter.sandbox_id == "fresh-sandbox"
        # The bound sandbox is closed but not deleted.
        shutdown.assert_awaited_once_with()
        assert save.await_args.args[3] is booter
        computer_client.session_booter.pop("s-reuse", None)


class TestTtlRenewal:
    """Bay's TTL is a total lifetime; a sandbox in use is renewed to a full TTL
    from now, so only one nobody uses expires (with its workspace)."""

    def _booter(self, expires_in: float | None, ttl: int = 7 * 86400):
        from datetime import datetime, timedelta

        from astrbot.core.computer.booters.shipyard_neo import ShipyardNeoBooter

        booter = ShipyardNeoBooter(
            endpoint_url="http://localhost:8114", access_token="sk", ttl=ttl
        )
        # Bay returns naive UTC times.
        expires_at = (
            None
            if expires_in is None
            else datetime.utcnow() + timedelta(seconds=expires_in)
        )
        booter._sandbox = SimpleNamespace(
            id="sandbox-x", expires_at=expires_at, extend_ttl=AsyncMock()
        )
        return booter

    @pytest.mark.asyncio
    async def test_a_used_sandbox_is_renewed_to_a_full_ttl(self):
        booter = self._booter(expires_in=3 * 3600)
        await booter.renew_ttl()
        (extend_by,), _ = booter._sandbox.extend_ttl.call_args
        assert abs(extend_by - (7 * 86400 - 3 * 3600)) < 5
        # Within the hour it is not asked again.
        await booter.renew_ttl()
        booter._sandbox.extend_ttl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_fresh_or_ttl_less_sandbox_is_left_alone(self):
        for booter in (
            self._booter(expires_in=7 * 86400 - 60),
            self._booter(expires_in=None),
            self._booter(expires_in=3600, ttl=0),
            self._booter(expires_in=-10),
        ):
            await booter.renew_ttl()
            booter._sandbox.extend_ttl.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_renewal_is_tried_again_on_the_next_use(self):
        booter = self._booter(expires_in=86400)
        booter._sandbox.extend_ttl.side_effect = [RuntimeError("bay down"), None]
        await booter.renew_ttl()
        await booter.renew_ttl()
        assert booter._sandbox.extend_ttl.await_count == 2
