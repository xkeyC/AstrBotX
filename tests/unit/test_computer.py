"""Tests for astrbot/core/computer module.

This module tests the ComputerClient and the sandbox booter implementations
(shipyard, boxlite) and the skill sync to sandboxes.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from astrbot.core.computer.booters.base import ComputerBooter


class TestComputerBooterBase:
    """Tests for ComputerBooter base class interface."""

    def test_base_class_is_protocol(self):
        """Test ComputerBooter has expected interface."""
        booter = ComputerBooter
        assert hasattr(booter, "fs")
        assert hasattr(booter, "python")
        assert hasattr(booter, "shell")
        assert hasattr(booter, "boot")
        assert hasattr(booter, "shutdown")
        assert hasattr(booter, "upload_file")
        assert hasattr(booter, "download_file")
        assert hasattr(booter, "available")


class TestShipyardBooter:
    """Tests for ShipyardBooter."""

    @pytest.mark.asyncio
    async def test_shipyard_shell_unwraps_bay_exec_response(self):
        """Bay exec responses keep shell output under the data field."""
        from astrbot.core.computer.booters.shipyard import ShipyardShellWrapper

        raw_shell = MagicMock()
        raw_shell.exec = AsyncMock(
            return_value={
                "success": True,
                "data": {
                    "success": True,
                    "return_code": 0,
                    "stdout": "hello_from_shipyard_bay\n/workspace\n",
                    "stderr": "",
                    "pid": 16,
                    "process_id": None,
                    "error": None,
                },
                "error": None,
            },
        )

        result = await ShipyardShellWrapper(raw_shell).exec(
            "echo hello_from_shipyard_bay && pwd"
        )

        assert result["stdout"] == "hello_from_shipyard_bay\n/workspace\n"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_shipyard_shell_keeps_flat_exec_response_compatible(self):
        """Flat shell responses remain compatible with ShipyardShellWrapper."""
        from astrbot.core.computer.booters.shipyard import ShipyardShellWrapper

        raw_shell = MagicMock()
        raw_shell.exec = AsyncMock(
            return_value={
                "success": True,
                "stdout": "hello_from_legacy_shell\n/workspace\n",
                "stderr": "",
                "exit_code": 0,
                "execution_id": "exec-legacy",
                "execution_time_ms": 45,
                "command": "echo hello_from_legacy_shell && pwd",
            },
        )

        result = await ShipyardShellWrapper(raw_shell).exec(
            "echo hello_from_legacy_shell && pwd"
        )

        assert result == {
            "stdout": "hello_from_legacy_shell\n/workspace\n",
            "stderr": "",
            "exit_code": 0,
            "success": True,
            "execution_id": "exec-legacy",
            "execution_time_ms": 45,
            "command": "echo hello_from_legacy_shell && pwd",
        }

    @pytest.mark.asyncio
    async def test_shipyard_shell_maps_nonzero_return_code(self):
        """A non-zero Bay return_code is exposed as exit_code and failure."""
        from astrbot.core.computer.booters.shipyard import ShipyardShellWrapper

        raw_shell = MagicMock()
        raw_shell.exec = AsyncMock(
            return_value={
                "success": True,
                "data": {
                    "success": False,
                    "return_code": 3,
                    "stdout": "",
                    "stderr": "oops",
                },
                "error": None,
            },
        )

        result = await ShipyardShellWrapper(raw_shell).exec("echo oops >&2; exit 3")

        assert result["stdout"] == ""
        assert result["stderr"] == "oops"
        assert result["exit_code"] == 3
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_shipyard_shell_failure_without_success_field(self):
        """A nonzero return code without `success` is reported as a failure."""
        from astrbot.core.computer.booters.shipyard import ShipyardShellWrapper

        raw_shell = MagicMock()
        raw_shell.exec = AsyncMock(
            return_value={"data": {"return_code": 1, "stdout": "", "stderr": ""}}
        )

        result = await ShipyardShellWrapper(raw_shell).exec("false")

        assert result["exit_code"] == 1
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_shipyard_shell_keeps_envelope_error(self):
        """A nested `data.error = None` must not hide the envelope error."""
        from astrbot.core.computer.booters.shipyard import ShipyardShellWrapper

        raw_shell = MagicMock()
        raw_shell.exec = AsyncMock(
            return_value={
                "success": False,
                "data": {
                    "success": False,
                    "return_code": 1,
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                },
                "error": "Ship returned 500: exec failed",
            },
        )

        result = await ShipyardShellWrapper(raw_shell).exec("false")

        assert result["stderr"] == "Ship returned 500: exec failed"
        assert result["exit_code"] == 1
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_shipyard_file_tools_unwrap_bay_exec_response(self):
        """Bay file responses keep their payload under the data field too."""
        from astrbot.core.computer.booters.shipyard import (
            ShipyardFileSystemWrapper,
        )

        raw_fs = MagicMock()
        raw_fs.read_file = AsyncMock(
            return_value={
                "success": True,
                "data": {"content": "hello-file", "path": "probe.txt", "size": 10},
                "error": None,
            },
        )
        raw_fs.list_dir = AsyncMock(
            return_value={
                "success": True,
                "data": {
                    "files": [{"name": "probe.txt", "is_file": True}],
                    "current_path": "/workspace",
                },
                "error": None,
            },
        )

        wrapper = ShipyardFileSystemWrapper(raw_fs, MagicMock())

        read_result = await wrapper.read_file("probe.txt")
        assert read_result["content"] == "hello-file"
        assert "data" not in read_result

        list_result = await wrapper.list_dir(".")
        assert list_result["files"] == [{"name": "probe.txt", "is_file": True}]
        assert list_result["current_path"] == "/workspace"
        assert "data" not in list_result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "kwargs", "inner"),
        [
            (
                "create_file",
                {"path": "probe.txt", "content": "hello-file"},
                {"success": True, "path": "probe.txt", "size": 10},
            ),
            (
                "write_file",
                {"path": "probe.txt", "content": "hello-file"},
                {"success": True, "path": "probe.txt", "bytes_written": 10},
            ),
            (
                "delete_file",
                {"path": "probe.txt"},
                {"success": True, "path": "probe.txt"},
            ),
            (
                "edit_file",
                {"path": "probe.txt", "old_string": "a", "new_string": "b"},
                {"success": True, "replacements": 1},
            ),
        ],
    )
    async def test_shipyard_remaining_file_tools_unwrap_bay_exec_response(
        self, method, kwargs, inner
    ):
        """Every fs wrapper flattens the envelope, not only read_file/list_dir."""
        from astrbot.core.computer.booters.shipyard import (
            ShipyardFileSystemWrapper,
        )

        raw_fs = MagicMock()
        setattr(
            raw_fs,
            method,
            AsyncMock(return_value={"success": True, "data": inner, "error": None}),
        )

        wrapper = ShipyardFileSystemWrapper(raw_fs, MagicMock())
        result = await getattr(wrapper, method)(**kwargs)

        for key, value in inner.items():
            assert result[key] == value
        assert "data" not in result

    @pytest.mark.asyncio
    async def test_shipyard_shell_envelope_failure_wins_over_data_success(self):
        """A failed request is not reported as a success by data.success."""
        from astrbot.core.computer.booters.shipyard import ShipyardShellWrapper

        raw_shell = MagicMock()
        raw_shell.exec = AsyncMock(
            return_value={
                "success": False,
                "data": {
                    "success": True,
                    "return_code": 0,
                    "stdout": "stale-body",
                    "stderr": "",
                    "error": None,
                },
                "error": "Ship returned 500: exec failed",
            },
        )

        result = await ShipyardShellWrapper(raw_shell).exec("echo hi")

        assert result["success"] is False
        assert result["stderr"] == "Ship returned 500: exec failed"

    @pytest.mark.asyncio
    async def test_shipyard_shell_envelope_failure_without_data(self):
        """A request-level failure carrying no `data` is still a failure."""
        from astrbot.core.computer.booters.shipyard import ShipyardShellWrapper

        raw_shell = MagicMock()
        raw_shell.exec = AsyncMock(
            return_value={
                "success": False,
                "data": None,
                "error": "Ship returned 500: exec failed",
            },
        )

        result = await ShipyardShellWrapper(raw_shell).exec("echo hi")

        assert result["stdout"] == ""
        assert result["stderr"] == "Ship returned 500: exec failed"
        assert result["exit_code"] is None
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_shipyard_booter_init(self):
        """Test ShipyardBooter initialization."""
        with patch("astrbot.core.computer.booters.shipyard.ShipyardClient"):
            from astrbot.core.computer.booters.shipyard import ShipyardBooter

            booter = ShipyardBooter(
                endpoint_url="http://localhost:8080",
                access_token="test_token",
                ttl=3600,
                session_num=10,
            )
            assert booter._ttl == 3600
            assert booter._session_num == 10

    @pytest.mark.asyncio
    async def test_shipyard_booter_boot(self):
        """Test ShipyardBooter boot method."""
        mock_ship = MagicMock()
        mock_ship.id = "test-ship-id"
        mock_ship.fs = MagicMock()
        mock_ship.python = MagicMock()
        mock_ship.shell = MagicMock()

        mock_client = MagicMock()
        mock_client.create_ship = AsyncMock(return_value=mock_ship)

        with patch(
            "astrbot.core.computer.booters.shipyard.ShipyardClient",
            return_value=mock_client,
        ):
            from astrbot.core.computer.booters.shipyard import ShipyardBooter

            booter = ShipyardBooter(
                endpoint_url="http://localhost:8080",
                access_token="test_token",
            )
            await booter.boot("test-session")
            assert booter._ship == mock_ship

    @pytest.mark.asyncio
    async def test_shipyard_available_healthy(self):
        """Test ShipyardBooter available when healthy."""
        mock_ship = MagicMock()
        mock_ship.id = "test-ship-id"

        mock_client = MagicMock()
        mock_client.get_ship = AsyncMock(return_value={"status": 1})

        with patch(
            "astrbot.core.computer.booters.shipyard.ShipyardClient",
            return_value=mock_client,
        ):
            from astrbot.core.computer.booters.shipyard import ShipyardBooter

            booter = ShipyardBooter(
                endpoint_url="http://localhost:8080",
                access_token="test_token",
            )
            booter._ship = mock_ship
            booter._sandbox_client = mock_client

            result = await booter.available()
            assert result is True

    @pytest.mark.asyncio
    async def test_shipyard_available_unhealthy(self):
        """Test ShipyardBooter available when unhealthy."""
        mock_ship = MagicMock()
        mock_ship.id = "test-ship-id"

        mock_client = MagicMock()
        mock_client.get_ship = AsyncMock(return_value={"status": 0})

        with patch(
            "astrbot.core.computer.booters.shipyard.ShipyardClient",
            return_value=mock_client,
        ):
            from astrbot.core.computer.booters.shipyard import ShipyardBooter

            booter = ShipyardBooter(
                endpoint_url="http://localhost:8080",
                access_token="test_token",
            )
            booter._ship = mock_ship
            booter._sandbox_client = mock_client

            result = await booter.available()
            assert result is False


class TestBoxliteBooter:
    """Tests for BoxliteBooter."""

    @pytest.mark.asyncio
    async def test_boxlite_booter_init(self):
        """Test BoxliteBooter can be instantiated via __new__."""
        # Need to mock boxlite module before importing
        mock_boxlite = MagicMock()
        mock_boxlite.SimpleBox = MagicMock()

        with patch.dict(sys.modules, {"boxlite": mock_boxlite}):
            from astrbot.core.computer.booters.boxlite import BoxliteBooter

            # Just verify class exists and can be instantiated (boot is async)
            booter = BoxliteBooter.__new__(BoxliteBooter)
            assert booter is not None


class TestComputerClient:
    """Tests for computer_client module functions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("runtime", [None, "none", "local"])
    async def test_get_booter_requires_explicit_runtime(self, runtime):
        """Only the sandbox runtime starts or reuses a computer environment.

        A stale ``local`` value of the removed local runtime must not fall
        through to the sandbox.
        """
        from astrbot.core.computer import computer_client

        provider_settings = {} if runtime is None else {"computer_use_runtime": runtime}
        context = MagicMock()
        context.get_config.return_value = {"provider_settings": provider_settings}
        existing = MagicMock()
        with patch.object(computer_client, "session_booter", {"session": existing}):
            with pytest.raises(RuntimeError, match="disabled by configuration"):
                await computer_client.get_booter(context, "session")
        existing.available.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_booter_shipyard(self):
        """Test get_booter with shipyard type."""
        from astrbot.core.computer import computer_client
        from astrbot.core.computer.booters.shipyard import ShipyardBooter

        # Clear session booter
        computer_client.session_booter.clear()

        mock_context = MagicMock()
        mock_config = MagicMock()
        mock_config.get = lambda key, default=None: {
            "provider_settings": {
                "computer_use_runtime": "sandbox",
                "sandbox": {
                    "booter": "shipyard",
                    "shipyard_endpoint": "http://localhost:8080",
                    "shipyard_access_token": "test_token",
                    "shipyard_ttl": 3600,
                    "shipyard_max_sessions": 10,
                },
            }
        }.get(key, default)
        mock_context.get_config = MagicMock(return_value=mock_config)

        # Mock the ShipyardBooter
        mock_ship = MagicMock()
        mock_ship.id = "test-ship-id"
        mock_ship.fs = MagicMock()
        mock_ship.python = MagicMock()
        mock_ship.shell = MagicMock()

        mock_booter = MagicMock()
        mock_booter.boot = AsyncMock()
        mock_booter.available = AsyncMock(return_value=True)
        mock_booter.shell = MagicMock()
        mock_booter.upload_file = AsyncMock(return_value={"success": True})

        with (
            patch.object(ShipyardBooter, "boot", new=AsyncMock()),
            patch(
                "astrbot.core.computer.computer_client._sync_skills_to_sandbox",
                AsyncMock(),
            ),
        ):
            # Directly set the booter in the session
            computer_client.session_booter["test-session-id"] = mock_booter

            booter = await computer_client.get_booter(mock_context, "test-session-id")
            assert booter is mock_booter

        # Cleanup
        computer_client.session_booter.clear()

    @pytest.mark.asyncio
    async def test_get_booter_unknown_type(self):
        """Test get_booter with unknown booter type raises ValueError."""
        from astrbot.core.computer import computer_client

        computer_client.session_booter.clear()

        mock_context = MagicMock()
        mock_config = MagicMock()
        mock_config.get = lambda key, default=None: {
            "provider_settings": {
                "computer_use_runtime": "sandbox",
                "sandbox": {
                    "booter": "unknown_type",
                },
            }
        }.get(key, default)
        mock_context.get_config = MagicMock(return_value=mock_config)

        with pytest.raises(ValueError) as exc_info:
            await computer_client.get_booter(mock_context, "test-session-id")
        assert "Unknown booter type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_booter_reuses_existing(self):
        """Test get_booter reuses existing booter for same session."""
        from astrbot.core.computer import computer_client
        from astrbot.core.computer.booters.shipyard import ShipyardBooter

        computer_client.session_booter.clear()

        mock_context = MagicMock()
        mock_config = MagicMock()
        mock_config.get = lambda key, default=None: {
            "provider_settings": {
                "computer_use_runtime": "sandbox",
                "sandbox": {
                    "booter": "shipyard",
                    "shipyard_endpoint": "http://localhost:8080",
                    "shipyard_access_token": "test_token",
                },
            }
        }.get(key, default)
        mock_context.get_config = MagicMock(return_value=mock_config)

        mock_booter = MagicMock()
        mock_booter.boot = AsyncMock()
        mock_booter.available = AsyncMock(return_value=True)
        mock_booter.shell = MagicMock()
        mock_booter.upload_file = AsyncMock(return_value={"success": True})

        with (
            patch.object(ShipyardBooter, "boot", new=AsyncMock()),
            patch(
                "astrbot.core.computer.computer_client._sync_skills_to_sandbox",
                AsyncMock(),
            ),
        ):
            # Pre-set the booter
            computer_client.session_booter["test-session"] = mock_booter

            booter1 = await computer_client.get_booter(mock_context, "test-session")
            booter2 = await computer_client.get_booter(mock_context, "test-session")
            assert booter1 is booter2

        # Cleanup
        computer_client.session_booter.clear()

    @pytest.mark.asyncio
    async def test_concurrent_first_calls_boot_one_sandbox(self):
        """Parallel first calls for a session share one boot and one skill sync."""
        from astrbot.core.computer import computer_client
        from astrbot.core.computer.booters.shipyard import ShipyardBooter

        computer_client.session_booter.clear()
        mock_context = MagicMock()
        mock_config = MagicMock()
        mock_config.get = lambda key, default=None: {
            "provider_settings": {
                "computer_use_runtime": "sandbox",
                "sandbox": {
                    "booter": "shipyard",
                    "shipyard_endpoint": "http://localhost:8080",
                    "shipyard_access_token": "test_token",
                },
            }
        }.get(key, default)
        mock_context.get_config = MagicMock(return_value=mock_config)

        async def slow_sync(_booter):
            await asyncio.sleep(0.05)

        sync = AsyncMock(side_effect=slow_sync)
        boot = AsyncMock()
        with (
            patch.object(ShipyardBooter, "boot", new=boot),
            patch.object(ShipyardBooter, "available", new=AsyncMock(return_value=True)),
            patch(
                "astrbot.core.computer.computer_client._sync_skills_to_sandbox",
                sync,
            ),
            patch(
                "astrbot.core.computer.computer_client._save_sandbox_binding",
                AsyncMock(),
            ),
        ):
            booters = await asyncio.gather(
                *(
                    computer_client.get_booter(mock_context, "race-session")
                    for _ in range(3)
                )
            )
        assert booters[0] is booters[1] is booters[2]
        assert boot.await_count == 1
        assert sync.await_count == 1
        computer_client.session_booter.clear()

    @pytest.mark.asyncio
    async def test_every_use_renews_the_sandbox_ttl(self):
        """get_booter is each tool's way in: it keeps the sandbox alive."""
        from astrbot.core.computer import computer_client

        computer_client.session_booter.clear()
        mock_context = MagicMock()
        mock_config = MagicMock()
        mock_config.get = lambda key, default=None: {
            "provider_settings": {
                "computer_use_runtime": "sandbox",
                "sandbox": {"booter": "shipyard"},
            }
        }.get(key, default)
        mock_context.get_config = MagicMock(return_value=mock_config)
        booter = MagicMock()
        booter.available = AsyncMock(return_value=True)
        booter.renew_ttl = AsyncMock()
        computer_client.session_booter["s"] = booter
        assert await computer_client.get_booter(mock_context, "s") is booter
        booter.renew_ttl.assert_awaited_once()
        computer_client.session_booter.clear()

    @pytest.mark.asyncio
    async def test_get_booter_rebuild_unavailable(self):
        """Test get_booter rebuilds when existing booter is unavailable."""
        from astrbot.core.computer import computer_client
        from astrbot.core.computer.booters.shipyard import ShipyardBooter

        computer_client.session_booter.clear()

        mock_context = MagicMock()
        mock_config = MagicMock()
        mock_config.get = lambda key, default=None: {
            "provider_settings": {
                "computer_use_runtime": "sandbox",
                "sandbox": {
                    "booter": "shipyard",
                    "shipyard_endpoint": "http://localhost:8080",
                    "shipyard_access_token": "test_token",
                },
            }
        }.get(key, default)
        mock_context.get_config = MagicMock(return_value=mock_config)

        mock_unavailable_booter = MagicMock(spec=ShipyardBooter)
        mock_unavailable_booter.available = AsyncMock(return_value=False)

        mock_new_booter = MagicMock(spec=ShipyardBooter)
        mock_new_booter.boot = AsyncMock()

        with (
            patch(
                "astrbot.core.computer.booters.shipyard.ShipyardBooter",
                return_value=mock_new_booter,
            ) as mock_booter_cls,
            patch(
                "astrbot.core.computer.computer_client._sync_skills_to_sandbox",
                AsyncMock(),
            ),
        ):
            session_id = "test-session-rebuild"
            # Pre-set the unavailable booter
            computer_client.session_booter[session_id] = mock_unavailable_booter

            # get_booter should detect the booter is unavailable and create a new one
            new_booter_instance = await computer_client.get_booter(
                mock_context, session_id
            )

            # Assert that a new booter was created and is now in the session
            mock_booter_cls.assert_called_once()
            mock_new_booter.boot.assert_awaited_once()
            assert new_booter_instance is mock_new_booter
            assert computer_client.session_booter[session_id] is mock_new_booter

        # Cleanup
        computer_client.session_booter.clear()


class TestSyncSkillsToSandbox:
    """Tests for _sync_skills_to_sandbox function."""

    @pytest.mark.asyncio
    async def test_sync_skills_no_skills_dir(self):
        """Test sync does nothing when skills directory doesn't exist."""
        from astrbot.core.computer import computer_client

        mock_booter = MagicMock()
        mock_booter.shell.exec = AsyncMock()
        mock_booter.upload_file = AsyncMock(return_value={"success": True})

        with (
            patch(
                "astrbot.core.computer.computer_client.get_astrbot_skills_path",
                return_value="/nonexistent/path",
            ),
            patch(
                "astrbot.core.computer.computer_client.os.path.isdir",
                return_value=False,
            ),
        ):
            await computer_client._sync_skills_to_sandbox(mock_booter)
            mock_booter.upload_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_skills_empty_dir(self):
        """Test sync does nothing when skills directory is empty."""
        from astrbot.core.computer import computer_client

        mock_booter = MagicMock()
        mock_booter.shell.exec = AsyncMock()
        mock_booter.upload_file = AsyncMock(return_value={"success": True})

        with (
            patch(
                "astrbot.core.computer.computer_client.get_astrbot_skills_path",
                return_value="/tmp/empty",
            ),
            patch(
                "astrbot.core.computer.computer_client.os.path.isdir",
                return_value=True,
            ),
            patch(
                "astrbot.core.computer.computer_client.Path.iterdir",
                return_value=iter([]),
            ),
        ):
            await computer_client._sync_skills_to_sandbox(mock_booter)
            mock_booter.upload_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_skills_success(self):
        """Test successful skills sync."""
        from astrbot.core.computer import computer_client

        mock_booter = MagicMock()
        mock_booter.shell.exec = AsyncMock(return_value={"exit_code": 0})
        mock_booter.upload_file = AsyncMock(return_value={"success": True})

        mock_skill_file = MagicMock()
        mock_skill_file.name = "skill.py"
        mock_skill_file.__str__ = lambda: "/tmp/skills/skill.py"

        with (
            patch(
                "astrbot.core.computer.computer_client.get_astrbot_skills_path",
                return_value="/tmp/skills",
            ),
            patch(
                "astrbot.core.computer.computer_client.os.path.isdir",
                return_value=True,
            ),
            patch(
                "astrbot.core.computer.computer_client.Path.iterdir",
                return_value=iter([mock_skill_file]),
            ),
            patch(
                "astrbot.core.computer.computer_client.get_astrbot_temp_path",
                return_value="/tmp",
            ),
            patch(
                "astrbot.core.computer.computer_client.shutil.make_archive",
            ),
            patch(
                "astrbot.core.computer.computer_client.os.path.exists",
                return_value=True,
            ),
            patch(
                "astrbot.core.computer.computer_client.os.remove",
            ),
        ):
            # Should not raise
            await computer_client._sync_skills_to_sandbox(mock_booter)
