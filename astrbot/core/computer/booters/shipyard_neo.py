from __future__ import annotations

import asyncio
import os
import re
import shlex
import time
from datetime import datetime, timezone
from typing import Any, cast

from astrbot.api import logger

from ..olayer import (
    BrowserComponent,
    FileSystemComponent,
    PythonComponent,
    ShellComponent,
)
from .base import ComputerBooter
from .shell_background import build_detached_shell_command
from .shipyard_search_file_util import search_files_via_shell

try:
    from shipyard_neo import BayClient
    from shipyard_neo.sandbox import Sandbox
except ImportError:
    logger.warning(
        "shipyard_neo_sdk is not installed. ShipyardNeoBooter will not work without it."
    )


def _maybe_model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _slice_content_by_lines(
    content: str,
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    lines = content.splitlines(keepends=True)
    start = 0 if offset is None else offset
    selected = lines[start:] if limit is None else lines[start : start + limit]
    return "".join(selected)


class NeoPythonComponent(PythonComponent):
    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def exec(
        self,
        code: str,
        kernel_id: str | None = None,
        timeout: int = 30,
        silent: bool = False,
    ) -> dict[str, Any]:
        _ = kernel_id  # Bay runtime does not expose kernel_id in current SDK.
        result = await self._sandbox.python.exec(code, timeout=timeout)
        payload = _maybe_model_dump(result)

        output_text = payload.get("output", "") or ""
        error_text = payload.get("error", "") or ""
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        rich_output = data.get("output") if isinstance(data.get("output"), dict) else {}
        if not isinstance(rich_output.get("images"), list):
            rich_output["images"] = []
        if "text" not in rich_output:
            rich_output["text"] = output_text

        if silent:
            rich_output["text"] = ""

        return {
            "success": bool(payload.get("success", error_text == "")),
            "data": {
                "output": rich_output,
                "error": error_text,
            },
            "execution_id": payload.get("execution_id"),
            "execution_time_ms": payload.get("execution_time_ms"),
            "code": payload.get("code"),
            "output": output_text,
            "error": error_text,
        }


class NeoShellComponent(ShellComponent):
    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = 300,
        shell: bool = True,
        background: bool = False,
    ) -> dict[str, Any]:
        if not shell:
            return {
                "stdout": "",
                "stderr": "error: only shell mode is supported in shipyard_neo booter.",
                "exit_code": 2,
                "success": False,
            }

        run_command = command
        if env:
            env_prefix = " ".join(
                f"{k}={shlex.quote(str(v))}" for k, v in sorted(env.items())
            )
            run_command = f"{env_prefix} {run_command}"

        if background:
            run_command = build_detached_shell_command(run_command)

        result = await self._sandbox.shell.exec(
            run_command,
            timeout=timeout or 300,
            cwd=cwd,
        )
        payload = _maybe_model_dump(result)

        stdout = payload.get("output", "") or ""
        stderr = payload.get("error", "") or ""
        exit_code = payload.get("exit_code")
        if background:
            pid: int | None = None
            try:
                pid = int(stdout.strip().splitlines()[-1])
            except Exception:
                pid = None
            return {
                "pid": pid,
                "stdout": (
                    f"Command is running in the background. pid={pid}"
                    if pid is not None
                    else "Command was submitted in the background."
                ),
                "stderr": stderr,
                "exit_code": exit_code,
                "success": bool(payload.get("success", not stderr)),
                "execution_id": payload.get("execution_id"),
                "execution_time_ms": payload.get("execution_time_ms"),
                "command": payload.get("command"),
            }

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "success": bool(payload.get("success", not stderr)),
            "execution_id": payload.get("execution_id"),
            "execution_time_ms": payload.get("execution_time_ms"),
            "command": payload.get("command"),
        }


class NeoFileSystemComponent(FileSystemComponent):
    def __init__(self, sandbox: Sandbox, shell: ShellComponent) -> None:
        self._sandbox = sandbox
        self._shell = shell

    async def create_file(
        self,
        path: str,
        content: str = "",
        mode: int = 0o644,
    ) -> dict[str, Any]:
        _ = mode
        await self._sandbox.filesystem.write_file(path, content)
        return {"success": True, "path": path}

    async def read_file(
        self,
        path: str,
        encoding: str = "utf-8",
        offset: int | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        _ = encoding
        content = await self._sandbox.filesystem.read_file(path)
        return {
            "success": True,
            "path": path,
            "content": _slice_content_by_lines(
                content,
                offset=offset,
                limit=limit,
            ),
        }

    async def search_files(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        after_context: int | None = None,
        before_context: int | None = None,
    ) -> dict[str, Any]:
        return await search_files_via_shell(
            self._shell,
            pattern=pattern,
            path=path,
            glob=glob,
            after_context=after_context,
            before_context=before_context,
        )

    async def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        _ = encoding
        content = await self._sandbox.filesystem.read_file(path)
        occurrences = content.count(old_string)
        if occurrences == 0:
            return {
                "success": False,
                "error": "old string not found in file",
                "replacements": 0,
            }
        if replace_all:
            updated = content.replace(old_string, new_string)
            replacements = occurrences
        else:
            updated = content.replace(old_string, new_string, 1)
            replacements = 1
        await self._sandbox.filesystem.write_file(path, updated)
        return {
            "success": True,
            "path": path,
            "replacements": replacements,
        }

    async def write_file(
        self,
        path: str,
        content: str,
        mode: str = "w",
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        _ = mode
        _ = encoding
        await self._sandbox.filesystem.write_file(path, content)
        return {"success": True, "path": path}

    async def delete_file(self, path: str) -> dict[str, Any]:
        await self._sandbox.filesystem.delete(path)
        return {"success": True, "path": path}

    async def list_dir(
        self,
        path: str = ".",
        show_hidden: bool = False,
    ) -> dict[str, Any]:
        entries = await self._sandbox.filesystem.list_dir(path)
        data = []
        for entry in entries:
            item = _maybe_model_dump(entry)
            if not show_hidden and str(item.get("name", "")).startswith("."):
                continue
            data.append(item)
        return {"success": True, "path": path, "entries": data}


class NeoBrowserComponent(BrowserComponent):
    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def exec(
        self,
        cmd: str,
        timeout: int = 30,
        description: str | None = None,
        tags: str | None = None,
        learn: bool = False,
        include_trace: bool = False,
    ) -> dict[str, Any]:
        result = await self._sandbox.browser.exec(
            cmd,
            timeout=timeout,
            description=description,
            tags=tags,
            learn=learn,
            include_trace=include_trace,
        )
        return _maybe_model_dump(result)

    async def exec_batch(
        self,
        commands: list[str],
        timeout: int = 60,
        stop_on_error: bool = True,
        description: str | None = None,
        tags: str | None = None,
        learn: bool = False,
        include_trace: bool = False,
    ) -> dict[str, Any]:
        result = await self._sandbox.browser.exec_batch(
            commands,
            timeout=timeout,
            stop_on_error=stop_on_error,
            description=description,
            tags=tags,
            learn=learn,
            include_trace=include_trace,
        )
        return _maybe_model_dump(result)

    async def run_skill(
        self,
        skill_key: str,
        timeout: int = 60,
        stop_on_error: bool = True,
        include_trace: bool = False,
        description: str | None = None,
        tags: str | None = None,
    ) -> dict[str, Any]:
        result = await self._sandbox.browser.run_skill(
            skill_key=skill_key,
            timeout=timeout,
            stop_on_error=stop_on_error,
            include_trace=include_trace,
            description=description,
            tags=tags,
        )
        return _maybe_model_dump(result)


#: Process limit a sandbox should offer. Each shell session costs a tmux pane
#: and a shell, and the limit is part of the Bay profile, not anything AstrBot
#: can set on a sandbox it creates.
RECOMMENDED_NPROC = 1024
# A sandbox in use has its TTL renewed at most this often, so its expiry
# stays within this of a full TTL after its last use.
TTL_RENEW_INTERVAL_SECONDS = 3600


def _quantity(value: Any) -> float:
    """Best-effort number for a Kubernetes-style resource quantity.

    Profiles report values such as ``2``, ``"1500m"`` or ``"4Gi"``. Only the
    ordering matters here, so anything unparsable scores zero rather than
    failing profile selection.
    """
    if isinstance(value, (int, float)):
        return float(value)
    match = re.match(r"^\s*([0-9.]+)\s*([A-Za-z]*)\s*$", str(value or ""))
    if not match:
        return 0.0
    factors = {
        "": 1.0,
        "m": 1e-3,
        "k": 1e3,
        "K": 1e3,
        "Ki": 1024.0,
        "M": 1e6,
        "Mi": 1024.0**2,
        "G": 1e9,
        "Gi": 1024.0**3,
        "T": 1e12,
        "Ti": 1024.0**4,
    }
    try:
        return float(match.group(1)) * factors.get(match.group(2), 1.0)
    except ValueError:
        return 0.0


class ShipyardNeoBooter(ComputerBooter):
    """Booter backed by Shipyard Neo (Bay).

    If *endpoint_url* is empty or set to ``"__auto__"``, Bay will be
    started automatically as a Docker container (like Boxlite does for
    Ship containers).
    """

    AUTO_SENTINEL = "__auto__"
    DEFAULT_PROFILE = "python-default"

    def __init__(
        self,
        endpoint_url: str,
        access_token: str,
        profile: str = "",
        ttl: int = 43200,
        reuse_sandbox_id: str = "",
    ) -> None:
        # Sandbox bound to this session before a restart; reused if still alive.
        self._reuse_sandbox_id = reuse_sandbox_id
        # Constructor settings, before boot resolves auto mode.
        self._init_settings = (endpoint_url, access_token, profile, ttl)
        self._endpoint_url = endpoint_url
        self._access_token = access_token
        self._profile = profile.strip() if profile else ""
        self._ttl = ttl
        self._client: BayClient | None = None
        self._sandbox: Sandbox | None = None
        self._bay_manager: Any = None  # BayContainerManager when auto-started
        self._fs: FileSystemComponent | None = None
        self._python: PythonComponent | None = None
        self._shell: ShellComponent | None = None
        self._browser: BrowserComponent | None = None
        self._ttl_renewed_at = float("-inf")

    @property
    def bay_client(self) -> Any:
        return self._client

    @property
    def sandbox(self) -> Any:
        return self._sandbox

    @property
    def capabilities(self) -> tuple[str, ...] | None:
        """Sandbox capabilities from the Bay profile.

        Returns an immutable tuple after :meth:`boot`; ``None`` before boot.
        """
        if self._sandbox is None:
            return None
        caps = getattr(self._sandbox, "capabilities", None)
        return tuple(caps) if caps is not None else None

    @property
    def is_auto_mode(self) -> bool:
        """True when Bay should be auto-started."""
        ep = (self._endpoint_url or "").strip()
        return not ep or ep == self.AUTO_SENTINEL

    async def boot(self, session_id: str) -> None:
        _ = session_id

        # --- Auto-start Bay if needed ---
        if self.is_auto_mode:
            from .bay_manager import BayContainerManager

            # Clean up previous manager if re-booting
            if self._bay_manager is not None:
                await self._bay_manager.close_client()

            logger.info("[Computer] Neo auto-start mode: launching Bay container")
            self._bay_manager = BayContainerManager()
            self._endpoint_url = await self._bay_manager.ensure_running()
            await self._bay_manager.wait_healthy()
            # Read auto-provisioned credentials
            if not self._access_token:
                self._access_token = await self._bay_manager.read_credentials()
            logger.info("[Computer] Bay auto-started at %s", self._endpoint_url)

        if not self._endpoint_url or not self._access_token:
            if self._bay_manager is not None:
                raise ValueError(
                    "Bay container started but credentials could not be read. "
                    "Ensure Bay generated credentials.json, or set access_token manually."
                )
            raise ValueError(
                "Shipyard Neo sandbox configuration is incomplete. "
                "Set endpoint (default http://127.0.0.1:8114) and access token, "
                "or ensure Bay's credentials.json is accessible for auto-discovery."
            )

        self._client = BayClient(
            endpoint_url=self._endpoint_url,
            access_token=self._access_token,
        )
        await self._client.__aenter__()

        # Resolve profile: user-specified > smart selection > default.
        # An empty profile means auto-select; any non-empty profile must be
        # honoured as an explicit choice, including "python-default".
        resolved_profile = await self._resolve_profile(self._client)

        self._sandbox = await self._reuse_sandbox(resolved_profile)
        if self._sandbox is None:
            self._sandbox = await self._client.create_sandbox(
                profile=resolved_profile,
                ttl=self._ttl,
            )
            # --- Readiness gate: wait until sandbox session is READY ---
            await self._wait_until_ready(self._sandbox)
        # A reused (idle) sandbox starts a session on its next call.

        self._shell = NeoShellComponent(self._sandbox)
        self._fs = NeoFileSystemComponent(self._sandbox, self._shell)
        self._python = NeoPythonComponent(self._sandbox)

        caps = self.capabilities or ()
        self._browser = (
            NeoBrowserComponent(self._sandbox) if "browser" in caps else None
        )

        # Resources, including the process limit every tmux session competes
        # for, are fixed by the profile; nothing here can raise them.
        logger.info(
            "Got Shipyard Neo sandbox: %s (profile=%s, capabilities=%s, auto=%s)",
            self._sandbox.id,
            resolved_profile,
            list(caps),
            bool(self._bay_manager),
        )
        await self._warn_if_process_limit_is_low()

    async def _warn_if_process_limit_is_low(self) -> None:
        """Says so when the sandbox cannot host many shell sessions.

        Every `exec_command` session is a tmux pane plus a shell, so a low
        process limit shows up as commands failing to fork, which is hard to
        recognise for what it is. The limit belongs to the Bay profile and
        cannot be raised from here, so name it at boot instead.
        """
        if self._shell is None:
            return
        try:
            result = await self._shell.exec("ulimit -Hu", timeout=15)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not fail boot
            logger.debug("[Computer] Could not read the sandbox process limit: %s", exc)
            return
        text = str(
            (result or {}).get("stdout") if isinstance(result, dict) else result or ""
        ).strip()
        if text == "unlimited":
            return
        try:
            limit = int(text.splitlines()[-1])
        except (ValueError, IndexError):
            return
        if limit < RECOMMENDED_NPROC:
            logger.warning(
                "[Computer] Sandbox process limit is %d, below the %d shell "
                "sessions this setup expects to be able to run. Raise nproc/pids "
                "in the Bay profile, or lower "
                "provider_settings.sandbox.max_shell_sessions.",
                limit,
                RECOMMENDED_NPROC,
            )

    @property
    def sandbox_id(self) -> str:
        return str(getattr(self._sandbox, "id", "") or "")

    def without_reuse(self) -> ShipyardNeoBooter:
        """A fresh booter with the same settings that creates a new sandbox."""
        endpoint_url, access_token, profile, ttl = self._init_settings
        return type(self)(
            endpoint_url=endpoint_url,
            access_token=access_token,
            profile=profile,
            ttl=ttl,
        )

    @property
    def reused_sandbox(self) -> bool:
        """Whether boot attached to the previously bound sandbox."""
        return (
            bool(self._reuse_sandbox_id) and self.sandbox_id == self._reuse_sandbox_id
        )

    async def _reuse_sandbox(self, profile: str = "") -> Sandbox | None:
        if not self._reuse_sandbox_id or self._client is None:
            return None
        try:
            sandbox = await self._client.get_sandbox(self._reuse_sandbox_id)
        except Exception as e:  # noqa: BLE001 - gone or unreachable: make a new one
            logger.info(
                "[Computer] Bound sandbox %s unavailable (%s); creating a new one",
                self._reuse_sandbox_id,
                e,
            )
            return None
        status = str(getattr(sandbox.status, "value", sandbox.status)).lower()
        if status in ("failed", "expired"):
            return None
        bound_profile = str(getattr(sandbox, "profile", "") or "")
        if profile and bound_profile and bound_profile != profile:
            logger.info(
                "[Computer] Bound sandbox %s uses profile %s, not %s; creating a new one",
                sandbox.id,
                bound_profile,
                profile,
            )
            return None
        logger.info("[Computer] Reusing bound sandbox %s", sandbox.id)
        return sandbox

    async def _wait_until_ready(self, sandbox: Sandbox) -> None:
        """Poll sandbox status until READY, or raise on FAILED / timeout.

        Covers both warm-pool hits (near-instant) and cold starts (up to 180s).
        On FAILED, EXPIRED, or timeout the sandbox is deleted before raising
        so no orphan resources leak on Bay.
        """
        READINESS_TIMEOUT = 180  # seconds
        POLL_INTERVAL = 2  # seconds

        sandbox_id = sandbox.id
        deadline = asyncio.get_running_loop().time() + READINESS_TIMEOUT

        while True:
            await sandbox.refresh()
            status = getattr(sandbox.status, "value", str(sandbox.status))

            if status == "ready":
                logger.info(
                    "[Computer] Sandbox %s is ready (profile=%s)",
                    sandbox_id,
                    sandbox.profile,
                )
                return

            if status in {"failed", "expired"}:
                logger.error(
                    "[Computer] Sandbox %s reached terminal state: %s",
                    sandbox_id,
                    status,
                )
                try:
                    await sandbox.delete()
                except Exception as del_err:
                    logger.warning(
                        "[Computer] Failed to delete failed sandbox %s: %s",
                        sandbox_id,
                        del_err,
                    )
                raise RuntimeError(
                    f"Sandbox {sandbox_id} is in terminal state: {status}"
                )

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.error(
                    "[Computer] Sandbox %s did not become ready within %ds "
                    "(last status: %s)",
                    sandbox_id,
                    READINESS_TIMEOUT,
                    status,
                )
                try:
                    await sandbox.delete()
                except Exception as del_err:
                    logger.warning(
                        "[Computer] Failed to delete timed-out sandbox %s: %s",
                        sandbox_id,
                        del_err,
                    )
                raise TimeoutError(
                    f"Sandbox {sandbox_id} did not become ready within "
                    f"{READINESS_TIMEOUT}s (last status: {status})"
                )

            logger.debug(
                "[Computer] Sandbox %s status=%s, waiting...",
                sandbox_id,
                status,
            )
            await asyncio.sleep(POLL_INTERVAL)

    async def _resolve_profile(self, client: Any) -> str:
        """Pick the best profile for this session.

        Resolution order:
        1. User-specified profile (non-empty) → use as-is.
        2. Query ``GET /v1/profiles`` and pick the profile with the most
           capabilities, preferring profiles that include ``"browser"``.
        3. Fall back to :attr:`DEFAULT_PROFILE`.

        Auth errors (401/403) are re-raised immediately — they indicate a
        misconfigured token, and silently falling back would just delay the
        real failure to ``create_sandbox``.
        """
        # User explicitly set a profile → honour it.
        if self._profile:
            logger.info("[Computer] Using user-specified profile: %s", self._profile)
            return self._profile

        # Query Bay for available profiles
        from shipyard_neo.errors import ForbiddenError, UnauthorizedError

        try:
            profile_list = await client.list_profiles()
            profiles = profile_list.items
        except (UnauthorizedError, ForbiddenError):
            raise  # auth errors must not be silenced
        except Exception as exc:
            logger.warning(
                "[Computer] Failed to query Bay profiles, falling back to %s: %s",
                self.DEFAULT_PROFILE,
                exc,
            )
            return self.DEFAULT_PROFILE

        if not profiles:
            return self.DEFAULT_PROFILE

        def _score(p: Any) -> tuple[int, int, float, float]:
            """(has_browser, capability_count, cpu, memory) — higher is better.

            Capabilities still come first: a profile that cannot do what the
            session needs is no use however large it is. Size breaks the tie,
            because every shell session costs processes and memory, and the
            profile is the only place those limits are set.
            """
            caps = getattr(p, "capabilities", []) or []
            resources = getattr(p, "resources", {}) or {}
            return (
                1 if "browser" in caps else 0,
                len(caps),
                _quantity(resources.get("cpu")),
                _quantity(resources.get("memory")),
            )

        best = max(profiles, key=_score)
        chosen = getattr(best, "id", self.DEFAULT_PROFILE)

        logger.info(
            "[Computer] Auto-selected profile %s (capabilities=%s, resources=%s)",
            chosen,
            getattr(best, "capabilities", []),
            getattr(best, "resources", {}),
        )

        return chosen

    async def shutdown(self, *, delete_sandbox: bool = False) -> None:
        if self._client is not None:
            sandbox_id = getattr(self._sandbox, "id", "unknown")

            # Delete sandbox on Bay BEFORE closing the HTTP client.
            # This is critical for cleanup — calling delete after
            # __aexit__ would fail because the httpx session is already
            # torn down.
            if delete_sandbox and self._sandbox is not None:
                try:
                    logger.info(
                        "[Computer] Deleting Shipyard Neo sandbox: id=%s", sandbox_id
                    )
                    await self._sandbox.delete()
                    logger.info(
                        "[Computer] Shipyard Neo sandbox deleted: id=%s", sandbox_id
                    )
                except Exception as e:
                    logger.warning(
                        "[Computer] Failed to delete sandbox %s (may already be "
                        "cleaned up by Bay GC): %s",
                        sandbox_id,
                        e,
                    )

            logger.info(
                "[Computer] Shutting down Shipyard Neo sandbox client: id=%s",
                sandbox_id,
            )
            await self._client.__aexit__(None, None, None)
            self._client = None
            self._sandbox = None
            logger.info(
                "[Computer] Shipyard Neo sandbox client shut down: id=%s", sandbox_id
            )

        # NOTE: We intentionally do NOT stop the Bay container here.
        # It stays running for reuse by future sessions.  The user can
        # stop it manually or via ``BayContainerManager.stop()``.
        if self._bay_manager is not None:
            await self._bay_manager.close_client()

    @property
    def fs(self) -> FileSystemComponent:
        if self._fs is None:
            raise RuntimeError("ShipyardNeoBooter is not initialized.")
        return self._fs

    @property
    def python(self) -> PythonComponent:
        if self._python is None:
            raise RuntimeError("ShipyardNeoBooter is not initialized.")
        return self._python

    @property
    def shell(self) -> ShellComponent:
        if self._shell is None:
            raise RuntimeError("ShipyardNeoBooter is not initialized.")
        return self._shell

    @property
    def browser(self) -> BrowserComponent:
        if self._browser is None:
            raise RuntimeError("ShipyardNeoBooter is not initialized.")
        return self._browser

    async def upload_file(self, path: str, file_name: str) -> dict:
        if self._sandbox is None:
            raise RuntimeError("ShipyardNeoBooter is not initialized.")
        with open(path, "rb") as f:
            content = f.read()
        remote_path = file_name.lstrip("/")
        await self._sandbox.filesystem.upload(remote_path, content)
        logger.info("[Computer] File uploaded to Neo sandbox: %s", remote_path)
        return {
            "success": True,
            "message": "File uploaded successfully",
            "file_path": remote_path,
        }

    async def download_file(self, remote_path: str, local_path: str) -> None:
        if self._sandbox is None:
            raise RuntimeError("ShipyardNeoBooter is not initialized.")
        content = await self._sandbox.filesystem.download(remote_path.lstrip("/"))
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(cast(bytes, content))
        logger.info(
            "[Computer] File downloaded from Neo sandbox: %s -> %s",
            remote_path,
            local_path,
        )

    async def renew_ttl(self) -> None:
        """Keeps a sandbox in use alive: its TTL counts from the last use.

        Bay deletes a sandbox, and its workspace with it, when its TTL from
        creation runs out, however busy it is; activity only renews the idle
        timeout. A chat's files would be lost every TTL. So each use (at most
        hourly) moves the expiry to a full TTL from now: the TTL is no longer a
        total lifetime, only how long a sandbox nobody uses is kept.
        """
        if self._sandbox is None or not self._ttl or self._ttl <= 0:
            return
        now = time.monotonic()
        if now - self._ttl_renewed_at < TTL_RENEW_INTERVAL_SECONDS:
            return
        self._ttl_renewed_at = now
        expires_at = getattr(self._sandbox, "expires_at", None)
        if not isinstance(expires_at, datetime):
            return  # no TTL
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        left = (expires_at - datetime.now(timezone.utc)).total_seconds()
        if left <= 0 or left > self._ttl - TTL_RENEW_INTERVAL_SECONDS:
            return
        try:
            # Bay extends from max(expiry, now): this ends a full TTL from now.
            await self._sandbox.extend_ttl(int(self._ttl - left))
        except Exception as e:  # noqa: BLE001 - retried on a later use
            self._ttl_renewed_at = float("-inf")
            logger.warning(
                "[Computer] Could not renew sandbox %s TTL: %s", self.sandbox_id, e
            )
            return
        logger.info(
            "[Computer] Renewed sandbox %s TTL: %.1f days were left, now %.1f",
            self.sandbox_id,
            left / 86400,
            self._ttl / 86400,
        )

    async def available(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            await self._sandbox.refresh()
            status = getattr(self._sandbox.status, "value", str(self._sandbox.status))
            healthy = status not in {"failed", "expired"}
            logger.info(
                "[Computer] Neo sandbox health check: id=%s, status=%s, healthy=%s",
                getattr(self._sandbox, "id", "unknown"),
                status,
                healthy,
            )
            return healthy
        except Exception as e:
            logger.error(f"Error checking Shipyard Neo sandbox availability: {e}")
            return False
