from pathlib import Path

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.utils.astrbot_path import get_astrbot_workspaces_path
from astrbot.core.workspace import normalize_umo_for_workspace


def workspace_root(umo: str) -> Path:
    """Return the legacy workspace root for compatibility.

    Args:
        umo: Unified message origin.

    Returns:
        Legacy per-session workspace root.
    """
    return (
        Path(get_astrbot_workspaces_path()) / normalize_umo_for_workspace(umo)
    ).resolve(strict=False)


def check_admin_permission(
    context: ContextWrapper[AstrAgentContext], operation_name: str
) -> str | None:
    cfg = context.context.context.get_config(
        umo=context.context.event.unified_msg_origin
    )
    provider_settings = cfg.get("provider_settings", {})
    require_admin = provider_settings.get("computer_use_require_admin", True)
    if require_admin and context.context.event.role != "admin":
        return (
            f"error: Permission denied. {operation_name} is only allowed for admin users. "
            "Tell user to add their user ID under `AstrBot WebUI -> Config -> "
            "Access Control -> Administrators -> Administrator IDs` in the "
            "profile used by this bot if they need this feature. "
            f"User's ID is: {context.context.event.get_sender_id()}. User's ID can be found by using /sid command."
        )
    return None


# Files that must never be handed to a model, even when it may run commands:
# AstrBot's own secrets and the usual credential stores of the host account.
_SECRET_DIR_NAMES = {
    ".aws",
    ".codex",
    ".config/gcloud",
    ".docker",
    ".gnupg",
    ".kube",
    ".ssh",
}
_SECRET_FILE_NAMES = {
    ".env",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pgpass",
    "auth.json",
    "cmd_config.json",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
_SECRET_FILE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}


def reject_secret_path(path: str | Path) -> str | None:
    """Refuse host paths that hold credentials.

    Tools that copy host files somewhere a model can read them (the sandbox,
    a chat message) call this first. AstrBot's own config, database, backups
    and Codex home are refused, together with the usual credential stores of
    the host account; ``data/temp`` and the rest of the data directory stay
    available so attachments and generated files still work.

    Args:
        path: Host path the model asked for.

    Returns:
        An error message to return to the model, or None when the path is fine.
    """
    from astrbot.core.utils.astrbot_path import (
        get_astrbot_config_path,
        get_astrbot_data_path,
    )

    resolved = Path(path).expanduser().resolve(strict=False)
    data_path = Path(get_astrbot_data_path()).resolve(strict=False)
    denied_roots = [
        Path(get_astrbot_config_path()).resolve(strict=False),
        data_path / "codex_home",
        data_path / "backups",
    ]
    home = Path.home().resolve(strict=False)
    denied_roots += [home / name for name in _SECRET_DIR_NAMES]

    for root in denied_roots:
        if resolved == root or root in resolved.parents:
            return f"error: {root} holds credentials and cannot be read by tools."
    name = resolved.name.lower()
    if (
        name in _SECRET_FILE_NAMES
        or resolved.suffix.lower() in _SECRET_FILE_SUFFIXES
        or (resolved.parent == data_path and name.startswith("data_v"))
    ):
        return f"error: {resolved.name} looks like a credential file and cannot be read by tools."
    return None
