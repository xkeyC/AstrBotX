"""Filesystem tools of the sandbox runtime.

The sandbox runtime exposes `astrbot_upload_file`, `astrbot_download_file`,
`astrbot_file_read_tool`, `astrbot_file_write_tool`, `astrbot_file_edit_tool`,
and `astrbot_grep_tool`. Paths are passed to the sandbox unchanged; the
sandbox enforces its own filesystem boundaries.

Read/write/edit/grep are not path-restricted by this module. Upload and
download copy files between the host and the sandbox and are denied by
`check_admin_permission` for non-admin users when `computer_use_require_admin`
is on.
"""

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from astrbot.api import FunctionTool, logger
from astrbot.api.event import MessageChain
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.computer.computer_client import get_booter
from astrbot.core.computer.file_read_utils import read_file_tool_result
from astrbot.core.message.components import File, Image
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from ..registry import builtin_tool
from .util import check_admin_permission, reject_secret_path

_SANDBOX_RUNTIME_TOOL_CONFIG = {
    "provider_settings.computer_use_runtime": "sandbox",
}
_IMAGE_FILE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}


def _remote_basename(path: str) -> str:
    # Sandbox paths may come from POSIX or Windows runtimes; normalize separators
    # without interpreting the path against the host filesystem.
    return path.replace("\\", "/").rstrip("/").split("/")[-1]


def _decode_escaped_text(value: str) -> str:
    """Decode common escaped control sequences used in tool arguments."""
    return (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
    )


@builtin_tool(config=_SANDBOX_RUNTIME_TOOL_CONFIG)
@dataclass
class FileReadTool(FunctionTool):
    name: str = "astrbot_file_read_tool"
    description: str = "Read file content. Supports text and image files."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to read. If relative, will be in workspace root.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Optional line offset to start reading from. 0-based index.",
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum number of lines to read.",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }
    )

    def _validate_read_window(
        self,
        offset: int | None,
        limit: int | None,
    ) -> tuple[int | None, int | None]:
        if offset is not None and offset < 0:
            raise ValueError("`offset` must be greater than or equal to 0.")
        if limit is not None and limit < 1:
            raise ValueError("`limit` must be greater than or equal to 1.")
        return offset, limit

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> ToolExecResult:
        try:
            normalized_path = path.strip()
            if not normalized_path:
                raise ValueError("`path` must be a non-empty string.")
            offset, limit = self._validate_read_window(offset, limit)
            sb = await get_booter(
                context.context.context,
                context.context.event.unified_msg_origin,
            )
            return await read_file_tool_result(
                sb,
                path=normalized_path,
                offset=offset,
                limit=limit,
            )
        except IsADirectoryError:
            return (
                f"Error: '{normalized_path}' is a directory, not a file. "
                "Use a file path instead, or use 'astrbot_execute_shell' to list "
                "directory contents."
            )
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            logger.error(f"Error reading file: {exc}")
            return f"Error reading file: {exc}"


@builtin_tool(config=_SANDBOX_RUNTIME_TOOL_CONFIG)
@dataclass
class FileWriteTool(FunctionTool):
    name: str = "astrbot_file_write_tool"
    description: str = "Write UTF-8 text content to a file."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to write. If relative, will be in workspace root.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["path", "content"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        path: str,
        content: str,
    ) -> ToolExecResult:
        try:
            normalized_path = path.strip()
            if not normalized_path:
                raise ValueError("`path` must be a non-empty string.")
            sb = await get_booter(
                context.context.context,
                context.context.event.unified_msg_origin,
            )
            result = await sb.fs.write_file(
                path=normalized_path,
                content=content,
                mode="w",
                encoding="utf-8",
            )
            if not result.get("success", False):
                error_detail = str(result.get("error", "") or "").strip()
                return (
                    "Error writing file: "
                    f"{error_detail or 'unknown filesystem write error'}"
                )
            return f"File written successfully: {normalized_path}"
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            logger.error(f"Error writing file: {exc}")
            return f"Error writing file: {exc}"


@builtin_tool(config=_SANDBOX_RUNTIME_TOOL_CONFIG)
@dataclass
class FileEditTool(FunctionTool):
    name: str = "astrbot_file_edit_tool"
    description: str = "Editing files."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to edit. If relative, will be in workspace root.",
                },
                "old": {
                    "type": "string",
                    "description": "The exact old text to replace.",
                },
                "new": {
                    "type": "string",
                    "description": "The replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Whether to replace all matches. Defaults to false.",
                },
            },
            "required": ["path", "old", "new"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> ToolExecResult:
        try:
            normalized_path = path.strip()
            if not normalized_path:
                raise ValueError("`path` must be a non-empty string.")
            # The read, write and grep tools all pass their strings through
            # unchanged, so a file may legitimately contain a literal "\n".
            # Use the arguments as they were given and only fall back to
            # decoding escape sequences when that finds nothing to replace.
            attempts = [(old, new)]
            decoded = (_decode_escaped_text(old), _decode_escaped_text(new))
            if decoded != (old, new):
                attempts.append(decoded)
            sb = await get_booter(
                context.context.context,
                context.context.event.unified_msg_origin,
            )
            for old_string, new_string in attempts:
                result = await sb.fs.edit_file(
                    path=normalized_path,
                    old_string=old_string,
                    new_string=new_string,
                    replace_all=replace_all,
                    encoding="utf-8",
                )
                if result.get("success", False):
                    break
            if not result.get("success", False):
                error_detail = str(result.get("error", "") or "").strip()
                return (
                    "Error editing file: "
                    f"{error_detail or 'unknown filesystem edit error'}"
                )
            replacements = int(result.get("replacements", 0) or 0)
            mode_text = "all matches" if replace_all else "first match"
            return (
                f"Edited {normalized_path}. "
                f"Replaced {replacements} occurrence(s) using {mode_text} mode."
            )
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            logger.error(f"Error editing file: {exc}")
            return f"Error editing file: {exc}"


@builtin_tool(config=_SANDBOX_RUNTIME_TOOL_CONFIG)
@dataclass
class GrepTool(FunctionTool):
    name: str = "astrbot_grep_tool"
    description: str = "Search and read file contents using ripgrep."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The expression pattern to search for in file contents.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (rg PATH). If relative, will be in workspace root.",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob filter such as `*.py`, `*.{ts,tsx}`.",
                },
                "-A": {
                    "type": "integer",
                    "description": "Number of trailing context lines to include after each match.",
                    "minimum": 0,
                },
                "-B": {
                    "type": "integer",
                    "description": "Number of leading context lines to include before each match.",
                    "minimum": 0,
                },
                "-C": {
                    "type": "integer",
                    "description": "Number of leading and trailing context lines to include around each match.",
                    "minimum": 0,
                },
                "result_limit": {
                    "type": "integer",
                    "description": "Maximum number of result groups returned by the tool. Defaults to 100.",
                    "minimum": 1,
                },
            },
            "required": ["pattern"],
        }
    )

    def _resolve_context_options(
        self,
        after_context: int | None,
        before_context: int | None,
        context: int | None,
    ) -> tuple[int | None, int | None]:
        if context is not None and context < 0:
            raise ValueError("`-C` must be greater than or equal to 0.")
        if after_context is not None and after_context < 0:
            raise ValueError("`-A` must be greater than or equal to 0.")
        if before_context is not None and before_context < 0:
            raise ValueError("`-B` must be greater than or equal to 0.")

        resolved_after = context if after_context is None else after_context
        resolved_before = context if before_context is None else before_context
        return resolved_after, resolved_before

    def _split_output_groups(self, output: str, *, has_context: bool) -> list[str]:
        if not output.strip():
            return []

        if not has_context:
            return [f"{line}\n" for line in output.splitlines() if line.strip()]

        groups: list[str] = []
        current: list[str] = []

        for line in output.splitlines(keepends=True):
            if line.strip() == "--":
                if current:
                    groups.append("".join(current))
                    current = []
                continue
            if not line.strip():
                continue
            current.append(line)

        if current:
            groups.append("".join(current))
        return groups

    def _apply_result_limit(
        self,
        output: str,
        *,
        result_limit: int,
        has_context: bool,
    ) -> str:
        if result_limit < 1:
            raise ValueError("`result_limit` must be greater than or equal to 1.")

        groups = self._split_output_groups(output, has_context=has_context)
        if len(groups) <= result_limit:
            return output if output.strip() else "No matches found."

        limited_output = "".join(groups[:result_limit]).rstrip()
        return f"{limited_output}\n\n[Truncated to first {result_limit} result groups.]"

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        result_limit: int = 100,
        **kwargs,
    ) -> ToolExecResult:
        normalized_pattern = pattern.strip()
        if not normalized_pattern:
            return "Error: `pattern` must be a non-empty string."

        try:
            after_context, before_context = self._resolve_context_options(
                kwargs.get("-A"),
                kwargs.get("-B"),
                kwargs.get("-C"),
            )
            has_context = (after_context or 0) > 0 or (before_context or 0) > 0
            sb = await get_booter(
                context.context.context,
                context.context.event.unified_msg_origin,
            )
            result = await sb.fs.search_files(
                pattern=normalized_pattern,
                path=path.strip() if path and path.strip() else ".",
                glob=glob,
                after_context=after_context,
                before_context=before_context,
            )
            if not result.get("success", False):
                error_detail = str(result.get("error", "") or "").strip()
                logger.error("GrepTool search failed: %s", error_detail)
                return (
                    "Error searching files: "
                    f"{error_detail or 'unknown filesystem search error'}"
                )
            return self._apply_result_limit(
                str(result.get("content", "") or ""),
                result_limit=result_limit,
                has_context=has_context,
            )
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            logger.error(f"Error searching files: {exc}")
            return f"Error searching files: {exc}"


@builtin_tool(config=_SANDBOX_RUNTIME_TOOL_CONFIG)
@dataclass
class FileUploadTool(FunctionTool):
    name: str = "astrbot_upload_file"
    description: str = (
        "Transfer a file FROM the host machine INTO the sandbox so that sandbox "
        "code can access it. Use this when the user sends/attaches a file and you "
        "need to process it inside the sandbox. The local_path must point to an "
        "existing file on the host filesystem."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "local_path": {
                    "type": "string",
                    "description": "Absolute path to the file on the host filesystem that will be copied into the sandbox.",
                },
                # "remote_path": {
                #     "type": "string",
                #     "description": "The filename to use in the sandbox. If not provided, file will be saved to the working directory with the same name as the local file.",
                # },
            },
            "required": ["local_path"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        local_path: str,
    ) -> str | None:
        if permission_error := check_admin_permission(context, "File upload/download"):
            return permission_error
        # This tool reads a host path into the sandbox, where the model can
        # read it back, so credentials are refused before anything is opened.
        if secret_error := reject_secret_path(local_path):
            return secret_error
        sb = await get_booter(
            context.context.context,
            context.context.event.unified_msg_origin,
        )
        try:
            # Check if file exists
            if not os.path.exists(local_path):
                return f"Error: File does not exist: {local_path}"

            if not os.path.isfile(local_path):
                return f"Error: Path is not a file: {local_path}"

            # Use basename if sandbox_filename is not provided
            remote_path = os.path.basename(local_path)

            # Upload file to sandbox
            result = await sb.upload_file(local_path, remote_path)
            logger.debug(f"Upload result: {result}")
            success = result.get("success", False)

            if not success:
                return f"Error uploading file: {result.get('message', 'Unknown error')}"

            file_path = result.get("file_path", "")
            logger.info(f"File {local_path} uploaded to sandbox at {file_path}")

            return f"File uploaded successfully to {file_path}"
        except Exception as e:
            logger.error(f"Error uploading file {local_path}: {e}")
            return f"Error uploading file: {str(e)}"


@builtin_tool(config=_SANDBOX_RUNTIME_TOOL_CONFIG)
@dataclass
class FileDownloadTool(FunctionTool):
    name: str = "astrbot_download_file"
    description: str = (
        "Copy a file from the sandbox out to the host, where the copy stays. Use "
        "it only when a copy on the host is wanted. To send images or files to "
        "the user, use send_message_to_user with the sandbox path instead: it "
        "leaves no copy behind."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "remote_path": {
                    "type": "string",
                    "description": "Path of the file in the sandbox.",
                },
                "also_send_to_user": {
                    "type": "boolean",
                    "description": "Also send the copy to the user (default true).",
                },
            },
            "required": ["remote_path"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        remote_path: str,
        also_send_to_user: bool = True,
    ) -> ToolExecResult:
        if permission_error := check_admin_permission(context, "File upload/download"):
            return permission_error
        sb = await get_booter(
            context.context.context,
            context.context.event.unified_msg_origin,
        )
        try:
            name = _remote_basename(remote_path) or os.path.basename(remote_path)

            local_path = os.path.join(
                get_astrbot_temp_path(), f"sandbox_{uuid.uuid4().hex[:4]}_{name}"
            )

            # Download file from sandbox
            await sb.download_file(remote_path, local_path)
            logger.info(f"File {remote_path} downloaded from sandbox to {local_path}")

            if also_send_to_user:
                try:
                    name = _remote_basename(remote_path) or os.path.basename(local_path)
                    if Path(local_path).suffix.lower() in _IMAGE_FILE_SUFFIXES:
                        message_component = Image.fromFileSystem(local_path)
                        sent_as = "image"
                    else:
                        message_component = File(name=name, file=local_path)
                        sent_as = "file"
                    await context.context.event.send(
                        MessageChain(chain=[message_component])
                    )
                except Exception as e:
                    logger.error(f"Error sending file message: {e}")
                    return (
                        f"NOT confirmed sent: sending {name} to the user failed "
                        f"({e}). After a timeout it may still have arrived. Do "
                        "not say it was sent; tell the user it may be missing, "
                        "and send it again only if they ask. The copy on the "
                        f"host is {local_path}."
                    )

                # remove
                # try:
                #     os.remove(local_path)
                # except Exception as e:
                #     logger.error(f"Error removing temp file {local_path}: {e}")

                return f"Sent {name} to the user as {sent_as}."

            return f"Copied {remote_path} to the host: {local_path}"
        except Exception as e:
            logger.error(f"Error downloading file {remote_path}: {e}")
            return f"Error downloading file: {str(e)}"
