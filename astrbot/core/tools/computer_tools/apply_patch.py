"""Codex-style ``apply_patch`` over AstrBot's execution environment.

Codex models are trained on this patch format, so offering it (next to the
plain edit tool) makes file edits in the sandbox or local workspace reliable:

    *** Begin Patch
    *** Add File: notes/todo.md
    +first line
    *** Update File: src/app.py
    *** Move to: src/main.py
    @@ def handler():
    -    return 1
    +    return 2
    *** Delete File: old.txt
    *** End Patch
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astrbot.api import FunctionTool, logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.computer.computer_client import get_booter

from ..registry import builtin_tool
from .fs import _is_restricted_env, _normalize_rw_path
from .util import is_local_runtime, workspace_root_for_context

_COMPUTER_RUNTIME_TOOL_CONFIG = {
    "provider_settings.computer_use_runtime": ("local", "sandbox"),
}

BEGIN = "*** Begin Patch"
END = "*** End Patch"
ADD = "*** Add File: "
UPDATE = "*** Update File: "
DELETE = "*** Delete File: "
MOVE = "*** Move to: "
EOF_MARK = "*** End of File"


class PatchError(ValueError):
    pass


@dataclass
class Hunk:
    context: str = ""
    old: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)


@dataclass
class FileOp:
    kind: str  # add | update | delete
    path: str
    move_to: str = ""
    add_lines: list[str] = field(default_factory=list)
    hunks: list[Hunk] = field(default_factory=list)


def parse_patch(text: str) -> list[FileOp]:
    lines = text.replace("\r\n", "\n").strip("\n").split("\n")
    if not lines or lines[0].strip() != BEGIN or lines[-1].strip() != END:
        raise PatchError(
            "patch must start with '*** Begin Patch' and end with '*** End Patch'"
        )
    ops: list[FileOp] = []
    body = lines[1:-1]
    i = 0
    while i < len(body):
        line = body[i]
        if line.startswith(ADD):
            op = FileOp("add", line[len(ADD) :].strip())
            i += 1
            while i < len(body) and not body[i].startswith("*** "):
                if not body[i].startswith("+"):
                    raise PatchError(
                        f"added file lines must start with '+': {body[i]!r}"
                    )
                op.add_lines.append(body[i][1:])
                i += 1
            ops.append(op)
        elif line.startswith(DELETE):
            ops.append(FileOp("delete", line[len(DELETE) :].strip()))
            i += 1
        elif line.startswith(UPDATE):
            op = FileOp("update", line[len(UPDATE) :].strip())
            i += 1
            if i < len(body) and body[i].startswith(MOVE):
                op.move_to = body[i][len(MOVE) :].strip()
                i += 1
            hunk: Hunk | None = None
            while i < len(body) and not (
                body[i].startswith("*** ") and body[i].strip() != EOF_MARK
            ):
                cur = body[i]
                if cur.startswith("@@"):
                    hunk = Hunk(context=cur[2:].strip())
                    op.hunks.append(hunk)
                elif cur.strip() == EOF_MARK:
                    pass
                else:
                    if hunk is None:
                        hunk = Hunk()
                        op.hunks.append(hunk)
                    tag, rest = (cur[:1], cur[1:]) if cur else (" ", "")
                    if tag == " ":
                        hunk.old.append(rest)
                        hunk.new.append(rest)
                    elif tag == "-":
                        hunk.old.append(rest)
                    elif tag == "+":
                        hunk.new.append(rest)
                    else:
                        raise PatchError(f"invalid hunk line: {cur!r}")
                i += 1
            if not op.hunks and not op.move_to:
                raise PatchError(f"update of {op.path} has no hunks")
            ops.append(op)
        elif not line.strip():
            i += 1
        else:
            raise PatchError(f"unexpected line: {line!r}")
    if not ops:
        raise PatchError("patch contains no file operations")
    return ops


def _find(lines: list[str], needle: list[str], start: int) -> int:
    if not needle:
        return start
    for strip in (False, True):
        for idx in range(start, len(lines) - len(needle) + 1):
            window = lines[idx : idx + len(needle)]
            if strip:
                if [w.rstrip() for w in window] == [n.rstrip() for n in needle]:
                    return idx
            elif window == needle:
                return idx
    return -1


def apply_hunks(content: str, hunks: list[Hunk]) -> str:
    trailing_newline = content.endswith("\n")
    lines = content.split("\n")
    if trailing_newline:
        lines = lines[:-1]
    cursor = 0
    for hunk in hunks:
        if hunk.context:
            ctx_idx = _find(lines, [hunk.context], cursor)
            if ctx_idx >= 0:
                cursor = ctx_idx
        pos = _find(lines, hunk.old, cursor)
        if pos < 0:
            pos = _find(lines, hunk.old, 0)
        if pos < 0:
            preview = "\n".join(hunk.old[:3])
            raise PatchError(f"could not locate hunk:\n{preview}")
        lines[pos : pos + len(hunk.old)] = hunk.new
        cursor = pos + len(hunk.new)
    out = "\n".join(lines)
    return out + "\n" if trailing_newline or not content else out


@builtin_tool(config=_COMPUTER_RUNTIME_TOOL_CONFIG)
@dataclass
class ApplyPatchTool(FunctionTool):
    name: str = "apply_patch"
    description: str = (
        "Edit files in the execution environment with a Codex-style patch "
        "(*** Begin Patch / *** Add File: / *** Update File: / *** Move to: / "
        "*** Delete File: / @@ hunks / *** End Patch). Paths are relative to the workspace."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "The full patch text."}
            },
            "required": ["patch"],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], patch: str) -> str:
        try:
            ops = parse_patch(patch)
        except PatchError as e:
            return f"Error: invalid patch: {e}"
        event = context.context.event
        umo = str(event.unified_msg_origin)
        local_env = is_local_runtime(context)
        restricted = _is_restricted_env(context)
        root = await workspace_root_for_context(context) if local_env else None

        def norm(path: str) -> str:
            if not local_env:
                return path.strip()
            return _normalize_rw_path(
                path,
                restricted=restricted,
                local_env=True,
                umo=umo,
                write=True,
                current_workspace_root=root,
            )

        try:
            sb = await get_booter(context.context.context, event.unified_msg_origin)
            done: list[str] = []
            for op in ops:
                path = norm(op.path)
                if op.kind == "add":
                    content = "\n".join(op.add_lines) + "\n"
                    result = await sb.fs.write_file(path=path, content=content)
                    done.append(f"A {op.path}")
                elif op.kind == "delete":
                    result = await sb.fs.delete_file(path)
                    done.append(f"D {op.path}")
                else:
                    read = await sb.fs.read_file(path=path)
                    if not read.get("success", True) and "content" not in read:
                        return f"Error: cannot read {op.path}: {read.get('error')}"
                    new_content = apply_hunks(str(read.get("content") or ""), op.hunks)
                    target = norm(op.move_to) if op.move_to else path
                    result = await sb.fs.write_file(path=target, content=new_content)
                    if op.move_to and target != path:
                        await sb.fs.delete_file(path)
                    done.append(
                        f"M {op.path}" + (f" -> {op.move_to}" if op.move_to else "")
                    )
                if isinstance(result, dict) and result.get("success") is False:
                    return f"Error applying patch at {op.path}: {result.get('error')}"
            return "Patch applied:\n" + "\n".join(done)
        except PatchError as e:
            return f"Error: {e}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:  # noqa: BLE001
            logger.error("apply_patch failed: %s", e)
            return f"Error applying patch: {e}"
