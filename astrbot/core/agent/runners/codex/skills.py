"""AstrBot skills for the Codex runner.

Codex threads run without a local execution environment by default, so the
model cannot `cat` a SKILL.md. Skills are listed in the prompt the way Codex
lists its own (the format and trigger rules its models are trained on) and
read on demand through the ``astrbot_read_skill`` tool, which only opens files
inside the directory of a skill offered to the current event -- on this host,
or in the sandbox when skills live there (the sandbox runtime).

A skill the user names outright is not left to the model: its SKILL.md is
loaded into that turn, as Codex does for an explicit ``$skill`` mention.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.skills.skill_manager import (
    SANDBOX_SKILLS_ROOT,
    SANDBOX_WORKSPACE_ROOT,
    SkillInfo,
)

READ_SKILL_TOOL = "astrbot_read_skill"
MAX_READ_BYTES = 64 * 1024
# At most this many named skills are loaded into one turn.
MAX_MENTIONED_SKILLS = 3
SKILLS_EXTRA = "_codex_skills"
SKILLS_IN_SANDBOX_EXTRA = "_codex_skills_in_sandbox"

_READ_CALL = f"tools.astrbot__{READ_SKILL_TOOL}({{name}})"

# Adapted from Codex's own skills prompt (codex-rs ext/skills catalog_prompt).
_HOW_TO_USE = f"""\
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR \
the task clearly matches a skill's description above, you must use that skill \
for that turn. Multiple mentions mean use them all. Do not carry skills across \
turns unless re-mentioned.
- How to use a skill: read its `SKILL.md` completely with `{_READ_CALL}` before \
taking task actions; pass `path` (e.g. `scripts/run.py`, `references/`) to open \
a file or folder it references. A skill already loaded into this turn under \
`<request_context name="skill:NAME">` needs no second read.
- Prefer running or adapting a skill's scripts and assets over recreating them. \
Announce which skill you are using in one short line.
- If a skill cannot be read or applied, say so briefly and continue with the \
best fallback."""


def sandbox_skill_dir(name: str) -> str:
    return f"{SANDBOX_WORKSPACE_ROOT}/{SANDBOX_SKILLS_ROOT}/{name}"


def build_codex_skills_prompt(
    skills: list[SkillInfo], *, in_sandbox: bool = False
) -> str:
    """Render the skill list for a Codex turn, in Codex's own format.

    Args:
        skills: Skills offered to the current event.
        in_sandbox: Whether skills are read from the sandbox copy instead of
            this host, which is what the sandbox runtime does.

    Returns:
        Prompt text listing each skill and how to use it.
    """
    lines = []
    for skill in skills:
        description = (skill.description or "Read SKILL.md for details.").strip()
        where = (
            f" (sandbox: {sandbox_skill_dir(skill.name)}/SKILL.md)"
            if in_sandbox
            else ""
        )
        lines.append(f"- {skill.name}: {description}{where}")
    location = (
        "Skill files live in the sandbox, so run a skill's scripts there "
        "(e.g. with exec_command)."
        if in_sandbox
        else "Skill files live on the AstrBot host and are read through the tool."
    )
    return (
        "## Skills\n"
        "A skill is a set of instructions provided through a `SKILL.md` file. "
        f"Below is the list of skills that can be used. {location}\n"
        "### Available skills\n"
        + "\n".join(lines)
        + "\n### How to use skills\n"
        + _HOW_TO_USE
    )


def _relative_path(rel_path: str) -> tuple[str, ...] | None:
    """Parts of a path inside a skill, or None for anything that escapes it."""
    rel = PureWindowsPath(rel_path or "SKILL.md")
    # Reject before touching the filesystem: absolute / drive / UNC paths
    # would otherwise be opened by resolve() (e.g. an outbound SMB request).
    if rel.is_absolute() or rel.drive or rel.anchor or ".." in rel.parts:
        return None
    if PurePosixPath(rel_path or "").is_absolute():
        return None
    return rel.parts


def _skill_dir(skill: SkillInfo) -> Path | None:
    path = Path(skill.path)
    if not skill.local_exists or not path.exists():
        return None
    return path.parent if path.is_file() else path


def _find(skills: list[SkillInfo], name: str) -> SkillInfo | str:
    skill = next((s for s in skills if s.name == name), None)
    if skill is None:
        available = ", ".join(s.name for s in skills) or "none"
        return f"error: unknown skill {name!r}. Available: {available}"
    return skill


def read_skill_file(skills: list[SkillInfo], name: str, rel_path: str = "") -> str:
    """Reads a file or lists a folder of a skill on this host."""
    skill = _find(skills, name)
    if isinstance(skill, str):
        return skill
    base = _skill_dir(skill)
    if base is None:
        return f"error: skill {name!r} is not available on this host."
    parts = _relative_path(rel_path)
    if parts is None:
        return "error: path must be relative to the skill directory."
    target = (base / Path(*parts)).resolve()
    base = base.resolve()
    if target != base and base not in target.parents:
        return "error: path escapes the skill directory."
    if target.is_dir():
        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()
        )
        return f"Directory {target.relative_to(base) or '.'}:\n" + "\n".join(entries)
    if not target.exists():
        return f"error: {rel_path or 'SKILL.md'} not found in skill {name!r}."
    with target.open("rb") as f:
        data = f.read(MAX_READ_BYTES)
    text = data.decode("utf-8", errors="replace")
    if target.stat().st_size > MAX_READ_BYTES:
        text += f"\n...[truncated at {MAX_READ_BYTES} bytes]"
    return text


async def read_sandbox_skill_file(
    booter: Any, skills: list[SkillInfo], name: str, rel_path: str = ""
) -> str:
    """Reads a file or lists a folder of a skill in the sandbox."""
    skill = _find(skills, name)
    if isinstance(skill, str):
        return skill
    parts = _relative_path(rel_path)
    if parts is None or any(part in ("", ".") for part in parts):
        return "error: path must be relative to the skill directory."
    target = "/".join([sandbox_skill_dir(skill.name), *parts])
    quoted = shlex.quote(target)
    # One round trip: a folder is listed, a file is read up to the cap.
    command = (
        f"if [ -d {quoted} ]; then ls -1p {quoted}; "
        f"elif [ -f {quoted} ]; then head -c {MAX_READ_BYTES + 1} {quoted}; "
        "else echo __ASTRBOT_SKILL_MISSING__; fi"
    )
    try:
        result = await booter.shell.exec(command, timeout=30)
    except Exception as e:  # noqa: BLE001 - sandbox unavailable
        return f"error: could not read skill {name!r} from the sandbox: {e}"
    output = str((result or {}).get("stdout") or "")
    if output.strip() == "__ASTRBOT_SKILL_MISSING__":
        return f"error: {rel_path or 'SKILL.md'} not found in skill {name!r}."
    if len(output.encode("utf-8", errors="replace")) > MAX_READ_BYTES:
        output = output[:MAX_READ_BYTES] + f"\n...[truncated at {MAX_READ_BYTES} bytes]"
    return output


async def read_skill(event: Any, context: Any, name: str, rel_path: str = "") -> str:
    """Reads from wherever this event's skills live (host or sandbox)."""
    skills = event.get_extra(SKILLS_EXTRA) or []
    if not isinstance(skills, list):
        return "error: no skills are offered for this message."
    if not event.get_extra(SKILLS_IN_SANDBOX_EXTRA):
        return read_skill_file(skills, name, rel_path)
    from astrbot.core.computer.computer_client import get_booter

    try:
        booter = await get_booter(context, event.unified_msg_origin)
    except Exception as e:  # noqa: BLE001
        return f"error: the sandbox is not available: {e}"
    return await read_sandbox_skill_file(booter, skills, name, rel_path)


def mentioned_skills(text: str, skills: list[SkillInfo]) -> list[SkillInfo]:
    """Skills the message names: `$name`, or the name as a word of its own.

    A plain name only counts when nothing alphanumeric touches it, so `pdf` is
    named in "用 pdf 技能" or "做个pdf" but not in "pdfkit".
    """
    found = []
    for skill in skills:
        name = re.escape(skill.name)
        pattern = rf"(?<![A-Za-z0-9_\-]){name}(?![A-Za-z0-9_\-])"
        if re.search(rf"\${name}(?![A-Za-z0-9_\-])", text) or (
            len(skill.name) >= 3 and re.search(pattern, text, re.IGNORECASE)
        ):
            found.append(skill)
    return found[:MAX_MENTIONED_SKILLS]


async def load_mentioned_skills(event: Any, context: Any, req: Any) -> list[str]:
    """Loads the SKILL.md of each skill the user named into this turn.

    Returns the names loaded. A skill that cannot be read is left to the model,
    which the prompt tells to read it.
    """
    skills = event.get_extra(SKILLS_EXTRA) or []
    if not isinstance(skills, list) or not req.prompt:
        return []
    loaded = []
    for skill in mentioned_skills(req.prompt, skills):
        body = await read_skill(event, context, skill.name)
        if body.startswith("error:"):
            continue
        req.add_temporary_context(
            f"skill:{skill.name}",
            f"The user named the `{skill.name}` skill; its SKILL.md follows. "
            "Follow it for this turn.\n\n" + body,
        )
        loaded.append(skill.name)
    return loaded


@dataclass
class ReadSkillTool(FunctionTool):
    name: str = READ_SKILL_TOOL
    description: str = (
        "Read a skill's SKILL.md, or a file / folder inside that skill "
        "(relative `path`, e.g. `scripts/run.py`). Works wherever the skills "
        "live, sandbox included."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name."},
                "path": {
                    "type": "string",
                    "description": "File or folder inside the skill; default SKILL.md.",
                },
            },
            "required": ["name"],
        }
    )

    async def call(self, context: ContextWrapper, **kwargs: Any) -> str:
        agent_ctx = context.context
        return await read_skill(
            agent_ctx.event,  # type: ignore[attr-defined]
            agent_ctx.context,  # type: ignore[attr-defined]
            str(kwargs.get("name") or ""),
            str(kwargs.get("path") or ""),
        )
