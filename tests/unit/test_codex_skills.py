"""Codex runner skills: Codex-style listing, one reader, named skills loaded."""

from types import SimpleNamespace

import pytest

from astrbot.core.agent.runners.codex import skills as skills_mod
from astrbot.core.agent.runners.codex.skills import (
    SKILLS_EXTRA,
    SKILLS_IN_SANDBOX_EXTRA,
    ReadSkillTool,
    build_codex_skills_prompt,
    load_mentioned_skills,
    mentioned_skills,
    read_sandbox_skill_file,
)
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.skills.skill_manager import SkillInfo


def _skill(name, description="d", path="x/SKILL.md", **kw):
    return SkillInfo(name=name, description=description, path=path, active=True, **kw)


# ------------------------------------------------------------ the listing


def test_the_listing_uses_codex_trigger_rules():
    prompt = build_codex_skills_prompt([_skill("pdf", "Make PDFs")])

    assert prompt.startswith("## Skills")
    assert "### Available skills\n- pdf: Make PDFs" in prompt
    # The rule that makes the model reach for a matching skill on its own.
    assert "you must use that skill" in prompt
    assert "before taking task actions" in prompt


def test_sandbox_skills_are_listed_with_their_sandbox_location():
    prompt = build_codex_skills_prompt([_skill("pdf")], in_sandbox=True)

    assert "- pdf: d (sandbox: /workspace/skills/pdf/SKILL.md)" in prompt
    assert "tools.astrbot__astrbot_read_skill" in prompt


# ------------------------------------------------------------ sandbox reads


class _Shell:
    def __init__(self, files):
        self.files = files
        self.commands = []

    async def exec(self, command, timeout=None, **kw):
        self.commands.append(command)
        for path, body in self.files.items():
            if f"'{path}'" in command or f" {path};" in command:
                return {"stdout": body, "exit_code": 0}
        return {"stdout": "__ASTRBOT_SKILL_MISSING__\n", "exit_code": 0}


def _booter(files):
    return SimpleNamespace(shell=_Shell(files))


@pytest.mark.asyncio
async def test_a_sandbox_skill_is_read_from_the_sandbox():
    booter = _booter({"/workspace/skills/pdf/SKILL.md": "BODY"})

    body = await read_sandbox_skill_file(booter, [_skill("pdf")], "pdf")

    assert body == "BODY"


@pytest.mark.asyncio
async def test_a_missing_sandbox_file_is_an_error():
    body = await read_sandbox_skill_file(_booter({}), [_skill("pdf")], "pdf", "x.py")

    assert body.startswith("error: x.py not found")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["../x", "/etc/passwd", r"C:\x", r"\\host\share"])
async def test_sandbox_reads_stay_inside_the_skill(bad):
    booter = _booter({})

    body = await read_sandbox_skill_file(booter, [_skill("pdf")], "pdf", bad)

    assert body.startswith("error: path must be relative")
    assert booter.shell.commands == []


@pytest.mark.asyncio
async def test_an_unknown_skill_is_not_read():
    booter = _booter({})

    body = await read_sandbox_skill_file(booter, [_skill("pdf")], "rm -rf")

    assert body.startswith("error: unknown skill")
    assert booter.shell.commands == []


@pytest.mark.asyncio
async def test_the_path_is_quoted_for_the_shell():
    booter = _booter({})
    await read_sandbox_skill_file(booter, [_skill("pdf")], "pdf", "a b;rm x")

    assert "'/workspace/skills/pdf/a b;rm x'" in booter.shell.commands[0]


# ------------------------------------------------------------ named skills


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("用 $pdf 帮我导出", ["pdf"]),
        ("做个pdf文件", ["pdf"]),
        ("use the PDF skill", ["pdf"]),
        ("install pdfkit", []),
        ("my-pdf-tool", []),
        ("hello", []),
    ],
)
def test_a_skill_is_named_by_dollar_or_as_a_word(text, names):
    found = mentioned_skills(text, [_skill("pdf"), _skill("docx")])

    assert [s.name for s in found] == names


def test_short_names_count_only_with_a_dollar():
    skills = [_skill("go")]

    assert mentioned_skills("let's go", skills) == []
    assert [s.name for s in mentioned_skills("$go now", skills)] == ["go"]


def _event(skills, in_sandbox=False):
    extras = {SKILLS_EXTRA: skills, SKILLS_IN_SANDBOX_EXTRA: in_sandbox}
    return SimpleNamespace(
        get_extra=lambda key, default=None: extras.get(key, default),
        unified_msg_origin="qq:GroupMessage:g",
    )


@pytest.mark.asyncio
async def test_a_named_skill_is_loaded_into_the_turn(tmp_path):
    (tmp_path / "pdf").mkdir()
    (tmp_path / "pdf" / "SKILL.md").write_text("PDF STEPS", encoding="utf-8")
    skills = [_skill("pdf", path=str(tmp_path / "pdf" / "SKILL.md"))]
    req = ProviderRequest(prompt="用 $pdf 导出一下")

    loaded = await load_mentioned_skills(_event(skills), None, req)

    assert loaded == ["pdf"]
    [part] = req.dynamic_user_context_parts
    assert 'name="skill:pdf"' in part.text
    assert "PDF STEPS" in part.text


@pytest.mark.asyncio
async def test_a_named_sandbox_skill_is_loaded_from_the_sandbox(monkeypatch):
    booter = _booter({"/workspace/skills/pdf/SKILL.md": "SANDBOX STEPS"})

    async def get_booter(context, umo):
        return booter

    import astrbot.core.computer.computer_client as computer_client

    monkeypatch.setattr(computer_client, "get_booter", get_booter)
    req = ProviderRequest(prompt="$pdf please")

    loaded = await load_mentioned_skills(
        _event([_skill("pdf", local_exists=False)], in_sandbox=True), object(), req
    )

    assert loaded == ["pdf"]
    assert "SANDBOX STEPS" in req.dynamic_user_context_parts[0].text


@pytest.mark.asyncio
async def test_an_unreadable_named_skill_is_left_to_the_model():
    req = ProviderRequest(prompt="$pdf please")

    loaded = await load_mentioned_skills(
        _event([_skill("pdf", path="missing/SKILL.md")]), None, req
    )

    assert loaded == []
    assert req.dynamic_user_context_parts == []


@pytest.mark.asyncio
async def test_the_tool_reads_from_the_sandbox_with_the_sandbox_runtime(monkeypatch):
    booter = _booter({"/workspace/skills/pdf/SKILL.md": "SANDBOX STEPS"})

    async def get_booter(context, umo):
        return booter

    import astrbot.core.computer.computer_client as computer_client

    monkeypatch.setattr(computer_client, "get_booter", get_booter)
    wrapper = SimpleNamespace(
        context=SimpleNamespace(
            event=_event([_skill("pdf")], in_sandbox=True), context=object()
        )
    )

    assert await ReadSkillTool().call(wrapper, name="pdf") == "SANDBOX STEPS"


def test_the_module_keeps_the_legacy_extra_name():
    # codex_request and astr_main_agent share this key.
    assert skills_mod.SKILLS_EXTRA == "_codex_skills"
