from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, ImageContent
from PIL import Image

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.computer import file_read_utils
from astrbot.core.tools.computer_tools import fs as fs_tools


def _make_sandbox_context(
    *,
    role: str = "admin",
    umo: str = "qq:friend:user-1",
):
    config_holder = SimpleNamespace(
        get_config=lambda umo=None: {
            "provider_settings": {
                "computer_use_require_admin": True,
                "computer_use_runtime": "sandbox",
            }
        }
    )
    event = SimpleNamespace(
        role=role,
        unified_msg_origin=umo,
        send=AsyncMock(),
    )
    astr_ctx = SimpleNamespace(context=config_holder, event=event)
    return ContextWrapper(context=astr_ctx)


@pytest.mark.asyncio
async def test_sandbox_file_download_handles_windows_remote_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    temp_root = tmp_path / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        fs_tools,
        "get_astrbot_temp_path",
        lambda: str(temp_root),
    )

    async def _download_file(_remote_path, local_path):
        # The local temp path keeps the original remote basename; separators
        # are platform-native, so only the basename is asserted here.
        assert local_path.endswith("report.txt")

    booter = SimpleNamespace(download_file=AsyncMock(side_effect=_download_file))

    async def _fake_get_booter(_ctx, _umo):
        return booter

    monkeypatch.setattr(fs_tools, "get_booter", _fake_get_booter)

    context = _make_sandbox_context()
    result = await fs_tools.FileDownloadTool().call(
        context,
        remote_path=r"C:\Users\AstrBot\report.txt",
        also_send_to_user=True,
    )

    assert "report.txt" in result
    sent_chain = context.context.event.send.await_args.args[0]
    sent_file = sent_chain.chain[0]
    assert sent_file.name == "report.txt"


@pytest.mark.asyncio
async def test_sandbox_file_download_strips_trailing_remote_slash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    temp_root = tmp_path / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        fs_tools,
        "get_astrbot_temp_path",
        lambda: str(temp_root),
    )

    booter = SimpleNamespace(download_file=AsyncMock())

    async def _fake_get_booter(_ctx, _umo):
        return booter

    monkeypatch.setattr(fs_tools, "get_booter", _fake_get_booter)

    context = _make_sandbox_context()
    result = await fs_tools.FileDownloadTool().call(
        context,
        remote_path="reports/export/",
        also_send_to_user=True,
    )

    assert "export" in result
    sent_chain = context.context.event.send.await_args.args[0]
    sent_file = sent_chain.chain[0]
    assert sent_file.name == "export"


class _HeredocPythonShell:
    """Runs the python heredoc a sandbox shell command carries, locally."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, **_kwargs: Any) -> dict[str, Any]:
        import asyncio
        import subprocess
        import sys

        self.commands.append(command)
        script = command.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-"],
            input=script,
            capture_output=True,
            text=True,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "success": proc.returncode == 0,
        }


@pytest.mark.asyncio
async def test_sandbox_file_read_runs_python_through_shell(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")
    shell = _HeredocPythonShell()
    python = SimpleNamespace(exec=AsyncMock())
    booter = SimpleNamespace(shell=shell, python=python)

    result = await file_read_utils.read_file_tool_result(
        booter,
        path=str(target),
        offset=1,
        limit=1,
    )

    assert "line2" in str(result)
    assert "line1" not in str(result)
    assert len(shell.commands) == 2
    python.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_sandbox_file_read_reports_shell_failure():
    shell = SimpleNamespace(
        exec=AsyncMock(
            return_value={
                "stdout": "",
                "stderr": "FileNotFoundError: missing.txt",
                "exit_code": 1,
                "success": False,
            }
        )
    )

    with pytest.raises(RuntimeError, match="file probe failed: FileNotFoundError"):
        await file_read_utils.read_file_tool_result(
            SimpleNamespace(shell=shell),
            path="missing.txt",
            offset=None,
            limit=None,
        )


class _MemoryFs:
    """Sandbox filesystem stand-in that keeps files in memory."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.search_calls: list[dict[str, Any]] = []
        self.search_content = ""

    async def write_file(self, path: str, content: str, **_kwargs: Any) -> dict:
        self.files[path] = content
        return {"success": True}

    async def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **_kwargs: Any,
    ) -> dict:
        content = self.files[path]
        count = content.count(old_string)
        if not count:
            return {"success": False, "error": "old string not found"}
        self.files[path] = content.replace(
            old_string, new_string, -1 if replace_all else 1
        )
        return {"success": True, "replacements": count if replace_all else 1}

    async def search_files(self, **kwargs: Any) -> dict:
        self.search_calls.append(kwargs)
        return {"success": True, "content": self.search_content}


def _use_booter(monkeypatch: pytest.MonkeyPatch, booter: Any) -> None:
    async def _fake_get_booter(_ctx, _umo):
        return booter

    monkeypatch.setattr(fs_tools, "get_booter", _fake_get_booter)


def _use_heredoc_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> _HeredocPythonShell:
    shell = _HeredocPythonShell()
    _use_booter(monkeypatch, SimpleNamespace(shell=shell))
    monkeypatch.setattr(
        file_read_utils, "get_astrbot_temp_path", lambda: str(tmp_path / "temp")
    )
    return shell


def test_detect_text_encoding_allows_utf8_probe_cut_mid_character():
    sample = '{"results": ["中文内容"]}'.encode()[:-1]

    assert file_read_utils.detect_text_encoding(sample) in {"utf-8", "utf-8-sig"}


@pytest.mark.asyncio
async def test_file_read_tool_rejects_large_full_text_read_before_reading_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    shell = _use_heredoc_shell(monkeypatch, tmp_path)
    large_file = tmp_path / "large.txt"
    large_file.write_text(
        "".join(f"line-{index:05d}-{'x' * 48}\n" for index in range(6000)),
        encoding="utf-8",
    )

    result = await fs_tools.FileReadTool().call(
        _make_sandbox_context(), path=str(large_file)
    )

    assert "text file exceeds 262144 bytes" in result
    assert "Use `offset` and `limit`" in result
    # Only the probe ran; the full text was never read.
    assert len(shell.commands) == 1


@pytest.mark.asyncio
async def test_file_read_tool_allows_partial_read_for_large_text_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    _use_heredoc_shell(monkeypatch, tmp_path)
    large_file = tmp_path / "large.txt"
    lines = [f"line-{index:05d}\n" for index in range(50000)]
    large_file.write_text("".join(lines), encoding="utf-8")

    result = await fs_tools.FileReadTool().call(
        _make_sandbox_context(),
        path=str(large_file),
        offset=1000,
        limit=3,
    )

    assert result == "".join(lines[1000:1003])


@pytest.mark.asyncio
async def test_file_read_tool_returns_image_call_tool_result_for_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    _use_heredoc_shell(monkeypatch, tmp_path)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 16), color=(255, 0, 0)).save(image_path, format="PNG")

    result = await fs_tools.FileReadTool().call(
        _make_sandbox_context(), path=str(image_path)
    )

    assert isinstance(result, CallToolResult)
    assert len(result.content) == 1
    assert isinstance(result.content[0], ImageContent)
    assert result.content[0].mimeType == "image/jpeg"
    assert base64.b64decode(result.content[0].data).startswith(b"\xff\xd8\xff")


@pytest.mark.asyncio
async def test_file_read_tool_treats_svg_as_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    _use_heredoc_shell(monkeypatch, tmp_path)
    svg_path = tmp_path / "shape.svg"
    svg_text = (
        "<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>"
    )
    svg_path.write_text(svg_text, encoding="utf-8")

    result = await fs_tools.FileReadTool().call(
        _make_sandbox_context(), path=str(svg_path)
    )

    assert result == svg_text


@pytest.mark.asyncio
async def test_file_read_tool_rejects_binary_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    _use_heredoc_shell(monkeypatch, tmp_path)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n")

    result = await fs_tools.FileReadTool().call(
        _make_sandbox_context(), path=str(pdf_path)
    )

    assert result == "Error reading file: binary files are not supported by this tool."


@pytest.mark.asyncio
async def test_grep_tool_applies_result_limit(monkeypatch: pytest.MonkeyPatch):
    fs = _MemoryFs()
    fs.search_content = "grep.txt:match-1\ngrep.txt:match-2\ngrep.txt:match-3\n"
    _use_booter(monkeypatch, SimpleNamespace(fs=fs))

    result = await fs_tools.GrepTool().call(
        _make_sandbox_context(),
        pattern="match",
        path=" grep.txt ",
        result_limit=2,
    )

    assert "match-1" in result
    assert "match-2" in result
    assert "match-3" not in result
    assert "[Truncated to first 2 result groups.]" in result
    assert fs.search_calls[0]["path"] == "grep.txt"


@pytest.mark.asyncio
async def test_grep_tool_searches_the_working_directory_without_a_path(
    monkeypatch: pytest.MonkeyPatch,
):
    fs = _MemoryFs()
    _use_booter(monkeypatch, SimpleNamespace(fs=fs))

    result = await fs_tools.GrepTool().call(_make_sandbox_context(), pattern="x")

    assert result == "No matches found."
    assert fs.search_calls[0]["path"] == "."


@pytest.mark.asyncio
async def test_file_edit_matches_source_that_contains_a_literal_escape(
    monkeypatch: pytest.MonkeyPatch,
):
    """A backslash-n inside source code is content, not a line break."""
    fs = _MemoryFs()
    _use_booter(monkeypatch, SimpleNamespace(fs=fs))
    source = 'def greet():\n    print("hello\\n")\n'
    write_result = await fs_tools.FileWriteTool().call(
        _make_sandbox_context(),
        path="greet.py",
        content=source,
    )

    edit_result = await fs_tools.FileEditTool().call(
        _make_sandbox_context(),
        path="greet.py",
        old='print("hello\\n")',
        new='print("goodbye\\n")',
    )

    assert "File written successfully" in write_result
    assert "Replaced 1 occurrence" in edit_result
    assert fs.files["greet.py"] == 'def greet():\n    print("goodbye\\n")\n'


@pytest.mark.asyncio
async def test_file_edit_inserts_a_literal_escape_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    """The replacement keeps the escape the model wrote when `old` matched as given."""
    fs = _MemoryFs()
    fs.files["paths.py"] = 'LOG = "PLACEHOLDER"\n'
    _use_booter(monkeypatch, SimpleNamespace(fs=fs))

    result = await fs_tools.FileEditTool().call(
        _make_sandbox_context(),
        path="paths.py",
        old="PLACEHOLDER",
        new="C:\\new\\temp\\report.txt",
    )

    assert "Replaced 1 occurrence" in result
    assert fs.files["paths.py"] == 'LOG = "C:\\new\\temp\\report.txt"\n'


@pytest.mark.asyncio
async def test_file_edit_still_decodes_escapes_when_the_text_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
):
    """A model that escapes its arguments keeps working: the decode is a fallback."""
    fs = _MemoryFs()
    fs.files["notes.txt"] = "alpha\nbeta\n"
    _use_booter(monkeypatch, SimpleNamespace(fs=fs))

    result = await fs_tools.FileEditTool().call(
        _make_sandbox_context(),
        path="notes.txt",
        old="alpha\\nbeta",
        new="alpha\\ngamma",
    )

    assert "Replaced 1 occurrence" in result
    assert fs.files["notes.txt"] == "alpha\ngamma\n"
