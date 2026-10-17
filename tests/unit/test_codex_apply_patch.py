import pytest

from astrbot.core.tools.computer_tools.apply_patch import (
    PatchError,
    apply_hunks,
    parse_patch,
)

PATCH = """*** Begin Patch
*** Add File: notes.md
+hello
+world
*** Update File: app.py
*** Move to: main.py
@@ def handler():
-    return 1
+    return 2
*** Delete File: old.txt
*** End Patch"""


def test_parse_patch_ops():
    ops = parse_patch(PATCH)
    assert [o.kind for o in ops] == ["add", "update", "delete"]
    assert ops[0].add_lines == ["hello", "world"]
    assert ops[1].move_to == "main.py"
    assert ops[1].hunks[0].context == "def handler():"
    assert ops[1].hunks[0].old == ["    return 1"]


def test_apply_hunks_with_context_and_whitespace_tolerance():
    src = "x = 0\ndef handler():\n    return 1\n"
    ops = parse_patch(PATCH)
    assert apply_hunks(src, ops[1].hunks) == "x = 0\ndef handler():\n    return 2\n"
    assert apply_hunks("def handler():\n    return 1   \n", ops[1].hunks).endswith(
        "return 2\n"
    )


def test_invalid_patches():
    with pytest.raises(PatchError):
        parse_patch("*** Update File: a\n")
    with pytest.raises(PatchError):
        apply_hunks("nothing here\n", parse_patch(PATCH)[1].hunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["member", "admin"])
async def test_a_role_without_local_file_access_cannot_patch(
    monkeypatch, tmp_path, role
):
    """Filesystem scope "none" refuses apply_patch like the file tools."""
    from types import SimpleNamespace

    from astrbot.core.tools.computer_tools import apply_patch

    target = tmp_path / "victim.txt"
    config = {
        "admins_id": ["1"] if role == "admin" else [],
        "provider_settings": {
            "computer_use_runtime": "local",
            "computer_use_local_permissions": {
                role: {
                    "allow_execution": False,
                    "allow_network": False,
                    "filesystem_scope": "none",
                }
            },
        },
    }
    event = SimpleNamespace(
        unified_msg_origin="qq:GroupMessage:g",
        get_sender_id=lambda: "1",
        role=role,
    )
    ctx = SimpleNamespace(
        context=SimpleNamespace(
            event=event, context=SimpleNamespace(get_config=lambda umo: config)
        )
    )

    async def no_booter(*_args):
        raise AssertionError("a refused patch must not reach the booter")

    monkeypatch.setattr(apply_patch, "get_booter", no_booter)
    patch = f"*** Begin Patch\n*** Add File: {target}\n+owned\n*** End Patch"

    result = await apply_patch.ApplyPatchTool().call(ctx, patch)

    assert "Permission denied" in result
    assert not target.exists()
