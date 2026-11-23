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
