# Copyright 2026 Vidip Gaur, Vidu Gaur
"""The repository tools must be able to fail, and must never print a secret."""
import sys

from conftest import REPO

sys.path.insert(0, str(REPO / "tools"))
import audit_paths  # noqa: E402
import scan_secrets  # noqa: E402

# join() rather than + so the compiler cannot fold a full token into the .pyc
FAKE_TOKEN = "".join(["hf_", "Ab3" * 10])


def test_secret_scan_flags_token_without_printing_it(tmp_path, capsys):
    (tmp_path / "a.md").write_text(f"export HF_TOKEN={FAKE_TOKEN}\n")
    assert scan_secrets.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "a.md:1" in out
    assert FAKE_TOKEN not in out
    assert FAKE_TOKEN[:6] in out


def test_secret_scan_ignores_word_internal_sk(tmp_path):
    (tmp_path / "b.md").write_text("see the task-instructions-for-the-run file\n")
    assert scan_secrets.main([str(tmp_path)]) == 0


def test_secret_scan_strict_fails_on_windows_user_path(tmp_path):
    # assembled at run time so this test file does not itself trip the scan
    (tmp_path / "c.py").write_text("".join(['ROOT = r"C:', '\\Users\\someone\\data"\n']))
    assert scan_secrets.main([str(tmp_path)]) == 0
    assert scan_secrets.main(["--strict", str(tmp_path)]) == 1


def test_secret_scan_strict_allows_only_the_published_contacts(tmp_path):
    (tmp_path / "ok.md").write_text("report to gaur.vidip@gmail.com or vidugaur07@gmail.com\n")
    assert scan_secrets.main(["--strict", str(tmp_path)]) == 0
    # assembled at run time so this test file does not itself trip the scan
    (tmp_path / "other.md").write_text("".join(["write to someone.else@", "example.com\n"]))
    assert scan_secrets.main(["--strict", str(tmp_path)]) == 1


def test_path_audit_separates_code_prose_and_windows_branch(tmp_path):
    src = (
        '"""Run from PowerShell."""\n'
        "import os, subprocess\n"
        'ROOT = r"C:\\data\\x"\n'
        "if os.name == \"nt\":\n"
        '    subprocess.run(["tasklist"])\n'
        'p = base + r"\\*.npz"\n'
        'q = "a\\n" + "b"\n'
    )
    f = tmp_path / "s.py"
    f.write_text(src)
    hits = {(c, k) for c, _, k in audit_paths.audit_file(f)}
    assert ("powershell", "prose") in hits
    assert ("windows_drive", "code") in hits
    assert ("windows_cmd", "winbranch") in hits
    assert ("backslash_join", "code") in hits
    assert sum(1 for c, _, _ in audit_paths.audit_file(f) if c == "backslash_join") == 1


def test_path_audit_gate_passes_on_scripts():
    assert audit_paths.main([str(REPO / "scripts")]) == 0
