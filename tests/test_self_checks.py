# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Synthetic self-checks that ship with the scripts, run as tests."""
from conftest import script_path


def test_graph_builder_self_test(run):
    r = run(str(script_path("build_streamline_graphs_bridge.py")), "--self-test", timeout=900)
    print(r.stdout[-6000:])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "NOT RAISED" not in r.stdout


def test_verify_arch(run):
    # Prints PASS/FAIL per check without setting the exit status.
    r = run(str(script_path("verify_arch.py")), timeout=600)
    print(r.stdout[-4000:])
    assert r.returncode == 0, r.stderr[-3000:]
    assert "FAIL" not in r.stdout
    assert "PASS" in r.stdout


def test_verify_phase3(run):
    r = run(str(script_path("verify_phase3.py")), timeout=600)
    print(r.stdout[-4000:])
    assert r.returncode == 0, r.stderr[-3000:]
    assert "FAIL" not in r.stdout
