# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Every non-DEV-ONLY script compiles; library-style scripts import and print
--help with no dataset, no GPU and UBC_ROOT unset.

Scripts without an `if __name__ == "__main__":` guard run their whole body on
import, so importing them is running them. They are compiled but not imported,
and grouped below by what their body needs. Wrapping them in a main() would be
a restructure, which the portability pass does not do.
"""
import ast
import py_compile

import pytest

from conftest import dev_only, script_files, script_path

REQUIRES_DATA_AT_IMPORT = {
    "batch_residual.py", "bt_check_edits.py", "bt_graph_audit.py", "bt_reference.py",
    "check_dims.py", "check_drivers.py", "check_eosin_classes.py", "check_groups.py",
    "check_nuc.py", "check_nuc2.py", "check_tma.py", "cmp_gbm_vs_mil.py",
    "cmp_stagec.py", "cmp_staged.py", "corpus_stats.py", "dump_names.py",
    "excluded.py", "final_ledger.py", "gap.py", "inspect_ckpt.py", "inspect_graph.py",
    "list_stain_cols.py", "paired_clean.py", "phase3_final.py", "phase3_pairs.py",
    "probe_wsi.py", "profile_chrom.py", "profile_eosin.py", "stability_table.py",
    "timeline_stats.py",
}
REQUIRES_NETWORK_AT_IMPORT = {
    "check_fm_env.py", "check_gating.py", "hf_access_check.py", "hibou_load.py",
    "hibou_probe.py", "test_phikon.py", "verify_timm_path.py",
}
# Script-style files whose body is a synthetic self-check; run in
# test_self_checks.py instead of imported.
SELF_CHECKS = {"verify_arch.py", "verify_phase3.py"}

ALL = sorted(p.name for p in script_files() if p.name not in dev_only())
SCRIPT_STYLE = REQUIRES_DATA_AT_IMPORT | REQUIRES_NETWORK_AT_IMPORT | SELF_CHECKS
IMPORTABLE = [n for n in ALL if n not in SCRIPT_STYLE]


def has_main_guard(name):
    tree = ast.parse(script_path(name).read_text(encoding="utf-8"))
    return any(isinstance(n, ast.If) and "__main__" in ast.dump(n.test) for n in tree.body)


def uses_argparse(name):
    return "ArgumentParser" in script_path(name).read_text(encoding="utf-8")


def test_exclusion_lists_are_current():
    unknown = SCRIPT_STYLE - set(ALL)
    assert not unknown, f"listed but not in scripts/: {sorted(unknown)}"
    unguarded = {n for n in ALL if not has_main_guard(n)} - {"corpus_paths.py"}
    assert unguarded == SCRIPT_STYLE, (
        f"unguarded but unlisted: {sorted(unguarded - SCRIPT_STYLE)}; "
        f"listed but guarded: {sorted(SCRIPT_STYLE - unguarded)}")


@pytest.mark.parametrize("name", ALL)
def test_compiles(name, tmp_path):
    py_compile.compile(str(script_path(name)), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.parametrize("name", IMPORTABLE)
def test_imports_without_data(name, run):
    r = run("-c", f"import {name[:-3]}")
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.parametrize("name", [n for n in IMPORTABLE if uses_argparse(n)])
def test_help(name, run):
    r = run(str(script_path(name)), "--help")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage:" in r.stdout
