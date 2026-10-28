# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
# Copyright 2026 Vidip Gaur, Vidu Gaur
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_abmil_nuclear import bags_for            # noqa: E402
from train_abmil import resolve_groups                # noqa: E402
import corpus_paths  # noqa: E402

X, y, ids, fn = bags_for(
    str(corpus_paths.ROOT), "v5nucchromeos_stainprobe", True,
    chromatin=True, eosin=True)

g = resolve_groups(fn)
print("total cols:", len(fn))
for k in ("chrom_context", "chrom_level", "eos_other", "eos_clear"):
    print(f"\n{k} ({len(g.get(k, []))}):")
    for i in g.get(k, []):
        print("   ", fn[i])

print("\nall tile_* columns:")
for i, f in enumerate(fn):
    if f.startswith("tile_"):
        print(f"  {i:3d} {f}")
