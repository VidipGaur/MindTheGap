# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
# Copyright 2026 Vidip Gaur, Vidu Gaur
import collections
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_abmil_nuclear import bags_for            # noqa: E402
from train_abmil import resolve_groups                # noqa: E402
import corpus_paths  # noqa: E402

X, y, ids, fn = bags_for(
    str(corpus_paths.ROOT), "v5nucchromeos_clean_final", True,
    drop_groups=["graph", "chrom_texture", "eos_other", "chrom_context"],
    exclude_prefix=["dead_"], chromatin=True, eosin=True)

g = resolve_groups(fn)
print("n_cols:", len(fn))
print("resolve_groups returned:", type(g))

if isinstance(g, dict):
    c = collections.Counter()
    for k, v in g.items():
        c[k] = len(v) if hasattr(v, "__len__") else 1
    for k, v in c.most_common():
        print(f"  {k:32s} {v}")
    ctx = g.get("chrom_context")
    print("\nchrom_context members:", ctx)
else:
    c = collections.Counter(g)
    for k, v in c.most_common():
        print(f"  {str(k):32s} {v}")

print("\ndead_ columns still present:", [n for n in fn if n.startswith("dead_")])
print("ctx-looking columns:", [n for n in fn if "tissue" in n or "ctx" in n
                               or n.startswith("tile_")][:20])
