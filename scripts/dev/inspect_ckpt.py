# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
# Copyright 2026 Vidip Gaur, Vidu Gaur
import glob
import os

import torch
import corpus_paths

C = str(corpus_paths.ROOT / "cellvit_run" / "graphs" / "gatv2_run" / "abmil_ckpt")

for arm in ("v5nucchromeos_clean_final", "v5fmuni2h_fm_addmil",
            "v5fmphikonv2_fm_addmil", "v5fmhibouL_fm_addmil",
            "v5fmhoptimus0_fm_addmil"):
    fs = sorted(glob.glob(os.path.join(C, arm, "*")))
    print("=" * 78)
    print(f"{arm}   ({len(fs)} files)")
    if not fs:
        continue
    print("  example filenames:", [os.path.basename(f) for f in fs[:3]])
    d = torch.load(fs[0], map_location="cpu", weights_only=False)
    print("  ckpt keys :", sorted(d.keys()))
    print("  in_dim    :", d["in_dim"])
    print("  classes   :", d["classes"])
    print("  config    :", d["config"])
    print("  select    :", d.get("select"), "|", d.get("select_detail"))
    n = d["normaliser"]
    print("  norm mean shape:", tuple(n["mean"].shape),
          " n log_idx:", len(n["log_idx"]))
    fn = d["feat_names"]
    print("  n feat_names:", len(fn))
    if len(fn) <= 60:
        for i in range(0, len(fn), 4):
            print("      " + "  ".join(f"{x:<24s}" for x in fn[i:i + 4]))
    else:
        print("      first 8:", fn[:8])
        print("      last  8:", fn[-8:])
    print("  model params:", [(k, tuple(v.shape))
                              for k, v in list(d["model"].items())[:4]])
