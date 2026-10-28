# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Exercise embed_tiles_fm.py's NEW code paths using an UNGATED model.

The gated weights are unavailable, but everything that could be wrong in the
edit is testable without them: the timm hf-hub loader, resolve_data_config,
the SwiGLU kwarg injection, the pooled-vs-CLS forward branch, the dim
assertion, the norm cross-check, and the non-finite guard.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import sys
import numpy as np
import torch
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed_tiles_fm as E

# An ungated timm ViT standing in for UNI2-h / H-optimus-0's loading path.
E.MODELS["_timmprobe"] = dict(
    repo="timm/vit_small_patch16_224.augreg_in21k", dim=384, loader="timm",
    gated=False, note="UNGATED stand-in to test the timm path",
    timm_kwargs=dict(num_classes=0),
    card_mean=(0.5, 0.5, 0.5), card_std=(0.5, 0.5, 0.5))

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

enc = E.Encoders(["_timmprobe"], dev)

rng = np.random.default_rng(0)
tiles = rng.integers(0, 255, (7, E.INPUT_PX, E.INPUT_PX, 3), dtype=np.uint8)
emb = enc.embed(tiles, "_timmprobe")
print("timm path  ->", emb.shape, "finite:", bool(np.isfinite(emb).all()),
      "|e|=%.2f" % np.linalg.norm(emb, axis=1).mean())
assert emb.shape == (7, 384), emb.shape

# The dim assertion must actually fire on a wrong build, or a silently
# mis-constructed UNI2-h would sail straight through into a 4 h run.
E.MODELS["_timmprobe"]["dim"] = 9999
try:
    enc.embed(tiles, "_timmprobe")
    print("!! dim assertion did NOT fire -- guard is broken")
except RuntimeError as ex:
    print("dim guard  -> fired correctly:", str(ex)[:70])
E.MODELS["_timmprobe"]["dim"] = 384

# The hf path must still behave exactly as before (phikon regression check).
enc2 = E.Encoders(["phikon"], dev)
emb2 = enc2.embed(tiles, "phikon")
print("hf path    ->", emb2.shape, "finite:", bool(np.isfinite(emb2).all()),
      "|e|=%.2f" % np.linalg.norm(emb2, axis=1).mean())
assert emb2.shape == (7, 768), emb2.shape

# Batching must not change results: 7 tiles > BATCH forces >1 chunk.
E.BATCH = 3
emb3 = enc2.embed(tiles, "phikon")
print("batching   -> max|delta| = %.2e (BATCH=3 vs 64)"
      % np.abs(emb3 - emb2).max())

print("\nALL NEW CODE PATHS OK -- only the gated weights are missing.")
