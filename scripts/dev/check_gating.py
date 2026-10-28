# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Which pathology encoders can we actually download without a human in the loop?

The positive control has to run unattended, so a gated repo that needs the user
to accept terms in a browser is a hard blocker, not an inconvenience. Rank the
candidates by reachability so the run script can pick the best OPEN one and
only prefer a gated one if credentials already exist.
"""
from huggingface_hub import HfApi

CANDIDATES = [
    ("MahmoodLab/UNI",        "timm ViT-L/16, 100k slides, field standard"),
    ("MahmoodLab/UNI2-h",     "timm ViT-H/14, successor to UNI"),
    ("owkin/phikon-v2",       "transformers ViT-L, 460k slides (PANCAN-XL)"),
    ("owkin/phikon",          "transformers ViT-B, iBOT on TCGA"),
    ("bioptimus/H-optimus-0", "timm ViT-g/14, 1.1B params"),
    ("histai/hibou-L",        "transformers ViT-L"),
    ("paige-ai/Virchow2",     "timm ViT-H/14"),
]

api = HfApi()
print(f"{'repo':26s} {'status':12s} {'gated':10s} note")
print("-" * 100)
for rid, note in CANDIDATES:
    try:
        info = api.model_info(rid, timeout=20)
        g = getattr(info, "gated", None)
        files = [s.rfilename for s in (info.siblings or [])]
        wt = [f for f in files if f.endswith((".bin", ".safetensors"))]
        print(f"{rid:26s} {'OK':12s} {str(g):10s} {note}")
        if wt:
            print(f"{'':26s} weights: {', '.join(wt[:3])}")
    except Exception as e:
        msg = str(e).split("\n")[0][:60]
        print(f"{rid:26s} {'BLOCKED':12s} {'?':10s} {type(e).__name__}: {msg}")
