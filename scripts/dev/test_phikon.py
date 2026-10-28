# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Smoke-test the encoders before committing an overnight embedding run.

Downloads, runs a real forward pass, and reports the embedding dim and
throughput -- the three things that decide whether the plan is feasible.
Do this BEFORE the full run, per the project's standing smoke-test rule.
"""
import time

import torch

device = torch.device("cuda")

for rid, dim_note in [("owkin/phikon-v2", "ViT-L expect 1024"),
                      ("owkin/phikon", "ViT-B expect 768")]:
    print(f"\n=== {rid} ({dim_note}) ===")
    t0 = time.time()
    try:
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(rid)
        model = AutoModel.from_pretrained(rid).to(device).eval().half()
        print(f"  loaded in {time.time()-t0:.0f}s")
        print(f"  processor size={getattr(proc,'size',None)} "
              f"mean={getattr(proc,'image_mean',None)}")
        n = 32
        x = torch.rand(n, 3, 224, 224, device=device).half()
        with torch.inference_mode():
            out = model(pixel_values=x)
            emb = out.last_hidden_state[:, 0, :]   # CLS token
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(3):
            with torch.inference_mode():
                model(pixel_values=x)
        torch.cuda.synchronize()
        rate = 3 * n / (time.time() - t1)
        print(f"  CLS embedding {tuple(emb.shape)}  dtype={emb.dtype}")
        print(f"  throughput ~{rate:.0f} tiles/s -> "
              f"292k tiles in {292000/rate/60:.0f} min of GPU")
        print(f"  params {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED {type(e).__name__}: {str(e)[:200]}")

print("\n=== UNI (gated, no token present) ===")
try:
    import timm
    from huggingface_hub import hf_hub_download
    hf_hub_download("MahmoodLab/UNI", "pytorch_model.bin")
    print("  unexpectedly OK")
except Exception as e:
    print(f"  blocked as expected: {type(e).__name__}: {str(e)[:150]}")
