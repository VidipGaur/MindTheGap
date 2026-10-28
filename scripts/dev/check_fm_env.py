# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Can this machine run a pathology foundation-model positive control?

Checks the three things that actually block it: the packages, network reach to
the HF hub, and whether any encoder weights are already cached locally.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import importlib
import os
import shutil
from pathlib import Path
import corpus_paths

print("=== packages ===")
for m in ["torch", "timm", "transformers", "huggingface_hub", "PIL",
          "tiffslide", "openslide", "safetensors", "einops"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:18s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:18s} MISSING ({type(e).__name__})")

print("\n=== gpu ===")
try:
    import torch
    print(f"  cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    if torch.cuda.is_available():
        free, tot = torch.cuda.mem_get_info()
        print(f"  vram free {free/2**30:.1f} / {tot/2**30:.1f} GB")
except Exception as e:
    print("  ", e)

print("\n=== disk ===")
for d in [str(Path.home()), str(corpus_paths.ROOT)]:
    try:
        t, u, f = shutil.disk_usage(d)
        print(f"  {d:40s} free {f/2**30:.0f} GB / {t/2**30:.0f} GB")
    except Exception as e:
        print(f"  {d}: {e}")

print("\n=== hf cache ===")
hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
hub = hub / "hub" if (hub / "hub").exists() else hub
print(f"  {hub}  exists={hub.exists()}")
if hub.exists():
    for d in sorted(hub.glob("models--*")):
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        print(f"    {d.name:55s} {sz/2**20:8.0f} MB")

print("\n=== hf token ===")
tok = Path.home() / ".cache" / "huggingface" / "token"
print(f"  env HF_TOKEN set: {bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))}")
print(f"  token file: {tok.exists()}")

print("\n=== network ===")
import urllib.request
for url in ["https://huggingface.co", "https://pypi.org"]:
    try:
        with urllib.request.urlopen(url, timeout=12) as r:
            print(f"  {url:28s} HTTP {r.status}")
    except Exception as e:
        print(f"  {url:28s} FAIL {type(e).__name__}: {e}")

# is the gated UNI repo reachable / already permitted?
try:
    from huggingface_hub import HfApi
    api = HfApi()
    for rid in ["MahmoodLab/UNI", "MahmoodLab/UNI2-h", "owkin/phikon-v2",
                "owkin/phikon", "bioptimus/H-optimus-0", "histai/hibou-L"]:
        try:
            info = api.model_info(rid, timeout=15)
            print(f"  {rid:26s} REACHABLE gated={getattr(info,'gated',None)}")
        except Exception as e:
            msg = str(e).split("\n")[0][:90]
            print(f"  {rid:26s} {type(e).__name__}: {msg}")
except Exception as e:
    print("  hub api unavailable:", e)
