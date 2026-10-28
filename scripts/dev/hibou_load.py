# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Load hibou-L with STOCK transformers classes, no remote code.

The only thing that makes this safe is checking missing/unexpected keys: a
from_pretrained that silently drops half the weights still returns a working
model that emits confident garbage. output_loading_info=True is the check.
"""
import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download

cfgd = json.load(open(hf_hub_download("histai/hibou-L", "config.json")))

import transformers
print("transformers", transformers.__version__)
has = [n for n in dir(transformers) if "Registers" in n]
print("register-capable classes:", has)

from transformers import (Dinov2WithRegistersConfig,
                          Dinov2WithRegistersModel)

keep = {k: v for k, v in cfgd.items()
        if k not in ("architectures", "auto_map", "model_type",
                     "_name_or_path", "transformers_version", "torch_dtype")}
cfg = Dinov2WithRegistersConfig(**keep)
print("\nbuilt config: hidden=%d layers=%d heads=%d patch=%d reg=%d swiglu=%s "
      "layerscale=%s" % (cfg.hidden_size, cfg.num_hidden_layers,
                         cfg.num_attention_heads, cfg.patch_size,
                         cfg.num_register_tokens, cfg.use_swiglu_ffn,
                         cfg.layerscale_value))

model, info = Dinov2WithRegistersModel.from_pretrained(
    "histai/hibou-L", config=cfg, output_loading_info=True)

for k in ("missing_keys", "unexpected_keys", "mismatched_keys"):
    v = info.get(k) or []
    print("%-16s %d %s" % (k, len(v), v[:6]))

n_par = sum(p.numel() for p in model.parameters())
print("params: %.1fM" % (n_par / 1e6))

model = model.cuda().eval().half()
mean = np.array([0.7068, 0.5755, 0.722], np.float32).reshape(1, 3, 1, 1)
std = np.array([0.195, 0.2316, 0.1816], np.float32).reshape(1, 3, 1, 1)
rng = np.random.default_rng(0)
x = rng.integers(0, 255, (4, 224, 224, 3), dtype=np.uint8)
c = (x.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
c = (c - mean) / std
with torch.inference_mode():
    o = model(pixel_values=torch.from_numpy(c).cuda().half())
E = o.last_hidden_state[:, 0, :].float().cpu().numpy()
print("\nCLS embedding:", E.shape, "finite:", bool(np.isfinite(E).all()),
      "|e|=%.2f" % np.linalg.norm(E, axis=1).mean())
print("seq len (1 CLS + 4 reg? + patches):", o.last_hidden_state.shape)
