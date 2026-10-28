# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Can hibou-L load WITHOUT its custom remote code?

Its configuration_dinov2.py imports transformers.onnx, gone in transformers 5.
If the config is really just a stock Dinov2Config, the stock classes load the
same weights and the remote code is dead weight we can skip entirely.
"""
import json
import os

from huggingface_hub import hf_hub_download

p = hf_hub_download("histai/hibou-L", "config.json")
cfg = json.load(open(p))
print("config.json keys:", sorted(cfg.keys()))
print()
for k in ["model_type", "architectures", "auto_map", "hidden_size",
          "num_hidden_layers", "num_attention_heads", "patch_size",
          "image_size", "layerscale_value", "use_swiglu_ffn"]:
    if k in cfg:
        print("  %-22s %s" % (k, cfg[k]))

try:
    pp = hf_hub_download("histai/hibou-L", "preprocessor_config.json")
    print("\npreprocessor_config.json:")
    print(json.dumps(json.load(open(pp)), indent=2)[:1200])
except Exception as e:
    print("\nno preprocessor_config.json:", type(e).__name__)
