# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Pathology foundation-model tile embeddings -- the Task 4 positive control.

WHY, AND WHY THESE MODELS
-------------------------
Every result in this project so far compares the pipeline against itself. The
one claim that has never been tested is how 0.67 balanced accuracy compares to
a known-good external standard, and no interpretability claim can be published
without it: "our named pathology features explain the prediction" only means
something if those features are competitive with the opaque state of the art.

The encoder choice is not arbitrary. The UBC-OCEAN Kaggle competition -- this
exact dataset -- was won at ~0.66 balanced accuracy by Owkin, using an ensemble
of Chowder models on top of PHIKON embeddings. So:

  owkin/phikon      ViT-B/16, iBOT on TCGA. The LITERAL encoder behind the
                    winning solution on this benchmark. Nothing else gives a
                    more direct comparison.
  owkin/phikon-v2   ViT-L/16, trained on 460k slides (PANCAN-XL). Its
                    successor, and a stronger current baseline.

Both are ungated on the HF hub, which is what made the first unattended run
possible.

THREE GATED ENCODERS -- the audit needs more than one subject
-------------------------------------------------------------
Under the foundation-model-AUDIT framing, a result measured on Phikon-v2 alone
is a statement about Phikon-v2, not about pathology foundation models. Three
more encoders, each from a different lab, pretraining corpus and objective, so
"how much of what these models know is expressible in named pathology
vocabulary" becomes a claim with an n:

  MahmoodLab/UNI2-h   ViT-H/14 + 8 register tokens, SwiGLU MLP, 1536-d. Mahmood
                      Lab, ~200k slides. The field's most-cited baseline family.
  bioptimus/H-optimus-0  ViT-g/14, 1536-d. Bioptimus, ~500k slides. The largest
                      of the four by parameter count.
  histai/hibou-L      DINOv2 ViT-L/14, 1024-d. HistAI, ~1.2M slides -- trained
                      on the most slides but the smallest of the three new ones.

All three are GATED (`gated: auto`): they need HF_TOKEN in the environment and
a human to have accepted terms on each model page. Set the token as a process
environment variable only -- never in this file, never in a log:

    $env:HF_TOKEN = "hf_..."        # PowerShell, session-scoped

Weights land in ~/.cache/huggingface on first load, after which the token is no
longer needed and can be revoked even mid-run.

Two of the three load through `timm` (hf-hub:), not transformers, and their
normalisation statistics are NOT ImageNet's -- H-optimus-0 in particular uses
stain-specific means near (0.707, 0.579, 0.704). Getting that wrong degrades
embeddings silently, so the per-model mean/std actually resolved is PRINTED at
load and asserted against the model card in --check.

WHAT MAKES THIS A FAIR CONTROL
------------------------------
The embeddings are computed on EXACTLY the tiles the graph already defines --
same 512um grid, same node order, same grid_rc_checksum guard as every other
feature block. So the control differs from the interpretable model in the
instance representation and nothing else: same bags, same folds, same
StratifiedKFold(random_state=repeat), same pooling, same selection rule. Any
gap is attributable to the features alone.

    python scripts/pipeline/embed_tiles_fm.py --check --limit 2      # ALWAYS run first
    python scripts/pipeline/embed_tiles_fm.py --workers 3
"""
from __future__ import annotations

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)

import argparse
import json
import os
import sys
import time
from pathlib import Path

# MUST happen before huggingface_hub is imported anywhere: it reads
# HF_HUB_OFFLINE into a module constant at import time.
#
# This flag is not a convenience. A GATED repo returns 401 to an
# unauthenticated HEAD request even when the weights are already cached, so
# without it every run needs HF_TOKEN present -- meaning the token could not
# be revoked until the last training job finished. With it, the hub is never
# contacted and cached weights are used directly, so the token is needed
# exactly once (--load-only) and can be revoked immediately after.
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np

from extract_chromatin_features import (GRAPH_DIR, TIFF_DIR,
                                        load_graph_geometry, open_slide,
                                        read_tile, slide_path, tissue_mask)

OUT_DIR = GRAPH_DIR / "fm_feats"

MODELS = {
    "phikonv2": dict(repo="owkin/phikon-v2", dim=1024, loader="hf", gated=False,
                     note="ViT-L, PANCAN-XL 460k slides"),
    "phikon":   dict(repo="owkin/phikon", dim=768, loader="hf", gated=False,
                     note="ViT-B, iBOT on TCGA -- the Kaggle winner's encoder"),
    "uni2h":    dict(repo="MahmoodLab/UNI2-h", dim=1536, loader="timm",
                     gated=True, note="ViT-H/14+8reg SwiGLU, Mahmood ~200k WSI",
                     # UNI2-h is NOT a stock timm vit: the architecture is only
                     # recoverable from these kwargs, and timm will happily
                     # build a DIFFERENT (silently wrong) model without them.
                     timm_kwargs=dict(
                         img_size=224, patch_size=14, depth=24, num_heads=24,
                         init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
                         num_classes=0, no_embed_class=True, reg_tokens=8,
                         dynamic_img_size=True),
                     # SwiGLUPacked / SiLU are callables, injected at load time
                     # rather than stored here so this stays an import-free dict
                     swiglu=True,
                     # model card: standard ImageNet statistics
                     card_mean=(0.485, 0.456, 0.406),
                     card_std=(0.229, 0.224, 0.225)),
    "hoptimus0": dict(repo="bioptimus/H-optimus-0", dim=1536, loader="timm",
                      gated=True, note="ViT-g/14, Bioptimus ~500k WSI",
                      timm_kwargs=dict(init_values=1e-5,
                                       dynamic_img_size=False),
                      # model card: stain-specific, NOT ImageNet
                      card_mean=(0.707223, 0.578729, 0.703617),
                      card_std=(0.211883, 0.230117, 0.177517)),
    # hibou-L ships custom remote code whose configuration_dinov2.py imports
    # transformers.onnx -- removed in transformers 5.x, so trust_remote_code
    # crashes outright here. It is genuinely a DINOv2-with-registers model, and
    # the STOCK transformers classes load it with 0 missing / 0 unexpected /
    # 0 mismatched keys (verified), so we bypass the remote code entirely.
    # That is both the working path and the safer one -- no third-party code
    # execution. Loading is checked strictly rather than trusted: a silent key
    # mismatch would yield a model that runs and emits confident garbage.
    "hibouL":   dict(repo="histai/hibou-L", dim=1024, loader="dinov2reg",
                     gated=True, note="DINOv2 ViT-L/14+4reg, HistAI ~1.2M WSI",
                     # from the repo's own preprocessor_config.json, NOT the
                     # values quoted in the README (their std disagrees)
                     card_mean=(0.7068, 0.5755, 0.7220),
                     card_std=(0.1950, 0.2316, 0.1816)),
}

INPUT_PX = 224
BATCH = 64
# A tile with almost no tissue carries no signal and its embedding is dominated
# by glass. It is still emitted (the node must exist for the bag to align) but
# is recorded so the loader can zero it if wanted.
MIN_TISSUE_FRAC = 0.05


def prepare_slide(gpath):
    """Read every node's tile and return them as one uint8 (N, 224, 224, 3).

    Runs in a worker process: this is the disk-bound half, and it is the actual
    bottleneck (GPU inference for the whole corpus is well under an hour).
    """
    import cv2
    g = load_graph_geometry(gpath)
    iid = g["image_id"]
    tif = slide_path(iid)
    if not tif.exists():
        return None, f"missing slide {tif.name}"
    tile_px = g["tile_px"]
    n = g["n_nodes"]
    sl = open_slide(tif)
    try:
        w0, h0 = sl.level_dimensions[0]
        if (h0, w0) != g["wsi_shape"]:
            return None, (f"dim mismatch: slide (h,w)=({h0},{w0}) vs graph "
                          f"{g['wsi_shape']}")
        out = np.zeros((n, INPUT_PX, INPUT_PX, 3), np.uint8)
        tis = np.zeros(n, np.float32)
        for i in range(n):
            ox = int(g["grid_rc"][i, 1] * tile_px)
            oy = int(g["grid_rc"][i, 0] * tile_px)
            try:
                rgb = read_tile(sl, ox, oy, tile_px)
            except Exception:
                continue
            tis[i] = float(tissue_mask(rgb[::4, ::4]).mean())
            out[i] = cv2.resize(rgb, (INPUT_PX, INPUT_PX),
                                interpolation=cv2.INTER_AREA)
    finally:
        sl.close()
    return dict(image_id=iid, label=g["label"], tiles=out, tissue_frac=tis,
                n_nodes=n, grid_rc_sum=int(g["grid_rc"].sum()),
                grid_shape=g["grid_shape"]), None


def _worker(gpath):
    t0 = time.time()
    try:
        payload, err = prepare_slide(Path(gpath))
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:200], time.time() - t0
    return payload, err, time.time() - t0


class Encoders:
    """The requested encoders, loaded once, run in fp16 on one batch each.

    Two loading paths, because the field has not converged on one:

      loader="hf"    transformers AutoModel; embedding = CLS token of
                     last_hidden_state. Phikon, Phikon-v2, Hibou-L.
      loader="timm"  timm.create_model("hf-hub:...", num_classes=0); the
                     forward pass already returns the pooled (CLS) vector, so
                     there is no last_hidden_state to index. UNI2-h,
                     H-optimus-0.

    Normalisation is taken from the model's OWN config in both paths (the HF
    processor, or timm's resolved data config) and then cross-checked against
    the mean/std written on the model card. A mismatch there is the classic
    silent failure for these encoders -- H-optimus-0's statistics are nowhere
    near ImageNet's -- so it warns loudly rather than proceeding quietly.
    """

    def __init__(self, names, device, verbose=True):
        import torch
        self.torch = torch
        self.device = device
        self.models, self.meta = {}, {}
        for nm in names:
            spec = MODELS[nm]
            loader = spec.get("loader", "hf")
            if loader == "timm":
                model, mean, std = self._load_timm(spec, device)
            elif loader == "dinov2reg":
                model, mean, std = self._load_dinov2reg(spec, device)
            else:
                model, mean, std = self._load_hf(spec, device)
            self.models[nm] = (model, mean, std)
            self.meta[nm] = spec
            if verbose:
                print(f"  loaded {nm:10s} {spec['repo']:22s} dim={spec['dim']} "
                      f"({spec['note']})")
                print(f"    mean={np.round(mean.ravel(), 4).tolist()} "
                      f"std={np.round(std.ravel(), 4).tolist()}")
            self._check_norm(nm, spec, mean, std)

    @staticmethod
    def _check_norm(nm, spec, mean, std):
        """Warn if the resolved normalisation disagrees with the model card."""
        for key, got in (("card_mean", mean), ("card_std", std)):
            want = spec.get(key)
            if want is None:
                continue
            if not np.allclose(np.array(want, np.float32), got.ravel(),
                               atol=2e-3):
                print(f"    !! {nm} {key} MISMATCH: resolved "
                      f"{np.round(got.ravel(), 4).tolist()} vs card "
                      f"{list(want)} -- embeddings may be degraded")

    def _load_hf(self, spec, device):
        from transformers import AutoImageProcessor, AutoModel
        kw = {}
        if spec.get("trust_remote_code"):
            kw["trust_remote_code"] = True
        proc = AutoImageProcessor.from_pretrained(spec["repo"], **kw)
        model = AutoModel.from_pretrained(spec["repo"], **kw)
        model = model.to(device).eval().half()
        mean = np.array(proc.image_mean, np.float32).reshape(1, 3, 1, 1)
        std = np.array(proc.image_std, np.float32).reshape(1, 3, 1, 1)
        return model, mean, std

    def _load_dinov2reg(self, spec, device):
        """hibou-L: stock DINOv2-with-registers, remote code bypassed.

        The hub config declares model_type "dinov2", which AutoConfig would
        resolve to the register-less Dinov2Config and silently drop the 4
        register tokens -- so the config class is named explicitly. Loading is
        then verified key-by-key.
        """
        import json

        from huggingface_hub import hf_hub_download
        from transformers import (AutoImageProcessor,
                                  Dinov2WithRegistersConfig,
                                  Dinov2WithRegistersModel)
        raw = json.load(open(hf_hub_download(spec["repo"], "config.json")))
        keep = {k: v for k, v in raw.items()
                if k not in ("architectures", "auto_map", "model_type",
                             "_name_or_path", "transformers_version",
                             "torch_dtype")}
        cfg = Dinov2WithRegistersConfig(**keep)
        model, info = Dinov2WithRegistersModel.from_pretrained(
            spec["repo"], config=cfg, output_loading_info=True)
        bad = {k: (info.get(k) or []) for k in
               ("missing_keys", "unexpected_keys", "mismatched_keys")}
        if any(bad.values()):
            raise RuntimeError(f"{spec['repo']} loaded with key problems "
                               f"{ {k: v[:5] for k, v in bad.items()} } -- "
                               f"refusing to embed with partial weights")
        model = model.to(device).eval().half()
        proc = AutoImageProcessor.from_pretrained(spec["repo"])
        mean = np.array(proc.image_mean, np.float32).reshape(1, 3, 1, 1)
        std = np.array(proc.image_std, np.float32).reshape(1, 3, 1, 1)
        return model, mean, std

    def _load_timm(self, spec, device):
        import timm
        import torch
        kw = dict(spec.get("timm_kwargs") or {})
        if spec.get("swiglu"):
            kw["mlp_layer"] = timm.layers.SwiGLUPacked
            kw["act_layer"] = torch.nn.SiLU
        kw.setdefault("num_classes", 0)
        model = timm.create_model(f"hf-hub:{spec['repo']}", pretrained=True,
                                  **kw)
        model = model.to(device).eval().half()
        cfg = timm.data.resolve_data_config({}, model=model)
        mean = np.array(cfg["mean"], np.float32).reshape(1, 3, 1, 1)
        std = np.array(cfg["std"], np.float32).reshape(1, 3, 1, 1)
        return model, mean, std

    def embed(self, tiles, name):
        """(N,224,224,3) uint8 -> (N, dim) float32 embeddings."""
        torch = self.torch
        model, mean, std = self.models[name]
        is_timm = MODELS[name].get("loader", "hf") == "timm"
        outs = []
        for i in range(0, len(tiles), BATCH):
            chunk = tiles[i:i + BATCH].astype(np.float32) / 255.0
            chunk = chunk.transpose(0, 3, 1, 2)
            chunk = (chunk - mean) / std
            x = torch.from_numpy(chunk).to(self.device).half()
            with torch.inference_mode():
                if is_timm:
                    # num_classes=0 -> forward() already returns the pooled
                    # (CLS) feature vector; there is no last_hidden_state.
                    o = model(x)
                else:
                    o = model(pixel_values=x).last_hidden_state[:, 0, :]
            outs.append(o.float().cpu().numpy())
        if not outs:
            return np.zeros((0, 0), np.float32)
        E = np.concatenate(outs, 0)
        want = MODELS[name].get("dim")
        if want and E.shape[1] != want:
            # a wrongly-built timm model is otherwise completely silent
            raise RuntimeError(f"{name}: got {E.shape[1]}-d embeddings, "
                               f"expected {want} -- model built wrong")
        if not np.isfinite(E).all():
            raise RuntimeError(f"{name}: non-finite embeddings (fp16 overflow?)")
        return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["phikonv2", "phikon"],
                    choices=sorted(MODELS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="use only cached weights, never contact the hub. "
                         "Handled at import time; see the note at the top. "
                         "Pass this on every run after --load-only so HF_TOKEN "
                         "can be revoked.")
    ap.add_argument("--load-only", action="store_true",
                    help="download + build the encoders, run one synthetic "
                         "batch, and exit. Reads no slides. This is the step "
                         "that caches gated weights, so HF_TOKEN is only "
                         "needed until it finishes.")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gated = [m for m in args.models if MODELS[m].get("gated")]
    if gated:
        tok = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN")
        if args.offline:
            print(f"gated models requested: {gated} -- OFFLINE, using cached "
                  f"weights only (no token needed)")
        else:
            print(f"gated models requested: {gated} -- HF_TOKEN "
                  f"{'present' if tok else 'ABSENT (will fail)'}")

    if args.load_only:
        enc = Encoders(args.models, device)
        rng = np.random.default_rng(0)
        probe = rng.integers(0, 255, (4, INPUT_PX, INPUT_PX, 3), dtype=np.uint8)
        for nm in args.models:
            E = enc.embed(probe, nm)
            nz = np.linalg.norm(E, axis=1)
            print(f"  {nm:10s} probe -> {E.shape} |e|={nz.mean():.2f} "
                  f"finite={bool(np.isfinite(E).all())}")
        print("\nweights cached; HF_TOKEN is no longer needed.")
        return

    graphs = sorted(GRAPH_DIR.glob("*_graph.npz"))
    if not args.check and not args.force:
        # image_id is everything before the "_graph" suffix, NOT the first
        # underscore-delimited token: UBC-OCEAN ids are bare numbers but other
        # corpora use ids like "10_OOU198_EC_SC", which split("_")[0] truncates
        # to "10" -- silently making every slide look un-embedded.
        graphs = [g for g in graphs
                  if not all((OUT_DIR / f"{g.name[:-len('_graph.npz')]}_{m}.npz").exists()
                             for m in args.models)]
    if args.limit:
        graphs = graphs[:args.limit]
    print(f"foundation-model embedding: {len(graphs)} slide(s), "
          f"models={args.models}, device={device}")
    if not graphs:
        return
    enc = Encoders(args.models, device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0, done, failed = time.time(), 0, []
    jobs = [str(g) for g in graphs]

    def handle(payload, err, secs, i):
        nonlocal done
        if err or payload is None:
            failed.append((Path(jobs[i - 1]).name, err))
            print(f"  [{i}/{len(jobs)}] ERROR {err}")
            return
        iid = payload["image_id"]
        line = [f"  [{i}/{len(jobs)}] {iid} {payload['label']:5s} "
                f"n={payload['n_nodes']} read={secs:.0f}s"]
        for nm in args.models:
            te = time.time()
            E = enc.embed(payload["tiles"], nm)
            line.append(f"{nm}={E.shape[1]}d/{time.time()-te:.0f}s")
            if not args.check:
                np.savez_compressed(
                    OUT_DIR / f"{iid}_{nm}.npz",
                    image_id=np.array([iid]), label=np.array([payload["label"]]),
                    fm_features=E.astype(np.float32),
                    fm_model=np.array([MODELS[nm]["repo"]]),
                    tissue_frac=payload["tissue_frac"],
                    grid_shape=np.array(payload["grid_shape"]),
                    grid_rc_checksum=np.array([payload["grid_rc_sum"]]),
                    params=np.array([json.dumps(dict(
                        INPUT_PX=INPUT_PX, MIN_TISSUE_FRAC=MIN_TISSUE_FRAC,
                        pooling="CLS token", dtype="fp16 inference"))]))
            if args.check:
                nz = np.linalg.norm(E, axis=1)
                line.append(f"|e|={nz.mean():.1f}+-{nz.std():.1f}")
        done += 1
        el = (time.time() - t0) / 60
        rate = el / max(done, 1)
        print(" ".join(line) + f"  ({el:.1f}m, eta {rate*(len(jobs)-i):.0f}m)")

    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, (payload, err, secs) in enumerate(
                    pool.imap(_worker, jobs), 1):
                handle(payload, err, secs, i)
    else:
        for i, j in enumerate(jobs, 1):
            payload, err, secs = _worker(j)
            handle(payload, err, secs, i)

    print(f"\ndone: {done} ok, {len(failed)} failed, "
          f"{(time.time()-t0)/60:.1f} min")
    for nm, e in failed[:20]:
        print(f"  {nm}: {e}")


if __name__ == "__main__":
    main()
