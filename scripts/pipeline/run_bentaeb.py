# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Orchestrator: Instrument 1 on the Ben Taeb et al. external corpus.

Mirrors the UBC-OCEAN chain stage for stage, on the parallel `bentaeb_run`
corpus that corpus_paths.py resolves. Every stage is the SAME script the
primary corpus was built with -- nothing about the feature definitions,
the tile lattice or the model is re-derived here.

    S1  graphs          build_streamline_graphs_bridge.py   (tile lattice)
    S2  nuclear         extract_nuclear_features.py         (contours)
    S3  chromatin       extract_chromatin_features.py       (pixels)
    S4  eosin           extract_eosin_features.py           (pixels)
    S5  embeddings      embed_tiles_fm.py                   (4 encoders, 1 pass)
    S6  instrument 1    eval_external.py                    (frozen inference)

Stages are skipped when their outputs are already complete, so the chain is
restartable. Each stage logs to bentaeb_run/logs/.

NOTE ON CACHES. train_abmil.py's bag cache is deliberately not used anywhere in
this chain -- eval_external.py rebuilds bags every run. A stale cache surviving a change to
the on-disk feature corpus silently trains on the wrong bags, and inference over
80 slides is cheap enough that the cache buys nothing worth that risk.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import corpus_paths
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent.parent
RUN = corpus_paths.ROOT / "bentaeb_run"
GDIR = RUN / "graphs"
LOGS = RUN / "logs"
MODELS = ["uni2h", "hoptimus0", "phikonv2", "hibouL"]

ENV = dict(os.environ, UBC_RUN_DIR="bentaeb_run", PYTHONUNBUFFERED="1")


def banner(s):
    print(f"\n{'=' * 74}\n{s}\n{'=' * 74}", flush=True)


def n_graphs():
    return len(list(GDIR.glob("*_graph.npz")))


def n_out(sub, suffix):
    d = GDIR / sub
    return len(list(d.glob(f"*{suffix}"))) if d.exists() else 0


def sh(args, logname, expect=None):
    """Run one stage. stdout and stderr go to separate files: a 2>&1 pipe through
    PowerShell fully buffers Python's stdout when it is not a TTY, which makes a
    healthy run look hung."""
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / f"{logname}.log"
    err = LOGS / f"{logname}.err"
    print(f"  $ {' '.join(str(a) for a in args)}", flush=True)
    print(f"    -> {out}", flush=True)
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fo, \
            open(err, "w", encoding="utf-8") as fe:
        rc = subprocess.call([PY, "-u"] + [str(a) for a in args],
                             cwd=str(SCRIPTS), env=ENV, stdout=fo, stderr=fe)
    mins = (time.time() - t0) / 60
    print(f"    rc={rc}  {mins:.1f} min", flush=True)
    if rc != 0:
        tail = out.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        etail = err.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        print("    --- stdout tail ---", flush=True)
        for t in tail:
            print(f"    {t}", flush=True)
        print("    --- stderr tail ---", flush=True)
        for t in etail:
            print(f"    {t}", flush=True)
        raise SystemExit(f"stage {logname} failed (rc={rc})")
    if expect is not None:
        got = expect()
        print(f"    outputs: {got}", flush=True)
    return rc


def wait_for_graphs(target=None, poll=60, quiet_for=180):
    """Block until the graph build is done.

    Done means either the builder's own 'Done.' line has appeared in its log, or
    the file count has stopped moving for `quiet_for` seconds -- the second
    condition covers the case where the build was launched outside this chain
    and its log is elsewhere.
    """
    log = RUN / "graph_build.log"
    last, since = -1, time.time()
    while True:
        n = n_graphs()
        if log.exists():
            txt = log.read_text(encoding="utf-8", errors="replace")
            if "\nDone." in txt or txt.startswith("Done."):
                print(f"  builder reported Done. ({n} graphs)", flush=True)
                return n
        if target and n >= target:
            print(f"  reached target {target} graphs", flush=True)
            return n
        if n != last:
            last, since = n, time.time()
            print(f"  graphs: {n}" + (f" / {target}" if target else ""),
                  flush=True)
        elif time.time() - since > quiet_for:
            print(f"  graph count static at {n} for {quiet_for}s -- proceeding",
                  flush=True)
            return n
        time.sleep(poll)


def main():
    only = set(sys.argv[1:]) or None

    def want(s):
        return only is None or s in only

    banner("S1  tile lattice (graphs)")
    if want("s1"):
        n = wait_for_graphs(target=80)
    else:
        n = n_graphs()
    print(f"  graphs on disk: {n}", flush=True)
    if n == 0:
        raise SystemExit("no graphs -- run build_streamline_graphs_bridge.py first")

    banner("S2  named features: nuclear morphometry (from cells.json contours)")
    if want("s2") and n_out("nuclear_feats", "_nuclei.npz") < n:
        sh([str(SCRIPTS / "pipeline" / "extract_nuclear_features.py")], "s2_nuclear",
           lambda: n_out("nuclear_feats", "_nuclei.npz"))
    else:
        print(f"  skip -- {n_out('nuclear_feats', '_nuclei.npz')}/{n} present",
              flush=True)

    banner("S3  named features: chromatin (haematoxylin channel)")
    if want("s3") and n_out("chromatin_feats", "_chromatin.npz") < n:
        sh([str(SCRIPTS / "pipeline" / "extract_chromatin_features.py"), "--workers", "3"], "s3_chromatin",
           lambda: n_out("chromatin_feats", "_chromatin.npz"))
    else:
        print(f"  skip -- {n_out('chromatin_feats', '_chromatin.npz')}/{n} present",
              flush=True)

    banner("S4  named features: eosin / cytoplasm")
    if want("s4") and n_out("eosin_feats", "_eosin.npz") < n:
        sh([str(SCRIPTS / "pipeline" / "extract_eosin_features.py"), "--workers", "3"], "s4_eosin",
           lambda: n_out("eosin_feats", "_eosin.npz"))
    else:
        print(f"  skip -- {n_out('eosin_feats', '_eosin.npz')}/{n} present",
              flush=True)

    banner("S5  foundation-model embeddings (4 encoders, one pass over tiles)")
    need = sum(1 for g in GDIR.glob("*_graph.npz")
               for m in MODELS
               if not (GDIR / "fm_feats" /
                       f"{g.name[:-len('_graph.npz')]}_{m}.npz").exists())
    if want("s5") and need:
        print(f"  {need} (slide, model) pairs outstanding", flush=True)
        # one tile read serves all four encoders, which is the whole reason to
        # pass them together: WSI reading is the bottleneck, not the GPU
        sh([str(SCRIPTS / "pipeline" / "embed_tiles_fm.py"), "--models"] + MODELS + ["--workers", "2",
            "--offline"], "s5_embed",
           lambda: n_out("fm_feats", ".npz"))
    else:
        print(f"  skip -- all {len(MODELS)} encoders present for every slide",
              flush=True)

    banner("S6  Instrument 1 -- frozen inference, 5 arms x 50 checkpoints")
    sh([str(SCRIPTS / "pipeline" / "eval_external.py"), "--boot", "1000",
        "--out", str(RUN / "instrument1.json")], "s6_eval")
    print((LOGS / "s6_eval.log").read_text(encoding="utf-8", errors="replace"),
          flush=True)


if __name__ == "__main__":
    main()
