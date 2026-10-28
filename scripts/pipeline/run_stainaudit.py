# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""How much of each encoder's advantage is the BATCH SHORTCUT we removed from
our own model?

WHY THIS EXISTS
---------------
The interpretable arm was cleaned. `dead_count` was convicted as a
staining/scanning artefact (Task 8: rho=-0.42 with tissue brightness; staining
descriptors alone predict subtype at 0.358 vs 0.20 chance; a staining-only
clustering yields a cluster with zero CC and zero MC slides at 10x the corpus
dead_count), and the chrom_context tile-staining descriptors went with it.
Removing them cost 0.0272 balanced accuracy, paired p=0.0009 -- a real loss,
knowingly taken, because the accuracy was being bought with "which batch is
this slide from" rather than with tumour morphology.

We could do that only because our features have names. A foundation model's
1024 dimensions have none, so the identical shortcut cannot be located and
removed from the encoders. Stain and scanner variation is well documented to
survive into these embeddings -- it is why stain normalisation and stain
augmentation exist as techniques at all.

So the reported gap (UNI2-h 0.8352 vs clean interpretable 0.6603, +0.170) is
contaminated in the ENCODERS' favour by an unknown amount: we handicapped one
side and not the other, then compared them. This run makes the amount known.

THE INSTRUMENT IS THE ONE WE ALREADY BUILT
------------------------------------------
fm_subspace_split.py deletes a subspace of the embedding aligned to a named
feature block, with a rank-matched null and a variance-matched control.
Nothing in it cares what the block MEANS. Pointing it at the 10 per-tile
staining descriptors instead of the 49 morphometry columns asks: delete
everything about the embedding that stain level, stain balance and focus can
linearly predict, retrain, and see what the encoder loses.

  fm_stainresid      encoder minus its stain-predictable component
  fm_stainresidnull  rank-matched map fitted from MISMATCHED slides (primary)
  fm_stainresidctl   variance-matched random PCA subspace (second bound)

READING IT, fixed before the run
--------------------------------
  stainresid ~= stainresidnull (p>0.10)  -> the encoder's accuracy does not
      live in the stain-aligned subspace. The comparison was fair after all
      and the gap stands as biology. This is the outcome that HELPS the paper,
      which is precisely why it must be pre-registered rather than decided
      afterwards.
  stainresid <  stainresidnull (p<0.10)  -> part of the encoder's advantage IS
      the shortcut. The gap between them, as a share of (encoder - clean
      interpretable), is the contaminated fraction, and the headline gap must
      be reported as an upper bound with that number attached.

Either way this goes in the paper. A null is a limitation retired with
evidence; a positive is a limitation quantified. The one unacceptable outcome
was leaving it unmeasured after noticing it.

WHAT IT STILL DOES NOT SETTLE
-----------------------------
The deletion is linear. An encoder could encode stain nonlinearly and keep the
shortcut through a route this map cannot reach, so a null here bounds the
LINEARLY accessible batch signal, not all of it. State that; do not upgrade a
null into a clean bill of health.

Encoder order is deliberate: UNI2-h first because it sets the headline gap,
Phikon-v2 second because it is the weakest encoder and the original
comparator, so those two alone already bracket the range if the run has to be
cut short. Results are re-reported after every encoder, so partial output is
usable.

USAGE
    python scripts/pipeline/run_stainaudit.py                 # all four encoders
    python scripts/pipeline/run_stainaudit.py --models uni2h  # just one
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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_multiencoder import (ADD, EPOCHS, PY, REPEATS, SCRIPTS, banner,  # noqa: E402
                              cells, paired, results_path, sh, train)

# (cli name, display name). Order is the fallback priority, see docstring.
ENCODERS = [
    ("uni2h", "UNI2-h"),
    ("phikonv2", "Phikon-v2"),
    ("hoptimus0", "H-optimus-0"),
    ("hibouL", "Hibou-L"),
]
ARMS = ("stainresid", "stainresidnull", "stainresidctl")
CLEAN = ("abmil_v5nucchromeos_clean_final_results.csv", ADD)      # 0.6603


def report(models):
    banner("STAIN AUDIT RESULTS -- paired on shared (repeat, fold)")
    print("  THE TEST is stainresid vs stainresidnull. A gap there is the\n"
          "  encoder's accuracy that lived in the stain-aligned subspace --\n"
          "  the shortcut the interpretable arm had surgically removed.",
          flush=True)
    for m, nice in models:
        if not results_path(f"v5fm{m}stainresid", "fm_stainresid").exists():
            continue
        print(f"\n  {nice}:", flush=True)
        R = {a: (f"abmil_v5fm{m}{a}_fm_{a}_results.csv", ADD) for a in ARMS}
        FMA = (f"abmil_v5fm{m}_fm_addmil_results.csv", ADD)
        d = paired(R["stainresid"], R["stainresidnull"], "stainresid vs stainresidnull (THE TEST)")
        paired(R["stainresid"], R["stainresidctl"], "stainresid vs stainresidctl (variance-matched)")
        paired(R["stainresid"], FMA, "stainresid vs full encoder")
        paired(R["stainresidnull"], FMA, "stainresidnull vs full encoder (sanity: ~0)")
        adv = paired(FMA, CLEAN, "full encoder vs clean interpretable (the gap)")

        # contaminated share = (what deleting stain costs beyond the matched
        # null) / (the encoder's whole advantage over the cleaned model)
        #
        # paired() returns a (delta, p) TUPLE, not a float. The first version of
        # this block did abs(d) on the tuple, which raises TypeError -- and the
        # guard below only caught ZeroDivisionError, so it escaped report(),
        # escaped main(), and killed the chain AFTER UNI2-h's three arms had
        # finished. Unpack explicitly, and only report a share when the test is
        # actually significant; a share computed off a null delta is noise
        # divided by a gap and means nothing.
        if d is not None and adv is not None and abs(adv[0]) > 1e-9:
            delta, pval = d
            share = abs(delta) / abs(adv[0]) * 100.0
            verdict = ("SIGNIFICANT -- report the gap as an upper bound"
                       if pval < 0.10 else
                       "n.s. -- no linearly accessible stain shortcut; this "
                       "share is not a real quantity, shown for scale only")
            print(f"    -> stain-aligned subspace accounts for {share:.0f}% "
                  f"of this encoder's advantage  [{verdict}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[m for m, _ in ENCODERS])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    models = [(m, n) for m, n in ENCODERS if m in a.models]
    if a.report_only:
        report(models)
        return 0

    banner(f"STAIN AUDIT -- {len(models)} encoder(s) x 3 arms")
    t0 = time.time()
    for m, nice in models:
        banner(f"{nice} -- stain subspace")
        # the split is idempotent per encoder; skip if the corpora exist
        done = results_path(f"v5fm{m}stainresidctl", "fm_stainresidctl").exists()
        if not done:
            sh([PY, "-u", str(SCRIPTS / "pipeline" / "fm_subspace_split.py"), "--model", m,
                "--target", "stain"], f"stain_split_{m}")
        for arm in ARMS:
            train(f"v5fm{m}{arm}", f"fm_{arm}",
                  ["--fm-features", f"{m}{arm}"], config=ADD, rebuild=True)
        report(models)          # re-report after every encoder
        print(f"\n  elapsed {(time.time()-t0)/3600:.1f} h", flush=True)

    banner("STAIN AUDIT DONE")
    report(models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
