# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Stage 9 of the smoke run: load the slides through the training-side loaders
and run one Additive MIL forward pass.

Checks that fm_subspace_split.named_features() returns the 49 named columns of
the clean interpretable model and the 10 stain columns, and that the per-tile
contributions of Additive MIL sum to the bag logit on a real bag.

The morph block is requested first because that is the call the fix
described in docs/NAMED_FEATURES_CACHE.md changed: on a root with no bag cache it used to
return 38 columns and leave a nuclear-only cache the stain call could not use.

On every load, train_abmil.load_bags() prints flag_leak_report(), which
cross-validates with StratifiedKFold(5) and so refuses a corpus with fewer than
five slides per class, as the five-slide panel is. That report is a printed
diagnostic, not a gate (its own docstring says so). --skip-flag-leak-report
replaces it with a no-op for this process only, so the column counts and the
forward pass can still be checked; run_smoke.sh runs once without it to record
the refusal, then once with it.

Usage:
    UBC_ROOT=<smoke root> python smoke/integration_check.py [--skip-flag-leak-report] [--json OUT]

Exit status: 0 when every check passes, 1 when a check fails, 2 when a loader
refuses the corpus (a limit of a five-slide test, not a pipeline failure).
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _d in ("pipeline", "analysis", "verify", "dev"):
    sys.path.insert(0, str(REPO / "scripts" / _d))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="also write the results here")
    ap.add_argument("--skip-flag-leak-report", action="store_true",
                    help="do not run the load-time flag-leak diagnostic (see above)")
    args = ap.parse_args(argv)

    import torch

    import fm_subspace_split as S
    import train_abmil as T

    if args.skip_flag_leak_report:
        T.flag_leak_report = lambda *a, **k: print("  flag-leak diagnostic skipped")

    rows = []

    def record(check, expected, observed, ok):
        rows.append(dict(check=check, expected=str(expected), observed=str(observed),
                         status="PASS" if ok else "FAIL"))
        print(f"{'PASS' if ok else 'FAIL'}  {check}: expected {expected}, observed {observed}")

    print(f"dataset root: {S.ROOT}")
    try:
        morph, morph_names = S.named_features("morph")
        stain, stain_names = S.named_features("stain")
    except BaseException as e:  # the loaders refuse with SystemExit as well as exceptions
        traceback.print_exc()
        msg = f"{type(e).__name__}: {e}"
        print(f"LIMIT: a loader refused this corpus: {msg}")
        if args.json:
            Path(args.json).write_text(json.dumps(dict(limit=msg, rows=rows), indent=1))
        return 2

    record("named_features('morph') columns", 49, len(morph_names), len(morph_names) == 49)
    record("named_features('stain') columns", 10, len(stain_names), len(stain_names) == 10)
    record("same slides in both blocks", sorted(morph), sorted(stain), sorted(morph) == sorted(stain))

    iid = sorted(morph)[0]
    X = morph[iid]
    cfg = next(c for c in T.CONFIGS if c[0] == "addmil_h128_do5")
    torch.manual_seed(0)
    model = T.build_model(cfg, X.shape[1], len(T.CLASSES)).double().eval()
    x = torch.from_numpy(X)[None]
    mask = torch.ones(1, X.shape[0], dtype=torch.bool)
    with torch.no_grad():
        out, _, contrib = model(x, mask, return_contrib=True)
    err = float((contrib.sum(1) - out).abs().max())
    record(f"Additive MIL on slide {iid}: max |sum of tile contributions - logit|",
           "< 1e-9", f"{err:.2e}", err < 1e-9)

    if args.json:
        Path(args.json).write_text(json.dumps(dict(rows=rows), indent=1))
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
