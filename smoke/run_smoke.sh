#!/usr/bin/env bash
# Five-slide smoke run of the extraction pipeline: fetch, convert, manifest,
# CellViT, KDE, graph, nuclear/chromatin/eosin blocks, embeddings, loader
# integration check, then verify_smoke_outputs.py. See smoke/README.md.
#
# UBC_ROOT must be a separate dataset root holding only this run's files. The
# script refuses a non-empty root it did not create, so no cache or output from
# another run can be picked up. Nothing is ever deleted.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
S="$REPO/scripts"
PY=${PYTHON:-$REPO/.venv/bin/python}
export PYTHON=$PY MPLBACKEND=Agg

: "${UBC_ROOT:?set UBC_ROOT to a fresh directory outside the repository}"
export UBC_ROOT
unset UBC_RUN_DIR

MARK="$UBC_ROOT/.smoke_root"
if [ -d "$UBC_ROOT" ] && [ -n "$(ls -A "$UBC_ROOT" 2>/dev/null)" ] && [ ! -f "$MARK" ]; then
  echo "$UBC_ROOT is not empty and was not created by this harness; use a fresh directory"
  exit 2
fi
mkdir -p "$UBC_ROOT/cellvit_run/logs"
[ -f "$MARK" ] || date -Is >"$MARK"
LOGS="$UBC_ROOT/cellvit_run/logs"

read -r -a IDS <<<"$("$PY" -c 'import json,sys; print(" ".join(s["image_id"] for s in json.load(open(sys.argv[1]))["slides"]))' "$HERE/slides.json")"
IDS_CSV=$(IFS=,; echo "${IDS[*]}")

# Every stage writes stdout and stderr to separate files; merging them through
# PowerShell fully buffers output and hides warnings until the run ends.
stage() {
  local name=$1; shift
  echo "== $name: $*"
  local t0=$SECONDS
  if "$@" >"$LOGS/smoke_$name.log" 2>"$LOGS/smoke_$name.err.log"; then
    echo "   ok ($((SECONDS - t0)) s)"
  else
    local rc=$?
    echo "   FAILED with exit $rc after $((SECONDS - t0)) s; see $LOGS/smoke_$name.err.log"
    exit "$rc"
  fi
}

stage fetch bash "$HERE/fetch_slides.sh"
stage convert "$PY" "$HERE/convert_to_pyramid_tiff.py" "${IDS[@]}"
stage manifest "$PY" "$HERE/build_manifest.py" "${IDS[@]}"
stage cellvit bash "$HERE/run_cellvit.sh" "${IDS[@]}"
for id in "${IDS[@]}"; do
  stage "kde_$id" "$PY" -u "$S/pipeline/build_kde_density_maps_new.py" --only "$id"
  stage "graph_$id" "$PY" -u "$S/pipeline/build_streamline_graphs_bridge.py" --only "$id"
done
stage nuclear_check "$PY" -u "$S/pipeline/extract_nuclear_features.py" --check --only "$IDS_CSV"
stage nuclear "$PY" -u "$S/pipeline/extract_nuclear_features.py" --only "$IDS_CSV"
stage chromatin_check "$PY" -u "$S/pipeline/extract_chromatin_features.py" --check --only "$IDS_CSV"
stage chromatin "$PY" -u "$S/pipeline/extract_chromatin_features.py" --only "$IDS_CSV"
# The eosin extractor has no --only; the root holds only the panel's graphs.
stage eosin_check "$PY" -u "$S/pipeline/extract_eosin_features.py" --check
stage eosin "$PY" -u "$S/pipeline/extract_eosin_features.py"

MODELS=(phikonv2)
if [ -n "${HF_TOKEN:-}" ]; then MODELS+=(uni2h hoptimus0 hibouL); fi
stage embed_check "$PY" -u "$S/pipeline/embed_tiles_fm.py" --check --limit 2 --models phikonv2
stage embed "$PY" -u "$S/pipeline/embed_tiles_fm.py" --models "${MODELS[@]}"

# Expected to stop at the flag-leak diagnostic with one slide per class; the
# result is recorded, then the check is repeated without that diagnostic.
set +e
"$PY" -u "$HERE/integration_check.py" --json "$UBC_ROOT/cellvit_run/integration_plain.json" \
  >"$LOGS/smoke_integration_plain.log" 2>"$LOGS/smoke_integration_plain.err.log"
echo "== integration (plain): exit $? (2 = loader refused the five-slide corpus)"
set -e
stage integration "$PY" -u "$HERE/integration_check.py" --skip-flag-leak-report \
  --json "$UBC_ROOT/cellvit_run/integration.json"

set +e
"$PY" "$HERE/verify_smoke_outputs.py" --root "$UBC_ROOT" --models "${MODELS[@]}" \
  --json "$UBC_ROOT/cellvit_run/smoke_verify.json" | tee "$LOGS/smoke_verify.log"
rc=${PIPESTATUS[0]}
set -e
echo "verify_smoke_outputs.py exit $rc; table in $LOGS/smoke_verify.log"
exit "$rc"
