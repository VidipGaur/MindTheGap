#!/usr/bin/env bash
# Run CellViT-SAM-H on the given slide ids with the settings used for the
# reported results (docs/PROVENANCE.md).
#
# Usage: smoke/run_cellvit.sh ID [ID ...]
# Needs UBC_ROOT, CELLVIT_CACHE (a folder for the checkpoint, outside the
# repository), the cellvit package in the active Python, and a CUDA GPU.
set -euo pipefail

: "${UBC_ROOT:?set UBC_ROOT}"
: "${CELLVIT_CACHE:?set CELLVIT_CACHE to a folder for the CellViT checkpoint}"
export CELLVIT_CACHE
PY=${PYTHON:-python}
RUN="$UBC_ROOT/cellvit_run"
OUT="$RUN/detections"
LOGS="$RUN/logs"
mkdir -p "$OUT" "$LOGS" "$CELLVIT_CACHE"
REPORT="$LOGS/cellvit_runs.tsv"

{
  echo "# $(date -Is)"
  echo "# cellvit: $("$PY" -c 'import importlib.metadata as m; print("Version:", m.version("cellvit"))' 2>/dev/null || echo 'not installed')  (version at time of re-run, not original)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/# gpu: /' || echo "# gpu: nvidia-smi unavailable"
} >>"$REPORT"
[ -s "$REPORT" ] && grep -q $'^image_id\t' "$REPORT" || printf 'image_id\texit\tseconds\n' >>"$REPORT"

for id in "$@"; do
  if [ -f "$OUT/$id/cells.json" ]; then echo "have detections for $id"; continue; fi
  echo "CellViT on $id ..."
  t0=$(date +%s)
  rc=0
  # The CLI asks for confirmation on start-up; the April runner piped "yes" in.
  printf 'yes\n%.0s' $(seq 20) | "$PY" -m cellvit.detect_cells \
      --model SAM --outdir "$OUT" --geojson \
      --cpu_count 4 --ray_worker 1 --ray_remote_cpus 2 --memory 12000 --batch_size 2 \
      process_wsi --wsi_path "$RUN/wsi_pyramid_tiff/$id.tiff" --wsi_mpp 0.25 \
      >"$LOGS/cellvit_$id.log" 2>"$LOGS/cellvit_$id.err.log" || rc=$?
  secs=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\n' "$id" "$rc" "$secs" >>"$REPORT"
  echo "  exit $rc after ${secs}s"
  [ "$rc" -eq 0 ] || { echo "CellViT failed on $id; see $LOGS/cellvit_$id.err.log"; exit "$rc"; }
done

ckpt=$(find "$CELLVIT_CACHE" -name 'CellViT-SAM-H-x40-AMP.pth' -print -quit 2>/dev/null || true)
if [ -n "$ckpt" ]; then
  echo "# checkpoint $(basename "$ckpt") sha256 $(sha256sum "$ckpt" | cut -c1-64)" >>"$REPORT"
else
  echo "# checkpoint CellViT-SAM-H-x40-AMP.pth not found under CELLVIT_CACHE" >>"$REPORT"
fi
