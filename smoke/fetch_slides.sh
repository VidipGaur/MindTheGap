#!/usr/bin/env bash
# Download train.csv, test.csv and the panel's PNG slides from the UBC-OCEAN
# Kaggle competition into $UBC_ROOT, one file at a time.
#
# Needs the Kaggle CLI (pip install kaggle), credentials in ~/.kaggle/kaggle.json
# (mode 600) or KAGGLE_USERNAME / KAGGLE_KEY in the environment, and the
# competition rules accepted on that account. Credentials are never printed.
set -euo pipefail

: "${UBC_ROOT:?set UBC_ROOT to the smoke dataset root}"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
COMP=UBC-OCEAN
MIN_FREE_GB=${MIN_FREE_GB:-60}
read -r -a IDS <<<"$("$PY" -c 'import json,sys; print(" ".join(s["image_id"] for s in json.load(open(sys.argv[1]))["slides"]))' "$HERE/slides.json")"

# Slides placed by hand need no Kaggle access; test.csv is optional.
have_all=1
[ -f "$UBC_ROOT/train.csv" ] || have_all=0
for id in "${IDS[@]}"; do [ -f "$UBC_ROOT/train_images/$id.png" ] || have_all=0; done
if [ "$have_all" = 1 ]; then
  echo "train.csv and all panel PNGs already under $UBC_ROOT; nothing to fetch"
  exit 0
fi

command -v kaggle >/dev/null || { echo "kaggle CLI not found; install it with: pip install kaggle"; exit 2; }
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && { [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; }; then
  echo "no Kaggle credentials: put kaggle.json in ~/.kaggle (chmod 600) or export KAGGLE_USERNAME and KAGGLE_KEY"
  exit 2
fi

mkdir -p "$UBC_ROOT/train_images"
free_gb=$(df -BG --output=avail "$UBC_ROOT" | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  echo "only ${free_gb} GB free under $UBC_ROOT; at least ${MIN_FREE_GB} GB are needed"
  exit 2
fi

# A metadata call first, so a missing rules acceptance stops the run before any download.
err=$(mktemp)
if ! kaggle competitions files -c "$COMP" >/dev/null 2>"$err"; then
  echo "Kaggle refused the metadata call for $COMP:"
  sed 's/^/  /' "$err"
  echo "If this is a 403, accept the rules at https://www.kaggle.com/competitions/$COMP/rules"
  echo "with the same account, then run this script again."
  rm -f "$err"
  exit 3
fi
rm -f "$err"

fetch() {  # fetch <file in the competition> <destination folder>
  local name
  name=$(basename "$1")
  if [ -f "$2/$name" ]; then echo "have $2/$name"; return; fi
  kaggle competitions download -c "$COMP" -f "$1" -p "$2" --quiet
  if [ -f "$2/$name.zip" ]; then
    unzip -q -o "$2/$name.zip" -d "$2" && rm "$2/$name.zip"
  fi
  [ -f "$2/$name" ] || { echo "downloading $1 did not produce $2/$name"; exit 4; }
  echo "got $2/$name ($(du -h "$2/$name" | cut -f1))"
}

fetch train.csv "$UBC_ROOT"
fetch test.csv "$UBC_ROOT"
for id in "${IDS[@]}"; do
  fetch "train_images/$id.png" "$UBC_ROOT/train_images"
done
