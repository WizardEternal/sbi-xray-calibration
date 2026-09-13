#!/usr/bin/env bash
# Phase 2: seed-averaging follow-up.
# Run after the probe's measured scatter is known
# (probe = run_probe_medium.sh / run_probe_bright.sh, single seed, 5 amplitudes).
#
# Usage: run_phase2.sh <level: medium|bright> <amp> <seed_start> <seed_end>
# Loops seed s from seed_start to seed_end inclusive, running the production
# script for <level> at gain g=1+<amp>, one run per seed, into
# BASE/<level>/g<amp>/seed<s>/. Resume-safe: skips a seed if its results JSON
# already exists. Same OMP thread settings as the probe.
set -u
if [ "$#" -ne 4 ]; then
  echo "usage: $0 <level: medium|bright> <amp> <seed_start> <seed_end>" >&2
  exit 2
fi
LEVEL="$1"
AMP="$2"
SEED_START="$3"
SEED_END="$4"

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 1
PY="./.venv/Scripts/python.exe"
BASE="outputs/gain_marg/subpercent_bias_2026-09-08"

case "$LEVEL" in
  medium)
    SCRIPT="outputs/gain_marg/eval_gainmarg_paired.py"
    JSON_NAME="paired_gain_bias_medium.json"
    ;;
  bright)
    SCRIPT="outputs/gain_marg/eval_gainmarg_paired_bright.py"
    JSON_NAME="paired_gain_bias_bright.json"
    ;;
  *)
    echo "level must be 'medium' or 'bright', got: $LEVEL" >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

PROGRESS="$BASE/phase2_${LEVEL}_g${AMP}.progress"
g=$(awk "BEGIN{print 1+$AMP}")

for s in $(seq "$SEED_START" "$SEED_END"); do
  outdir="$BASE/${LEVEL}/g${AMP}/seed${s}"
  mkdir -p "$outdir"
  jsonpath="$outdir/$JSON_NAME"
  if [ -f "$jsonpath" ]; then
    echo "SKIP $LEVEL amp=$AMP seed=$s (json exists: $jsonpath) $(date -u +%FT%TZ)" >> "$PROGRESS"
    continue
  fi
  "$PY" "$SCRIPT" --gain "$g" --n-test 500 --seed "$s" --out "$outdir" > "$outdir/run.log" 2>&1
  code=$?
  echo "DONE $LEVEL g=$g seed=$s $(date -u +%FT%TZ) exit=$code" >> "$PROGRESS"
done

echo "ALL DONE $LEVEL g=$g seeds=${SEED_START}-${SEED_END} $(date -u +%FT%TZ)" >> "$PROGRESS"
