#!/usr/bin/env bash
# Probe: sub-percent paired Gamma-bias, BRIGHT level, single seed (default 20260611).
# Resume-safe: skips an amplitude if its results JSON already exists.
set -u
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 1
PY="./.venv/Scripts/python.exe"
BASE="outputs/gain_marg/subpercent_bias_2026-09-08"
SCRIPT="outputs/gain_marg/eval_gainmarg_paired_bright.py"
LEVEL="bright"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

PROGRESS="$BASE/probe_${LEVEL}.progress"

for amp in 0.001 0.002 0.003 0.005 0.01; do
  outdir="$BASE/${LEVEL}/g${amp}"
  mkdir -p "$outdir"
  jsonpath="$outdir/paired_gain_bias_bright.json"
  if [ -f "$jsonpath" ]; then
    echo "SKIP $LEVEL amp=$amp (json exists: $jsonpath) $(date -u +%FT%TZ)" >> "$PROGRESS"
    continue
  fi
  g=$(awk "BEGIN{print 1+$amp}")
  "$PY" "$SCRIPT" --gain "$g" --n-test 500 --out "$outdir" > "$outdir/run.log" 2>&1
  code=$?
  echo "DONE $LEVEL g=$g $(date -u +%FT%TZ) exit=$code" >> "$PROGRESS"
done

echo "ALL DONE $LEVEL $(date -u +%FT%TZ)" >> "$PROGRESS"
