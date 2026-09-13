#!/bin/bash
# Sequential, crash-resistant driver for the 3 per-source paired evals.
# One eval at a time (memory-bound). Each writes a per-run DONE marker
# and EXITCODE so a killed driver can resume-poll instead of
# re-launching. Never overwrites an existing run (checked before each launch).
set -u
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=4
PY=".venv/Scripts/python.exe"
BASE="outputs/gain_marg/paired_per_source_2026-09-03"

run_one () {
  name="$1"; shift
  outdir="$BASE/$name"
  if [ -f "$outdir/DONE" ]; then
    echo "[skip] $name already DONE"
    return 0
  fi
  echo "[start] $name $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  t0=$(date +%s)
  "$PY" "$@" > "$outdir/run.log" 2>&1
  ec=$?
  t1=$(date +%s)
  echo "$ec" > "$outdir/EXITCODE"
  echo "$((t1-t0))" > "$outdir/WALL_S"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$outdir/DONE"
  echo "[done] $name exit=$ec wall=$((t1-t0))s"
}

run_one bright_uncapped outputs/gain_marg/eval_gainmarg_paired_bright.py \
  --fixed-dir outputs/models/gonogo_uncapped_bright \
  --out "$BASE/bright_uncapped" --save-per-source

run_one bright_production outputs/gain_marg/eval_gainmarg_paired_bright.py \
  --out "$BASE/bright_production" --save-per-source

run_one medium outputs/gain_marg/eval_gainmarg_paired.py \
  --out "$BASE/medium" --save-per-source

date -u +%Y-%m-%dT%H:%M:%SZ > "$BASE/ALL_DONE"
echo "[all done]"
