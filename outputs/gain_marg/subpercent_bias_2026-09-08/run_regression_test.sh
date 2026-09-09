#!/usr/bin/env bash
# Regression test for the --seed edit (STEP 4): NOT launched by this campaign.
# Runs the MEDIUM production script at --gain 1.03 (the pre-existing default
# shift) --n-test 500 with the DEFAULT seed (no --seed passed) into
# BASE/regression_test/medium_g0.03_default/, then compares the headline
# paired Gamma-bias fields (mean, sd, se, n) against the shipped, committed
# 3% medium JSON. If the --seed edit is truly a no-op at the default, these
# must match to floating-point precision (same seed, same code path modulo
# the added argument plumbing).
set -u
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 1
PY="./.venv/Scripts/python.exe"
BASE="outputs/gain_marg/subpercent_bias_2026-09-08"
OUTDIR="$BASE/regression_test/medium_g0.03_default"
mkdir -p "$OUTDIR"

# Shipped 3% medium JSON (scoping report ITEM 1(a)):
#   C\outputs\gain_marg\paired_gain_bias_medium.json (= e4b_paired_results.json)
SHIPPED="outputs/gain_marg/paired_gain_bias_medium.json"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "[regression] running eval_gainmarg_paired.py --gain 1.03 --n-test 500 (default seed) -> $OUTDIR"
"$PY" outputs/gain_marg/eval_gainmarg_paired.py \
  --gain 1.03 --n-test 500 --out "$OUTDIR" \
  > "$OUTDIR/run.log" 2>&1
code=$?
echo "[regression] exit=$code"

NEW="$OUTDIR/paired_gain_bias_medium.json"

echo "[regression] comparing headline paired.gamma_bias_delta.{fixed,gain_marg}.{mean,sd,se,n} against $SHIPPED"
"$PY" -c "
import json
with open(r'$SHIPPED') as f:
    old = json.load(f)
with open(r'$NEW') as f:
    new = json.load(f)
fields = ['mean', 'sd', 'se', 'n']
for flow in ('fixed', 'gain_marg'):
    o = old['paired']['gamma_bias_delta'][flow]
    n = new['paired']['gamma_bias_delta'][flow]
    print(f'--- {flow} ---')
    all_match = True
    for k in fields:
        ov, nv = o[k], n[k]
        match = (ov == nv)
        all_match = all_match and match
        print(f'  {k:5s} old={ov!r:>22} new={nv!r:>22}  {\"MATCH\" if match else \"MISMATCH\"}')
    print(f'  ALL FIELDS MATCH: {all_match}')
print()
print('old config.seed_theta/seed_poisson_base:', old['config'].get('seed_theta'), old['config'].get('seed_poisson_base'))
print('new config.seed/seed_theta/seed_poisson_base:', new['config'].get('seed'), new['config'].get('seed_theta'), new['config'].get('seed_poisson_base'))
"
