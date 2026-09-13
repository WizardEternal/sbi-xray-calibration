#!/usr/bin/env bash
# Regression test for the --seed edit.
# Runs the MEDIUM production script at --gain 1.03 (the pre-existing default
# shift) --n-test 500 with the DEFAULT seed (no --seed passed) into
# BASE/regression_test/medium_g0.03_default/, then compares the reported
# paired Gamma-bias fields (mean, sd, se, n) against the shipped, committed
# 3% medium JSON.
#
# n must match exactly; the three float fields are compared at a relative
# tolerance of 1e-5. The spectra and the flow run in float32, whose epsilon is
# 1.2e-7, so run-to-run accumulation noise of a few ulps is the size to expect:
# the measured relative differences are 1.8e-7 to 3.5e-7, e.g. fixed.mean
# 0.018299196600914003 against 0.018299203038215636. seed_theta (20320611) and
# seed_poisson_base (20330611) are identical in both files, so the --seed edit
# is a no-op at the default.
set -u
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 1
PY="./.venv/Scripts/python.exe"
BASE="outputs/gain_marg/subpercent_bias_2026-09-08"
OUTDIR="$BASE/regression_test/medium_g0.03_default"
mkdir -p "$OUTDIR"

# Shipped 3% medium JSON:
#   outputs/gain_marg/paired_gain_bias_medium.json  (= e4b_paired_results.json)
SHIPPED="outputs/gain_marg/paired_gain_bias_medium.json"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "[regression] running eval_gainmarg_paired.py --gain 1.03 --n-test 500 (default seed) -> $OUTDIR"
"$PY" outputs/gain_marg/eval_gainmarg_paired.py \
  --gain 1.03 --n-test 500 --out "$OUTDIR" \
  > "$OUTDIR/run.log" 2>&1
code=$?
echo "[regression] exit=$code"

NEW="$OUTDIR/paired_gain_bias_medium.json"

echo "[regression] comparing paired.gamma_bias_delta.{fixed,gain_marg}.{mean,sd,se,n} against $SHIPPED"
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
        match = (ov == nv) if k == 'n' else (abs(ov - nv) <= 1e-5 * max(abs(ov), abs(nv), 1e-12))
        all_match = all_match and match
        print(f'  {k:5s} old={ov!r:>22} new={nv!r:>22}  {\"MATCH\" if match else \"MISMATCH\"}')
    print(f'  ALL FIELDS MATCH: {all_match}')
print()
print('old config.seed_theta/seed_poisson_base:', old['config'].get('seed_theta'), old['config'].get('seed_poisson_base'))
print('new config.seed/seed_theta/seed_poisson_base:', new['config'].get('seed'), new['config'].get('seed_theta'), new['config'].get('seed_poisson_base'))
"
