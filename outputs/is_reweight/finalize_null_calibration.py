r"""Combine the null calibration and the four seed sets into one summary
block, using a theoretical (Hanley-McNeil large-sample Mann-Whitney) null SE
matched to the n_clean=150, n_mis=100 structure of the seed-set amplitude
points. The n=150-vs-150 empirical null-calibration pairs have a slightly
tighter sampling variance and are not mutually independent (28 pairs drawn
from 8 populations, a fully connected graph), so the percentile-of-28 figures
in report_null_percentiles.py understate how correlated that empirical
distribution is. This script adds the like-for-like parametric cross-check
and the pooled-across-seed-sets analysis, which averages out seed-to-seed
draw noise and is the most direct test of whether an amplitude-driven trend
is common to all four seed sets.

Read-only against null_calibration_results.json and
subpercent_sweep_results.json, except that it appends one new top-level key,
"summary", to null_calibration_results.json. No other key written by
run_null_calibration.py is touched.

    .venv\Scripts\python.exe outputs\is_reweight\finalize_null_calibration.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
NULLCAL_PATH = HERE / "null_calibration_results.json"

with open(NULLCAL_PATH) as f:
    NULLCAL = json.load(f)
with open(HERE / "subpercent_sweep_results.json") as f:
    SUBPCT = json.load(f)

STRENGTHS = [0.1, 0.2, 0.3, 0.5]
N_CLEAN, N_MIS = 150, 100

# Large-sample null SE for the Mann-Whitney/AUC statistic under H0 (Hanley &
# McNeil 1982 large-sample limit == normalized-U-statistic variance), matched
# to the ACTUAL sample sizes used at every seed-set amplitude point.
SE_THEORY = math.sqrt((N_CLEAN + N_MIS + 1) / (12.0 * N_CLEAN * N_MIS))

def get_aucs(src, key):
    return {f"{s:g}pct": src[key]["points"][f"{s:g}pct"]["auc_ess_eff"] for s in STRENGTHS}

def get_perm_p(src, key):
    return {f"{s:g}pct": src[key]["points"][f"{s:g}pct"]["perm_p_vs_0p5"] for s in STRENGTHS}

seed_aucs = {
    "seed_set0": get_aucs(SUBPCT, "seed_set0"),
    "seed_set1": get_aucs(SUBPCT, "seed_set1"),
    "seed_set2": get_aucs(NULLCAL, "seed_set2"),
    "seed_set3": get_aucs(NULLCAL, "seed_set3"),
}
seed_perm_p = {
    "seed_set0": get_perm_p(SUBPCT, "seed_set0"),
    "seed_set1": get_perm_p(SUBPCT, "seed_set1"),
    "seed_set2": get_perm_p(NULLCAL, "seed_set2"),
    "seed_set3": get_perm_p(NULLCAL, "seed_set3"),
}

print(f"theoretical null SE (n_clean={N_CLEAN}, n_mis={N_MIS}, large-sample "
      f"Mann-Whitney): {SE_THEORY:.5f}")
print()

# ---- per-point z-score cross-check against each point's own permutation p ----
print("z-score (theoretical SE) vs the machinery's own permutation p (sanity cross-check):")
z_table = {}
for sname, aucs in seed_aucs.items():
    z_table[sname] = {}
    for k, auc in aucs.items():
        z = (auc - 0.5) / SE_THEORY
        p_norm = 2 * (1 - stats.norm.cdf(abs(z)))
        p_perm = seed_perm_p[sname][k]
        z_table[sname][k] = {"auc": auc, "z": z, "p_normal_approx": p_norm, "p_perm": p_perm}
        print(f"  {sname} {k}: AUC={auc:.4f} z={z:+.3f} p_normal={p_norm:.4f} "
              f"p_perm(actual)={p_perm:.4f}")
print()

# ---- count of nominal p<0.05 crossings, and their direction ----
n_sig = 0
sig_list = []
for sname, pp in seed_perm_p.items():
    for k, p in pp.items():
        if p < 0.05:
            n_sig += 1
            direction = "ELEVATED (AUC>0.5)" if seed_aucs[sname][k] > 0.5 else "DEPRESSED (AUC<0.5)"
            sig_list.append((sname, k, p, direction))
print(f"nominal p<0.05 uncorrected crossings: {n_sig} / 16 seed-set*amplitude cells "
      f"(expected ~0.8 under a true null at alpha=0.05)")
for row in sig_list:
    print(f"  {row}")
print()

# ---- pooled-across-seed-sets mean AUC per amplitude (averages out seed noise) ----
print("pooled mean AUC across the 4 independent seed sets, per amplitude "
      "(the most direct test of a REPRODUCIBLE amplitude-driven trend):")
se_pooled = SE_THEORY / math.sqrt(4)
pooled = {}
for k in [f"{s:g}pct" for s in STRENGTHS]:
    vals = [seed_aucs[sname][k] for sname in seed_aucs]
    m = float(np.mean(vals))
    z = (m - 0.5) / se_pooled
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    pooled[k] = {"mean_auc": m, "values": vals, "se_pooled": se_pooled, "z": z, "p_normal_approx": p}
    print(f"  {k}: mean={m:.4f} (values {[f'{v:.4f}' for v in vals]}) "
          f"z={z:+.3f} p={p:.3f}")
print()

# ---- null calibration recap (for the summary block) ----
nc = NULLCAL["null_calibration"]
null_summary = nc["auc_summary"]
null_dev_summary = nc["abs_dev_from_0p5_summary"]

summary = {
    "theoretical_null_se_matched_to_seedset_n": SE_THEORY,
    "z_score_cross_check": z_table,
    "n_nominal_p_lt_0p05_of_16": n_sig,
    "nominal_p_lt_0p05_cells": sig_list,
    "pooled_across_seedsets_per_amplitude": pooled,
    "null_calibration_recap": {
        "auc_summary": null_summary,
        "abs_dev_summary": null_dev_summary,
        "caveat": ("The 28 pairs are not mutually independent: they are all "
                   "C(8,2) pairs drawn from only 8 independent populations, "
                   "and each population appears in 7 pairs, so percentile-of-28 "
                   "figures understate the tail uncertainty. The null-pair "
                   "structure is n=150-vs-150 while the seed-set amplitude "
                   "points are n=150-vs-100 (theoretical SE 0.0334 against "
                   "0.0373), so the theoretical z-score cross-check above is "
                   "the like-for-like comparison and the empirical 28-pair "
                   "null is a supporting sanity check."),
    },
}
NULLCAL["summary"] = summary
with open(NULLCAL_PATH, "w") as f:
    json.dump(NULLCAL, f, indent=2)
print(f"appended 'summary' key to {NULLCAL_PATH}")
