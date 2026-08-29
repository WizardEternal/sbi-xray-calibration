r"""Locate all four seed sets inside the clean-vs-clean null spread.

seed_set0 and seed_set1 come from run_subpercent_sweep.py, seed_set2 and
seed_set3 from run_null_calibration.py. Read-only against
null_calibration_results.json and subpercent_sweep_results.json; it writes
nothing and only prints.

    .venv\Scripts\python.exe outputs\is_reweight\report_null_percentiles.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

with open(HERE / "null_calibration_results.json") as f:
    NULLCAL = json.load(f)
with open(HERE / "subpercent_sweep_results.json") as f:
    SUBPCT = json.load(f)

nc = NULLCAL["null_calibration"]
pair_aucs = np.array([v["auc"] for v in nc["pairs"].values()])
pair_devs = np.abs(pair_aucs - 0.5)

print(f"K={nc['k_null']} populations, {nc['n_pairs']} pairs")
print(f"null AUC: median={np.median(pair_aucs):.4f} mean={np.mean(pair_aucs):.4f} "
      f"std={np.std(pair_aucs, ddof=1):.4f} min={np.min(pair_aucs):.4f} max={np.max(pair_aucs):.4f}")
print(f"null AUC central95%: [{np.percentile(pair_aucs,2.5):.4f},{np.percentile(pair_aucs,97.5):.4f}]")
print(f"null |AUC-0.5|: median={np.median(pair_devs):.4f} p95={np.percentile(pair_devs,95):.4f} "
      f"max={np.max(pair_devs):.4f}")
print(f"frac(|dev|>=0.05, i.e. AUC>=0.55 or <=0.45): {np.mean(pair_devs>=0.05):.3f}")
print(f"frac(|dev|>=0.09, i.e. AUC>=0.59 or <=0.41): {np.mean(pair_devs>=0.09):.3f}")
print()

STRENGTHS = [0.1, 0.2, 0.3, 0.5]


def report_seedset(name, src, key):
    print(f"-- {name} --")
    for s in STRENGTHS:
        p = src[key]["points"][f"{s:g}pct"]
        auc = p["auc_ess_eff"]
        dev = abs(auc - 0.5)
        pctile = float(np.mean(pair_devs <= dev)) * 100  # % of null pairs with SMALLER |dev| than this
        print(f"  {s:g}%: AUC={auc:.4f} [{p['ci95_lo']:.4f},{p['ci95_hi']:.4f}] "
              f"perm_p={p['perm_p_vs_0p5']:.4f} | |AUC-0.5|={dev:.4f}, "
              f"{pctile:.0f}th percentile of the null |AUC-0.5| distribution "
              f"(i.e. {100-pctile:.0f}% of null pairs deviate from 0.5 at least this much)")
    print()


report_seedset("seed_set0 (sub-percent sweep, reused clean)", SUBPCT, "seed_set0")
report_seedset("seed_set1 (sub-percent sweep, fresh clean)", SUBPCT, "seed_set1")
if "seed_set2" in NULLCAL:
    report_seedset("seed_set2 (null-calibration run, fresh clean)", NULLCAL, "seed_set2")
if "seed_set3" in NULLCAL:
    report_seedset("seed_set3 (null-calibration run, fresh clean)", NULLCAL, "seed_set3")

print(f"wall_s_null={NULLCAL.get('wall_s_null')} wall_s_set2={NULLCAL.get('wall_s_set2')} "
      f"wall_s_set3={NULLCAL.get('wall_s_set3')} wall_s_total={NULLCAL.get('wall_s_total')}")
