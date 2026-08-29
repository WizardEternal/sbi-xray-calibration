r"""Canonical ESS-efficiency AUCs with uncertainty.

For B4 (3% gain shift) and B1 (Fe-K line) at medium and bright, report the
ESS-efficiency ROC AUC (same detector as run_is_ess_sweep.py: suspicion score
= -ess_frac, since a misspecified spectrum should have LOWER importance-
sampling efficiency) with:
  - a nonparametric bootstrap 95% CI (resample clean/misspec scores with
    replacement, recompute AUC, percentile CI),
  - a permutation p-value against the null AUC=0.5 (shuffle clean/misspec
    labels, recompute AUC, two-sided p = P(|AUC_perm - 0.5| >= |AUC_obs - 0.5|)),
  - across >=2 INDEPENDENT seed sets so the numbers aren't an artifact of one
    particular draw of clean/misspec spectra.

Seed set 0 = reused from the already-committed is_ess_sweep_results.json
main sweep (clean n=150, B4/B1 n=100 each, seeds 80000-83999 range),
no recomputation, just re-analysis of committed IS weights.
Seed set 1 = FRESH population (different global seed offset, disjoint from
every seed used anywhere else in the sweep/amplitude scripts) drawn and
re-weighted from scratch with the same n's and budget.

Uses only the validated run_is_ess_sweep.py functions (Level, is_reweight,
gen_clean, gen_misspec, roc_auc), with no method changes.

Run per level as its own process (mirrors the crash-isolation used for the
main sweep):

    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe outputs\is_reweight\run_canonical_auc.py --level medium
    .venv\Scripts\python.exe outputs\is_reweight\run_canonical_auc.py --level bright
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_is_ess_sweep as R  # noqa: E402
roc_auc = R.roc_auc  # reuse the validated detector; do not reimplement

ROOT = R.ROOT
OUT = R.OUT
N_BUDGET = 6000
N_CLEAN = 150
N_MIS = 100
LEVELS = {"faint": 0, "medium": 1, "bright": 2}
FAM_GRID = {"B4": (3.0, {}), "B1": (3.0e-4, {"line_energy_kev": 6.4, "line_sigma_kev": 0.05})}
FRESH_SEED_BASE = 900000  # disjoint from main sweep (60000-89999 range) and amplitude sweep (950000+)


def bootstrap_auc_ci(clean: np.ndarray, mis: np.ndarray, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    n_c, n_m = clean.size, mis.size
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        c = clean[rng.integers(0, n_c, n_c)]
        m = mis[rng.integers(0, n_m, n_m)]
        _, _, aucs[b] = roc_auc(-c, -m)
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), aucs


def permutation_p(clean: np.ndarray, mis: np.ndarray, n_perm=3000, seed=0):
    _, _, auc_obs = roc_auc(-clean, -mis)
    n_c, n_m = clean.size, mis.size
    pooled = np.concatenate([clean, mis])
    rng = np.random.default_rng(seed)
    dev = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(pooled)
        c, m = perm[:n_c], perm[n_c:]
        _, _, a = roc_auc(-c, -m)
        dev[p] = abs(a - 0.5)
    p_val = float(np.mean(dev >= abs(auc_obs - 0.5)))
    return float(auc_obs), p_val


def analyze(tag: str, clean: np.ndarray, mis: np.ndarray, seed: int):
    _, _, auc = roc_auc(-clean, -mis)
    lo, hi, _ = bootstrap_auc_ci(clean, mis, n_boot=3000, seed=seed + 1)
    auc_chk, pval = permutation_p(clean, mis, n_perm=3000, seed=seed + 2)
    assert abs(auc_chk - auc) < 1e-9
    out = {
        "auc": float(auc), "ci95_lo": lo, "ci95_hi": hi,
        "perm_p_vs_0p5": pval, "n_clean": int(clean.size), "n_mis": int(mis.size),
        "ess_frac_median_clean": float(np.nanmedian(clean)),
        "ess_frac_median_mis": float(np.nanmedian(mis)),
    }
    print(f"    [{tag}] AUC={auc:.4f} 95%CI=[{lo:.4f},{hi:.4f}] perm_p={pval:.4f} "
          f"n_clean={clean.size} n_mis={mis.size}", flush=True)
    return out


def seed_set0_from_sweep(lname: str, fam: str):
    """Pull the already-committed main-sweep ess_frac arrays (no recomputation)."""
    with open(OUT / "is_ess_sweep_results.json") as f:
        res = json.load(f)
    sweep = res["sweep"][lname]
    strength, _ = FAM_GRID[fam]
    clean = np.asarray(sweep["cells"]["clean"]["ess_frac"], dtype=np.float64)
    key = f"{fam}_s{strength:g}"
    mis = np.asarray(sweep["cells"][key]["ess_frac"], dtype=np.float64)
    return clean, mis


def seed_set1_fresh(lev: R.Level, block_idx: int, lname: str):
    """Fresh clean + B4 + B1 population, disjoint seeds from every other script."""
    xc, _ = R.gen_clean(lev, N_CLEAN, seed=FRESH_SEED_BASE + block_idx * 1000)
    clean_eff = np.array([
        R.is_reweight(lev, xc[i], N_BUDGET, seed=FRESH_SEED_BASE + 1000 + block_idx * 10000 + i)["ess_frac"]
        for i in range(N_CLEAN)
    ])
    out = {"clean": clean_eff}
    for fi, (fam, (strength, fixed)) in enumerate(FAM_GRID.items()):
        xm, _ = R.gen_misspec(lev, fam, strength, N_MIS,
                               seed=FRESH_SEED_BASE + 2000 + block_idx * 10000 + fi * 100, fixed=fixed)
        eff = np.array([
            R.is_reweight(lev, xm[i], N_BUDGET,
                           seed=FRESH_SEED_BASE + 3000 + block_idx * 10000 + fi * 1000 + i)["ess_frac"]
            for i in range(N_MIS)
        ])
        out[fam] = eff
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["faint", "medium", "bright"], required=True)
    args = ap.parse_args()
    lname = args.level
    block_idx = LEVELS[lname]

    t0 = time.time()
    print(f"===== CANONICAL AUC: {lname} =====", flush=True)
    lev = R.Level(lname)

    out_path = OUT / "canonical_auc_results.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    level_out = {"seed_set0_source": "reused from main sweep JSON (seeds 80000-83999 range)",
                 "seed_set1_source": f"fresh draw, seed base {FRESH_SEED_BASE}", "families": {}}

    print("  seed set 1 (fresh draw)...", flush=True)
    fresh = seed_set1_fresh(lev, block_idx, lname)

    FAM_SALT = {"B4": 4, "B1": 1}  # stable per-family salt; NEVER use builtin hash()
                                    # here; it is PYTHONHASHSEED-salted per process
                                    # and would make the bootstrap/perm seed (and hence
                                    # the CI) non-reproducible across runs.
    for fam in FAM_GRID:
        print(f"  family {fam}", flush=True)
        c0, m0 = seed_set0_from_sweep(lname, fam)
        seed0_res = analyze(f"{lname}/{fam}/seed_set0", c0, m0, seed=1000 + block_idx * 10 + FAM_SALT[fam])
        c1, m1 = fresh["clean"], fresh[fam]
        seed1_res = analyze(f"{lname}/{fam}/seed_set1", c1, m1, seed=2000 + block_idx * 10 + FAM_SALT[fam])
        level_out["families"][fam] = {"seed_set0": seed0_res, "seed_set1": seed1_res}

    level_out["wall_s"] = time.time() - t0
    all_results[lname] = level_out
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"DONE {lname} in {level_out['wall_s']:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
