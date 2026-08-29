r"""Placement-flag (q) recompute: in-prior acceptance on the CANONICAL ESS-AUC
populations (n=150 clean / n=100 misspec per cell, medium + bright, family B4
only [the 3% gain shift the paper's ESS/AUC claim is about], seed set 0 +
seed set 1).

Why this script exists: the acceptance range quoted in the paper,

    "Acceptance of the fixed-budget in-prior weighting scheme sits at 0.91 to
    0.97, close between clean and gain-shifted populations, so nothing below
    is an artifact of differential rejection."

was first measured on a smaller focused run (clean n=50-60, misspec n=50).
This script recomputes it on the canonical population, the one that produced
the committed ESS AUCs in canonical_auc_results.json and
is_ess_sweep_results.json, so the quoted range is measured where the AUCs
are measured.

Definition of "acceptance" used (copied verbatim, not reimplemented, from the
committed pipeline, outputs/is_reweight/run_is_ess_sweep.py):

    inside = np.all((s >= level.lo) & (s <= level.hi), axis=1)   # line 156
    n_in = int(inside.sum())                                     # line 157
    ...
    "acc": float(n_in / n_budget),                                # line 181

i.e. acceptance = fraction of the FIXED RAW BUDGET of flow draws (n_budget =
6000, reject_outside_prior=False) that land inside the uniform prior box, for
one spectrum. This is NOT a rejection-sampling acceptance rate to fill a
target n; it is a leakage-rate diagnostic per spectrum (see run_is_ess_sweep.py
docstring, "Prior leakage").

Two seed sets, both reusing the committed generator/reweight functions
(R.Level, R.gen_clean, R.gen_misspec, R.is_reweight) with NO method changes:

  seed_set0: exact reproduction of the main-sweep seeds used to build
    outputs/is_reweight/is_ess_sweep_results.json (clean seed 80000+block_idx,
    is_reweight seed 81000+block_idx*1000+i; B4 is fi=1 in that script's MIS
    list [B1,B4,B2,B3], so gen_misspec seed 82000+block_idx*1000+100,
    is_reweight seed 83000+block_idx*10000+1000+i). That JSON already stores a
    per-spectrum "acc" field for this exact population, so recomputing it here
    both gives the acceptance numbers directly AND is a reproducibility check
    (recomputed acc must match the committed acc arrays to numerical
    precision, since torch.manual_seed is reset inside is_reweight per call).

  seed_set1: exact reproduction of run_canonical_auc.py's "fresh draw" seeds
    (FRESH_SEED_BASE=900000; note FAM_GRID there is {"B4":..., "B1":...} so B4
    is fi=0, not fi=1. This matters for seed reproducibility and is easy to
    get wrong by copying the main-sweep fi).

Levels: medium, bright (per the flag; faint is out of scope for this claim).
Families: clean control + B4 (3% gain) only. B1/B2/B3 are not part of the
sentence being checked.

Crash-safe: results are written to acceptance_canonical_results.json after
EVERY population (8 total: {medium,bright} x {seed_set0,seed_set1} x
{clean,B4}), so a crash mid-run loses at most one population's compute.

Run:
    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe outputs\is_reweight\acceptance_canonical.py --level medium
    .venv\Scripts\python.exe outputs\is_reweight\acceptance_canonical.py --level bright
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_is_ess_sweep as R  # noqa: E402  (reuse committed Level/gen_clean/gen_misspec/is_reweight)

ROOT = R.ROOT
OUT = R.OUT
RESULTS_PATH = OUT / "acceptance_canonical_results.json"
SWEEP_JSON = OUT / "is_ess_sweep_results.json"

N_BUDGET = 6000
N_CLEAN = 150
N_MIS = 100
LEVELS = {"faint": 0, "medium": 1, "bright": 2}
FRESH_SEED_BASE = 900000  # == run_canonical_auc.py's FRESH_SEED_BASE

# B4's position in run_is_ess_sweep.py's own MIS list ([B1, B4, B2, B3] -> fi=1).
# Needed to reproduce the seed_set0 (main-sweep) draw exactly.
MAIN_SWEEP_FI_B4 = 1
# B4's position in run_canonical_auc.py's FAM_GRID dict ({"B4":..., "B1":...} -> fi=0).
# Needed to reproduce the seed_set1 (canonical_auc fresh draw) exactly.
FRESH_FI_B4 = 0


def stats(acc: np.ndarray) -> dict:
    acc = np.asarray(acc, dtype=np.float64)
    return {
        "n": int(acc.size),
        "median": float(np.median(acc)),
        "mean": float(np.mean(acc)),
        "min": float(np.min(acc)),
        "max": float(np.max(acc)),
        "std": float(np.std(acc)),
    }


def recompute_seed_set0(lev: R.Level, block_idx: int) -> dict:
    """Exact reproduction of the main-sweep seeds in run_is_ess_sweep.py:main()."""
    xc, _ = R.gen_clean(lev, N_CLEAN, seed=80000 + block_idx)
    acc_clean = np.array([
        R.is_reweight(lev, xc[i], N_BUDGET, seed=81000 + block_idx * 1000 + i)["acc"]
        for i in range(N_CLEAN)
    ])
    xm, _ = R.gen_misspec(lev, "B4", 3.0, N_MIS,
                           seed=82000 + block_idx * 1000 + MAIN_SWEEP_FI_B4 * 100, fixed={})
    acc_b4 = np.array([
        R.is_reweight(lev, xm[i], N_BUDGET,
                       seed=83000 + block_idx * 10000 + MAIN_SWEEP_FI_B4 * 1000 + i)["acc"]
        for i in range(N_MIS)
    ])
    return {"clean": acc_clean, "B4": acc_b4}


def recompute_seed_set1(lev: R.Level, block_idx: int) -> dict:
    """Exact reproduction of run_canonical_auc.py's seed_set1_fresh() seeds."""
    xc, _ = R.gen_clean(lev, N_CLEAN, seed=FRESH_SEED_BASE + block_idx * 1000)
    acc_clean = np.array([
        R.is_reweight(lev, xc[i], N_BUDGET,
                       seed=FRESH_SEED_BASE + 1000 + block_idx * 10000 + i)["acc"]
        for i in range(N_CLEAN)
    ])
    xm, _ = R.gen_misspec(lev, "B4", 3.0, N_MIS,
                           seed=FRESH_SEED_BASE + 2000 + block_idx * 10000 + FRESH_FI_B4 * 100,
                           fixed={})
    acc_b4 = np.array([
        R.is_reweight(lev, xm[i], N_BUDGET,
                       seed=FRESH_SEED_BASE + 3000 + block_idx * 10000 + FRESH_FI_B4 * 1000 + i)["acc"]
        for i in range(N_MIS)
    ])
    return {"clean": acc_clean, "B4": acc_b4}


def committed_acc_check(lname: str) -> dict | None:
    """Cross-check: pull the "acc" arrays already stored in the committed
    is_ess_sweep_results.json for this exact population (same seeds as
    seed_set0) and compare to our freshly recomputed seed_set0 arrays.
    Returns None if the committed file/level is missing."""
    if not SWEEP_JSON.exists():
        return None
    with open(SWEEP_JSON) as f:
        d = json.load(f)
    if lname not in d.get("sweep", {}):
        return None
    cells = d["sweep"][lname]["cells"]
    return {
        "clean": np.asarray(cells["clean"]["acc"], dtype=np.float64),
        "B4": np.asarray(cells["B4_s3"]["acc"], dtype=np.float64),
    }


def load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_population(all_results: dict, lname: str, seed_set: str, fam: str, acc: np.ndarray,
                     extra: dict | None = None):
    all_results.setdefault(lname, {}).setdefault(seed_set, {})
    entry = stats(acc)
    entry["acc_values"] = [float(a) for a in acc]
    if extra:
        entry.update(extra)
    all_results[lname][seed_set][fam] = entry
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"    [saved] {lname}/{seed_set}/{fam}: "
          f"median={entry['median']:.4f} mean={entry['mean']:.4f} "
          f"min={entry['min']:.4f} max={entry['max']:.4f} (n={entry['n']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["medium", "bright"], required=True)
    args = ap.parse_args()
    lname = args.level
    block_idx = LEVELS[lname]

    t0 = time.time()
    print(f"===== ACCEPTANCE RECOMPUTE (canonical): {lname} =====", flush=True)
    lev = R.Level(lname)

    all_results = load_results()
    all_results.setdefault(lname, {})
    all_results[lname]["definition_source"] = (
        "outputs/is_reweight/run_is_ess_sweep.py lines 156-157,181, "
        "acc = fraction of n_budget=6000 raw flow draws (reject_outside_prior=False) "
        "landing inside the uniform prior box, per spectrum"
    )
    all_results[lname]["median_counts"] = lev.median_counts

    # --- seed_set0: exact reproduction of the main-sweep population ---
    print("  seed_set0 (reproduce main-sweep seeds)...", flush=True)
    s0 = recompute_seed_set0(lev, block_idx)
    for fam in ("clean", "B4"):
        save_population(all_results, lname, "seed_set0", fam, s0[fam])

    # cross-check against the already-committed acc arrays in is_ess_sweep_results.json
    committed = committed_acc_check(lname)
    if committed is not None:
        for fam in ("clean", "B4"):
            recomputed = s0[fam]
            stored = committed[fam]
            if recomputed.size == stored.size:
                maxabsdiff = float(np.max(np.abs(recomputed - stored)))
            else:
                maxabsdiff = float("nan")
            check = {
                "stored_n": int(stored.size),
                "recomputed_n": int(recomputed.size),
                "max_abs_diff": maxabsdiff,
                "exact_match": bool(recomputed.size == stored.size and maxabsdiff < 1e-12),
            }
            all_results[lname].setdefault("seed_set0_reproducibility_check", {})[fam] = check
            print(f"    [repro-check] {lname}/{fam}: stored_n={check['stored_n']} "
                  f"recomputed_n={check['recomputed_n']} max_abs_diff={maxabsdiff:.3g} "
                  f"exact_match={check['exact_match']}", flush=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        print(f"    [repro-check] committed is_ess_sweep_results.json has no '{lname}' "
              f"sweep entry, skipping cross-check", flush=True)

    # --- seed_set1: exact reproduction of run_canonical_auc.py's fresh draw ---
    print("  seed_set1 (reproduce canonical_auc.py fresh-draw seeds)...", flush=True)
    s1 = recompute_seed_set1(lev, block_idx)
    for fam in ("clean", "B4"):
        save_population(all_results, lname, "seed_set1", fam, s1[fam])

    all_results[lname]["wall_s"] = time.time() - t0
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"DONE {lname} in {all_results[lname]['wall_s']:.0f}s -> {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
