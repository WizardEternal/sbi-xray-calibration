r"""Sub-percent gain-amplitude sweep (medium, B4). It extends the
amplitude control (run_amplitude_sweep.py, medium/B4, 1/3/5/10%, all
null-consistent: AUC 0.443-0.484, all perm p>=0.13) down into the REALISTIC
detector energy-scale-accuracy band. Real EPIC-pn gain accuracy is ~0.2% at
6 keV and NICER ~0.08%, so the question is whether the blindness holds at
the realistic amplitude and not only at ten times it.

Strength -> gain factor (sbixcal.misspec.simulate_misspec_population, family
B4): gain = 1.0 + strength/100, so
    strength 0.1  -> 0.1% gain
    strength 0.2  -> 0.2% gain  (~ EPIC-pn energy-scale accuracy at 6 keV)
    strength 0.3  -> 0.3% gain
    strength 0.5  -> 0.5% gain  (fills the gap below the committed 1% point;
                                 not covered by the existing 1/3/5/10% grid)

Two independent seed sets (mirrors run_canonical_auc.py's stability-check
pattern, extended to every amplitude on the grid instead of just one point):
  seed_set0 = REUSED clean cell from the committed is_ess_sweep_results.json
              main sweep (n=150, seeds 80000-81999), the same convention
              run_amplitude_sweep.py used, plus a fresh B4 misspec draw per
              amplitude, seed base 970000.
  seed_set1 = a FULLY FRESH clean population (n=150, drawn once, shared
              across all 4 amplitudes) AND a fresh B4 misspec draw per
              amplitude, seed base 980000. All seeds disjoint from every
              other script here (main sweep 60000-89999, canonical_auc
              900000+, run_amplitude_sweep.py 950000-~957000).

Uses ONLY the validated run_is_ess_sweep.py / run_canonical_auc.py functions
(Level, is_reweight, gen_clean, gen_misspec, roc_auc, bootstrap_auc_ci,
permutation_p), with no method changes and no reimplementation. Does not modify
run_amplitude_sweep.py, run_is_ess_sweep.py, run_canonical_auc.py, or
amplitude_sweep_results.json. Writes a NEW file,
outputs/is_reweight/subpercent_sweep_results.json.

Crash isolation: pass --seedset {0,1} to run just that seed set as its own
process (merge-safe write into the same JSON), mirroring the --level flag
added to run_is_ess_sweep.py after an earlier real crash from resource growth
across a long multi-stage process. Default (no flag) runs both seed sets
sequentially in one process; the observed cost of the original 4-point
amplitude sweep (167s, one process, no crash) suggests this is low-risk, but
the flag is here in case it is not.

    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe outputs\is_reweight\run_subpercent_sweep.py
    .venv\Scripts\python.exe outputs\is_reweight\run_subpercent_sweep.py --seedset 0
    .venv\Scripts\python.exe outputs\is_reweight\run_subpercent_sweep.py --seedset 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_is_ess_sweep as R  # noqa: E402
from run_canonical_auc import bootstrap_auc_ci, permutation_p  # noqa: E402

ROOT = R.ROOT
OUT = R.OUT
N_MIS = 100
N_CLEAN = 150
N_BUDGET = 6000
LEVEL = "medium"
STRENGTHS = [0.1, 0.2, 0.3, 0.5]  # gain percent
SEED_BASE0 = 970000  # seed_set0: reused clean cell + fresh B4 draw
SEED_BASE1 = 980000  # seed_set1: fully fresh clean + fresh B4 draw


def _one_point(lev, clean, strength, si, seed_base):
    xm, _ = R.gen_misspec(lev, "B4", strength, N_MIS, seed=seed_base + si * 1000)
    mis = np.array([
        R.is_reweight(lev, xm[i], N_BUDGET, seed=seed_base + 1000 + si * 1000 + i)["ess_frac"]
        for i in range(N_MIS)
    ])
    _, _, auc = R.roc_auc(-clean, -mis)
    lo, hi, _ = bootstrap_auc_ci(clean, mis, n_boot=3000, seed=seed_base + 2000 + si)
    auc_chk, pval = permutation_p(clean, mis, n_perm=3000, seed=seed_base + 3000 + si)
    assert abs(auc_chk - auc) < 1e-9
    return {
        "gain_factor": 1.0 + strength / 100.0,
        "auc_ess_eff": float(auc), "ci95_lo": lo, "ci95_hi": hi,
        "perm_p_vs_0p5": pval,
        "ess_frac_median_mis": float(np.nanmedian(mis)),
        "ess_frac_median_clean": float(np.nanmedian(clean)),
        "n_mis": N_MIS, "n_clean": int(clean.size),
    }


def run_seedset0(lev, out_path, results):
    """Reused clean cell (n=150, committed) + fresh B4 draw per amplitude."""
    with open(OUT / "is_ess_sweep_results.json") as f:
        sweep = json.load(f)
    clean = np.asarray(sweep["sweep"][LEVEL]["cells"]["clean"]["ess_frac"], dtype=np.float64)
    print(f"    reused clean cell: n={clean.size} median_ess_frac={np.nanmedian(clean):.4f}", flush=True)
    pts = {}
    for si, strength in enumerate(STRENGTHS):
        print(f"    strength={strength:g}% ...", flush=True)
        pts[f"{strength:g}pct"] = _one_point(lev, clean, strength, si, SEED_BASE0)
        p = pts[f"{strength:g}pct"]
        print(f"      AUC={p['auc_ess_eff']:.4f} 95%CI=[{p['ci95_lo']:.4f},{p['ci95_hi']:.4f}] "
              f"perm_p={p['perm_p_vs_0p5']:.4f} eff_mis_med={p['ess_frac_median_mis']:.4f} "
              f"(clean {np.nanmedian(clean):.4f})", flush=True)
        results.setdefault("seed_set0", {})["points"] = pts
        results["seed_set0"]["source"] = ("clean cell reused from committed "
            "is_ess_sweep_results.json (n=150, seeds 80000-81999); fresh B4 misspec "
            f"draw per amplitude, seed base {SEED_BASE0}")
        results["seed_set0"]["clean_n"] = int(clean.size)
        results["seed_set0"]["clean_ess_frac_median"] = float(np.nanmedian(clean))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)  # incremental crash-safe save
    return results


def run_seedset1(lev, out_path, results):
    """Fully fresh clean (n=150, drawn once) + fresh B4 draw per amplitude."""
    xc, _ = R.gen_clean(lev, N_CLEAN, seed=SEED_BASE1)
    clean = np.array([
        R.is_reweight(lev, xc[i], N_BUDGET, seed=SEED_BASE1 + 1000 + i)["ess_frac"]
        for i in range(N_CLEAN)
    ])
    print(f"    fresh clean cell: n={clean.size} median_ess_frac={np.nanmedian(clean):.4f}", flush=True)
    pts = {}
    for si, strength in enumerate(STRENGTHS):
        print(f"    strength={strength:g}% ...", flush=True)
        pts[f"{strength:g}pct"] = _one_point(lev, clean, strength, si, SEED_BASE1 + 2000)
        p = pts[f"{strength:g}pct"]
        print(f"      AUC={p['auc_ess_eff']:.4f} 95%CI=[{p['ci95_lo']:.4f},{p['ci95_hi']:.4f}] "
              f"perm_p={p['perm_p_vs_0p5']:.4f} eff_mis_med={p['ess_frac_median_mis']:.4f} "
              f"(clean {np.nanmedian(clean):.4f})", flush=True)
        results.setdefault("seed_set1", {})["points"] = pts
        results["seed_set1"]["source"] = ("fully fresh clean (n=150, drawn once, shared "
            f"across amplitudes) + fresh B4 misspec draw per amplitude, seed base {SEED_BASE1 + 2000}"
            f" (clean draw seed base {SEED_BASE1})")
        results["seed_set1"]["clean_n"] = int(clean.size)
        results["seed_set1"]["clean_ess_frac_median"] = float(np.nanmedian(clean))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)  # incremental crash-safe save
    return results


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedset", choices=["0", "1"], default=None,
                     help="Run only this seed set as its own process (crash isolation); "
                          "merges into the existing subpercent_sweep_results.json instead "
                          "of overwriting the other seed set.")
    args = ap.parse_args()

    print(f"===== SUB-PERCENT AMPLITUDE SWEEP: {LEVEL}, B4 gain, strengths={STRENGTHS} =====",
          flush=True)
    lev = R.Level(LEVEL)
    print(f"  flow loaded, ~{lev.median_counts:.0f} median counts", flush=True)

    out_path = OUT / "subpercent_sweep_results.json"
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {"level": LEVEL, "n_mis": N_MIS, "n_clean": N_CLEAN, "n_budget": N_BUDGET,
                   "strengths_pct": STRENGTHS}

    do0 = args.seedset in (None, "0")
    do1 = args.seedset in (None, "1")

    if do0:
        t0 = time.time()
        print("  ---- seed_set0 (reused clean cell) ----", flush=True)
        results = run_seedset0(lev, out_path, results)
        results["wall_s_seedset0"] = time.time() - t0
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [saved] seed_set0 done in {results['wall_s_seedset0']:.1f}s", flush=True)

    if do1:
        t1 = time.time()
        print("  ---- seed_set1 (fully fresh) ----", flush=True)
        results = run_seedset1(lev, out_path, results)
        results["wall_s_seedset1"] = time.time() - t1
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [saved] seed_set1 done in {results['wall_s_seedset1']:.1f}s", flush=True)

    results["wall_s_total"] = results.get("wall_s_seedset0", 0.0) + results.get("wall_s_seedset1", 0.0)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"DONE (this process) total {time.time()-t_start:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
