r"""Clean-vs-clean null calibration for the sub-percent gain sweep, plus two
more independent seed sets at the flagged sub-percent amplitudes. It settles
the seed_set0/seed_set1 disagreement left open by run_subpercent_sweep.py.

Hypothesis under test: seed_set1's elevated sub-percent AUCs (0.53-0.59,
2 of 4 nominal p<0.05) are clean-population DRAW NOISE, not a real
gain-shift signal, because seed_set1's fresh clean reference has 36%
higher median ESS-efficiency than seed_set0's reused reference, and that
alone can shift a clean-vs-misspec AUC by a comparable amount. Counter-
hypothesis (a real finding, would matter for the paper): sub-percent gain
is genuinely weakly detectable.

Three things, reusing ONLY the already-validated machinery (Level,
gen_clean, gen_misspec, is_reweight, roc_auc from run_is_ess_sweep.py;
bootstrap_auc_ci, permutation_p from run_canonical_auc.py), with no method
changes and no reimplementation:

  (1) NULL CALIBRATION: draw K independent fresh clean populations (same
      Level="medium", n=150, budget=6000 as every sub-percent population) and
      compute ESS-efficiency AUC for every pairwise clean-vs-clean
      combination (K=8 -> C(8,2)=28 pairs). Under the true null (both
      populations really are draws from the SAME clean generative process)
      this gives the empirical sampling distribution of |AUC-0.5| from
      finite-n draw noise alone, with NO injected misspecification at all.
  (2) SEED SETS 2 & 3: two more fully independent runs (fresh clean AND
      fresh B4 misspec draws, own seed blocks) at all four sub-percent
      amplitudes (0.1/0.2/0.3/0.5%), same statistic/CI/permutation
      procedure as run_subpercent_sweep.py's seed_set1.
  (3) Placement: locate seed_set0, seed_set1, seed_set2, seed_set3's AUCs
      inside the calibrated null spread from (1).

Seeds are plain integer offsets, disjoint from every other script's range
(main sweep 60000-89999, canonical_auc 900000-925000ish, amplitude_sweep
950000-957000, subpercent_sweep 970000-990000ish). This script uses
1,000,000+:
  null calibration: 1,000,000 + p*10,000  for p in range(K_NULL)   (population draw)
  seed_set2:         1,200,000 + ...                                (mirrors seed_set1's pattern)
  seed_set3:         1,300,000 + ...

Does NOT modify run_is_ess_sweep.py, run_canonical_auc.py,
run_amplitude_sweep.py, run_subpercent_sweep.py, or any of their output
JSONs. Writes a new file, outputs/is_reweight/null_calibration_results.json.

Crash isolation (mirrors the --level / --seedset pattern used elsewhere):
pass --stage {null,set2,set3} to run just that stage as its own process,
merge-safe into the shared JSON. Default (no flag) runs all three stages
sequentially, saving incrementally after every population / every
amplitude point.

    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe outputs\is_reweight\run_null_calibration.py
    .venv\Scripts\python.exe outputs\is_reweight\run_null_calibration.py --stage null
    .venv\Scripts\python.exe outputs\is_reweight\run_null_calibration.py --stage set2
    .venv\Scripts\python.exe outputs\is_reweight\run_null_calibration.py --stage set3

Reduced-size smoke test (fast, correctness check only, writes a separate
file so it can never collide with the real run):
    .venv\Scripts\python.exe outputs\is_reweight\run_null_calibration.py \
        --stage null --k-null 2 --n-clean 5 --n-budget 200 \
        --out outputs\is_reweight\null_calibration_smoketest.json
"""
from __future__ import annotations

import argparse
import itertools
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
LEVEL = "medium"
N_BUDGET_DEFAULT = 6000
N_CLEAN_DEFAULT = 150
N_MIS_DEFAULT = 100
K_NULL_DEFAULT = 8
STRENGTHS = [0.1, 0.2, 0.3, 0.5]  # gain percent, same grid as the sub-percent sweep

SEED_BASE_NULL = 1_000_000   # K independent clean populations, 10000 apart
SEED_BASE_SET2 = 1_200_000   # seed_set2: fresh clean + fresh B4 draws, all 4 amps
SEED_BASE_SET3 = 1_300_000   # seed_set3: fresh clean + fresh B4 draws, all 4 amps


def draw_clean_population(lev, base_seed, n_clean, n_budget):
    """Fresh clean draw + IS reweight, same recipe as run_subpercent_sweep's
    seed_set1 / run_canonical_auc's seed_set1_fresh: gen_clean(seed=base_seed)
    (uses base_seed, base_seed+1 internally), then is_reweight per sample at
    base_seed+1000+i."""
    xc, _ = R.gen_clean(lev, n_clean, seed=base_seed)
    eff = np.array([
        R.is_reweight(lev, xc[i], n_budget, seed=base_seed + 1000 + i)["ess_frac"]
        for i in range(n_clean)
    ])
    return eff


def pair_stats(eff_a, eff_b, seed):
    """ESS-eff AUC + CI + perm-p for one clean-vs-clean pair, treating a as
    the negative/'clean' class and b as the positive/'mis' class (same
    -ess_frac suspicion-score convention as every other script here)."""
    _, _, auc = R.roc_auc(-eff_a, -eff_b)
    lo, hi, _ = bootstrap_auc_ci(eff_a, eff_b, n_boot=3000, seed=seed + 1)
    auc_chk, pval = permutation_p(eff_a, eff_b, n_perm=3000, seed=seed + 2)
    assert abs(auc_chk - auc) < 1e-9
    return {"auc": float(auc), "ci95_lo": lo, "ci95_hi": hi, "perm_p_vs_0p5": pval}


def save(results, out_path):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def run_null_stage(lev, results, out_path, k_null, n_clean, n_budget):
    print(f"  ---- null calibration: K={k_null} independent clean populations ----", flush=True)
    pops = {}
    medians = {}
    for p in range(k_null):
        t0 = time.time()
        base = SEED_BASE_NULL + p * 10_000
        eff = draw_clean_population(lev, base, n_clean, n_budget)
        pops[p] = eff.tolist()
        medians[p] = float(np.nanmedian(eff))
        print(f"    pop {p}: seed_base={base} median_ess_eff={medians[p]:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        results.setdefault("null_calibration", {})["populations"] = pops
        results["null_calibration"]["population_medians"] = medians
        results["null_calibration"]["k_null"] = k_null
        results["null_calibration"]["n_clean"] = n_clean
        results["null_calibration"]["n_budget"] = n_budget
        results["null_calibration"]["seed_base"] = SEED_BASE_NULL
        save(results, out_path)

    print("  computing all pairwise clean-vs-clean AUCs...", flush=True)
    pairs = {}
    for i, j in itertools.combinations(range(k_null), 2):
        seed = 1_090_000 + i * 100 + j  # cheap, no new flow draws; distinct seed per pair
        st = pair_stats(np.asarray(pops[i]), np.asarray(pops[j]), seed=seed)
        pairs[f"{i}v{j}"] = st
        print(f"    pair ({i},{j}): AUC={st['auc']:.4f} |AUC-0.5|={abs(st['auc']-0.5):.4f} "
              f"perm_p={st['perm_p_vs_0p5']:.4f}", flush=True)

    aucs = np.array([v["auc"] for v in pairs.values()])
    devs = np.abs(aucs - 0.5)
    results["null_calibration"]["pairs"] = pairs
    results["null_calibration"]["n_pairs"] = int(aucs.size)
    results["null_calibration"]["auc_summary"] = {
        "median": float(np.median(aucs)),
        "mean": float(np.mean(aucs)),
        "std": float(np.std(aucs, ddof=1)),
        "min": float(np.min(aucs)),
        "max": float(np.max(aucs)),
        "central95_lo": float(np.percentile(aucs, 2.5)),
        "central95_hi": float(np.percentile(aucs, 97.5)),
    }
    results["null_calibration"]["abs_dev_from_0p5_summary"] = {
        "median": float(np.median(devs)),
        "mean": float(np.mean(devs)),
        "std": float(np.std(devs, ddof=1)),
        "min": float(np.min(devs)),
        "max": float(np.max(devs)),
        "p95": float(np.percentile(devs, 95)),
        "frac_ge_0p05": float(np.mean(devs >= 0.05)),  # AUC>=0.55 or <=0.45
        "frac_ge_0p09": float(np.mean(devs >= 0.09)),  # AUC>=0.59 or <=0.41
    }
    save(results, out_path)
    print(f"  [null] median AUC={np.median(aucs):.4f} central95=[{np.percentile(aucs,2.5):.4f},"
          f"{np.percentile(aucs,97.5):.4f}] max|AUC-0.5|={np.max(devs):.4f}", flush=True)
    return results


def run_seedset_stage(lev, results, out_path, tag, base_clean, n_clean, n_mis, n_budget, strengths):
    print(f"  ---- {tag}: fresh clean (n={n_clean}) + fresh B4 draws, amps={strengths} ----",
          flush=True)
    eff_clean = draw_clean_population(lev, base_clean, n_clean, n_budget)
    print(f"    clean: median_ess_eff={np.nanmedian(eff_clean):.4f}", flush=True)
    pts = {}
    base_pts = base_clean + 2000  # disjoint block for the misspec/CI/perm seeds
    for si, strength in enumerate(strengths):
        xm, _ = R.gen_misspec(lev, "B4", strength, n_mis, seed=base_pts + si * 1000)
        mis = np.array([
            R.is_reweight(lev, xm[i], n_budget, seed=base_pts + 1000 + si * 1000 + i)["ess_frac"]
            for i in range(n_mis)
        ])
        _, _, auc = R.roc_auc(-eff_clean, -mis)
        lo, hi, _ = bootstrap_auc_ci(eff_clean, mis, n_boot=3000, seed=base_pts + 7000 + si)
        auc_chk, pval = permutation_p(eff_clean, mis, n_perm=3000, seed=base_pts + 8000 + si)
        assert abs(auc_chk - auc) < 1e-9
        pts[f"{strength:g}pct"] = {
            "gain_factor": 1.0 + strength / 100.0,
            "auc_ess_eff": float(auc), "ci95_lo": lo, "ci95_hi": hi,
            "perm_p_vs_0p5": pval,
            "ess_frac_median_mis": float(np.nanmedian(mis)),
            "ess_frac_median_clean": float(np.nanmedian(eff_clean)),
            "n_mis": n_mis, "n_clean": int(eff_clean.size),
        }
        p = pts[f"{strength:g}pct"]
        print(f"      strength={strength:g}%: AUC={p['auc_ess_eff']:.4f} "
              f"95%CI=[{p['ci95_lo']:.4f},{p['ci95_hi']:.4f}] perm_p={p['perm_p_vs_0p5']:.4f}",
              flush=True)
        results[tag] = {
            "points": pts,
            "source": f"fully fresh clean (n={n_clean}, drawn once, shared across amplitudes) "
                      f"+ fresh B4 misspec draw per amplitude, seed base {base_clean}",
            "clean_n": int(eff_clean.size),
            "clean_ess_frac_median": float(np.nanmedian(eff_clean)),
        }
        save(results, out_path)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["null", "set2", "set3"], default=None,
                     help="Run only this stage as its own process (crash isolation); "
                          "merges into the existing output JSON instead of overwriting "
                          "other stages. Default: run all three stages sequentially.")
    ap.add_argument("--k-null", type=int, default=K_NULL_DEFAULT)
    ap.add_argument("--n-clean", type=int, default=N_CLEAN_DEFAULT)
    ap.add_argument("--n-mis", type=int, default=N_MIS_DEFAULT)
    ap.add_argument("--n-budget", type=int, default=N_BUDGET_DEFAULT)
    ap.add_argument("--strengths", type=float, nargs="+", default=STRENGTHS)
    ap.add_argument("--out", type=str, default=str(OUT / "null_calibration_results.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    t_start = time.time()
    print(f"===== null calibration + seed sets 2/3 (level={LEVEL}) =====", flush=True)
    lev = R.Level(LEVEL)
    print(f"  flow loaded, ~{lev.median_counts:.0f} median counts", flush=True)

    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {"level": LEVEL, "n_clean": args.n_clean, "n_mis": args.n_mis,
                   "n_budget": args.n_budget, "strengths_pct": args.strengths,
                   "k_null": args.k_null}

    do_null = args.stage in (None, "null")
    do_set2 = args.stage in (None, "set2")
    do_set3 = args.stage in (None, "set3")

    if do_null:
        t0 = time.time()
        results = run_null_stage(lev, results, out_path, args.k_null, args.n_clean, args.n_budget)
        results["wall_s_null"] = time.time() - t0
        save(results, out_path)
        print(f"  [saved] null calibration done in {results['wall_s_null']:.1f}s", flush=True)

    if do_set2:
        t0 = time.time()
        results = run_seedset_stage(lev, results, out_path, "seed_set2", SEED_BASE_SET2,
                                     args.n_clean, args.n_mis, args.n_budget, args.strengths)
        results["wall_s_set2"] = time.time() - t0
        save(results, out_path)
        print(f"  [saved] seed_set2 done in {results['wall_s_set2']:.1f}s", flush=True)

    if do_set3:
        t0 = time.time()
        results = run_seedset_stage(lev, results, out_path, "seed_set3", SEED_BASE_SET3,
                                     args.n_clean, args.n_mis, args.n_budget, args.strengths)
        results["wall_s_set3"] = time.time() - t0
        save(results, out_path)
        print(f"  [saved] seed_set3 done in {results['wall_s_set3']:.1f}s", flush=True)

    results["wall_s_total"] = (results.get("wall_s_null", 0.0) + results.get("wall_s_set2", 0.0)
                                + results.get("wall_s_set3", 0.0))
    save(results, out_path)
    print(f"DONE (this process) total {time.time()-t_start:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
