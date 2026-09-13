r"""E6b: residual Fisher power R = (1-I) * ||b||^2, extending the E6
invisibility-index mechanism computation.

Motivation: I = ||P_A b||^2 / ||b||^2 is a *fraction* --
level-invariant by construction (a uniform rescaling of b leaves the ratio of
squared norms unchanged). For B2/B3, I ~ 0.99-0.9999 makes them look "more
invisible than B4" (I ~ 0.976) by the fraction alone, but B2/B3 are in fact
CLEARLY detected by the project's committed AUC benchmark
(outputs/detect/auc_table.md) at these same strengths, while B4 is not. The
missing ingredient is scale: a whitened perturbation vector b can have a tiny
*fraction* left outside the model's span while that fraction is still large
in *absolute* (chi^2-scale) terms if ||b||^2 itself is large. This script
computes the out-of-span absolute residual power

    R = (1 - I) * ||b||^2

which is an expected chi^2-excess-like scale in Fisher-whitened units (NOT
bounded in [0,1], and expected to grow ~linearly with exposure, unlike I).

Reuses outputs/mechanism/run_mechanism_index.py's machinery by IMPORT (same
theta draws, same seed, same Jacobian/whitening/projection code) -- nothing
in that module is edited. This both (a) avoids duplicating validated code and
(b) gives a built-in reproduction check: re-deriving I via the imported
functions and comparing against the already-committed
mechanism_index_results.json is sanity check #1 below.

No training, no MCMC/NS sampling, no Poisson sampling anywhere (same as the
parent script -- apply_stat=False throughout, inherited from the imported
functions).

Reproduce (repo root, repo venv):
    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe outputs\mechanism\run_residual_power.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `import run_mechanism_index` finds the sibling module

import run_mechanism_index as mech  # noqa: E402  (reused machinery, not edited)

ROOT = mech.ROOT
OUT = mech.OUT
EXISTING_RESULTS = OUT / "mechanism_index_results.json"


def summarize_values(values, flagged):
    """Same convention as run_mechanism_index.summarize: median/q25/q75 over
    unflagged, finite values. Works for I, bnorm2, or R alike."""
    arr = np.asarray(values, dtype=np.float64)
    flagged = np.asarray(flagged, dtype=bool)
    good = arr[~flagged]
    good = good[np.isfinite(good)]
    if good.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0, "n_flagged": int(flagged.sum())}
    return {
        "median": float(np.median(good)),
        "q25": float(np.percentile(good, 25)),
        "q75": float(np.percentile(good, 75)),
        "n": int(good.size),
        "n_flagged": int(flagged.sum()),
    }


def main():
    t0 = time.time()
    theta, param_order, lows, highs, prior_cfg = mech.build_theta()
    print(f"theta drawn: {theta.shape}, seed={mech.GLOBAL_SEED}, params={param_order}")

    results = {
        "meta": {
            "source_module": "run_mechanism_index.py (imported, not modified)",
            "response": mech.RESPONSE,
            "base_model": mech.BASE_MODEL,
            "param_order": param_order,
            "seed": mech.GLOBAL_SEED,
            "n_draws": mech.N_DRAWS,
            "rel_step": mech.REL_STEP,
            "levels_exposure_s": mech.LEVELS,
            "family_strength": mech.FAMILY_STRENGTH,
            "bnorm2_guard": mech.BNORM2_GUARD,
            "definition": "R = (1 - I) * ||b||^2  (Fisher-whitened out-of-span "
                           "residual power; I = ||P_A b||^2 / ||b||^2 as in "
                           "run_mechanism_index.py)",
        },
        "per_theta": {},
        "summary": {},
    }

    for level in mech.LEVELS:
        print(f"\n=== level={level} (exposure {mech.LEVELS[level]} s) ===")
        obsconf = mech.obsconf_for_level(level)
        J, mu0, clip_frac = mech.central_diff_jacobian(theta, param_order, lows, highs, obsconf)
        assert np.all(mu0 > 0), "mu0 has non-positive channels -- whitening undefined"

        results["per_theta"][level] = {}
        results["summary"][level] = {}

        for fam in mech.FAMILIES:
            mu_mis = mech.DELTA_FN[fam](theta, param_order, obsconf)
            res = mech.invisibility_index(J, mu0, mu_mis)
            I = res["I"]
            bnorm2 = res["bnorm2"]
            flagged = res["flagged"]
            R = np.array(
                [(1.0 - Iv) * bn if (not fl and np.isfinite(Iv)) else np.nan
                 for Iv, bn, fl in zip(I, bnorm2, flagged)],
                dtype=np.float64,
            )

            results["per_theta"][level][fam] = {
                "I": [None if not np.isfinite(v) else float(v) for v in I],
                "bnorm2": [float(v) for v in bnorm2],
                "R": [None if not np.isfinite(v) else float(v) for v in R],
                "flagged": [bool(v) for v in flagged],
            }
            summ_I = summarize_values(I, flagged)
            summ_b = summarize_values(bnorm2, flagged)
            summ_R = summarize_values(R, flagged)
            results["summary"][level][fam] = {"I": summ_I, "bnorm2": summ_b, "R": summ_R}
            print(f"  {fam}: median I={summ_I['median']:.4f}  "
                  f"median ||b||^2={summ_b['median']:.4g}  "
                  f"median R={summ_R['median']:.4g}")

    # -----------------------------------------------------------------
    # sanity check 1: I from this run reproduces the committed
    # mechanism_index_results.json (same code path via import, same seed --
    # should match to floating-point precision; this validates the reuse is
    # wired correctly, not an independent re-derivation of the algorithm).
    # -----------------------------------------------------------------
    sanity = {"reproduces_I": {}, "exposure_scaling": {}}
    with open(EXISTING_RESULTS) as f:
        existing = json.load(f)
    max_abs_diff_overall = 0.0
    for level in mech.LEVELS:
        sanity["reproduces_I"][level] = {}
        for fam in mech.FAMILIES:
            new_I = np.array(
                [np.nan if v is None else v for v in results["per_theta"][level][fam]["I"]]
            )
            old_I = np.array(
                [np.nan if v is None else v for v in existing["per_theta"][level][fam]["I"]]
            )
            diff = np.abs(new_I - old_I)
            diff = diff[np.isfinite(diff)]
            max_diff = float(diff.max()) if diff.size else 0.0
            max_abs_diff_overall = max(max_abs_diff_overall, max_diff)
            sanity["reproduces_I"][level][fam] = {
                "max_abs_diff": max_diff,
                "matches": bool(max_diff < 1e-9),
            }

    # -----------------------------------------------------------------
    # sanity check 2: R should scale ~linearly with exposure between medium
    # and bright (nominal exposure ratio = 3534.0 / 353.4 = 10.0 exactly).
    # -----------------------------------------------------------------
    exposure_ratio = mech.LEVELS["bright"] / mech.LEVELS["medium"]
    sanity["exposure_scaling"]["nominal_exposure_ratio"] = exposure_ratio
    sanity["exposure_scaling"]["per_family"] = {}
    for fam in mech.FAMILIES:
        med_R = results["summary"]["medium"][fam]["R"]["median"]
        bri_R = results["summary"]["bright"][fam]["R"]["median"]
        med_b = results["summary"]["medium"][fam]["bnorm2"]["median"]
        bri_b = results["summary"]["bright"][fam]["bnorm2"]["median"]
        sanity["exposure_scaling"]["per_family"][fam] = {
            "median_R_medium": med_R,
            "median_R_bright": bri_R,
            "measured_R_ratio": (bri_R / med_R) if med_R else None,
            "median_bnorm2_medium": med_b,
            "median_bnorm2_bright": bri_b,
            "measured_bnorm2_ratio": (bri_b / med_b) if med_b else None,
        }

    results["sanity_checks"] = sanity

    print("\n--- sanity check 1: I reproduces committed mechanism_index_results.json ---")
    print(f"  max |delta I| over all cells = {max_abs_diff_overall:.3e}  "
          f"({'PASS' if max_abs_diff_overall < 1e-9 else 'FAIL'}, threshold 1e-9)")
    print("\n--- sanity check 2: R exposure scaling (nominal ratio = "
          f"{exposure_ratio:.4f}) ---")
    for fam in mech.FAMILIES:
        d = sanity["exposure_scaling"]["per_family"][fam]
        print(f"  {fam}: measured R ratio (bright/medium) = {d['measured_R_ratio']:.4f}  "
              f"(||b||^2 ratio = {d['measured_bnorm2_ratio']:.4f})")

    out_json = OUT / "residual_power_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out_json}  ({out_json.stat().st_size/1024:.0f} KiB)")
    print(f"total wall time {time.time()-t0:.1f} s")
    return results


if __name__ == "__main__":
    main()
