"""Replication 2: stability check for run_conformal.py.

Identical protocol to ``run_conformal.py`` (same populations, same
checkpoint, same B4 3%-gain-shift setup, same conformal fit-on-clean /
apply-to-shifted deployment-realistic design) with a DIFFERENT stable integer
seed base, to check whether rep 1's numbers are stable to resampling noise
rather than a one-off artifact of a single seed draw. Writes to a separate
``outputs/conformal/rep2/`` subdirectory so rep 1's files are never touched.

Run (repo venv):
    .venv\\Scripts\\python.exe outputs\\conformal\\run_conformal_rep2.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbixcal import responses as R
from sbixcal import train_npe as tn
from sbixcal import calibrate as C
from sbixcal.misspec import simulate_misspec_population

ROOT = Path(__file__).resolve().parents[2]
FIXED_DIR = ROOT / "outputs" / "models" / "train_npe_prod_medium"
OUT = ROOT / "outputs" / "conformal" / "rep2"
OUT.mkdir(parents=True, exist_ok=True)

RESPONSE_NAME = "NGC7793_ULX4_PN"   # real XMM-Newton EPIC-pn response (Quintin+21)
EXPOSURE_S = 353.4
BASE_MODEL = "tbabs_powerlaw_bb"
PHYS_PRIORS = {
    "tbabs_1_nh":          {"dist": "uniform",    "low": 0.15,   "high": 0.35},
    "powerlaw_1_alpha":    {"dist": "uniform",    "low": 1.0,    "high": 3.0},
    "powerlaw_1_norm":     {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":   {"dist": "uniform",    "low": 0.3,    "high": 3.0},
    "blackbodyrad_1_norm": {"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}
PHYS_ORDER = ["tbabs_1_nh", "powerlaw_1_alpha", "powerlaw_1_norm",
              "blackbodyrad_1_kT", "blackbodyrad_1_norm"]
GAMMA_I = 1  # powerlaw_1_alpha index

N = 500              # population size per split (cal / shift-test / clean-test)
N_SAMPLES = 1000      # posterior draws per spectrum
# REPLICATION-2 seed base: distinct stable integer literal from rep 1's
# SEED=20260611. Not derived from hash(); it is the run date, 2026-07-23, so
# the seed base is a stable literal rather than a machine-dependent value.
SEED = 20260723

# disjoint, stable integer seed offsets; no builtin hash() (same offset
# scheme as rep 1, applied to the rep-2 SEED base)
SEED_CAL = SEED + 811          # clean g=1.00, fits the ConformalRecalibrator
SEED_SHIFT = SEED + 823        # B4 gain g=1.03, the misspecified deployment test set
SEED_CLEANCTL = SEED + 831     # clean g=1.00, DISJOINT from SEED_CAL, in-distribution control
SEED_SAMP_CAL = SEED + 901
SEED_SAMP_SHIFT = SEED + 902
SEED_SAMP_CLEANCTL = SEED + 903

NOMINAL_CURVE = np.round(np.linspace(0.05, 0.95, 19), 4)  # matches run_calibration.py's grid
NOMINAL_LEVELS = [0.68, 0.95]


def make_pop(strength: float, seed: int):
    """B4 gain-shift population at medium exposure/response.
    strength=0.0 -> gain 1.00 (clean); strength=3.0 -> gain 1.03 (the injection used throughout)."""
    base = R.load_base_obsconf(RESPONSE_NAME)
    oc = R.scale_exposure(base, EXPOSURE_S)
    x, theta, present = simulate_misspec_population(
        BASE_MODEL, PHYS_PRIORS, oc, "B4", float(strength), N, seed)
    assert present == PHYS_ORDER, present
    return np.asarray(x, dtype=np.float32), np.asarray(theta, dtype=np.float64)


def cov_at(nominal, cov, target):
    """Per-parameter coverage vector at the nominal level nearest `target`."""
    j = int(np.argmin(np.abs(np.asarray(nominal) - target)))
    return np.asarray(cov)[j], float(np.asarray(nominal)[j])


def mean_std_per_spectrum(samples_per_obs):
    """Per-spectrum posterior mean and std, stacked -> (N, n_params) each."""
    means = np.stack([s.mean(axis=0) for s in samples_per_obs], axis=0)
    stds = np.stack([s.std(axis=0) for s in samples_per_obs], axis=0)
    return means, stds


def median_per_spectrum(samples_per_obs):
    """Per-spectrum posterior median, stacked -> (N, n_params)."""
    return np.stack([np.median(s, axis=0) for s in samples_per_obs], axis=0)


def bias_report(means, stds, truths, idx, medians=None):
    b = means[:, idx] - truths[:, idx]
    n = len(b)
    mean_b = float(b.mean())
    se_b = float(b.std(ddof=1) / np.sqrt(n))
    b_in_std = b / stds[:, idx]
    mean_b_std = float(b_in_std.mean())
    se_b_std = float(b_in_std.std(ddof=1) / np.sqrt(n))
    out = {
        "n": int(n),
        "bias_mean_abs": mean_b, "bias_se_abs": se_b,
        "bias_mean_in_post_std": mean_b_std, "bias_se_in_post_std": se_b_std,
        "median_posterior_std": float(np.median(stds[:, idx])),
    }
    if medians is not None:
        bm = medians[:, idx] - truths[:, idx]
        out["biasmedian_mean_abs"] = float(bm.mean())
        out["biasmedian_se_abs"] = float(bm.std(ddof=1) / np.sqrt(n))
        out["biasmedian_median_abs"] = float(np.median(bm))
    return out


def main():
    t_start = time.time()
    print("[conformal-rep2] loading fixed-response baseline flow:", FIXED_DIR)
    post, info = tn.load_posterior(FIXED_DIR, device="cpu")
    assert info["param_names"] == PHYS_ORDER, info["param_names"]
    print(f"[conformal-rep2] checkpoint median_total_counts={info['median_total_counts']:.0f} "
          f"exposure_s={info['exposure_s']}")

    # ---------------------------------------------------------------
    # populations
    # ---------------------------------------------------------------
    t0 = time.time()
    x_cal, th_cal = make_pop(0.0, SEED_CAL)
    x_shift, th_shift = make_pop(3.0, SEED_SHIFT)
    x_cleanctl, th_cleanctl = make_pop(0.0, SEED_CLEANCTL)
    print(f"[conformal-rep2] populations generated in {time.time()-t0:.1f}s: "
          f"cal(clean) median counts={np.median(x_cal.sum(1)):.0f}, "
          f"shift(g=1.03) median counts={np.median(x_shift.sum(1)):.0f}, "
          f"cleanctl median counts={np.median(x_cleanctl.sum(1)):.0f}")

    # ---------------------------------------------------------------
    # posterior sampling (reject_outside_prior=True, the run_calibration.py protocol)
    # ---------------------------------------------------------------
    t0 = time.time()
    s_cal = C.sample_posterior_batch(post, x_cal, N_SAMPLES, seed=SEED_SAMP_CAL, device="cpu")
    print(f"[conformal-rep2] s_cal sampled in {time.time()-t0:.1f}s")
    t0 = time.time()
    s_shift = C.sample_posterior_batch(post, x_shift, N_SAMPLES, seed=SEED_SAMP_SHIFT, device="cpu")
    print(f"[conformal-rep2] s_shift sampled in {time.time()-t0:.1f}s")
    t0 = time.time()
    s_cleanctl = C.sample_posterior_batch(post, x_cleanctl, N_SAMPLES, seed=SEED_SAMP_CLEANCTL, device="cpu")
    print(f"[conformal-rep2] s_cleanctl sampled in {time.time()-t0:.1f}s")

    # ---------------------------------------------------------------
    # fit ConformalRecalibrator on CLEAN calibration data ONLY
    # ---------------------------------------------------------------
    recal = C.ConformalRecalibrator.fit(s_cal, th_cal, PHYS_ORDER)

    # ---------------------------------------------------------------
    # coverage: raw vs conformal-recalibrated, on shifted test + clean control
    # ---------------------------------------------------------------
    cov_raw_shift = C.empirical_coverage_curve(s_shift, th_shift, NOMINAL_CURVE)
    cov_recal_shift = recal.coverage_curve(s_shift, th_shift, NOMINAL_CURVE)
    cov_raw_clean = C.empirical_coverage_curve(s_cleanctl, th_cleanctl, NOMINAL_CURVE)
    cov_recal_clean = recal.coverage_curve(s_cleanctl, th_cleanctl, NOMINAL_CURVE)

    # sanity: the calibration set's OWN before/after
    cov_raw_cal = C.empirical_coverage_curve(s_cal, th_cal, NOMINAL_CURVE)
    cov_recal_cal = recal.coverage_curve(s_cal, th_cal, NOMINAL_CURVE)

    # ---------------------------------------------------------------
    # bias: posterior MEAN minus truth
    # ---------------------------------------------------------------
    means_shift, stds_shift = mean_std_per_spectrum(s_shift)
    means_clean, stds_clean = mean_std_per_spectrum(s_cleanctl)

    bias_shift = {PHYS_ORDER[p]: bias_report(means_shift, stds_shift, th_shift, p)
                  for p in range(len(PHYS_ORDER))}
    bias_clean = {PHYS_ORDER[p]: bias_report(means_clean, stds_clean, th_cleanctl, p)
                  for p in range(len(PHYS_ORDER))}

    has_point_estimate_method = any(
        hasattr(recal, m) for m in ("mean", "point_estimate", "median", "predict"))
    assert not has_point_estimate_method, (
        "ConformalRecalibrator unexpectedly exposes a point-estimate method; "
        "the before==after bias identity assumed below would not hold.")

    # ---------------------------------------------------------------
    # coverage table
    # ---------------------------------------------------------------
    cov_at_nominal = {}
    for lvl in NOMINAL_LEVELS:
        raw_s, nom_used = cov_at(NOMINAL_CURVE, cov_raw_shift, lvl)
        rec_s, _ = cov_at(NOMINAL_CURVE, cov_recal_shift, lvl)
        raw_c, _ = cov_at(NOMINAL_CURVE, cov_raw_clean, lvl)
        rec_c, _ = cov_at(NOMINAL_CURVE, cov_recal_clean, lvl)
        cov_at_nominal[str(lvl)] = {
            "nominal_used": nom_used,
            "shifted": {"raw": {PHYS_ORDER[i]: float(raw_s[i]) for i in range(5)},
                        "conformal": {PHYS_ORDER[i]: float(rec_s[i]) for i in range(5)}},
            "clean_control": {"raw": {PHYS_ORDER[i]: float(raw_c[i]) for i in range(5)},
                              "conformal": {PHYS_ORDER[i]: float(rec_c[i]) for i in range(5)}},
        }

    def mad(cov):
        return float(np.mean(np.abs(cov - NOMINAL_CURVE[:, None])))

    dev_summary = {
        "shifted": {"raw": mad(cov_raw_shift), "conformal": mad(cov_recal_shift)},
        "clean_control": {"raw": mad(cov_raw_clean), "conformal": mad(cov_recal_clean)},
        "calibration_set_in_sample": {"raw": mad(cov_raw_cal), "conformal": mad(cov_recal_cal)},
    }

    print("\n[conformal-rep2] mean |emp cov - nominal| over the 5%-95% curve, per split:")
    for k, v in dev_summary.items():
        print(f"  {k:28s} raw={v['raw']:.4f} -> conformal={v['conformal']:.4f}")

    print("\n[conformal-rep2] Gamma bias (posterior mean - truth), shifted vs clean control:")
    for k, d in (("shifted", bias_shift["powerlaw_1_alpha"]),
                 ("clean_control", bias_clean["powerlaw_1_alpha"])):
        print(f"  {k:14s} bias={d['bias_mean_abs']:+.4f} +/- {d['bias_se_abs']:.4f}  "
              f"({d['bias_mean_in_post_std']:+.3f} +/- {d['bias_se_in_post_std']:.3f} "
              f"post-std units, N={d['n']})")

    # ---------------------------------------------------------------
    # save numbers
    # ---------------------------------------------------------------
    results = {
        "config": {
            "checkpoint": "outputs/models/train_npe_prod_medium",
            "response": RESPONSE_NAME, "exposure_s": EXPOSURE_S, "base_model": BASE_MODEL,
            "family": "B4", "gain_shift_pct": 3.0, "level": "medium",
            "n_cal": N, "n_shift_test": N, "n_cleanctl_test": N, "n_samples": N_SAMPLES,
            "seed_base": SEED, "replication": 2,
            "seed_cal": SEED_CAL, "seed_shift": SEED_SHIFT, "seed_cleanctl": SEED_CLEANCTL,
            "seed_samp_cal": SEED_SAMP_CAL, "seed_samp_shift": SEED_SAMP_SHIFT,
            "seed_samp_cleanctl": SEED_SAMP_CLEANCTL,
            "median_counts_cal": float(np.median(x_cal.sum(1))),
            "median_counts_shift": float(np.median(x_shift.sum(1))),
            "median_counts_cleanctl": float(np.median(x_cleanctl.sum(1))),
            "reject_outside_prior": True,
            "nominal_curve": NOMINAL_CURVE.tolist(),
        },
        "coverage_deviation_summary": dev_summary,
        "coverage_at_nominal": cov_at_nominal,
        "gamma_bias": {"shifted": bias_shift["powerlaw_1_alpha"],
                      "clean_control": bias_clean["powerlaw_1_alpha"]},
        "all_param_bias_shifted": bias_shift,
        "all_param_bias_clean_control": bias_clean,
    }
    with open(OUT / "conformal_results_rep2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[conformal-rep2] -> {OUT / 'conformal_results_rep2.json'}")

    np.savez(OUT / "conformal_coverage_curves_rep2.npz",
             nominal=NOMINAL_CURVE, param_names=np.array(PHYS_ORDER),
             cov_raw_shift=cov_raw_shift, cov_recal_shift=cov_recal_shift,
             cov_raw_clean=cov_raw_clean, cov_recal_clean=cov_recal_clean,
             cov_raw_cal=cov_raw_cal, cov_recal_cal=cov_recal_cal,
             means_shift=means_shift, stds_shift=stds_shift, th_shift=th_shift,
             means_clean=means_clean, stds_clean=stds_clean, th_cleanctl=th_cleanctl)
    print(f"[conformal-rep2] -> {OUT / 'conformal_coverage_curves_rep2.npz'}")

    print(f"\n[conformal-rep2] total wall time {time.time()-t_start:.1f}s")
    return results


if __name__ == "__main__":
    main()
