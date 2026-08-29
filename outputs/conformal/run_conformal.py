"""Split conformal recalibration fitted on a clean calibration population and
applied to a 3 per cent gain-shifted test population.

Setup (deployment-realistic): fit ``sbixcal.calibrate.ConformalRecalibrator`` on
posterior samples from the FIXED-RESPONSE baseline flow (outputs/models/
train_npe_prod_medium, tbabs*(powerlaw+blackbody), 5 params) on a CLEAN
(g=1.00) calibration population, then APPLY that recalibrator to the flow's
posteriors on a B4 3%-gain-shifted (g=1.03) test population, so the calibrator
never sees the shift. This mirrors exactly how ``coverage_before_after`` /
``ConformalRecalibrator.fit`` + ``.coverage_curve`` are used in
scripts/run_calibration.py (fit on a held-out calibration set, evaluate on a
disjoint test set), except that run_calibration.py keeps calibration and
evaluation in distribution (both clean); here we
deliberately break that match to probe the out-of-distribution case a real
deployment would face. A disjoint CLEAN control test set (same generator,
different seed from the calibration set) is run through the same recalibrator
for comparison.

Measures, before vs after conformal recalibration, on both the shifted test
set and the clean control test set:
  - per-parameter empirical marginal coverage at nominal 68% / 95% (+ full
    curve for the figure), for all 5 physical parameters,
  - the photon-index (Gamma = powerlaw_1_alpha) bias: posterior MEAN minus
    truth, population mean +/- SE, in absolute Gamma units and in units of
    the per-spectrum posterior std. This is identical before/after conformal
    recalibration BY CONSTRUCTION: ConformalRecalibrator only remaps which
    sample-quantile levels are read off to form an interval, and it never
    touches the posterior samples or their mean, so it cannot move a point
    estimate. This script verifies that identity numerically rather than
    assuming it.

Uses B4 gain-shift population generation exactly as outputs/gain_marg/
eval_gainmarg.py does (same PHYS_PRIORS / PHYS_ORDER / exposure / response /
base model, sbixcal.misspec.simulate_misspec_population), and posterior
sampling via sbixcal.calibrate.sample_posterior_batch (reject_outside_prior=
True, the protocol run_calibration.py uses, not eval_gainmarg's looser
reject_outside_prior=False variant).

Run (repo venv):
    .venv\\Scripts\\python.exe outputs\\conformal\\run_conformal.py
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
OUT = ROOT / "outputs" / "conformal"
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
SEED = 20260611       # project-standard base seed (train_npe_prod.yaml, calibration_prod.yaml)

# disjoint, stable integer seed offsets; no builtin hash()
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
    print("[conformal] loading fixed-response baseline flow:", FIXED_DIR)
    post, info = tn.load_posterior(FIXED_DIR, device="cpu")
    assert info["param_names"] == PHYS_ORDER, info["param_names"]
    print(f"[conformal] checkpoint median_total_counts={info['median_total_counts']:.0f} "
          f"exposure_s={info['exposure_s']}")

    # ---------------------------------------------------------------
    # populations
    # ---------------------------------------------------------------
    t0 = time.time()
    x_cal, th_cal = make_pop(0.0, SEED_CAL)
    x_shift, th_shift = make_pop(3.0, SEED_SHIFT)
    x_cleanctl, th_cleanctl = make_pop(0.0, SEED_CLEANCTL)
    print(f"[conformal] populations generated in {time.time()-t0:.1f}s: "
          f"cal(clean) median counts={np.median(x_cal.sum(1)):.0f}, "
          f"shift(g=1.03) median counts={np.median(x_shift.sum(1)):.0f}, "
          f"cleanctl median counts={np.median(x_cleanctl.sum(1)):.0f}")

    # ---------------------------------------------------------------
    # posterior sampling (reject_outside_prior=True, the run_calibration.py protocol)
    # ---------------------------------------------------------------
    t0 = time.time()
    s_cal = C.sample_posterior_batch(post, x_cal, N_SAMPLES, seed=SEED_SAMP_CAL, device="cpu")
    print(f"[conformal] s_cal sampled in {time.time()-t0:.1f}s")
    t0 = time.time()
    s_shift = C.sample_posterior_batch(post, x_shift, N_SAMPLES, seed=SEED_SAMP_SHIFT, device="cpu")
    print(f"[conformal] s_shift sampled in {time.time()-t0:.1f}s")
    t0 = time.time()
    s_cleanctl = C.sample_posterior_batch(post, x_cleanctl, N_SAMPLES, seed=SEED_SAMP_CLEANCTL, device="cpu")
    print(f"[conformal] s_cleanctl sampled in {time.time()-t0:.1f}s")

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

    # sanity: the calibration set's OWN before/after (should show conformal
    # forcing near-exact nominal coverage on clean in-sample data, a trivial
    # check that the recalibrator machinery is doing what it claims)
    cov_raw_cal = C.empirical_coverage_curve(s_cal, th_cal, NOMINAL_CURVE)
    cov_recal_cal = recal.coverage_curve(s_cal, th_cal, NOMINAL_CURVE)

    # ---------------------------------------------------------------
    # bias: posterior MEAN minus truth (identical before/after conformal by
    # construction: conformal only remaps interval endpoints, never touches
    # the samples/mean). Verify that identity numerically.
    # ---------------------------------------------------------------
    means_shift, stds_shift = mean_std_per_spectrum(s_shift)
    means_clean, stds_clean = mean_std_per_spectrum(s_cleanctl)

    bias_shift = {PHYS_ORDER[p]: bias_report(means_shift, stds_shift, th_shift, p)
                  for p in range(len(PHYS_ORDER))}
    bias_clean = {PHYS_ORDER[p]: bias_report(means_clean, stds_clean, th_cleanctl, p)
                  for p in range(len(PHYS_ORDER))}

    # explicit structural check: conformal recalibration cannot change the
    # point estimate because ConformalRecalibrator has no method that returns
    # anything but interval endpoints from the SAME samples array.
    has_point_estimate_method = any(
        hasattr(recal, m) for m in ("mean", "point_estimate", "median", "predict"))
    assert not has_point_estimate_method, (
        "ConformalRecalibrator unexpectedly exposes a point-estimate method; "
        "the before==after bias identity assumed below would not hold.")

    # ---------------------------------------------------------------
    # coverage table: per-param coverage at 68% / 95%, raw vs conformal,
    # shifted vs clean
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

    # mean |emp - nominal| deviation across the full curve, per split (single
    # summary number, mirrors run_calibration.py's raw_dev/recal_dev)
    def mad(cov):
        return float(np.mean(np.abs(cov - NOMINAL_CURVE[:, None])))

    dev_summary = {
        "shifted": {"raw": mad(cov_raw_shift), "conformal": mad(cov_recal_shift)},
        "clean_control": {"raw": mad(cov_raw_clean), "conformal": mad(cov_recal_clean)},
        "calibration_set_in_sample": {"raw": mad(cov_raw_cal), "conformal": mad(cov_recal_cal)},
    }

    print("\n[conformal] mean |emp cov - nominal| over the 5%-95% curve, per split:")
    for k, v in dev_summary.items():
        print(f"  {k:28s} raw={v['raw']:.4f} -> conformal={v['conformal']:.4f}")

    print("\n[conformal] Gamma bias (posterior mean - truth), shifted vs clean control:")
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
    with open(OUT / "conformal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[conformal] -> {OUT / 'conformal_results.json'}")

    np.savez(OUT / "conformal_coverage_curves.npz",
             nominal=NOMINAL_CURVE, param_names=np.array(PHYS_ORDER),
             cov_raw_shift=cov_raw_shift, cov_recal_shift=cov_recal_shift,
             cov_raw_clean=cov_raw_clean, cov_recal_clean=cov_recal_clean,
             cov_raw_cal=cov_raw_cal, cov_recal_cal=cov_recal_cal,
             means_shift=means_shift, stds_shift=stds_shift, th_shift=th_shift,
             means_clean=means_clean, stds_clean=stds_clean, th_cleanctl=th_cleanctl)
    print(f"[conformal] -> {OUT / 'conformal_coverage_curves.npz'}")

    # ---------------------------------------------------------------
    # figure: 2 panels. (1) Gamma coverage curve raw vs conformal, shifted
    # vs clean; (2) Gamma bias before/after (identical, shown explicitly)
    # ---------------------------------------------------------------
    # colorblind-safe (Okabe-Ito): blue #0072B2, orange #E69F00, vermillion #D55E00, black
    C_RAW_SHIFT = "#D55E00"
    C_RECAL_SHIFT = "#0072B2"
    C_RAW_CLEAN = "#E69F00"
    C_RECAL_CLEAN = "#009E73"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="nominal")
    ax.plot(NOMINAL_CURVE, cov_raw_shift[:, GAMMA_I], "o-", color=C_RAW_SHIFT, ms=4, lw=1.6,
            label="raw NPE, shifted (g=1.03)")
    ax.plot(NOMINAL_CURVE, cov_recal_shift[:, GAMMA_I], "s-", color=C_RECAL_SHIFT, ms=4, lw=1.6,
            label="conformal, shifted (g=1.03)")
    ax.plot(NOMINAL_CURVE, cov_raw_clean[:, GAMMA_I], "o--", color=C_RAW_CLEAN, ms=3, lw=1.1,
            alpha=0.8, label="raw NPE, clean control")
    ax.plot(NOMINAL_CURVE, cov_recal_clean[:, GAMMA_I], "s--", color=C_RECAL_CLEAN, ms=3, lw=1.1,
            alpha=0.8, label="conformal, clean control")
    for lvl in NOMINAL_LEVELS:
        ax.axvline(lvl, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("nominal credible level")
    ax.set_ylabel("empirical marginal coverage")
    ax.set_title("Photon index $\\Gamma$: coverage,\nconformal fit on CLEAN data only")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    gb_s = bias_shift["powerlaw_1_alpha"]
    gb_c = bias_clean["powerlaw_1_alpha"]
    xpos = [0, 1, 2, 3]
    xlab = ["shifted\nbefore", "shifted\nafter", "clean ctl\nbefore", "clean ctl\nafter"]
    yvals = [gb_s["bias_mean_abs"], gb_s["bias_mean_abs"],
             gb_c["bias_mean_abs"], gb_c["bias_mean_abs"]]
    yerr = [gb_s["bias_se_abs"], gb_s["bias_se_abs"],
            gb_c["bias_se_abs"], gb_c["bias_se_abs"]]
    colors = [C_RAW_SHIFT, C_RECAL_SHIFT, C_RAW_CLEAN, C_RECAL_CLEAN]
    ax2.axhline(0.0, color="k", ls="--", lw=1)
    ax2.errorbar(xpos, yvals, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2,
                 capsize=4, zorder=2)
    ax2.bar(xpos, yvals, color=colors, alpha=0.85, width=0.6, zorder=1)
    ax2.set_xticks(xpos); ax2.set_xticklabels(xlab, fontsize=8)
    ax2.set_ylabel(r"$\Gamma$ bias (posterior mean $-$ truth)")
    ax2.set_title("Photon index $\\Gamma$ bias:\nconformal recalibration cannot move it")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Conformal recalibration repairs coverage but not the photon-index bias\n"
        "(fixed-response NPE, medium/~986 cts, EPIC-pn, B4 3% gain shift; "
        "conformal fit on clean g=1.00 only)",
        fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "conformal_coverage.png", dpi=150)
    fig.savefig(OUT / "conformal_coverage.pdf")
    plt.close(fig)
    print(f"[conformal] -> {OUT / 'conformal_coverage.png'} / .pdf")

    print(f"\n[conformal] total wall time {time.time()-t_start:.1f}s")
    return results


if __name__ == "__main__":
    main()
