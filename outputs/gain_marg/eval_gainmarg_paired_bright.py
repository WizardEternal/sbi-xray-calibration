"""Paired gain-shift evaluation at bright counts (~10^4), the counterpart to
``eval_gainmarg_paired.py`` at medium counts.

This is a copy of ``eval_gainmarg_paired.py`` with ONLY the following
changed, per the config block that script deliberately parameterized for
this exact purpose:
  - EXPOSURE_S: 353.4 -> 3534.0
  - FIXED_DIR: train_npe_prod_medium -> train_npe_prod_bright
  - GM_DIR: model_medium -> model_bright
  - output paths: paired_gain_bias_medium.json -> paired_gain_bias_bright.json,
    paired_gain_bias_medium.png -> paired_gain_bias_bright.png
  - figure suptitle: now reports the actual computed median counts instead
    of the medium script's hardcoded "~986 counts" (a data-correctness fix,
    not a design change; the label must match the data it describes)

Seed convention: UNCHANGED from the medium script. The medium script does
NOT encode "medium" into any of its seed constants (SEED / SEED_THETA /
SEED_POISSON_BASE / SEED_SAMPLE are level-agnostic fixed offsets from a
common base); only the upstream *training-data-generation* scripts
(``gen_and_train_gainmarg{,_bright}.py``) tag seeds by level, via
SEED_OFFSET = 2 (medium) / 3 (bright) on the flow-training simulation.
This eval script's job is different: it draws its OWN N_TEST-theta paired
population at eval time, independent of whichever training set the
checkpoints were fit on, so there is no level-derived salt to carry
forward here. Keeping every SEED_* constant identical to the medium script
is therefore the correct application of "keep its convention": it means
the bright run draws the exact same theta population (same physical
prior, same seed) as the medium run, so bright-vs-medium comparisons are
not confounded by different underlying source parameters, only by the
count level. Only the OUTPUT paths and the two checkpoint dirs differ.

For the rationale behind the paired design, the common random numbers and the
rejection/clip step, see the docstring of ``eval_gainmarg_paired.py``.

Run (repo venv):
    .venv\\Scripts\\python.exe outputs\\gain_marg\\eval_gainmarg_paired_bright.py
    .venv\\Scripts\\python.exe outputs\\gain_marg\\eval_gainmarg_paired_bright.py --n-test 8 --max-sampling-time 3
        (quick wiring check; does not touch the committed N_TEST=500 config)

Never modifies eval_gainmarg_paired.py, eval_gainmarg.py, or their outputs.
Writes only:
    outputs/gain_marg/paired_gain_bias_bright.json
    outputs/gain_marg/paired_gain_bias_bright.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbixcal import responses as R
from sbixcal import models as M
from sbixcal import priors as P
from sbixcal import simulate as SIM
from sbixcal import train_npe as tn

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------
# Config: bright variant of eval_gainmarg_paired.py's config block.
# ----------------------------------------------------------------------------
FIXED_DIR = ROOT / "outputs" / "models" / "train_npe_prod_bright"
GM_DIR = ROOT / "outputs" / "gain_marg" / "model_bright"
OUT = ROOT / "outputs" / "gain_marg"

EXPOSURE_S = 3534.0
BASE_MODEL = "tbabs_powerlaw_bb"
RESPONSE_NAME = "NGC7793_ULX4_PN"

PHYS_PRIORS = {
    "tbabs_1_nh":          {"dist": "uniform",    "low": 0.15,   "high": 0.35},
    "powerlaw_1_alpha":    {"dist": "uniform",    "low": 1.0,    "high": 3.0},
    "powerlaw_1_norm":     {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":   {"dist": "uniform",    "low": 0.3,    "high": 3.0},
    "blackbodyrad_1_norm": {"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}
GAIN_LO, GAIN_HI = 0.95, 1.05
GAIN_PRIOR = {"gain_g": {"dist": "uniform", "low": GAIN_LO, "high": GAIN_HI}}
GAIN_PRIOR_WIDTH = GAIN_HI - GAIN_LO

PHYS_ORDER = M.MODEL_PARAMS[BASE_MODEL]
assert PHYS_ORDER == ["tbabs_1_nh", "powerlaw_1_alpha", "powerlaw_1_norm",
                      "blackbodyrad_1_kT", "blackbodyrad_1_norm"], PHYS_ORDER
GM_ORDER = PHYS_ORDER + ["gain_g"]
GAMMA_I = PHYS_ORDER.index("powerlaw_1_alpha")
NORM_I = PHYS_ORDER.index("powerlaw_1_norm")

N_TEST = 500
N_SAMPLES = 1000
CRED = 0.90
MAX_SAMPLING_TIME = 8.0  # seconds, per-spectrum rejection-sampling cap (detect.py pattern)

# stable integer seeds (never builtin hash()); UNCHANGED from the medium
# script; see the module docstring for why this is the correct convention
SEED = 20260611
SEED_THETA = SEED + 60000            # single draw of the N_TEST paired thetas
SEED_POISSON_BASE = SEED + 70000     # per-spectrum CRN base; seed_i = BASE + i, reused for both g
SEED_SAMPLE = {
    ("fixed", "clean"):    SEED + 100100,
    ("fixed", "gain"):     SEED + 100200,
    ("gainmarg", "clean"): SEED + 100300,
    ("gainmarg", "gain"):  SEED + 100400,
}
GAIN_CASES = [("clean", 1.00), ("gain", 1.03)]

Z95 = 1.959963985  # standard-normal 97.5th percentile, for the 95% CI


# ----------------------------------------------------------------------------
# paired population: one theta draw, folded at g=1.00 and g=1.03, CRN Poisson
# ----------------------------------------------------------------------------
def make_paired_population(n_test, theta_seed, poisson_seed_base):
    """Draw ``n_test`` thetas ONCE from the physical prior (mirrors
    ``misspec._params_b4`` -> ``_base_nuisance`` -> ``priors.sample_prior``,
    since B4 adds no source parameters of its own), then fold each theta
    through the g=1.00 and g=1.03 gain-shifted EPIC-pn responses (the exact
    ``misspec.simulate_misspec_population`` B4 path:
    ``responses.gain_shift_obsconf(obsconf, 1.0 + strength/100)`` then
    ``fakeit``). We use ``simulate.fold_theta`` for the noiseless expected
    counts (lambda) and apply the Poisson draw ourselves with a per-spectrum
    seed reused identically across both g values (common random numbers), so
    the two populations differ ONLY in the response.

    Returns (theta (n,5) float64, x_clean (n,102) float32, x_gain (n,102)
    float32, lam_clean, lam_gain).
    """
    rng_theta = np.random.default_rng(int(theta_seed))
    phys = P.sample_prior(PHYS_PRIORS, PHYS_ORDER, n_test, rng_theta)
    theta = np.stack([phys[p] for p in PHYS_ORDER], axis=1).astype(np.float64)

    base = R.load_base_obsconf(RESPONSE_NAME)
    oc_exp = R.scale_exposure(base, EXPOSURE_S)

    lam = {}
    for _, g in GAIN_CASES:
        oc_g = R.gain_shift_obsconf(oc_exp, float(g))
        lam[g] = SIM.fold_theta(BASE_MODEL, PHYS_ORDER, theta, oc_g)

    x = {g: np.zeros_like(lam[g]) for _, g in GAIN_CASES}
    for i in range(n_test):
        seed_i = int(poisson_seed_base) + i
        for _, g in GAIN_CASES:
            rng_p = np.random.default_rng(seed_i)  # SAME seed for both g -> CRN
            x[g][i] = rng_p.poisson(np.clip(lam[g][i], 0.0, None))

    return (theta,
            x[1.00].astype(np.float32), x[1.03].astype(np.float32),
            lam[1.00], lam[1.03])


# ----------------------------------------------------------------------------
# posterior sampling with rejection + clip (exact detect.py pattern)
# ----------------------------------------------------------------------------
def sample_flow(model_dir, x, lo, hi, seed, expected_names, n_samples=N_SAMPLES,
                max_sampling_time=MAX_SAMPLING_TIME, cred=CRED):
    """Sample the flow at ``model_dir`` for every spectrum in ``x``.

    Mirrors ``detect.py::posterior_predictive_replicates`` lines ~228-249:
    try rejection sampling with a time cap; on failure or a short draw, top
    up with unrejected samples; then clip everything into [lo, hi] before
    computing the credible interval / median. Returns per-spectrum
    (med, lo90, hi90) plus diagnostics (mean fraction of raw samples that
    fell outside the prior box pre-clip, and how many spectra needed the
    unrejected top-up).
    """
    post, info = tn.load_posterior(model_dir, device="cpu")
    names = info["param_names"]
    assert names == expected_names, (names, expected_names)
    n_par = len(names)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    n = x.shape[0]
    med = np.zeros((n, n_par)); lo_arr = np.zeros((n, n_par)); hi_arr = np.zeros((n, n_par))
    frac_outside = np.zeros(n)
    n_topup = 0
    torch.manual_seed(seed)
    with torch.no_grad():
        for i in range(n):
            try:
                s_t = post.sample(
                    (n_samples,), x=x_t[i], show_progress_bars=False,
                    reject_outside_prior=True, max_sampling_time=float(max_sampling_time),
                )
            except (RuntimeError, ValueError, TypeError):
                s_t = post.sample(
                    (n_samples,), x=x_t[i], show_progress_bars=False,
                    reject_outside_prior=False,
                )
            s = s_t.detach().cpu().numpy().astype(np.float64)
            if s.shape[0] < n_samples:
                n_topup += 1
                extra = post.sample(
                    (n_samples - s.shape[0],), x=x_t[i], show_progress_bars=False,
                    reject_outside_prior=False,
                ).detach().cpu().numpy().astype(np.float64)
                s = np.vstack([s, extra])
            outside = ((s < lo[None, :]) | (s > hi[None, :])).any(axis=1)
            frac_outside[i] = outside.mean()
            s = np.clip(s, lo[None, :], hi[None, :])
            lo_i, hi_i, med_i = tn.credible_interval(s, cred)
            lo_arr[i] = lo_i; hi_arr[i] = hi_i; med[i] = med_i
    return med, lo_arr, hi_arr, names, float(frac_outside.mean()), n_topup


def metrics(truth, med, lo, hi, idx):
    bias = med[:, idx] - truth[:, idx]
    cover = float(((truth[:, idx] >= lo[:, idx]) & (truth[:, idx] <= hi[:, idx])).mean())
    width = float(np.median(hi[:, idx] - lo[:, idx]))
    return {"bias_mean": float(bias.mean()), "bias_median": float(np.median(bias)),
            "bias_std": float(bias.std(ddof=1)), "coverage90": cover, "med_width90": width}


def paired_stats(delta):
    """Mean/SD/SE/95% CI of a paired difference array. SE = SD/sqrt(n)."""
    n = int(delta.shape[0])
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    se = sd / np.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "ci95": [mean - Z95 * se, mean + Z95 * se]}


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------
def run(n_test=N_TEST, n_samples=N_SAMPLES, max_sampling_time=MAX_SAMPLING_TIME,
       write_outputs=True):
    t_start = time.perf_counter()

    from sbixcal import priors as _priors
    lo5, hi5 = _priors.prior_bounds(PHYS_PRIORS, PHYS_ORDER)
    lo6, hi6 = _priors.prior_bounds({**PHYS_PRIORS, **GAIN_PRIOR}, GM_ORDER)

    print(f"[gen] drawing {n_test} paired thetas (seed={SEED_THETA}), "
         f"folding at g=1.00/1.03 with CRN Poisson (base seed={SEED_POISSON_BASE})...")
    t0 = time.perf_counter()
    theta, x_clean, x_gain, lam_clean, lam_gain = make_paired_population(
        n_test, SEED_THETA, SEED_POISSON_BASE)
    dt_gen = time.perf_counter() - t0
    median_counts_clean = float(np.median(x_clean.sum(1)))
    print(f"[gen done] {dt_gen:.1f}s median_counts clean={median_counts_clean:.0f} "
         f"gain={np.median(x_gain.sum(1)):.0f} "
         f"(mean fold ratio gain/clean={float((lam_gain.sum(1)/lam_clean.sum(1)).mean()):.4f})")

    x_by_case = {"clean": x_clean, "gain": x_gain}
    results = {
        "config": {"n_test": n_test, "n_samples": n_samples, "cred": CRED,
                  "exposure_s": EXPOSURE_S, "gain_prior": [GAIN_LO, GAIN_HI],
                  "max_sampling_time": max_sampling_time,
                  "seed_theta": SEED_THETA, "seed_poisson_base": SEED_POISSON_BASE,
                  "design": "paired (single theta draw, CRN Poisson per pair), "
                            "reject_outside_prior=True + clip-to-box",
                  "level": "bright", "median_counts_clean": median_counts_clean},
        "cases": {}, "diagnostics": {}, "paired": {},
    }

    flow_out = {}  # (flow, gcase) -> (med, lo, hi)
    for flow_key, model_dir, lo_b, hi_b, expected_names in [
        ("fixed", FIXED_DIR, lo5, hi5, PHYS_ORDER),
        ("gainmarg", GM_DIR, lo6, hi6, GM_ORDER),
    ]:
        for gcase, g in GAIN_CASES:
            x = x_by_case[gcase]
            seed = SEED_SAMPLE[(flow_key, gcase)]
            t0 = time.perf_counter()
            med, lo, hi, names, frac_out, n_topup = sample_flow(
                model_dir, x, lo_b, hi_b, seed, expected_names,
                n_samples=n_samples, max_sampling_time=max_sampling_time)
            dt = time.perf_counter() - t0
            flow_out[(flow_key, gcase)] = (med, lo, hi)
            results["diagnostics"][f"{flow_key}_{gcase}"] = {
                "wall_s": dt, "mean_frac_outside_prior_preclip": frac_out,
                "n_spectra_needed_topup": n_topup,
            }
            print(f"[sample] {flow_key:8s} {gcase:5s} (g={g:.2f}): {dt:.1f}s "
                 f"frac_outside_preclip={frac_out:.3f} topup={n_topup}/{n_test}")

    # ---- per-case metrics on clipped samples (bias / coverage90 / width) ----
    for gcase, g in GAIN_CASES:
        medF, loF, hiF = flow_out[("fixed", gcase)]
        medG, loG, hiG = flow_out[("gainmarg", gcase)]
        gF = metrics(theta, medF, loF, hiF, GAMMA_I)
        nF = metrics(theta, medF, loF, hiF, NORM_I)
        gG = metrics(theta, medG, loG, hiG, GAMMA_I)
        nG = metrics(theta, medG, loG, hiG, NORM_I)

        g_med = medG[:, 5]; g_lo = loG[:, 5]; g_hi = hiG[:, 5]
        g_width = float(np.median(g_hi - g_lo))
        g_excludes_1 = float(((g_lo > 1.0) | (g_hi < 1.0)).mean())
        g_stats = {"g_post_median_mean": float(g_med.mean()),
                  "g_post_median_median": float(np.median(g_med)),
                  "g_med_width90": g_width,
                  "g_width_frac_of_prior": g_width / GAIN_PRIOR_WIDTH,
                  "frac_excluding_g1.0": g_excludes_1,
                  "injected_g": g}

        results["cases"][gcase] = {
            "gain": g,
            "median_counts": float(np.median(x_by_case[gcase].sum(1))),
            "fixed": {"gamma": gF, "norm": nF},
            "gain_marg": {"gamma": gG, "norm": nG, "g_marginal": g_stats},
        }
        print(f"  [{gcase}] FIXED   bias={gF['bias_mean']:+.4f} cov90={gF['coverage90']:.3f} | "
             f"GAINMRG bias={gG['bias_mean']:+.4f} cov90={gG['coverage90']:.3f} | "
             f"g_med={g_stats['g_post_median_mean']:.4f} w={g_width:.4f}")

    # ---- PAIRED delta stats: shifted minus clean, per theta, both flows ----
    medF_clean, _, _ = flow_out[("fixed", "clean")]
    medF_gain, _, _ = flow_out[("fixed", "gain")]
    medG_clean, _, _ = flow_out[("gainmarg", "clean")]
    medG_gain, _, _ = flow_out[("gainmarg", "gain")]

    d_gamma_fixed = medF_gain[:, GAMMA_I] - medF_clean[:, GAMMA_I]
    d_gamma_gm = medG_gain[:, GAMMA_I] - medG_clean[:, GAMMA_I]
    d_lognorm_fixed = np.log10(medF_gain[:, NORM_I]) - np.log10(medF_clean[:, NORM_I])
    d_lognorm_gm = np.log10(medG_gain[:, NORM_I]) - np.log10(medG_clean[:, NORM_I])

    results["paired"] = {
        "gamma_bias_delta": {"fixed": paired_stats(d_gamma_fixed),
                             "gain_marg": paired_stats(d_gamma_gm)},
        "log10norm_bias_delta": {"fixed": paired_stats(d_lognorm_fixed),
                                 "gain_marg": paired_stats(d_lognorm_gm)},
    }
    print("\n[paired Gamma delta, shifted-clean] "
         f"fixed mean={results['paired']['gamma_bias_delta']['fixed']['mean']:+.5f} "
         f"SE={results['paired']['gamma_bias_delta']['fixed']['se']:.5f} | "
         f"gain_marg mean={results['paired']['gamma_bias_delta']['gain_marg']['mean']:+.5f} "
         f"SE={results['paired']['gamma_bias_delta']['gain_marg']['se']:.5f}")

    dt_total = time.perf_counter() - t_start
    results["config"]["wall_s_total"] = dt_total
    print(f"\n[total wall] {dt_total:.1f}s ({dt_total/60:.2f} min)")

    if write_outputs:
        # ---- figure: paired-delta histograms per flow + g-marginal panels ----
        fig, axes = plt.subplots(3, 2, figsize=(11, 13))

        for col, (d, title) in enumerate([(d_gamma_fixed, "fixed"), (d_gamma_gm, "gain-marg")]):
            ax = axes[0, col]
            ax.hist(d, bins=30, color="tab:red" if col == 0 else "tab:blue", alpha=0.75)
            ax.axvline(0.0, color="k", ls="--", lw=1)
            ps = results["paired"]["gamma_bias_delta"]["fixed" if col == 0 else "gain_marg"]
            ax.axvline(ps["mean"], color="green", ls="-", lw=1.5)
            ax.set_title(f"{title}: paired $\\Delta\\Gamma$ (shifted - clean)\n"
                        f"mean={ps['mean']:+.4f} SE={ps['se']:.4f} "
                        f"95% CI [{ps['ci95'][0]:+.4f}, {ps['ci95'][1]:+.4f}]", fontsize=9)
            ax.set_xlabel("$\\Delta\\Gamma$"); ax.set_ylabel("count"); ax.grid(alpha=0.3)

        for col, (d, title) in enumerate([(d_lognorm_fixed, "fixed"), (d_lognorm_gm, "gain-marg")]):
            ax = axes[1, col]
            ax.hist(d, bins=30, color="tab:red" if col == 0 else "tab:blue", alpha=0.75)
            ax.axvline(0.0, color="k", ls="--", lw=1)
            ps = results["paired"]["log10norm_bias_delta"]["fixed" if col == 0 else "gain_marg"]
            ax.axvline(ps["mean"], color="green", ls="-", lw=1.5)
            ax.set_title(f"{title}: paired $\\Delta\\log_{{10}}$norm (shifted - clean)\n"
                        f"mean={ps['mean']:+.4f} SE={ps['se']:.4f} "
                        f"95% CI [{ps['ci95'][0]:+.4f}, {ps['ci95'][1]:+.4f}]", fontsize=9)
            ax.set_xlabel("$\\Delta\\log_{10}$(powerlaw norm)"); ax.set_ylabel("count")
            ax.grid(alpha=0.3)

        for col, gcase in enumerate(["clean", "gain"]):
            ax = axes[2, col]
            medG, loG, hiG = flow_out[("gainmarg", gcase)]
            g_med = medG[:, 5]
            g = dict(GAIN_CASES)[gcase]
            ax.hist(g_med, bins=30, range=(GAIN_LO, GAIN_HI), color="tab:blue", alpha=0.75,
                   label="per-spectrum g posterior median")
            ax.axvline(g, color="k", ls="--", lw=1.5, label=f"injected g={g:.2f}")
            ax.axvline(1.0, color="gray", ls=":", lw=1)
            gs = results["cases"][gcase]["gain_marg"]["g_marginal"]
            ax.set_title(f"g-marginal ({gcase}, g={g:.2f}): "
                        f"width={gs['g_med_width90']:.3f} "
                        f"({gs['g_width_frac_of_prior']:.0%} of prior)", fontsize=9)
            ax.set_xlabel("gain g"); ax.set_ylabel("count"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        fig.suptitle(f"Paired gain-shift eval (bright, ~{median_counts_clean:.0f} counts): "
                    "same-theta CRN pairs, N=500, rejection+clip sampling", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        figpath = OUT / "paired_gain_bias_bright.png"
        fig.savefig(figpath, dpi=130)
        plt.close(fig)
        print(f"[fig] {figpath}")

        jp = OUT / "paired_gain_bias_bright.json"
        jp.write_text(json.dumps(results, indent=2))
        print(f"[json] {jp}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--max-sampling-time", type=float, default=MAX_SAMPLING_TIME)
    ap.add_argument("--no-write", action="store_true",
                    help="skip writing json/png (for quick wiring checks)")
    args = ap.parse_args()
    run(n_test=args.n_test, n_samples=args.n_samples,
       max_sampling_time=args.max_sampling_time, write_outputs=not args.no_write)


if __name__ == "__main__":
    main()
