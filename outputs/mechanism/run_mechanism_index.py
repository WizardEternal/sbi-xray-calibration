r"""E6: Fisher-metric invisibility index (absorbable-fraction mechanism figure).

Explanatory MECHANISM figure for the successor paper (not a headline result).
Answers: for a given misspecification family, how much of the data perturbation
it induces can be reabsorbed by shifting the well-specified model's own 5
parameters, in the local Poisson-Fisher metric? This is the standard Fisher-
information / sloppy-model-geometry statement of "degenerate with the model,"
not a new statistic (see the report for citations: Cutler & Vallisneri 2007;
Amara & Refregier 2008; Transtrum, Machta & Sethna sloppy-model geometry; the
complement of Alsing & Wandelt 2019 nuisance-hardened score compression).

For parameter point theta (5 physical params of Model A production:
tbabs*(powerlaw+blackbody)):

    mu(theta)   = expected folded counts per channel (fold_theta; deterministic,
                  NEVER Poisson-sampled here)
    J = d mu/d theta            (channels x 5, central finite differences)
    delta_mu    = mu_misspec(theta) - mu(theta)   (misspecification at its
                  committed strength, SAME theta, SAME response otherwise)
    G = diag(1/mu)              (Poisson-Fisher metric)
    W = diag(1/sqrt(mu)),  A = W . J,  b = W . delta_mu
    I = || P_A b ||^2 / || b ||^2

P_A is the orthogonal projector onto col(A), built via a reduced QR
decomposition (Q R = A, P_A b = Q Q^T b); (A^T A)^-1 is never formed explicitly.
I in [0,1]: I ~ 1 means the systematic's whitened data-space signature lies (to
first order, at this theta) almost entirely inside the span the model's own
parameters can move in -- so a likelihood-based check that only compares
best-fit residuals cannot see it, because refitting theta reabsorbs it. Lower I
means a whitened residual survives outside that span -- a likelihood-based
check CAN in principle see it (whether a specific detector's summary is
sensitive enough is a separate, measured question -- this is a mechanism, not a
guarantee).

No training, no MCMC/NS sampling, no Poisson sampling anywhere in this script.
Only cheap deterministic forward folds through `sbixcal.simulate.fold_theta`
and `jaxspec.data.util.fakeit_for_multiple_parameters(..., apply_stat=False)`.

Reproduce (repo root, repo venv):
    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe outputs\mechanism\run_mechanism_index.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs" / "mechanism"
OUT.mkdir(parents=True, exist_ok=True)

from sbixcal import models as _models
from sbixcal import priors as _priors
from sbixcal import responses as _responses
from sbixcal.simulate import fold_theta, load_config
from jaxspec.data.util import fakeit_for_multiple_parameters

# ---------------------------------------------------------------------------
# fixed configuration (matches the note's production setup and its committed
# misspecification grids -- see configs/sim_modelA_prod.yaml and
# configs/detect.yaml; strengths below match the E1 IS-ESS committed-grid
# comparison point, deliberation/E1_is_ess_report.md)
# ---------------------------------------------------------------------------
RESPONSE = "NGC7793_ULX4_PN"                 # bundled real XMM EPIC-pn (Quintin+2021)
BASE_MODEL = "tbabs_powerlaw_bb"             # Model A production, 5 params
GLOBAL_SEED = 20260611                       # repo convention; single shared theta draw
N_DRAWS = 200
REL_STEP = 1.0e-3                            # central finite-diff relative step

LEVELS = {
    "medium": 353.4,   # seconds; sim_modelA_prod.yaml level 'medium' (~986 counts median)
    "bright": 3534.0,  # seconds; sim_modelA_prod.yaml level 'bright' (~9982 counts median)
}

# committed single-point strengths (strongest / headline grid point for each
# family, matching configs/detect.yaml production grids and the E1 report's
# single-point comparison: B1 norm=3e-4, B4 gain=3%).
FAMILY_STRENGTH = {
    "B1": {"line_energy_kev": 6.4, "line_sigma_kev": 0.05, "gauss_1_norm": 3.0e-4},
    "B4": {"gain_pct": 3.0},
    "B2": {"tbpcf_1_f": 0.3},
    "B3": {"kT": 1.5, "use_diskbb": False},
}
FAMILIES = ["B1", "B4", "B2", "B3"]  # folding cost is one extra vectorized fold
                                       # per family (~102 channels x N draws);
                                       # cheap, so all four are run, not just
                                       # the required B1/B4 minimum.

# guard: exclude/flag theta draws where the whitened perturbation norm-squared
# is numerically negligible (I is a 0/0-flavoured ratio there). Set from the
# observed population (see report): the bulk of ||b||^2 sits at >=1e-2 for
# every family/level tried; 1e-8 is >=6 orders of magnitude below that, i.e. a
# floating-point floor, not a physical cut.
BNORM2_GUARD = 1.0e-8


def build_theta():
    cfg = load_config(str(ROOT / "configs" / "sim_modelA_prod.yaml"))
    prior_cfg = cfg["priors"]
    param_order = _models.MODEL_PARAMS[BASE_MODEL]
    rng = np.random.default_rng(GLOBAL_SEED)
    samples = _priors.sample_prior(prior_cfg, param_order, N_DRAWS, rng)
    theta = np.stack([np.asarray(samples[p]) for p in param_order], axis=1).astype(np.float64)
    lows, highs = _priors.prior_bounds(prior_cfg, param_order)
    return theta, param_order, lows, highs, prior_cfg


def obsconf_for_level(level_name):
    base = _responses.load_base_obsconf(RESPONSE)
    return _responses.scale_exposure(base, LEVELS[level_name])


BOUND_MARGIN = 0.5   # a clipped step never uses more than this fraction of the
                      # remaining room to a bound -- see the Gamma=1 landmine
                      # note below; without a margin, clipping a step to land
                      # EXACTLY on a bound can land exactly on an unrelated
                      # analytic singularity of the spectral model that happens
                      # to coincide with that bound.


def central_diff_jacobian(theta, param_order, lows, highs, obsconf):
    """J[i, :, k] = d mu / d theta_k at theta[i], central finite diff, relative
    step REL_STEP * |theta_k|, clipped per-sample so theta_k +/- h stays
    STRICTLY inside (lows[k], highs[k]) (the prior box == the physical bound
    for every one of these 5 params, all of which are positive-valued) with a
    BOUND_MARGIN safety factor. Returns (J (N,C,P), mu0 (N,C), step_clip_frac
    (P,) = fraction of draws whose step was shrunk by clipping).

    LANDMINE (hit and fixed): jaxspec's Powerlaw component integrates
    E^-Gamma analytically, which switches to a log form at Gamma=1 exactly
    (the textbook power-law-integral special case). The production prior's
    powerlaw_1_alpha lower bound is exactly 1.0 (Barret & Dupourque 2024,
    Table 1). One N=200 draw (Gamma=1.00047) is close enough to that bound
    that clipping the backward step to exactly `theta - (theta - low)` landed
    the probe point at Gamma=1.0 EXACTLY, which returned an all-NaN folded
    spectrum (not a bounds violation -- an unrelated analytic singularity that
    happens to sit on the prior edge). BOUND_MARGIN keeps every clipped probe
    point strictly interior to the box, not on its edge, which avoids this
    without changing the well-interior (unclipped) majority of draws at all.
    """
    N, P = theta.shape
    mu0 = fold_theta(BASE_MODEL, param_order, theta, obsconf)
    C = mu0.shape[1]
    J = np.empty((N, C, P), dtype=np.float64)
    clip_frac = np.zeros(P)
    for k in range(P):
        h_nom = REL_STEP * np.abs(theta[:, k])
        h = np.minimum(h_nom, BOUND_MARGIN * (highs[k] - theta[:, k]))
        h = np.minimum(h, BOUND_MARGIN * (theta[:, k] - lows[k]))
        h = np.maximum(h, 1e-300)  # never exactly zero (would only occur at a
                                    # sample sitting exactly on a bound, which
                                    # sample_prior's open-interval draws do not
                                    # produce)
        clip_frac[k] = float(np.mean(h < 0.999 * h_nom))
        theta_p = theta.copy(); theta_p[:, k] += h
        theta_m = theta.copy(); theta_m[:, k] -= h
        mu_p = fold_theta(BASE_MODEL, param_order, theta_p, obsconf)
        mu_m = fold_theta(BASE_MODEL, param_order, theta_m, obsconf)
        if not (np.all(np.isfinite(mu_p)) and np.all(np.isfinite(mu_m))):
            bad = np.where(~np.all(np.isfinite(mu_p), axis=1)
                            | ~np.all(np.isfinite(mu_m), axis=1))[0]
            raise FloatingPointError(
                f"non-finite fold at param {param_order[k]}, draw idx {bad.tolist()}; "
                f"theta={theta[bad].tolist()}, h={h[bad].tolist()}"
            )
        J[:, :, k] = (mu_p - mu_m) / (2.0 * h)[:, None]
    return J, mu0, clip_frac


def delta_mu_b1(theta, param_order, obsconf):
    N = theta.shape[0]
    model = _models.build_model_b1(BASE_MODEL)
    params = {p: theta[:, j] for j, p in enumerate(param_order)}
    s = FAMILY_STRENGTH["B1"]
    params["gauss_1_El"] = np.full(N, s["line_energy_kev"])
    params["gauss_1_sigma"] = np.full(N, s["line_sigma_kev"])
    params["gauss_1_norm"] = np.full(N, s["gauss_1_norm"])
    mu = np.asarray(
        fakeit_for_multiple_parameters(obsconf, model, params, rng_key=0, apply_stat=False),
        dtype=np.float64,
    )
    return mu


def delta_mu_b4(theta, param_order, obsconf):
    gain = 1.0 + FAMILY_STRENGTH["B4"]["gain_pct"] / 100.0
    oc_shift = _responses.gain_shift_obsconf(obsconf, gain)
    mu = fold_theta(BASE_MODEL, param_order, theta, oc_shift)
    return mu


def delta_mu_b2(theta, param_order, obsconf):
    N = theta.shape[0]
    model = _models.build_model_b2(BASE_MODEL)
    params = {p: theta[:, j] for j, p in enumerate(param_order)}
    nh = params.pop("tbabs_1_nh")
    params["tbpcf_1_nh"] = nh
    params["tbpcf_1_f"] = np.full(N, FAMILY_STRENGTH["B2"]["tbpcf_1_f"])
    mu = np.asarray(
        fakeit_for_multiple_parameters(obsconf, model, params, rng_key=0, apply_stat=False),
        dtype=np.float64,
    )
    return mu


def delta_mu_b3(theta, param_order, obsconf):
    N = theta.shape[0]
    model = _models.build_model_b3(BASE_MODEL, use_diskbb=FAMILY_STRENGTH["B3"]["use_diskbb"])
    params = {p: theta[:, j] for j, p in enumerate(param_order)}
    pl_norm = params.pop("powerlaw_1_norm")
    params.pop("powerlaw_1_alpha")
    params["brems_1_kT"] = np.full(N, FAMILY_STRENGTH["B3"]["kT"])
    params["brems_1_norm"] = pl_norm
    mu = np.asarray(
        fakeit_for_multiple_parameters(obsconf, model, params, rng_key=0, apply_stat=False),
        dtype=np.float64,
    )
    return mu


DELTA_FN = {"B1": delta_mu_b1, "B4": delta_mu_b4, "B2": delta_mu_b2, "B3": delta_mu_b3}


def invisibility_index(J, mu0, mu_misspec):
    """Per-theta I = ||P_A b||^2 / ||b||^2, QR projector, guarded.

    Returns dict with arrays I (N,), bnorm2 (N,), flagged (N, bool),
    rank_deficient (N, bool, True iff QR found col(A) rank < P at that theta).
    """
    N, C, P = J.shape
    delta = mu_misspec - mu0
    I = np.full(N, np.nan)
    bnorm2 = np.zeros(N)
    flagged = np.zeros(N, dtype=bool)
    rank_def = np.zeros(N, dtype=bool)
    for i in range(N):
        w = 1.0 / np.sqrt(mu0[i])            # (C,) Poisson-Fisher whitening
        A = w[:, None] * J[i]                # (C,P)
        b = w * delta[i]                     # (C,)
        bn2 = float(b @ b)
        bnorm2[i] = bn2
        if bn2 < BNORM2_GUARD:
            flagged[i] = True
            continue
        Q, R = np.linalg.qr(A, mode="reduced")     # A = QR, never (A^T A)^-1
        rank = int(np.sum(np.abs(np.diag(R)) > 1e-10 * np.abs(np.diag(R)).max()))
        if rank < P:
            rank_def[i] = True
        proj = Q @ (Q.T @ b)                 # P_A b
        val = float(proj @ proj) / bn2
        I[i] = min(max(val, 0.0), 1.0)        # clip tiny float overshoot past [0,1]
    return {"I": I, "bnorm2": bnorm2, "flagged": flagged, "rank_deficient": rank_def}


def summarize(I, flagged):
    good = I[~flagged]
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
    theta, param_order, lows, highs, prior_cfg = build_theta()
    print(f"theta drawn: {theta.shape}, seed={GLOBAL_SEED}, params={param_order}")

    results = {
        "meta": {
            "response": RESPONSE,
            "base_model": BASE_MODEL,
            "param_order": param_order,
            "seed": GLOBAL_SEED,
            "n_draws": N_DRAWS,
            "rel_step": REL_STEP,
            "levels_exposure_s": LEVELS,
            "family_strength": FAMILY_STRENGTH,
            "bnorm2_guard": BNORM2_GUARD,
            "theta": theta.tolist(),
        },
        "per_theta": {},
        "summary": {},
        "diagnostics": {},
    }

    for level in LEVELS:
        print(f"\n=== level={level} (exposure {LEVELS[level]} s) ===")
        obsconf = obsconf_for_level(level)
        J, mu0, clip_frac = central_diff_jacobian(theta, param_order, lows, highs, obsconf)
        print(f"  mu0: min={mu0.min():.4g} median={np.median(mu0):.4g} max={mu0.max():.4g}")
        print(f"  finite-diff step clip fraction per param: "
              f"{dict(zip(param_order, np.round(clip_frac, 4)))}")
        assert np.all(mu0 > 0), "mu0 has non-positive channels -- whitening undefined"

        results["per_theta"][level] = {}
        results["summary"][level] = {}
        results["diagnostics"][level] = {
            "mu0_min": float(mu0.min()),
            "mu0_median": float(np.median(mu0)),
            "mu0_max": float(mu0.max()),
            "step_clip_frac": {p: float(c) for p, c in zip(param_order, clip_frac)},
        }

        for fam in FAMILIES:
            mu_mis = DELTA_FN[fam](theta, param_order, obsconf)
            res = invisibility_index(J, mu0, mu_mis)
            results["per_theta"][level][fam] = {
                "I": [None if not np.isfinite(v) else float(v) for v in res["I"]],
                "bnorm2": [float(v) for v in res["bnorm2"]],
                "flagged": [bool(v) for v in res["flagged"]],
                "rank_deficient": [bool(v) for v in res["rank_deficient"]],
            }
            summ = summarize(res["I"], res["flagged"])
            results["summary"][level][fam] = summ
            n_rankdef = int(res["rank_deficient"].sum())
            print(f"  {fam}: median I = {summ['median']:.4f}  "
                  f"IQR [{summ['q25']:.4f}, {summ['q75']:.4f}]  "
                  f"(n={summ['n']}, flagged={summ['n_flagged']}, "
                  f"rank_deficient={n_rankdef}, "
                  f"bnorm2 median={np.median(res['bnorm2']):.4g})"
                  if summ["median"] is not None else
                  f"  {fam}: ALL FLAGGED (n_flagged={summ['n_flagged']})")

    out_json = OUT / "mechanism_index_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out_json}  ({out_json.stat().st_size/1024:.0f} KiB)")
    print(f"total wall time {time.time()-t0:.1f} s")
    return results


if __name__ == "__main__":
    main()
