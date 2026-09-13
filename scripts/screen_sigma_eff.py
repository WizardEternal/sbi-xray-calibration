r"""Fisher gain-information screen: sigma_eff(g) for a simulated spectrum population.

WHAT THIS BACKS
---------------
The successor paper's section on the gain-marginalized NS cross-check says
(the sigma_eff sentence):

    "The finite-shift Fisher error is sigma_eff = Delta g / sqrt(R), with R the
     residual power a gain shift Delta g leaves after the other five parameters
     absorb what they can. For the three it reads 0.037, 0.050, and 0.039
     against a prior standard deviation of 0.0289. ... We use sigma_eff
     ordinally, only to screen which spectra carry gain information at all."

Those per-spectrum numbers came from 2026-08-12 working files outside the repo
that no longer exist, so they were not reproducible from anything committed.
This script is the committed replacement. It re-derives sigma_eff from first
principles and screens the whole simulated population, not only the ten spectra
that were run.

DEFINITION (exactly what is computed)
-------------------------------------
At a spectrum's true parameter vector theta (5 physical parameters of Model A,
tbabs*(powerlaw + blackbodyrad)) and at the population's exposure:

    mu0    = fold_theta(theta)                        expected counts per channel
    J      = d mu / d theta   (C x 5)                 central differences, relative
                                                      step REL_STEP, clipped to stay
                                                      strictly inside the prior box
    dmu/dg = [mu(g = 1 + dg) - mu(g = 1 - dg)] / (2 dg)
                                                      gain applied to the response by
                                                      responses.gain_shift_obsconf,
                                                      the same operator the simulator,
                                                      the NS likelihood and the E6
                                                      misspecification family B4 use
    W      = diag(1 / sqrt(mu0))                      Poisson-Fisher whitening
    A      = W J          b = W dmu/dg
    I      = ||P_A b||^2 / ||b||^2                    fraction of the gain signature
                                                      the other five parameters absorb
                                                      (P_A from a reduced QR of A)
    R      = (1 - I) ||b||^2                          out-of-span residual power

    sigma_eff_marginal    = 1 / sqrt(R)          = sqrt([F^-1]_gg) of the 6x6 Fisher
    sigma_eff_conditional = 1 / sqrt(||b||^2)    = 1 / sqrt(F_gg), other five fixed

sigma_eff_marginal is identically the paper's Delta g / sqrt(R_paper): with the
symmetric finite shift, b_paper = W [mu(1+dg) - mu(1-dg)] / 2 = dg * b, so
R_paper = dg^2 R and Delta g / sqrt(R_paper) = 1 / sqrt(R). Writing it as a
Fisher inverse just makes the "marginal vs conditional" choice explicit.

The gain prior is U[0.95, 1.05], sd 0.10 / sqrt(12) = 0.0288675. Combining the
likelihood width with that prior in quadrature (Gaussian approximation),

    sigma_post = 1 / sqrt(1 / sigma_prior^2 + 1 / sigma_eff_marginal^2)
    shrink_pred = sigma_post / sigma_prior

gives a predicted posterior shrink. THE PAPER DOES NOT CLAIM THIS AS A WIDTH
PREDICTION -- sigma_eff is used ordinally, to screen which spectra carry gain
information at all. shrink_pred is emitted here for diagnostics only, and the
report records that on the four bright spectra actually run it puts i8 and i394
in the wrong order relative to the measured NS shrinks.

WHY dg = 0.03
-------------
0.03 is E6's committed B4 strength (outputs/mechanism/run_mechanism_index.py,
FAMILY_STRENGTH["B4"]["gain_pct"] = 3.0), so the screen and the E6 mechanism
figure describe the same perturbation. It is also the step that reproduces the
recorded values: over dg in {0.02, 0.025, 0.03, 0.04, 0.05}, only dg = 0.03
lands all seven recorded sigma_eff values within 4 per cent (the others scatter
by 10-16 per cent, because the folded difference is a small residual after the
five-parameter projection and jaxspec folds in float32). See
outputs/gain_marg/sigma_eff_screen/REPORT.md for the scan.

REPRODUCTION OF THE RECORDED VALUES (2026-09-02)
------------------------------------------------
Against the seven per-spectrum sigma_eff values on record from the 2026-08-12
working files, this definition gives:

    i8   0.0385 vs 0.0390 (-1.3%)      i22  0.4673 vs 0.4568 (+2.3%)
    i394 0.0506 vs 0.0500 (+1.3%)      i87  0.2059 vs 0.2026 (+1.6%)
    i416 0.0374 vs 0.0366 (+2.4%)      i482 0.1262 vs 0.1228 (+2.8%)
    i238 0.2032 vs 0.1970 (+3.2%)

So the paper's 0.037 / 0.050 / 0.039 for i416 / i394 / i8 reproduces. The
residual is a systematic near +2 per cent whose source is not pinned down; it is
larger than the estimator's own numerical scatter (below), so it is a real
small difference in how the original was computed, not noise. It does not touch
the ordinal use: the population spans sigma_eff 0.024 to 0.43 at bright.

NUMERICAL SCATTER. jaxspec folds in float32 and the gain derivative is a small
residual left after projecting out five parameters, so sigma_eff depends weakly
on how many spectra are folded in one call. Over batch sizes 4 to 500 the
coefficient of variation is 0.03 to 0.74 per cent per spectrum; a single-row
fold can sit 4 per cent high. Repeating a run at a fixed batch size is exact.
This script always folds the whole population in one call, so a given --n-pop is
reproducible; do not compare sigma_eff across different --n-pop at the third
digit.

SEEDS AND POPULATION
--------------------
The population is the one outputs/gain_marg/run_ns_smallset.py draws from:
misspec.simulate_misspec_population(base_model, PHYS_PRIORS, obsconf, "B4",
strength, n_pop, seed_eval_base + int(strength * 10)), with
seed_eval_base = 20260611 + 40000 = 20300611 and strength = 0.0 (clean control,
injected g = 1.0). NOTE: docstrings predating 2026-08-12 quote 20260651; that is
wrong. The counts gate below checks the regenerated spectra against the ten
totals recorded in the E7 verification reports.

Deterministic: no RNG beyond that population draw, no sampling, no training.

USAGE (repo root, repo venv)
----------------------------
    .venv\Scripts\python.exe scripts\screen_sigma_eff.py --level bright
    .venv\Scripts\python.exe scripts\screen_sigma_eff.py --level medium

Runtime is a few tens of seconds for 500 spectra (13 vectorized folds plus 500
QR decompositions of a 102 x 5 matrix).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sbixcal import models as _models          # noqa: E402
from sbixcal import priors as _priors          # noqa: E402
from sbixcal import responses as _responses    # noqa: E402
from sbixcal.misspec import simulate_misspec_population  # noqa: E402
from sbixcal.simulate import fold_theta        # noqa: E402

# --------------------------------------------------------------------------
# configuration, mirroring outputs/gain_marg/ns_gainmarg.py exactly
# --------------------------------------------------------------------------
BASE_MODEL = "tbabs_powerlaw_bb"
RESPONSE = "NGC7793_ULX4_PN"
EXPOSURE_S = {"medium": 353.4, "bright": 3534.0}

PHYS_PRIORS = {
    "tbabs_1_nh":          {"dist": "uniform",    "low": 0.15,   "high": 0.35},
    "powerlaw_1_alpha":    {"dist": "uniform",    "low": 1.0,    "high": 3.0},
    "powerlaw_1_norm":     {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":   {"dist": "uniform",    "low": 0.3,    "high": 3.0},
    "blackbodyrad_1_norm": {"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}
PARAM_ORDER = _models.MODEL_PARAMS[BASE_MODEL]
GAIN_LO, GAIN_HI = 0.95, 1.05
GAIN_PRIOR_SD = (GAIN_HI - GAIN_LO) / np.sqrt(12.0)     # 0.0288675

# spectra already run through gain-marginalized NS (outputs/gain_marg/ns_smallset)
ALREADY_RUN = {"medium": [22, 87, 91, 95, 197, 482], "bright": [8, 238, 394, 416]}

# the recorded counts gate
COUNTS_GATE = {
    "medium": {22: 3853, 87: 384, 91: 429, 95: 291, 197: 599, 482: 5204},
    "bright": {8: 25067, 238: 10496, 394: 6829, 416: 54509},
}

BOUND_MARGIN = 0.5    # a clipped finite-difference step never uses more than this
                      # fraction of the room left to a bound. Also keeps probe
                      # points strictly interior, which matters because jaxspec's
                      # Powerlaw switches to a log form at Gamma = 1 exactly, and
                      # 1.0 is the prior's lower bound (E6 hit this; same guard).


def build_population(level: str, n_pop: int, seed_eval_base: int, strength: float):
    """Regenerate the eval_gainmarg population deterministically."""
    obsconf = _responses.scale_exposure(
        _responses.load_base_obsconf(RESPONSE), EXPOSURE_S[level])
    seed = seed_eval_base + int(strength * 10)
    x, theta, present = simulate_misspec_population(
        BASE_MODEL, PHYS_PRIORS, obsconf, "B4", float(strength), n_pop, seed)
    assert present == PARAM_ORDER, (present, PARAM_ORDER)
    return obsconf, np.asarray(x, float), np.asarray(theta, float), seed


def physics_jacobian(theta, obsconf, rel_step):
    """J[i, :, k] = d mu / d theta_k, central differences, box-clipped steps."""
    lows, highs = _priors.prior_bounds(PHYS_PRIORS, PARAM_ORDER)
    n, npar = theta.shape
    mu0 = np.asarray(fold_theta(BASE_MODEL, PARAM_ORDER, theta, obsconf), float)
    J = np.empty((n, mu0.shape[1], npar))
    for k in range(npar):
        h = rel_step * np.abs(theta[:, k])
        h = np.minimum(h, BOUND_MARGIN * (highs[k] - theta[:, k]))
        h = np.minimum(h, BOUND_MARGIN * (theta[:, k] - lows[k]))
        h = np.maximum(h, 1e-300)
        tp = theta.copy(); tp[:, k] += h
        tm = theta.copy(); tm[:, k] -= h
        mp = np.asarray(fold_theta(BASE_MODEL, PARAM_ORDER, tp, obsconf), float)
        mm = np.asarray(fold_theta(BASE_MODEL, PARAM_ORDER, tm, obsconf), float)
        if not (np.all(np.isfinite(mp)) and np.all(np.isfinite(mm))):
            raise FloatingPointError(f"non-finite fold probing {PARAM_ORDER[k]}")
        J[:, :, k] = (mp - mm) / (2.0 * h)[:, None]
    return J, mu0


def gain_derivative(theta, obsconf, delta_g):
    """d mu / d g at g = 1, symmetric finite shift of +/- delta_g."""
    oc_p = _responses.gain_shift_obsconf(obsconf, 1.0 + delta_g)
    oc_m = _responses.gain_shift_obsconf(obsconf, 1.0 - delta_g)
    mp = np.asarray(fold_theta(BASE_MODEL, PARAM_ORDER, theta, oc_p), float)
    mm = np.asarray(fold_theta(BASE_MODEL, PARAM_ORDER, theta, oc_m), float)
    return (mp - mm) / (2.0 * delta_g)


def sigma_eff(J, mu0, dmu_dg):
    """Per-spectrum I, ||b||^2, R, and the marginal / conditional gain widths."""
    n = J.shape[0]
    out = {k: np.full(n, np.nan) for k in
           ("I", "bnorm2", "R", "sigma_marginal", "sigma_conditional")}
    rank_def = np.zeros(n, dtype=bool)
    for i in range(n):
        w = 1.0 / np.sqrt(mu0[i])
        A = w[:, None] * J[i]
        b = w * dmu_dg[i]
        bn2 = float(b @ b)
        Q, Rq = np.linalg.qr(A, mode="reduced")
        diag = np.abs(np.diag(Rq))
        if int(np.sum(diag > 1e-10 * diag.max())) < A.shape[1]:
            rank_def[i] = True
        proj = Q @ (Q.T @ b)
        Ival = min(max(float(proj @ proj) / bn2, 0.0), 1.0)
        Rres = (1.0 - Ival) * bn2
        out["I"][i] = Ival
        out["bnorm2"][i] = bn2
        out["R"][i] = Rres
        out["sigma_conditional"][i] = 1.0 / np.sqrt(bn2)
        out["sigma_marginal"][i] = np.inf if Rres <= 0 else 1.0 / np.sqrt(Rres)
    out["rank_deficient"] = rank_def
    return out


def prior_edge_margins(theta):
    """Margin to each prior edge as a fraction of the prior range, measured in the
    coordinate the prior is uniform in (linear for uniform, log for loguniform).

    Returns (lower (n,5), upper (n,5)) in [0, 0.5]-ish units of prior range.
    """
    n = theta.shape[0]
    lo_m = np.empty((n, len(PARAM_ORDER)))
    hi_m = np.empty((n, len(PARAM_ORDER)))
    for k, p in enumerate(PARAM_ORDER):
        cfg = PHYS_PRIORS[p]
        lo, hi = float(cfg["low"]), float(cfg["high"])
        v = theta[:, k]
        if cfg["dist"] == "loguniform":
            lo, hi, v = np.log(lo), np.log(hi), np.log(v)
        rng = hi - lo
        lo_m[:, k] = (v - lo) / rng
        hi_m[:, k] = (hi - v) / rng
    return lo_m, hi_m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", default="bright", choices=sorted(EXPOSURE_S))
    ap.add_argument("--n-pop", type=int, default=500)
    ap.add_argument("--seed-eval-base", type=int, default=20260611 + 40000)
    ap.add_argument("--strength", type=float, default=0.0,
                    help="B4 injected gain in per cent; 0.0 = the clean control "
                         "population the NS small-set was drawn from")
    ap.add_argument("--delta-g", type=float, default=0.03)
    ap.add_argument("--rel-step", type=float, default=1.0e-3,
                    help="relative central-difference step for the 5 physics columns")
    ap.add_argument("--soft-band-kev", type=float, default=1.2,
                    help="channels with centre below this are the soft band, where "
                         "the gain information lives (tbabs turnover)")
    ap.add_argument("--out", default=None,
                    help="output directory (default outputs/gain_marg/sigma_eff_screen)")
    args = ap.parse_args(argv)

    outdir = Path(args.out) if args.out else ROOT / "outputs" / "gain_marg" / "sigma_eff_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    obsconf, x, theta, pop_seed = build_population(
        args.level, args.n_pop, args.seed_eval_base, args.strength)
    totals = x.sum(axis=1)
    e_lo = np.asarray(obsconf.coords["e_min_folded"], float)
    e_hi = np.asarray(obsconf.coords["e_max_folded"], float)
    soft = 0.5 * (e_lo + e_hi) < args.soft_band_kev
    soft_counts = x[:, soft].sum(axis=1)

    gate = {}
    for idx, want in COUNTS_GATE.get(args.level, {}).items():
        if idx < args.n_pop:
            got = int(round(totals[idx]))
            gate[str(idx)] = {"expected": want, "got": got, "pass": got == want}
    n_gate_fail = sum(1 for v in gate.values() if not v["pass"])
    print(f"counts gate: {len(gate) - n_gate_fail}/{len(gate)} pass"
          + ("" if not n_gate_fail else
             "  FAIL: " + str({k: v for k, v in gate.items() if not v['pass']})))

    print(f"folding Jacobian for {args.n_pop} spectra ...")
    J, mu0 = physics_jacobian(theta, obsconf, args.rel_step)
    dmu = gain_derivative(theta, obsconf, args.delta_g)
    res = sigma_eff(J, mu0, dmu)

    sm = res["sigma_marginal"]
    sigma_post = 1.0 / np.sqrt(1.0 / GAIN_PRIOR_SD**2 + 1.0 / sm**2)
    shrink_pred = sigma_post / GAIN_PRIOR_SD

    lo_m, hi_m = prior_edge_margins(theta)
    kt_k = PARAM_ORDER.index("blackbodyrad_1_kT")
    kt_hi = float(PHYS_PRIORS["blackbodyrad_1_kT"]["high"])
    kt_ceiling_frac_range = hi_m[:, kt_k]                       # fraction of prior range
    kt_ceiling_frac_value = (kt_hi - theta[:, kt_k]) / kt_hi    # fraction of 3.0
    min_margin = np.minimum(lo_m, hi_m).min(axis=1)

    already = set(ALREADY_RUN.get(args.level, []))
    rows = []
    for i in range(args.n_pop):
        row = {
            "index": i,
            "total_counts": int(round(totals[i])),
            "soft_counts": int(round(soft_counts[i])),
            "soft_frac": round(float(soft_counts[i] / max(totals[i], 1.0)), 5),
            "sigma_eff_marginal": float(sm[i]),
            "sigma_eff_conditional": float(res["sigma_conditional"][i]),
            "absorbed_fraction_I": float(res["I"][i]),
            "bnorm2": float(res["bnorm2"][i]),
            "R_residual_power": float(res["R"][i]),
            "sigma_post_pred": float(sigma_post[i]),
            "shrink_pred": float(shrink_pred[i]),
            "min_prior_edge_margin": float(min_margin[i]),
            "kT_ceiling_margin_frac_range": float(kt_ceiling_frac_range[i]),
            "kT_ceiling_margin_frac_value": float(kt_ceiling_frac_value[i]),
            "rank_deficient": bool(res["rank_deficient"][i]),
            "already_run_ns": i in already,
        }
        for k, p in enumerate(PARAM_ORDER):
            row[f"theta_{p}"] = float(theta[i, k])
        rows.append(row)

    finite = sm[np.isfinite(sm)]
    summary = {
        "n": int(args.n_pop),
        "sigma_eff_marginal": {
            "median": float(np.median(finite)),
            "q25": float(np.percentile(finite, 25)),
            "q75": float(np.percentile(finite, 75)),
            "min": float(finite.min()),
            "max": float(finite.max()),
        },
        "gain_prior_sd": float(GAIN_PRIOR_SD),
        "fraction_below_threshold": {
            f"{t:g}": float(np.mean(finite < t))
            for t in (0.02, 0.0289, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20)
        },
        "median_total_counts": float(np.median(totals)),
        "median_shrink_pred": float(np.median(shrink_pred)),
    }

    meta = {
        "script": "scripts/screen_sigma_eff.py",
        "level": args.level,
        "exposure_s": EXPOSURE_S[args.level],
        "response": RESPONSE,
        "base_model": BASE_MODEL,
        "param_order": PARAM_ORDER,
        "n_pop": args.n_pop,
        "strength_pct": args.strength,
        "population_seed": pop_seed,
        "seed_eval_base": args.seed_eval_base,
        "delta_g": args.delta_g,
        "rel_step": args.rel_step,
        "soft_band_kev": args.soft_band_kev,
        "gain_prior": [GAIN_LO, GAIN_HI],
        "gain_prior_sd": float(GAIN_PRIOR_SD),
        "definition": ("sigma_eff_marginal = 1/sqrt(R), R = (1-I)*||b||^2, "
                       "b = W*dmu/dg, W = diag(1/sqrt(mu0)), I = ||P_A b||^2/||b||^2 "
                       "with A = W*dmu/dtheta over the 5 physics parameters; "
                       "identical to Delta_g/sqrt(R_paper) for the symmetric shift"),
        "counts_gate": gate,
        "wall_time_s": round(time.time() - t0, 1),
    }

    js = outdir / f"sigma_eff_{args.level}.json"
    with open(js, "w") as f:
        json.dump({"meta": meta, "summary": summary, "rows": rows}, f, indent=1)
    cs = outdir / f"sigma_eff_{args.level}.csv"
    with open(cs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nsigma_eff (marginal), {args.level}, n={args.n_pop}:")
    s = summary["sigma_eff_marginal"]
    print(f"  median {s['median']:.4f}   IQR [{s['q25']:.4f}, {s['q75']:.4f}]"
          f"   range [{s['min']:.4f}, {s['max']:.4f}]")
    print("  fraction below threshold: " +
          "  ".join(f"{k}:{v:.3f}" for k, v in summary["fraction_below_threshold"].items()))
    print(f"\n  already-run indices ({args.level}):")
    for i in sorted(already):
        if i < args.n_pop:
            print(f"    i{i:<4d} counts {int(round(totals[i])):>6d}  soft {int(round(soft_counts[i])):>6d}"
                  f"  sigma_marg {sm[i]:.4f}  sigma_cond {res['sigma_conditional'][i]:.5f}"
                  f"  shrink_pred {shrink_pred[i]:.4f}")
    print(f"\nwrote {js}\nwrote {cs}\nwall time {time.time() - t0:.1f} s")
    return {"meta": meta, "summary": summary, "rows": rows}


if __name__ == "__main__":
    main()
