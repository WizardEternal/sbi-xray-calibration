"""Diagnostics on top of rederive_R.py.

1. Linearized parameter shift implied by the projection, in prior-width units
   (tests whether the LOCAL linear projection is a valid approximation).
2. Nonlinear R: min over theta' in the prior box of ||W (mu_F - mu(theta'))||^2,
   batched Gauss-Newton / Levenberg-Marquardt. Generalizes the linear R.
3. Per-channel breakdown of B1's R at medium (the low-count-channel question).
4. sigma_eff vs total counts for B4 at bright.
5. Paired B3-vs-B4 comparison.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

from jaxspec.data.util import fakeit_for_multiple_parameters  # noqa: E402

from sbixcal import misspec as _misspec  # noqa: E402
from sbixcal import models as _models  # noqa: E402
from sbixcal import priors as _priors  # noqa: E402
from sbixcal import responses as _responses  # noqa: E402

BASE_MODEL = "tbabs_powerlaw_bb"
PARAM_ORDER = _models.MODEL_PARAMS[BASE_MODEL]
SEED, N_DRAWS = 20260611, 200
FAMILIES = ["B1", "B2", "B3", "B4"]
LEVELS = ["medium", "bright"]
LOG_IDX = [2, 4]
COORD_WIDTH = np.array([0.20, 2.00, 2.00, 2.70, 2.00])
COORD_LO = np.array([0.15, 1.0, -4.0, 0.3, -2.0])
COORD_HI = np.array([0.35, 3.0, -2.0, 3.0, 0.0])
REL_STEP = 1e-3

THETA_MAP = {
    "B1": {"tbabs_1_nh": 0, "powerlaw_1_alpha": 1, "powerlaw_1_norm": 2,
           "blackbodyrad_1_kT": 3, "blackbodyrad_1_norm": 4},
    "B2": {"tbpcf_1_nh": 0, "powerlaw_1_alpha": 1, "powerlaw_1_norm": 2,
           "blackbodyrad_1_kT": 3, "blackbodyrad_1_norm": 4},
    "B3": {"tbabs_1_nh": 0, "brems_1_norm": 2,
           "blackbodyrad_1_kT": 3, "blackbodyrad_1_norm": 4},
    "B4": {"tbabs_1_nh": 0, "powerlaw_1_alpha": 1, "powerlaw_1_norm": 2,
           "blackbodyrad_1_kT": 3, "blackbodyrad_1_norm": 4},
}
LOGL = []


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    LOGL.append(s)


def fold(model, params, oc, chunk=400):
    n = len(next(iter(params.values())))
    out = []
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        sub = {k: np.asarray(v[sl], dtype=np.float64) for k, v in params.items()}
        out.append(np.asarray(fakeit_for_multiple_parameters(
            oc, model, sub, rng_key=0, apply_stat=False), dtype=np.float64))
    return np.concatenate(out, axis=0)


def c_to_theta(c):
    th = np.array(c, dtype=np.float64, copy=True)
    th[:, LOG_IDX] = 10.0 ** th[:, LOG_IDX]
    return th


def theta_to_c(th):
    c = np.array(th, dtype=np.float64, copy=True)
    c[:, LOG_IDX] = np.log10(c[:, LOG_IDX])
    return c


MODEL_A = _models.build_model(BASE_MODEL)


def mu_at_c(c, oc):
    th = c_to_theta(c)
    return fold(MODEL_A, {p: th[:, k] for k, p in enumerate(PARAM_ORDER)}, oc)


def jac_at_c(c, oc, rel=REL_STEP):
    n, p = c.shape
    h = rel * COORD_WIDTH
    blocks = [c]
    for j in range(p):
        for sgn in (1.0, -1.0):
            cj = c.copy()
            cj[:, j] += sgn * h[j]
            blocks.append(cj)
    big = np.concatenate(blocks, axis=0)
    mu = mu_at_c(big, oc)
    mu0 = mu[:n]
    J = np.empty((n, mu.shape[1], p))
    for j in range(p):
        J[:, :, j] = (mu[n + 2 * j * n: n + (2 * j + 1) * n]
                      - mu[n + (2 * j + 1) * n: n + (2 * j + 2) * n]) / (2 * h[j])
    return mu0, J


def misspec_mu(family, theta, oc, strength, fixed):
    n = theta.shape[0]
    model, params = _misspec.FAMILIES[family](
        BASE_MODEL, PRIOR_CFG, n, np.random.default_rng(0), float(strength), fixed)
    for key, col in THETA_MAP[family].items():
        params[key] = theta[:, col].copy()
    oc_use = _responses.gain_shift_obsconf(oc, 1 + float(strength) / 100.0) \
        if family == "B4" else oc
    return fold(model, params, oc_use)


def linear_fit(mu0, delta, J):
    """Returns R, ||bw||^2, and the linearized coefficient vector per draw."""
    n = mu0.shape[0]
    R = np.empty(n); tot = np.empty(n); coef = np.empty((n, J.shape[2]))
    for i in range(n):
        w = 1.0 / np.sqrt(mu0[i])
        bw = w * delta[i]
        Jw = w[:, None] * J[i]
        c, *_ = np.linalg.lstsq(Jw, bw, rcond=None)
        r = bw - Jw @ c
        R[i] = r @ r; tot[i] = bw @ bw; coef[i] = c
    return R, tot, coef


def nonlinear_R(theta, muF, mu0, oc, n_iter=30, box=True):
    """min_{theta'} ||W (mu_F - mu(theta'))||^2 with W fixed at the true theta."""
    n = theta.shape[0]
    w = 1.0 / np.sqrt(mu0)
    c = theta_to_c(theta).copy()
    r = w * (muF - mu_at_c(c, oc))
    cost = np.einsum("ij,ij->i", r, r)
    lam = np.full(n, 1e-3)
    for _ in range(n_iter):
        mu_c, J = jac_at_c(c, oc)
        r = w * (muF - mu_c)
        cost = np.einsum("ij,ij->i", r, r)
        c_new = c.copy()
        for i in range(n):
            Jw = w[i][:, None] * J[i]
            A = Jw.T @ Jw
            g = Jw.T @ r[i]
            step = np.linalg.solve(A + lam[i] * np.diag(np.diag(A) + 1e-30), g)
            c_new[i] = c[i] + step
        if box:
            c_new = np.clip(c_new, COORD_LO, COORD_HI)
        r_new = w * (muF - mu_at_c(c_new, oc))
        cost_new = np.einsum("ij,ij->i", r_new, r_new)
        acc = cost_new < cost
        c[acc] = c_new[acc]
        lam[acc] /= 3.0
        lam[~acc] *= 3.0
        lam = np.clip(lam, 1e-9, 1e9)
    r = w * (muF - mu_at_c(c, oc))
    return np.einsum("ij,ij->i", r, r), c


def summ(a):
    a = np.asarray(a, float)
    return {"median": float(np.median(a)), "p16": float(np.percentile(a, 16)),
            "p84": float(np.percentile(a, 84)), "min": float(np.min(a)),
            "max": float(np.max(a))}


# ---------------------------------------------------------------- setup
t0 = time.time()
with open(REPO / "configs" / "sim_modelA_prod.yaml") as fh:
    SIM = yaml.safe_load(fh)
with open(REPO / "configs" / "detect.yaml") as fh:
    DET = yaml.safe_load(fh)
PRIOR_CFG = SIM["priors"]
EXPO = {l["name"]: float(l["exposure_s"]) for l in SIM["levels"]}
STR = {f: float(DET["families"][f]["strength_grid"][-1]) for f in FAMILIES}
FIX = {f: dict(DET["families"][f].get("fixed") or {}) for f in FAMILIES}

base_oc = _responses.load_base_obsconf(SIM.get("response", "NGC7793_ULX4_PN"))
e_out = np.asarray(base_oc.out_energies)
e_mid = 0.5 * (e_out[0] + e_out[1])
rng = np.random.default_rng(SEED)
theta = np.stack([_priors.sample_prior(PRIOR_CFG, PARAM_ORDER, N_DRAWS, rng)[p]
                  for p in PARAM_ORDER], axis=1)

out = {"linear_shift": {}, "nonlinear_R": {}, "b1_channels": {}, "sigma_eff": {},
       "b3_vs_b4": {}}

for level in LEVELS:
    oc = _responses.scale_exposure(base_oc, EXPO[level])
    mu0, J = jac_at_c(theta_to_c(theta), oc)
    for fam in FAMILIES:
        muF = misspec_mu(fam, theta, oc, STR[fam], FIX[fam])
        delta = muF - mu0
        R, tot, coef = linear_fit(mu0, delta, J)
        # implied shift in prior-width units
        shift = np.abs(coef) / COORD_WIDTH
        out["linear_shift"][f"{fam}_{level}"] = {
            "max_over_params_median": float(np.median(shift.max(axis=1))),
            "max_over_params_p84": float(np.percentile(shift.max(axis=1), 84)),
            "per_param_median": [float(x) for x in np.median(shift, axis=0)],
            "frac_draws_shift_gt_0.5_priorwidth": float(np.mean(shift.max(axis=1) > 0.5)),
        }
        log(f"{fam}_{level} linear shift |dc|/priorwidth: median max {np.median(shift.max(1)):.3f}, "
            f"per-param medians {np.round(np.median(shift,0),3)}")

        t1 = time.time()
        Rnl, c_end = nonlinear_R(theta, muF, mu0, oc)
        at_bound = np.mean(np.any((c_end <= COORD_LO + 1e-9) | (c_end >= COORD_HI - 1e-9), axis=1))
        out["nonlinear_R"][f"{fam}_{level}"] = {
            "R_linear": summ(R), "R_nonlinear": summ(Rnl),
            "I_nonlinear": summ(1 - Rnl / tot),
            "ratio_nl_over_lin_median": float(np.median(Rnl / R)),
            "frac_at_prior_bound": float(at_bound),
            "R_nl_all": Rnl.tolist(),
        }
        log(f"{fam}_{level} R_lin median {np.median(R):.4g} -> R_nonlin median "
            f"{np.median(Rnl):.4g} (x{np.median(Rnl/R):.3g}), I_nl median "
            f"{np.median(1-Rnl/tot):.6f}, at-bound {at_bound:.2f} ({time.time()-t1:.0f}s)")

    # per-channel breakdown for B1 and B4
    for fam in ("B1", "B4"):
        muF = misspec_mu(fam, theta, oc, STR[fam], FIX[fam])
        delta = muF - mu0
        n = mu0.shape[0]
        contrib = np.zeros((n, mu0.shape[1]))
        for i in range(n):
            w = 1.0 / np.sqrt(mu0[i])
            bw = w * delta[i]
            Jw = w[:, None] * J[i]
            q, _ = np.linalg.qr(Jw)
            r = bw - q @ (q.T @ bw)
            contrib[i] = r ** 2
        frac = contrib / contrib.sum(1, keepdims=True)
        top = np.argsort(-np.median(frac, axis=0))[:6]
        out["b1_channels"][f"{fam}_{level}"] = {
            "top_channels": [int(k) for k in top],
            "top_e_mid_kev": [float(e_mid[k]) for k in top],
            "top_median_frac_of_R": [float(np.median(frac[:, k])) for k in top],
            "top_median_mu0": [float(np.median(mu0[:, k])) for k in top],
            "top_min_mu0": [float(mu0[:, k].min()) for k in top],
        }
        log(f"{fam}_{level} top-6 channels for R: E={np.round(e_mid[top],2)} keV, "
            f"fracR={np.round([np.median(frac[:,k]) for k in top],3)}, "
            f"median mu0={np.round([np.median(mu0[:,k]) for k in top],2)}")

    if level == "bright":
        muF = misspec_mu("B4", theta, oc, STR["B4"], FIX["B4"])
        R, tot, _ = linear_fit(mu0, muF - mu0, J)
        se = 0.03 / np.sqrt(R)
        cts = mu0.sum(1)
        out["sigma_eff"]["bright"] = {
            "summary": summ(se), "total_counts": summ(cts),
            "n_below_0.06": int((se < 0.06).sum()),
            "n_below_prior_sd_0.0289": int((se < 0.0289).sum()),
            "spearman_se_vs_counts": float(np.corrcoef(
                np.argsort(np.argsort(se)), np.argsort(np.argsort(cts)))[0, 1]),
            "se_at_54509cts_scaled_median": float(np.median(se) * np.sqrt(np.median(cts) / 54509.0)),
        }
        log(f"sigma_eff bright: {summ(se)}; n<0.06 = {(se<0.06).sum()}/200; "
            f"median rescaled to 54509 counts = {np.median(se)*np.sqrt(np.median(cts)/54509.):.4f}")

# B3 vs B4 paired
with open(HERE / "results.json") as fh:
    prev = json.load(fh)
for level in LEVELS:
    r3 = np.array(prev["cells"][f"B3_{level}"]["R_all"])
    r4 = np.array(prev["cells"][f"B4_{level}"]["R_all"])
    out["b3_vs_b4"][level] = {
        "frac_R_B4_gt_R_B3": float(np.mean(r4 > r3)),
        "median_ratio_B4_over_B3": float(np.median(r4 / r3)),
        "R_B3_median": float(np.median(r3)), "R_B4_median": float(np.median(r4))}
    log(f"B3 vs B4 {level}: frac(R_B4>R_B3)={np.mean(r4>r3):.3f}, "
        f"median ratio {np.median(r4/r3):.3f}")

out["runtime_s"] = time.time() - t0
with open(HERE / "diagnostics.json", "w") as fh:
    json.dump(out, fh, indent=1)
with open(HERE / "diagnose.log", "w") as fh:
    fh.write("\n".join(LOGL) + "\n")
log(f"done in {out['runtime_s']:.0f}s")
