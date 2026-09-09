"""Independent re-derivation of the residual power R and the invisibility index I.

R = || b_w - P b_w ||^2, with b_w = W (mu_F - mu_0), W = diag(1/sqrt(mu_0)),
P the orthogonal projector onto the column span of J_w = W dmu_0/dtheta.
I = 1 - R / ||b_w||^2.

Deterministic expected counts throughout (apply_stat=False): no Poisson sampling.

Run:  .venv/Scripts/python.exe outputs/mechanism_rederive_2026-09-08/rederive_R.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # must precede any array creation

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

# ---------------------------------------------------------------- configuration

BASE_MODEL = "tbabs_powerlaw_bb"
PARAM_ORDER = _models.MODEL_PARAMS[BASE_MODEL]
SEED = 20260611
N_DRAWS = 200
FAMILIES = ["B1", "B2", "B3", "B4"]
LEVELS = ["medium", "bright"]

# differencing coordinates: linear for N_H/Gamma/kT, log10 for the two norms
LOG_IDX = (2, 4)
# prior widths in those coordinates (N_H 0.15-0.35, Gamma 1-3, log NormPL 2 dex,
# kT 0.3-3.0, log NormBB 2 dex)
COORD_WIDTH = np.array([0.20, 2.00, 2.00, 2.70, 2.00])
REL_STEP = 1.0e-3

# which family parameter key takes which column of theta
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
FIXED_KEYS = {
    "B1": {"gauss_1_El", "gauss_1_sigma", "gauss_1_norm"},
    "B2": {"tbpcf_1_f"},
    "B3": {"brems_1_kT"},
    "B4": set(),
}

LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


# ---------------------------------------------------------------- helpers

def load_cfgs():
    with open(REPO / "configs" / "sim_modelA_prod.yaml") as fh:
        sim = yaml.safe_load(fh)
    with open(REPO / "configs" / "detect.yaml") as fh:
        det = yaml.safe_load(fh)
    return sim, det


def fold(model, params: dict, oc, chunk: int = 400) -> np.ndarray:
    """Expected counts (no Poisson) for a batch of parameter vectors."""
    n = len(next(iter(params.values())))
    out = []
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        sub = {k: np.asarray(v[sl], dtype=np.float64) for k, v in params.items()}
        out.append(np.asarray(
            fakeit_for_multiple_parameters(oc, model, sub, rng_key=0, apply_stat=False),
            dtype=np.float64,
        ))
    return np.concatenate(out, axis=0)


def theta_to_c(th):
    c = np.array(th, dtype=np.float64, copy=True)
    c[:, list(LOG_IDX)] = np.log10(c[:, list(LOG_IDX)])
    return c


def c_to_theta(c):
    th = np.array(c, dtype=np.float64, copy=True)
    th[:, list(LOG_IDX)] = 10.0 ** th[:, list(LOG_IDX)]
    return th


def jacobian(theta, oc, rel=REL_STEP, log_coords=True):
    """Central-difference J = dmu/dc, shape (n, C, 5)."""
    model = _models.build_model(BASE_MODEL)
    n, p = theta.shape
    if log_coords:
        c0 = theta_to_c(theta)
        width = COORD_WIDTH
    else:
        c0 = np.array(theta, dtype=np.float64, copy=True)
        width = np.array([0.20, 2.00, 1.0e-2 - 1.0e-4, 2.70, 1.0 - 1.0e-2])
    h = rel * width

    blocks = [theta]
    for j in range(p):
        for sgn in (+1.0, -1.0):
            cj = c0.copy()
            cj[:, j] += sgn * h[j]
            blocks.append(c_to_theta(cj) if log_coords else cj)
    big = np.concatenate(blocks, axis=0)
    params = {name: big[:, k] for k, name in enumerate(PARAM_ORDER)}
    mu_all = fold(model, params, oc)
    nch = mu_all.shape[1]
    mu0 = mu_all[:n]
    J = np.empty((n, nch, p))
    for j in range(p):
        mp = mu_all[n + (2 * j) * n: n + (2 * j + 1) * n]
        mm = mu_all[n + (2 * j + 1) * n: n + (2 * j + 2) * n]
        J[:, :, j] = (mp - mm) / (2.0 * h[j])
    return mu0, J


def misspec_mu(family, theta, oc, strength, fixed):
    """mu_F for one family at MY theta, using the package's own assembly."""
    n = theta.shape[0]
    rng = np.random.default_rng(0)  # draws are discarded, overwritten below
    model, params = _misspec.FAMILIES[family](
        BASE_MODEL, PRIOR_CFG, n, rng, float(strength), fixed
    )
    exp_keys = set(THETA_MAP[family]) | FIXED_KEYS[family]
    assert set(params) == exp_keys, (family, sorted(params), sorted(exp_keys))
    for key, col in THETA_MAP[family].items():
        params[key] = theta[:, col].copy()
    oc_use = oc
    if family == "B4":
        oc_use = _responses.gain_shift_obsconf(oc, 1.0 + float(strength) / 100.0)
    return fold(model, params, oc_use)


def residual_power(mu0, delta, J, floor=1e-12):
    """R, ||bw||^2, I per draw. Also returns the Jw condition number."""
    n, nch = mu0.shape
    R = np.empty(n)
    tot = np.empty(n)
    cond = np.empty(n)
    for i in range(n):
        w = 1.0 / np.sqrt(np.maximum(mu0[i], floor))
        bw = w * delta[i]
        Jw = w[:, None] * J[i]
        q, _ = np.linalg.qr(Jw)
        proj = q @ (q.T @ bw)
        r = bw - proj
        R[i] = float(r @ r)
        tot[i] = float(bw @ bw)
        cond[i] = float(np.linalg.cond(Jw))
    I = 1.0 - R / tot
    return R, tot, I, cond


def summarize(a):
    a = np.asarray(a, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "p16": float(np.percentile(a, 16)),
        "p84": float(np.percentile(a, 84)),
        "mean": float(np.mean(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


# ---------------------------------------------------------------- main

t_start = time.time()
SIM_CFG, DET_CFG = load_cfgs()
PRIOR_CFG = SIM_CFG["priors"]
EXPOSURE = {lv["name"]: float(lv["exposure_s"]) for lv in SIM_CFG["levels"]}
STRENGTH = {f: float(DET_CFG["families"][f]["strength_grid"][-1]) for f in FAMILIES}
FIXED = {f: dict(DET_CFG["families"][f].get("fixed") or {}) for f in FAMILIES}

log(f"exposures: {EXPOSURE}")
log(f"strengths (last grid entry): {STRENGTH}")
log(f"fixed: {FIXED}")

base_oc = _responses.load_base_obsconf(SIM_CFG.get("response", _responses.EXAMPLE_NAME))
e_out = np.asarray(base_oc.out_energies)
e_mid = 0.5 * (e_out[0] + e_out[1])
log(f"channels: {e_mid.size}, band {e_out[0][0]:.4f}-{e_out[1][-1]:.4f} keV")

rng = np.random.default_rng(SEED)
theta = np.stack(
    [_priors.sample_prior(PRIOR_CFG, PARAM_ORDER, N_DRAWS, rng)[p] for p in PARAM_ORDER],
    axis=1,
)
log(f"theta {theta.shape} seed {SEED}; medians {np.median(theta, axis=0)}")

results = {
    "meta": {
        "seed": SEED, "n_draws": N_DRAWS, "levels": LEVELS, "families": FAMILIES,
        "strengths": STRENGTH, "fixed": FIXED, "exposures_s": EXPOSURE,
        "n_channels": int(e_mid.size),
        "band_kev": [float(e_out[0][0]), float(e_out[1][-1])],
        "param_order": PARAM_ORDER,
        "diff_coords": "linear N_H/Gamma/kT, log10 NormPL/NormBB",
        "rel_step": REL_STEP, "x64": True,
    },
    "cells": {}, "checks": {},
}

per_level = {}
for level in LEVELS:
    oc = _responses.scale_exposure(base_oc, EXPOSURE[level])
    t0 = time.time()
    mu0, J = jacobian(theta, oc)
    log(f"{level}: mu0 total counts median {np.median(mu0.sum(1)):.1f} "
        f"[{np.percentile(mu0.sum(1),16):.0f}, {np.percentile(mu0.sum(1),84):.0f}]; "
        f"min per-channel mu0 {mu0.min():.3e}; "
        f"channels<0.1 ct median {np.median((mu0<0.1).sum(1)):.0f}; "
        f"J in {time.time()-t0:.1f}s")
    per_level[level] = (oc, mu0, J)

    for fam in FAMILIES:
        t1 = time.time()
        muF = misspec_mu(fam, theta, oc, STRENGTH[fam], FIXED[fam])
        delta = muF - mu0
        R, tot, I, cond = residual_power(mu0, delta, J)
        key = f"{fam}_{level}"
        results["cells"][key] = {
            "family": fam, "level": level, "strength": STRENGTH[fam],
            "R": summarize(R), "I": summarize(I), "bw_sq": summarize(tot),
            "frac_I_gt_0.99": float(np.mean(I > 0.99)),
            "frac_I_gt_0.999": float(np.mean(I > 0.999)),
            "cond_Jw": summarize(cond),
            "R_all": R.tolist(), "I_all": I.tolist(),
        }
        log(f"{key}: median R={np.median(R):.4g}  median I={np.median(I):.6f}  "
            f"median ||bw||^2={np.median(tot):.4g}  fracI>0.99={np.mean(I>0.99):.3f}  "
            f"({time.time()-t1:.1f}s)")

# ---- sanity check (a): sigma_eff for B4 at 3 per cent
for level in LEVELS:
    R = np.asarray(results["cells"][f"B4_{level}"]["R_all"])
    se = 0.03 / np.sqrt(R)
    results["checks"][f"sigma_eff_B4_{level}"] = summarize(se)
    log(f"CHECK a: sigma_eff B4 {level}: median {np.median(se):.4f} "
        f"[{np.percentile(se,16):.4f}, {np.percentile(se,84):.4f}] "
        f"(prior sd 0.0289)")

# ---- sanity check (b): in-span null
rng_b = np.random.default_rng(11)
for level in LEVELS:
    oc, mu0, J = per_level[level]
    step = rng_b.normal(size=(N_DRAWS, 5)) * (1e-3 * COORD_WIDTH)
    delta_null = np.einsum("icj,ij->ic", J, step)
    R, tot, I, _ = residual_power(mu0, delta_null, J)
    ratio = R / tot
    results["checks"][f"null_inspan_{level}"] = {
        "R_over_bwsq": summarize(ratio), "I": summarize(I)}
    log(f"CHECK b: in-span null {level}: max R/||bw||^2 = {ratio.max():.3e}, "
        f"min I = {I.min():.9f}")

# ---- sanity check (c): out-of-span single-channel spike at 6.4 keV
k64 = int(np.argmin(np.abs(e_mid - 6.4)))
log(f"CHECK c: spike channel {k64}, e_mid {e_mid[k64]:.4f} keV")
for level in LEVELS:
    oc, mu0, J = per_level[level]
    delta_sp = np.zeros_like(mu0)
    delta_sp[:, k64] = 0.05 * mu0[:, k64]
    R, tot, I, _ = residual_power(mu0, delta_sp, J)
    results["checks"][f"spike_{level}"] = {
        "channel": k64, "e_mid_kev": float(e_mid[k64]),
        "I": summarize(I), "R": summarize(R)}
    log(f"CHECK c: spike {level}: median I = {np.median(I):.6f} "
        f"[{np.percentile(I,16):.4f}, {np.percentile(I,84):.4f}], "
        f"median R = {np.median(R):.4g}")

# ---- extra check: reparametrization invariance (linear vs log norms)
sub = theta[:40]
oc_b = per_level["bright"][0]
mu0_l, J_lin = jacobian(sub, oc_b, log_coords=False)
mu0_g, J_log = jacobian(sub, oc_b, log_coords=True)
muF_b4 = misspec_mu("B4", sub, oc_b, STRENGTH["B4"], FIXED["B4"])
d = muF_b4 - mu0_l
R_lin, _, _, _ = residual_power(mu0_l, d, J_lin)
R_log, _, _, _ = residual_power(mu0_g, d, J_log)
rel = np.abs(R_lin - R_log) / R_log
results["checks"]["reparam_invariance_B4_bright"] = {
    "max_rel_diff": float(rel.max()), "median_rel_diff": float(np.median(rel)),
    "R_lin_median": float(np.median(R_lin)), "R_log_median": float(np.median(R_log))}
log(f"CHECK extra: reparam invariance (B4 bright, 40 draws): "
    f"max |dR|/R = {rel.max():.3e}")

# ---- extra check: finite-difference step convergence
step_conv = {}
for r in (1e-4, 1e-3, 3e-3):
    mu0_s, J_s = jacobian(sub, oc_b, rel=r)
    row = {}
    for fam in FAMILIES:
        muF = misspec_mu(fam, sub, oc_b, STRENGTH[fam], FIXED[fam])
        R, _, I, _ = residual_power(mu0_s, muF - mu0_s, J_s)
        row[fam] = {"R_median": float(np.median(R)), "I_median": float(np.median(I))}
    step_conv[f"rel_{r:g}"] = row
    log(f"CHECK extra: step rel={r:g} -> " +
        " ".join(f"{f}:R={row[f]['R_median']:.4g}" for f in FAMILIES))
results["checks"]["step_convergence_bright_40draws"] = step_conv

# ---- extra check: drop low-count channels (whitening robustness)
robust = {}
for level in LEVELS:
    oc, mu0, J = per_level[level]
    keep = mu0.min(axis=0) >= 0.1
    row = {"n_channels_kept": int(keep.sum())}
    for fam in FAMILIES:
        muF = misspec_mu(fam, theta, oc, STRENGTH[fam], FIXED[fam])
        R, _, I, _ = residual_power(mu0[:, keep], (muF - mu0)[:, keep], J[:, keep, :])
        row[fam] = {"R_median": float(np.median(R)), "I_median": float(np.median(I))}
    robust[level] = row
    log(f"CHECK extra: {level} drop mu0<0.1ct ({int(keep.sum())}/102 kept) -> " +
        " ".join(f"{f}:R={row[f]['R_median']:.4g},I={row[f]['I_median']:.4f}" for f in FAMILIES))
results["checks"]["low_count_channel_cut"] = robust

results["meta"]["runtime_s"] = time.time() - t_start
with open(HERE / "results.json", "w") as fh:
    json.dump(results, fh, indent=1)
log(f"wrote results.json; total runtime {results['meta']['runtime_s']:.1f}s")
with open(HERE / "run.log", "w") as fh:
    fh.write("\n".join(LOG_LINES) + "\n")
