"""Check A helper: draw pair 4 and pair 0 clean/gain spectra via the exact
committed path in scripts/paired_ns_gain_check.py (no NS run), and compare
per-channel.

Read-only. Does not modify any file, does not run NS, does not touch git.
Run from the repo root with the repo .venv:

    .venv/Scripts/python.exe outputs/ns_bench/revision_2026-09-02/pair4_check.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from sbixcal import responses as R, simulate as S, models as M, priors as P

# --- exact constants from scripts/paired_ns_gain_check.py ---
BASE_MODEL = "tbabs_powerlaw_bb"
RESP = "NGC7793_ULX4_PN"
EXPOSURE = 353.4
GAIN = 1.03
N = 12
THETA_SEED = 20260630
POISSON_SEED_BASE = 1000
PRIOR_CFG = {
    "tbabs_1_nh":         {"dist": "uniform",    "low": 0.15,   "high": 0.35},
    "powerlaw_1_alpha":   {"dist": "uniform",    "low": 1.0,    "high": 3.0},
    "powerlaw_1_norm":    {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":  {"dist": "uniform",    "low": 0.3,    "high": 3.0},
    "blackbodyrad_1_norm":{"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}

param_order = M.MODEL_PARAMS[BASE_MODEL]
base = R.load_base_obsconf(RESP)
clean_oc = R.scale_exposure(base, EXPOSURE)
gain_oc = R.gain_shift_obsconf(clean_oc, GAIN)

rng = np.random.default_rng(THETA_SEED)
samples = P.sample_prior(PRIOR_CFG, param_order, N, rng)
theta = np.stack([np.asarray(samples[p]) for p in param_order], axis=1)


def draw_pair(i: int):
    th = theta[i:i + 1]
    lam_clean = np.asarray(S.fold_theta(BASE_MODEL, param_order, th, clean_oc))[0]
    lam_gain = np.asarray(S.fold_theta(BASE_MODEL, param_order, th, gain_oc))[0]
    data_clean = np.random.default_rng(POISSON_SEED_BASE + i).poisson(np.maximum(lam_clean, 0)).astype(float)
    data_gain = np.random.default_rng(POISSON_SEED_BASE + i).poisson(np.maximum(lam_gain, 0)).astype(float)
    return lam_clean, lam_gain, data_clean, data_gain


def report(i: int):
    lam_clean, lam_gain, data_clean, data_gain = draw_pair(i)
    n_channels = data_clean.shape[0]
    diff = data_gain - data_clean
    n_diff = int(np.count_nonzero(diff))
    max_abs = float(np.max(np.abs(diff))) if n_channels else 0.0
    sum_abs = float(np.sum(np.abs(diff)))
    lam_diff = lam_gain - lam_clean
    lam_max_abs = float(np.max(np.abs(lam_diff))) if n_channels else 0.0
    lam_sum_abs = float(np.sum(np.abs(lam_diff)))
    print(f"=== pair {i} ===")
    print(f"  n_channels          = {n_channels}")
    print(f"  counts_clean total  = {int(data_clean.sum())}")
    print(f"  counts_gain  total  = {int(data_gain.sum())}")
    print(f"  n channels differing (data) = {n_diff}")
    print(f"  max |data_gain - data_clean| (per channel) = {max_abs}")
    print(f"  sum |data_gain - data_clean| over channels = {sum_abs}")
    print(f"  lambda_clean sum = {lam_clean.sum():.6f}  lambda_gain sum = {lam_gain.sum():.6f}")
    print(f"  max |lambda_gain - lambda_clean| (per channel, model mean) = {lam_max_abs:.6f}")
    print(f"  sum |lambda_gain - lambda_clean| over channels (model mean) = {lam_sum_abs:.6f}")
    print(f"  max lambda_clean (peak channel rate) = {float(lam_clean.max()):.6f}")
    return dict(i=i, n_channels=n_channels, counts_clean=int(data_clean.sum()),
                counts_gain=int(data_gain.sum()), n_diff=n_diff, max_abs=max_abs,
                sum_abs=sum_abs, lam_max_abs=lam_max_abs, lam_sum_abs=lam_sum_abs,
                data_clean=data_clean, data_gain=data_gain)


if __name__ == "__main__":
    print(f"[cfg] exposure={EXPOSURE} gain={GAIN} n={N} theta_seed={THETA_SEED} "
          f"poisson_seed_base={POISSON_SEED_BASE} resp={RESP} model={BASE_MODEL}")
    r4 = report(4)
    print()
    r0 = report(0)
