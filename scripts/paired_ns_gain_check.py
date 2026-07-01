"""Paired nested-sampling test of the B4 gain-shift evidence.

A level-matched mean(logZ_mis) - mean(logZ_clean) over unmatched spectra is
confounded by total counts. This removes the confound by construction: the same
parameter draw is folded through the clean response and the 3%-gain-shifted
response, Poisson-realized with a MATCHED seed (so total counts are ~identical),
and the clean (well-specified) model is fit to each by UltraNest. Paired
Delta logZ = logZ(gain-shifted) - logZ(clean); ~0 means no evidence penalty from
the gain shift.

Medium count regime. CPU-only (set by caller env). Resumable: appends one row per
spectrum to outputs/ns_bench/paired_gain_check.jsonl.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

from sbixcal import responses as R, simulate as S, models as M, priors as P
from sbixcal import ns_bench as NB

BASE_MODEL = "tbabs_powerlaw_bb"
RESP = "NGC7793_ULX4_PN"
EXPOSURE = 353.4            # medium (~1000 counts), matches sim_modelA_prod
GAIN = 1.03                 # 3% gain shift (the headline B4 strength)
N = 12                      # matched-count pairs (deterministic from the seeds below)
PRIOR_CFG = {
    "tbabs_1_nh":         {"dist": "uniform",    "low": 0.15,   "high": 0.35},
    "powerlaw_1_alpha":   {"dist": "uniform",    "low": 1.0,    "high": 3.0},
    "powerlaw_1_norm":    {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":  {"dist": "uniform",    "low": 0.3,    "high": 3.0},
    "blackbodyrad_1_norm":{"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}
OUT = Path("outputs/ns_bench/paired_gain_check.jsonl")


def main():
    param_order = M.MODEL_PARAMS[BASE_MODEL]
    base = R.load_base_obsconf(RESP)
    clean_oc = R.scale_exposure(base, EXPOSURE)
    gain_oc = R.gain_shift_obsconf(clean_oc, GAIN)

    # well-specified model: fold through the NOMINAL (clean) response
    def model_counts_fn(theta_arr):
        return S.fold_theta(BASE_MODEL, param_order, theta_arr, clean_oc)

    rng = np.random.default_rng(20260630)
    samples = P.sample_prior(PRIOR_CFG, param_order, N, rng)
    theta = np.stack([np.asarray(samples[p]) for p in param_order], axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for l in open(OUT):
            if l.strip():
                done.add(json.loads(l)["i"])

    for i in range(N):
        if i in done:
            print(f"[skip] spectrum {i} already done")
            continue
        th = theta[i:i + 1]
        lam_clean = np.asarray(S.fold_theta(BASE_MODEL, param_order, th, clean_oc))[0]
        lam_gain = np.asarray(S.fold_theta(BASE_MODEL, param_order, th, gain_oc))[0]
        # matched-seed Poisson so the two realizations are paired
        data_clean = np.random.default_rng(1000 + i).poisson(np.maximum(lam_clean, 0)).astype(float)
        data_gain = np.random.default_rng(1000 + i).poisson(np.maximum(lam_gain, 0)).astype(float)
        t0 = time.time()
        rc = NB.run_ns_one(data_clean, model_counts_fn, PRIOR_CFG, param_order,
                           min_num_live_points=400, dlogz=0.5, max_ncalls=400000, seed=i)
        rg = NB.run_ns_one(data_gain, model_counts_fn, PRIOR_CFG, param_order,
                           min_num_live_points=400, dlogz=0.5, max_ncalls=400000, seed=i)
        row = {
            "i": i,
            "counts_clean": int(data_clean.sum()),
            "counts_gain": int(data_gain.sum()),
            "logz_clean": rc.logz, "logzerr_clean": rc.logzerr,
            "logz_gain": rg.logz, "logzerr_gain": rg.logzerr,
            "d_paired": rg.logz - rc.logz,
            "wall_s": round(time.time() - t0, 1),
        }
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[done] i={i} counts {row['counts_clean']}/{row['counts_gain']} "
              f"logZ_clean {rc.logz:.1f} logZ_gain {rg.logz:.1f} "
              f"d_paired {row['d_paired']:+.2f} (+/- {(rc.logzerr**2+rg.logzerr**2)**0.5:.2f}) "
              f"[{row['wall_s']}s]", flush=True)

    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    d = np.array([r["d_paired"] for r in rows])
    print("\n=== PAIRED RESULT (clean model fit to gain-shifted vs clean data, same theta) ===")
    print(f"n={len(d)}  mean paired Delta logZ = {d.mean():+.2f} +/- {d.std(ddof=1)/len(d)**0.5:.2f} (SEM)")
    print(f"per-spectrum: {[round(x,2) for x in d]}")
    print("paired Delta logZ ~ 0 => no evidence penalty from the 3% gain shift.")


if __name__ == "__main__":
    main()
