r"""IS-reweighting ESS / PSIS-khat sweep: does the Barret & Dupourque /
Dingo-IS importance-sampling efficiency (and PSIS k-hat) flag a detector GAIN
shift (B4) the way it flags an unmodeled emission LINE (B1)?

Recipe (Barret & Dupourque Paper III, arXiv:2512.16709; Dingo-IS, Dax+23):
  1. draw theta_j ~ q_NPE(.|x)  (the trained flow),
  2. weight w_j = p_Poisson(x|theta_j) * pi_box(theta_j) / q_NPE(theta_j|x),
  3. sampling efficiency  eff = ESS/N = (sum w)^2 / (N * sum w^2),
  4. PSIS k-hat: generalized-Pareto fit to the upper tail of w (Vehtari+2024).

Prior leakage: the flow leaks mass outside the box prior
on misspecified spectra (~1% acceptance on B4), so reject-sampling to a fixed n
stalls. We instead draw a FIXED BUDGET of RAW flow samples (reject_outside_prior
=False), keep the in-prior ones (weight 0 outside via pi=-inf), self-normalize,
and report the in-prior acceptance rate. k-hat flags unreliable tails; we never
force n by rejection.

Prior convention: the flow's BoxUniform prior and the NS benchmark both use a
UNIFORM box in LINEAR theta (train_npe.build_prior / ns_bench.make_box_transform),
so the IS evidence estimate is directly comparable to the committed nested-sampling
log Z on the SAME clean spectra.

Everything is seeded. Writes results JSON + weight/ESS figures under
outputs/is_reweight/. Reproducible from this committed script:

    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe outputs\is_reweight\run_is_ess_sweep.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

# repo root = two parents up from outputs/is_reweight/
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "is_reweight"
OUT.mkdir(parents=True, exist_ok=True)

from sbixcal import train_npe as _tn
from sbixcal import simulate as _sim
from sbixcal import misspec as _MS
from sbixcal import responses as _resp
from sbixcal import priors as _pr
from sbixcal.calibrate import poisson_loglik
from sbixcal.detect import roc_auc

RESPONSE = "NGC7793_ULX4_PN"          # XMM EPIC-pn (the prod flows' response)
TRAIN_RUN = "train_npe_prod"
GLOBAL_SEED = 20260611


# ==========================================================================
# PSIS k-hat  (Vehtari, Simpson, Gelman, Yao, Gabry 2024; Zhang & Stephens 2009 GPD fit)
# ==========================================================================

def _gpdfit(ary: np.ndarray):
    """Empirical-Bayes generalized-Pareto shape/scale fit (Zhang & Stephens 2009),
    the exact estimator ArviZ ``stats.stats._gpdfit`` uses. ``ary`` is the sorted
    (ascending), strictly-positive tail exceedances. Returns (k, sigma)."""
    prior_bs = 3.0
    prior_k = 10.0
    n = len(ary)
    m_est = 30 + int(n ** 0.5)

    b_ary = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1.0 / ary[-1]

    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1.0)
    weights = 1.0 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    weights /= weights.sum()

    b_post = np.sum(b_ary * weights)
    k_post = np.log1p(-b_post * ary).mean()
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return float(k_post), float(sigma)


def psis_khat(log_w_inprior: np.ndarray):
    """PSIS k-hat from the (finite) in-prior log importance weights (ArviZ psislw
    tail procedure). Returns k_hat (np.nan if too few weights). k_hat > 0.7 => the
    IS weights have a heavy tail: ESS and any evidence/expectation estimate from
    them are unreliable (Vehtari, Simpson, Gelman, Yao, Gabry 2024)."""
    lw = np.asarray(log_w_inprior, dtype=np.float64)
    lw = lw[np.isfinite(lw)]
    S = lw.size
    if S < 30:
        return float("nan")
    lw = lw - lw.max()                       # stability: largest log weight -> 0
    lw_sorted = np.sort(lw)
    M = int(np.ceil(min(0.2 * S, 3.0 * np.sqrt(S))))
    if M < 5:
        return float("nan")
    log_cutoff = lw_sorted[S - M - 1]        # threshold just below the tail
    tail = lw_sorted[S - M:]                  # M largest log weights, ascending
    exceed = np.exp(tail) - np.exp(log_cutoff)
    exceed = exceed[exceed > 0]
    if exceed.size < 5:
        return float("nan")
    k, _ = _gpdfit(np.sort(exceed))
    return k


# ==========================================================================
# per-level state
# ==========================================================================

class Level:
    def __init__(self, level: str):
        ck = ROOT / "outputs" / "models" / f"{TRAIN_RUN}_{level}"
        self.post, _ = _tn.load_posterior(ck, device="cpu")
        arch = json.load(open(ck / "arch.json"))
        self.pcfg = arch["prior_cfg"]
        self.base = arch["base_model"]
        self.names = list(arch["param_names"])
        self.exp = float(arch["exposure_s"])
        self.median_counts = float(arch.get("median_total_counts", np.nan))
        lo, hi = _pr.prior_bounds(self.pcfg, self.names)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self.log_vol = float(np.log(np.prod(self.hi - self.lo)))
        self.obs = _resp.scale_exposure(_resp.load_base_obsconf(RESPONSE), self.exp)

    def fold(self, theta):
        return _sim.fold_theta(self.base, self.names, theta, self.obs)


# ==========================================================================
# raw-budget IS reweighting (the landmine-safe core)
# ==========================================================================

def is_reweight(level: Level, x_obs: np.ndarray, n_budget: int, seed: int):
    """Draw n_budget RAW flow samples (no rejection), self-normalized IS weights
    against the exact Poisson likelihood + uniform box prior. Returns a dict."""
    torch.manual_seed(int(seed))
    x_t = torch.as_tensor(np.asarray(x_obs, dtype=np.float32))
    with torch.no_grad():
        s_t = level.post.sample((n_budget,), x=x_t, show_progress_bars=False,
                                reject_outside_prior=False)
        log_q = level.post.log_prob(s_t, x=x_t, norm_posterior=False).cpu().numpy().astype(np.float64)
    s = s_t.cpu().numpy().astype(np.float64)

    inside = np.all((s >= level.lo) & (s <= level.hi), axis=1)
    n_in = int(inside.sum())
    log_w = np.full(n_budget, -np.inf)
    if n_in > 0:
        ll = poisson_loglik(x_obs, level.fold(s[inside]))
        # log_w = loglike + log_prior_uniform - log_q ; avoid (-inf)-(-inf) NaN
        log_w[inside] = ll - level.log_vol - log_q[inside]

    finite = np.isfinite(log_w)
    if finite.any():
        m = np.max(log_w[finite])
        w = np.where(finite, np.exp(log_w - m), 0.0)
        sw = w.sum()
        W = w / sw if sw > 0 else np.full(n_budget, 1.0 / n_budget)
        ess = 1.0 / np.sum(W ** 2)
        logZ = logsumexp(log_w[finite]) - np.log(n_budget)
    else:
        W = np.full(n_budget, 1.0 / n_budget)
        ess = float(n_budget)
        logZ = float("nan")

    khat = psis_khat(log_w[inside]) if n_in > 0 else float("nan")
    return {
        "ess": float(ess),
        "ess_frac": float(ess / n_budget),
        "acc": float(n_in / n_budget),
        "logZ": float(logZ),
        "khat": float(khat),
        "log_w_inprior": log_w[inside],
        "weights": W,
    }


# ==========================================================================
# population generators (deterministic seeds)
# ==========================================================================

def gen_clean(level: Level, n: int, seed: int):
    rng = np.random.default_rng(seed)
    th, xexp, _ = _sim.simulate_spectra(level.base, level.pcfg, level.obs, n, rng,
                                         apply_poisson=False, seed_for_fakeit=seed)
    x = np.random.default_rng(seed + 1).poisson(np.clip(xexp, 0.0, None)).astype(np.float64)
    return x, np.asarray(th, dtype=np.float64)


def gen_misspec(level: Level, family: str, strength: float, n: int, seed: int, fixed=None):
    x, th, pres = _MS.simulate_misspec_population(level.base, level.pcfg, level.obs,
                                                  family, strength, n, seed=seed,
                                                  fixed=fixed or {})
    return np.asarray(x, dtype=np.float64), np.asarray(th, dtype=np.float64)


# ==========================================================================
# NS clean spectra reproduction (for the logZ control)
# ==========================================================================

def reproduce_ns_clean(level: Level, level_name: str, block_idx: int, n: int):
    """Regenerate the EXACT clean spectra the committed NS benchmark ran on (same
    block seed as scripts/run_ns_benchmark.draw_block), so log Z_IS can be compared
    to the stored nested-sampling log Z per spectrum_id."""
    bseed = (GLOBAL_SEED + 1000 * (block_idx + 1)) % (2 ** 31 - 1)
    rng = np.random.default_rng(bseed)
    th, xexp, _ = _sim.simulate_spectra(level.base, level.pcfg, level.obs, n, rng,
                                        apply_poisson=False, seed_for_fakeit=bseed)
    x = np.random.default_rng(bseed + 1).poisson(np.clip(xexp, 0.0, None)).astype(np.float64)
    ids = [f"clean|{level_name}|clean|{block_idx}|{i}" for i in range(n)]
    return x, ids


def prior_mc_logz(level: "Level", x_obs: np.ndarray, n_mc: int, seed: int):
    """Assumption-free brute-force evidence: log Z = log E_{theta~Uniform(box)}[L(theta)]
    = logmeanexp of the Poisson log-likelihood over uniform-prior draws. Needs no
    flow; the ground truth the IS estimate must reproduce."""
    rng = np.random.default_rng(seed)
    lls = []
    done = 0
    while done < n_mc:
        nb = min(200000, n_mc - done)
        th = rng.uniform(level.lo, level.hi, size=(nb, len(level.names)))
        lls.append(poisson_loglik(x_obs, level.fold(th)))
        done += nb
    ll = np.concatenate(lls)
    return float(logsumexp(ll) - np.log(ll.size))


def load_ns_clean_logz():
    rows = [json.loads(l) for l in open(ROOT / "outputs/ns_bench/results.jsonl") if l.strip()]
    out = {}
    for r in rows:
        if r.get("family") == "clean":
            out[r["spectrum_id"]] = (float(r["ns"]["logz"]), float(r["ns"].get("logzerr", np.nan)))
    return out


# ==========================================================================
# figures
# ==========================================================================

def fig_weights(examples: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(examples.keys())
    fig, axes = plt.subplots(2, len(fams), figsize=(4.2 * len(fams), 7), squeeze=False)
    for j, fam in enumerate(fams):
        lw = examples[fam]["log_w_inprior"]
        lw = lw[np.isfinite(lw)]
        W = examples[fam]["weights"]
        W = W[W > 0]
        ax = axes[0, j]
        ax.hist(lw - np.max(lw), bins=50, color="C0", alpha=0.8)
        ax.set_title(f"{fam}\nlog w (shifted), eff={examples[fam]['ess_frac']:.3f} "
                     f"khat={examples[fam]['khat']:.2f}", fontsize=9)
        ax.set_xlabel("log w - max log w"); ax.set_ylabel("count")
        ax2 = axes[1, j]
        Ws = np.sort(W)[::-1]
        ax2.plot(np.arange(1, Ws.size + 1), np.cumsum(Ws), "C3-")
        ax2.set_xscale("log")
        ax2.set_xlabel("# top weights"); ax2.set_ylabel("cumulative norm. weight")
        ax2.set_title(f"acc(in-prior)={examples[fam]['acc']:.2f}", fontsize=9)
        ax2.grid(alpha=0.3)
    fig.suptitle("IS-weight distributions (one representative spectrum per family)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_ess(cells: dict, level_name: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    fams = list(cells.keys())
    colors = {"clean": "C7", "B1": "C3", "B2": "C2", "B3": "C4", "B4": "C0"}
    for fam in fams:
        c = colors.get(fam[:2], "C1")
        ef = np.asarray(cells[fam]["ess_frac"])
        axL.hist(ef, bins=np.linspace(0, max(0.05, np.nanpercentile(ef, 98)), 40),
                 histtype="step", lw=2, label=f"{fam} (med {np.nanmedian(ef):.4f})", color=c)
        kh = np.asarray(cells[fam]["khat"])
        kh = kh[np.isfinite(kh)]
        axR.hist(kh, bins=40, histtype="step", lw=2,
                 label=f"{fam} (med {np.nanmedian(kh):.2f})", color=c)
    axL.set_xlabel("ESS efficiency (n_eff / N_budget)"); axL.set_ylabel("count")
    axL.set_title(f"{level_name}: ESS-efficiency per spectrum"); axL.legend(fontsize=8)
    axL.grid(alpha=0.3)
    axR.axvline(0.7, color="k", ls="--", lw=1, label="k-hat=0.7")
    axR.set_xlabel("PSIS k-hat"); axR.set_ylabel("count")
    axR.set_title(f"{level_name}: PSIS k-hat per spectrum"); axR.legend(fontsize=8)
    axR.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ==========================================================================
# main
# ==========================================================================

def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    t_start = time.time()

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["faint", "medium", "bright"], default=None,
                     help="Run only this level as its own process (crash-isolation across "
                          "levels); merges into the existing results JSON instead of "
                          "overwriting other levels.")
    args = ap.parse_args()

    N_BUDGET = 6000
    N_CLEAN = 150        # shared negative class per level
    N_MIS = 100          # per misspec family per level
    ALL_LEVELS = {"faint": 0, "medium": 1, "bright": 2}   # name -> NS clean block_idx
    LEVELS = {args.level: ALL_LEVELS[args.level]} if args.level else ALL_LEVELS
    # misspec families: (family, strength, fixed); grids match configs/detect.yaml
    MIS = [
        ("B1", 3.0e-4, {"line_energy_kev": 6.4, "line_sigma_kev": 0.05}),   # unmodeled Fe-K line
        ("B4", 3.0, {}),                                                    # 3% gain shift
        ("B2", 0.5, {}),                                                    # partial covering f=0.5
        ("B3", 3.0, {"use_diskbb": False}),                                 # continuum swap kT=3
    ]

    results_path = OUT / "is_ess_sweep_results.json"
    if args.level is not None and results_path.exists():
        # merge mode: keep whatever other levels already saved, only (re)write this one
        with open(results_path) as f:
            results = json.load(f)
        results.setdefault("clean_control", {})
        results.setdefault("sweep", {})
    else:
        results = {"config": {"n_budget": N_BUDGET, "n_clean": N_CLEAN, "n_mis": N_MIS,
                              "response": RESPONSE, "train_run": TRAIN_RUN, "seed": GLOBAL_SEED,
                              "misspec_grid": [(f, s) for f, s, _ in MIS]},
                   "clean_control": {}, "sweep": {}}

    for lname, block_idx in LEVELS.items():
        print(f"\n===== LEVEL {lname} =====", flush=True)
        lev = Level(lname)
        print(f"  flow loaded, ~{lev.median_counts:.0f} counts", flush=True)

        # ---------- CLEAN CONTROL ----------
        # (a) assumption-free: IS logZ vs brute-force prior-Monte-Carlo logZ on the
        #     SAME freshly-drawn clean spectra (works at every level, no flow).
        # (b) cross-check vs committed nested sampling on the EXACT reproduced NS
        #     spectra (verified reproducible at faint & bright; medium NS rows come
        #     from a different run, flagged ns_reproduced=False there).
        N_CTRL = 12
        MC_BUDGET_ES = 60000 if lname == "bright" else 20000   # bigger IS budget at low-ESS bright
        xc_ctrl, _ = gen_clean(lev, N_CTRL, seed=60000 + block_idx)
        ctrl = []
        for i in range(N_CTRL):
            r = is_reweight(lev, xc_ctrl[i], MC_BUDGET_ES, seed=70000 + block_idx * 100 + i)
            zmc = prior_mc_logz(lev, xc_ctrl[i], 1_000_000, seed=71000 + block_idx * 100 + i)
            ctrl.append({"logZ_is": r["logZ"], "logZ_mc": zmc,
                         "ess_frac": r["ess_frac"], "acc": r["acc"], "khat": r["khat"]})
        dz_mc = np.array([c["logZ_is"] - c["logZ_mc"] for c in ctrl], dtype=float)
        effc = np.array([c["ess_frac"] for c in ctrl])
        khc = np.array([c["khat"] for c in ctrl])

        # (b) reproduced-NS cross-check
        n_ns = {"faint": 25, "medium": 16, "bright": 15}[lname]
        x_nsclean, ids = reproduce_ns_clean(lev, lname, block_idx, n_ns)
        rows_ns = {r["spectrum_id"]: r for r in
                   (json.loads(l) for l in open(ROOT / "outputs/ns_bench/results.jsonl") if l.strip())}
        dz_ns = []
        for i in range(n_ns):
            ns = rows_ns.get(ids[i])
            if ns is None or int(x_nsclean[i].sum()) != ns["n_counts"]:
                continue  # spectrum did not reproduce (medium) -> skip NS compare
            r = is_reweight(lev, x_nsclean[i], MC_BUDGET_ES, seed=72000 + block_idx * 100 + i)
            dz_ns.append(r["logZ"] - float(ns["ns"]["logz"]))
        dz_ns = np.array(dz_ns, dtype=float)
        ns_reproduced = dz_ns.size > 0

        results["clean_control"][lname] = {
            "median_counts": lev.median_counts,
            "n_ctrl": N_CTRL,
            "is_budget": MC_BUDGET_ES,
            "ess_frac_median": float(np.nanmedian(effc)),
            "ess_frac_mean": float(np.nanmean(effc)),
            "khat_median": float(np.nanmedian(khc)),
            "khat_frac_gt_0p7": float(np.mean(np.asarray(khc)[np.isfinite(khc)] > 0.7)) if np.isfinite(khc).any() else float("nan"),
            "logZ_is_vs_priorMC_median": float(np.nanmedian(dz_mc)),
            "logZ_is_vs_priorMC_mean": float(np.nanmean(dz_mc)),
            "logZ_is_vs_priorMC_std": float(np.nanstd(dz_mc)),
            "logZ_is_vs_priorMC_maxabs": float(np.nanmax(np.abs(dz_mc))),
            "ns_reproduced": bool(ns_reproduced),
            "n_ns_matched": int(dz_ns.size),
            "logZ_is_vs_NS_median": float(np.nanmedian(dz_ns)) if ns_reproduced else None,
            "logZ_is_vs_NS_std": float(np.nanstd(dz_ns)) if ns_reproduced else None,
            "logZ_is_vs_NS_maxabs": float(np.nanmax(np.abs(dz_ns))) if ns_reproduced else None,
            "per_spectrum_priorMC": ctrl,
        }
        print(f"  [clean-ctrl] eff_med={np.nanmedian(effc):.4f} khat_med={np.nanmedian(khc):.2f} "
              f"dlogZ(IS-priorMC) med={np.nanmedian(dz_mc):+.3f} std={np.nanstd(dz_mc):.3f} "
              f"maxabs={np.nanmax(np.abs(dz_mc)):.2f} | "
              f"NS-repro={ns_reproduced} dlogZ(IS-NS) med="
              f"{(np.nanmedian(dz_ns) if ns_reproduced else float('nan')):+.3f}", flush=True)

        # ---------- SWEEP: clean (neg) vs each misspec family ----------
        xc, _ = gen_clean(lev, N_CLEAN, seed=80000 + block_idx)
        clean_cell = {"ess_frac": [], "khat": [], "acc": []}
        rep_examples = {}
        for i in range(N_CLEAN):
            r = is_reweight(lev, xc[i], N_BUDGET, seed=81000 + block_idx * 1000 + i)
            clean_cell["ess_frac"].append(r["ess_frac"])
            clean_cell["khat"].append(r["khat"])
            clean_cell["acc"].append(r["acc"])
            if i == 0:
                rep_examples["clean"] = r
        cells = {"clean": clean_cell}
        aucs = {}
        for fi, (fam, strength, fixed) in enumerate(MIS):
            xm, _ = gen_misspec(lev, fam, strength, N_MIS, seed=82000 + block_idx * 1000 + fi * 100,
                                fixed=fixed)
            cell = {"ess_frac": [], "khat": [], "acc": []}
            for i in range(N_MIS):
                r = is_reweight(lev, xm[i], N_BUDGET, seed=83000 + block_idx * 10000 + fi * 1000 + i)
                cell["ess_frac"].append(r["ess_frac"])
                cell["khat"].append(r["khat"])
                cell["acc"].append(r["acc"])
                if i == 0:
                    rep_examples[f"{fam}"] = r
            cells[f"{fam}_s{strength:g}"] = cell
            # ROC: suspicion = LOW ess -> use -ess_frac ; and high khat -> khat
            eff_clean = np.array(clean_cell["ess_frac"]); eff_mis = np.array(cell["ess_frac"])
            _, _, auc_eff = roc_auc(-eff_clean, -eff_mis)
            kh_clean = np.array(clean_cell["khat"]); kh_mis = np.array(cell["khat"])
            _, _, auc_khat = roc_auc(kh_clean, kh_mis)
            aucs[f"{fam}_s{strength:g}"] = {
                "auc_ess_eff": float(auc_eff),
                "auc_khat": float(auc_khat),
                "ess_frac_median_clean": float(np.nanmedian(eff_clean)),
                "ess_frac_median_mis": float(np.nanmedian(eff_mis)),
                "khat_median_clean": float(np.nanmedian(kh_clean)),
                "khat_median_mis": float(np.nanmedian(kh_mis)),
                "acc_median_mis": float(np.nanmedian(cell["acc"])),
            }
            print(f"  [{fam} s{strength:g}] AUC_ess={auc_eff:.3f} AUC_khat={auc_khat:.3f} "
                  f"eff_mis_med={np.nanmedian(eff_mis):.4f} (clean {np.nanmedian(eff_clean):.4f}) "
                  f"acc_mis={np.nanmedian(cell['acc']):.2f}", flush=True)

        results["sweep"][lname] = {
            "median_counts": lev.median_counts,
            "clean_ess_frac_median": float(np.nanmedian(clean_cell["ess_frac"])),
            "clean_khat_median": float(np.nanmedian(clean_cell["khat"])),
            "aucs": aucs,
            "cells": {k: {"ess_frac": list(map(float, v["ess_frac"])),
                          "khat": list(map(float, v["khat"])),
                          "acc": list(map(float, v["acc"]))} for k, v in cells.items()},
        }

        # figures
        fig_weights(rep_examples, OUT / f"weights_{lname}.png")
        fig_ess({k: cells[k] for k in cells}, lname, OUT / f"ess_khat_{lname}.png")

        # incremental crash-safe dump after each level (own file write, own process
        # when --level is used, so a crash on one level cannot lose another's results)
        results.setdefault("wall_s_by_level", {})[lname] = time.time() - t_start
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [saved] partial results through level {lname}", flush=True)

    results["wall_s"] = sum(results.get("wall_s_by_level", {}).values())
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE in {time.time()-t_start:.0f}s (this process) -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
