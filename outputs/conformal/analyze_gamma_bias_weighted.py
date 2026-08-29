"""Photon-index bias of the conformal populations under a precision-weighted
estimator, with stratified permutation tests.

Written independently of analyze_gamma_bias_sigma.py, which it does not
import. It reads only the npz/json artifacts run_conformal.py and
run_conformal_rep2.py wrote and, for two population-identity checks,
regenerates the calibration-split truths with those drivers' seeds.

Everything it computes is written to
outputs/conformal/sigma_replication/gamma_bias_weighted_results.json.

Run:
    .venv\\Scripts\\python.exe outputs\\conformal\\analyze_gamma_bias_weighted.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONF = ROOT / "outputs" / "conformal"
OUT = CONF / "sigma_replication"
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = ["tbabs_1_nh", "powerlaw_1_alpha", "powerlaw_1_norm",
          "blackbodyrad_1_kT", "blackbodyrad_1_norm"]
GI = 1
GLO, GHI = 1.0, 3.0          # Gamma prior box

REPS = {
    "rep1": {"base": 20260611, "npz": CONF / "conformal_coverage_curves.npz",
             "json": CONF / "conformal_results.json"},
    "rep2": {"base": 20260723, "npz": CONF / "rep2" / "conformal_coverage_curves_rep2.npz",
             "json": CONF / "rep2" / "conformal_results_rep2.json"},
}
NPERM = 20000
RNG = np.random.default_rng(31337)          # distinct from analyze_gamma_bias_sigma.py's


# ----------------------------------------------------------------- estimators
def raw_mean(b):
    n = b.size
    m = float(b.mean())
    se = float(b.std(ddof=1) / np.sqrt(n))
    return m, se


def pull_mean(b, s):
    r = b / s
    n = r.size
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(n)), float(r.std(ddof=1))


def iv_weighted(b, s):
    """Inverse-VARIANCE weighted mean of b (the textbook precision-weighted
    estimator for b_i ~ N(beta, s_i^2)).  Model SE assumes s_i is the true
    error scale; robust (sandwich) SE does not."""
    w = 1.0 / s ** 2
    sw = w.sum()
    beta = float((w * b).sum() / sw)
    se_model = float(1.0 / np.sqrt(sw))
    se_rob = float(np.sqrt((w ** 2 * (b - beta) ** 2).sum()) / sw)
    return beta, se_model, se_rob


def two_sample(m1, se1, m2, se2):
    d = m1 - m2
    se = float(np.hypot(se1, se2))
    return {"delta": float(d), "se": se, "z": float(d / se)}


def iv_combine(pairs):
    w = np.array([1.0 / se ** 2 for _, se in pairs])
    m = np.array([mu for mu, _ in pairs])
    mc = float((w * m).sum() / w.sum())
    se = float(1.0 / np.sqrt(w.sum()))
    return {"mean": mc, "se": se, "z": mc / se}


def perm_p(b1, s1, b2, s2, stat, nperm=NPERM, rng=None):
    """Two-sided label-permutation p for stat(b,s) difference between splits."""
    rng = rng or RNG
    b = np.concatenate([b1, b2]); s = np.concatenate([s1, s2])
    n1 = b1.size
    obs = stat(b1, s1) - stat(b2, s2)
    cnt = 0
    n = b.size
    for _ in range(nperm):
        idx = rng.permutation(n)
        a, c = idx[:n1], idx[n1:]
        d = stat(b[a], s[a]) - stat(b[c], s[c])
        if abs(d) >= abs(obs) - 1e-15:
            cnt += 1
    return {"observed": float(obs), "p_two_sided": (cnt + 1) / (nperm + 1)}


def sig_summary(s):
    inv = 1.0 / s
    return {"n": int(s.size), "median": float(np.median(s)), "mean": float(s.mean()),
            "rms": float(np.sqrt((s ** 2).mean())),
            "harmonic_mean": float(1.0 / inv.mean()),
            "mean_inv_sigma": float(inv.mean()),
            "p16": float(np.percentile(s, 16)), "p84": float(np.percentile(s, 84)),
            "min": float(s.min()), "max": float(s.max()),
            "frac_above_0.5": float((s > 0.5).mean()),
            "frac_above_0.45": float((s > 0.45).mean()),
            "frac_below_0.15": float((s < 0.15).mean())}


# ----------------------------------------------------------------- main
def main():
    t0 = time.time()
    out = {"generated": "2026-08-14", "nperm": NPERM,
           "rng_seed": 31337}
    D = {}

    # ---------------- load, and independently reproduce the ORIGINAL json ----
    out["original_json_reproduction"] = {}
    for rep, cfg in REPS.items():
        d = np.load(cfg["npz"], allow_pickle=True)
        oj = json.load(open(cfg["json"]))
        # seed bookkeeping check, done my own way
        base = cfg["base"]
        out.setdefault("seed_checks", {})[rep] = {
            "config_seed_shift": oj["config"]["seed_shift"],
            "base_plus_823": base + 823,
            "match_shift": oj["config"]["seed_shift"] == base + 823,
            "config_seed_cleanctl": oj["config"]["seed_cleanctl"],
            "base_plus_831": base + 831,
            "match_cleanctl": oj["config"]["seed_cleanctl"] == base + 831,
            "median_counts_cal": oj["config"]["median_counts_cal"],
            "median_counts_shift": oj["config"]["median_counts_shift"],
            "median_counts_cleanctl": oj["config"]["median_counts_cleanctl"],
        }
        for split, keys, ojkey in (
                ("shifted", ("means_shift", "stds_shift", "th_shift"), "all_param_bias_shifted"),
                ("clean", ("means_clean", "stds_clean", "th_cleanctl"), "all_param_bias_clean_control")):
            mu = np.asarray(d[keys[0]], dtype=np.float64)
            sg = np.asarray(d[keys[1]], dtype=np.float64)
            th = np.asarray(d[keys[2]], dtype=np.float64)
            D[(rep, split)] = (mu, sg, th)
            rep_chk = {}
            for p, name in enumerate(PARAMS):
                b = mu[:, p] - th[:, p]
                s = sg[:, p]
                m, se = raw_mean(b)
                pm, pse, _ = pull_mean(b, s)
                o = oj[ojkey][name]
                rep_chk[name] = {
                    "d_bias_mean_abs": abs(o["bias_mean_abs"] - m),
                    "d_bias_se_abs": abs(o["bias_se_abs"] - se),
                    "d_bias_mean_in_post_std_vs_MEANOFRATIOS": abs(o["bias_mean_in_post_std"] - pm),
                    "d_bias_se_in_post_std_vs_MEANOFRATIOS": abs(o["bias_se_in_post_std"] - pse),
                    "d_median_posterior_std": abs(o["median_posterior_std"] - float(np.median(s))),
                    # the alternative reading the report ASSUMED:
                    "orig_value": o["bias_mean_in_post_std"],
                    "mean_b_over_median_sigma": m / float(np.median(s)),
                    "mean_b_over_mean_sigma": m / float(s.mean()),
                }
            out["original_json_reproduction"][f"{rep}/{split}"] = rep_chk
        # float32 provenance
        d32 = np.load(cfg["npz"], allow_pickle=True)
        out.setdefault("dtypes", {})[rep] = {
            k: str(d32[k].dtype) for k in ("means_shift", "stds_shift", "th_shift")}

    # ---------------- per-split Gamma statistics, my own code ---------------
    out["gamma_per_split"] = {}
    for key, (mu, sg, th) in D.items():
        rep, split = key
        b = mu[:, GI] - th[:, GI]
        s = sg[:, GI]
        m, se = raw_mean(b)
        pm, pse, psd = pull_mean(b, s)
        beta_w, se_m, se_r = iv_weighted(b, s)
        ss = sig_summary(s)
        out["gamma_per_split"][f"{rep}/{split}"] = {
            "n": int(b.size),
            "raw_mean": m, "raw_se": se, "raw_z": m / se,
            "raw_median": float(np.median(b)),
            "pull_mean": pm, "pull_se": pse, "pull_sd": psd,
            "pull_z": pm / pse,
            "iv_beta": beta_w, "iv_se_model": se_m, "iv_se_robust": se_r,
            "iv_z_robust": beta_w / se_r, "iv_z_model": beta_w / se_m,
            "sigma": ss,
            "implied_sigma_meanb_over_meanpull": (m / pm) if pm != 0 else float("nan"),
            "harmonic_mean_sigma": ss["harmonic_mean"],
            "implied_over_harmonic": (m / pm) / ss["harmonic_mean"] if pm != 0 else float("nan"),
            "corr_b_sigma": float(np.corrcoef(b, s)[0, 1]),
            "corr_absb_sigma": float(np.corrcoef(np.abs(b), s)[0, 1]),
        }

    # ---------------- shifted - clean, per rep, three estimators ------------
    out["gamma_delta_per_rep"] = {}
    for rep in REPS:
        mu1, sg1, th1 = D[(rep, "shifted")]
        mu0, sg0, th0 = D[(rep, "clean")]
        b1, s1 = mu1[:, GI] - th1[:, GI], sg1[:, GI]
        b0, s0 = mu0[:, GI] - th0[:, GI], sg0[:, GI]
        m1, se1 = raw_mean(b1); m0, se0 = raw_mean(b0)
        p1, pse1, _ = pull_mean(b1, s1); p0, pse0, _ = pull_mean(b0, s0)
        w1, _, r1 = iv_weighted(b1, s1); w0, _, r0 = iv_weighted(b0, s0)
        e = {
            "raw": two_sample(m1, se1, m0, se0),
            "pull": two_sample(p1, pse1, p0, pse0),
            "iv_robust": two_sample(w1, r1, w0, r0),
        }
        e["raw"]["perm"] = perm_p(b1, s1, b0, s0, lambda b, s: b.mean())
        e["pull"]["perm"] = perm_p(b1, s1, b0, s0, lambda b, s: (b / s).mean())
        e["iv_robust"]["perm"] = perm_p(
            b1, s1, b0, s0,
            lambda b, s: ((b / s ** 2).sum() / (1.0 / s ** 2).sum()))
        # Welch t on the two sigma distributions: are the splits' sigma
        # populations even comparable?
        e["sigma_split_compare"] = {
            "median_shift": float(np.median(s1)), "median_clean": float(np.median(s0)),
            "mean_shift": float(s1.mean()), "mean_clean": float(s0.mean()),
            "ks_like_max_cdf_gap": float(np.max(np.abs(
                np.searchsorted(np.sort(s1), np.sort(np.concatenate([s1, s0]))) / s1.size
                - np.searchsorted(np.sort(s0), np.sort(np.concatenate([s1, s0]))) / s0.size))),
        }
        out["gamma_delta_per_rep"][rep] = e

    # ---------------- cross-rep combination --------------------------------
    comb = {}
    for est in ("raw", "pull", "iv_robust"):
        pairs = [(out["gamma_delta_per_rep"][r][est]["delta"],
                  out["gamma_delta_per_rep"][r][est]["se"]) for r in REPS]
        comb[est] = iv_combine(pairs)
    # shifted-vs-ZERO combination (the OTHER estimand)
    for est, key in (("raw", ("raw_mean", "raw_se")), ("pull", ("pull_mean", "pull_se")),
                     ("iv_robust", ("iv_beta", "iv_se_robust"))):
        for split in ("shifted", "clean"):
            pairs = [(out["gamma_per_split"][f"{r}/{split}"][key[0]],
                      out["gamma_per_split"][f"{r}/{split}"][key[1]]) for r in REPS]
            comb[f"{est}_{split}_vs_zero"] = iv_combine(pairs)
    out["gamma_combined"] = comb

    # pooled sigma, all 2000 spectra
    sall = np.concatenate([D[(r, s)][1][:, GI] for r in REPS for s in ("shifted", "clean")])
    out["sigma_gamma_pooled_2000"] = sig_summary(sall)
    conv = {}
    for nm, val in (("median", np.median(sall)), ("mean", sall.mean()),
                    ("rms", np.sqrt((sall ** 2).mean())),
                    ("harmonic", 1.0 / (1.0 / sall).mean())):
        conv[nm] = {
            "sigma": float(val),
            "raw_delta_in_sigma": comb["raw"]["mean"] / float(val),
            "raw_delta_se_in_sigma": comb["raw"]["se"] / float(val),
            "shifted_vs_zero_in_sigma": comb["raw_shifted_vs_zero"]["mean"] / float(val),
        }
    out["sigma_convention_sensitivity"] = conv

    # ---------------- multiplicity: all 5 params ---------------------------
    allp = {}
    for p, name in enumerate(PARAMS):
        per_rep = {}
        for rep in REPS:
            mu1, sg1, th1 = D[(rep, "shifted")]
            mu0, sg0, th0 = D[(rep, "clean")]
            b1, s1 = mu1[:, p] - th1[:, p], sg1[:, p]
            b0, s0 = mu0[:, p] - th0[:, p], sg0[:, p]
            m1, se1 = raw_mean(b1); m0, se0 = raw_mean(b0)
            p1, pse1, _ = pull_mean(b1, s1); p0, pse0, _ = pull_mean(b0, s0)
            per_rep[rep] = {"raw": two_sample(m1, se1, m0, se0),
                            "pull": two_sample(p1, pse1, p0, pse0),
                            "clean_vs_zero_raw_z": m0 / se0,
                            "shift_vs_zero_raw_z": m1 / se1}
        allp[name] = {
            "per_rep": per_rep,
            "combined_raw": iv_combine([(per_rep[r]["raw"]["delta"], per_rep[r]["raw"]["se"])
                                        for r in REPS]),
            "combined_pull": iv_combine([(per_rep[r]["pull"]["delta"], per_rep[r]["pull"]["se"])
                                         for r in REPS]),
        }
    out["all_params_delta"] = allp

    # ---------------- rail / Bhatia-Davis attack ---------------------------
    rail = {}
    for key, (mu, sg, th) in D.items():
        rep, split = key
        m = mu[:, GI]; s = sg[:, GI]
        edge = np.minimum(m - GLO, GHI - m)
        bd = np.sqrt(np.clip((m - GLO) * (GHI - m), 1e-12, None))   # Bhatia-Davis bound
        ratio = s / bd
        rail[f"{rep}/{split}"] = {
            "corr_sigma_edge": float(np.corrcoef(s, edge)[0, 1]),
            "corr_sigmaOverBound_edge": float(np.corrcoef(ratio, edge)[0, 1]),
            "frac_sigma_exceeds_bound": float((s > bd).mean()),
            "bound_ratio_median": float(np.median(ratio)),
            "mean_edge": float(edge.mean()),
        }
    out["rail_bhatia_davis"] = rail

    # NULL SIMULATION: no bias anywhere, posteriors = truncated normal on
    # [1,3] with width that depends on truth only.  Does corr(sigma, edge)
    # appear anyway?  Does a positive mean pull appear anyway?
    def null_sim(width_fn, n=500, ndraw=1000, seed=7):
        rng = np.random.default_rng(seed)
        th = rng.uniform(GLO, GHI, n)
        mus, sds, bs = [], [], []
        for t in th:
            w = width_fn(t)
            # unbiased posterior centred on truth + independent centring noise
            c = t + rng.normal(0.0, w)
            x = rng.normal(c, w, size=ndraw * 3)
            x = x[(x >= GLO) & (x <= GHI)][:ndraw]
            if x.size < 50:
                x = np.clip(rng.normal(c, w, size=ndraw), GLO, GHI)
            mus.append(x.mean()); sds.append(x.std())
        mus = np.array(mus); sds = np.array(sds)
        b = mus - th
        edge = np.minimum(mus - GLO, GHI - mus)
        m, se = raw_mean(b)
        pm, pse, _ = pull_mean(b, sds)
        return {"corr_sigma_edge": float(np.corrcoef(sds, edge)[0, 1]),
                "raw_mean": m, "raw_se": se, "raw_z": m / se,
                "pull_mean": pm, "pull_se": pse, "pull_z": pm / pse,
                "sigma_median": float(np.median(sds))}

    out["null_simulation"] = {
        "constant_width_0.30": null_sim(lambda t: 0.30),
        "width_rises_with_Gamma_0.15_to_0.55": null_sim(lambda t: 0.15 + 0.20 * (t - 1.0)),
        "width_falls_with_Gamma_0.55_to_0.15": null_sim(lambda t: 0.55 - 0.20 * (t - 1.0)),
    }

    # ---------------- terciles: within-split vs COMMON boundaries ----------
    terc = {}
    for rep in REPS:
        mu1, sg1, th1 = D[(rep, "shifted")]
        mu0, sg0, th0 = D[(rep, "clean")]
        b1, s1 = mu1[:, GI] - th1[:, GI], sg1[:, GI]
        b0, s0 = mu0[:, GI] - th0[:, GI], sg0[:, GI]
        pooled = np.concatenate([s1, s0])
        qc = np.percentile(pooled, [100 / 3, 200 / 3])
        q1 = np.percentile(s1, [100 / 3, 200 / 3])
        q0 = np.percentile(s0, [100 / 3, 200 / 3])

        def bands(b, s, q):
            o = {}
            for nm, msk in (("tight", s <= q[0]), ("mid", (s > q[0]) & (s <= q[1])),
                            ("wide", s > q[1])):
                if msk.sum() < 3:
                    o[nm] = None; continue
                m, se = raw_mean(b[msk])
                o[nm] = {"n": int(msk.sum()), "mean": m, "se": se,
                         "sigma_median": float(np.median(s[msk]))}
            return o

        own = {"shifted": bands(b1, s1, q1), "clean": bands(b0, s0, q0)}
        com = {"shifted": bands(b1, s1, qc), "clean": bands(b0, s0, qc)}
        terc[rep] = {
            "q_shifted": q1.tolist(), "q_clean": q0.tolist(), "q_common": qc.tolist(),
            "own_boundaries": own, "common_boundaries": com,
            "paired_own": {k: two_sample(own["shifted"][k]["mean"], own["shifted"][k]["se"],
                                         own["clean"][k]["mean"], own["clean"][k]["se"])
                           for k in ("tight", "mid", "wide")},
            "paired_common": {k: two_sample(com["shifted"][k]["mean"], com["shifted"][k]["se"],
                                            com["clean"][k]["mean"], com["clean"][k]["se"])
                              for k in ("tight", "mid", "wide") if com["shifted"][k] and com["clean"][k]},
        }
    out["terciles"] = terc

    # ---------------- bias vs TRUE Gamma (rail asymmetry) ------------------
    tb = {}
    edges = np.array([1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
    for key, (mu, sg, th) in D.items():
        rep, split = key
        b = mu[:, GI] - th[:, GI]; s = sg[:, GI]; t = th[:, GI]
        rows = []
        for i in range(len(edges) - 1):
            msk = (t >= edges[i]) & (t < edges[i] + (edges[i + 1] - edges[i]) + (1e-9 if i == len(edges) - 2 else 0))
            msk = (t >= edges[i]) & ((t < edges[i + 1]) | ((i == len(edges) - 2) & (t <= edges[i + 1])))
            if msk.sum() < 3:
                rows.append(None); continue
            m, se = raw_mean(b[msk])
            rows.append({"bin": [float(edges[i]), float(edges[i + 1])], "n": int(msk.sum()),
                         "raw_mean": m, "raw_se": se,
                         "sigma_median": float(np.median(s[msk])),
                         "pull_mean": float((b[msk] / s[msk]).mean())})
        tb[f"{rep}/{split}"] = rows
    out["bias_vs_true_gamma"] = tb

    # ---------------- population identity / overlap ------------------------
    ident = {}
    for rep in REPS:
        th1 = D[(rep, "shifted")][2]; th0 = D[(rep, "clean")][2]
        # min pairwise distance between the two truth sets (any duplicate rows?)
        dmin = np.min(np.abs(th1[:, None, :] - th0[None, :, :]).max(axis=2))
        ident[rep] = {"min_chebyshev_dist_shift_vs_clean_truths": float(dmin),
                      "identical_rows": int(np.sum(np.all(th1 == th0, axis=1)))}
    out["population_identity"] = ident

    # regenerate the CALIBRATION truths to test the disjointness claim
    if "--no-regen" not in sys.argv:
        try:
            from sbixcal import responses as R
            from sbixcal.misspec import simulate_misspec_population
            PHYS_PRIORS = {
                "tbabs_1_nh": {"dist": "uniform", "low": 0.15, "high": 0.35},
                "powerlaw_1_alpha": {"dist": "uniform", "low": 1.0, "high": 3.0},
                "powerlaw_1_norm": {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
                "blackbodyrad_1_kT": {"dist": "uniform", "low": 0.3, "high": 3.0},
                "blackbodyrad_1_norm": {"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
            }
            base = R.load_base_obsconf("NGC7793_ULX4_PN")
            oc = R.scale_exposure(base, 353.4)
            for rep, cfg in REPS.items():
                x, thc, present = simulate_misspec_population(
                    "tbabs_powerlaw_bb", PHYS_PRIORS, oc, "B4", 0.0, 500, cfg["base"] + 811)
                thc = np.asarray(thc, dtype=np.float64)
                th1 = D[(rep, "shifted")][2]; th0 = D[(rep, "clean")][2]
                out["population_identity"][rep].update({
                    "cal_median_counts_regen": float(np.median(np.asarray(x).sum(1))),
                    "cal_vs_clean_min_chebyshev": float(
                        np.min(np.abs(thc[:, None, :] - th0[None, :, :]).max(axis=2))),
                    "cal_vs_shift_min_chebyshev": float(
                        np.min(np.abs(thc[:, None, :] - th1[None, :, :]).max(axis=2))),
                    "cal_identical_rows_vs_clean": int(np.sum(
                        np.all(thc[:, None, :] == th0[None, :, :], axis=2))),
                })
        except Exception as exc:                                # noqa: BLE001
            out["population_identity"]["regen_error"] = repr(exc)

    out["wall_time_s"] = time.time() - t0
    with open(OUT / "gamma_bias_weighted_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- console digest ---------------------------------------
    def g(k):
        return out["gamma_per_split"][k]
    print("\n=== reproduction of original json (max abs diff over 5 params x 4 splits) ===")
    worst = {}
    for k, v in out["original_json_reproduction"].items():
        for name, dd in v.items():
            for kk, vv in dd.items():
                if kk.startswith("d_"):
                    worst[kk] = max(worst.get(kk, 0.0), vv)
    for k, v in worst.items():
        print(f"  {k:52s} {v:.3e}")

    print("\n=== Gamma per split ===")
    for k in ["rep1/shifted", "rep1/clean", "rep2/shifted", "rep2/clean"]:
        v = g(k)
        print(f"  {k:14s} raw={v['raw_mean']:+.5f}+/-{v['raw_se']:.5f} z={v['raw_z']:+.2f} | "
              f"pull={v['pull_mean']:+.4f}+/-{v['pull_se']:.4f} z={v['pull_z']:+.2f} "
              f"sd(pull)={v['pull_sd']:.3f} | IV={v['iv_beta']:+.5f}+/-{v['iv_se_robust']:.5f} "
              f"z={v['iv_z_robust']:+.2f}")
        print(f"                 sigma med={v['sigma']['median']:.4f} mean={v['sigma']['mean']:.4f} "
              f"harm={v['sigma']['harmonic_mean']:.4f} | implied(meanb/meanpull)="
              f"{v['implied_sigma_meanb_over_meanpull']:+.4f} "
              f"(ratio to harmonic {v['implied_over_harmonic']:+.2f})")

    print("\n=== shifted - clean, per rep ===")
    for rep in REPS:
        for est in ("raw", "pull", "iv_robust"):
            e = out["gamma_delta_per_rep"][rep][est]
            print(f"  {rep} {est:10s} d={e['delta']:+.5f}+/-{e['se']:.5f} z={e['z']:+.2f} "
                  f"perm p={e['perm']['p_two_sided']:.4f}")

    print("\n=== combined across reps ===")
    for k, v in out["gamma_combined"].items():
        print(f"  {k:28s} {v['mean']:+.5f} +/- {v['se']:.5f}  z={v['z']:+.2f}")

    print("\n=== sigma convention sensitivity (combined raw delta) ===")
    for nm, v in out["sigma_convention_sensitivity"].items():
        print(f"  {nm:9s} sigma={v['sigma']:.4f} -> delta={v['raw_delta_in_sigma']:+.4f} "
              f"+/- {v['raw_delta_se_in_sigma']:.4f} sigma; shifted-vs-0 = "
              f"{v['shifted_vs_zero_in_sigma']:+.4f} sigma")

    print("\n=== rail / Bhatia-Davis ===")
    for k, v in out["rail_bhatia_davis"].items():
        print(f"  {k:14s} corr(sigma,edge)={v['corr_sigma_edge']:+.3f}  "
              f"corr(sigma/bound,edge)={v['corr_sigmaOverBound_edge']:+.3f}  "
              f"frac sigma>bound={v['frac_sigma_exceeds_bound']:.3f}")
    print("  NULL sims (no bias by construction):")
    for k, v in out["null_simulation"].items():
        print(f"    {k:34s} corr(sigma,edge)={v['corr_sigma_edge']:+.3f} "
              f"raw z={v['raw_z']:+.2f} pull mean={v['pull_mean']:+.4f} z={v['pull_z']:+.2f}")

    print("\n=== terciles: paired tight, own vs common boundaries ===")
    for rep in REPS:
        po = terc[rep]["paired_own"]["tight"]; pc = terc[rep]["paired_common"]["tight"]
        print(f"  {rep} tight own z={po['z']:+.2f} (d={po['delta']:+.4f})   "
              f"common z={pc['z']:+.2f} (d={pc['delta']:+.4f})")

    print("\n=== all params, combined shifted-clean ===")
    for name, v in out["all_params_delta"].items():
        print(f"  {name:22s} raw z={v['combined_raw']['z']:+.2f}   pull z={v['combined_pull']['z']:+.2f}")

    print("\n=== population identity ===")
    print(json.dumps(out["population_identity"], indent=2))
    print(f"\nwall {out['wall_time_s']:.1f}s -> {OUT / 'gamma_bias_weighted_results.json'}")


if __name__ == "__main__":
    main()
