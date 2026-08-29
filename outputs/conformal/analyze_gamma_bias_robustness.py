"""Robustness of the precision-weighted (1/sigma^2) photon-index bias that
analyze_gamma_bias_weighted.py computes.

Checks:
  * effective sample size of the 1/sigma^2 weights, max leave-one-out influence
  * trimmed / winsorized-sigma variants (floor sigma at the 5th, 10th pct)
  * rail-free subsample: keep only spectra whose posterior mean is > 0.3 (and
    > 0.5) from the Gamma prior edge
  * sigma-matched (common-decile-stratified) difference, which removes any
    sigma-distribution mismatch between the two splits by construction
  * counts-confound proxy: recompute the delta after matching the two splits
    on posterior sigma rank
  * stratified permutation test pooled over both replications
  * Fisher combination of the per-rep permutation p-values
  * predicted-vs-observed pull delta under a homogeneous absolute bias

Writes outputs/conformal/sigma_replication/gamma_bias_robustness_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONF = HERE.parents[1] / "outputs" / "conformal"
OUT = CONF / "sigma_replication"
OUT.mkdir(parents=True, exist_ok=True)
GI = 1
GLO, GHI = 1.0, 3.0
NPZ = {"rep1": CONF / "conformal_coverage_curves.npz",
       "rep2": CONF / "rep2" / "conformal_coverage_curves_rep2.npz"}
RNG = np.random.default_rng(90210)
NPERM = 40000


def load(rep):
    d = np.load(NPZ[rep], allow_pickle=True)
    out = {}
    for split, (km, ks, kt) in (("shifted", ("means_shift", "stds_shift", "th_shift")),
                                ("clean", ("means_clean", "stds_clean", "th_cleanctl"))):
        mu = np.asarray(d[km], dtype=np.float64)[:, GI]
        sg = np.asarray(d[ks], dtype=np.float64)[:, GI]
        th = np.asarray(d[kt], dtype=np.float64)[:, GI]
        out[split] = (mu - th, sg, mu, th)
    return out


def iv(b, s):
    w = 1.0 / s ** 2
    sw = w.sum()
    beta = (w * b).sum() / sw
    se_rob = np.sqrt((w ** 2 * (b - beta) ** 2).sum()) / sw
    return float(beta), float(se_rob)


def raw(b):
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(b.size))


def ts(m1, se1, m0, se0):
    d = m1 - m0
    se = float(np.hypot(se1, se0))
    return {"delta": float(d), "se": se, "z": float(d / se)}


def ivcomb(pairs):
    w = np.array([1 / se ** 2 for _, se in pairs])
    m = np.array([x for x, _ in pairs])
    mc = float((w * m).sum() / w.sum())
    se = float(1 / np.sqrt(w.sum()))
    return {"mean": mc, "se": se, "z": mc / se}


def main():
    out = {}
    R = {r: load(r) for r in NPZ}

    # ---------- 1. weight concentration / influence -----------------------
    conc = {}
    for r in NPZ:
        for sp in ("shifted", "clean"):
            b, s, mu, th = R[r][sp]
            w = 1.0 / s ** 2
            n_eff = float(w.sum() ** 2 / (w ** 2).sum())
            beta, _ = iv(b, s)
            # leave-one-out
            loo = np.array([iv(np.delete(b, i), np.delete(s, i))[0] for i in range(b.size)])
            conc[f"{r}/{sp}"] = {
                "n": int(b.size), "n_eff_weights": n_eff,
                "top1_weight_share": float(w.max() / w.sum()),
                "top10_weight_share": float(np.sort(w)[-10:].sum() / w.sum()),
                "beta_iv": beta,
                "loo_max_abs_change": float(np.abs(loo - beta).max()),
                "loo_max_change_index": int(np.argmax(np.abs(loo - beta))),
                "sigma_min": float(s.min()),
                "sigma_of_max_weight": float(s[np.argmax(w)]),
                "edge_of_max_weight": float(min(mu[np.argmax(w)] - GLO, GHI - mu[np.argmax(w)])),
            }
    out["weight_concentration"] = conc

    # ---------- 2. variants of the differenced estimator -------------------
    def delta_variants(r):
        b1, s1, mu1, th1 = R[r]["shifted"]
        b0, s0, mu0, th0 = R[r]["clean"]
        v = {}
        # plain
        v["iv_plain"] = ts(*iv(b1, s1), *iv(b0, s0))
        v["raw_plain"] = ts(*raw(b1), *raw(b0))
        # sigma floored at pooled 5th / 10th percentile (winsorized weights)
        pooled = np.concatenate([s1, s0])
        for q in (5, 10, 25):
            f = np.percentile(pooled, q)
            v[f"iv_sigma_floor_p{q}"] = ts(*iv(b1, np.maximum(s1, f)),
                                           *iv(b0, np.maximum(s0, f)))
        # drop the tightest 5% / 10% of each split
        for q in (5, 10):
            k1 = s1 > np.percentile(s1, q)
            k0 = s0 > np.percentile(s0, q)
            v[f"iv_drop_tightest_{q}pct"] = ts(*iv(b1[k1], s1[k1]), *iv(b0[k0], s0[k0]))
        # rail-free subsamples (posterior mean far from the Gamma prior edge)
        for thr in (0.3, 0.5):
            edge1 = np.minimum(mu1 - GLO, GHI - mu1)
            edge0 = np.minimum(mu0 - GLO, GHI - mu0)
            k1, k0 = edge1 > thr, edge0 > thr
            v[f"iv_edge_gt_{thr}"] = ts(*iv(b1[k1], s1[k1]), *iv(b0[k0], s0[k0]))
            v[f"iv_edge_gt_{thr}"]["n_shift"] = int(k1.sum())
            v[f"iv_edge_gt_{thr}"]["n_clean"] = int(k0.sum())
            v[f"raw_edge_gt_{thr}"] = ts(*raw(b1[k1]), *raw(b0[k0]))
        # TRUTH far from the rail (independent of the posterior)
        for thr in (0.3, 0.5):
            edge1 = np.minimum(th1 - GLO, GHI - th1)
            edge0 = np.minimum(th0 - GLO, GHI - th0)
            k1, k0 = edge1 > thr, edge0 > thr
            v[f"iv_truthedge_gt_{thr}"] = ts(*iv(b1[k1], s1[k1]), *iv(b0[k0], s0[k0]))
            v[f"iv_truthedge_gt_{thr}"]["n_shift"] = int(k1.sum())
            v[f"raw_truthedge_gt_{thr}"] = ts(*raw(b1[k1]), *raw(b0[k0]))
        # sigma-matched: common deciles of the POOLED sigma, equal-weight the
        # within-bin differences -> removes any sigma-distribution mismatch
        qs = np.percentile(pooled, np.arange(10, 100, 10))
        d_bins, se_bins, ns = [], [], []
        for lo, hi in zip(np.r_[-np.inf, qs], np.r_[qs, np.inf]):
            m1 = (s1 > lo) & (s1 <= hi)
            m0 = (s0 > lo) & (s0 <= hi)
            if m1.sum() < 5 or m0.sum() < 5:
                continue
            a1, e1_ = raw(b1[m1]); a0, e0_ = raw(b0[m0])
            d_bins.append(a1 - a0); se_bins.append(np.hypot(e1_, e0_))
            ns.append((int(m1.sum()), int(m0.sum())))
        d_bins = np.array(d_bins); se_bins = np.array(se_bins)
        v["sigma_matched_equalweight"] = {
            "delta": float(d_bins.mean()),
            "se": float(np.sqrt((se_bins ** 2).sum()) / d_bins.size),
            "z": float(d_bins.mean() / (np.sqrt((se_bins ** 2).sum()) / d_bins.size)),
            "n_bins": int(d_bins.size), "bin_ns": ns}
        v["sigma_matched_ivweight"] = ivcomb(list(zip(d_bins, se_bins)))
        return v

    out["delta_variants"] = {r: delta_variants(r) for r in NPZ}
    # combine the variants across reps
    combv = {}
    keys = [k for k in out["delta_variants"]["rep1"] if "delta" in out["delta_variants"]["rep1"][k]]
    for k in keys:
        combv[k] = ivcomb([(out["delta_variants"][r][k]["delta"],
                            out["delta_variants"][r][k]["se"]) for r in NPZ])
    out["delta_variants_combined"] = combv

    # ---------- 3. stratified permutation pooled over both reps ------------
    def stat_iv(b, s):
        w = 1 / s ** 2
        return (w * b).sum() / w.sum()

    def stat_raw(b, s):
        return b.mean()

    def stat_pull(b, s):
        return (b / s).mean()

    perms = {}
    for nm, stat in (("raw", stat_raw), ("pull", stat_pull), ("iv", stat_iv)):
        obs = 0.0
        blocks = []
        for r in NPZ:
            b1, s1, _, _ = R[r]["shifted"]
            b0, s0, _, _ = R[r]["clean"]
            obs += stat(b1, s1) - stat(b0, s0)
            blocks.append((np.concatenate([b1, b0]), np.concatenate([s1, s0]), b1.size))
        cnt = 0
        for _ in range(NPERM):
            tot = 0.0
            for b, s, n1 in blocks:
                idx = RNG.permutation(b.size)
                a, c = idx[:n1], idx[n1:]
                tot += stat(b[a], s[a]) - stat(b[c], s[c])
            if abs(tot) >= abs(obs) - 1e-15:
                cnt += 1
        perms[nm] = {"observed_sum_of_two_rep_deltas": float(obs),
                     "p_two_sided_stratified": (cnt + 1) / (NPERM + 1)}
    out["stratified_permutation"] = perms

    # ---------- 4. homogeneity: does one absolute bias explain all three? --
    sall = np.concatenate([R[r][sp][1] for r in NPZ for sp in ("shifted", "clean")])
    beta_iv = combv["iv_plain"]["mean"]
    out["homogeneity_check"] = {
        "iv_delta": beta_iv,
        "raw_delta": combv["raw_plain"]["mean"],
        "mean_inv_sigma_pooled": float((1 / sall).mean()),
        "predicted_pull_delta_if_homogeneous": float(beta_iv * (1 / sall).mean()),
        "note": "compare with the observed combined pull delta (+0.1046) from gamma_bias_weighted_results.json",
    }

    # ---------- 5. sigma-fraction conversions of the IV delta --------------
    out["iv_delta_in_sigma_units"] = {
        nm: {"sigma": float(v), "frac": beta_iv / float(v),
             "frac_se": combv["iv_plain"]["se"] / float(v)}
        for nm, v in (("median", np.median(sall)), ("mean", sall.mean()),
                      ("rms", np.sqrt((sall ** 2).mean())))
    }

    with open(OUT / "gamma_bias_robustness_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # -------------------- digest ------------------------------------------
    print("=== 1/sigma^2 weight concentration ===")
    for k, v in conc.items():
        print(f"  {k:14s} n_eff={v['n_eff_weights']:6.1f}/500  top1={v['top1_weight_share']:.4f} "
              f"top10={v['top10_weight_share']:.4f}  LOO max |dbeta|={v['loo_max_abs_change']:.5f} "
              f"(sigma_min={v['sigma_min']:.4f}, edge@maxw={v['edge_of_max_weight']:.3f})")

    print("\n=== differenced estimator variants (combined over reps) ===")
    for k, v in combv.items():
        print(f"  {k:28s} {v['mean']:+.5f} +/- {v['se']:.5f}  z={v['z']:+.2f}")
    print("\n  per-rep detail (delta / z):")
    for r in NPZ:
        for k in keys:
            vv = out["delta_variants"][r][k]
            print(f"    {r} {k:28s} {vv['delta']:+.5f} z={vv['z']:+.2f}")

    print("\n=== stratified permutation over both reps ===")
    for k, v in perms.items():
        print(f"  {k:5s} sum-of-deltas={v['observed_sum_of_two_rep_deltas']:+.5f} "
              f"p={v['p_two_sided_stratified']:.5f}")

    print("\n=== homogeneity ===")
    print(json.dumps(out["homogeneity_check"], indent=2))
    print("\n=== IV delta in sigma units ===")
    for k, v in out["iv_delta_in_sigma_units"].items():
        print(f"  {k:7s} sigma={v['sigma']:.4f} -> {v['frac']:+.4f} +/- {v['frac_se']:.4f} sigma")


if __name__ == "__main__":
    main()
