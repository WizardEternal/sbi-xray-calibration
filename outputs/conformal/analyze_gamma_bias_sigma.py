"""Photon-index bias of the conformal populations, in absolute units and in
units of the posterior width.

Why this exists
---------------
Standardizing a population bias by a per-spectrum posterior width is not the
same operation as dividing the population bias by a typical width, and the
two differ by a factor of about 2.7 on this population. The first reading
implies sigma(Gamma) ~ 0.10, while a direct measurement of the same medium
flow gives 0.295-0.30, so the two have to be separated explicitly.

The quantity `run_conformal.py` stores as `bias_mean_in_post_std` is

    (1/N) * sum_i [ (mu_i - theta_i) / sigma_i ]        (mean of ratios)

which is NOT

    [ (1/N) * sum_i (mu_i - theta_i) ] / sigma_typ      (ratio of means)

The two differ by ~2.7x here, and "implied sigma" = mean(b) / mean(b/sigma) is
a meaningless quantity (it goes NEGATIVE for the clean-control split, where
mean(b) < 0 but mean(b/sigma) > 0).

What it computes
----------------
Per replication (rep1 seed base 20260611, rep2 seed base 20260723) and per
split (shifted g=1.03, clean control g=1.00), for all 5 parameters and in
detail for Gamma = powerlaw_1_alpha:

  * the per-spectrum posterior sigma distribution (median / mean / rms /
    [16,84] percentiles / min / max)
  * the raw bias mean +/- SE (posterior mean - truth), absolute units
  * BOTH standardizations: mean(b)/median(sigma) [constant denominator] and
    mean(b/sigma) [per-spectrum, the original], plus their z-scores
  * paired footing: shifted minus clean control, raw and in sigma units
  * sigma-tercile breakdown, to test whether the per-spectrum standardization
    is concentrating real signal or manufacturing it
  * bootstrap CIs for both estimators

Two independent paths, both run:
  (A) STORED: read means/stds/truths straight out of the two npz files the
      original runs wrote.
  (B) REGENERATED: re-simulate the populations with the same seeds and re-sample
      the NPE posterior from scratch (same checkpoint, same
      sample_posterior_batch, reject_outside_prior=True), then recompute
      everything. Path B is the from-scratch axis; it is compared to path A.

Run (repo venv, from the repo root):
    .venv\\Scripts\\python.exe outputs\\conformal\\analyze_gamma_bias_sigma.py
    .venv\\Scripts\\python.exe outputs\\conformal\\analyze_gamma_bias_sigma.py --stored-only
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "outputs" / "conformal"
OUT = CONF / "sigma_replication"
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run_log.txt"

PHYS_ORDER = ["tbabs_1_nh", "powerlaw_1_alpha", "powerlaw_1_norm",
              "blackbodyrad_1_kT", "blackbodyrad_1_norm"]
GAMMA_I = 1

# ---- exact config copied from the two original drivers (verified by reading
# them; see the report). Kept here so path B is standalone. ----
RESPONSE_NAME = "NGC7793_ULX4_PN"
EXPOSURE_S = 353.4
BASE_MODEL = "tbabs_powerlaw_bb"
PHYS_PRIORS = {
    "tbabs_1_nh":          {"dist": "uniform",    "low": 0.15,   "high": 0.35},
    "powerlaw_1_alpha":    {"dist": "uniform",    "low": 1.0,    "high": 3.0},
    "powerlaw_1_norm":     {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":   {"dist": "uniform",    "low": 0.3,    "high": 3.0},
    "blackbodyrad_1_norm": {"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}
N = 500
N_SAMPLES = 1000
FIXED_DIR = ROOT / "outputs" / "models" / "train_npe_prod_medium"

REPS = {
    "rep1": {"seed_base": 20260611, "npz": CONF / "conformal_coverage_curves.npz",
             "json": CONF / "conformal_results.json"},
    "rep2": {"seed_base": 20260723, "npz": CONF / "rep2" / "conformal_coverage_curves_rep2.npz",
             "json": CONF / "rep2" / "conformal_results_rep2.json"},
}
# offsets are identical in both drivers
OFF = {"cal": 811, "shift": 823, "cleanctl": 831,
       "samp_cal": 901, "samp_shift": 902, "samp_cleanctl": 903}

BOOT = 20000
BOOT_SEED = 20260814


def log_state(line: str) -> None:
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
    print(line)


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------

def sigma_summary(sig: np.ndarray) -> dict:
    return {
        "n": int(sig.size),
        "median": float(np.median(sig)),
        "mean": float(sig.mean()),
        "rms": float(np.sqrt((sig ** 2).mean())),
        "p16": float(np.percentile(sig, 16)),
        "p84": float(np.percentile(sig, 84)),
        "min": float(sig.min()),
        "max": float(sig.max()),
    }


def split_stats(mu: np.ndarray, sig: np.ndarray, th: np.ndarray, rng) -> dict:
    """All bias statistics for one (split, parameter)."""
    b = mu.astype(np.float64) - th.astype(np.float64)
    s = sig.astype(np.float64)
    n = b.size
    mean_b = float(b.mean())
    se_b = float(b.std(ddof=1) / np.sqrt(n))
    med_s = float(np.median(s))
    mean_s = float(s.mean())

    r = b / s                                    # the ORIGINAL per-spectrum ratio
    mean_r = float(r.mean())
    se_r = float(r.std(ddof=1) / np.sqrt(n))

    # bootstrap (paired resample of spectra) for both estimators
    idx = rng.integers(0, n, size=(BOOT, n))
    bb = b[idx]
    ss = s[idx]
    boot_const = bb.mean(axis=1) / np.median(ss, axis=1)
    boot_ratio = (bb / ss).mean(axis=1)

    return {
        "n": n,
        # --- raw, absolute Gamma units ---
        "bias_mean_abs": mean_b,
        "bias_se_abs": se_b,
        "bias_median_abs": float(np.median(b)),
        "bias_z_abs": mean_b / se_b,
        # --- sigma ---
        "sigma": sigma_summary(s),
        # --- CORRECTED: constant denominator (median sigma) ---
        "std_const_median": mean_b / med_s,
        "std_const_median_se": se_b / med_s,
        "std_const_mean": mean_b / mean_s,
        "std_const_z": mean_b / se_b,           # identical to bias_z_abs, by construction
        "std_const_boot_ci95": [float(np.percentile(boot_const, 2.5)),
                                float(np.percentile(boot_const, 97.5))],
        # --- ORIGINAL: per-spectrum denominator (mean of ratios) ---
        "std_perspec_mean": mean_r,
        "std_perspec_se": se_r,
        "std_perspec_median": float(np.median(r)),
        "std_perspec_z": mean_r / se_r,
        "std_perspec_boot_ci95": [float(np.percentile(boot_ratio, 2.5)),
                                  float(np.percentile(boot_ratio, 97.5))],
        # --- what the mean-of-ratios form implies about the denominator ---
        "implied_sigma_from_perspec": (mean_b / mean_r) if mean_r != 0 else float("nan"),
        "perspec_over_const_ratio": mean_r / (mean_b / med_s) if mean_b != 0 else float("nan"),
    }


def tercile_breakdown(mu, sig, th) -> dict:
    """Is the per-spectrum standardization concentrating real signal, or
    manufacturing it? Split spectra by posterior width and report the raw bias
    in each tercile."""
    b = mu.astype(np.float64) - th.astype(np.float64)
    s = sig.astype(np.float64)
    q = np.percentile(s, [100 / 3, 200 / 3])
    out = {}
    for name, m in (("tight", s <= q[0]),
                    ("mid", (s > q[0]) & (s <= q[1])),
                    ("wide", s > q[1])):
        bi, si = b[m], s[m]
        out[name] = {
            "n": int(m.sum()),
            "sigma_median": float(np.median(si)),
            "bias_mean_abs": float(bi.mean()),
            "bias_se_abs": float(bi.std(ddof=1) / np.sqrt(m.sum())),
            "bias_over_median_sigma": float(bi.mean() / np.median(si)),
            "mean_ratio": float((bi / si).mean()),
        }
    return out


def leverage(mu, sig, th) -> dict:
    """How much of mean(b_i/sigma_i) is carried by the narrowest posteriors?
    Sort spectra by sigma, report the cumulative contribution of the tightest
    decile / quartile to the total sum of ratios."""
    b = mu.astype(np.float64) - th.astype(np.float64)
    s = sig.astype(np.float64)
    r = b / s
    n = r.size
    order = np.argsort(s)
    tot = r.sum()
    out = {"total_mean_ratio": float(tot / n)}
    for frac in (0.1, 0.25, 0.5):
        k = int(round(frac * n))
        out[f"tightest_{int(frac*100)}pct_share_of_sum"] = float(r[order[:k]].sum() / tot)
        out[f"tightest_{int(frac*100)}pct_share_of_raw_sum"] = float(
            b[order[:k]].sum() / b.sum()) if b.sum() != 0 else float("nan")
    return out


def two_sample_z(m1, se1, m2, se2):
    d = m1 - m2
    sd = float(np.hypot(se1, se2))
    return {"delta": float(d), "se": sd, "z": float(d / sd)}


def inverse_variance_combine(pairs):
    """pairs: list of (mean, se). Returns combined mean, se, z."""
    w = np.array([1.0 / se ** 2 for _, se in pairs])
    m = np.array([mu for mu, _ in pairs])
    mc = float((w * m).sum() / w.sum())
    sec = float(1.0 / np.sqrt(w.sum()))
    return {"mean": mc, "se": sec, "z": mc / sec}


# ----------------------------------------------------------------------
# path B: regenerate from scratch
# ----------------------------------------------------------------------

def regenerate(seed_base: int):
    """Re-simulate the populations and re-sample the NPE posterior from
    scratch, using exactly the original drivers' seeds. Returns
    {split: (means, stds, truths)}."""
    from sbixcal import responses as R
    from sbixcal import train_npe as tn
    from sbixcal import calibrate as Cc
    from sbixcal.misspec import simulate_misspec_population

    post, info = tn.load_posterior(FIXED_DIR, device="cpu")
    assert info["param_names"] == PHYS_ORDER, info["param_names"]

    base = R.load_base_obsconf(RESPONSE_NAME)
    oc = R.scale_exposure(base, EXPOSURE_S)

    def pop(strength, seed):
        x, theta, present = simulate_misspec_population(
            BASE_MODEL, PHYS_PRIORS, oc, "B4", float(strength), N, seed)
        assert present == PHYS_ORDER, present
        return np.asarray(x, dtype=np.float32), np.asarray(theta, dtype=np.float64)

    out = {}
    for split, (strength, off_pop, off_samp) in (
            ("shifted", (3.0, OFF["shift"], OFF["samp_shift"])),
            ("clean_control", (0.0, OFF["cleanctl"], OFF["samp_cleanctl"]))):
        t0 = time.time()
        x, th = pop(strength, seed_base + off_pop)
        s = Cc.sample_posterior_batch(post, x, N_SAMPLES,
                                      seed=seed_base + off_samp, device="cpu")
        means = np.stack([a.mean(axis=0) for a in s], axis=0)
        stds = np.stack([a.std(axis=0) for a in s], axis=0)
        out[split] = (means, stds, th)
        print(f"    [regen] {split}: {time.time()-t0:.1f}s "
              f"median counts={np.median(x.sum(1)):.0f}")
    return out


# ----------------------------------------------------------------------

def main():
    stored_only = "--stored-only" in sys.argv
    rng = np.random.default_rng(BOOT_SEED)
    results = {"generated": "2026-08-14", "boot_draws": BOOT, "boot_seed": BOOT_SEED,
               "stored": {}, "regenerated": {}, "cross_checks": {}}
    arrays = {}

    log_state("")
    log_state(f"- [run] started {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"(stored_only={stored_only})")

    # ---------------- path A: stored arrays ----------------
    for rep, cfg in REPS.items():
        d = np.load(cfg["npz"], allow_pickle=True)
        orig = json.load(open(cfg["json"]))
        # verify the seeds the driver actually used match the brief
        sb = cfg["seed_base"]
        assert orig["config"]["seed_shift"] == sb + OFF["shift"], (rep, orig["config"])
        assert orig["config"]["seed_cleanctl"] == sb + OFF["cleanctl"], (rep, orig["config"])
        assert orig["config"]["seed_samp_shift"] == sb + OFF["samp_shift"]
        assert orig["config"]["seed_samp_cleanctl"] == sb + OFF["samp_cleanctl"]

        rr = {"seed_base": sb, "splits": {}}
        for split, (km, ks, kt) in (
                ("shifted", ("means_shift", "stds_shift", "th_shift")),
                ("clean_control", ("means_clean", "stds_clean", "th_cleanctl"))):
            mu, sg, th = d[km], d[ks], d[kt]
            arrays[f"{rep}_{split}_means"] = mu
            arrays[f"{rep}_{split}_stds"] = sg
            arrays[f"{rep}_{split}_truths"] = th
            per_param = {}
            for p, name in enumerate(PHYS_ORDER):
                per_param[name] = split_stats(mu[:, p], sg[:, p], th[:, p], rng)
            rr["splits"][split] = {
                "per_param": per_param,
                "gamma_terciles": tercile_breakdown(mu[:, GAMMA_I], sg[:, GAMMA_I],
                                                    th[:, GAMMA_I]),
                "gamma_leverage": leverage(mu[:, GAMMA_I], sg[:, GAMMA_I], th[:, GAMMA_I]),
            }
            # reproduce the ORIGINAL json numbers exactly, as a wiring check
            og = orig["gamma_bias"]["shifted" if split == "shifted" else "clean_control"]
            g = per_param["powerlaw_1_alpha"]
            rr["splits"][split]["original_json_match"] = {
                "bias_mean_abs": [og["bias_mean_abs"], g["bias_mean_abs"],
                                  abs(og["bias_mean_abs"] - g["bias_mean_abs"])],
                "bias_mean_in_post_std_vs_perspec": [
                    og["bias_mean_in_post_std"], g["std_perspec_mean"],
                    abs(og["bias_mean_in_post_std"] - g["std_perspec_mean"])],
                "median_posterior_std": [og["median_posterior_std"],
                                         g["sigma"]["median"],
                                         abs(og["median_posterior_std"] - g["sigma"]["median"])],
            }
        # paired footing: shifted minus clean control (independent pops -> 2-sample)
        gs = rr["splits"]["shifted"]["per_param"]["powerlaw_1_alpha"]
        gc = rr["splits"]["clean_control"]["per_param"]["powerlaw_1_alpha"]
        sig_pool = float(np.median(np.concatenate(
            [arrays[f"{rep}_shifted_stds"][:, GAMMA_I],
             arrays[f"{rep}_clean_control_stds"][:, GAMMA_I]]).astype(np.float64)))
        raw_pair = two_sample_z(gs["bias_mean_abs"], gs["bias_se_abs"],
                                gc["bias_mean_abs"], gc["bias_se_abs"])
        std_pair = two_sample_z(gs["std_perspec_mean"], gs["std_perspec_se"],
                                gc["std_perspec_mean"], gc["std_perspec_se"])
        rr["gamma_paired_shifted_minus_clean"] = {
            "pooled_median_sigma": sig_pool,
            "raw": raw_pair,
            "raw_in_sigma_units": {"delta": raw_pair["delta"] / sig_pool,
                                   "se": raw_pair["se"] / sig_pool,
                                   "z": raw_pair["z"]},
            "perspec_standardized": std_pair,
        }
        # is the tight-posterior sub-population where the shift signature lives?
        ts = rr["splits"]["shifted"]["gamma_terciles"]
        tc = rr["splits"]["clean_control"]["gamma_terciles"]
        rr["gamma_tercile_paired"] = {
            k: two_sample_z(ts[k]["bias_mean_abs"], ts[k]["bias_se_abs"],
                            tc[k]["bias_mean_abs"], tc[k]["bias_se_abs"])
            for k in ("tight", "mid", "wide")}
        results["stored"][rep] = rr

    # ---------------- cross-rep combination ----------------
    g1 = results["stored"]["rep1"]["gamma_paired_shifted_minus_clean"]
    g2 = results["stored"]["rep2"]["gamma_paired_shifted_minus_clean"]
    results["cross_checks"]["paired_raw_combined"] = inverse_variance_combine(
        [(g1["raw"]["delta"], g1["raw"]["se"]), (g2["raw"]["delta"], g2["raw"]["se"])])
    results["cross_checks"]["paired_perspec_combined"] = inverse_variance_combine(
        [(g1["perspec_standardized"]["delta"], g1["perspec_standardized"]["se"]),
         (g2["perspec_standardized"]["delta"], g2["perspec_standardized"]["se"])])
    sig_all = np.concatenate([arrays[f"{r}_{s}_stds"][:, GAMMA_I]
                              for r in REPS for s in ("shifted", "clean_control")]).astype(np.float64)
    results["cross_checks"]["sigma_gamma_all_2000_spectra"] = sigma_summary(sig_all)
    results["cross_checks"]["paired_raw_combined_in_sigma"] = (
        results["cross_checks"]["paired_raw_combined"]["mean"] / float(np.median(sig_all)))
    # The 0.295-0.30 anchor is the mean per-spectrum sigma (our 2000-spectrum
    # mean is 0.3028), not the median (0.2781). Report both.
    results["cross_checks"]["paired_raw_combined_in_mean_sigma"] = (
        results["cross_checks"]["paired_raw_combined"]["mean"] / float(sig_all.mean()))
    # prior-rail leverage: sigma_i is small precisely where the posterior is
    # truncated against the Gamma prior box [1, 3] (reject_outside_prior=True),
    # so mean(b_i/sigma_i) up-weights spectra pinned against the prior rail.
    rail = {}
    for rep in REPS:
        for split in ("shifted", "clean_control"):
            mu = arrays[f"{rep}_{split}_means"][:, GAMMA_I].astype(np.float64)
            sg = arrays[f"{rep}_{split}_stds"][:, GAMMA_I].astype(np.float64)
            edge = np.minimum(mu - 1.0, 3.0 - mu)
            tight = sg <= np.percentile(sg, 100 / 3)
            rail[f"{rep}_{split}"] = {
                "corr_sigma_vs_dist_to_prior_edge": float(np.corrcoef(sg, edge)[0, 1]),
                "frac_within_0.3_of_edge_tight_tercile": float((edge[tight] < 0.3).mean()),
                "frac_within_0.3_of_edge_all": float((edge < 0.3).mean()),
            }
    results["cross_checks"]["gamma_prior_rail_diagnostic"] = rail
    results["cross_checks"]["external_anchors"] = {
        "paired_measurement_sigma_gamma": [0.295, 0.30],
        "paired_measurement_raw_delta": [0.018, 0.023],
        "paired_measurement_sigma_fraction_medium": [0.05, 0.08],
    }

    log_state("- [path A] stored-array statistics computed for both reps, both splits.")
    for rep in REPS:
        for split in ("shifted", "clean_control"):
            g = results["stored"][rep]["splits"][split]["per_param"]["powerlaw_1_alpha"]
            log_state(f"    {rep}/{split}: raw={g['bias_mean_abs']:+.5f}+/-{g['bias_se_abs']:.5f} "
                      f"(z={g['bias_z_abs']:+.3f})  sigma med={g['sigma']['median']:.4f} "
                      f"mean={g['sigma']['mean']:.4f}  const-std={g['std_const_median']:+.4f}  "
                      f"perspec-std={g['std_perspec_mean']:+.4f} (z={g['std_perspec_z']:+.3f})")

    # ---------------- path B: regenerate ----------------
    if not stored_only:
        for rep, cfg in REPS.items():
            print(f"  [regen] {rep} (seed base {cfg['seed_base']})")
            regen = regenerate(cfg["seed_base"])
            rr = {"seed_base": cfg["seed_base"], "splits": {}, "vs_stored": {}}
            for split, (mu, sg, th) in regen.items():
                arrays[f"{rep}_{split}_means_REGEN"] = mu
                arrays[f"{rep}_{split}_stds_REGEN"] = sg
                arrays[f"{rep}_{split}_truths_REGEN"] = th
                rr["splits"][split] = {
                    "powerlaw_1_alpha": split_stats(mu[:, GAMMA_I], sg[:, GAMMA_I],
                                                    th[:, GAMMA_I], rng)}
                mu0 = arrays[f"{rep}_{split}_means"]
                sg0 = arrays[f"{rep}_{split}_stds"]
                th0 = arrays[f"{rep}_{split}_truths"]
                rr["vs_stored"][split] = {
                    "truths_max_abs_diff": float(np.abs(th - th0).max()),
                    "means_max_abs_diff": float(np.abs(mu - mu0).max()),
                    "stds_max_abs_diff": float(np.abs(sg - sg0).max()),
                    "gamma_sigma_median_regen": float(np.median(sg[:, GAMMA_I])),
                    "gamma_sigma_median_stored": float(np.median(sg0[:, GAMMA_I])),
                }
            results["regenerated"][rep] = rr
            log_state(f"- [path B] {rep} regenerated. "
                      f"max|truth diff|="
                      f"{max(v['truths_max_abs_diff'] for v in rr['vs_stored'].values()):.2e}, "
                      f"max|std diff|="
                      f"{max(v['stds_max_abs_diff'] for v in rr['vs_stored'].values()):.2e}")

    # ---------------- persist ----------------
    with open(OUT / "gamma_bias_sigma_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(OUT / "gamma_bias_sigma_arrays.npz", **arrays)
    log_state(f"- [persist] -> {OUT / 'gamma_bias_sigma_results.json'} and "
              f"{OUT / 'gamma_bias_sigma_arrays.npz'}")
    return results


if __name__ == "__main__":
    main()
