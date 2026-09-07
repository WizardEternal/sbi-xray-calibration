"""Gate check + paired-SE analysis for the 2026-09-03 per-source npz runs.
Read-only: writes only outputs/gain_marg/paired_per_source_2026-09-03/analysis.json.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs" / "gain_marg" / "paired_per_source_2026-09-03"

RUNS = {
    "bright_uncapped": {
        "new_json": BASE / "bright_uncapped" / "paired_gain_bias_bright.json",
        "new_npz": BASE / "bright_uncapped" / "paired_gain_bias_bright_per_source.npz",
        "committed_json": ROOT / "outputs" / "uncapped_bright" / "gain_marg" / "paired_gain_bias_bright.json",
    },
    "bright_production": {
        "new_json": BASE / "bright_production" / "paired_gain_bias_bright.json",
        "new_npz": BASE / "bright_production" / "paired_gain_bias_bright_per_source.npz",
        "committed_json": ROOT / "outputs" / "gain_marg" / "paired_gain_bias_bright.json",
    },
    "medium": {
        "new_json": BASE / "medium" / "paired_gain_bias_medium.json",
        "new_npz": BASE / "medium" / "paired_gain_bias_medium_per_source.npz",
        "committed_json": ROOT / "outputs" / "gain_marg" / "paired_gain_bias_medium.json",
    },
}

Z95 = 1.959963985


def gate_compare(new_j, committed_j):
    """Compare aggregate bias means (gamma, log10norm-delta-mean via gamma_bias_delta
    and log10norm_bias_delta) between new run and committed json. Report max abs delta."""
    out = {}
    for stat in ["gamma_bias_delta", "log10norm_bias_delta"]:
        for arm in ["fixed", "gain_marg"]:
            new_v = new_j["paired"][stat][arm]
            old_v = committed_j["paired"][stat][arm]
            out[f"{stat}.{arm}.mean"] = {
                "new": new_v["mean"], "committed": old_v["mean"],
                "delta": new_v["mean"] - old_v["mean"],
            }
            out[f"{stat}.{arm}.se"] = {
                "new": new_v["se"], "committed": old_v["se"],
                "delta": new_v["se"] - old_v["se"],
            }
    # also per-case bias_mean for gamma/norm, fixed/gain_marg, clean/gain
    for gcase in ["clean", "gain"]:
        for arm in ["fixed", "gain_marg"]:
            for stat in ["gamma", "norm"]:
                new_v = new_j["cases"][gcase][arm][stat]["bias_mean"]
                old_v = committed_j["cases"][gcase][arm][stat]["bias_mean"]
                out[f"cases.{gcase}.{arm}.{stat}.bias_mean"] = {
                    "new": new_v, "committed": old_v, "delta": new_v - old_v,
                }
    return out


def paired_analysis(npz_path):
    d = np.load(npz_path)
    out = {}
    for stat in ["gamma_bias", "log10norm_bias"]:
        for gcase in ["clean", "gain"]:
            f = d[f"fixed_{gcase}_{stat}"]
            g = d[f"gainmarg_{gcase}_{stat}"]
            diff = f - g
            n = diff.shape[0]
            mean_d = float(diff.mean())
            sd_d = float(diff.std(ddof=1))
            se_paired = sd_d / np.sqrt(n)
            corr = float(np.corrcoef(f, g)[0, 1])
            se_f = float(f.std(ddof=1) / np.sqrt(n))
            se_g = float(g.std(ddof=1) / np.sqrt(n))
            se_naive = float(np.sqrt(se_f**2 + se_g**2))
            t_paired = mean_d / se_paired if se_paired > 0 else float("nan")
            two_sided_p = float(2 * (1 - stats.norm.cdf(abs(t_paired))))
            out[f"{stat}.{gcase}"] = {
                "n": n, "mean_diff": mean_d, "sd_diff": sd_d,
                "se_paired": se_paired, "se_naive": se_naive,
                "corr_f_g": corr, "se_fixed": se_f, "se_gainmarg": se_g,
                "t_paired": t_paired, "p_paired_2sided": two_sided_p,
                "ratio_paired_over_naive": se_paired / se_naive if se_naive > 0 else float("nan"),
            }
    return out


def main():
    report = {"gate": {}, "paired": {}, "seed_pairing_check": {}, "fixed_vs_fixed_bright": {}}

    for name, paths in RUNS.items():
        new_j = json.loads(paths["new_json"].read_text())
        committed_j = json.loads(paths["committed_json"].read_text())
        report["gate"][name] = gate_compare(new_j, committed_j)
        report["paired"][name] = paired_analysis(paths["new_npz"])

    # seed / theta pairing check across the two bright runs
    du = np.load(RUNS["bright_uncapped"]["new_npz"])
    dp = np.load(RUNS["bright_production"]["new_npz"])
    same_seed_theta = int(du["seed_theta"]) == int(dp["seed_theta"])
    same_seed_poisson = int(du["seed_poisson_base"]) == int(dp["seed_poisson_base"])
    same_theta_array = bool(np.array_equal(du["theta"], dp["theta"]))
    report["seed_pairing_check"] = {
        "same_seed_theta": same_seed_theta,
        "same_seed_poisson_base": same_seed_poisson,
        "theta_arrays_bytewise_equal": same_theta_array,
        "seed_theta_uncapped": int(du["seed_theta"]), "seed_theta_production": int(dp["seed_theta"]),
    }

    # fixed-vs-fixed: uncapped minus production, same spectra (bright), per stat, per case
    if same_theta_array:
        for stat in ["gamma_bias", "log10norm_bias"]:
            for gcase in ["clean", "gain"]:
                u = du[f"fixed_{gcase}_{stat}"]
                p = dp[f"fixed_{gcase}_{stat}"]
                diff = u - p
                n = diff.shape[0]
                mean_d = float(diff.mean())
                sd_d = float(diff.std(ddof=1))
                se_paired = sd_d / np.sqrt(n)
                t_paired = mean_d / se_paired if se_paired > 0 else float("nan")
                two_sided_p = float(2 * (1 - stats.norm.cdf(abs(t_paired))))
                corr = float(np.corrcoef(u, p)[0, 1])
                report["fixed_vs_fixed_bright"][f"{stat}.{gcase}"] = {
                    "n": n, "mean_diff_uncapped_minus_production": mean_d,
                    "sd_diff": sd_d, "se_paired": se_paired, "corr": corr,
                    "t_paired": t_paired, "p_paired_2sided": two_sided_p,
                }

    out_path = BASE / "analysis.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[written] {out_path}")

    # print a compact summary
    print("\n=== GATE (aggregate JSON new vs committed; flag |delta|>5e-4) ===")
    for name, g in report["gate"].items():
        print(f"-- {name} --")
        for k, v in g.items():
            if "mean" in k:
                flag = " <<<< FLAG" if abs(v["delta"]) > 5e-4 else ""
                print(f"  {k}: new={v['new']:+.6f} committed={v['committed']:+.6f} delta={v['delta']:+.6f}{flag}")

    print("\n=== SEED/THETA PAIRING (bright uncapped vs production) ===")
    print(report["seed_pairing_check"])

    print("\n=== PAIRED SE (fixed - gainmarg), per run ===")
    for name, p in report["paired"].items():
        print(f"-- {name} --")
        for k, v in p.items():
            print(f"  {k}: mean_d={v['mean_diff']:+.5f} se_paired={v['se_paired']:.5f} "
                  f"se_naive={v['se_naive']:.5f} corr={v['corr_f_g']:.3f} t={v['t_paired']:+.2f} "
                  f"p={v['p_paired_2sided']:.4f}")

    print("\n=== FIXED vs FIXED (bright, uncapped - production) ===")
    for k, v in report["fixed_vs_fixed_bright"].items():
        print(f"  {k}: mean_d={v['mean_diff_uncapped_minus_production']:+.5f} "
              f"se_paired={v['se_paired']:.5f} corr={v['corr']:.3f} t={v['t_paired']:+.2f} "
              f"p={v['p_paired_2sided']:.4f}")


if __name__ == "__main__":
    main()
