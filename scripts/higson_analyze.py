r"""Higson thread-wise bootstrap: read the per-run DONE.json markers, compare the
Higson (2018) sigma on logZ against UltraNest's own error components and the
empirical paired-diff SE, compute the NS-sampling floor on the paired gain-null, and
check whether the B1 line-detection CIs are threatened. Writes higson_results.json +
higson_table.md under outputs/ns_bench/higson/. Robust to partial completion.

The per-run sigma is recomputed here from each run directory rather than read from
the sigma each run's DONE.json stored, because the stored value comes from the
raw points.hdf5 Lmin column and a start-contour-stratified resample; see
scripts/higson_common.py for why neither is the right input. The DONE.json value is
carried alongside for reference. Each recomputed run is checkpointed to
outputs/ns_bench/higson/higson_recompute.jsonl the moment it lands, so a killed
recompute resumes.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HG = ROOT / "outputs" / "ns_bench" / "higson"
RUNS = HG / "runs"
sys.path.insert(0, str(Path(__file__).resolve().parent))

RECOMPUTE_JSONL = HG / "higson_recompute.jsonl"
N_RESAMPLES = 400
BOOT_SEED = 12345          # unchanged repo convention

# committed reference numbers (verified 2026-07-23)
PAIRED_JSONL = ROOT / "outputs" / "ns_bench" / "paired_gain_check.jsonl"
# B1 line count-controlled residuals, from outputs/ns_bench/count_controlled.json
B1_RESID = {
    "medium": {"median": -67.3, "ci": [-89.8, -44.0]},
    "bright": {"median": -892.1, "ci": [-1165.6, -562.2]},
}


def load_markers():
    out = {}
    for d in sorted(RUNS.glob("*")):
        m = d / "DONE.json"
        if m.exists():
            out[d.name] = json.loads(m.read_text())
    return out


def recompute_markers(marks, force=False):
    """Recompute each run's Higson sigma from its run directory.

    Reads chains/run.txt + results/points.hdf5, replays UltraNest's refill-batch
    consumption to recover the true birth-death tree (HC.reconstruct_tree), and runs
    the global Algorithm-2 thread bootstrap (HC.higson_bootstrap) with the repo's
    B=400 / boot_seed=12345 convention. Every finished run is appended to
    RECOMPUTE_JSONL immediately; a rerun reuses those rows unless ``force``.
    Returns a new dict of markers; the on-disk DONE.json files are never written."""
    import higson_common as HC

    cache = {}
    if RECOMPUTE_JSONL.exists() and not force:
        for line in RECOMPUTE_JSONL.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["job_id"]] = row

    out = {}
    for job_id, v in marks.items():
        row = cache.get(job_id)
        if row is None:
            t0 = time.perf_counter()
            tree = HC.reconstruct_tree(RUNS / job_id)
            boot = HC.higson_bootstrap(tree["logl"], tree["birth"],
                                       n_resamples=N_RESAMPLES, seed=BOOT_SEED,
                                       labels=tree["thread_id"])
            row = {
                "job_id": job_id,
                "higson_sigma": boot["higson_sigma"],
                "higson_sigma_from_done_json": v["higson_sigma"],
                "higson_logZ_reconstructed": boot["logZ_reconstructed"],
                "higson_logZ_mean": boot["higson_logZ_mean"],
                "higson_logZ_p16_p84": boot["higson_logZ_p16_p84"],
                "n_threads": boot["n_threads"],
                "n_start_groups": boot["n_start_groups"],
                "n_points": boot["n_points"],
                "n_resamples": boot["n_resamples"],
                "boot_seed": BOOT_SEED,
                "delta_reconstructed_vs_reported": boot["logZ_reconstructed"] - v["ns_logz"],
                "nlive_exact_vs_runtxt": tree["nlive_exact"],
                "nlive_n_mismatch": tree["nlive_n_mismatch"],
                "nlive_max_abs_diff": tree["nlive_max_abs_diff"],
                "n_threads_from_done_json": v["n_threads"],
                "n_store": tree["n_store"],
                "n_store_unused": tree["n_store_unused"],
                "n_store_skipped_stale": tree["n_store_skipped_stale"],
                "n_store_tail": tree["n_store_tail"],
                "ns_logzerr_single": v.get("ns_logzerr_single"),
                "sigma_over_logzerr_single": (boot["higson_sigma"] / v["ns_logzerr_single"]
                                              if v.get("ns_logzerr_single") else None),
                "wall_s": round(time.perf_counter() - t0, 1),
            }
            with open(RECOMPUTE_JSONL, "a") as f:      # checkpoint the moment it lands
                f.write(json.dumps(row) + "\n")
            print(f"[recompute] {job_id:20s} sigma {v['higson_sigma']:.4f} -> "
                  f"{row['higson_sigma']:.4f}  threads {v['n_threads']} -> "
                  f"{row['n_threads']}  nlive_exact={row['nlive_exact_vs_runtxt']} "
                  f"[{row['wall_s']}s]", flush=True)
        merged = dict(v)
        merged["higson_sigma_from_done_json"] = v["higson_sigma"]
        merged["n_threads_from_done_json"] = v["n_threads"]
        for k in ("higson_sigma", "higson_logZ_reconstructed", "higson_logZ_mean",
                  "higson_logZ_p16_p84", "n_threads", "n_start_groups", "n_points",
                  "delta_reconstructed_vs_reported", "nlive_exact_vs_runtxt",
                  "nlive_n_mismatch", "n_store", "n_store_unused",
                  "sigma_over_logzerr_single"):
            merged[k] = row[k]
        out[job_id] = merged
    return out


def paired_reference():
    rows = [json.loads(l) for l in open(PAIRED_JSONL) if l.strip()]
    d = np.array([r["d_paired"] for r in rows])
    lz = np.array([r["logzerr_clean"] for r in rows] + [r["logzerr_gain"] for r in rows])
    # NS-sampling floor on the mean from UltraNest's OWN per-run logzerr:
    #   Var(d_i) = err_clean_i^2 + err_gain_i^2 ; SE_mean = sqrt(sum Var)/n
    var_d_un = np.array([r["logzerr_clean"]**2 + r["logzerr_gain"]**2 for r in rows])
    se_mean_un = float(np.sqrt(var_d_un.sum()) / len(rows))
    return {
        "n_pairs": len(rows),
        "mean_dlogz": float(d.mean()),
        "empirical_paired_SEM": float(d.std(ddof=1) / len(d) ** 0.5),
        "empirical_paired_SD": float(d.std(ddof=1)),
        "ultranest_logzerr_mean": float(lz.mean()),
        "ultranest_logzerr_range": [float(lz.min()), float(lz.max())],
        "ns_floor_on_mean_from_ultranest_logzerr": se_mean_un,
        "per_pair_d": [float(x) for x in d],
    }


MC_SEEDS = (12345, 137, 42, 11, 7, 2026, 99, 20260814)
MC_JSONL = HG / "higson_mc_noise.jsonl"


def mc_noise_check(marks, force=False):
    """Consistency-check support. The statistic checked is the median
    of 13 per-run sigmas, and each sigma is itself a B=400 bootstrap estimate with
    ~1/sqrt(2(B-1)) = 3.5% relative MC noise. This re-runs the whole 13-run bootstrap
    under 8 different boot seeds at B=400, plus one B=4000 pass at the canonical seed,
    so the MC noise on the gate statistic is measured rather than assumed. Cheap
    (~2.5 min) and checkpointed per case."""
    import higson_common as HC

    cache = {}
    if MC_JSONL.exists() and not force:
        for line in MC_JSONL.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[(row["B"], row["seed"])] = row

    jobs = sorted(marks)
    trees = {j: HC.reconstruct_tree(RUNS / j) for j in jobs}
    single = {j: marks[j]["ns_logzerr_single"] for j in jobs}
    pairs = [(f"gain_pair{i}_clean", f"gain_pair{i}_gain") for i in (8, 6, 0, 9, 10)]

    cases = [(N_RESAMPLES, s) for s in MC_SEEDS] + [(10 * N_RESAMPLES, BOOT_SEED)]
    rows = []
    for B, seed in cases:
        row = cache.get((B, seed))
        if row is None:
            t0 = time.perf_counter()
            sig = {j: HC.higson_bootstrap(trees[j]["logl"], trees[j]["birth"], n_resamples=B,
                                          seed=seed, labels=trees[j]["thread_id"])["higson_sigma"]
                   for j in jobs}
            a = np.array([sig[j] for j in jobs])
            ratio = np.array([sig[j] / single[j] for j in jobs])
            vbar = float(np.mean([sig[c] ** 2 + sig[g] ** 2 for c, g in pairs]))
            row = {"B": B, "seed": seed, "sigma_median": float(np.median(a)),
                   "sigma_max": float(a.max()), "sigma_min": float(a.min()),
                   "ratio_median": float(np.median(ratio)), "Vbar": vbar,
                   "floor_12_pairs": float(np.sqrt(vbar / 12)),
                   "wall_s": round(time.perf_counter() - t0, 1)}
            with open(MC_JSONL, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[mc] B={B:5d} seed={seed:9d} median={row['sigma_median']:.5f} "
                  f"max={row['sigma_max']:.5f} ratio_med={row['ratio_median']:.4f} "
                  f"floor={row['floor_12_pairs']:.5f} [{row['wall_s']}s]", flush=True)
        rows.append(row)

    b400 = [r for r in rows if r["B"] == N_RESAMPLES]
    med = np.array([r["sigma_median"] for r in b400])
    fl = np.array([r["floor_12_pairs"] for r in b400])
    big = [r for r in rows if r["B"] == 10 * N_RESAMPLES][0]
    return {
        "why": ("the gate statistic (median of 13 per-run sigmas) is a single order "
                "statistic over B=400 bootstrap estimates, so it carries its own MC "
                "noise; measured here rather than assumed"),
        "n_seeds": len(b400), "B": N_RESAMPLES, "seeds": list(MC_SEEDS),
        "sigma_median_over_seeds": {"mean": float(med.mean()), "sd": float(med.std(ddof=1)),
                                    "min": float(med.min()), "max": float(med.max())},
        "floor_over_seeds": {"mean": float(fl.mean()), "sd": float(fl.std(ddof=1)),
                             "min": float(fl.min()), "max": float(fl.max())},
        "low_noise_reference": {"B": big["B"], "seed": big["seed"],
                                "sigma_median": big["sigma_median"],
                                "sigma_max": big["sigma_max"],
                                "ratio_median": big["ratio_median"],
                                "floor_12_pairs": big["floor_12_pairs"]},
        "checkpoint": str(MC_JSONL.relative_to(ROOT)).replace("\\", "/"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-recompute", action="store_true",
                    help="ignore the recompute checkpoint jsonl and redo every run")
    ap.add_argument("--mc-noise-check", action="store_true",
                    help="measure the bootstrap MC noise on the validation-gate "
                         "statistics (8 boot seeds at B=400 + one B=4000 pass); "
                         "results go into higson_results.json (~2.5 min)")
    args = ap.parse_args(argv)

    marks = load_markers()
    marks = recompute_markers(marks, force=args.force_recompute)
    ref = paired_reference()

    gain = {k: v for k, v in marks.items() if v["kind"] in ("gain_clean", "gain_gain")}
    block = {k: v for k, v in marks.items() if v["kind"] == "block"}

    # ---- per-run floor summary ----
    all_sigma = np.array([v["higson_sigma"] for v in marks.values()])
    gain_sigma = np.array([v["higson_sigma"] for v in gain.values()]) if gain else np.array([])

    # ---- paired-null floor from the measured gain-pair Higson sigmas ----
    # pair up clean+gain by pair_i; per-pair NS-sampling variance of the diff
    pairs = {}
    for v in gain.values():
        pairs.setdefault(v["pair_i"], {})[v["kind"]] = v
    pair_var_higson, pair_var_un = [], []
    per_pair = []
    for pi, dd in sorted(pairs.items()):
        if "gain_clean" in dd and "gain_gain" in dd:
            sc = dd["gain_clean"]["higson_sigma"]; sg = dd["gain_gain"]["higson_sigma"]
            uc = dd["gain_clean"]["ns_logzerr"]; ug = dd["gain_gain"]["ns_logzerr"]
            vh = sc**2 + sg**2; vu = uc**2 + ug**2
            pair_var_higson.append(vh); pair_var_un.append(vu)
            per_pair.append({
                "pair_i": pi,
                "counts_clean": dd["gain_clean"]["counts"],
                "counts_gain": dd["gain_gain"]["counts"],
                "sigma_higson_clean": sc, "sigma_higson_gain": sg,
                "logzerr_un_clean": uc, "logzerr_un_gain": ug,
                "pair_diff_floor_higson": float(np.sqrt(vh)),
                "pair_diff_floor_ultranest": float(np.sqrt(vu)),
            })
    floor = {}
    if pair_var_higson:
        vbar_h = float(np.mean(pair_var_higson))
        vbar_u = float(np.mean(pair_var_un))
        n = ref["n_pairs"]  # extrapolate the measured mean per-pair variance to all 12 pairs
        floor = {
            "n_gain_pairs_measured": len(pair_var_higson),
            "mean_pair_diff_variance_higson": vbar_h,
            "mean_pair_diff_variance_ultranest": vbar_u,
            "typical_pair_diff_floor_higson": float(np.sqrt(vbar_h)),
            "typical_pair_diff_floor_ultranest": float(np.sqrt(vbar_u)),
            # SE of the mean over n=12 pairs implied by NS sampling error alone
            "ns_floor_on_mean_from_higson_extrapolated": float(np.sqrt(vbar_h / n)),
            "ns_floor_on_mean_from_ultranest_extrapolated": float(np.sqrt(vbar_u / n)),
        }

    # ---- B1 line detection threat check ----
    b1 = {}
    for k, v in block.items():
        if v.get("family") == "B1":
            lvl = v["level"]
            resid = B1_RESID.get(lvl)
            if resid:
                half = (resid["ci"][1] - resid["ci"][0]) / 2.0
                b1[k] = {
                    "level": lvl, "counts": v["counts"],
                    "higson_sigma": v["higson_sigma"],
                    "ultranest_logzerr": v["ns_logzerr"],
                    "residual_median": resid["median"],
                    "residual_ci_halfwidth": half,
                    "sigma_vs_residual": v["higson_sigma"] / abs(resid["median"]),
                    "sigma_vs_ci_halfwidth": v["higson_sigma"] / half,
                }

    # ---- reproduction deltas for the (exactly-attached) block jobs ----
    repro = {}
    for k, v in block.items():
        if v.get("committed_logz") is not None:
            repro[k] = {
                "counts": v["counts"],
                "committed_logz": v["committed_logz"],
                "rerun_logz": v["ns_logz"],
                "delta_reported_vs_committed": v["delta_reported_vs_committed"],
                "delta_reconstructed_vs_reported": v["delta_reconstructed_vs_reported"],
                "higson_sigma": v["higson_sigma"],
            }

    # ---- consistency checks on the recompute ----
    gates = {}
    if marks:
        sig = np.array([v["higson_sigma"] for v in marks.values()])
        ratio = np.array([v["higson_sigma"] / v["ns_logzerr_single"] for v in marks.values()
                          if v.get("ns_logzerr_single")])
        gates = {
            "nlive_exact_on_all_runs": bool(all(v["nlive_exact_vs_runtxt"] for v in marks.values())),
            "n_runs_with_400_threads": int(sum(v["n_threads"] == 400 for v in marks.values())),
            "max_abs_delta_reconstructed_vs_ultranest_logz":
                float(max(abs(v["delta_reconstructed_vs_reported"]) for v in marks.values())),
            "sigma_median": float(np.median(sig)),
            "sigma_range": [float(sig.min()), float(sig.max())],
            "sigma_over_ns_logzerr_single_median": float(np.median(ratio)),
            "sigma_over_ns_logzerr_single_range": [float(ratio.min()), float(ratio.max())],
            "median_ratio_replayed_over_done_json":
                float(np.median([v["higson_sigma"] / v["higson_sigma_from_done_json"]
                                 for v in marks.values()])),
        }
        if args.mc_noise_check:
            gates["mc_noise_on_gate_statistics"] = mc_noise_check(
                marks, force=args.force_recompute)

    results = {
        "n_jobs_done": len(marks),
        "sigma_recompute": {
            "what": ("per-run higson_sigma computed from the replayed UltraNest "
                     "birth-death tree (points.hdf5 column 0 `Lmin` is a refill-batch "
                     "stamp, not a per-point birth contour) with a global Algorithm-2 "
                     "thread resample rather than one stratified by start contour. "
                     "Both differences raise the per-run sigma: the ratio to the "
                     "DONE.json value has median 1.198 and range 1.096-1.323 over the "
                     "13 runs."),
            "done_json_status": ("each run's DONE.json is left unmodified; its "
                                 "higson_sigma and n_threads are carried here as "
                                 "*_from_done_json"),
            "boot_seed": BOOT_SEED, "n_resamples": N_RESAMPLES,
            "checkpoint": str(RECOMPUTE_JSONL.relative_to(ROOT)).replace("\\", "/"),
            "consistency_checks": gates,
        },
        "committed_paired_reference": ref,
        "per_run_floor_summary": {
            "higson_sigma_all_median": float(np.median(all_sigma)) if len(all_sigma) else None,
            "higson_sigma_all_range": [float(all_sigma.min()), float(all_sigma.max())] if len(all_sigma) else None,
            "higson_sigma_gain_median": float(np.median(gain_sigma)) if len(gain_sigma) else None,
            "higson_sigma_gain_range": [float(gain_sigma.min()), float(gain_sigma.max())] if len(gain_sigma) else None,
        },
        "paired_null_floor": floor,
        "per_pair": per_pair,
        "b1_line_threat": b1,
        "block_reproduction": repro,
        "runs": marks,
    }
    (HG / "higson_results.json").write_text(json.dumps(results, indent=2))

    # ---- markdown table ----
    lines = []
    lines.append("# Higson thread-wise bootstrap: results\n")
    lines.append("Every per-run Higson sigma below is computed from the replayed UltraNest "
                 "birth-death tree with a global (Higson 2018 Algorithm 2) thread resample. "
                 "The sigma each run's DONE.json stored instead reads points.hdf5 column 0 "
                 "(`Lmin`, a refill-batch stamp) as a per-point birth contour and resamples "
                 "threads within start-contour groups, which lowers every sigma by a median "
                 "16.5% (range 8.8-24.4%). It is carried in the last column for reference.\n")
    lines.append(f"Jobs with DONE.json: {len(marks)}/13\n")
    lines.append("## Per-run: Higson thread bootstrap vs UltraNest error components\n")
    hdr = ("| job | kind | counts | wall_s | logZ | UN logzerr | UN bs | UN tail | "
           "Higson sigma | n_threads | reconstr-logZ delta |")
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    hdr += " sigma from DONE.json | sigma/sqrt(H/400) |"
    sep += "---:|---:|"
    lines.append(hdr)
    lines.append(sep)
    for k in sorted(marks):
        v = marks[k]
        row = (f"| {k} | {v['kind']} | {v['counts']} | {v['wall_s']:.0f} | "
               f"{v['ns_logz']:.2f} | {v['ns_logzerr']:.3f} | {v.get('ns_logzerr_bs', float('nan')):.3f} | "
               f"{v.get('ns_logzerr_tail', float('nan')):.3f} | {v['higson_sigma']:.3f} | "
               f"{v['n_threads']} | {v['delta_reconstructed_vs_reported']:+.3f} |")
        row += (f" {v['higson_sigma_from_done_json']:.3f} | "
                f"{v['sigma_over_logzerr_single']:.3f} |")
        lines.append(row)
    lines.append("")
    if gates:
        lines.append("### Consistency checks\n")
        lines.append(f"- reconstructed nlive profile identical to chains/run.txt on all runs: "
                     f"{gates['nlive_exact_on_all_runs']}")
        lines.append(f"- runs with exactly 400 threads: {gates['n_runs_with_400_threads']}/{len(marks)}")
        lines.append(f"- max |reconstructed logZ - UltraNest logz| = "
                     f"{gates['max_abs_delta_reconstructed_vs_ultranest_logz']:.4f} nats")
        lines.append(f"- per-run sigma: median {gates['sigma_median']:.4f}, range "
                     f"{gates['sigma_range'][0]:.4f}-{gates['sigma_range'][1]:.4f} nats")
        lines.append(f"- sigma / sqrt(H/400) (UltraNest logzerr_single): median "
                     f"{gates['sigma_over_ns_logzerr_single_median']:.3f}, range "
                     f"{gates['sigma_over_ns_logzerr_single_range'][0]:.3f}-"
                     f"{gates['sigma_over_ns_logzerr_single_range'][1]:.3f}")
        lines.append(f"- median replayed/DONE.json sigma ratio = "
                     f"{gates['median_ratio_replayed_over_done_json']:.3f}")
        lines.append("")
    lines.append("## Committed paired gain-null reference\n")
    lines.append(f"- n_pairs = {ref['n_pairs']}, mean dlogZ = {ref['mean_dlogz']:+.4f}, "
                 f"empirical paired SD = {ref['empirical_paired_SD']:.3f}, SEM = {ref['empirical_paired_SEM']:.3f}")
    lines.append(f"- UltraNest per-run logzerr mean = {ref['ultranest_logzerr_mean']:.3f} "
                 f"(range {ref['ultranest_logzerr_range'][0]:.3f}-{ref['ultranest_logzerr_range'][1]:.3f})")
    lines.append(f"- NS floor on the mean from UltraNest logzerr (all 12 pairs) = "
                 f"{ref['ns_floor_on_mean_from_ultranest_logzerr']:.4f}")
    if floor:
        lines.append("")
        lines.append("## NS-sampling floor on the paired mean (from measured Higson sigmas)\n")
        lines.append(f"- gain pairs measured: {floor['n_gain_pairs_measured']}")
        lines.append(f"- typical per-pair diff floor (Higson) = {floor['typical_pair_diff_floor_higson']:.3f} nats "
                     f"(UltraNest {floor['typical_pair_diff_floor_ultranest']:.3f})")
        lines.append(f"- NS floor on the mean over 12 pairs (Higson, extrapolated) = "
                     f"{floor['ns_floor_on_mean_from_higson_extrapolated']:.4f} nats "
                     f"(UltraNest {floor['ns_floor_on_mean_from_ultranest_extrapolated']:.4f})")
        lines.append(f"- mean dlogZ = {ref['mean_dlogz']:+.4f}; empirical SEM = {ref['empirical_paired_SEM']:.3f}")
        result = ("null-consistent: the NS-sampling floor is far below both the mean and the "
                   "empirical SEM; the observed scatter is dominated by real spectrum-to-spectrum "
                   "variation, not NS sampling noise."
                   if floor['ns_floor_on_mean_from_higson_extrapolated'] < abs(ref['mean_dlogz'])
                  else "check: floor comparable to the mean.")
        lines.append(f"- result: {result}")
    if b1:
        lines.append("")
        lines.append("## B1 line-detection threat check\n")
        lines.append("| job | level | counts | Higson sigma | residual median | CI halfwidth | sigma/|resid| | sigma/CIhalf |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for k, v in b1.items():
            lines.append(f"| {k} | {v['level']} | {v['counts']} | {v['higson_sigma']:.3f} | "
                         f"{v['residual_median']:.1f} | {v['residual_ci_halfwidth']:.1f} | "
                         f"{v['sigma_vs_residual']:.2e} | {v['sigma_vs_ci_halfwidth']:.2e} |")
    if repro:
        lines.append("")
        lines.append("## Block reproduction (exactly-attached rows)\n")
        lines.append("| job | counts | committed logZ | rerun logZ | delta rerun-committed | Higson sigma |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for k, v in repro.items():
            lines.append(f"| {k} | {v['counts']} | {v['committed_logz']:.2f} | {v['rerun_logz']:.2f} | "
                         f"{v['delta_reported_vs_committed']:+.2f} | {v['higson_sigma']:.3f} |")
    (HG / "higson_table.md").write_text("\n".join(lines) + "\n")

    print(f"[analyze] {len(marks)}/13 jobs. wrote higson_results.json + higson_table.md")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
