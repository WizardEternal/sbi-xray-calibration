r"""Post-campaign analysis of a paired gain-null nested-sampling set.

Reads a paired jsonl (one row per pair, written by scripts/paired_ns_gain_check.py)
plus the per-run UltraNest point stores that ``--log-dir-root`` produced, and
reports, in one place:

  1. the per-run Higson (2018) thread-bootstrap sigma for every run (2 per pair);
  2. the NS-sampling floor on the paired mean, both measured directly over the
     pairs that have a point store and extrapolated to all pairs in the jsonl;
  3. the paired statistic: mean d_paired +/- SEM, 95% t CI, two-sided p;
  4. the NS share of the paired variance, and the p-value after deleting all of it.

The estimator itself is NOT reimplemented here: threads come from
``higson_common.reconstruct_tree`` and the bootstrap from
``higson_common.higson_bootstrap`` (the 2026-08-14 corrected versions; never the
``*.pre_fix_2026-08-14.py`` copies). The paired t-test is the same statistic
``scripts/reconcile_ns_gain_line.py::paired_gain_stat`` ships: a two-sided
one-sample t-test on d_paired with df = n-1.

Every recomputed run is appended to the checkpoint jsonl the moment it lands, so
a killed analysis resumes.

Usage
-----
    # the new medium campaign
    python scripts/paired_higson_analyze.py \
        --jsonl outputs/ns_bench/revision_2026-09-02/medium/pairs_merged.jsonl \
        --log-dir-root outputs/ns_bench/revision_2026-09-02/medium/runs \
        --label medium_2026-09-02 \
        --out-dir outputs/ns_bench/revision_2026-09-02/medium

    # self-validation against the shipped E2 numbers (uses the 2026-07 higson runs
    # and the committed, unreproducible paired jsonl; ~1 min, no NS)
    python scripts/paired_higson_analyze.py --validate-shipped

``--run-name-fmt`` handles the older directory naming: the 2026-07 Higson batch
wrote ``gain_pair{i}_{kind}`` under outputs/ns_bench/higson/runs, the campaign
writes ``pair{i}_{kind}`` under its own root.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

N_RESAMPLES = 400          # repo convention (higson_analyze.py)
BOOT_SEED = 12345          # repo convention (higson_analyze.py)


def load_pairs(jsonl_paths):
    """Merge one or more paired jsonls, last row per index wins, sorted by index."""
    rows = {}
    for p in jsonl_paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[int(r["i"])] = r
    return [rows[k] for k in sorted(rows)]


def paired_stat(d: np.ndarray) -> dict:
    """Two-sided one-sample t-test on the paired differences (df = n-1).

    Identical statistic to scripts/reconcile_ns_gain_line.py::paired_gain_stat."""
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    tcrit = float(stats.t.ppf(0.975, df=df))
    t_stat = mean / se
    return {
        "n": n, "mean_d_paired": mean, "sd": sd, "sem": float(se),
        "ci95_t": [mean - tcrit * se, mean + tcrit * se],
        "t_stat": float(t_stat), "df": df,
        "p_two_sided": float(2 * (1 - stats.t.cdf(abs(t_stat), df=df))),
    }


def p_after_deleting_variance(d: np.ndarray, vbar: float) -> dict:
    """Recompute the paired t-test after subtracting the NS-sampling variance
    ``vbar`` (mean per-pair variance of the difference) from the empirical
    variance of d. This is the "delete all NS variance" robustness check: if the
    p-value barely moves, the null is not an artefact of NS sampling noise."""
    n = len(d)
    var = float(d.var(ddof=1))
    var_def = max(var - float(vbar), 0.0)
    sd_def = var_def ** 0.5
    se_def = sd_def / np.sqrt(n)
    t_def = float(d.mean()) / se_def if se_def > 0 else float("inf")
    return {
        "empirical_var": var,
        "ns_variance_removed": float(vbar),
        "ns_share_of_paired_variance": float(vbar) / var if var > 0 else float("nan"),
        "deflated_sd": sd_def, "deflated_sem": float(se_def),
        "deflated_t": t_def,
        "p_two_sided_deflated": float(2 * (1 - stats.t.cdf(abs(t_def), df=n - 1))),
    }


def higson_for_runs(pairs, log_dir_root, run_name_fmt, ckpt_path,
                    n_resamples=N_RESAMPLES, boot_seed=BOOT_SEED, force=False):
    """Per-run Higson sigma for every clean/gain run that has a point store.

    Returns {(pair_i, kind): row}. Runs with no directory are skipped and
    reported, so a partially finished campaign still analyses."""
    import higson_common as HC

    cache = {}
    if ckpt_path.exists() and not force:
        for line in ckpt_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[(int(r["pair_i"]), r["kind"])] = r

    out, missing = {}, []
    for pr in pairs:
        i = int(pr["i"])
        for kind in ("clean", "gain"):
            key = (i, kind)
            if key in cache:
                out[key] = cache[key]
                continue
            # a row written by paired_ns_gain_check.py records its own point-store
            # paths; prefer them, so a merged jsonl can mix runs living under
            # different roots (e.g. the 2026-07 Higson runs reused as pairs of the
            # new set) without copying or symlinking anything.
            recorded = pr.get(f"log_dir_{kind}")
            if recorded:
                run_dir = Path(recorded)
                if not run_dir.is_absolute():
                    run_dir = ROOT / run_dir
            elif log_dir_root is not None:
                run_dir = Path(log_dir_root) / run_name_fmt.format(i=i, kind=kind)
            else:
                missing.append(f"pair{i}_{kind}: no --log-dir-root and no log_dir in the row")
                continue
            if not (run_dir / "chains" / "run.txt").exists():
                missing.append(str(run_dir).replace("\\", "/"))
                continue
            t0 = time.perf_counter()
            tree = HC.reconstruct_tree(run_dir)
            boot = HC.higson_bootstrap(tree["logl"], tree["birth"],
                                       n_resamples=n_resamples, seed=boot_seed,
                                       labels=tree["thread_id"])
            row = {
                "pair_i": i, "kind": kind, "run_dir": str(run_dir).replace("\\", "/"),
                "higson_sigma": boot["higson_sigma"],
                "higson_logZ_reconstructed": boot["logZ_reconstructed"],
                "n_threads": boot["n_threads"], "n_start_groups": boot["n_start_groups"],
                "n_points": boot["n_points"], "n_resamples": boot["n_resamples"],
                "boot_seed": boot_seed,
                "nlive_exact_vs_runtxt": tree["nlive_exact"],
                "nlive_n_mismatch": tree["nlive_n_mismatch"],
                "n_store": tree["n_store"], "n_store_unused": tree["n_store_unused"],
                "jsonl_logz": pr.get(f"logz_{kind}"),
                "jsonl_logzerr": pr.get(f"logzerr_{kind}"),
                "counts": pr.get(f"counts_{kind}"),
                "wall_s": round(time.perf_counter() - t0, 1),
            }
            row["delta_reconstructed_vs_reported"] = (
                row["higson_logZ_reconstructed"] - row["jsonl_logz"]
                if row["jsonl_logz"] is not None else None)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[higson] pair{i:>2} {kind:<5} sigma={row['higson_sigma']:.4f} "
                  f"threads={row['n_threads']} nlive_exact={row['nlive_exact_vs_runtxt']} "
                  f"[{row['wall_s']}s]", flush=True)
            out[key] = row
    return out, missing


def analyse(pairs, sig, label):
    d = np.array([r["d_paired"] for r in pairs], float)
    stat = paired_stat(d)

    per_pair, var_h, var_u = [], [], []
    for pr in pairs:
        i = int(pr["i"])
        vu = pr["logzerr_clean"] ** 2 + pr["logzerr_gain"] ** 2
        var_u.append(vu)
        sc = sig.get((i, "clean"), {}).get("higson_sigma")
        sg = sig.get((i, "gain"), {}).get("higson_sigma")
        vh = (sc ** 2 + sg ** 2) if (sc is not None and sg is not None) else None
        if vh is not None:
            var_h.append(vh)
        per_pair.append({
            "pair_i": i, "counts_clean": pr["counts_clean"], "counts_gain": pr["counts_gain"],
            "d_paired": pr["d_paired"], "wall_s": pr.get("wall_s"),
            "logzerr_clean": pr["logzerr_clean"], "logzerr_gain": pr["logzerr_gain"],
            "sigma_higson_clean": sc, "sigma_higson_gain": sg,
            "pair_diff_var_higson": vh,
            "pair_diff_floor_higson": (vh ** 0.5 if vh is not None else None),
            "pair_diff_floor_ultranest": vu ** 0.5,
        })

    n = len(d)
    var_u = np.asarray(var_u, float)
    floor = {
        "n_pairs_in_statistic": n,
        "n_pairs_with_higson": len(var_h),
        "floor_on_mean_from_ultranest_logzerr": float(np.sqrt(var_u.sum()) / n),
        "mean_pair_diff_variance_ultranest": float(var_u.mean()),
    }
    if var_h:
        vh_arr = np.asarray(var_h, float)
        vbar_h = float(vh_arr.mean())
        floor.update({
            "mean_pair_diff_variance_higson": vbar_h,
            "typical_pair_diff_floor_higson": float(np.sqrt(vbar_h)),
            # extrapolated: mean per-pair variance transferred to all n pairs
            "floor_on_mean_higson_extrapolated": float(np.sqrt(vbar_h / n)),
            # direct: only valid, and equal to the above, when every pair is measured
            "floor_on_mean_higson_direct": (float(np.sqrt(vh_arr.sum()) / n)
                                            if len(var_h) == n else None),
            "extrapolation_needed": len(var_h) != n,
            "sem_over_floor_higson": stat["sem"] / float(np.sqrt(vbar_h / n)),
        })
        deflated = p_after_deleting_variance(d, vbar_h)
        deflated_un = p_after_deleting_variance(d, float(var_u.mean()))
        share = {
            "ns_share_higson": deflated["ns_share_of_paired_variance"],
            "ns_share_ultranest": deflated_un["ns_share_of_paired_variance"],
            "ns_share_range_percent": [
                100 * min(deflated["ns_share_of_paired_variance"],
                          deflated_un["ns_share_of_paired_variance"]),
                100 * max(deflated["ns_share_of_paired_variance"],
                          deflated_un["ns_share_of_paired_variance"])],
            "p_original": stat["p_two_sided"],
            "p_deleting_all_ns_variance_higson": deflated["p_two_sided_deflated"],
            "p_deleting_all_ns_variance_ultranest": deflated_un["p_two_sided_deflated"],
        }
    else:
        deflated = deflated_un = share = None

    sigmas = np.array([v["higson_sigma"] for v in sig.values()], float)
    return {
        "label": label,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_runs_with_higson": len(sig),
        "per_run_sigma_summary": ({
            "median": float(np.median(sigmas)),
            "range": [float(sigmas.min()), float(sigmas.max())],
            "n": int(sigmas.size),
        } if sigmas.size else None),
        "paired_statistic": stat,
        "paired_null_floor": floor,
        "variance_attribution": share,
        "deflated_higson": deflated,
        "deflated_ultranest": deflated_un,
        "per_pair": per_pair,
        "per_run": [sig[k] for k in sorted(sig)],
        "conventions": {
            "higson_n_resamples": N_RESAMPLES, "higson_boot_seed": BOOT_SEED,
            "estimator": "higson_common.reconstruct_tree + higson_bootstrap "
                         "(2026-08-14 corrected)",
            "p_value_test": "two-sided one-sample t-test on d_paired, df=n-1 "
                            "(same as scripts/reconcile_ns_gain_line.py)",
        },
    }


def render_md(res) -> str:
    L = [f"# Paired gain-null + Higson floor: {res['label']}\n",
         f"Generated {res['generated']}.\n"]
    s = res["paired_statistic"]
    L.append("## Paired statistic\n")
    L.append(f"- n = {s['n']} pairs; mean d_paired = {s['mean_d_paired']:+.4f} nats, "
             f"SD {s['sd']:.4f}, SEM {s['sem']:.4f}")
    L.append(f"- 95% t CI [{s['ci95_t'][0]:+.4f}, {s['ci95_t'][1]:+.4f}], "
             f"t = {s['t_stat']:.4f} (df {s['df']}), two-sided p = {s['p_two_sided']:.6f}")
    f = res["paired_null_floor"]
    L.append("\n## NS-sampling floor\n")
    if res["per_run_sigma_summary"]:
        q = res["per_run_sigma_summary"]
        L.append(f"- per-run Higson sigma over {q['n']} runs: median {q['median']:.5f} nats, "
                 f"range {q['range'][0]:.5f} to {q['range'][1]:.5f}")
    if "mean_pair_diff_variance_higson" in f:
        L.append(f"- pairs with a point store: {f['n_pairs_with_higson']}/{f['n_pairs_in_statistic']}"
                 f"  (extrapolation needed: {f['extrapolation_needed']})")
        L.append(f"- mean per-pair difference variance = {f['mean_pair_diff_variance_higson']:.6f} nats^2; "
                 f"typical per-pair floor {f['typical_pair_diff_floor_higson']:.4f} nats")
        L.append(f"- floor on the paired mean = {f['floor_on_mean_higson_extrapolated']:.6f} nats"
                 + (f" (direct {f['floor_on_mean_higson_direct']:.6f})"
                    if f.get("floor_on_mean_higson_direct") is not None else ""))
        L.append(f"- SEM / floor = {f['sem_over_floor_higson']:.2f}")
    L.append(f"- floor from UltraNest's own logzerr = "
             f"{f['floor_on_mean_from_ultranest_logzerr']:.6f} nats")
    v = res["variance_attribution"]
    if v:
        L.append("\n## Variance attribution\n")
        L.append(f"- NS share of the paired variance: {100*v['ns_share_higson']:.3f} per cent "
                 f"(Higson), {100*v['ns_share_ultranest']:.3f} per cent (UltraNest logzerr) "
                 f"-> {v['ns_share_range_percent'][0]:.2f} to "
                 f"{v['ns_share_range_percent'][1]:.2f} per cent")
        L.append(f"- deleting ALL NS variance moves p from {v['p_original']:.6f} to "
                 f"{v['p_deleting_all_ns_variance_higson']:.6f} (Higson) / "
                 f"{v['p_deleting_all_ns_variance_ultranest']:.6f} (UltraNest)")
    L.append("\n## Per pair\n")
    L.append("| pair | counts clean/gain | d_paired | wall_s | sigma_H clean | sigma_H gain | "
             "pair floor (H) | pair floor (UN) |")
    L.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for p in res["per_pair"]:
        fh = p["pair_diff_floor_higson"]
        L.append(f"| {p['pair_i']} | {p['counts_clean']}/{p['counts_gain']} | "
                 f"{p['d_paired']:+.4f} | {p['wall_s'] if p['wall_s'] is not None else ''} | "
                 f"{p['sigma_higson_clean']:.4f} | {p['sigma_higson_gain']:.4f} | "
                 f"{fh:.4f} | {p['pair_diff_floor_ultranest']:.4f} |"
                 if fh is not None else
                 f"| {p['pair_i']} | {p['counts_clean']}/{p['counts_gain']} | "
                 f"{p['d_paired']:+.4f} | {p['wall_s'] if p['wall_s'] is not None else ''} | "
                 f"- | - | - | {p['pair_diff_floor_ultranest']:.4f} |")
    L.append("\n## Per run\n")
    L.append("| pair | kind | counts | logZ (jsonl) | UN logzerr | sigma_H | threads | "
             "nlive exact | reconstr-logZ delta |")
    L.append("|---:|---|---:|---:|---:|---:|---:|---|---:|")
    for r in res["per_run"]:
        L.append(f"| {r['pair_i']} | {r['kind']} | {r['counts']} | {r['jsonl_logz']:.3f} | "
                 f"{r['jsonl_logzerr']:.3f} | {r['higson_sigma']:.4f} | {r['n_threads']} | "
                 f"{r['nlive_exact_vs_runtxt']} | {r['delta_reconstructed_vs_reported']:+.4f} |")
    return "\n".join(L) + "\n"


SHIPPED = {   # E2 report §8 corrected numbers, for --validate-shipped
    "p_two_sided": 0.8116289751447194,
    "mean_d_paired": 0.3344784737589137,
    "sem": 1.3701221698877286,
    "vbar_higson": 0.04383723433320786,
    "floor": 0.060440903873954885,
    "sem_over_floor": 22.67,
    # NOT 0.811506. The paper (journal_revision/main.tex L318) and E2 §8 quote
    # "0.8116 to 0.8115" / 0.811506, which this script traces to the SUPERSEDED
    # pre-fix variance Vbar = 0.030028 (floor 0.050023), the pair the 2026-08-14
    # fix replaced with Vbar = 0.043837 / floor 0.060441. With the corrected
    # estimator the deflated p is 0.811449 and the variance share is 0.195 per
    # cent, so the shipped sentence should read "0.20 to 0.43 per cent" and
    # "0.8116 to 0.8114". Same class of pre-fix leftover as FIXLIST A2b. No
    # conclusion moves; the number is simply stale.
    "p_deflated": 0.8114493584751092,
    "p_deflated_prefix_leftover": 0.811506,
    "vbar_higson_prefix": 0.030028,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", nargs="*", type=Path, default=None,
                    help="one or more paired jsonls; later files win on a repeated index")
    ap.add_argument("--log-dir-root", type=Path, default=None)
    ap.add_argument("--run-name-fmt", default="pair{i}_{kind}",
                    help="directory name under --log-dir-root; the 2026-07 batch used "
                         "gain_pair{i}_{kind}")
    ap.add_argument("--label", default="paired_gain_null")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where paired_higson_results.json + .md go (default: the "
                         "first --jsonl's directory)")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--validate-shipped", action="store_true",
                    help="run against the committed paired jsonl + the 2026-07 Higson "
                         "runs and assert the shipped E2 numbers come back")
    args = ap.parse_args(argv)

    if args.validate_shipped:
        args.jsonl = [ROOT / "outputs/ns_bench/paired_gain_check.jsonl"]
        args.log_dir_root = ROOT / "outputs/ns_bench/higson/runs"
        args.run_name_fmt = "gain_pair{i}_{kind}"
        args.label = "SHIPPED committed 12 pairs + 2026-07 Higson runs (validation)"
        args.out_dir = args.out_dir or (ROOT / "outputs/ns_bench/revision_2026-09-02")

    if not args.jsonl:
        ap.error("--jsonl is required (or use --validate-shipped)")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.jsonl[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "paired_higson_validation" if args.validate_shipped else "paired_higson_results"
    ckpt = out_dir / f"{stem}_recompute.jsonl"

    pairs = load_pairs(args.jsonl)
    print(f"[in] {len(pairs)} pairs from {[str(p) for p in args.jsonl]}", flush=True)
    sig, missing = ({}, [])
    has_recorded = any(p.get("log_dir_clean") or p.get("log_dir_gain") for p in pairs)
    if args.log_dir_root is not None or has_recorded:
        sig, missing = higson_for_runs(pairs, args.log_dir_root, args.run_name_fmt,
                                       ckpt, force=args.force_recompute)
    if missing:
        print(f"[warn] {len(missing)} runs have no point store yet:", flush=True)
        for m in missing[:24]:
            print(f"       {m}", flush=True)

    res = analyse(pairs, sig, args.label)
    res["inputs"] = {"jsonl": [str(p).replace("\\", "/") for p in args.jsonl],
                     "log_dir_root": (str(args.log_dir_root).replace("\\", "/")
                                      if args.log_dir_root else None),
                     "run_name_fmt": args.run_name_fmt,
                     "runs_missing_point_store": missing}
    (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2))
    md = render_md(res)
    (out_dir / f"{stem}.md").write_text(md)
    print(md)
    print(f"[out] {out_dir / (stem + '.json')}\n[out] {out_dir / (stem + '.md')}")

    if args.validate_shipped:
        s, f = res["paired_statistic"], res["paired_null_floor"]
        v = res["variance_attribution"]
        checks = [
            ("p_two_sided", s["p_two_sided"], SHIPPED["p_two_sided"], 1e-9),
            ("mean_d_paired", s["mean_d_paired"], SHIPPED["mean_d_paired"], 1e-9),
            ("sem", s["sem"], SHIPPED["sem"], 1e-9),
            ("Vbar_higson", f["mean_pair_diff_variance_higson"], SHIPPED["vbar_higson"], 1e-9),
            ("floor", f["floor_on_mean_higson_extrapolated"], SHIPPED["floor"], 1e-9),
            ("SEM/floor", f["sem_over_floor_higson"], SHIPPED["sem_over_floor"], 5e-3),
            ("p deflated", v["p_deleting_all_ns_variance_higson"], SHIPPED["p_deflated"], 1e-6),
        ]
        ok = True
        print("\n=== VALIDATION vs the shipped E2 numbers ===")
        for name, got, want, tol in checks:
            good = abs(got - want) <= tol
            ok &= good
            print(f"  {'OK  ' if good else 'FAIL'} {name:<16} got {got!r:<24} want {want!r}")
        print(f"NS share range (per cent): {v['ns_share_range_percent'][0]:.3f} to "
              f"{v['ns_share_range_percent'][1]:.3f}")
        print("  NOTE the paper says 0.13 to 0.43 and 'p 0.8116 -> 0.8115'. The 0.13 end "
              "and 0.811506\n       both come from the PRE-FIX Vbar "
              f"{SHIPPED['vbar_higson_prefix']} that the 2026-08-14 fix superseded; "
              "with the\n       corrected estimator they are 0.195 per cent and "
              f"{SHIPPED['p_deflated']:.6f}.")
        print("  NOTE the 'reconstr-logZ delta' column is meaningless in this mode: it "
              "compares the\n       2026-07 rerun's logZ against the committed jsonl's "
              "logZ, and those are different\n       spectra (the committed set is "
              "unreproducible). In a real campaign both come from the\n       same runs "
              "and the delta is the tree-reconstruction check.")
        print("VALIDATION", "PASSED" if ok else "FAILED")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
