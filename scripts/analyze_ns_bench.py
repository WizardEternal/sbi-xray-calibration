r"""NS-vs-NPE benchmark analysis: the speed-vs-agreement table, the NS-flag-vs-
detector-flag cross-check, and the canonical count-controlled nested-sampling
evidence analysis.

Usage (repo venv):
    .venv\Scripts\python.exe scripts\analyze_ns_bench.py --config configs\ns_bench.yaml
    .venv\Scripts\python.exe scripts\analyze_ns_bench.py --config configs\ns_bench_nicer.yaml

    # count-controlled analysis only, no config (the standalone invocation
    # analyze_ns_bench_countctl.py used to document): pass a results.jsonl path
    # directly, writes only <dir>/count_controlled.json next to it.
    .venv\Scripts\python.exe scripts\analyze_ns_bench.py outputs\ns_bench\results.jsonl
    .venv\Scripts\python.exe scripts\analyze_ns_bench.py outputs\ns_bench_nicer\results.jsonl

Reads outputs/ns_bench/results.jsonl (written by run_ns_benchmark.py) and prints
(and writes to outputs/ns_bench/analysis.md), in --config mode:

  1. speed-vs-agreement table, per count level (clean Model-A spectra only, the
     well-specified spine): NS wall-clock (s/spectrum) and n_like_evals vs NPE
     sampling (ms/spectrum), the NS/NPE speed ratio, and the NS-vs-NPE posterior
     quantile agreement (mean |q_NS - q_NPE| / prior-width). Small agreement where
     raw NPE is well calibrated (faint/medium) = NS validates the amortized flow;
     larger at bright = NS exposes the over-confidence the calibration suite measured.

  2. NS misspecification-flag check on the B1/B4 spectra: per misspecified
     spectrum we form two NS-side flags:
        * residual flag: the best-fit (max-likelihood) Poisson chi2-like residual
          (reduced), where a poor fit to the well-specified Model A flags misspec;
        * evidence flag: logZ relative to the clean-population logZ at the same
          level (a misspecified spectrum the model cannot fit has lower evidence).
     and reports them next to the detector AUC for the matching
     (family, strength, level) cell. The d-logZ count-controlled column here is
     the same count-controlled evidence residual as (3) below, computed with the
     same ported functions and RNG-consumption order.

  3. Count-controlled nested-sampling evidence analysis (written separately to
     outputs/ns_bench/count_controlled.json): logZ is the log marginal likelihood
     of one dataset, so it scales ~linearly with total counts
     (logZ ~ a*log10(counts)+b), and a count "level" spans a ~30x count range. A
     level-matched mean(logZ_mis) - mean(logZ_clean) over unmatched spectra is
     therefore confounded by which spectra land in each group. This fits
     logZ vs log10(counts) on the clean spectra (all levels), then reports each
     misspecified cell's mean residual from that trend, with a bootstrap CI and a
     per-spectrum breakdown. A real model error sits below the clean trend
     (negative residual); a count artifact sits on it. This is the canonical
     count-controlled analysis, and the numbers the README quotes.

  *** Detector cross-check status ***  The detector benchmark
  (outputs/detect/results.jsonl) may still be running (the full 144-cell grid).
  This script reads that file if present (read-only; it never writes there) and
  fills the detector-AUC column where the matching cell exists; cells not yet
  computed are shown as "pending" and the whole cross-check is labelled a stub
  until the detector grid finishes. The NS-side flags are always computed.

Writes to outputs/ns_bench/analysis.md, analysis_summary.json and
count_controlled.json in --config mode; only count_controlled.json in the bare
results.jsonl-path mode. Reads outputs/ns_bench/results.jsonl and (read-only, if
present) outputs/detect/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sbixcal._shared import _repo_root, load_config, _read_jsonl


def _out_dir(cfg: dict) -> Path:
    """Resolve the ns_bench output dir from the config's optional out_dir key.
    Default (out_dir absent) is exactly outputs/ns_bench, matching
    run_ns_benchmark.py's default."""
    out_dir = cfg.get("out_dir", "outputs/ns_bench")
    p = Path(out_dir)
    return p if p.is_absolute() else _repo_root() / p


# 1. speed-vs-agreement (clean spine)

def speed_agreement_table(rows):
    """Per-level aggregates over the clean Model-A spectra."""
    by_level = defaultdict(list)
    for r in rows:
        if r["family"] == "clean":
            by_level[r["level"]].append(r)

    order = ["faint", "medium", "bright"]
    lines = []
    lines.append("| level | ~counts | n | NS s/spec | NS n_like_evals | "
                 "NPE ms/spec | NS/NPE speedup | q-agreement (mean |dq|/width) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    agg = {}
    for lvl in order:
        rs = by_level.get(lvl, [])
        if not rs:
            continue
        ns_wall = np.array([r["ns"]["wall_s"] for r in rs])
        ns_eval = np.array([r["ns"]["n_like_evals"] for r in rs])
        npe_ms = np.array([r["npe"]["sample_wall_s"] * 1e3 for r in rs])
        agree = np.array([r["agreement"]["mean_abs_norm"] for r in rs])
        counts = np.median([r["n_counts"] for r in rs])
        speedup = (ns_wall.mean() * 1e3) / npe_ms.mean()  # NS ms / NPE ms
        lines.append(
            f"| {lvl} | {counts:.0f} | {len(rs)} | {ns_wall.mean():.1f} | "
            f"{ns_eval.mean():.0f} | {npe_ms.mean():.0f} | {speedup:.0f}x | "
            f"{agree.mean():.3f} |")
        agg[lvl] = {
            "n": len(rs), "ns_wall_mean": float(ns_wall.mean()),
            "ns_wall_median": float(np.median(ns_wall)),
            "ns_eval_mean": float(ns_eval.mean()),
            "npe_ms_mean": float(npe_ms.mean()),
            "speedup": float(speedup), "q_agree_mean": float(agree.mean()),
            "counts": float(counts),
        }
    return "\n".join(lines), agg


# 2. NS misspecification flags vs detector AUC, and
# 3. the count-controlled evidence analysis they both draw on
#
# ns_flag_table's markdown/AUC
# cross-check and count_controlled_report's outputs/ns_bench/count_controlled.json
# both call them, so the count-controlled numbers are the same computation
# everywhere they are reported, not two implementations that can drift apart.

def fit_clean_trend(rows, rng):
    """Fit logZ = a*log10(counts)+b on the clean spectra (all levels). logZ scales
    with total counts, so this trend is the count-controlled baseline every
    misspecified cell's residual is compared against."""
    cc = np.array([r["n_counts"] for r in rows if r["family"] == "clean"], float)
    cz = np.array([r["ns"]["logz"] for r in rows if r["family"] == "clean"], float)
    X = np.vstack([np.log10(cc), np.ones_like(cc)]).T
    coef, *_ = np.linalg.lstsq(X, cz, rcond=None)
    resid = cz - X @ coef
    sd = resid.std(ddof=2)
    boot = []
    n = len(cc)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        c, *_ = np.linalg.lstsq(X[idx], cz[idx], rcond=None)
        boot.append(c)
    boot = np.array(boot)
    return coef, sd, boot, n


def cell_residual(sel, coef, boot, fam, level, rng):
    """Count-controlled residual + bootstrap CI for one cell's spectra (`sel`,
    already filtered by the caller. count_controlled_report filters by
    family+level only, matching the cell list below; ns_flag_table filters by
    family+strength+level, a no-op today since the ns_bench subsample draws
    exactly one strength per family/level cell). Bootstraps over both sources of
    uncertainty: the clean-trend fit (each `boot` row) and the finite cell itself
    (resampling the n cell spectra); dropping the cell resample understates the
    CI badly for small n."""
    if not sel:
        return None
    gc = np.array([r["n_counts"] for r in sel], float)
    gz = np.array([r["ns"]["logz"] for r in sel], float)
    pred = coef[0] * np.log10(gc) + coef[1]
    res = gz - pred
    bmeans = []
    n = len(res)
    for b in range(len(boot)):
        predb = boot[b, 0] * np.log10(gc) + boot[b, 1]
        idx = rng.integers(0, n, n)
        bmeans.append((gz[idx] - predb[idx]).mean())
    lo, hi = np.percentile(bmeans, [2.5, 97.5])
    # significance from the cell's own residual scatter (per-cell t-statistic);
    # dividing by the clean-trend SD understates the noise badly for high-count
    # cells, where the per-spectrum penalty varies by hundreds of nats.
    # The bootstrap CI above is the robust statement and the one reported.
    cell_sd = res.std(ddof=1)
    sig = res.mean() / (cell_sd / np.sqrt(n)) if n > 1 and cell_sd > 0 else float("nan")
    return dict(fam=fam, level=level, n=n, mean_resid=float(res.mean()),
                ci=[float(lo), float(hi)], sigma=float(sig),
                per_spec=[round(float(x), 1) for x in res],
                counts=[int(c) for c in gc])


# the fixed (family, level) cells the ns_bench subsample draws misspecified
# spectra for; count_controlled_report visits them in exactly this order so the
# shared-rng bootstrap draws land the same way run to run.
_COUNT_CONTROLLED_CELLS = [("B1", "medium"), ("B1", "bright"),
                           ("B4", "medium"), ("B4", "bright")]


def count_controlled_report(rows, rng):
    """The canonical count-controlled NS-evidence analysis (outputs/ns_bench/
    count_controlled.json): fit_clean_trend once, then cell_residual over the
    fixed cell list above, in that order, sharing one rng stream."""
    coef, sd, boot, n_clean = fit_clean_trend(rows, rng)
    out = {"trend": {"slope": float(coef[0]), "intercept": float(coef[1]),
                     "resid_sd": float(sd), "n_clean": n_clean}, "cells": []}
    for fam, lvl in _COUNT_CONTROLLED_CELLS:
        sel = [r for r in rows if r["family"] == fam and r["level"] == lvl]
        r = cell_residual(sel, coef, boot, fam, lvl, rng)
        if r is None:
            continue
        r["caught"] = bool(r["ci"][1] < 0)  # CI entirely below 0 = a real evidence penalty
        out["cells"].append(r)
    return out, coef, sd


def _print_count_controlled(label, out, coef, sd):
    """Console report of the count-controlled cells."""
    print(f"# Count-controlled NS analysis: {label}")
    print(f"clean trend: logZ = {coef[0]:.2f}*log10(counts) + {coef[1]:.2f}  "
          f"(n_clean={out['trend']['n_clean']}, resid sd={sd:.2f})")
    print(f"{'cell':22s} {'n':>3s} {'mean_resid':>11s} {'95% CI':>20s} {'sigma':>7s}  flag")
    for r in out["cells"]:
        flag = "penalty" if r["caught"] else "none"
        print(f"{r['fam']+'/'+r['level']:22s} {r['n']:3d} {r['mean_resid']:+11.1f} "
              f"[{r['ci'][0]:+7.1f},{r['ci'][1]:+7.1f}] {r['sigma']:+7.1f}  {flag}")


def _detector_auc_lookup(detect_rows):
    """Map (family, strength, level) -> {detector: auc} from a detect results.jsonl."""
    out = defaultdict(dict)
    for r in detect_rows:
        key = (r["family"], float(r["strength"]), r["level"])
        out[key][r["detector"]] = r["auc"]
    return out


def ns_flag_table(rows, detect_rows):
    """Per misspecified cell: the count-controlled evidence residual (mean logZ minus
    the clean logZ-vs-log10(counts) trend, with a bootstrap CI), shown next to the
    detector AUCs. Below the trend is a real evidence penalty; on the trend is none."""
    rng = np.random.default_rng(20260630)
    coef, sd, boot, n_clean = fit_clean_trend(rows, rng)
    det = _detector_auc_lookup(detect_rows)

    groups = defaultdict(list)
    for r in rows:
        if r["family"] == "clean":
            continue
        # strength from the label (B1_s0.0003 -> 0.0003)
        slab = r["strength_label"]
        try:
            strength = float(slab.split("_s")[-1])
        except ValueError:
            strength = float("nan")
        groups[(r["family"], strength, r["level"])].append(r)

    lines = []
    lines.append("| family | strength | level | n | d-logZ count-controlled [95% CI] | "
                 "D1 AUC | D2 AUC | D3 AUC | detector status |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    rowsout = []
    # iterate in (family, canonical count-level order), not alphabetical on level,
    # since "bright" < "medium" as strings would consume the shared bootstrap rng
    # stream in a different order per cell than the canonical reproducer
    # (count_controlled_report / _COUNT_CONTROLLED_CELLS above, which visits medium
    # before bright for each family) and silently change every cell's CI despite an
    # identical seed.
    _level_order = {"faint": 0, "medium": 1, "bright": 2}
    for (fam, strength, lvl), rs in sorted(
            groups.items(), key=lambda kv: (kv[0][0], _level_order.get(kv[0][2], 99), kv[0][1])):
        cr = cell_residual(rs, coef, boot, fam, lvl, rng)
        mean_r = cr["mean_resid"]
        lo, hi = cr["ci"]
        aucs = det.get((fam, strength, lvl), {})
        d1 = aucs.get("D1"); d2 = aucs.get("D2"); d3 = aucs.get("D3")
        have = sum(a is not None for a in (d1, d2, d3))
        status = ("ready" if have == 3 else
                  (f"pending ({have}/3)" if have else "pending (STUB)"))
        def f(a):
            return f"{a:.3f}" if a is not None else "n/a"
        lines.append(
            f"| {fam} | {strength:g} | {lvl} | {cr['n']} | {mean_r:+.0f} [{lo:+.0f}, {hi:+.0f}] | "
            f"{f(d1)} | {f(d2)} | {f(d3)} | {status} |")
        rowsout.append({
            "family": fam, "strength": strength, "level": lvl, "n": cr["n"],
            "dlogz_count_controlled": mean_r, "ci95": [lo, hi],
            "detector_auc": aucs, "detector_status": status,
        })
    return "\n".join(lines), rowsout


# truth-recovery summary (clean spectra: NS 90% interval contains truth?)

def truth_recovery(rows):
    """Per level, fraction of clean spectra whose truth falls in the NS 5-95%
    interval, per parameter then averaged, a sanity coverage proxy for NS."""
    by_level = defaultdict(list)
    for r in rows:
        if r["family"] == "clean" and r.get("truth") is not None:
            by_level[r["level"]].append(r)
    order = ["faint", "medium", "bright"]
    lines = ["| level | n | NS 90% interval contains truth (mean over params) |",
             "|---|---|---|"]
    for lvl in order:
        rs = by_level.get(lvl, [])
        if not rs:
            continue
        hits = []
        for r in rs:
            names = r["param_names"]
            t = np.asarray(r["truth"], dtype=float)
            inside = []
            for j, nm in enumerate(names):
                lo = r["ns"]["quantiles"][nm]["0.05"]
                hi = r["ns"]["quantiles"][nm]["0.95"]
                inside.append(lo <= t[j] <= hi)
            hits.append(np.mean(inside))
        lines.append(f"| {lvl} | {len(rs)} | {np.mean(hits):.2f} |")
    return "\n".join(lines)


def _run_full(config_path: str) -> int:
    """--config mode: the speed/agreement + misspec-flag + truth-recovery
    analysis, plus the count-controlled evidence analysis. Writes analysis.md,
    analysis_summary.json and count_controlled.json under the config's out_dir."""
    cfg = load_config(config_path)

    out = _out_dir(cfg)
    all_rows = _read_jsonl(out / "results.jsonl")
    # error rows (a worker caught a per-spectrum exception and wrote a keyed error
    # row instead of a result) carry an "error" key and no ns/npe payload; skip them
    # from every table but keep a count so the run's fault budget is visible.
    rows = [r for r in all_rows if "error" not in r]
    n_err = len(all_rows) - len(rows)
    if n_err:
        print(f"[note] skipping {n_err} error row(s) "
              f"(per-spectrum failures recorded by the runner).")
    if not rows:
        print("No usable outputs/ns_bench/results.jsonl rows yet"
              f"{f' ({n_err} error rows only)' if n_err else ''}. Run "
              "scripts/run_ns_benchmark.py first.")
        return 0

    detect_rows = _read_jsonl(_repo_root() / "outputs" / "detect" / "results.jsonl")

    speed_tbl, speed_agg = speed_agreement_table(rows)
    flag_tbl, flag_rows = ns_flag_table(rows, detect_rows)
    recov_tbl = truth_recovery(rows)

    n_clean = sum(1 for r in rows if r["family"] == "clean")
    n_mis = len(rows) - n_clean
    det_status = ("READY" if detect_rows else "STUB (no outputs/detect/results.jsonl)")

    md = []
    md.append("# NS-vs-NPE benchmark: analysis\n")
    md.append(f"Spectra analyzed: {len(rows)} ({n_clean} clean, {n_mis} misspecified)"
              f"{f'; {n_err} error row(s) skipped' if n_err else ''}. "
              f"Detector cross-check: **{det_status}** "
              f"({len(detect_rows)} detector cells available).\n")
    md.append("## 1. Speed vs agreement (clean Model-A spine)\n")
    md.append(speed_tbl + "\n")
    md.append("## 2. NS truth recovery (clean; 90% interval coverage proxy)\n")
    md.append(recov_tbl + "\n")
    md.append("## 3. NS misspecification flags vs detector AUC\n")
    md.append("logZ scales with total counts, so each cell's flag is the count-"
              "controlled residual: mean logZ minus the clean logZ-vs-log10(counts) "
              "trend (95% CI). Below the trend => the well-specified Model A fits the "
              "misspecified spectra worse => flagged; on the trend => no penalty. "
              "Detector AUCs are read-only from outputs/detect/results.jsonl; "
              "cells the detector grid has not produced yet show as pending.\n")
    md.append(flag_tbl + "\n")

    md_text = "\n".join(md)
    (out / "analysis.md").write_text(md_text, encoding="utf-8")

    # console-safe print (Windows cp1252 can choke on non-ASCII; the .md file is utf-8)
    try:
        print(md_text)
    except UnicodeEncodeError:
        print(md_text.encode("ascii", "replace").decode("ascii"))
    print(f"\n[written] {out / 'analysis.md'}")

    # also drop a compact JSON of the aggregates for downstream use
    (out / "analysis_summary.json").write_text(json.dumps({
        "n_spectra": len(rows), "n_clean": n_clean, "n_misspec": n_mis,
        "n_error_rows": n_err,
        "speed_agreement": speed_agg, "ns_misspec_flags": flag_rows,
        "detector_status": det_status,
    }, indent=2), encoding="utf-8")

    # canonical count-controlled evidence analysis, written separately, with its
    # own fresh rng stream (seed 20260630), independent of ns_flag_table's.
    cc_rng = np.random.default_rng(20260630)
    cc_out, cc_coef, cc_sd = count_controlled_report(rows, cc_rng)
    _print_count_controlled(str(out / "results.jsonl"), cc_out, cc_coef, cc_sd)
    (out / "count_controlled.json").write_text(
        json.dumps(cc_out, indent=2), encoding="utf-8")
    print(f"\n[written] {out / 'count_controlled.json'}")
    return 0


def _run_countctl_only(results_path: str) -> int:
    """Bare results.jsonl-path mode: writes only <dir>/count_controlled.json next
    to the given results.jsonl, with no config and no analysis.md/
    analysis_summary.json."""
    rows = _read_jsonl(Path(results_path))
    if not rows:
        print(f"No usable rows in {results_path}.")
        return 0
    rng = np.random.default_rng(20260630)
    out, coef, sd = count_controlled_report(rows, rng)
    _print_count_controlled(results_path, out, coef, sd)
    outpath = str(results_path).replace("results.jsonl", "count_controlled.json")
    Path(outpath).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {outpath}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="NS-vs-NPE benchmark analysis + count-controlled evidence")
    ap.add_argument("--config", help="ns_bench config (e.g. configs/ns_bench.yaml); "
                     "runs the full analysis and writes analysis.md, "
                     "analysis_summary.json and count_controlled.json")
    ap.add_argument("results_path", nargs="?", default=None,
                     help="results.jsonl path for a count-controlled-only run "
                     "(no config); writes only count_controlled.json next to it")
    args = ap.parse_args(argv)

    if args.config:
        return _run_full(args.config)
    if args.results_path:
        return _run_countctl_only(args.results_path)
    ap.error("either --config <ns_bench.yaml> or a results.jsonl path is required")


if __name__ == "__main__":
    raise SystemExit(main())
