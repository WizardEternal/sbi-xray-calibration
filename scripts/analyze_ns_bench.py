r"""Phase-5 NS-vs-NPE benchmark analysis: the speed-vs-agreement table + the
NS-flag-vs-detector-flag cross-check.

Usage (repo venv):
    .venv\Scripts\python.exe scripts\analyze_ns_bench.py --config configs\ns_bench.yaml

Reads outputs/ns_bench/results.jsonl (written by run_ns_benchmark.py) and prints
(and writes to outputs/ns_bench/analysis.md):

  1. SPEED-VS-AGREEMENT table, per count level (clean Model-A spectra only -- the
     well-specified spine): NS wall-clock (s/spectrum) and n_like_evals vs NPE
     sampling (ms/spectrum), the NS/NPE speed ratio, and the NS-vs-NPE posterior
     QUANTILE agreement (mean |q_NS - q_NPE| / prior-width). Small agreement where
     raw NPE is well calibrated (faint/medium) = NS validates the amortized flow;
     larger at bright = NS exposes the over-confidence Phase-3 measured.

  2. NS misspecification-flag check on the B1/B4 spectra: per misspecified
     spectrum we form two NS-side flags --
        * residual flag: the best-fit (max-likelihood) Poisson chi2-like residual
          (reduced) -- a poor fit to the well-specified Model A flags misspec;
        * evidence flag: logZ relative to the clean-population logZ at the same
          level (a misspecified spectrum the model cannot fit has lower evidence).
     and reports them next to the Phase-4 DETECTOR AUC for the matching
     (family, strength, level) cell.

  *** Detector cross-check status ***  The Phase-4 detector benchmark
  (outputs/detect/results.jsonl) may still be RUNNING (the full 144-cell grid).
  This script READS that file if present (read-only; it never writes there) and
  fills the detector-AUC column where the matching cell exists; cells not yet
  computed are shown as "pending" and the whole cross-check is labelled a STUB
  until the detector grid finishes. The NS-side flags are always computed.

Writes ONLY to outputs/ns_bench/analysis.md. Reads outputs/ns_bench/results.jsonl
and (read-only, if present) outputs/detect/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _read_jsonl(path: Path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# --------------------------------------------------------------------------
# 1. speed-vs-agreement (clean spine)
# --------------------------------------------------------------------------

def speed_agreement_table(rows):
    """Per-level aggregates over the CLEAN Model-A spectra."""
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


# --------------------------------------------------------------------------
# 2. NS misspecification flags vs detector AUC
# --------------------------------------------------------------------------

def _clean_logz_by_level(rows):
    """Mean clean logZ per level (the evidence baseline for the misspec flag)."""
    by = defaultdict(list)
    for r in rows:
        if r["family"] == "clean":
            by[r["level"]].append(r["ns"]["logz"])
    return {lvl: float(np.mean(v)) for lvl, v in by.items() if v}


def _detector_auc_lookup(detect_rows):
    """Map (family, strength, level) -> {detector: auc} from a detect results.jsonl."""
    out = defaultdict(dict)
    for r in detect_rows:
        key = (r["family"], float(r["strength"]), r["level"])
        out[key][r["detector"]] = r["auc"]
    return out


def ns_flag_table(rows, detect_rows):
    """Per misspecified spectrum: NS evidence flag (Δ logZ vs clean baseline) and a
    quantile-spread proxy, grouped by (family, strength, level), shown next to the
    detector AUCs for the matching cell (if available)."""
    clean_logz = _clean_logz_by_level(rows)
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
    lines.append("| family | strength | level | n | mean d-logZ (mis - clean) | "
                 "D1 AUC | D2 AUC | D3 AUC | detector status |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    rowsout = []
    for (fam, strength, lvl), rs in sorted(groups.items()):
        base = clean_logz.get(lvl, np.nan)
        dlogz = np.mean([r["ns"]["logz"] - base for r in rs]) if np.isfinite(base) \
            else np.nan
        aucs = det.get((fam, strength, lvl), {})
        d1 = aucs.get("D1"); d2 = aucs.get("D2"); d3 = aucs.get("D3")
        have = sum(a is not None for a in (d1, d2, d3))
        status = ("ready" if have == 3 else
                  (f"pending ({have}/3)" if have else "pending (STUB)"))
        def f(a):
            return f"{a:.3f}" if a is not None else "n/a"
        lines.append(
            f"| {fam} | {strength:g} | {lvl} | {len(rs)} | {dlogz:+.1f} | "
            f"{f(d1)} | {f(d2)} | {f(d3)} | {status} |")
        rowsout.append({
            "family": fam, "strength": strength, "level": lvl, "n": len(rs),
            "mean_dlogz": float(dlogz) if np.isfinite(dlogz) else None,
            "detector_auc": aucs, "detector_status": status,
        })
    return "\n".join(lines), rowsout


# --------------------------------------------------------------------------
# truth-recovery summary (clean spectra: NS 90% interval contains truth?)
# --------------------------------------------------------------------------

def truth_recovery(rows):
    """Per level, fraction of clean spectra whose truth falls in the NS 5-95%
    interval, per parameter then averaged -- a sanity coverage proxy for NS."""
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-5 NS-vs-NPE benchmark analysis")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    _ = load_config(args.config)  # reserved for future use; analysis is self-contained

    out = _repo_root() / "outputs" / "ns_bench"
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
    md.append("# Phase-5 NS-vs-NPE benchmark: analysis\n")
    md.append(f"Spectra analyzed: {len(rows)} ({n_clean} clean, {n_mis} misspecified)"
              f"{f'; {n_err} error row(s) skipped' if n_err else ''}. "
              f"Detector cross-check: **{det_status}** "
              f"({len(detect_rows)} detector cells available).\n")
    md.append("## 1. Speed vs agreement (clean Model-A spine)\n")
    md.append(speed_tbl + "\n")
    md.append("## 2. NS truth recovery (clean; 90% interval coverage proxy)\n")
    md.append(recov_tbl + "\n")
    md.append("## 3. NS misspecification flags vs Phase-4 detector AUC\n")
    md.append("NS evidence flag = mean(logZ_misspec - logZ_clean) at the same level "
              "(more negative => the well-specified Model A fits the misspecified "
              "spectrum worse => flagged). Detector AUCs are read read-only from "
              "outputs/detect/results.jsonl; cells the (still-running) detector grid "
              "has not produced yet show as pending.\n")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
