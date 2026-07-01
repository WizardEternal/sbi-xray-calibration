"""Count-controlled nested-sampling evidence analysis.

logZ is the log marginal likelihood of one dataset, so it scales ~linearly with
total counts (logZ ~ a*log10(counts)+b here), and a count "level" spans a ~30x
count range. A level-matched mean(logZ_mis) - mean(logZ_clean) over unmatched
spectra is therefore confounded by which spectra land in each group.

This script instead fits logZ vs log10(counts) on the CLEAN spectra (all levels),
then reports each misspecified cell's mean residual from that clean trend, with a
bootstrap CI and a per-spectrum breakdown. A real model error sits BELOW the clean
trend (negative residual); a count artifact sits on it.

Usage:
    python scripts/analyze_ns_bench_countctl.py outputs/ns_bench/results.jsonl
    python scripts/analyze_ns_bench_countctl.py outputs/ns_bench_nicer/results.jsonl
"""
from __future__ import annotations
import json, sys
import numpy as np


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


def fit_clean_trend(rows, rng):
    cc = np.array([r["n_counts"] for r in rows if r["family"] == "clean"], float)
    cz = np.array([r["ns"]["logz"] for r in rows if r["family"] == "clean"], float)
    X = np.vstack([np.log10(cc), np.ones_like(cc)]).T
    coef, *_ = np.linalg.lstsq(X, cz, rcond=None)
    resid = cz - X @ coef
    sd = resid.std(ddof=2)
    # bootstrap the fit
    boot = []
    n = len(cc)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        c, *_ = np.linalg.lstsq(X[idx], cz[idx], rcond=None)
        boot.append(c)
    boot = np.array(boot)
    return coef, sd, boot, len(cc)


def cell_residual(rows, coef, sd, boot, fam, level, rng):
    sel = [r for r in rows if r["family"] == fam and r["level"] == level]
    if not sel:
        return None
    gc = np.array([r["n_counts"] for r in sel], float)
    gz = np.array([r["ns"]["logz"] for r in sel], float)
    pred = coef[0] * np.log10(gc) + coef[1]
    res = gz - pred
    # bootstrap CI on the mean residual, propagating clean-trend uncertainty
    bmeans = []
    n = len(res)
    for b in range(len(boot)):
        predb = boot[b, 0] * np.log10(gc) + boot[b, 1]
        idx = rng.integers(0, n, n)
        bmeans.append((gz[idx] - predb[idx]).mean())
    lo, hi = np.percentile(bmeans, [2.5, 97.5])
    # significance from the cell's OWN residual scatter (honest per-cell t-statistic).
    # dividing by the clean-trend SD understates the noise badly for high-count cells,
    # where the per-spectrum penalty varies by hundreds of nats; the bootstrap CI is
    # the robust statement and the one the note reports.
    cell_sd = res.std(ddof=1)
    sig = res.mean() / (cell_sd / np.sqrt(n)) if n > 1 and cell_sd > 0 else float("nan")
    return dict(fam=fam, level=level, n=n, mean_resid=float(res.mean()),
                ci=[float(lo), float(hi)], sigma=float(sig),
                per_spec=[round(float(x), 1) for x in res],
                counts=[int(c) for c in gc])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/ns_bench/results.jsonl"
    rng = np.random.default_rng(20260630)
    rows = load(path)
    coef, sd, boot, n_clean = fit_clean_trend(rows, rng)
    print(f"# Count-controlled NS analysis: {path}")
    print(f"clean trend: logZ = {coef[0]:.2f}*log10(counts) + {coef[1]:.2f}  "
          f"(n_clean={n_clean}, resid sd={sd:.2f})")
    print(f"{'cell':22s} {'n':>3s} {'mean_resid':>11s} {'95% CI':>20s} {'sigma':>7s}  verdict")
    cells = [("B1", "medium"), ("B1", "bright"), ("B4", "medium"), ("B4", "bright")]
    out = {"trend": {"slope": float(coef[0]), "intercept": float(coef[1]),
                     "resid_sd": float(sd), "n_clean": n_clean}, "cells": []}
    for fam, lvl in cells:
        r = cell_residual(rows, coef, sd, boot, fam, lvl, rng)
        if r is None:
            continue
        # caught = CI entirely below 0 (real evidence penalty)
        caught = r["ci"][1] < 0
        verdict = "CAUGHT (penalty)" if caught else ("null/boost" )
        print(f"{fam+'/'+lvl:22s} {r['n']:3d} {r['mean_resid']:+11.1f} "
              f"[{r['ci'][0]:+7.1f},{r['ci'][1]:+7.1f}] {r['sigma']:+7.1f}  {verdict}")
        r["caught"] = bool(caught)
        out["cells"].append(r)
    outpath = path.replace("results.jsonl", "count_controlled.json")
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
