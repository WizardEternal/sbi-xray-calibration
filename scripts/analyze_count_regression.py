r"""Regress the paired gain-null log-evidence differences on the count ratio.

The paired gain check (scripts/paired_ns_gain_check.py, output
outputs/ns_bench/paired_gain_check.jsonl) folds the same clean model through the
nominal and through a 3 per cent gain-shifted response, draws one Poisson
realization of each with common random numbers, and runs nested sampling on both.
The two members of a pair therefore differ in total counts by a Poisson draw, and
log Z depends steeply on counts: the clean subset in
outputs/ns_bench/count_controlled.json gives log Z = -116.9 log10(counts) + 91.1
with correlation -0.99.

This script measures how much of the paired difference that count mismatch
explains. It fits, by ordinary least squares over the 12 pairs,

    d_paired = a + b * log(counts_gain / counts_clean)

with a natural log, so b is nats per e-fold of the count ratio, and reports:

  * b and its standard error,
  * R^2, i.e. the fraction of paired variance the count ratio accounts for,
  * a two-sided permutation p-value for b, shuffling the count ratios against the
    differences (seeded, so it is reproducible),
  * the intercept a, which is the count-adjusted gain effect, with its standard
    error and its t p-value,
  * the residual standard deviation, the scatter left once the count ratio is
    removed.

Reads only outputs/ns_bench/paired_gain_check.jsonl and
outputs/ns_bench/count_controlled.json, both committed. Writes nothing.

Run (repo venv, from repo root):
    .venv\Scripts\python.exe scripts\analyze_count_regression.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ROOT / "outputs" / "ns_bench" / "paired_gain_check.jsonl"
COUNTCTL = ROOT / "outputs" / "ns_bench" / "count_controlled.json"

N_PERM = 200_000
PERM_SEED = 0


def load_pairs():
    rows = [json.loads(ln) for ln in PAIRED.read_text().splitlines() if ln.strip()]
    d = np.array([r["d_paired"] for r in rows], float)
    x = np.log(np.array([r["counts_gain"] for r in rows], float) /
               np.array([r["counts_clean"] for r in rows], float))
    return x, d


def ols(x, y):
    """Slope, intercept and their standard errors, plus R^2 and residual sd."""
    n = x.size
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot
    return {
        "intercept": float(beta[0]),
        "intercept_se": float(np.sqrt(cov[0, 0])),
        "slope": float(beta[1]),
        "slope_se": float(np.sqrt(cov[1, 1])),
        "r2": r2,
        "resid_sd": float(np.std(resid, ddof=2)),
        "dof": dof,
    }


def permutation_p(x, y, n_perm=N_PERM, seed=PERM_SEED):
    """Two-sided permutation p for the slope: shuffle x against y."""
    obs = abs(ols(x, y)["slope"])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        if abs(ols(rng.permutation(x), y)["slope"]) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def t_p(stat, dof):
    """Two-sided t p-value without scipy: incomplete-beta via a series is overkill,
    so use the normal approximation only as a cross-check and report both."""
    from math import erfc, sqrt
    return erfc(abs(stat) / sqrt(2.0))


def main():
    x, d = load_pairs()
    fit = ols(x, d)
    p_perm = permutation_p(x, d)

    print(f"pairs: n={x.size}")
    print(f"log count ratio: mean {x.mean():+.4f}, sd {x.std(ddof=1):.4f}, "
          f"range {x.min():+.4f} to {x.max():+.4f}")
    print(f"paired difference: mean {d.mean():+.4f}, sd {d.std(ddof=1):.4f}")
    print()
    print(f"slope      = {fit['slope']:+.1f} +/- {fit['slope_se']:.1f} nats per e-fold "
          f"(t = {fit['slope'] / fit['slope_se']:+.2f}, dof {fit['dof']})")
    print(f"R^2        = {fit['r2']:.3f}")
    print(f"perm p     = {p_perm:.4f}  ({N_PERM} permutations, seed {PERM_SEED})")
    print(f"intercept  = {fit['intercept']:+.2f} +/- {fit['intercept_se']:.2f} "
          f"(normal-approx p = {t_p(fit['intercept'] / fit['intercept_se'], fit['dof']):.2f})")
    print(f"resid sd   = {fit['resid_sd']:.2f} nats")
    print()

    trend = json.loads(COUNTCTL.read_text())["trend"]
    print(f"clean logZ-counts trend (from count_controlled.json): "
          f"slope {trend['slope']:.1f}, intercept {trend['intercept']:.1f}, "
          f"n_clean {trend['n_clean']}")


if __name__ == "__main__":
    main()
