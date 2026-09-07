"""Independent cross-check (fresh code, not copied from analyze_count_regression.py)
of the paired count-ratio statistics, run on both the old and new 12-pair files.
Uses scipy.stats.linregress for the log-ratio regression (matches the committed
script's specification: d_paired = a + b*log(counts_gain/counts_clean)) and also
reports the plain percent-ratio regression for transparency (different spec, not
expected to match the printed intercept).
"""
import sys, json
import numpy as np
from scipy import stats

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cc = np.array([r["counts_clean"] for r in rows], float)
    cg = np.array([r["counts_gain"] for r in rows], float)
    d = np.array([r["d_paired"] for r in rows], float)
    return cc, cg, d

def report(path):
    cc, cg, d = load(path)
    pct = (cg / cc - 1) * 100
    logr = np.log(cg / cc)
    n = len(d)
    print(f"\n=== {path} (n={n}) ===")
    print(f"percent ratio: median {np.median(pct):+.3f}%  min {pct.min():+.3f}%  max {pct.max():+.3f}%")

    # natural-log regression (the paper's actual spec)
    sl, ic, r, p_sl, se_sl = stats.linregress(logr, d)
    # SE of intercept (linregress doesn't give it directly)
    X = np.column_stack([np.ones(n), logr])
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    resid = d - X @ beta
    dof = n - 2
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se_ic = np.sqrt(cov[0, 0])
    t_ic = beta[0] / se_ic
    p_ic = 2 * stats.t.sf(abs(t_ic), dof)
    print(f"[log-ratio spec, matches committed script] intercept a={beta[0]:+.4f} SE={se_ic:.4f} p={p_ic:.4f}   slope b={sl:+.4f} SE={se_sl:.4f} p={p_sl:.4f}  R2={r**2:.4f}")

    # plain percent-ratio regression (alternate spec, for transparency only)
    slp, icp, rp, p_slp, se_slp = stats.linregress(pct, d)
    Xp = np.column_stack([np.ones(n), pct])
    betap, *_ = np.linalg.lstsq(Xp, d, rcond=None)
    residp = d - Xp @ betap
    s2p = (residp @ residp) / dof
    covp = s2p * np.linalg.inv(Xp.T @ Xp)
    se_icp = np.sqrt(covp[0, 0])
    t_icp = betap[0] / se_icp
    p_icp = 2 * stats.t.sf(abs(t_icp), dof)
    print(f"[percent-linear spec, alternate/NOT the paper's stat] intercept a={betap[0]:+.4f} SE={se_icp:.4f} p={p_icp:.4f}   slope b={betap[1]:+.4f} SE={se_slp:.4f} p={p_slp:.4f}  R2={rp**2:.4f}")

    mean_d = d.mean()
    sem_d = d.std(ddof=1) / np.sqrt(n)
    print(f"plain mean +/- SEM of d_paired: {mean_d:+.4f} +/- {sem_d:.4f}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
