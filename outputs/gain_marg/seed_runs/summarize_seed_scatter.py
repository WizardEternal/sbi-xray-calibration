"""Seed-to-seed scatter of the bright paired gain-shift photon-index bias.

Reads the four independent bright evaluations that exist for the gain-marginalized
flow and reports their mean and standard deviation. Each is the paired
(gain minus clean) Gamma offset at g=1.03 over N=500 common-parameter pairs, but
they were produced by two independent implementations and differ in the point
estimate used:

  outputs/gain_marg/paired_gain_bias_bright.json
      paired.gamma_bias_delta.gain_marg.mean, from eval_gainmarg_paired_bright.py
      (posterior mean, 1000 draws per spectrum, seed_theta 20320611)
  seed_runs/seed_11.json        bright_gainmarg.gamma_bias
  seed_runs/seed_137.json       gainmarg_bright.gamma_bias
  seed_runs/seed_20260724_gainmarg_bright.json   gamma_bias_mean
      the three independent re-implementations (eval_bright_seed.py and two
      siblings), each with its own training seed and its own evaluation code.

The three re-implementations use the posterior median rather than the mean, so
the spread below mixes training-seed scatter with a small estimator difference.
Both are real sources of variation in the reported bright number and neither is
separable from the four runs alone.

Run (repo venv, from repo root):
    .venv\\Scripts\\python.exe outputs\\gain_marg\\seed_runs\\summarize_seed_scatter.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GM = HERE.parent

SOURCES = [
    ("eval_gainmarg_paired_bright (posterior mean)",
     GM / "paired_gain_bias_bright.json",
     lambda d: d["paired"]["gamma_bias_delta"]["gain_marg"]["mean"]),
    ("independent seed 11 (posterior median)",
     HERE / "seed_11.json", lambda d: d["bright_gainmarg"]["gamma_bias"]),
    ("independent seed 137 (posterior median)",
     HERE / "seed_137.json", lambda d: d["gainmarg_bright"]["gamma_bias"]),
    ("independent seed 20260724 (posterior median)",
     HERE / "seed_20260724_gainmarg_bright.json", lambda d: d["gamma_bias_mean"]),
]

FIXED = [
    ("eval_gainmarg_paired_bright (posterior mean)",
     GM / "paired_gain_bias_bright.json",
     lambda d: d["paired"]["gamma_bias_delta"]["fixed"]["mean"]),
    ("independent seed 11 (posterior median)",
     HERE / "seed_11.json", lambda d: d["bright_fixed"]["gamma_bias"]),
    ("independent seed 137 (posterior median)",
     HERE / "seed_137.json", lambda d: d["fixed_bright"]["gamma_bias"]),
    ("independent seed 20260724 (posterior median)",
     HERE / "seed_20260724_fixed_bright.json", lambda d: d["gamma_bias_mean"]),
]


def report(title, sources):
    print(title)
    vals = []
    for label, path, get in sources:
        v = float(get(json.loads(path.read_text())))
        vals.append(v)
        print(f"  {v:+.6f}   {label}")
    a = np.array(vals)
    print(f"  mean {a.mean():+.6f}   sd(ddof=1) {a.std(ddof=1):.6f}   "
          f"range {a.min():+.6f} to {a.max():+.6f}   n={a.size}")
    print()
    return a


if __name__ == "__main__":
    gm = report("Bright, gain-marginalized flow, paired Gamma offset at g=1.03:", SOURCES)
    fx = report("Bright, fixed-response flow, same quantity:", FIXED)
    both = np.concatenate([gm, fx])
    print(f"All eight bright cells: mean {both.mean():+.6f}  sd(ddof=1) "
          f"{both.std(ddof=1):.6f}  range {both.min():+.6f} to {both.max():+.6f}")
