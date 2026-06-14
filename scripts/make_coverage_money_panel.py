"""Phase-3 money plot: the coverage-vs-nominal panel.

Reads the finished Phase-3 calibration artifacts in
``outputs/calibration/{faint,medium,bright}/`` and produces

    outputs/diagnostics/coverage_money_panel.png

a single panel of coverage-vs-nominal curves, one count level per subplot, raw
NPE vs the BETTER recalibration at that level, with the diagonal reference. The
"coverage" plotted is the mean over the 5 Model-A parameters of the per-parameter
empirical equal-tailed credible-interval coverage (the same numbers the
before/after npz stores), so a curve ON the diagonal = perfectly calibrated,
BELOW = overconfident (under-covers), ABOVE = conservative.

Which recalibration is plotted per level (chosen by smallest mean |emp-nominal|
deviation, read from each level's summary.json and printed):
  * faint / medium -- already near-diagonal; conformal is a near-no-op and is the
    better (or tied) recalibration, so conformal is drawn and LABELLED.
  * bright -- strongly overconfident raw; conformal repairs it best (the IS
    refinement is starved by ~97% low-ESS at this count level, see RESULTS.md),
    so conformal is drawn and labelled.
The label on each panel states explicitly which recalibration is shown and its
mean-abs-deviation, so the figure is self-documenting.

Colorblind-safe: Wong/Okabe-Ito palette (blue = raw, vermilion = recalibrated),
distinct markers + linestyles so the two curves are separable without color.
dpi=220.

Regenerable, no config: ::

    .venv\\Scripts\\python.exe scripts\\make_coverage_money_panel.py

It READS outputs/calibration/<level>/{coverage_before_after,is_coverage}.npz and
summary.json and WRITES only outputs/diagnostics/coverage_money_panel.png. It
never touches outputs/detect/ or outputs/models/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEVELS = ["faint", "medium", "bright"]

# Okabe-Ito colorblind-safe palette
C_RAW = "#0072B2"      # blue
C_RECAL = "#D55E00"    # vermilion
C_DIAG = "#444444"     # neutral gray


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _calib_dir(level: str) -> Path:
    return _repo_root() / "outputs" / "calibration" / level


def _mean_abs_dev(nominal: np.ndarray, cov_mean: np.ndarray) -> float:
    return float(np.mean(np.abs(cov_mean - nominal)))


def load_level(level: str):
    """Return a dict with nominal levels and the mean-over-params coverage curves
    (raw, conformal, IS-all, IS-okess) for one count level, plus the coverage
    deviations and low-ESS fraction read from summary.json."""
    cdir = _calib_dir(level)
    cov = np.load(cdir / "coverage_before_after.npz", allow_pickle=True)
    isc = np.load(cdir / "is_coverage.npz", allow_pickle=True)
    with open(cdir / "summary.json") as f:
        summ = json.load(f)

    nominal = np.asarray(cov["nominal_levels"], dtype=float)
    # mean over the 5 parameters -> one coverage curve per method
    raw = np.asarray(cov["cov_raw"]).mean(axis=1)
    conformal = np.asarray(cov["cov_recal"]).mean(axis=1)
    is_all = np.asarray(isc["cov_is_all"]).mean(axis=1)
    is_okess = np.asarray(isc["cov_is_okess"]).mean(axis=1)

    return {
        "level": level,
        "nominal": nominal,
        "raw": raw,
        "conformal": conformal,
        "is_all": is_all,
        "is_okess": is_okess,
        "counts": float(summ["median_total_counts"]),
        "raw_dev": _mean_abs_dev(nominal, raw),
        "conformal_dev": _mean_abs_dev(nominal, conformal),
        "is_all_dev": _mean_abs_dev(nominal, is_all),
        "low_ess_frac": float(
            summ["is_refinement"]["cov_testset_low_ess_fraction"]),
    }


def choose_recal(d: dict):
    """Pick the recalibration to plot for a level: the one with the smaller
    mean-abs-deviation between conformal and IS-refined(all-cases). Returns
    (curve, label, dev, method_key)."""
    if d["conformal_dev"] <= d["is_all_dev"]:
        return d["conformal"], "conformal-recalibrated", d["conformal_dev"], "conformal"
    return d["is_all"], "IS-refined", d["is_all_dev"], "is"


def main():
    data = [load_level(lv) for lv in LEVELS]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharex=True, sharey=True)

    for ax, d in zip(axes, data):
        recal_curve, recal_label, recal_dev, _ = choose_recal(d)

        # diagonal reference
        ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=C_DIAG, lw=1.3,
                label="perfect calibration", zorder=1)

        # raw NPE
        ax.plot(d["nominal"], d["raw"], marker="o", ms=6, mfc="white", mew=1.6,
                color=C_RAW, ls="-", lw=2.0, label="raw NPE", zorder=3)

        # better recalibration for this level
        ax.plot(d["nominal"], recal_curve, marker="s", ms=5.5, mfc="white",
                mew=1.6, color=C_RECAL, ls="--", lw=2.0,
                label=recal_label, zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25, lw=0.7)
        ax.set_xlabel("nominal credibility level")

        title = (f"{d['level']}  (~{d['counts']:.0f} counts)")
        ax.set_title(title, fontsize=12, pad=8)

        # annotate the deviations + low-ESS in-panel (self-documenting)
        verdict = ("near-diagonal" if d["raw_dev"] < 0.05
                   else "over-confident")
        txt = (f"raw mean|$\\Delta$|={d['raw_dev']:.3f}  ({verdict})\n"
               f"{recal_label}: {recal_dev:.3f}\n"
               f"IS low-ESS frac: {d['low_ess_frac']:.0%}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc",
                          alpha=0.92))

    axes[0].set_ylabel("empirical coverage (mean over 5 params)")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle("NPE coverage vs nominal, raw and recalibrated (one flow per level)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = _repo_root() / "outputs" / "diagnostics" / "coverage_money_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] {out}")
    print("\nper-level deviations (mean |emp-nominal|, mean over params):")
    print("| level | ~counts | raw | conformal | IS-all | low-ESS frac | plotted recal |")
    print("|---|---|---|---|---|---|---|")
    for d in data:
        _, lbl, dev, key = choose_recal(d)
        print(f"| {d['level']} | {d['counts']:.0f} | {d['raw_dev']:.3f} | "
              f"{d['conformal_dev']:.3f} | {d['is_all_dev']:.3f} | "
              f"{d['low_ess_frac']:.0%} | {key} ({dev:.3f}) |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
