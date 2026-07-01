"""Supporting README figures for Phase 6 (read-only over committed artifacts).

Produces three figures into outputs/diagnostics/ :

  detector_auc_grid.png
      A heatmap of detection ROC AUC, 3 detectors (D1 PPC / D2 emb-OOD /
      D3 marginal-C2ST) x 4 families (B1 line / B2 partial-cover / B3 brems /
      B4 gain), one panel per count level, each cell showing the BEST AUC
      across that family's strength grid for that (detector, family). Diverging
      colormap centred at 0.5 (chance); annotated with the numeric AUC. Reads
      outputs/detect/results.jsonl + configs/detect.yaml.

  dgamma_silent_failure.png
      The consequence figure: signed Gamma-bias (DeltaGamma) imposed by an
      UNDETECTED / weakly-detected misspecification, as a function of strength,
      one line per count level, for B1 (the unmodeled Fe-K line). Reads
      outputs/detect/consequence.jsonl. The point: the line silently biases the
      photon index, and the signed bias grows with counts even where the
      detector AUC has not yet reached 1.

  tarp_bright_curve.png
      The bright-level TARP expected-coverage (ECP-vs-alpha) curve, with the
      |ECP-alpha| area shaded. The figure: the SIGNED area-to-curve
      (ATC ~ -0.002) cancels because the curve bows above the diagonal at
      alpha<0.5 and below at alpha>0.5, hiding the over-confidence; the UNSIGNED
      abs-area (0.053) and max|ECP-alpha| (0.102 at alpha~0.19) catch it. Reads
      outputs/calibration/bright/tarp.npz. The lesson is "read the curve / use
      abs-area or KS", not "TARP is insufficient".

Writes ONLY outputs/diagnostics/. No model loading, no heavy compute.

    .venv\\Scripts\\python.exe scripts\\make_support_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


LEVELS = ["faint", "medium", "bright"]
FAMILIES = ["B1", "B2", "B3", "B4"]
DETECTORS = ["D1", "D2", "D3"]
FAM_LABEL = {
    "B1": "B1\nFe-K line",
    "B2": "B2\npartial-cover",
    "B3": "B3\nbrems cont.",
    "B4": "B4\ngain shift",
}
DET_LABEL = {"D1": "D1 PPC", "D2": "D2 emb-OOD", "D3": "D3 marg-C2ST"}

# Okabe-Ito for the dGamma lines
LEVEL_COLOR = {"faint": "#56B4E9", "medium": "#009E73", "bright": "#D55E00"}
LEVEL_MARKER = {"faint": "o", "medium": "s", "bright": "^"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def best_auc_grid(results):
    """level -> (det x family) array of the best AUC across each family's grid."""
    out = {}
    for lv in LEVELS:
        grid = np.full((len(DETECTORS), len(FAMILIES)), np.nan)
        for i, det in enumerate(DETECTORS):
            for j, fam in enumerate(FAMILIES):
                cells = [r["auc"] for r in results
                         if r["level"] == lv and r["family"] == fam
                         and r["detector"] == det]
                if cells:
                    grid[i, j] = max(cells)
        out[lv] = grid
    return out


def make_auc_grid(results, out_path: Path):
    grids = best_auc_grid(results)
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(13.5, 4.2), squeeze=False,
                             constrained_layout=True)
    # linear norm over the full AUC range; RdBu_r puts chance (0.5) at white. A
    # two-slope norm here made the colourbar nonlinear (0.5-1.0 and 0.4-0.5 rendered
    # as equal spans), which misreads the scale.
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdBu_r")
    im = None
    for ax, lv in zip(axes[0], LEVELS):
        g = grids[lv]
        im = ax.imshow(g, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=9)
        ax.set_yticks(range(len(DETECTORS)))
        ax.set_yticklabels([DET_LABEL[d] for d in DETECTORS], fontsize=9)
        ax.set_title(lv, fontsize=12)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g[i, j]
                if np.isnan(v):
                    continue
                txt_color = "white" if (v > 0.85 or v < 0.45) else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9.5, color=txt_color,
                        fontweight="bold" if v >= 0.9 else "normal")
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("best ROC AUC over strength grid (0.5 = chance)", fontsize=9)
    fig.suptitle("Detector x family best AUC (clean vs misspecified), per count level",
                 fontsize=12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_dgamma_fig(cons, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axhline(0.0, color="#444444", lw=1.0, ls=(0, (4, 3)), zorder=1,
               label="no bias")
    for lv in LEVELS:
        rows = sorted([r for r in cons if r["level"] == lv],
                      key=lambda r: r["strength"])
        if not rows:
            continue
        x = [r["strength"] for r in rows]
        y = [r["dGamma_bias_mean"] for r in rows]
        yerr = [r["dGamma_bias_std"] / np.sqrt(max(r["n"], 1)) for r in rows]
        ax.errorbar(x, y, yerr=yerr, marker=LEVEL_MARKER[lv], ms=7,
                    color=LEVEL_COLOR[lv], lw=2.0, capsize=3,
                    label=f"{lv}")
    ax.set_xscale("log")
    ax.set_xlabel("unmodeled Fe-K line norm (B1 strength, ~ equivalent width)")
    ax.set_ylabel(r"mean photon-index bias  $\langle\Delta\Gamma\rangle$  "
                  r"($\hat\Gamma - \Gamma_{\rm true}$)")
    ax.set_title("Inferred photon-index bias from an unmodeled Fe-K line vs counts",
                 fontsize=11.5)
    ax.grid(alpha=0.25, lw=0.7)
    ax.legend(title="count level", fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_tarp_bright_curve(tarp_npz: Path, out_path: Path):
    """Bright TARP ECP-vs-alpha curve with the |ECP-alpha| area shaded.

    The signed ATC cancels (the curve bows above the diagonal at
    alpha<0.5 and below at alpha>0.5) and hides the over-confidence; the unsigned
    abs-area and max|ECP-alpha| catch it. All numbers are recomputed from the npz.
    """
    d = np.load(tarp_npz, allow_pickle=True)
    ecp = np.asarray(d["ecp"]); alpha = np.asarray(d["alpha"])
    atc = float(d["atc"])
    absdiff = np.abs(ecp - alpha)
    abs_area = float(np.trapezoid(absdiff, alpha))
    max_dev = float(np.max(absdiff))
    amax = float(alpha[np.argmax(absdiff)])

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color="#444444", lw=1.3,
            label="ideal (ECP = nominal)")
    ax.plot(alpha, ecp, color="#D55E00", lw=2.3, label="bright ECP (~9982 ct)")
    ax.fill_between(alpha, alpha, ecp, color="#D55E00", alpha=0.18,
                    label=f"|ECP−α|, abs-area {abs_area:.3f}")
    ax.annotate(f"max|ECP−α| = {max_dev:.3f}\nat α = {amax:.2f}",
                xy=(amax, ecp[np.argmax(absdiff)]), xytext=(0.40, 0.18),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))
    ax.text(0.22, 0.42, "bows ABOVE\n(α<0.5)", fontsize=8, color="#7a3500",
            ha="center")
    ax.text(0.78, 0.66, "bows below\n(α>0.5)", fontsize=8, color="#7a3500",
            ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", "box")
    ax.grid(alpha=0.25, lw=0.7)
    ax.set_xlabel("nominal credibility level α")
    ax.set_ylabel("expected coverage probability (ECP)")
    ax.set_title("Bright TARP: signed ATC vs unsigned abs-area\n"
                 f"signed ATC = {atc:+.3f}  vs  abs-area = "
                 f"{abs_area:.3f}, max|dev| = {max_dev:.3f}", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    root = _repo_root()
    diag = root / "outputs" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    results = _read_jsonl(root / "outputs" / "detect" / "results.jsonl")
    cons = _read_jsonl(root / "outputs" / "detect" / "consequence.jsonl")

    p1 = make_auc_grid(results, diag / "detector_auc_grid.png")
    p2 = make_dgamma_fig(cons, diag / "dgamma_silent_failure.png")
    print(f"[done] {p1}")
    print(f"[done] {p2}")

    tarp_npz = root / "outputs" / "calibration" / "bright" / "tarp.npz"
    if tarp_npz.exists():
        p3 = make_tarp_bright_curve(tarp_npz, diag / "tarp_bright_curve.png")
        print(f"[done] {p3}")
    else:
        print(f"[skip] {tarp_npz} not found -> tarp_bright_curve.png not built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
