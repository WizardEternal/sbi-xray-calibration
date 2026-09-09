r"""Figure for E6: distribution of the Fisher-metric invisibility index I per
misspecification family x count level. Reads
outputs/mechanism/mechanism_index_results.json (written by
run_mechanism_index.py) and writes invisibility_index.{png,pdf}.

    .venv\Scripts\python.exe outputs\mechanism\make_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "mechanism"

# repo-standard Okabe-Ito family palette (matches scripts/make_money_plot.py
# FAMILY_STYLE so this figure reads consistently with the rest of the note).
FAMILY_COLOR = {
    "B1": "#0072B2",   # blue
    "B2": "#009E73",   # bluish green
    "B3": "#CC79A7",   # reddish purple
    "B4": "#999999",   # neutral gray
}
FAMILY_LABEL = {
    "B1": "B1\nFe-K line",
    "B2": "B2\npartial cover",
    "B3": "B3\nwrong continuum",
    "B4": "B4\ngain shift",
}
FAMILIES = ["B1", "B2", "B3", "B4"]
LEVELS = ["medium", "bright"]
LEVEL_ALPHA = {"medium": 0.55, "bright": 0.95}
LEVEL_HATCH = {"medium": "///", "bright": None}


def load_results():
    with open(OUT / "mechanism_index_results.json") as f:
        return json.load(f)


def main():
    d = load_results()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    box_width = 0.32
    group_gap = 1.0
    positions = []
    data = []
    colors = []
    hatches = []
    alphas = []
    tick_pos = []
    tick_lab = []

    for gi, fam in enumerate(FAMILIES):
        base = gi * group_gap
        tick_pos.append(base)
        tick_lab.append(FAMILY_LABEL[fam])
        for li, level in enumerate(LEVELS):
            I = np.array([v for v in d["per_theta"][level][fam]["I"] if v is not None])
            pos = base + (li - 0.5) * (box_width + 0.06)
            positions.append(pos)
            data.append(I)
            colors.append(FAMILY_COLOR[fam])
            hatches.append(LEVEL_HATCH[level])
            alphas.append(LEVEL_ALPHA[level])

    bp = ax.boxplot(
        data, positions=positions, widths=box_width, patch_artist=True,
        showfliers=True, flierprops=dict(marker="o", markersize=2.5, alpha=0.5,
                                          markeredgecolor="none"),
        medianprops=dict(color="black", lw=1.6), whiskerprops=dict(color="#333333"),
        capprops=dict(color="#333333"), boxprops=dict(lw=1.0, edgecolor="#333333"),
    )
    for patch, c, h, a in zip(bp["boxes"], colors, hatches, alphas):
        patch.set_facecolor(c)
        patch.set_alpha(a)
        if h:
            patch.set_hatch(h)
            patch.set_edgecolor("#333333")

    ax.axhline(1.0, color="#444444", lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel(r"invisibility index  $I = \Vert P_A b\Vert^2 / \Vert b\Vert^2$")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_title("Fisher-metric invisibility index by misspecification family and count level\n"
                 r"($I\to1$: reabsorbable by refitting $\theta$; $I\to0$: residual survives)",
                 fontsize=10.5)

    # legend for level (hatch = medium, solid = bright), family colors are the
    # x-axis groups themselves so no separate family legend is needed.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#bbbbbb", edgecolor="#333333", hatch="///", alpha=LEVEL_ALPHA["medium"],
              label="medium (353.4 s, ~1000 ct)"),
        Patch(facecolor="#bbbbbb", edgecolor="#333333", alpha=LEVEL_ALPHA["bright"],
              label="bright (3534.0 s, ~10000 ct)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5, framealpha=0.9)

    ax.grid(axis="y", color="#dddddd", lw=0.7, zorder=-1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    png = OUT / "invisibility_index.png"
    pdf = OUT / "invisibility_index.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
