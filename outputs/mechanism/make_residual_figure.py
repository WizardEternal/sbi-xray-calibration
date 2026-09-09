r"""E6b figure: invisibility index I (fraction) next to residual Fisher power R
(absolute out-of-span scale, log y). Reads mechanism_index_results.json and
residual_power_results.json (both written by the sibling run_*.py scripts)
and writes residual_power.{png,pdf}.

Left panel reproduces the I story from invisibility_index.png (fraction,
level-invariant). Right panel adds R = (1-I)*||b||^2 per family x level on a
log y-axis, which is NOT level-invariant (grows ~linearly with exposure) --
this is the panel that explains why B2/B3 (I ~ 0.99-0.9999, "more invisible"
than B4 by the fraction alone) are nonetheless measurably detected while B4
is not: their absolute residual power sits well above B4's.

    .venv\Scripts\python.exe outputs\mechanism\make_residual_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "mechanism"

# same Okabe-Ito colorblind-safe family palette as invisibility_index.png /
# scripts/make_money_plot.py FAMILY_STYLE, kept identical across figures.
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


def load_json(name):
    with open(OUT / name) as f:
        return json.load(f)


def boxplot_panel(ax, mech, resid, quantity, ylabel, title, logy=False):
    """quantity in {'I', 'R'}. For 'I' uses mech (mechanism_index_results.json
    per_theta), for 'R' uses resid (residual_power_results.json per_theta)."""
    src = mech if quantity == "I" else resid
    box_width = 0.32
    group_gap = 1.0
    positions, data, colors, hatches, alphas = [], [], [], [], []
    tick_pos, tick_lab = [], []

    for gi, fam in enumerate(FAMILIES):
        base = gi * group_gap
        tick_pos.append(base)
        tick_lab.append(FAMILY_LABEL[fam])
        for li, level in enumerate(LEVELS):
            vals = np.array([v for v in src["per_theta"][level][fam][quantity] if v is not None])
            pos = base + (li - 0.5) * (box_width + 0.06)
            positions.append(pos)
            data.append(vals)
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

    if logy:
        ax.set_yscale("log")
    else:
        ax.axhline(1.0, color="#444444", lw=1.0, ls=(0, (4, 3)), zorder=0)
        ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_title(title, fontsize=10.0)
    ax.grid(axis="y", color="#dddddd", lw=0.7, zorder=-1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    mech = load_json("mechanism_index_results.json")
    resid = load_json("residual_power_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    boxplot_panel(
        axes[0], mech, resid, "I",
        ylabel=r"invisibility index  $I = \Vert P_A b\Vert^2 / \Vert b\Vert^2$",
        title="fraction reabsorbable by refitting $\\theta$\n"
              r"(level-invariant $\to$ B2/B3 look ``more invisible'' than B4)",
        logy=False,
    )
    boxplot_panel(
        axes[1], mech, resid, "R",
        ylabel=r"residual power  $R = (1-I)\Vert b\Vert^2$  (Fisher-whitened, log scale)",
        title="absolute out-of-span residual (grows $\\propto$ exposure)\n"
              "B2/B3 sit well above B4 despite higher $I$",
        logy=True,
    )

    fig.suptitle(
        "E6b: invisibility fraction $I$ vs. absolute residual power $R$, by family and count level",
        fontsize=12, y=1.02,
    )

    legend_handles = [
        Patch(facecolor="#bbbbbb", edgecolor="#333333", hatch="///", alpha=LEVEL_ALPHA["medium"],
              label="medium (353.4 s, ~1000 ct)"),
        Patch(facecolor="#bbbbbb", edgecolor="#333333", alpha=LEVEL_ALPHA["bright"],
              label="bright (3534.0 s, ~10000 ct)"),
    ]
    axes[0].legend(handles=legend_handles, loc="lower left", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    png = OUT / "residual_power.png"
    pdf = OUT / "residual_power.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
