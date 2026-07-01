"""Cross-instrument SBC figure: per-parameter rank-uniformity KS p-value at each
count level, EPIC-pn vs NICER. The bright (10000-count) flow fails on both
responses, while faint/medium mostly pass -- the count-regime miscalibration.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "outputs" / "diagnostics"
PARAMS = [r"$N_{\rm H}$", r"$\Gamma$", r"$N_{\rm pl}$", r"$kT$", r"$N_{\rm bb}$"]
LEVELS = ["faint", "medium", "bright"]
COL = {"faint": "C0", "medium": "C2", "bright": "C3"}


def ksp(npz):
    d = np.load(npz, allow_pickle=True)
    k = [x for x in d.files if "rank" in x.lower()][0]
    r = d[k]; r = r[:, None] if r.ndim == 1 else r; L = r.max()
    return [float(stats.kstest((r[:, j] + 0.5) / (L + 1), "uniform").pvalue)
            for j in range(r.shape[1])]


fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
x = np.arange(5)
for ax, (inst, base) in zip(axes, [("XMM EPIC-pn", "outputs/calibration"),
                                   ("NICER XTI", "outputs/calibration_nicer")]):
    for lvl in LEVELS:
        p = np.array(ksp(REPO / base / lvl / "sbc.npz"))
        p = np.clip(p, 1e-14, 1)
        ax.plot(x, p, "o-", color=COL[lvl], ms=5, label=lvl)
    ax.axhline(0.05, ls=":", color="k", lw=1)
    ax.set_yscale("log"); ax.set_ylim(1e-14, 3)
    ax.set_xticks(x); ax.set_xticklabels(PARAMS, fontsize=8)
    ax.set_xlabel("Parameter")
    ax.set_title(inst, fontsize=9)
axes[0].set_ylabel("SBC Rank-Uniformity KS $p$")
fig.tight_layout(rect=[0, 0, 1, 0.9])
handles = [Line2D([0], [0], color=COL[l], marker="o", ms=5, label=l) for l in LEVELS]
handles.append(Line2D([0], [0], ls=":", color="k", lw=1, label=r"$p = 0.05$ threshold"))
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, 1.0))
out = FIGDIR / "nicer_sbc.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print("saved", out)
for inst, base in [("XMM", "outputs/calibration"), ("NICER", "outputs/calibration_nicer")]:
    for lvl in LEVELS:
        p = ksp(REPO / base / lvl / "sbc.npz")
        print(inst, lvl, [f"{v:.1e}" for v in p], "fails:", sum(v < 0.05 for v in p))
