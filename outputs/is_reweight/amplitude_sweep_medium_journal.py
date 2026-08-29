"""Journal-clean re-plot of the medium-level gain-amplitude sweep.

AUC points, a bootstrap 95% CI band and the AUC=0.5 reference line on a
linear 1-10 per cent x-axis, sized for a single journal column and with
plain axis labels. Nothing is recomputed here: every plotted value is read
from the persisted sweep-results json.

Source: outputs/is_reweight/amplitude_sweep_results.json
  (written by run_amplitude_sweep.py, level="medium", clean_n=150, n_mis=100,
  n_budget=6000 permutations per point; see that file's own header).
  For each of points["1pct"|"3pct"|"5pct"|"10pct"]:
    gain_factor      -> x-axis position (1, 3, 5, 10 per cent)
    auc_ess_eff      -> plotted AUC point
    ci95_lo/ci95_hi  -> plotted bootstrap 95% CI band

Output: amplitude_sweep_medium_journal.png (300 dpi) and .pdf, this
directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "amplitude_sweep_results.json"

with open(RESULTS_JSON) as f:
    results = json.loads(f.read())

assert results["level"] == "medium"

STRENGTHS = [1.0, 3.0, 5.0, 10.0]  # per cent, matches run_amplitude_sweep.py STRENGTHS
points = results["points"]

xs = STRENGTHS
aucs = [points[f"{s:g}pct"]["auc_ess_eff"] for s in xs]
los = [points[f"{s:g}pct"]["ci95_lo"] for s in xs]
his = [points[f"{s:g}pct"]["ci95_hi"] for s in xs]

# Cross-check against the values the coordinator quoted from memory
# (rounded to 3 d.p.): 0.443 / 0.445 / 0.462 / 0.484.
expected_auc_3dp = [0.443, 0.445, 0.462, 0.484]
for auc, want in zip(aucs, expected_auc_3dp):
    assert abs(round(auc, 3) - want) < 1e-9, (auc, want)

# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "DejaVu Sans",
})

FIG_W_IN = 3.4
FIG_H_IN = 2.7

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))

ax.fill_between(xs, los, his, color="#8a8a8a", alpha=0.30, lw=0,
                 label="bootstrap 95% CI")
ax.plot(xs, aucs, marker="o", markersize=4.5, color="#1a1a1a", lw=1.4,
         label="ESS-efficiency AUC")
ax.axhline(0.5, color="#555555", lw=1.0, ls=(0, (4, 2)), label="AUC = 0.5")

ax.set_xlabel("injected gain amplitude (per cent)")
ax.set_ylabel("ESS-efficiency ROC AUC")
ax.set_xticks(xs)
ax.set_xlim(0.3, 10.7)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.legend(loc="upper left", frameon=False, handlelength=1.6,
          borderaxespad=0.0, labelspacing=0.3)

fig.subplots_adjust(left=0.205, right=0.97, top=0.96, bottom=0.19)

out_png = HERE / "amplitude_sweep_medium_journal.png"
out_pdf = HERE / "amplitude_sweep_medium_journal.pdf"
fig.savefig(out_png, dpi=300)
fig.savefig(out_pdf)
print("wrote", out_png)
print("wrote", out_pdf)

print()
print("Plotted values (amplitude_sweep_results.json -> points):")
for s, auc, lo, hi in zip(xs, aucs, los, his):
    print(f"  {s:>4.0f}%  AUC={auc:.4f}  CI95=[{lo:.4f}, {hi:.4f}]")
