"""NICER replication figure for the article.
(a) best per-spectrum ROC AUC vs counts for B1 (Fe-K line) and B4 (3% gain shift),
    EPIC-pn vs NICER: B4 flat at chance on both, B1 rises but is weaker on NICER.
(b) effective area vs energy for both responses, with the Fe-K line marked, which
    explains the weaker NICER line detection (its area at 6.4 keV is ~6x its peak).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "outputs" / "diagnostics"
LEVELS = ["faint", "medium", "bright"]
COUNTS = [100, 1000, 10000]


def best_auc(path, fam, strength):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {}
    for lvl in LEVELS:
        # per-spectrum panel: D1/D2 only (D3 is the population statistic, excluded)
        a = [r["auc"] for r in rows if r["family"] == fam
             and abs(r["strength"] - strength) < 1e-9 and r["level"] == lvl
             and r["detector"] in ("D1", "D2")]
        if a:
            out[lvl] = max(a)
    return [out[l] for l in LEVELS]


def load_area(arf):
    a = fits.open(arf)["SPECRESP"].data
    e = 0.5 * (a["ENERG_LO"] + a["ENERG_HI"])
    return np.asarray(e, float), np.asarray(a["SPECRESP"], float)


b1x = best_auc(REPO / "outputs/detect/results.jsonl", "B1", 3e-4)
b1n = best_auc(REPO / "outputs/detect_nicer/results.jsonl", "B1", 3e-4)
b4x = best_auc(REPO / "outputs/detect/results.jsonl", "B4", 3.0)
b4n = best_auc(REPO / "outputs/detect_nicer/results.jsonl", "B4", 3.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))

ax1.plot(COUNTS, b1x, "o-", color="C0", label="B1 Line, EPIC-pn")
ax1.plot(COUNTS, b1n, "s--", color="C0", mfc="white", label="B1 Line, NICER")
ax1.plot(COUNTS, b4x, "o-", color="C1", label="B4 Gain, EPIC-pn")
ax1.plot(COUNTS, b4n, "s--", color="C1", mfc="white", label="B4 Gain, NICER")
ax1.axhline(0.5, ls=":", color="grey", lw=1)
ax1.set_xscale("log"); ax1.set_xticks(COUNTS); ax1.set_xticklabels(["100", "1000", "10000"])
ax1.set_xlabel("Total Counts"); ax1.set_ylabel("Best Per-Spectrum ROC AUC")
ax1.set_ylim(0.33, 1.04)
ax1.set_title("Detection Power", fontsize=9)
# legend below the data (the B4 lines sit at ~0.5; the band below 0.46 is empty)
ax1.legend(fontsize=6.5, loc="lower center", ncol=2, framealpha=0.9,
           columnspacing=1.1, handletextpad=0.4, borderpad=0.3)
ax1.text(0.02, 0.94, "(a)", transform=ax1.transAxes, fontweight="bold")

en, an = load_area(REPO / "data/nicer/nicer.arf")
ax2.plot(en, an, color="C2", lw=1.3, label="NICER XTI")
try:
    from jaxspec.data.util import table_manager
    ex, ax_pn = load_area(table_manager.fetch("example_data/NGC7793_ULX4/PN.arf"))
    ax2.plot(ex, ax_pn, color="C3", lw=1.3, label="XMM EPIC-pn")
except Exception as e:
    print("XMM ARF overlay skipped:", e)
ax2.axvline(6.4, ls=":", color="k", lw=1)
ax2.text(6.1, 300, "Fe-K 6.4 keV", rotation=90, fontsize=6.5, va="center", ha="right")
ax2.set_xscale("log"); ax2.set_xlim(0.3, 10)
ax2.set_xlabel("Energy (keV)"); ax2.set_ylabel("Effective Area (cm$^2$)")
ax2.set_title("Effective Area", fontsize=9)
ax2.legend(fontsize=7, loc="upper right")
ax2.text(0.02, 0.94, "(b)", transform=ax2.transAxes, fontweight="bold")

fig.tight_layout()
out = FIGDIR / "nicer_replication.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print("saved", out)
