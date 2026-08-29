"""Per-spectrum gain-posterior shrink, exact NS against the gain-marginalized flow.

Every number plotted here is read from a persisted artifact under
outputs/gain_marg/; see the inline comments at each read. Nothing is typed in by
hand.

Quantity plotted: shrink = std(g posterior) / std(g prior), for the gain
parameter g with committed prior U[0.95, 1.05]. shrink = 1 means the posterior
returned the prior (no information); shrink < 1 means the posterior narrowed.

  NS (exact) side: ns_std(gain_g) from each spectrum's
    committed_reference.comparison_ns_vs_npe.per_param.gain_g.ns_std block in
    outputs/gain_marg/ns_smallset/npe_draw_recompute/<spectrum_id>.json. That
    recompute asserts the field float32-exact against the original run
    (outputs/gain_marg/ns_smallset/*.json). It is divided by the
    continuous-uniform prior std 0.028867513459481315 = (1.05-0.95)/sqrt(12),
    read from the same file's top-level "prior_std_g" field (identical across
    all 10 files, checked below).

    For i394 and i416 the plotted point is the value from the original
    MLFriends run (0.7901 / 0.7313). The sampler-independent SliceSampler
    re-run (0.7709 / 0.7248, reproducing those to 2.4% and 0.9%) is a
    robustness check, not the plotted number, so this figure plots the same
    npe_draw_recompute/<spectrum_id>.json field for every spectrum.

    For i22 the plotted NS point is the original nominal-box value (0.7231),
    marked as a prior-boundary artifact: a wide-box NS re-run recovers shrink
    1.0216, essentially the prior, in field ns.g_shrink_std_over_prior_std of
    outputs/gain_marg/ns_smallset_v2/i22_widebox.json. That number goes in a
    figure footnote rather than as a second plotted point.

  Flow (gain-marginalized NPE) side: npe_std(gain_g) from the same
    npe_draw_recompute/<spectrum_id>.json files, field metrics_clipped.npe_g_std.
    That is the flow's g-marginal std under the detect.py draw convention
    (outside-prior rejection enabled, then clip), which is the convention used
    at inference time, rather than the raw reject_outside_prior=False draw.

    That std is divided here by the flow's true generative prior std for g,
    which is not the continuous 0.0288675: gen_and_train_gainmarg.py (and the
    _bright variant) draws g from a 200-point grid, centers =
    np.linspace(0.95, 1.05, 200) (GAIN_LO / GAIN_HI / N_GAIN_BINS,
    outputs/gain_marg/gen_and_train_gainmarg.py:51-92 and
    gen_and_train_gainmarg_bright.py:52-92, identical constants in both). A
    200-atom discrete-uniform grid has std sqrt(d**2*(n**2-1)/12) with
    d=(0.05-(-0.05))/199, which the script below evaluates directly as
    np.std(centers) rather than hardcoding a number (~0.0290122, 0.50% above
    the continuous value). Using the continuous denominator on the flow side
    would be a definition mismatch of exactly the kind that silently biases a
    ratio, so every flow-side shrink here divides by 1.005 relative to the
    continuous convention.

Counts (total_counts) are read from the same npe_draw_recompute files'
committed_reference.total_counts field.

Output: gain_shrink_capture.png (300 dpi) and .pdf, this directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

HERE = Path(__file__).resolve().parent
GAIN_MARG = HERE.parent  # outputs/gain_marg
RECOMPUTE_DIR = GAIN_MARG / "ns_smallset" / "npe_draw_recompute"
V2_DIR = GAIN_MARG / "ns_smallset_v2"

# ----------------------------------------------------------------------
# 1. The flow's TRUE g-prior std (200-atom grid), derived from the exact
#    grid definition in the training scripts, not hardcoded.
# ----------------------------------------------------------------------
GAIN_LO, GAIN_HI, N_GAIN_BINS = 0.95, 1.05, 200  # gen_and_train_gainmarg{,_bright}.py:51-53
flow_g_centers = np.linspace(GAIN_LO, GAIN_HI, N_GAIN_BINS)  # ...py:91
flow_prior_std_g = float(np.std(flow_g_centers))  # population std of the 200 grid centers
assert abs(flow_prior_std_g - 0.0290122124) < 1e-6, flow_prior_std_g

# ----------------------------------------------------------------------
# 2. Per-spectrum npe_draw_recompute JSONs: NS (original committed box) and
#    clipped-draw flow gain_g stds, plus counts.
# ----------------------------------------------------------------------
SPECTRUM_IDS = [
    "medium_s0_i22", "medium_s0_i87", "medium_s0_i91",
    "medium_s0_i95", "medium_s0_i197", "medium_s0_i482",
    "bright_s0_i8", "bright_s0_i238", "bright_s0_i394", "bright_s0_i416",
]

raw = {}
continuous_prior_std_g = None
for sid in SPECTRUM_IDS:
    d = json.loads((RECOMPUTE_DIR / f"{sid}.json").read_text())
    if continuous_prior_std_g is None:
        continuous_prior_std_g = d["prior_std_g"]
    else:
        assert d["prior_std_g"] == continuous_prior_std_g
    cr = d["committed_reference"]
    ns_std = cr["comparison_ns_vs_npe"]["per_param"]["gain_g"]["ns_std"]
    counts = cr["total_counts"]
    npe_std_clipped = d["metrics_clipped"]["npe_g_std"]
    raw[sid] = dict(ns_std=ns_std, counts=counts, npe_std_clipped=npe_std_clipped)

assert abs(continuous_prior_std_g - 0.028867513459481315) < 1e-12

# ----------------------------------------------------------------------
# 3. v2 wide-box NS re-run for the i22 artifact check (footnote only). i394
#    and i416 do not use their v2 slice re-runs here: the plotted value is
#    the one from the original MLFriends run (see docstring), so
#    this figure must match: i394/i416 NS shrink comes from the same
#    npe_draw_recompute committed_reference block as every other spectrum
#    (Sec.4 below). The v2 slice numbers (0.7709/0.7248) are cited in the
#    robustness cross-check rather than the plotted value, and is not read here.
# ----------------------------------------------------------------------
i22_widebox = json.loads((V2_DIR / "i22_widebox.json").read_text())
i22_widebox_shrink = i22_widebox["ns"]["g_shrink_std_over_prior_std"]

# ----------------------------------------------------------------------
# 4. Assemble the per-spectrum table: NS shrink (exact) + flow shrink
#    (corrected for the 200-atom grid prior).
# ----------------------------------------------------------------------
def flow_shrink(sid: str) -> float:
    return raw[sid]["npe_std_clipped"] / flow_prior_std_g

medium_ids = ["medium_s0_i22", "medium_s0_i197", "medium_s0_i482",
              "medium_s0_i87", "medium_s0_i91", "medium_s0_i95"]
bright_ids = ["bright_s0_i416", "bright_s0_i394", "bright_s0_i8", "bright_s0_i238"]

data = {}
for sid in medium_ids + bright_ids:
    # Value from the original run for every spectrum, i394 and i416 included:
    # npe_draw_recompute/<sid>.json -> committed_reference.comparison_ns_vs_npe
    # .per_param.gain_g.ns_std / prior_std_g (see docstring: the paper text
    # quotes these, not the v2 slice re-run).
    ns_s = raw[sid]["ns_std"] / continuous_prior_std_g
    data[sid] = dict(
        counts=raw[sid]["counts"],
        ns_shrink=ns_s,
        flow_shrink=flow_shrink(sid),
    )

# ----------------------------------------------------------------------
# 5. Cross-check against the citable markdown numbers (not the source --
#    just a redundant sanity check that this script reproduces them).
# ----------------------------------------------------------------------
expected_ns_shrink = {  # values from the original MLFriends runs
    "medium_s0_i22": 0.7231, "medium_s0_i87": 0.9783, "medium_s0_i91": 0.9897,
    "medium_s0_i95": 1.0103, "medium_s0_i197": 0.9470, "medium_s0_i482": 0.9778,
    "bright_s0_i8": 0.8557, "bright_s0_i238": 1.0050,
    "bright_s0_i394": 0.7901, "bright_s0_i416": 0.7313,  # committed, Sec.2 table
    # (v2 slice re-run reproduces these to 2.4%/0.9%: 0.7709/0.7248, Sec.5 --
    # the robustness check, not plotted)
}
expected_flow_shrink = {  # flow-side shrinks under the 200-atom grid prior std
    "medium_s0_i22": 0.8875, "medium_s0_i87": 0.9878, "medium_s0_i91": 0.9779,
    "medium_s0_i95": 0.9664, "medium_s0_i197": 0.9595, "medium_s0_i482": 0.9739,
    "bright_s0_i8": 0.9819, "bright_s0_i238": 0.9652, "bright_s0_i394": 0.9875,
    "bright_s0_i416": 0.9494,
}
for sid, want in expected_ns_shrink.items():
    got = data[sid]["ns_shrink"]
    assert abs(got - want) < 5e-4, (sid, got, want)
for sid, want in expected_flow_shrink.items():
    got = data[sid]["flow_shrink"]
    assert abs(got - want) < 2e-3, (sid, got, want)
assert abs(i22_widebox_shrink - 1.0216) < 1e-3

# ----------------------------------------------------------------------
# 6. Plot.
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
FIG_H_IN = 3.8

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))

# y-positions, top to bottom: header, 6 medium rows, gap, header, 4 bright rows
y_medium_header = 12
y_medium = {"medium_s0_i22": 11, "medium_s0_i197": 10, "medium_s0_i482": 9,
            "medium_s0_i87": 8, "medium_s0_i91": 7, "medium_s0_i95": 6}
y_bright_header = 4
y_bright = {"bright_s0_i416": 3, "bright_s0_i394": 2, "bright_s0_i8": 1,
            "bright_s0_i238": 0}
y_of = {**y_medium, **y_bright}

NS_COLOR = "#1a1a1a"
FLOW_COLOR = "#8a8a8a"
CONNECT_COLOR = "#bbbbbb"

# connecting segments (dumbbell) first, so markers draw on top
for sid, y in y_of.items():
    ax.plot([data[sid]["ns_shrink"], data[sid]["flow_shrink"]], [y, y],
             color=CONNECT_COLOR, lw=1.2, zorder=1, solid_capstyle="round")

# flow markers (open squares, grey)
for sid, y in y_of.items():
    ax.plot(data[sid]["flow_shrink"], y, marker="s", markersize=5.5,
             markerfacecolor="none", markeredgecolor=FLOW_COLOR,
             markeredgewidth=1.3, zorder=3, linestyle="none")

# NS markers (filled circles, black); i22 gets an extra hollow-x overlay
for sid, y in y_of.items():
    ax.plot(data[sid]["ns_shrink"], y, marker="o", markersize=5.0,
             markerfacecolor=NS_COLOR, markeredgecolor=NS_COLOR,
             zorder=4, linestyle="none")

sid = "medium_s0_i22"
ax.plot(data[sid]["ns_shrink"], y_of[sid], marker="x", markersize=6.5,
         markeredgecolor="white", markeredgewidth=1.3, zorder=5, linestyle="none")

# prior-returned reference line
ax.axvline(1.0, color="#555555", lw=1.0, ls=(0, (4, 2)), zorder=2)

# group headers (bold, left-aligned in axes x-fraction so xlim doesn't matter)
trans = ax.get_yaxis_transform()
ax.text(0.0, y_medium_header, "Medium spectra", transform=trans,
         fontsize=8, fontweight="bold", va="center", ha="left")
ax.text(0.0, y_bright_header, "Bright spectra", transform=trans,
         fontsize=8, fontweight="bold", va="center", ha="left")

# y tick labels: spectrum id + counts, only on data rows
tick_positions = list(y_of.values())
tick_labels = []
id_to_label = {
    "medium_s0_i22": "i22†", "medium_s0_i197": "i197", "medium_s0_i482": "i482",
    "medium_s0_i87": "i87", "medium_s0_i91": "i91", "medium_s0_i95": "i95",
    "bright_s0_i416": "i416", "bright_s0_i394": "i394", "bright_s0_i8": "i8",
    "bright_s0_i238": "i238",
}
order = ["medium_s0_i22", "medium_s0_i197", "medium_s0_i482",
         "medium_s0_i87", "medium_s0_i91", "medium_s0_i95",
         "bright_s0_i416", "bright_s0_i394", "bright_s0_i8", "bright_s0_i238"]
for sid in order:
    tick_labels.append(f"{id_to_label[sid]}  (n={data[sid]['counts']:,})")

ax.set_yticks([y_of[sid] for sid in order])
ax.set_yticklabels(tick_labels)
ax.set_ylim(-0.9, 12.9)
# Do not invert_yaxis(): matplotlib's default (y increases upward)
# already puts the larger y-values (medium header=12) at the top and the
# bright block (header=4, rows 3..0) at the bottom, which is the intended
# "6 medium above, 4 bright below" layout.

ax.set_xlim(0.66, 1.06)
ax.set_xlabel("gain posterior std / prior std\n(1.0 = prior returned)")

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=3)

# faint horizontal row bands for readability (very light, alternating)
for sid, y in y_of.items():
    if y % 2 == 0:
        ax.axhspan(y - 0.5, y + 0.5, color="#f2f2f2", zorder=0)

# legend (compact, above the axes)
ns_handle = mlines.Line2D([], [], color=NS_COLOR, marker="o", linestyle="none",
                            markersize=5.0, label="NS (exact)")
flow_handle = mlines.Line2D([], [], color=FLOW_COLOR, marker="s", linestyle="none",
                              markersize=5.5, markerfacecolor="none",
                              markeredgewidth=1.3, label="Flow (NPE)")
ax.legend(handles=[ns_handle, flow_handle], loc="upper center",
          bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False,
          handletextpad=0.35, columnspacing=1.0, borderaxespad=0.0)

# footnote for the i22 artifact (kept short so it fits the 3.4in column
# at the required >=8pt font without being clipped at the canvas edge)
fig.text(0.5, 0.008,
          f"† prior-boundary artifact; wide-box NS shrink = {i22_widebox_shrink:.2f}",
          ha="center", va="bottom", fontsize=8)

fig.subplots_adjust(left=0.34, right=0.97, top=0.87, bottom=0.20)

out_png = HERE / "gain_shrink_capture.png"
out_pdf = HERE / "gain_shrink_capture.pdf"
fig.savefig(out_png, dpi=300)
fig.savefig(out_pdf)
print("wrote", out_png)
print("wrote", out_pdf)

print()
print("Per-spectrum values used (NS shrink, flow shrink):")
for sid in order:
    print(f"  {sid:16s} counts={data[sid]['counts']:>6d}  "
          f"ns={data[sid]['ns_shrink']:.4f}  flow={data[sid]['flow_shrink']:.4f}")
print(f"i22 wide-box NS shrink (footnote only): {i22_widebox_shrink:.4f}")
print(f"flow_prior_std_g (200-atom grid, derived): {flow_prior_std_g:.7f}")
print(f"continuous_prior_std_g (NS side): {continuous_prior_std_g:.7f}")
