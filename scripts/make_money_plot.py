"""The README summary figure: outputs/money_plot.png.

Two panels, side by side, colorblind-safe (Okabe-Ito), dpi>=200:

  panel (a)  calibration: empirical coverage vs nominal credibility, raw NPE
             vs the better recalibration, across the three count regimes
             (faint ~100 / medium ~1000 / bright ~10000 counts). Below the
             diagonal = over-confident (under-covers). This is the
             coverage_money_panel collapsed onto one axes (one raw + one recal
             curve per level) so the count-regime trend is read at a glance:
             faint/medium sit on the diagonal, bright sags below it and the
             recalibration pulls it back.

  panel (b)  detection: ROC curves at the medium (~1000-count) level, the best
             detector per misspecification family (B1 line, B2 partial-cover,
             B3 brems-continuum, B4 gain-shift), each at its strongest grid
             point, plus B4 which stays on the chance diagonal at every strength
             (the gain-shift negative result). Rebuilt from the per-spectrum
             scores in outputs/detect/scores.jsonl (the benchmark stays a pure
             data producer).

Data sources (read-only):
  outputs/calibration/{faint,medium,bright}/coverage_before_after.npz + summary.json + is_coverage.npz
  outputs/detect/scores.jsonl  (+ results.jsonl for the AUC annotations)
  configs/detect.yaml          (family strength grids / labels)

Writes only:
  outputs/money_plot.png

The figure itself is committed. Regenerating it does no model loading and no
heavy compute, but it reads the calibration npz + detect scores.jsonl produced
by the benchmarks (those source artifacts are gitignored), so the pipeline
outputs must be present locally:

    .venv\\Scripts\\python.exe scripts\\make_money_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from sbixcal import detect as D
from sbixcal._shared import _repo_root, _mean_abs_dev, OKABE_ITO


LEVELS = ["faint", "medium", "bright"]

C_DIAG = OKABE_ITO["diag_gray"]

# per-level styling for panel (a): distinct color + marker + linestyle so the
# three regimes are separable without color.
LEVEL_STYLE = {
    "faint":  dict(color=OKABE_ITO["sky_blue"], marker="o", ls="-"),
    "medium": dict(color=OKABE_ITO["bluish_green"], marker="s", ls="-"),
    "bright": dict(color=OKABE_ITO["vermilion"], marker="^", ls="-"),
}

# per-family styling for panel (b)
FAMILY_STYLE = {
    "B1": dict(color=OKABE_ITO["blue"], ls="-",  label="B1 Fe-K line"),
    "B2": dict(color=OKABE_ITO["bluish_green"], ls="-",  label="B2 partial-covering"),
    "B3": dict(color=OKABE_ITO["reddish_purple"], ls="-",  label="B3 wrong continuum (brems)"),
    "B4": dict(color=OKABE_ITO["gray"], ls="--", label="B4 gain shift"),
}


# panel (a): calibration coverage curves
def _calib_dir(level: str) -> Path:
    return _repo_root() / "outputs" / "calibration" / level


# load_calibration/choose_recal below look like scripts/make_coverage_money_panel.py's
# load_level/choose_recal, but they are not byte-identical (this one sorts by
# nominal and returns a 3-tuple from choose_recal vs a 4-tuple there), so each
# file keeps its own local copy rather than being forced together.
def load_calibration(level: str):
    cdir = _calib_dir(level)
    cov = np.load(cdir / "coverage_before_after.npz", allow_pickle=True)
    isc = np.load(cdir / "is_coverage.npz", allow_pickle=True)
    with open(cdir / "summary.json") as f:
        summ = json.load(f)
    nominal = np.asarray(cov["nominal_levels"], dtype=float)
    raw = np.asarray(cov["cov_raw"]).mean(axis=1)
    conformal = np.asarray(cov["cov_recal"]).mean(axis=1)
    is_all = np.asarray(isc["cov_is_all"]).mean(axis=1)
    order = np.argsort(nominal)
    d = {
        "level": level,
        "nominal": nominal[order],
        "raw": raw[order],
        "conformal": conformal[order],
        "is_all": is_all[order],
        "counts": float(summ["median_total_counts"]),
        "raw_dev": _mean_abs_dev(nominal, raw),
        "conformal_dev": _mean_abs_dev(nominal, conformal),
        "is_all_dev": _mean_abs_dev(nominal, is_all),
    }
    return d


def choose_recal(d: dict):
    """Better recalibration = smaller mean-abs-deviation (conformal vs IS-all)."""
    if d["conformal_dev"] <= d["is_all_dev"]:
        return d["conformal"], "conformal", d["conformal_dev"]
    return d["is_all"], "IS-refined", d["is_all_dev"]


# GO/NO-GO robustness variants: the bright-level reseeds + the uncapped retrain.
# Their raw coverage curves form the "near-the-diagonal" envelope that shows the
# production-flow over-confidence is a single-flow training artifact, not the
# count regime. Loaded read-only from outputs/gonogo/<variant>/.
GONOGO_VARIANTS = ["gonogo_seed101", "gonogo_seed202", "gonogo_seed303",
                   "gonogo_uncapped"]


def load_gonogo_raw_band():
    """Return (nominal, lo, hi, devs) for the reseed/uncapped raw-coverage band at
    the bright level. ``lo``/``hi`` are the per-nominal min/max raw coverage across
    the variants (mean over params); ``devs`` is the list of mean-abs-deviations.
    Returns None if the artifacts are absent (band is then simply skipped)."""
    root = _repo_root()
    curves, devs = [], []
    nominal_ref = None
    for v in GONOGO_VARIANTS:
        npz = root / "outputs" / "gonogo" / v / "coverage_before_after.npz"
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=True)
        nominal = np.asarray(z["nominal_levels"], dtype=float)
        cov_pp = np.asarray(z["cov_raw"])                 # [n_levels, n_params]
        raw = cov_pp.mean(axis=1)                         # mean over params, for the band curve
        order = np.argsort(nominal)
        nominal, raw, cov_pp = nominal[order], raw[order], cov_pp[order]
        if nominal_ref is None:
            nominal_ref = nominal
        curves.append(raw)
        # report the deviation with the paper/gonogo per-param metric (mean over params
        # and levels of |per-param coverage - nominal|), not the mean-curve deviation.
        devs.append(float(np.mean(np.abs(cov_pp - nominal[:, None]))))
    if not curves:
        return None
    arr = np.vstack(curves)
    return nominal_ref, arr.min(axis=0), arr.max(axis=0), devs


def plot_calibration(ax):
    data = [load_calibration(lv) for lv in LEVELS]
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=C_DIAG, lw=1.3, zorder=1,
            label="perfect calibration")

    # reseed/uncapped raw-coverage band (bright level): the "it's a single-flow
    # artifact" evidence. Drawn under the curves so it reads as context.
    band = load_gonogo_raw_band()
    if band is not None:
        nb, lo, hi, devs = band
        ax.fill_between(nb, lo, hi, color="#999999", alpha=0.28, lw=0, zorder=2,
                        label=(f"bright reseeds + uncapped (raw),\n"
                               f"dev {min(devs):.3f}–{max(devs):.3f}"))

    for d in data:
        st = LEVEL_STYLE[d["level"]]
        lbl_lvl = f"{d['level']} (~{d['counts']:.0f} ct)"
        # raw NPE: open marker, solid line
        ax.plot(d["nominal"], d["raw"], marker=st["marker"], ms=6,
                mfc="white", mew=1.6, color=st["color"], ls="-", lw=1.9,
                zorder=3, label=f"{lbl_lvl} raw")
        # recalibrated (only draw if it differs meaningfully from raw, i.e. bright)
        recal_curve, recal_name, recal_dev = choose_recal(d)
        if d["raw_dev"] >= 0.05:  # only the miscalibrated bright flow needs repair
            ax.plot(d["nominal"], recal_curve, marker=st["marker"], ms=5.5,
                    mfc=st["color"], mew=1.2, color=st["color"], ls=":", lw=2.2,
                    zorder=4, label=f"{lbl_lvl} {recal_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, lw=0.7)
    ax.set_xlabel("nominal credibility level")
    ax.set_ylabel("empirical coverage (mean over 5 params)")
    ax.set_title("(a) coverage before vs after recalibration",
                 fontsize=11.5)
    ax.legend(loc="upper left", fontsize=6.8, framealpha=0.95)
    return data


# panel (b): detection ROC curves at the medium level
# kept local: unlike sbixcal._shared._read_jsonl this does not tolerate a
# missing file or skip malformed lines (crashes instead) -- unifying it would
# change behavior at this call site.
def _read_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Detector classes: D1/D2 are per-spectrum unlabeled novelty scores
# (deployable trust scores); D3 is a population separability statistic (supervised
# two-sample), not a per-spectrum trust score, so it must not be plotted as a peer
# per-spectrum ROC curve. We segregate it into its own (dotted, greyed) overlay.
PER_SPECTRUM = ("D1", "D2")


def strongest_cell_per_family(results, level, cfg, detectors=None):
    """For each family, the strongest grid point and the detector that wins it.

    ``detectors`` (optional) restricts the winner search to a subset (e.g. the
    per-spectrum detectors D1/D2); if None, all detectors are eligible.
    """
    fams = cfg["families"]
    out = {}
    for fam, fam_cfg in fams.items():
        grid = fam_cfg["strength_grid"]
        strongest = grid[-1]  # configs order each grid weak->strong (last = strongest)
        cells = [r for r in results if r["level"] == level and r["family"] == fam
                 and abs(float(r["strength"]) - float(strongest)) < 1e-12
                 and (detectors is None or r["detector"] in detectors)]
        if not cells:
            continue
        win = max(cells, key=lambda r: r["auc"])
        out[fam] = win
    return out


def roc_for_cell(scores, level, family, strength, detector):
    rows = [r for r in scores
            if r["level"] == level and r["family"] == family
            and abs(float(r["strength"]) - float(strength)) < 1e-12
            and r["detector"] == detector]
    y = np.array([1 if r["kind"] == "misspec" else 0 for r in rows])
    s = np.array([r["score"] for r in rows], dtype=float)
    fpr, tpr, _ = roc_curve(y, s)  # sklearn stays: curve shape only
    clean_s = np.array([r["score"] for r in rows if r["kind"] != "misspec"], dtype=float)
    mis_s = np.array([r["score"] for r in rows if r["kind"] == "misspec"], dtype=float)
    _, _, auc = D.roc_auc(clean_s, mis_s)  # the tested AUC implementation, not sklearn
    return fpr, tpr, auc


def plot_detection(ax, level="medium"):
    root = _repo_root()
    with open(root / "configs" / "detect.yaml") as f:
        cfg = yaml.safe_load(f)
    results = _read_jsonl(root / "outputs" / "detect" / "results.jsonl")
    scores = _read_jsonl(root / "outputs" / "detect" / "scores.jsonl")

    counts = None
    try:
        with open(root / "outputs" / "calibration" / level / "summary.json") as f:
            counts = float(json.load(f)["median_total_counts"])
    except Exception:
        pass

    # Per-spectrum detectors only (D1/D2) win the solid ROC curves -- those are the
    # deployable, single-unlabeled-spectrum trust scores. D3 (population
    # separability) is overlaid separately, greyed + dotted + flagged, so it is not
    # read as a peer per-spectrum detector.
    cells = strongest_cell_per_family(results, level, cfg, detectors=PER_SPECTRUM)
    d3_cells = strongest_cell_per_family(results, level, cfg, detectors=("D3",))

    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=C_DIAG, lw=1.2, zorder=1,
            label="chance (AUC 0.5)")

    order = ["B1", "B2", "B3", "B4"]
    # (i) per-spectrum detector ROC curves (solid)
    for fam in order:
        if fam not in cells:
            continue
        c = cells[fam]
        fpr, tpr, auc = roc_for_cell(scores, level, fam, c["strength"],
                                     c["detector"])
        st = FAMILY_STYLE[fam]
        ax.plot(fpr, tpr, color=st["color"], ls=st["ls"], lw=2.1, zorder=3,
                label=f"{st['label']} [{c['detector']}, AUC {auc:.2f}]")

    # (ii) D3 population-separability overlay (dotted) -- segregated by keeping the
    # dotted linestyle (vs the solid per-spectrum curves above), but colored by
    # family (same OKABE_ITO family color as its solid curve) so the three D3
    # curves are distinguishable instead of stacking as identical grey dashes.
    # Show only where it is well above its ~0.5-AUC control floor (B2/B3) so the
    # panel stays readable. One legend entry per family that is actually drawn
    # (checked against the rendered figure: 8 entries total fit the panel fine
    # at the existing fontsize, so no combined-entry fallback was needed).
    for fam in order:
        if fam not in d3_cells:
            continue
        c = d3_cells[fam]
        fpr, tpr, auc = roc_for_cell(scores, level, fam, c["strength"], "D3")
        if auc < 0.6:   # at/near the D3 control floor -> not informative to draw
            continue
        fam_color = FAMILY_STYLE[fam]["color"]
        lbl = f"{fam} D3 (population, not per-spectrum)"
        ax.plot(fpr, tpr, color=fam_color, ls=":", lw=1.8, zorder=2,
                alpha=0.9, label=lbl)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, lw=0.7)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    cnt = f" (~{counts:.0f} counts)" if counts else ""
    ax.set_title(f"(b) detection ROC at {level}{cnt}, best per-spectrum detector",
                 fontsize=11.5)
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.95)
    return cells


def main():
    # No suptitle: the interpretive caption lives in the README / paper caption,
    # not inside the image (a baked-in text block distorts the tight bounding box
    # and is unreadable at thumbnail size anyway).
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    cal_data = plot_calibration(axes[0])
    det_cells = plot_detection(axes[1], level="medium")
    fig.tight_layout()

    out = _repo_root() / "outputs" / "money_plot.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}")

    # echo the numbers that go into the caption
    print("\n-- panel (a) calibration deviations (mean |emp-nominal|) --")
    for d in cal_data:
        _, name, dev = choose_recal(d)
        print(f"  {d['level']:6} (~{d['counts']:.0f} ct): raw {d['raw_dev']:.3f} "
              f"-> {name} {dev:.3f}")
    print("\n-- panel (b) detection cells (medium, strongest strength) --")
    for fam in ["B1", "B2", "B3", "B4"]:
        if fam in det_cells:
            c = det_cells[fam]
            print(f"  {fam}: detector {c['detector']}  strength {c['strength']:g}"
                  f"  AUC {c['auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
