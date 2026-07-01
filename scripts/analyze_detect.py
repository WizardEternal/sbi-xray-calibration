"""Analyze the Phase-4 detection benchmark: AUC tables + ΔΓ-bias + figures.

Reads the JSONL artifacts written by scripts/run_detect_benchmark.py and
regenerates everything derived (so the benchmark stays a pure data-producer and
the figures are reproducible without rerunning the expensive scoring):

    outputs/detect/results[_pilot].jsonl      per-cell AUC
    outputs/detect/consequence[_pilot].jsonl  B1 Gamma-bias
        -> outputs/detect/auc_heatmap[_pilot].png   AUC grid (family x strength,
                                                     panel per level, line per detector)
        -> outputs/detect/auc_table[_pilot].md      markdown AUC table
        -> outputs/detect/consequence[_pilot].md    B1 ΔΓ-bias table

Usage:
    .venv\\Scripts\\python.exe scripts\\analyze_detect.py --config configs\\detect.yaml
    .venv\\Scripts\\python.exe scripts\\analyze_detect.py --config configs\\detect.yaml --pilot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _out_dir() -> Path:
    return _repo_root() / "outputs" / "detect"


def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def build_auc_table(results, levels, families, detectors):
    """(level, family, strength, detector) -> auc nested dict."""
    table = defaultdict(dict)
    for r in results:
        table[(r["level"], r["family"], float(r["strength"]))][r["detector"]] = r["auc"]
    return table


def write_auc_markdown(table, levels, families, detectors, out_path: Path):
    lines = ["# Detection-benchmark AUC (clean vs misspecified)\n"]
    for level in levels:
        lines.append(f"\n## level: {level}\n")
        header = "| family | strength | " + " | ".join(detectors) + " |"
        sep = "|" + "---|" * (2 + len(detectors))
        lines.append(header)
        lines.append(sep)
        for fam in families:
            grid = sorted({s for (lv, f, s) in table if lv == level and f == fam})
            for s in grid:
                cells = table.get((level, fam, s), {})
                vals = " | ".join(
                    f"{cells[d]:.3f}" if d in cells else "n/a" for d in detectors
                )
                lines.append(f"| {fam} | {s:g} | {vals} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_consequence_markdown(cons, out_path: Path):
    if not cons:
        out_path.write_text("# B1 dGamma-bias\n\n(no consequence rows)\n",
                            encoding="utf-8")
        return out_path
    lines = ["# B1 unmodeled-line silent-failure cost: NPE Gamma bias\n",
             "Posterior-median Gamma minus the clean-truth Gamma that generated "
             "the (line-contaminated) spectrum, per strength/level. The undetected-"
             "but-biased cases quantify the downstream bias cost.\n",
             "| level | line norm | <G_truth> | <G_hat> | <dG> | median dG | <|dG|> |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(cons, key=lambda r: (r["level"], r["strength"])):
        lines.append(
            f"| {r['level']} | {r['strength']:g} | {r['gamma_truth_mean']:.3f} | "
            f"{r['gamma_hat_mean']:.3f} | {r['dGamma_bias_mean']:+.3f} | "
            f"{r['dGamma_bias_median']:+.3f} | {r['abs_bias_mean']:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def make_heatmap(table, levels, families, detectors, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [l for l in levels if any(lv == l for (lv, _, _) in table)]
    if not levels:
        print("[analyze] no cells to plot")
        return None
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4),
                             squeeze=False)
    markers = {"D1": "o", "D2": "s", "D3": "^"}
    for ax, level in zip(axes[0], levels):
        for fam in families:
            grid = sorted({s for (lv, f, s) in table if lv == level and f == fam})
            if not grid:
                continue
            for d in detectors:
                ys = [table.get((level, fam, s), {}).get(d, np.nan) for s in grid]
                ax.plot(range(len(grid)), ys, marker=markers.get(d, "o"),
                        label=f"{fam}-{d}", alpha=0.7, ms=4)
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set_title(f"{level}")
        ax.set_xlabel("strength index (weak→strong)")
        ax.set_ylabel("ROC AUC")
        ax.set_ylim(0.3, 1.02)
    axes[0][-1].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle("Misspecification-detection AUC (clean vs misspecified)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    suffix = "_pilot" if args.pilot else ""
    out = _out_dir()

    results = _read_jsonl(out / f"results{suffix}.jsonl")
    cons = _read_jsonl(out / f"consequence{suffix}.jsonl")
    if not results:
        print(f"[analyze] no results in {out / f'results{suffix}.jsonl'}")
        return 1

    levels = cfg["levels"]
    families = list(cfg["families"].keys())
    detectors = cfg.get("detectors", ["D1", "D2", "D3"])

    table = build_auc_table(results, levels, families, detectors)
    p1 = write_auc_markdown(table, levels, families, detectors,
                            out / f"auc_table{suffix}.md")
    p2 = write_consequence_markdown(cons, out / f"consequence{suffix}.md")
    p3 = make_heatmap(table, levels, families, detectors,
                      out / f"auc_heatmap{suffix}.png")
    print(f"[analyze] wrote {p1.name}, {p2.name}" + (f", {p3.name}" if p3 else ""))

    # print the table to stdout too
    print("\n" + (out / f"auc_table{suffix}.md").read_text(encoding="utf-8"))
    if cons:
        print((out / f"consequence{suffix}.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
