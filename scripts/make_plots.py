"""Rebuild the README figures that derive from finished benchmark artifacts (config-driven).

The figures themselves are committed. This is the single documented entry point
for regenerating them by `python scripts/make_plots.py --config ...` from a fixed
seed. It reads outputs/calibration/*.npz and outputs/detect/*.jsonl (no model
loading, no simulation, no heavy compute), but those source artifacts are
gitignored, so the pipeline outputs must be present locally (i.e. after running
the benchmarks). It then dispatches to the per-figure builders:

    summary_figure       -> make_summary_figure.main()   (outputs/summary_figure.png)
    coverage_panel       -> make_coverage_panel.main()   (the calibration panel)
    support_figs         -> make_support_figs.main()     (AUC grid + dGamma)
    detect_tables        -> analyze_detect.main()        (AUC md/heatmap + consequence)
    nicer_replication    -> make_nicer_fig.main()        (NICER-vs-EPIC-pn detection figure)
    nicer_sbc            -> make_nicer_sbc_fig.main()    (cross-instrument SBC figure)

Each figure is skip-if-exists (per its declared output paths in the config)
unless --force is passed.

Usage:
    .venv\\Scripts\\python.exe scripts\\make_plots.py --config configs\\make_plots.yaml
    .venv\\Scripts\\python.exe scripts\\make_plots.py --config configs\\make_plots.yaml --force
    .venv\\Scripts\\python.exe scripts\\make_plots.py --config configs\\make_plots.yaml --only summary_figure
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml

from sbixcal._shared import _repo_root


# ensure sibling scripts are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _outputs_exist(rel_paths) -> bool:
    root = _repo_root()
    return all((root / p).exists() for p in rel_paths)


def _run_builder(builder: str, detect_config: str):
    """Dispatch one builder. Returns the module's main() exit code (0 = ok)."""
    if builder == "summary_figure":
        mod = importlib.import_module("make_summary_figure")
        return mod.main()
    if builder == "coverage_panel":
        mod = importlib.import_module("make_coverage_panel")
        return mod.main()
    if builder == "support_figs":
        mod = importlib.import_module("make_support_figs")
        return mod.main()
    if builder == "detect_tables":
        mod = importlib.import_module("analyze_detect")
        return mod.main(["--config", detect_config])
    if builder == "nicer_replication":
        mod = importlib.import_module("make_nicer_fig")
        return mod.main()
    if builder == "nicer_sbc":
        mod = importlib.import_module("make_nicer_sbc_fig")
        return mod.main()
    raise ValueError(f"unknown builder: {builder!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true",
                    help="rebuild figures even if their outputs already exist")
    ap.add_argument("--only", default=None,
                    help="build only this builder key (e.g. summary_figure)")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    detect_config = cfg.get("detect_config", "configs/detect.yaml")

    n_built = n_skipped = 0
    for entry in cfg["figures"]:
        builder = entry["builder"]
        outs = entry.get("outputs", [])
        if args.only and builder != args.only:
            continue
        if outs and _outputs_exist(outs) and not args.force:
            print(f"[skip] {builder} (outputs present: {', '.join(outs)})")
            n_skipped += 1
            continue
        print(f"[build] {builder} ...")
        rc = _run_builder(builder, detect_config)
        if rc not in (0, None):
            print(f"[error] builder {builder} returned {rc}")
            return rc
        n_built += 1

    print(f"\n[make_plots] done: {n_built} built, {n_skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
