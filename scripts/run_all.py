"""End-to-end pipeline driver for sbi-xray-calibration.

Chains the whole repo, each stage skip-if-exists so a cold checkout can be
rebuilt from config+seed and a warm checkout is a no-op:

    simulate  -> train  -> calibrate  -> detect  -> [ns_bench]  -> plots

Stages and what "already done" means (each is skipped when its artifact exists,
unless --force or the stage's own re-run is requested):

  1. simulate   data/sim/<name>_<level>.npz                       (5e4-row training sets + sanity sets)
  2. train      outputs/models/train_npe_prod_<level>/flow_state.pt
  3. calibrate  outputs/calibration/<level>/summary.json
  4. detect     outputs/detect/results.jsonl  (144 cells)  + analyze_detect tables
  5. ns_bench   outputs/ns_bench/results.jsonl              -- OPT-IN ONLY (--with-ns),
                because the NS run is multi-hour and is normally launched
                separately in the background. Default run_all does NOT touch it.
  6. plots      money_plot + supporting figures (make_plots.py)

Crash-safety: every stage delegates to a script that is itself append/skip
resumable (simulate npz skip-if-exists, train per-level checkpoint skip, detect
JSONL per-cell skip, calibrate per-level skip). run_all just sequences them and
prints a per-stage status line.

Caps compute (OMP_NUM_THREADS) for the heavy stages.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_all.py
    .venv\\Scripts\\python.exe scripts\\run_all.py --skip-train --skip-detect       # just (re)plot
    .venv\\Scripts\\python.exe scripts\\run_all.py --with-ns                          # include the long NS run
    .venv\\Scripts\\python.exe scripts\\run_all.py --force                            # rebuild everything

Skip flags: --skip-simulate --skip-train --skip-calibrate --skip-detect
            --skip-plots ; --with-ns to include NS ; --force to ignore
            skip-if-exists on every stage.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


PY = None  # resolved in main()
LEVELS = ["faint", "medium", "bright"]


def _py() -> str:
    """The repo venv python (this interpreter)."""
    return sys.executable


def _run(cmd, env=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(cmd, cwd=str(_repo_root()), env=env)
    if res.returncode != 0:
        raise SystemExit(f"[run_all] stage failed (rc={res.returncode}): {cmd}")


def _env_capped():
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


# --------------------------------------------------------------------------
# per-stage existence checks
# --------------------------------------------------------------------------
def _sim_done() -> bool:
    d = _repo_root() / "data" / "sim"
    if not d.exists():
        return False
    need = [f"modelA_prod_train_{lv}.npz" for lv in LEVELS]
    return all((d / n).exists() for n in need)


def _train_done() -> bool:
    root = _repo_root() / "outputs" / "models"
    return all((root / f"train_npe_prod_{lv}" / "flow_state.pt").exists()
               for lv in LEVELS)


def _calib_done() -> bool:
    root = _repo_root() / "outputs" / "calibration"
    return all((root / lv / "summary.json").exists() for lv in LEVELS)


def _detect_done() -> bool:
    p = _repo_root() / "outputs" / "detect" / "results.jsonl"
    if not p.exists():
        return False
    # 144-cell full grid
    with open(p) as f:
        n = sum(1 for line in f if line.strip())
    return n >= 144


def _ns_done() -> bool:
    p = _repo_root() / "outputs" / "ns_bench" / "results.jsonl"
    return p.exists() and p.stat().st_size > 0


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage_simulate(force, env):
    if _sim_done() and not force:
        print("[skip] simulate (training datasets present)")
        return
    py = _py()
    # production training sets (one per level) + production sanity sets
    _run([py, "-m", "sbixcal.simulate", "--config",
          "configs/sim_modelA_prod_train.yaml"] + (["--force"] if force else []),
         env=env)
    _run([py, "-m", "sbixcal.simulate", "--config",
          "configs/sim_modelA_prod.yaml"] + (["--force"] if force else []),
         env=env)


def stage_train(force, env):
    if _train_done() and not force:
        print("[skip] train (all 3 production checkpoints present)")
        return
    py = _py()
    _run([py, "scripts/run_train_npe.py", "--config",
          "configs/train_npe_prod.yaml"] + (["--force"] if force else []),
         env=env)


def stage_calibrate(force, env):
    if _calib_done() and not force:
        print("[skip] calibrate (all 3 levels have summary.json)")
        return
    py = _py()
    _run([py, "scripts/run_calibration.py", "--config",
          "configs/calibration.yaml"] + (["--force"] if force else []),
         env=env)
    # the Phase-3 coverage panel is rebuilt in the plots stage too, harmless here
    _run([py, "scripts/make_coverage_money_panel.py"], env=env)


def stage_detect(force, env):
    py = _py()
    if not (_detect_done() and not force):
        _run([py, "scripts/run_detect_benchmark.py", "--config",
              "configs/detect.yaml"], env=env)
    else:
        print("[skip] detect benchmark (144 cells present)")
    # always (cheaply) regenerate the derived tables/heatmap from the JSONL
    _run([py, "scripts/analyze_detect.py", "--config", "configs/detect.yaml"],
         env=env)


def stage_ns(force, env):
    py = _py()
    if _ns_done() and not force:
        print("[skip] ns_bench (results.jsonl present)")
    else:
        print("[run_all] launching the NESTED-SAMPLING benchmark (multi-hour).")
        _run([py, "scripts/run_ns_benchmark.py", "--config",
              "configs/ns_bench.yaml"], env=env)
    _run([py, "scripts/analyze_ns_bench.py", "--config", "configs/ns_bench.yaml"],
         env=env)


def stage_plots(force, env):
    py = _py()
    cmd = [py, "scripts/make_plots.py", "--config", "configs/make_plots.yaml"]
    if force:
        cmd.append("--force")
    _run(cmd, env=env)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-simulate", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-calibrate", action="store_true")
    ap.add_argument("--skip-detect", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--with-ns", action="store_true",
                    help="ALSO run the multi-hour nested-sampling benchmark "
                         "(off by default; normally launched separately)")
    ap.add_argument("--force", action="store_true",
                    help="ignore skip-if-exists on every stage")
    args = ap.parse_args(argv)

    env = _env_capped()
    print(f"[run_all] OMP_NUM_THREADS={env['OMP_NUM_THREADS']}  "
          f"python={_py()}")

    if not args.skip_simulate:
        stage_simulate(args.force, env)
    if not args.skip_train:
        stage_train(args.force, env)
    if not args.skip_calibrate:
        stage_calibrate(args.force, env)
    if not args.skip_detect:
        stage_detect(args.force, env)
    if args.with_ns:
        stage_ns(args.force, env)
    if not args.skip_plots:
        stage_plots(args.force, env)

    print("\n[run_all] pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
