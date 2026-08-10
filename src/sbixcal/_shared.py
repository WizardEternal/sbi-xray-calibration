"""Helpers shared across scripts/ and the sbixcal package.

Pulled out of ~14-15 near-identical copies scattered over scripts/ and
src/sbixcal/ (C9a cleanup pass). Every function here reproduces exactly what
its call sites' own local copy did before -- this is a pure extraction, not a
behavior change. Copies whose behavior actually differed between call sites
(different return shape, different JSONDecodeError handling, ...) were left
local rather than forced together; see the comments at each such site.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def _repo_root() -> Path:
    # this file lives at src/sbixcal/_shared.py -> repo root is two parents up.
    # Independent of the caller's own location: every script and package
    # module that imports this gets the same absolute repo root.
    return Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict]:
    """Read a jsonl file, tolerant of a missing file and malformed lines.

    Returns [] if path does not exist. Blank lines and lines that fail to
    parse as JSON are skipped rather than raised. This is the lenient variant
    used by the majority of call sites (analyze_ns_bench.py, analyze_detect.py,
    gonogo_verdict.load_summary); a couple of call sites keep a stricter local
    copy that must not silently swallow a missing file or bad line -- see
    scripts/make_money_plot.py and scripts/make_support_figs.py.
    """
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _mean_abs_dev(nominal, cov_mean) -> float:
    return float(np.mean(np.abs(np.asarray(cov_mean) - np.asarray(nominal))))


def _cov_at_full(nominal, cov, target):
    """Core lookup shared by run_calibration.py's and run_gonogo.py's
    ``_cov_at``: the nominal level nearest ``target``, its per-param coverage
    and the mean over params. Raises if the nearest available level is more
    than 0.005 from ``target``, instead of silently filing that level's
    coverage under the requested target.

    Returns (mean_over_params, per_param_array, nearest_nominal_level). The
    two call sites return different shapes (run_gonogo.py drops per_param;
    run_calibration.py returns it as a list), so each keeps its own thin
    ``_cov_at`` wrapper around this with its original signature -- only the
    validation/lookup logic itself is shared.
    """
    nominal = np.asarray(nominal)
    j = int(np.argmin(np.abs(nominal - target)))
    nearest = float(nominal[j])
    if abs(nearest - target) > 0.005:
        raise ValueError(
            f"requested nominal level {target} has no match within 0.005 in "
            f"nominal_levels (nearest is {nearest}); nominal_levels={nominal.tolist()}")
    per_param = np.asarray(cov)[j]
    return float(np.mean(per_param)), per_param, nearest


def _stable_cell_seed(seed: int, family: str, strength: float) -> int:
    """Deterministic per-(family,strength) misspec-draw seed (process-independent,
    so crash-resume reproduces the same misspecified population)."""
    h = hashlib.sha1(f"{family}|{float(strength):g}".encode()).hexdigest()
    return (int(seed) + int(h[:8], 16)) % (2**31 - 1)


# Okabe-Ito colorblind-safe palette. Named subset actually used by the two
# money-plot scripts (make_money_plot.py / make_coverage_money_panel.py); hex
# values only, unchanged from what each script had hardcoded.
OKABE_ITO = {
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#999999",
    "diag_gray": "#444444",
}
