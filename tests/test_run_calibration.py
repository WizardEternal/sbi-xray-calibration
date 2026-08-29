"""Unit tests for the ``_cov_at`` mismatch guard.

Run with the repo venv:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_run_calibration.py

``_cov_at`` (in scripts/run_calibration.py and its copy in scripts/run_reseed_pack.py)
picks the nominal level nearest a requested target. If a config's
nominal_levels doesn't actually contain the target (e.g. 0.68 missing) it must
raise instead of silently filing a different level's coverage under the
requested key.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_script_module(name: str):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    repo = Path(__file__).resolve().parents[1]
    scripts_dir = repo / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


@pytest.mark.parametrize("module_name", ["run_calibration", "run_reseed_pack"])
def test_cov_at_raises_on_missing_level(module_name):
    mod = _load_script_module(module_name)
    nominal = np.array([0.05, 0.1, 0.5, 0.7, 0.9])
    cov = np.zeros((len(nominal), 3))
    with pytest.raises(ValueError, match="0.68"):
        mod._cov_at(nominal, cov, 0.68)


@pytest.mark.parametrize("module_name", ["run_calibration", "run_reseed_pack"])
def test_cov_at_accepts_close_match(module_name):
    mod = _load_script_module(module_name)
    nominal = np.array([0.05, 0.1, 0.5, 0.679, 0.9])
    cov = np.tile(np.arange(len(nominal), dtype=float)[:, None], (1, 2))
    result = mod._cov_at(nominal, cov, 0.68)
    used = result[-1]
    assert used == pytest.approx(0.679)
