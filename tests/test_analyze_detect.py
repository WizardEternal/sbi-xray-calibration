"""Unit test for the empty-input guard in scripts/analyze_detect.py.

Run with the repo venv:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_analyze_detect.py

On a fresh clone, outputs/detect/consequence.jsonl is gitignored and absent
while outputs/detect/consequence.md is committed. analyze_detect.py must not
overwrite the committed table with a placeholder generated from empty/missing
input; it must leave it untouched and signal failure instead. This test pins
that contract directly against write_consequence_markdown, with no heavy
imports (numpy/yaml only, same as the module itself).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_analyze_detect_module():
    """Import scripts/analyze_detect.py as a module (scripts/ is not a package)."""
    repo = Path(__file__).resolve().parents[1]
    scripts_dir = repo / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    name = "analyze_detect"
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


def test_empty_consequence_leaves_committed_md_untouched(tmp_path):
    AD = _load_analyze_detect_module()

    md_path = tmp_path / "consequence.md"
    original = "# B1 unmodeled-line silent-failure cost: NPE Gamma bias\n\n(committed content)\n"
    md_path.write_text(original, encoding="utf-8")

    ok = AD.write_consequence_markdown([], md_path)

    assert ok is False
    assert md_path.read_text(encoding="utf-8") == original


def test_nonempty_consequence_writes_table(tmp_path):
    AD = _load_analyze_detect_module()

    md_path = tmp_path / "consequence.md"
    rows = [{
        "level": "medium", "strength": 0.0003,
        "gamma_truth_mean": 2.0, "gamma_hat_mean": 2.1,
        "dGamma_bias_mean": 0.1, "dGamma_bias_median": 0.09,
        "abs_bias_mean": 0.3,
    }]

    ok = AD.write_consequence_markdown(rows, md_path)

    assert ok is True
    text = md_path.read_text(encoding="utf-8")
    assert "medium" in text
    assert "0.1" in text or "+0.100" in text
