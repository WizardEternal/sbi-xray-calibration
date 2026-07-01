"""Unit tests for the GO/NO-GO verdict logic (scripts/gonogo_verdict.py).

Run with the repo venv:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_gonogo.py

The verdict classifier is the decision the whole robustness pack exists to make:
given the three full-reseed bright variants' raw coverage deviations (and the
uncapped flow's), decide whether the primary NPE over-confidence is ROBUST,
FRAGILE, or MIXED, and interpret the epoch-cap mechanism test. These tests pin the
classification rule against synthetic summary lines so a refactor can't silently
flip a verdict.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_verdict_module():
    """Import scripts/gonogo_verdict.py as a module (scripts/ is not a package)."""
    repo = Path(__file__).resolve().parents[1]
    scripts_dir = repo / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    name = "gonogo_verdict"
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


def _cal_row(variant, dev, **extra):
    """A synthetic kind=='calibration' summary row with a given raw deviation."""
    row = {
        "variant": variant,
        "kind": "calibration",
        "median_total_counts": 9900.0,
        "epochs_trained": 151,
        "max_num_epochs": 150,
        "cov50": 0.36, "cov68": 0.51, "cov90": 0.76,
        "cov_dev_raw": dev,
        "cov_dev_conformal": 0.031,
        "sbc_ks_p_min": 1e-20,
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------
# reseed verdict: ROBUST / FRAGILE / MIXED
# --------------------------------------------------------------------------

def test_all_reseeds_overconfident_is_robust():
    V = _load_verdict_module()
    rows = [
        _cal_row("gonogo_seed101", 0.113),
        _cal_row("gonogo_seed202", 0.098),
        _cal_row("gonogo_seed303", 0.121),
    ]
    res = V.classify(rows)
    assert res["verdict"] == "ROBUST"
    assert "ROBUST" in res["recommendation"]


def test_one_near_calibrated_reseed_is_fragile():
    V = _load_verdict_module()
    rows = [
        _cal_row("gonogo_seed101", 0.113),
        _cal_row("gonogo_seed202", 0.020),   # near-calibrated -> FRAGILE
        _cal_row("gonogo_seed303", 0.110),
    ]
    res = V.classify(rows)
    assert res["verdict"] == "FRAGILE"
    assert "SOFTEN" in res["recommendation"].upper() or "FRAGILE" in res["recommendation"]


def test_intermediate_reseeds_are_mixed():
    V = _load_verdict_module()
    # none below the fragile floor (0.03), but not all above the robust floor (0.06)
    rows = [
        _cal_row("gonogo_seed101", 0.113),
        _cal_row("gonogo_seed202", 0.045),   # between 0.03 and 0.06
        _cal_row("gonogo_seed303", 0.090),
    ]
    res = V.classify(rows)
    assert res["verdict"] == "MIXED"


def test_fragile_takes_precedence_even_if_others_robust():
    V = _load_verdict_module()
    # a fragile reseed wins over two strongly-overconfident ones
    rows = [
        _cal_row("gonogo_seed101", 0.200),
        _cal_row("gonogo_seed202", 0.200),
        _cal_row("gonogo_seed303", 0.010),   # fragile
    ]
    assert V.classify(rows)["verdict"] == "FRAGILE"


def test_incomplete_when_reseeds_missing():
    V = _load_verdict_module()
    rows = [_cal_row("gonogo_seed101", 0.113)]   # only one reseed present
    res = V.classify(rows)
    assert res["verdict"] == "INCOMPLETE"


def test_boundary_at_robust_threshold_is_not_robust():
    V = _load_verdict_module()
    # exactly at 0.06 is NOT > 0.06, so the all-robust condition fails -> MIXED
    rows = [
        _cal_row("gonogo_seed101", 0.060),
        _cal_row("gonogo_seed202", 0.113),
        _cal_row("gonogo_seed303", 0.090),
    ]
    assert V.classify(rows)["verdict"] == "MIXED"


# --------------------------------------------------------------------------
# uncapped (epoch-cap mechanism) interpretation
# --------------------------------------------------------------------------

def test_uncapped_near_calibrated_means_undertraining():
    V = _load_verdict_module()
    rows = [
        _cal_row("gonogo_seed101", 0.113),
        _cal_row("gonogo_seed202", 0.098),
        _cal_row("gonogo_seed303", 0.121),
        _cal_row("gonogo_uncapped", 0.018, epochs_trained=380, max_num_epochs=400),
    ]
    res = V.classify(rows)
    assert res["uncapped_dev"] == 0.018
    assert "undertraining" in res["uncapped_interpretation"].lower()
    assert "undertraining" in res["recommendation"].lower()


def test_uncapped_persists_means_not_just_the_cap():
    V = _load_verdict_module()
    rows = [
        _cal_row("gonogo_seed101", 0.113),
        _cal_row("gonogo_seed202", 0.098),
        _cal_row("gonogo_seed303", 0.121),
        _cal_row("gonogo_uncapped", 0.105, epochs_trained=260, max_num_epochs=400),
    ]
    res = V.classify(rows)
    assert "persists" in res["uncapped_interpretation"].lower()
    assert "not just the" in res["uncapped_interpretation"].lower()


def test_uncapped_intermediate_is_partial():
    V = _load_verdict_module()
    rows = [
        _cal_row("gonogo_seed101", 0.113),
        _cal_row("gonogo_seed202", 0.098),
        _cal_row("gonogo_seed303", 0.121),
        _cal_row("gonogo_uncapped", 0.045),
    ]
    res = V.classify(rows)
    assert "partial" in res["uncapped_interpretation"].lower()


# --------------------------------------------------------------------------
# robustness of the loader against the real summary.jsonl shape
# --------------------------------------------------------------------------

def test_loader_skips_blank_and_garbled_lines(tmp_path):
    V = _load_verdict_module()
    p = tmp_path / "summary.jsonl"
    p.write_text(
        '{"variant":"gonogo_seed101","kind":"calibration","cov_dev_raw":0.11}\n'
        "\n"
        "not json at all\n"
        '{"variant":"gonogo_seed202","kind":"calibration","cov_dev_raw":0.10}\n'
    )
    rows = V.load_summary(p)
    assert len(rows) == 2
    assert {r["variant"] for r in rows} == {"gonogo_seed101", "gonogo_seed202"}


def test_detect_spot_rows_ignored_by_classifier():
    V = _load_verdict_module()
    # spot-check rows (kind=='detect_spot') must not affect the reseed verdict
    rows = [
        _cal_row("gonogo_seed101", 0.113),
        _cal_row("gonogo_seed202", 0.098),
        _cal_row("gonogo_seed303", 0.121),
        {"variant": "gonogo_seed101", "kind": "detect_spot", "family": "B1",
         "strength": 3e-4, "detector": "D1", "auc": 0.97, "expected_auc": 0.97},
        {"variant": "gonogo_seed101", "kind": "detect_spot", "family": "B4",
         "strength": 3.0, "detector": "D1", "auc": 0.50, "expected_auc": 0.50},
    ]
    res = V.classify(rows)
    assert res["verdict"] == "ROBUST"
    assert set(res["reseed_devs"]) == {
        "gonogo_seed101", "gonogo_seed202", "gonogo_seed303"}
