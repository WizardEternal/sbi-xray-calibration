"""Reseed robustness summary: read outputs/reseed/summary.jsonl and classify the
bright-level NPE over-confidence as reproducible across training runs, confined
to a single run, or mixed.

Reads the one-line-per-variant summaries written by scripts/run_reseed_pack.py
and prints:
  * a per-variant table (cov@50/68/90, raw coverage deviation, SBC KS p-min,
    epochs/cap);
  * the detector spot-check rows (B1 / B4 D1 AUC vs expected), if present;
  * one line reading the numbers.

Classification rule, over the three full-reseed variants seed101/202/303:
  * ROBUST: every reseed shows raw coverage deviation > ROBUST_DEV (0.06), so
    the over-confidence reproduces across independent training runs.
  * FRAGILE: some reseed is near-calibrated (raw deviation < FRAGILE_DEV 0.03),
    so the bright over-confidence is at least partly a single-run artifact.
  * MIXED: neither. Some reseeds are clearly over-confident, none is cleanly
    near-calibrated, and not all clear the ROBUST floor.

The uncapped variant is the epoch-cap control and is reported separately:
  * uncapped raw deviation < FRAGILE_DEV (0.03) -> the over-confidence is an
    epoch-cap / undertraining artifact.
  * uncapped raw deviation > ROBUST_DEV (0.06) -> it survives 400 epochs, so
    the cap is not the mechanism.
  * in between -> the cap explains part of it; read the coverage curve.

The classification logic (``classify``) is pure and unit-tested
(tests/test_analyze_reseed.py); this script is its CLI and table formatter.

Usage (repo venv):
    .venv\\Scripts\\python.exe scripts\\analyze_reseed.py
    .venv\\Scripts\\python.exe scripts\\analyze_reseed.py --summary outputs/reseed/summary.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sbixcal._shared import _repo_root, _read_jsonl

# ---- thresholds ------------------------------------
ROBUST_DEV = 0.06      # reseed raw coverage deviation above this = over-confident
FRAGILE_DEV = 0.03     # reseed raw coverage deviation below this = near-calibrated

RESEED_VARIANTS = ("gonogo_seed101", "gonogo_seed202", "gonogo_seed303")
UNCAPPED_VARIANT = "gonogo_uncapped"


def load_summary(path: Path) -> list[dict]:
    """Load summary.jsonl -> list of row dicts (skips blank/garbled lines).

    Thin name-preserving wrapper around sbixcal._shared._read_jsonl (same
    behavior: [] for a missing file, malformed lines skipped)."""
    return _read_jsonl(path)


def classify(rows: list[dict],
             robust_dev: float = ROBUST_DEV,
             fragile_dev: float = FRAGILE_DEV) -> dict:
    """Classify the bright over-confidence from summary rows. Pure (unit-tested).

    Looks at the ``kind == "calibration"`` rows. Uses the three reseed variants'
    ``cov_dev_raw`` for the ROBUST/FRAGILE/MIXED classification and the uncapped
    variant's ``cov_dev_raw`` for the epoch-cap control.

    Returns a dict with: ``verdict`` (ROBUST|FRAGILE|MIXED|INCOMPLETE),
    ``reseed_devs`` ({variant: dev}), ``uncapped_dev`` (float|None),
    ``uncapped_interpretation`` (str|None), ``recommendation`` (str).
    """
    cal = {r["variant"]: r for r in rows
           if r.get("kind") == "calibration" and "variant" in r}

    reseed_devs = {v: cal[v]["cov_dev_raw"] for v in RESEED_VARIANTS if v in cal}
    uncapped_dev = cal[UNCAPPED_VARIANT]["cov_dev_raw"] if UNCAPPED_VARIANT in cal else None

    # ---- reseed classification ----
    n_present = len(reseed_devs)
    if n_present == 0:
        verdict = "INCOMPLETE"
    else:
        devs = list(reseed_devs.values())
        all_robust = all(d > robust_dev for d in devs)
        any_fragile = any(d < fragile_dev for d in devs)
        if any_fragile:
            verdict = "FRAGILE"
        elif all_robust and n_present == len(RESEED_VARIANTS):
            verdict = "ROBUST"
        elif all_robust and n_present < len(RESEED_VARIANTS):
            # all *present* reseeds are over-confident but not all three are in yet.
            verdict = "INCOMPLETE"
        else:
            verdict = "MIXED"

    # ---- uncapped (epoch-cap control) interpretation ----
    uncapped_interp = None
    if uncapped_dev is not None:
        if uncapped_dev < fragile_dev:
            uncapped_interp = (
                "the uncapped flow is near-calibrated (raw dev "
                f"{uncapped_dev:.3f} < {fragile_dev}), so the over-confidence is "
                "an epoch-cap / undertraining artifact.")
        elif uncapped_dev > robust_dev:
            uncapped_interp = (
                f"the over-confidence persists at 400 epochs (raw dev "
                f"{uncapped_dev:.3f} > {robust_dev}), so the mechanism is not just "
                "the epoch cap; it is structural to this single-round wide-prior "
                "amortized setup.")
        else:
            uncapped_interp = (
                f"the cap partially explains it (raw dev {uncapped_dev:.3f} sits "
                f"between {fragile_dev} and {robust_dev}); read the coverage "
                "curve.")

    # ---- reading line ----
    if verdict == "ROBUST":
        rec = ("ROBUST: all three reseeds reproduce the bright over-confidence "
               "(dev > %.2f). It is not confined to one training run." % robust_dev)
    elif verdict == "FRAGILE":
        rec = ("FRAGILE: at least one reseed is near-calibrated (dev < %.2f), so "
               "the over-confidence is training-run-dependent. Report the reseed "
               "spread." % fragile_dev)
    elif verdict == "MIXED":
        rec = ("MIXED: the reseeds neither all clear the robust floor nor go "
               "near-calibrated. The effect is real but varies across runs, so "
               "report the full spread.")
    else:
        rec = ("INCOMPLETE: not all three reseed variants are in summary.jsonl "
               "yet. Run the remaining variants before reading the "
               "classification.")
    if uncapped_interp is not None:
        rec += "  Uncapped: " + uncapped_interp

    return {
        "verdict": verdict,
        "reseed_devs": reseed_devs,
        "uncapped_dev": uncapped_dev,
        "uncapped_interpretation": uncapped_interp,
        "recommendation": rec,
        "robust_dev": robust_dev,
        "fragile_dev": fragile_dev,
    }


def _fmt_table(rows: list[dict]) -> str:
    """Format the per-variant table and the spot-check rows as text."""
    cal = [r for r in rows if r.get("kind") == "calibration"]
    spot = [r for r in rows if r.get("kind") == "detect_spot"]

    lines = []
    lines.append("| variant | ~counts | epochs/cap | cov@50 | cov@68 | cov@90 | "
                 "dev raw | dev conf | SBC KS p-min |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    order = list(RESEED_VARIANTS) + [UNCAPPED_VARIANT]
    cal_by_v = {r["variant"]: r for r in cal}
    for v in order + [r["variant"] for r in cal if r["variant"] not in order]:
        if v not in cal_by_v:
            continue
        r = cal_by_v[v]
        ep = f"{r.get('epochs_trained','?')}/{r.get('max_num_epochs','?')}"
        lines.append(
            f"| {v} | {r.get('median_total_counts',0):.0f} | {ep} | "
            f"{r.get('cov50',float('nan')):.3f} | {r.get('cov68',float('nan')):.3f} | "
            f"{r.get('cov90',float('nan')):.3f} | {r.get('cov_dev_raw',float('nan')):.3f} | "
            f"{r.get('cov_dev_conformal',float('nan')):.3f} | "
            f"{r.get('sbc_ks_p_min',float('nan')):.2e} |")

    if spot:
        lines.append("")
        lines.append("Detector spot-check (seed101 flow, D1 PPC):")
        lines.append("| family | strength | D1 AUC | expected |")
        lines.append("|---|---|---|---|")
        for r in spot:
            lines.append(
                f"| {r.get('family')} | {r.get('strength'):g} | "
                f"{r.get('auc',float('nan')):.3f} | ~{r.get('expected_auc',float('nan')):.2f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reseed robustness summary")
    ap.add_argument("--summary", default=None,
                    help="path to summary.jsonl (default outputs/reseed/summary.jsonl)")
    args = ap.parse_args(argv)
    summary_path = Path(args.summary) if args.summary else \
        _repo_root() / "outputs" / "reseed" / "summary.jsonl"

    rows = load_summary(summary_path)
    if not rows:
        print(f"No summary rows found at {summary_path}. Run "
              f"scripts/run_reseed_pack.py first.")
        return 1

    print(f"=== reseed robustness ({summary_path}) ===\n")
    print(_fmt_table(rows))

    res = classify(rows)
    print("\n--- reseed coverage deviations (raw) ---")
    for v in RESEED_VARIANTS:
        d = res["reseed_devs"].get(v)
        print(f"  {v}: {d:.3f}" if d is not None else f"  {v}: (missing)")
    if res["uncapped_dev"] is not None:
        print(f"  {UNCAPPED_VARIANT}: {res['uncapped_dev']:.3f}")

    print(f"\nclassification: {res['verdict']}")
    print(res["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
