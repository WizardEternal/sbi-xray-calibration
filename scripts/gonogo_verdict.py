"""GO/NO-GO verdict -- read outputs/gonogo/summary.jsonl and decide whether the
primary bright-level NPE over-confidence is ROBUST, FRAGILE, or MIXED.

Reads the one-line-per-variant summaries written by scripts/run_gonogo.py and
prints:
  * a verdict table (one row per variant: cov@50/68/90, raw coverage deviation,
    SBC KS p-min, epochs/cap);
  * the detector spot-check rows (B1 / B4 D1 AUC vs expected), if present;
  * a recommendation line.

Verdict rule (over the three FULL-RESEED variants seed101/202/303):
  * ROBUST   -- every reseed shows raw coverage deviation > ROBUST_DEV (0.06):
                the over-confidence reproduces across independent training runs.
  * FRAGILE  -- ANY reseed is near-calibrated (raw deviation < FRAGILE_DEV 0.03):
                the primary is at least partly a single-run artifact.
  * MIXED    -- neither (some reseeds clearly over-confident, none cleanly
                near-calibrated, but not all above the ROBUST floor).

Uncapped interpretation (the epoch-cap mechanism test), reported separately:
  * uncapped raw deviation < FRAGILE_DEV (0.03)  -> "over-confidence is
    undertraining; the claim must soften."
  * uncapped raw deviation > ROBUST_DEV  (0.06)  -> "persists at 400 epochs --
    the mechanism is not just the cap."
  * in-between -> "partially explained by the cap (read the curve)."

The classification logic (``classify``) is pure and unit-tested
(tests/test_gonogo.py); this script is just its CLI + table formatter.

Usage (repo venv):
    .venv\\Scripts\\python.exe scripts\\gonogo_verdict.py
    .venv\\Scripts\\python.exe scripts\\gonogo_verdict.py --summary outputs/gonogo/summary.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- thresholds ------------------------------------
ROBUST_DEV = 0.06      # reseed raw coverage deviation ABOVE this = over-confident
FRAGILE_DEV = 0.03     # reseed raw coverage deviation BELOW this = near-calibrated

RESEED_VARIANTS = ("gonogo_seed101", "gonogo_seed202", "gonogo_seed303")
UNCAPPED_VARIANT = "gonogo_uncapped"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_summary(path: Path) -> list[dict]:
    """Load summary.jsonl -> list of row dicts (skips blank/garbled lines)."""
    rows = []
    if not Path(path).exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def classify(rows: list[dict],
             robust_dev: float = ROBUST_DEV,
             fragile_dev: float = FRAGILE_DEV) -> dict:
    """Classify the primary result from summary rows. PURE (unit-tested).

    Looks at the ``kind == "calibration"`` rows. Uses the three reseed variants'
    ``cov_dev_raw`` for the ROBUST/FRAGILE/MIXED verdict and the uncapped variant's
    ``cov_dev_raw`` for the mechanism interpretation.

    Returns a dict with: ``verdict`` (ROBUST|FRAGILE|MIXED|INCOMPLETE),
    ``reseed_devs`` ({variant: dev}), ``uncapped_dev`` (float|None),
    ``uncapped_interpretation`` (str|None), ``recommendation`` (str).
    """
    cal = {r["variant"]: r for r in rows
           if r.get("kind") == "calibration" and "variant" in r}

    reseed_devs = {v: cal[v]["cov_dev_raw"] for v in RESEED_VARIANTS if v in cal}
    uncapped_dev = cal[UNCAPPED_VARIANT]["cov_dev_raw"] if UNCAPPED_VARIANT in cal else None

    # ---- reseed verdict ----
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

    # ---- uncapped (epoch-cap mechanism) interpretation ----
    uncapped_interp = None
    if uncapped_dev is not None:
        if uncapped_dev < fragile_dev:
            uncapped_interp = (
                "over-confidence is undertraining: the uncapped flow is "
                f"near-calibrated (raw dev {uncapped_dev:.3f} < {fragile_dev}). "
                "The claim must soften to an epoch-cap / undertraining artifact.")
        elif uncapped_dev > robust_dev:
            uncapped_interp = (
                f"over-confidence persists at 400 epochs (raw dev {uncapped_dev:.3f} "
                f"> {robust_dev}) -- the mechanism is NOT just the epoch cap; it is "
                "structural to this single-round wide-prior amortized setup.")
        else:
            uncapped_interp = (
                f"partially explained by the cap (raw dev {uncapped_dev:.3f} sits "
                f"between {fragile_dev} and {robust_dev}); read the coverage curve.")

    # ---- recommendation line ----
    if verdict == "ROBUST":
        rec = ("ROBUST -- all reseeds reproduce the bright over-confidence "
               "(dev > %.2f). The primary is not a single-run fluke; keep the "
               "claim (scoped as in RESULTS.md M1)." % robust_dev)
    elif verdict == "FRAGILE":
        rec = ("FRAGILE -- at least one reseed is near-calibrated (dev < %.2f). "
               "The primary over-confidence is training-run-dependent; SOFTEN the "
               "claim and report the reseed spread." % fragile_dev)
    elif verdict == "MIXED":
        rec = ("MIXED -- reseeds neither all clear the robust floor nor any go "
               "near-calibrated. Report the full reseed spread; the effect is "
               "real but variable across runs.")
    else:
        rec = ("INCOMPLETE -- not all three reseed variants are in summary.jsonl "
               "yet. Run the remaining variants before reading the verdict.")
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
    """Format the per-variant verdict table + spot-check rows as text."""
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
    ap = argparse.ArgumentParser(description="GO/NO-GO verdict from summary.jsonl")
    ap.add_argument("--summary", default=None,
                    help="path to summary.jsonl (default outputs/gonogo/summary.jsonl)")
    args = ap.parse_args(argv)
    summary_path = Path(args.summary) if args.summary else \
        _repo_root() / "outputs" / "gonogo" / "summary.jsonl"

    rows = load_summary(summary_path)
    if not rows:
        print(f"No summary rows found at {summary_path}. Run scripts/run_gonogo.py first.")
        return 1

    print(f"=== GO/NO-GO verdict ({summary_path}) ===\n")
    print(_fmt_table(rows))

    res = classify(rows)
    print("\n--- reseed coverage deviations (raw) ---")
    for v in RESEED_VARIANTS:
        d = res["reseed_devs"].get(v)
        print(f"  {v}: {d:.3f}" if d is not None else f"  {v}: (missing)")
    if res["uncapped_dev"] is not None:
        print(f"  {UNCAPPED_VARIANT}: {res['uncapped_dev']:.3f}")

    print(f"\nVERDICT: {res['verdict']}")
    print(f"RECOMMENDATION: {res['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
