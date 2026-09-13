r"""N* (source-count threshold) derivation from the paired sub-percent
gain-bias JSONs written by ``eval_gainmarg_paired.py`` /
``eval_gainmarg_paired_bright.py`` into
``outputs/gain_marg/subpercent_bias_2026-09-08/``.

Field paths used (pinned by reading the shipped 3% JSONs and the committed
regression test, ``BASE/run_regression_test.sh``):

    paired.gamma_bias_delta.{fixed,gain_marg}.{mean,sd,se,n}
        -- the PAIRED (shifted-minus-clean, same CRN-matched theta) delta in
        posterior-median Gamma, per flow. ``mean``/``sd`` are the mean and
        ddof=1 sample SD of the 500 (or N) per-pair deltas; ``se = sd /
        sqrt(n)``. This is the ONLY paired Gamma statistic in the JSON --
        there is no separate posterior-mean-vs-posterior-median split; "mean"
        already means "mean of the per-pair posterior-median deltas". The
        "fixed" and "gain_marg" blocks are two different FLOWS, not two
        different statistics of the same flow.

    cases.clean.fixed.gamma.bias_std
        -- sigma: the ddof=1 SD of (posterior median - truth) on the single
        CLEAN arm, fixed flow, 500 spectra. This is the point-estimate
        scatter used as "sigma" in N* = (sigma/b)^2 (paper: 0.37 medium,
        0.34 bright -- bright is unambiguous, "the bright one on the capped
        production flow" = fixed; medium is ~equally close to fixed (0.3704)
        and gain_marg (0.3670) -- this script uses fixed for both
        levels for consistency).

Only json / numpy / argparse / glob / pathlib / math / sys (stdlib + numpy).
No torch, no sbixcal import -- this script must run standalone off the
committed JSONs.

Usage (repo venv, from the repo root):
    .venv\Scripts\python.exe scripts\nstar_from_paired.py
    .venv\Scripts\python.exe scripts\nstar_from_paired.py --levels medium
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
FLOWS = ("fixed", "gain_marg")
DEFAULT_LEVELS = ("medium", "bright")
DEFAULT_AMPS = ("0.001", "0.002", "0.003", "0.005", "0.01")
DEFAULT_POOL_AMP = "0.003"

# Default-seed constants baked into eval_gainmarg_paired.py before the --seed
# flag was added (SEED = 20260611 -> SEED_THETA = SEED+60000, SEED_POISSON_BASE
# = SEED+70000). The g0.001 / g0.002 probe JSONs predate the --seed edit and
# so carry no explicit config['seed'] key; their seed_theta/seed_poisson_base
# are checked against these to confirm they used the same default seed before
# substituting it in (a documented, verified fallback -- not a silent one; a
# WARNING is printed every time this path is taken).
IMPLIED_DEFAULT_SEED = 20260611
IMPLIED_SEED_THETA = 20320611
IMPLIED_SEED_POISSON_BASE = 20330611

# Extrapolated bias points, taken verbatim from the paper
# (Section F17 text, "grows linearly in g-1"): medium is given directly at
# both 0.3% and 0.1%; bright is given only at 0.3% (0.1x the shipped 3%
# mean). The 0.1% bright point is this script's own linear extension (1/30x
# the 3% mean, matching the 0.3%/3% = 0.1x and 0.1%/3% = 1/30 amplitude
# ratios) -- flagged as EXTRAPOLATED-EXT in the output, not asked for
# verbatim in the brief.
MEDIUM_B_EXT = {"0.003": 0.0018, "0.001": 0.0006}
BRIGHT_B_EXT_SCALE = {"0.003": 0.1, "0.001": 1.0 / 30.0}

Z_3SIGMA = 3.0
Z_2SIGMA = 2.0
RESOLVE_THRESHOLD = 2.0  # |b|/SE >= this to call the pooled 0.3% bias resolved


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def fail(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing JSON: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_field(d: dict, dotted: str, ctx: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            fail(f"field path '{dotted}' missing in {ctx}")
        cur = cur[part]
    return cur


def probe_path(base: Path, level: str, amp: str) -> Path:
    return base / level / f"g{amp}" / f"paired_gain_bias_{level}.json"


def paired_block(d: dict, flow: str, ctx: str) -> dict:
    blk = get_field(d, f"paired.gamma_bias_delta.{flow}", ctx)
    for k in ("mean", "sd", "se", "n"):
        if k not in blk:
            fail(f"paired.gamma_bias_delta.{flow}.{k} missing in {ctx}")
    return blk


def sigma_fixed(d: dict, ctx: str) -> float:
    return float(get_field(d, "cases.clean.fixed.gamma.bias_std", ctx))


def sigma_gain_marg(d: dict, ctx: str) -> float:
    return float(get_field(d, "cases.clean.gain_marg.gamma.bias_std", ctx))


def resolved_seed(cfg: dict, ctx: str) -> tuple[int, bool]:
    """Return (seed, used_fallback). Fails loudly if seed_theta/seed_poisson_base
    don't match the known default when 'seed' itself is absent (i.e. this is
    NOT a silent substitution -- it is checked against known constants and
    reported)."""
    if "seed" in cfg:
        return int(cfg["seed"]), False
    st = cfg.get("seed_theta")
    sp = cfg.get("seed_poisson_base")
    if st == IMPLIED_SEED_THETA and sp == IMPLIED_SEED_POISSON_BASE:
        print(
            f"  WARNING: {ctx}: config has no 'seed' key (pre --seed-flag run); "
            f"seed_theta={st}, seed_poisson_base={sp} match the default-seed "
            f"({IMPLIED_DEFAULT_SEED}) derived constants -> treating as seed="
            f"{IMPLIED_DEFAULT_SEED}.",
            file=sys.stderr,
        )
        return IMPLIED_DEFAULT_SEED, True
    fail(
        f"{ctx}: config has no 'seed' key AND seed_theta/seed_poisson_base "
        f"({st}, {sp}) do not match the known default-seed derivation "
        f"({IMPLIED_SEED_THETA}, {IMPLIED_SEED_POISSON_BASE}) -- cannot infer "
        f"the seed used, refusing to substitute silently."
    )
    raise AssertionError("unreachable")  # fail() raises SystemExit


def discover_pool_seed_jsons(base: Path, level: str, pool_amp: str) -> list[tuple[int, Path]]:
    pattern = str(base / level / f"g{pool_amp}" / "seed*" / f"paired_gain_bias_{level}.json")
    found = []
    for p in glob.glob(pattern):
        m = re.search(r"seed(\d+)[\\/]", p.replace("\\", "/"))
        if not m:
            continue
        found.append((int(m.group(1)), Path(p)))
    found.sort(key=lambda t: t[0])
    return found


# ----------------------------------------------------------------------------
# pooling
# ----------------------------------------------------------------------------
def pool_groups(ns: np.ndarray, means: np.ndarray, sds: np.ndarray) -> dict:
    """Pool K independent groups (per-seed runs), each summarized by
    (n_k, mean_k, sd_k) with sd_k the ddof=1 sample SD, into a single pooled
    mean/SD/SE. ddof=1 throughout, matching eval_gainmarg_paired.py's own
    convention (`delta.std(ddof=1)` in `paired_stats()`).

    Two formulas are computed and reported:

    (a) "exact" -- SS decomposition via the parallel-axis theorem. This is
        the exact pooled ddof=1 variance for combining group summary
        statistics into one grand sample, valid for equal OR unequal n_k:
            SS_within  = sum_k (n_k - 1) * sd_k^2
            SS_between = sum_k n_k * (mean_k - grand_mean)^2
            pooled_var = (SS_within + SS_between) / (N_total - 1)

    (b) "approx" -- the equal-N formula: mean of the
        per-seed (ddof=1) variances, plus the (population, ddof=0) variance
        of the per-seed means about the pooled mean. This equals (a) up to
        an O(1/n) correction on the within-seed term (exactly (n-1)/n for
        equal n) and a ddof=0-vs-ddof=1 difference on the between-seed term
        (a further O(1/K) correction) -- both negligible at n=500, K=41/15
        (see the report's Choices section for the derivation and the
        measured size of the discrepancy).
    """
    ns = np.asarray(ns, dtype=float)
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    n_total = float(ns.sum())
    grand_mean = float((ns * means).sum() / n_total)

    ss_within = float(((ns - 1) * sds**2).sum())
    ss_between = float((ns * (means - grand_mean) ** 2).sum())
    pooled_var_exact = (ss_within + ss_between) / (n_total - 1)
    pooled_sd_exact = math.sqrt(pooled_var_exact)
    se_exact = pooled_sd_exact / math.sqrt(n_total)

    mean_var = float((sds**2).mean())
    between_var_pop = float(((means - grand_mean) ** 2).mean())
    pooled_var_approx = mean_var + between_var_pop
    pooled_sd_approx = math.sqrt(pooled_var_approx)
    se_approx = pooled_sd_approx / math.sqrt(n_total)

    k = len(means)
    per_seed_mean_sd = float(means.std(ddof=1)) if k > 1 else float("nan")
    per_seed_se = sds / np.sqrt(ns)
    mean_per_seed_se = float(per_seed_se.mean())
    consistency_ratio = (
        per_seed_mean_sd / mean_per_seed_se if mean_per_seed_se > 0 else float("nan")
    )

    return {
        "n_total": int(n_total),
        "n_seeds": k,
        "grand_mean": grand_mean,
        "pooled_sd_exact": pooled_sd_exact,
        "se_exact": se_exact,
        "pooled_sd_approx": pooled_sd_approx,
        "se_approx": se_approx,
        "per_seed_mean_min": float(means.min()),
        "per_seed_mean_max": float(means.max()),
        "consistency_ratio": consistency_ratio,
    }


# ----------------------------------------------------------------------------
# N* lines
# ----------------------------------------------------------------------------
def nstar_lines(level: str, flow: str, sigma: float, pooled: dict, b_ext: dict) -> dict:
    b_meas = pooled["grand_mean"]
    se_meas = pooled["se_exact"]
    sd_meas = pooled["pooled_sd_exact"]
    resolved = abs(b_meas) / se_meas >= RESOLVE_THRESHOLD if se_meas > 0 else False

    out = {"flow": flow, "sigma": sigma, "b_measured": b_meas, "se_measured": se_meas}

    if resolved:
        out["nstar_measured"] = (sigma / b_meas) ** 2
    else:
        out["nstar_measured"] = None
        out["nstar_measured_note"] = (
            f"unresolved: |b|/SE = {abs(b_meas) / se_meas:.3f} < {RESOLVE_THRESHOLD}"
        )

    out["nstar_extrapolated"] = {}
    out["n3sigma_extrapolated"] = {}
    for amp_key, b in b_ext.items():
        out["nstar_extrapolated"][amp_key] = (sigma / b) ** 2
        out["n3sigma_extrapolated"][amp_key] = (Z_3SIGMA * sd_meas / b) ** 2

    b_upper = abs(b_meas) + Z_2SIGMA * se_meas
    out["b_upper_2sigma"] = b_upper
    out["nstar_lower_2sigma"] = (sigma / b_upper) ** 2 if b_upper > 0 else None

    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--base",
        default="outputs/gain_marg/subpercent_bias_2026-09-08",
        help="BASE dir holding <level>/g<amp>/... probe and phase-2 seed JSONs.",
    )
    ap.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
    ap.add_argument("--amps", nargs="+", default=list(DEFAULT_AMPS))
    ap.add_argument("--pool-amp", default=DEFAULT_POOL_AMP)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument(
        "--shipped-json",
        default=None,
        help="dir holding the shipped 3%% JSONs (paired_gain_bias_{level}.json); "
        "default: BASE.parent (i.e. outputs/gain_marg/).",
    )
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        fail(f"--base does not exist: {base}")
    out_json = Path(args.out_json) if args.out_json else base / "POOLED_SUMMARY.json"
    out_md = Path(args.out_md) if args.out_md else base / "POOLED_SUMMARY.md"
    shipped_dir = Path(args.shipped_json) if args.shipped_json else base.parent

    report = {"base": str(base), "levels": {}}
    md_lines: list[str] = []
    md_lines.append("# Pooled sub-percent gain-bias summary (auto-generated)\n")
    md_lines.append(f"BASE = `{base}`\n")

    # ---- shipped 3% JSONs, for sigma cross-check + bright b_ext basis ----
    shipped = {}
    for level in ("medium", "bright"):
        p = shipped_dir / f"paired_gain_bias_{level}.json"
        if p.exists():
            shipped[level] = load_json(p)
        else:
            print(f"  NOTE: shipped 3% JSON not found at {p} (only needed for bright b_ext basis)", file=sys.stderr)

    if "bright" in shipped:
        bright_3pct_mean_fixed = paired_block(shipped["bright"], "fixed", str(shipped_dir / "paired_gain_bias_bright.json"))["mean"]
    else:
        bright_3pct_mean_fixed = None

    for level in args.levels:
        print(f"\n{'=' * 70}\nLEVEL = {level}\n{'=' * 70}")
        level_report: dict = {"amps": {}, "pooled": {}, "nstar": {}}
        md_lines.append(f"\n## {level}\n")

        # ---- §2: per-amplitude single-seed table ----
        md_lines.append("\n### Per-amplitude, single seed (probe, seed 20260611)\n")
        md_lines.append("| amp | flow | mean | sd | se | n | seed |")
        md_lines.append("|---|---|---|---|---|---|---|")
        sigma_values_fixed = []
        sigma_values_gm = []
        for amp in args.amps:
            p = probe_path(base, level, amp)
            d = load_json(p)
            cfg = d.get("config", {})
            seed, fallback = resolved_seed(cfg, str(p))
            amp_report = {"path": str(p), "seed": seed, "seed_is_fallback": fallback, "flows": {}}
            for flow in FLOWS:
                blk = paired_block(d, flow, str(p))
                amp_report["flows"][flow] = blk
                print(
                    f"[{level}][amp={amp}][{flow}] mean={blk['mean']:+.6f} "
                    f"sd={blk['sd']:.6f} se={blk['se']:.6f} n={blk['n']} seed={seed}"
                )
                md_lines.append(
                    f"| {amp} | {flow} | {blk['mean']:+.6f} | {blk['sd']:.6f} | "
                    f"{blk['se']:.6f} | {blk['n']} | {seed}{'*' if fallback else ''} |"
                )
            sf = sigma_fixed(d, str(p))
            sg = sigma_gain_marg(d, str(p))
            sigma_values_fixed.append(sf)
            sigma_values_gm.append(sg)
            amp_report["sigma_fixed"] = sf
            amp_report["sigma_gain_marg"] = sg
            level_report["amps"][amp] = amp_report
        md_lines.append("\n(`*` = seed inferred from seed_theta/seed_poisson_base, no explicit `config.seed` key)\n")

        sigma_arr_fixed = np.array(sigma_values_fixed)
        sigma_arr_gm = np.array(sigma_values_gm)
        print(
            f"\n[{level}] sigma (cases.clean.fixed.gamma.bias_std) across the "
            f"{len(args.amps)} probe amps: mean={sigma_arr_fixed.mean():.6f} "
            f"min={sigma_arr_fixed.min():.6f} max={sigma_arr_fixed.max():.6f} "
            f"(spread {(sigma_arr_fixed.max() - sigma_arr_fixed.min()) / sigma_arr_fixed.mean() * 100:.3f}% of mean)"
        )
        print(
            f"[{level}] sigma (cases.clean.gain_marg.gamma.bias_std) across the "
            f"{len(args.amps)} probe amps: mean={sigma_arr_gm.mean():.6f} "
            f"min={sigma_arr_gm.min():.6f} max={sigma_arr_gm.max():.6f}"
        )
        level_report["sigma_fixed_across_amps"] = {
            "mean": float(sigma_arr_fixed.mean()), "min": float(sigma_arr_fixed.min()), "max": float(sigma_arr_fixed.max()),
        }
        level_report["sigma_gain_marg_across_amps"] = {
            "mean": float(sigma_arr_gm.mean()), "min": float(sigma_arr_gm.min()), "max": float(sigma_arr_gm.max()),
        }

        # ---- §3: pooled 0.3% results ----
        pool_amp = args.pool_amp
        probe_p = probe_path(base, level, pool_amp)
        probe_d = load_json(probe_p)
        probe_cfg = probe_d.get("config", {})
        probe_seed, probe_fallback = resolved_seed(probe_cfg, str(probe_p))

        seed_jsons = discover_pool_seed_jsons(base, level, pool_amp)
        expected_max = 40 if level == "medium" else 14
        found_seed_nums = [s for s, _ in seed_jsons]
        missing = sorted(set(range(1, expected_max + 1)) - set(found_seed_nums))
        if missing:
            print(f"  WARNING [{level}]: missing phase-2 seeds at pool-amp {pool_amp}: {missing}", file=sys.stderr)
        if not seed_jsons:
            fail(f"[{level}] no phase-2 seed JSONs found under {base / level / ('g' + pool_amp)}/seed*/")

        print(
            f"\n[{level}] pooling amp={pool_amp}: probe (seed {probe_seed}) + "
            f"{len(seed_jsons)} phase-2 seeds {found_seed_nums[0]}..{found_seed_nums[-1]} "
            f"(missing: {missing if missing else 'none'})"
        )

        md_lines.append(f"\n### Pooled amp={pool_amp}, {level}\n")
        md_lines.append(f"Probe seed {probe_seed}{'*' if probe_fallback else ''} + phase-2 seeds "
                        f"{found_seed_nums[0]}-{found_seed_nums[-1]} "
                        f"({len(seed_jsons)} found, missing: {missing if missing else 'none'}).\n")
        md_lines.append("| flow | N_total | n_seeds | pooled mean | pooled SD (exact) | pooled SE (exact) | pooled SD (approx) | pooled SE (approx) | consistency ratio | per-seed mean min/max |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|")

        all_docs = [(probe_seed, probe_d)] + [
            (s, load_json(p)) for s, p in seed_jsons
        ]

        # distinct-seed check on seed_theta
        seed_theta_vals = []
        for s, d in all_docs:
            st = d.get("config", {}).get("seed_theta")
            if st is None:
                fail(f"[{level}] seed {s}: config.seed_theta missing")
            seed_theta_vals.append(st)
        distinct_count = len(set(seed_theta_vals))
        print(
            f"[{level}] distinct seed_theta values across {len(seed_theta_vals)} pooled runs: "
            f"{distinct_count} (expect {len(seed_theta_vals)})"
        )
        level_report["distinct_seed_theta_count"] = distinct_count
        level_report["distinct_seed_theta_expected"] = len(seed_theta_vals)

        # n != 500 check
        n_mismatches = []
        for s, d in all_docs:
            for flow in FLOWS:
                n = paired_block(d, flow, f"seed{s}")["n"]
                if n != 500:
                    n_mismatches.append((s, flow, n))
        if n_mismatches:
            print(f"  WARNING [{level}]: N != 500 found: {n_mismatches}", file=sys.stderr)
        level_report["n_mismatches"] = n_mismatches

        level_report["pooled"][pool_amp] = {}
        for flow in FLOWS:
            ns = np.array([paired_block(d, flow, f"seed{s}")["n"] for s, d in all_docs])
            means = np.array([paired_block(d, flow, f"seed{s}")["mean"] for s, d in all_docs])
            sds = np.array([paired_block(d, flow, f"seed{s}")["sd"] for s, d in all_docs])
            pooled = pool_groups(ns, means, sds)
            level_report["pooled"][pool_amp][flow] = pooled
            print(
                f"[{level}][{flow}] pooled: N_total={pooled['n_total']} n_seeds={pooled['n_seeds']} "
                f"mean={pooled['grand_mean']:+.6f} sd_exact={pooled['pooled_sd_exact']:.6f} "
                f"se_exact={pooled['se_exact']:.6f} sd_approx={pooled['pooled_sd_approx']:.6f} "
                f"se_approx={pooled['se_approx']:.6f} consistency_ratio={pooled['consistency_ratio']:.4f} "
                f"per_seed_mean=[{pooled['per_seed_mean_min']:+.5f}, {pooled['per_seed_mean_max']:+.5f}]"
            )
            md_lines.append(
                f"| {flow} | {pooled['n_total']} | {pooled['n_seeds']} | {pooled['grand_mean']:+.6f} | "
                f"{pooled['pooled_sd_exact']:.6f} | {pooled['se_exact']:.6f} | "
                f"{pooled['pooled_sd_approx']:.6f} | {pooled['se_approx']:.6f} | "
                f"{pooled['consistency_ratio']:.4f} | "
                f"[{pooled['per_seed_mean_min']:+.5f}, {pooled['per_seed_mean_max']:+.5f}] |"
            )

        # ---- §4: N* lines ----
        sigma_pool = level_report["amps"][pool_amp]["sigma_fixed"]
        print(f"\n[{level}] sigma (fixed-flow clean-arm bias_std @ pool-amp) = {sigma_pool:.6f}")

        md_lines.append(f"\n### N* lines, {level} (sigma = {sigma_pool:.6f}, fixed-flow clean-arm bias_std)\n")
        md_lines.append("| flow | b measured (0.3%) | SE | resolved (|b|/SE>=2)? | N* measured | b_ext (amp) | N* extrapolated | N_3sigma(amp) | 2sigma upper |b| | N* lower limit |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|")

        if level == "medium":
            b_ext_map = MEDIUM_B_EXT
        else:
            if bright_3pct_mean_fixed is None:
                fail("bright b_ext requested but shipped bright 3% JSON not found")
            b_ext_map = {k: v * bright_3pct_mean_fixed for k, v in BRIGHT_B_EXT_SCALE.items()}
            print(f"[{level}] b_ext basis: shipped 3% fixed-flow mean = {bright_3pct_mean_fixed:.6f}")

        for flow in FLOWS:
            pooled = level_report["pooled"][pool_amp][flow]
            lines = nstar_lines(level, flow, sigma_pool, pooled, b_ext_map)
            level_report["nstar"][flow] = lines
            resolved_str = (
                f"N*={lines['nstar_measured']:.4g}" if lines["nstar_measured"] is not None
                else lines["nstar_measured_note"]
            )
            print(
                f"[{level}][{flow}] b_measured={lines['b_measured']:+.6f} SE={lines['se_measured']:.6f} "
                f"-> {resolved_str}"
            )
            for amp_key, nstar_ext in lines["nstar_extrapolated"].items():
                tag = "" if (level == "medium" or amp_key == "0.003") else " [EXTRAPOLATED-EXT, script's own 1/30x extension]"
                n3s = lines["n3sigma_extrapolated"][amp_key]
                print(
                    f"[{level}][{flow}] amp={amp_key}: b_ext={b_ext_map[amp_key]:+.6f} "
                    f"N*_ext={nstar_ext:.4g} N_3sigma={n3s:.4g}{tag}"
                )
                md_lines.append(
                    f"| {flow} | {lines['b_measured']:+.6f} | {lines['se_measured']:.6f} | "
                    f"{'YES' if lines['nstar_measured'] is not None else 'NO'} | "
                    f"{resolved_str} | {amp_key} ({b_ext_map[amp_key]:+.6f}){tag} | "
                    f"{nstar_ext:.4g} | {n3s:.4g} | "
                    f"{lines['b_upper_2sigma']:.6f} | {lines['nstar_lower_2sigma']:.4g} |"
                )

        report["levels"][level] = level_report

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
