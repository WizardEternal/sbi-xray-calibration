r"""Phase-5 NS pilot TIMING probe (per-level wall-clock for the extrapolation).

A small, self-contained probe that measures NS wall-clock + n_like_evals at each
count level on a couple of MEDIAN-brightness clean spectra, so the per-level
extrapolation to the full ~100-spectrum benchmark is honest without running the
unbounded full pilot (whose per-spectrum cost has huge variance because the
log-uniform norm makes some "faint"-level draws actually high-count -> very tight
posterior -> slow NS).

To keep the probe bounded we (a) pick clean draws whose total counts are CLOSE TO
the level median (so the timing is representative, not a tail case), and (b) pass
a generous max_ncalls cap. Writes outputs/ns_bench/pilot_timing.json (separate from
results.jsonl so it never races the main runner).

Usage:
    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe scripts\ns_pilot_timing.py --per-level 2 --max-ncalls 300000
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from sbixcal import ns_bench as NB
from sbixcal import train_npe as _tn
from sbixcal import priors as _priors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-run", default="train_npe_prod")
    ap.add_argument("--response", default="NGC7793_ULX4_PN")
    ap.add_argument("--per-level", type=int, default=2)
    ap.add_argument("--max-ncalls", type=int, default=300000)
    ap.add_argument("--min-live", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args(argv)

    from sbixcal import responses as _responses
    from sbixcal import simulate as _sim

    out = _repo_root() / "outputs" / "ns_bench"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for level in ["faint", "medium", "bright"]:
        ckpt = _repo_root() / "outputs" / "models" / f"{args.train_run}_{level}"
        if not (ckpt / "arch.json").exists():
            print(f"[skip] {level}: no checkpoint")
            continue
        post, info = _tn.load_posterior(ckpt, device="cpu")
        with open(ckpt / "arch.json") as f:
            arch = json.load(f)
        prior_cfg = arch["prior_cfg"]
        base_model = arch["base_model"]
        exposure_s = float(arch["exposure_s"])
        param_names = list(arch["param_names"])
        median_counts = float(arch.get("median_total_counts", np.nan))
        base = _responses.load_base_obsconf(args.response)
        oc = _responses.scale_exposure(base, exposure_s)

        def model_counts_fn(theta_arr, _oc=oc):
            return _sim.fold_theta(base_model, param_names, theta_arr, _oc)

        # draw a pool, pick the spectra whose counts are nearest the level median
        rng = np.random.default_rng(args.seed + hash(level) % 1000)
        theta, x_exp, _ = _sim.simulate_spectra(
            base_model, prior_cfg, oc, 60, rng, apply_poisson=False,
            seed_for_fakeit=args.seed)
        rng_p = np.random.default_rng(args.seed + 1)
        x = rng_p.poisson(np.clip(x_exp, 0.0, None)).astype(np.float64)
        tot = x.sum(axis=1)
        order = np.argsort(np.abs(tot - median_counts))
        pick = order[: args.per_level]

        for k, idx in enumerate(pick):
            counts = x[idx]
            truth = theta[idx]
            ns = NB.run_ns_one(
                counts, model_counts_fn, prior_cfg, param_names,
                min_num_live_points=args.min_live, max_ncalls=args.max_ncalls,
                dlogz=0.5, seed=args.seed + k)
            npe = NB.run_npe_one(post, counts, param_names, n_samples=2000,
                                 seed=args.seed + k)
            low, high = _priors.prior_bounds(prior_cfg, param_names)
            agree = NB.quantile_agreement(ns.quantiles, npe.quantiles,
                                          param_names, low, high)
            inside = []
            for j, nm in enumerate(param_names):
                lo = ns.quantiles[nm]["0.05"]; hi = ns.quantiles[nm]["0.95"]
                inside.append(bool(lo <= truth[j] <= hi))
            row = {
                "level": level, "median_counts": median_counts,
                "spectrum_counts": int(round(float(tot[idx]))),
                "ns_wall_s": ns.wall_s, "ns_n_like_evals": ns.n_like_evals,
                "ns_niter": ns.niter, "ns_logz": ns.logz,
                "ns_capped": bool(ns.n_like_evals >= args.max_ncalls),
                "npe_ms": npe.sample_wall_s * 1e3,
                "q_agree": agree["mean_abs_norm"],
                "truth_in_90": float(np.mean(inside)),
            }
            rows.append(row)
            print(f"[{level} #{k}] counts={row['spectrum_counts']:6d} "
                  f"NS={ns.wall_s:7.1f}s evals={ns.n_like_evals:7d}"
                  f"{' (CAPPED)' if row['ns_capped'] else ''} "
                  f"NPE={row['npe_ms']:.0f}ms qagree={row['q_agree']:.3f} "
                  f"truth_in_90={row['truth_in_90']:.2f}")

    (out / "pilot_timing.json").write_text(json.dumps(rows, indent=2),
                                           encoding="utf-8")
    print(f"\n[written] {out / 'pilot_timing.json'}  ({len(rows)} spectra)")

    # per-level summary + 100-spectrum extrapolation
    print("\n=== per-level NS wall-clock (pilot) ===")
    print("| level | ~counts | NS s/spec (mean) | NS evals (mean) | NPE ms/spec | q-agree |")
    print("|---|---|---|---|---|---|")
    by = {}
    for lvl in ["faint", "medium", "bright"]:
        rs = [r for r in rows if r["level"] == lvl]
        if not rs:
            continue
        ns_s = np.mean([r["ns_wall_s"] for r in rs])
        ev = np.mean([r["ns_n_like_evals"] for r in rs])
        npe_ms = np.mean([r["npe_ms"] for r in rs])
        qa = np.mean([r["q_agree"] for r in rs])
        by[lvl] = ns_s
        print(f"| {lvl} | {rs[0]['median_counts']:.0f} | {ns_s:.1f} | {ev:.0f} | "
              f"{npe_ms:.0f} | {qa:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
