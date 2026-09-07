"""Paired nested-sampling test of the B4 gain-shift evidence.

A level-matched mean(logZ_mis) - mean(logZ_clean) over unmatched spectra is
confounded by total counts. This removes the confound by construction: the same
parameter draw is folded through the clean response and the 3%-gain-shifted
response, Poisson-realized with a matched seed (so total counts are ~identical),
and the clean (well-specified) model is fit to each by UltraNest. Paired
Delta logZ = logZ(gain-shifted) - logZ(clean); ~0 means no evidence penalty from
the gain shift.

Medium count regime by default. CPU-only (set by caller env). Resumable: appends
one row per spectrum to the output jsonl and skips indices already present.

CLI
---
Every default reproduces the historical behaviour exactly (exposure 353.4, gain
1.03, N=12, theta seed 20260630, Poisson seed 1000+i, NS seed i,
min_num_live_points=400, dlogz=0.5, max_ncalls=400000, no point store), so
``python scripts/paired_ns_gain_check.py`` with no arguments is the old script.

    # one pair as its own process, own jsonl, own point stores
    python scripts/paired_ns_gain_check.py --pairs 3 \
        --out outputs/ns_bench/revision_2026-09-02/medium/pair3.jsonl \
        --log-dir-root outputs/ns_bench/revision_2026-09-02/medium/runs

    # the bright arm
    python scripts/paired_ns_gain_check.py --exposure 3534.0 --pairs 0-11 \
        --out outputs/ns_bench/revision_2026-09-02/bright/pair{i}.jsonl ...

Point store
-----------
``--log-dir-root <root>`` gives every NS run an ISOLATED directory
``<root>/pair{i}_clean/`` and ``<root>/pair{i}_gain/``, so UltraNest writes
``chains/run.txt`` + ``results/points.hdf5`` and the Higson (2018) thread
bootstrap can be computed afterwards
(``scripts/higson_common.reconstruct_tree``). Without it the runs stay in
memory, exactly as the committed runs were. The layout matches
``outputs/ns_bench/higson/runs/gain_pair{i}_{clean,gain}/`` up to the directory
name, so the same analysis code reads both.

CAVEAT on the committed jsonl
-----------------------------
``outputs/ns_bench/paired_gain_check.jsonl`` (written 2026-07-01 19:00) is NOT
reproducible by this script: the script was edited at 20:04 the same day and the
current draw gives different spectra (i=0 -> 1403 counts vs the committed 468),
with no functional variant anywhere in git history. Do not treat a new run as a
reproduction of those rows; it is a fresh, reproducible replacement set.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse, json, time
from pathlib import Path
import numpy as np

from sbixcal import responses as R, simulate as S, models as M, priors as P
from sbixcal import ns_bench as NB

BASE_MODEL = "tbabs_powerlaw_bb"
RESP = "NGC7793_ULX4_PN"
EXPOSURE = 353.4            # medium (~1000 counts), matches sim_modelA_prod
GAIN = 1.03                 # 3% gain shift (the B4 strength the benchmark reports)
N = 12                      # matched-count pairs (deterministic from the seeds below)
THETA_SEED = 20260630       # draws the N parameter vectors
POISSON_SEED_BASE = 1000    # Poisson seed of pair i is POISSON_SEED_BASE + i
MIN_NUM_LIVE_POINTS = 400
DLOGZ = 0.5
MAX_NCALLS = 400000
PRIOR_CFG = {
    "tbabs_1_nh":         {"dist": "uniform",    "low": 0.15,   "high": 0.35},
    "powerlaw_1_alpha":   {"dist": "uniform",    "low": 1.0,    "high": 3.0},
    "powerlaw_1_norm":    {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":  {"dist": "uniform",    "low": 0.3,    "high": 3.0},
    "blackbodyrad_1_norm":{"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}
OUT = Path("outputs/ns_bench/paired_gain_check.jsonl")


def parse_pairs(spec: str, n: int) -> list[int]:
    """``"3"`` / ``"1,2,5"`` / ``"0-11"`` / ``"1-3,7,11"`` -> sorted unique indices.

    ``None`` or an empty string means all ``n``. Every index must be in
    ``[0, n)``; the draw is a single block of ``n`` theta vectors, so asking for
    an index outside it would silently mean a different experiment."""
    if not spec:
        return list(range(n))
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-") and not part.lstrip().startswith("-"):
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = sorted(i for i in out if not (0 <= i < n))
    if bad:
        raise SystemExit(f"[err] --pairs index out of range for --n {n}: {bad}")
    return sorted(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("CLI")[0].strip(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--exposure", type=float, default=EXPOSURE,
                    help="exposure in s; 353.4 = medium, 3534.0 = bright")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="append-resumable jsonl, one row per pair")
    ap.add_argument("--log-dir-root", type=Path, default=None,
                    help="if set, each run gets an isolated UltraNest point store at "
                         "<root>/pair{i}_{clean,gain}/ (default: no point store, "
                         "in-memory, the committed behaviour)")
    ap.add_argument("--pairs", default=None,
                    help='which pair indices to run: "3", "1,2,5", "0-11", "1-3,7" '
                         "(default: all --n of them)")
    ap.add_argument("--max-ncalls", type=int, default=MAX_NCALLS)
    ap.add_argument("--min-num-live-points", type=int, default=MIN_NUM_LIVE_POINTS)
    ap.add_argument("--dlogz", type=float, default=DLOGZ)
    ap.add_argument("--gain", type=float, default=GAIN)
    ap.add_argument("--n", type=int, default=N,
                    help="size of the theta block; changing it changes EVERY pair's "
                         "theta, because the draw is one block of --n vectors")
    ap.add_argument("--seed", type=int, default=THETA_SEED,
                    help="theta-draw seed (np.random.default_rng)")
    ap.add_argument("--resume", default="overwrite",
                    choices=("overwrite", "resume", "resume-similar", "subfolder"),
                    help="UltraNest resume mode; only used with --log-dir-root")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the spectra, print counts per pair, run no NS")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    idx = parse_pairs(args.pairs, args.n)

    param_order = M.MODEL_PARAMS[BASE_MODEL]
    base = R.load_base_obsconf(RESP)
    clean_oc = R.scale_exposure(base, args.exposure)
    gain_oc = R.gain_shift_obsconf(clean_oc, args.gain)

    # well-specified model: fold through the nominal (clean) response
    def model_counts_fn(theta_arr):
        return S.fold_theta(BASE_MODEL, param_order, theta_arr, clean_oc)

    rng = np.random.default_rng(args.seed)
    samples = P.sample_prior(PRIOR_CFG, param_order, args.n, rng)
    theta = np.stack([np.asarray(samples[p]) for p in param_order], axis=1)

    out_path = Path(args.out)
    done = set()
    if not args.dry_run:                      # a dry run touches no files and
        out_path.parent.mkdir(parents=True, exist_ok=True)   # never resume-skips,
        if out_path.exists():                 # so it always prints every index
            for l in open(out_path):
                if l.strip():
                    done.add(json.loads(l)["i"])

    ns_kw = dict(min_num_live_points=args.min_num_live_points,
                 dlogz=args.dlogz, max_ncalls=args.max_ncalls)
    print(f"[cfg] exposure={args.exposure} gain={args.gain} n={args.n} seed={args.seed} "
          f"pairs={idx} ns={ns_kw} log_dir_root={args.log_dir_root} out={out_path}",
          flush=True)

    for i in idx:
        if i in done:
            print(f"[skip] spectrum {i} already done")
            continue
        th = theta[i:i + 1]
        lam_clean = np.asarray(S.fold_theta(BASE_MODEL, param_order, th, clean_oc))[0]
        lam_gain = np.asarray(S.fold_theta(BASE_MODEL, param_order, th, gain_oc))[0]
        # matched-seed Poisson so the two realizations are paired
        data_clean = np.random.default_rng(POISSON_SEED_BASE + i).poisson(np.maximum(lam_clean, 0)).astype(float)
        data_gain = np.random.default_rng(POISSON_SEED_BASE + i).poisson(np.maximum(lam_gain, 0)).astype(float)
        if args.dry_run:
            print(f"[dry] i={i} counts {int(data_clean.sum())}/{int(data_gain.sum())}", flush=True)
            continue
        ld_clean = ld_gain = None
        if args.log_dir_root is not None:
            ld_clean = Path(args.log_dir_root) / f"pair{i}_clean"
            ld_gain = Path(args.log_dir_root) / f"pair{i}_gain"
        t0 = time.time()
        rc = NB.run_ns_one(data_clean, model_counts_fn, PRIOR_CFG, param_order,
                           seed=i, log_dir=ld_clean, resume=args.resume, **ns_kw)
        rg = NB.run_ns_one(data_gain, model_counts_fn, PRIOR_CFG, param_order,
                           seed=i, log_dir=ld_gain, resume=args.resume, **ns_kw)
        row = {
            "i": i,
            "counts_clean": int(data_clean.sum()),
            "counts_gain": int(data_gain.sum()),
            "logz_clean": rc.logz, "logzerr_clean": rc.logzerr,
            "logz_gain": rg.logz, "logzerr_gain": rg.logzerr,
            "d_paired": rg.logz - rc.logz,
            "wall_s": round(time.time() - t0, 1),
            "exposure": args.exposure, "gain": args.gain,
            "n_block": args.n, "theta_seed": args.seed,
            "ncall_clean": rc.n_like_evals, "ncall_gain": rg.n_like_evals,
            "max_ncalls": args.max_ncalls,
            "min_num_live_points": args.min_num_live_points, "dlogz": args.dlogz,
            "log_dir_clean": (str(ld_clean).replace("\\", "/") if ld_clean else None),
            "log_dir_gain": (str(ld_gain).replace("\\", "/") if ld_gain else None),
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[done] i={i} counts {row['counts_clean']}/{row['counts_gain']} "
              f"logZ_clean {rc.logz:.1f} logZ_gain {rg.logz:.1f} "
              f"d_paired {row['d_paired']:+.2f} (+/- {(rc.logzerr**2+rg.logzerr**2)**0.5:.2f}) "
              f"[{row['wall_s']}s]", flush=True)

    if args.dry_run:
        return 0
    rows = [json.loads(l) for l in open(out_path) if l.strip()]
    d = np.array([r["d_paired"] for r in rows])
    print("\n=== PAIRED RESULT (clean model fit to gain-shifted vs clean data, same theta) ===")
    print(f"file: {out_path}  exposure={args.exposure}")
    print(f"n={len(d)}  mean paired Delta logZ = {d.mean():+.2f} +/- "
          f"{(d.std(ddof=1)/len(d)**0.5) if len(d) > 1 else float('nan'):.2f} (SEM)")
    print(f"per-spectrum: {[round(x,2) for x in d]}")
    print("paired Delta logZ ~ 0 => no evidence penalty from the 3% gain shift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
