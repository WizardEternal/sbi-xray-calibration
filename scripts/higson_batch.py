r"""Batch runner for the thread-wise bootstrap.

For every job in ``higson_common.build_jobs()`` (13 runs, hard cap 16): reconstruct
the EXACT committed spectrum, run UltraNest with an ISOLATED per-run ``log_dir`` (the
only change from the committed runs, so the point store is written and threads can be
recovered), then read the point store and compute the Higson (2018) thread-wise
bootstrap sigma on logZ. Each job persists its result to a ``DONE.json`` marker the
MOMENT it finishes, so a killed batch resumes by re-running only the jobs missing a
marker (``--workers`` fans jobs out across processes; each writes its own marker).

Windows-safe: per-run isolated log_dir (no shared HDF5 -> no h5py locking), spawn
Pool with the ``if __name__ == "__main__"`` guard, OMP/MKL pinned to 1 thread per
worker (parallelism is ACROSS jobs).

Usage (repo venv):
    .venv\Scripts\python.exe scripts\higson_batch.py --workers 5
    .venv\Scripts\python.exe scripts\higson_batch.py --job gain_pair0_clean   # one job
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys, json, time, argparse, traceback, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import higson_common as HC

ROOT = Path(__file__).resolve().parents[1]
HGDIR = ROOT / "outputs" / "ns_bench" / "higson"
RUNS = HGDIR / "runs"
LOGS = HGDIR / "logs"
N_RESAMPLES = 400
BOOT_SEED = 12345


def committed_for(job: dict):
    """Committed reported logZ this job attaches to (for the reproduction delta).

    Gain jobs return None: the committed paired-gain spectra are unrecoverable
    (paired_ns_gain_check.py was edited after it wrote paired_gain_check.jsonl; the
    current script does not reproduce the committed counts). Only the block jobs
    (B1/clean, reconstructed byte-exactly from frozen arch.json) attach."""
    return job.get("committed_logz")


def reconstruct(job: dict):
    k = job["kind"]
    if k in ("gain_clean", "gain_gain"):
        dc, dg, mcf, pcfg, pnames = HC.reconstruct_gain_pair(job["pair_i"])
        counts = dc if k == "gain_clean" else dg
        return counts, mcf, pcfg, pnames
    counts, mcf, pcfg, pnames = HC.reconstruct_block(
        job["family"], job["level"], job["block_i"], job["strength"])
    return counts, mcf, pcfg, pnames


def run_job(job: dict) -> dict:
    """Run ONE NS job to completion, persist DONE.json, return a status dict."""
    run_id = job["job_id"]
    run_dir = RUNS / run_id
    marker = run_dir / "DONE.json"
    logpath = LOGS / f"{run_id}.log"

    def log(msg):
        with open(logpath, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    if marker.exists():
        return {"status": "skip", "job_id": run_id}

    # wipe any partial dir from a killed run, then let UltraNest create fresh
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        log(f"START kind={job['kind']} ns_seed={job['ns_seed']}")
        counts, mcf, pcfg, pnames = reconstruct(job)
        log(f"reconstructed spectrum counts_sum={int(counts.sum())}")

        from sbixcal import ns_bench as NB
        from ultranest import ReactiveNestedSampler

        np.random.seed(int(job["ns_seed"]))
        loglike = NB.make_poisson_loglike(counts, mcf)
        transform = NB.make_box_transform(pcfg, pnames)

        t0 = time.perf_counter()
        sampler = ReactiveNestedSampler(
            list(pnames), loglike, transform,
            log_dir=str(run_dir), resume="overwrite", vectorized=True)
        res = sampler.run(min_num_live_points=HC.NS_KW["min_num_live_points"],
                          dlogz=HC.NS_KW["dlogz"], max_ncalls=HC.NS_KW["max_ncalls"],
                          show_status=False, viz_callback=False)
        wall = time.perf_counter() - t0
        log(f"NS done wall={wall:.0f}s logz={res['logz']:.4f} logzerr={res['logzerr']:.4f} "
            f"ncall={res['ncall']} niter={res['niter']}")

        # ---- Higson thread bootstrap from the reconstructed birth-death tree ----
        # Note: points.hdf5 column 0 (`Lmin`) must not be read as the
        # per-point birth contour. That column is the contour at which the whole
        # refill BATCH was drawn, not the contour at which the point entered the
        # live set, and reading it that way manufactured thousands of phantom
        # single-point threads. HC.reconstruct_tree replays the batch consumption
        # against chains/run.txt and recovers the true 400-thread structure.
        tree = HC.reconstruct_tree(run_dir)
        boot = HC.higson_bootstrap(tree["logl"], tree["birth"], n_resamples=N_RESAMPLES,
                                   seed=BOOT_SEED, labels=tree["thread_id"])
        log(f"HIGSON sigma={boot['higson_sigma']:.4f} n_threads={boot['n_threads']} "
            f"reconstructed_logZ={boot['logZ_reconstructed']:.4f} "
            f"nlive_exact={tree['nlive_exact']}")

        committed = committed_for(job)
        out = {
            "job_id": run_id,
            "kind": job["kind"],
            "attaches_to": job.get("attaches_to"),
            "pair_i": job.get("pair_i"),
            "family": job.get("family"),
            "level": job.get("level"),
            "block_i": job.get("block_i"),
            "ns_seed": int(job["ns_seed"]),
            "counts": int(counts.sum()),
            "n_channels": int(counts.size),
            # UltraNest reported evidence + its native error components
            "ns_logz": float(res["logz"]),
            "ns_logzerr": float(res["logzerr"]),
            "ns_logzerr_bs": float(res.get("logzerr_bs", np.nan)),
            "ns_logzerr_tail": float(res.get("logzerr_tail", np.nan)),
            "ns_logzerr_single": float(res.get("logzerr_single", np.nan)),
            "ns_ncall": int(res["ncall"]),
            "ns_niter": int(res["niter"]),
            "wall_s": round(wall, 1),
            # Higson thread bootstrap
            "higson_sigma": boot["higson_sigma"],
            "higson_logZ_reconstructed": boot["logZ_reconstructed"],
            "higson_logZ_mean": boot["higson_logZ_mean"],
            "higson_logZ_p16_p84": boot["higson_logZ_p16_p84"],
            "n_threads": boot["n_threads"],
            "n_start_groups": boot["n_start_groups"],
            "n_points": boot["n_points"],
            "n_resamples": boot["n_resamples"],
            # tree-reconstruction validation
            "tree_fix": "2026-08-14 replay of UltraNest batch consumption",
            "nlive_exact_vs_runtxt": tree["nlive_exact"],
            "nlive_n_mismatch": tree["nlive_n_mismatch"],
            "n_store_unused": tree["n_store_unused"],
            # reproduction deltas (magnitude is the goal, not exact reproduction)
            "committed_logz": committed,
            "delta_reported_vs_committed": (float(res["logz"]) - committed
                                            if committed is not None else None),
            "delta_reconstructed_vs_reported": boot["logZ_reconstructed"] - float(res["logz"]),
        }
        with open(marker, "w") as f:
            json.dump(out, f, indent=2)
        log("MARKER written")
        return {"status": "done", "job_id": run_id, "higson_sigma": boot["higson_sigma"],
                "ns_logzerr": float(res["logzerr"]), "wall_s": round(wall, 1)}

    except Exception as exc:  # per-job fault isolation
        tb = traceback.format_exc()
        log(f"ERROR {type(exc).__name__}: {exc}\n{tb}")
        err = {"job_id": run_id, "status": "error",
               "error": f"{type(exc).__name__}: {exc}", "traceback": tb}
        with open(run_dir / "ERROR.json", "w") as f:
            json.dump(err, f, indent=2)
        return err


def _worker_init():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore")


def write_manifest(jobs):
    HGDIR.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    man = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "thread-wise bootstrap error floor under the paper's NS "
                   "evidence numbers. Each job reuses the EXACT committed spectrum/seed/"
                   "config; only change is an isolated per-run log_dir.",
        "ns_settings": HC.NS_KW,
        "n_resamples": N_RESAMPLES,
        "boot_seed": BOOT_SEED,
        "n_jobs": len(jobs),
        "jobs": {},
    }
    for j in jobs:
        man["jobs"][j["job_id"]] = {
            "kind": j["kind"],
            "attaches_to": j.get("attaches_to"),
            "pair_i": j.get("pair_i"),
            "family": j.get("family"),
            "level": j.get("level"),
            "block_i": j.get("block_i"),
            "strength": j.get("strength"),
            "ns_seed": j["ns_seed"],
            "committed_logz": committed_for(j),
        }
    with open(HGDIR / "MANIFEST.json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"[manifest] wrote {HGDIR / 'MANIFEST.json'} ({len(jobs)} jobs)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--job", default=None, help="run a single job_id (serial)")
    args = ap.parse_args(argv)

    jobs = HC.build_jobs()
    write_manifest(jobs)

    if args.job:
        jobs = [j for j in jobs if j["job_id"] == args.job]
        if not jobs:
            print(f"[err] no job {args.job}", flush=True)
            return 1

    pending = [j for j in jobs if not (RUNS / j["job_id"] / "DONE.json").exists()]
    print(f"[batch] {len(jobs)} jobs, {len(pending)} pending "
          f"({len(jobs)-len(pending)} already done)", flush=True)
    if not pending:
        print("[batch] all jobs already have DONE.json; nothing to run.", flush=True)
        return 0

    t_all = time.perf_counter()
    if args.job or args.workers <= 1:
        _worker_init()
        for j in pending:
            r = run_job(j)
            print(f"[{r['status']}] {r['job_id']} "
                  + (f"sigma={r.get('higson_sigma'):.4f} logzerr={r.get('ns_logzerr'):.4f} "
                     f"wall={r.get('wall_s')}s" if r["status"] == "done" else r.get("error", "")),
                  flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        nproc = min(args.workers, len(pending))
        print(f"[parallel] {len(pending)} jobs across {nproc} workers", flush=True)
        with ctx.Pool(processes=nproc, initializer=_worker_init, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(run_job, pending):
                print(f"[{r['status']}] {r['job_id']} "
                      + (f"sigma={r.get('higson_sigma'):.4f} logzerr={r.get('ns_logzerr'):.4f} "
                         f"wall={r.get('wall_s')}s" if r["status"] == "done" else r.get("error", "")),
                      flush=True)

    dt = time.perf_counter() - t_all
    done = sum(1 for j in jobs if (RUNS / j["job_id"] / "DONE.json").exists())
    print(f"\n[batch] finished in {dt:.0f}s: {done}/{len(jobs)} jobs have DONE.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
