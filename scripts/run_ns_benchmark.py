r"""Phase-5 nested-sampling benchmark runner: UltraNest vs amortized NPE.

Usage (repo venv; cap compute with OMP_NUM_THREADS):
    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe scripts\run_ns_benchmark.py --config configs\ns_bench.yaml
    .venv\Scripts\python.exe scripts\run_ns_benchmark.py --config configs\ns_bench.yaml --pilot 2
    .venv\Scripts\python.exe scripts\run_ns_benchmark.py --config configs\ns_bench.yaml --max-ncalls 40000
    .venv\Scripts\python.exe scripts\run_ns_benchmark.py --config configs\ns_bench.yaml --workers 10

For each spectrum in the config subsample (clean Model-A at 3 count levels + a few
B1/B4 misspecified) this runs UltraNest's ReactiveNestedSampler on the SAME Poisson
likelihood the Phase-3 IS-refinement uses (calibrate.poisson_loglik of the observed
counts vs simulate.fold_theta(theta)) and the SAME box prior, plus the amortized
NPE posterior, and appends one row per spectrum to outputs/ns_bench/results.jsonl
keyed by spectrum_id (resume-skips ids already present).

CRITICAL modelling note (B4): a B4 spectrum is GENERATED with a gain-shifted
response, but BOTH inferences (NPE and NS) use the NOMINAL (clean) response and the
well-specified Model A -- that mismatch is exactly the misspecification. So the
likelihood's model_counts_fn always folds through the clean exposure-scaled obsconf.

--pilot N keeps only the FIRST N spectra of EACH block (the 6-spectrum pilot uses
--pilot 2 with the clean-only blocks selected via --clean-only).

Parallelism (--workers N, default 1 == exact serial behavior)
-------------------------------------------------------------
With --workers > 1 the per-spectrum NS+NPE work is fanned out ACROSS spectra with a
multiprocessing.Pool (Windows-safe spawn; the ``if __name__ == "__main__":`` guard
below is required). Each worker rebuilds and caches its own ``LevelNS`` per count
level (the NPE posterior and the model_counts_fn closure are not picklable, so we
pass only picklable args -- the checkpoint dir, response, counts/truth arrays -- and
reconstruct per process), and pins OMP/MKL to a single thread to avoid oversubscribing
the cores the cross-spectrum parallelism is already using.

The PARENT process is the ONLY writer to results.jsonl: workers return completed row
dicts via ``Pool.imap_unordered``; the parent appends + flushes each row as it lands.
Resume semantics are unchanged (ids already present at startup are skipped before any
work is dispatched). A per-spectrum exception in a worker does NOT abort the run: the
worker returns an ERROR ROW (keyed by spectrum_id, carrying the exception string) which
the parent writes and counts; analyze_ns_bench.py skips error rows (with a count).

Writes ONLY to outputs/ns_bench/. Reads checkpoints in outputs/models/. Never
touches outputs/calibration/ or outputs/detect/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

from sbixcal import ns_bench as NB
from sbixcal import train_npe as _tn
from sbixcal import priors as _priors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _out_dir() -> Path:
    d = _repo_root() / "outputs" / "ns_bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_for_level(train_run: str, level: str) -> Path:
    return _repo_root() / "outputs" / "models" / f"{train_run}_{level}"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# per-level state: cold-loaded flow + clean exposure-scaled obsconf + model_fn
# --------------------------------------------------------------------------

class LevelNS:
    """Everything reused across all spectra at one count level: the cold-loaded
    NPE posterior, the prior box, and the model_counts_fn that folds theta through
    the NOMINAL (clean) exposure-scaled response (the inference model)."""

    def __init__(self, ckpt_dir: Path, response: str | None, device: str = "cpu"):
        from sbixcal import responses as _responses
        from sbixcal import simulate as _sim

        self.posterior, self.info = _tn.load_posterior(ckpt_dir, device=device)
        with open(ckpt_dir / "arch.json") as f:
            arch = json.load(f)
        self.prior_cfg = arch["prior_cfg"]
        self.base_model = arch["base_model"]
        self.exposure_s = float(arch["exposure_s"])
        self.param_names = list(arch["param_names"])
        self.median_counts = float(arch.get("median_total_counts", np.nan))
        self.device = device

        resp = response or _responses.EXAMPLE_NAME
        base = _responses.load_base_obsconf(resp)
        self.obsconf = _responses.scale_exposure(base, self.exposure_s)  # NOMINAL response
        self.low, self.high = _priors.prior_bounds(self.prior_cfg, self.param_names)

        def model_counts_fn(theta_arr):
            return _sim.fold_theta(self.base_model, self.param_names, theta_arr,
                                   self.obsconf)
        self.model_counts_fn = model_counts_fn


# --------------------------------------------------------------------------
# drawing the subsample blocks (deterministic, reproducible on resume)
# --------------------------------------------------------------------------

def draw_block(block: dict, state: LevelNS, block_idx: int, seed: int):
    """Return ``(x (n,C) Poisson counts, theta_truth (n,P) or None, ids list)``
    for one subsample block. Clean blocks use simulate.simulate_spectra; B1/B4 use
    misspec.simulate_misspec_population (B4 gain-shift on a COPY of the obsconf, the
    inference still uses the nominal one in ``state``)."""
    from sbixcal import simulate as _sim
    from sbixcal import misspec as _MS
    from sbixcal import responses as _responses

    family = block["family"]
    level = block["level"]
    n = int(block["n"])
    # deterministic per-block seed so resume reproduces the SAME spectra
    block_seed = (int(seed) + 1000 * (block_idx + 1)) % (2**31 - 1)

    if family == "clean":
        rng = np.random.default_rng(block_seed)
        theta, x_exp, names = _sim.simulate_spectra(
            state.base_model, state.prior_cfg, state.obsconf, n, rng,
            apply_poisson=False, seed_for_fakeit=block_seed)
        rng_p = np.random.default_rng(block_seed + 1)
        x = rng_p.poisson(np.clip(x_exp, 0.0, None)).astype(np.float64)
        theta_truth = np.asarray(theta, dtype=np.float64)
        slabel = "clean"
    else:
        strength = float(block["strength"])
        x, theta_truth, present = _MS.simulate_misspec_population(
            state.base_model, state.prior_cfg, state.obsconf,
            family, strength, n, seed=block_seed, fixed=block.get("fixed", {}))
        x = np.asarray(x, dtype=np.float64)
        # B1/B4 base-Model-A truth theta is `present`-ordered; only complete if all
        # base params survive (true for B1 and B4; B2/B3 drop params -> None).
        if theta_truth.shape[1] != len(state.param_names):
            theta_truth = None
        slabel = f"{family}_s{strength:g}"

    ids = [f"{family}|{level}|{slabel}|{block_idx}|{i}" for i in range(n)]
    return x, theta_truth, ids, slabel


# --------------------------------------------------------------------------
# building the per-spectrum task list (parent side; picklable payloads only)
# --------------------------------------------------------------------------

def build_tasks(cfg: dict, pilot: int | None, clean_only: bool,
                max_ncalls_override: int | None, device: str, done: set):
    """Draw every subsample block in the parent and return a flat list of picklable
    per-spectrum task dicts (those not already in ``done``), plus the set of count
    levels each task references (so workers know which checkpoints to load).

    Each task carries ONLY picklable data: the spectrum_id, the observed counts
    (list), truth (list or None), and the metadata needed to rebuild the row. The
    heavy ``LevelNS`` (NPE posterior + model_counts_fn) is reconstructed per process
    in the worker, keyed by ``level``."""
    seed = int(cfg["seed"])
    train_run = cfg["train_run"]
    response = cfg.get("response")
    ns_cfg = cfg.get("ns", {})
    npe_cfg = cfg.get("npe", {})
    min_live = int(ns_cfg.get("min_num_live_points", 400))
    dlogz = float(ns_cfg.get("dlogz", 0.5))
    max_ncalls = max_ncalls_override if max_ncalls_override is not None \
        else ns_cfg.get("max_ncalls")
    n_npe = int(npe_cfg.get("n_samples", 2000))

    blocks = list(cfg["subsample"])
    if clean_only:
        blocks = [b for b in blocks if b["family"] == "clean"]

    tasks: list[dict] = []
    states: dict[str, LevelNS] = {}
    for bidx, block in enumerate(blocks):
        level = block["level"]
        if level not in states:
            ckpt = _checkpoint_for_level(train_run, level)
            if not (ckpt / "arch.json").exists():
                print(f"[skip-level] {level}: no checkpoint at {ckpt}")
                states[level] = None  # sentinel: skip this level's blocks
                continue
            states[level] = LevelNS(ckpt, response, device=device)
            print(f"[level {level}] flow loaded "
                  f"(~{states[level].median_counts:.0f} counts)")
        state = states[level]
        if state is None:
            continue

        x, theta_truth, ids, slabel = draw_block(block, state, bidx, seed)
        take = int(pilot) if pilot else len(ids)

        for i in range(min(take, len(ids))):
            sid = ids[i]
            if sid in done:
                print(f"[skip] {sid}")
                continue
            truth = (theta_truth[i].tolist() if theta_truth is not None else None)
            tasks.append({
                "spectrum_id": sid,
                "family": block["family"],
                "level": level,
                "strength_label": slabel,
                "counts": x[i].tolist(),
                "truth": truth,
                "ckpt_dir": str(_checkpoint_for_level(train_run, level)),
                "response": response,
                "device": device,
                "min_live": min_live,
                "dlogz": dlogz,
                "max_ncalls": max_ncalls,
                "n_npe": n_npe,
                "seed": seed + i,
            })
    return tasks


# --------------------------------------------------------------------------
# the worker: rebuilds LevelNS per process (cached by level) and runs one spectrum
# --------------------------------------------------------------------------

# per-process LevelNS cache (one entry per count level seen in this worker)
_WORKER_STATES: dict[str, LevelNS] = {}


def _worker_init():
    """Pin BLAS/OpenMP to a single thread in each worker: the parallelism is ACROSS
    spectra, so per-spectrum thread pools would oversubscribe the cores."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    warnings.filterwarnings("ignore")


def _get_worker_state(task: dict) -> LevelNS:
    """Reconstruct (and cache) the LevelNS for this task's level inside this process."""
    level = task["level"]
    st = _WORKER_STATES.get(level)
    if st is None:
        st = LevelNS(Path(task["ckpt_dir"]), task["response"], device=task["device"])
        _WORKER_STATES[level] = st
    return st


def _row_from_results(task: dict, state: LevelNS, ns, npe, agree) -> dict:
    return {
        "spectrum_id": task["spectrum_id"],
        "family": task["family"], "level": task["level"],
        "strength_label": task["strength_label"],
        "n_counts": int(round(float(np.sum(task["counts"])))),
        "param_names": state.param_names,
        "truth": task["truth"],
        "ns": {
            "quantiles": ns.quantiles, "logz": ns.logz, "logzerr": ns.logzerr,
            "n_like_evals": ns.n_like_evals, "niter": ns.niter,
            "ess": ns.ess, "wall_s": ns.wall_s, "n_live": ns.n_live,
        },
        "npe": {
            "quantiles": npe.quantiles, "sample_wall_s": npe.sample_wall_s,
            "n_samples": npe.n_samples,
        },
        "agreement": agree,
    }


def run_one_task(task: dict) -> dict:
    """Run NS + NPE for ONE spectrum and return a completed row dict.

    A per-spectrum failure is caught and returned as an ERROR ROW (keyed by
    spectrum_id, carrying the exception string) so one bad spectrum cannot abort the
    whole run; the parent writes it and the analysis skips it with a count. This is
    the single unit of work dispatched to the Pool (and also called directly in the
    serial path)."""
    try:
        state = _get_worker_state(task)
        counts = np.asarray(task["counts"], dtype=np.float64)
        ns = NB.run_ns_one(
            counts, state.model_counts_fn, state.prior_cfg, state.param_names,
            min_num_live_points=task["min_live"], max_ncalls=task["max_ncalls"],
            dlogz=task["dlogz"], seed=task["seed"], show_status=False)
        npe = NB.run_npe_one(
            state.posterior, counts, state.param_names,
            n_samples=task["n_npe"], seed=task["seed"], device=task["device"])
        agree = NB.quantile_agreement(
            ns.quantiles, npe.quantiles, state.param_names, state.low, state.high)
        return _row_from_results(task, state, ns, npe, agree)
    except Exception as exc:  # noqa: BLE001 -- per-spectrum fault isolation
        import traceback
        return {
            "spectrum_id": task["spectrum_id"],
            "family": task.get("family"), "level": task.get("level"),
            "strength_label": task.get("strength_label"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_subsample(cfg: dict, pilot: int | None = None, clean_only: bool = False,
                  max_ncalls_override: int | None = None, device: str = "cpu",
                  workers: int = 1):
    out = _out_dir()
    results_path = out / "results.jsonl"
    done = NB.load_done_ids(results_path)

    tasks = build_tasks(cfg, pilot, clean_only, max_ncalls_override, device, done)
    n_total = len(tasks)
    if n_total == 0:
        print("\nnothing to do (all requested spectra already present).")
        return 0

    workers = max(1, int(workers))
    t_all = time.perf_counter()
    n_run = 0
    n_err = 0

    def _handle(row: dict):
        nonlocal n_run, n_err
        NB.append_jsonl(results_path, row)
        done.add(row["spectrum_id"])
        n_run += 1
        if "error" in row:
            n_err += 1
            print(f"[error] {row['spectrum_id']}: {row['error']}")
        else:
            ns = row["ns"]
            print(f"[done] {row['spectrum_id']}: NS {ns['wall_s']:.1f}s "
                  f"({ns['n_like_evals']} evals, logZ={ns['logz']:.1f}) | "
                  f"NPE {row['npe']['sample_wall_s']*1e3:.0f} ms | "
                  f"q-agree {row['agreement']['mean_abs_norm']:.3f}")

    if workers == 1:
        # exact serial behavior (default): run inline, no Pool, parent writes per row.
        _worker_init()
        for task in tasks:
            _handle(run_one_task(task))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # Windows-safe; explicit for cross-platform parity
        nproc = min(workers, n_total)
        print(f"[parallel] {n_total} spectra across {nproc} workers "
              f"(parent is the only writer)")
        # maxtasksperchild: recycle workers every few spectra. Long ultranest runs
        # accumulate ~GB-scale state per process and can OOM the host otherwise.
        # 3 amortizes the per-spawn torch/jax init cost while keeping accumulation
        # bounded (the faint level, worst accumulator, is the slowest anyway).
        with ctx.Pool(processes=nproc, initializer=_worker_init,
                      maxtasksperchild=3) as pool:
            # imap_unordered: rows land as they finish; parent appends + flushes each.
            for row in pool.imap_unordered(run_one_task, tasks):
                _handle(row)

    dt = time.perf_counter() - t_all
    print(f"\nran {n_run} new spectra ({n_err} errored) in {dt:.1f}s -> {results_path}")
    return n_run


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Phase-5 NS-vs-NPE benchmark runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot", type=int, default=None,
                    help="keep only the first N spectra of each block (pilot)")
    ap.add_argument("--clean-only", action="store_true",
                    help="restrict to clean Model-A blocks (pilot spine)")
    ap.add_argument("--max-ncalls", type=int, default=None,
                    help="cap likelihood evals per NS run (overrides config)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallelize across spectra with N processes "
                         "(default 1 = serial; each worker pins OMP/MKL to 1 thread)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_subsample(cfg, pilot=args.pilot, clean_only=args.clean_only,
                  max_ncalls_override=args.max_ncalls, device=args.device,
                  workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
