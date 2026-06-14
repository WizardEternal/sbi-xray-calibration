"""Phase-5 nested-sampling benchmark: UltraNest vs amortized NPE on the SAME
Poisson likelihood (the speed-vs-trust comparison).

Goal: for a config-driven subsample of spectra, run
UltraNest's ``ReactiveNestedSampler`` on the EXACT SAME Poisson likelihood the
Phase-3 IS-refinement uses (``calibrate.poisson_loglik`` of the observed counts
given the model counts ``fold_theta(theta)``), with the SAME box-uniform prior as
the flow, and compare its posterior to the amortized NPE posterior. Per spectrum
we record NS quantiles / logZ / n_like_evals / wall-clock and NPE quantiles /
sampling wall-clock.

Likelihood reuse, not duplication
---------------------------------
The log-likelihood is literally ``calibrate.poisson_loglik(counts, fold_theta(theta))``
and the prior is the box read from the checkpoint's ``arch.json`` (same
``priors.prior_bounds`` the flow's BoxUniform uses). UltraNest gets a *vectorized*
loglike (it passes an ``(M, n_params)`` block; ``fold_theta`` + ``poisson_loglik``
are already vectorized over the leading axis) and a unit-cube → box transform.
The ``test_ns_bench`` suite asserts NS and the IS-likelihood agree on ``logL`` at
sampled θ to machine precision (the exact-reuse check).

In-memory, Windows-safe
-----------------------
``log_dir=None`` keeps the whole run in memory, no HDF5 point-store, so the
UltraNest/h5py Windows file-locking quirk cannot bite.

Append-resumable JSONL
----------------------
``outputs/ns_bench/results.jsonl``, one row per spectrum keyed by a stable
``spectrum_id``. :func:`load_done_ids` returns the ids already present;
:func:`run_subsample` skips them. Safe to kill and rerun.

Pure / importable: the CLI lives in ``scripts/run_ns_benchmark.py`` and the
analysis in ``scripts/analyze_ns_bench.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import calibrate as _cal
from . import priors as _priors


# ==========================================================================
# the reused Poisson likelihood + box prior (NO duplication)
# ==========================================================================

def make_poisson_loglike(counts: np.ndarray, model_counts_fn: Callable):
    r"""Return a *vectorized* UltraNest log-likelihood for one observed spectrum.

    ``model_counts_fn(theta) -> (M, n_channels)`` folds an ``(M, n_params)`` block
    of parameter vectors through the SAME response used everywhere else
    (``simulate.fold_theta``); the per-θ log-likelihood is exactly
    ``calibrate.poisson_loglik(counts, model_counts)``, the identical function the
    Phase-3 IS-refinement calls. UltraNest passes a 2-D block when
    ``vectorized=True``; we also accept a single 1-D vector (for the test's
    exact-reuse check).
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)

    def loglike(theta: np.ndarray) -> np.ndarray:
        theta2 = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        lam = np.asarray(model_counts_fn(theta2), dtype=np.float64)
        ll = _cal.poisson_loglik(counts, lam)
        # UltraNest requires finite log-likelihoods; floor -inf to a large negative.
        ll = np.where(np.isfinite(ll), ll, -1e30)
        return ll if np.ndim(theta) > 1 else float(ll[0])

    return loglike


def make_box_transform(prior_cfg: dict, param_names: list[str]):
    """Return a unit-cube → box transform for the SAME box-uniform prior the flow
    uses (linear bounds from ``priors.prior_bounds``). Vectorized over rows."""
    low, high = _priors.prior_bounds(prior_cfg, param_names)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    span = high - low

    def transform(u: np.ndarray) -> np.ndarray:
        u2 = np.atleast_2d(np.asarray(u, dtype=np.float64))
        out = low[None, :] + u2 * span[None, :]
        return out if np.ndim(u) > 1 else out[0]

    return transform


# ==========================================================================
# NS run on one spectrum
# ==========================================================================

# the quantile levels recorded per parameter (median + the 50/68/90 brackets)
QUANTILES = (0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95)


@dataclass
class NSResult:
    quantiles: dict          # {param_name: {q_str: value}}
    logz: float
    logzerr: float
    n_like_evals: int
    niter: int
    ess: float
    wall_s: float
    n_live: int
    samples: np.ndarray      # (n_eq, n_params) equal-weight posterior samples
    param_names: list


def run_ns_one(
    counts: np.ndarray,
    model_counts_fn: Callable,
    prior_cfg: dict,
    param_names: list[str],
    min_num_live_points: int = 400,
    max_ncalls: int | None = None,
    dlogz: float = 0.5,
    seed: int = 0,
    show_status: bool = False,
) -> NSResult:
    """Run UltraNest on one observed spectrum's exact Poisson posterior.

    In-memory (``log_dir=None``), vectorized likelihood. Returns an
    :class:`NSResult` with per-parameter quantiles, logZ (+ error), n_like_evals,
    ESS, and wall-clock. ``min_num_live_points`` / ``max_ncalls`` / ``dlogz`` let
    tests run a cheap, reduced-live-point version.
    """
    from ultranest import ReactiveNestedSampler

    np.random.seed(seed)
    loglike = make_poisson_loglike(counts, model_counts_fn)
    transform = make_box_transform(prior_cfg, param_names)

    sampler = ReactiveNestedSampler(
        list(param_names), loglike, transform,
        log_dir=None, vectorized=True,
    )
    t0 = time.perf_counter()
    res = sampler.run(
        min_num_live_points=int(min_num_live_points),
        max_ncalls=(int(max_ncalls) if max_ncalls else None),
        dlogz=float(dlogz),
        show_status=bool(show_status),
        viz_callback=False,
    )
    wall = time.perf_counter() - t0

    samples = np.asarray(res["samples"], dtype=np.float64)  # equal-weight
    quantiles = {}
    for j, name in enumerate(param_names):
        qs = np.quantile(samples[:, j], QUANTILES)
        quantiles[name] = {f"{q:g}": float(v) for q, v in zip(QUANTILES, qs)}

    return NSResult(
        quantiles=quantiles,
        logz=float(res["logz"]),
        logzerr=float(res["logzerr"]),
        n_like_evals=int(res["ncall"]),
        niter=int(res["niter"]),
        ess=float(res.get("ess", np.nan)),
        wall_s=float(wall),
        n_live=int(min_num_live_points),
        samples=samples,
        param_names=list(param_names),
    )


# ==========================================================================
# NPE quantiles + wall-clock on the SAME spectrum
# ==========================================================================

@dataclass
class NPEResult:
    quantiles: dict          # {param_name: {q_str: value}}
    sample_wall_s: float
    n_samples: int
    rejection_timeout: bool = False  # flow mass leaked outside the prior; see run_npe_one


def run_npe_one(
    posterior,
    counts: np.ndarray,
    param_names: list[str],
    n_samples: int = 2000,
    seed: int = 0,
    device: str = "cpu",
) -> NPEResult:
    """Sample the amortized NPE posterior for one spectrum and record the SAME
    quantiles as NS plus the sampling wall-clock (the ms/spectrum amortized cost).

    Rejection sampling against the prior box can stall indefinitely when a
    MISSPECIFIED spectrum pushes the flow's mass outside the prior (observed:
    ~1% acceptance on B4-bright). We bound it at 120 s; on timeout we fall back
    to raw flow samples (reject_outside_prior=False) and flag the row; the
    leak itself is a trust signal."""
    torch.manual_seed(seed)
    x_t = torch.as_tensor(np.asarray(counts, dtype=np.float32), device=device)
    rejection_timeout = False
    t0 = time.perf_counter()
    s = posterior.sample((n_samples,), x=x_t, show_progress_bars=False,
                         reject_outside_prior=True,
                         max_sampling_time=120.0, return_partial_on_timeout=True)
    if s.shape[0] < n_samples:
        rejection_timeout = True
        if s.shape[0] < max(200, n_samples // 10):
            s = posterior.sample((n_samples,), x=x_t, show_progress_bars=False,
                                 reject_outside_prior=False)
    wall = time.perf_counter() - t0
    s = s.detach().cpu().numpy()

    quantiles = {}
    for j, name in enumerate(param_names):
        qs = np.quantile(s[:, j], QUANTILES)
        quantiles[name] = {f"{q:g}": float(v) for q, v in zip(QUANTILES, qs)}
    return NPEResult(quantiles=quantiles, sample_wall_s=float(wall),
                     n_samples=int(n_samples))


# ==========================================================================
# quantile-agreement metric (NS vs NPE)
# ==========================================================================

def quantile_agreement(ns_q: dict, npe_q: dict, param_names, low, high):
    """Per-parameter |NS − NPE| quantile difference, normalized by prior width.

    Returns a dict with per-parameter median/5–95-bracket agreement and a single
    scalar ``mean_abs_norm`` = the mean over parameters and over the recorded
    quantiles of |q_NS − q_NPE| / (prior_high − prior_low). Small = posteriors
    agree (the NS-validates-NPE result, when raw NPE is well calibrated)."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    span = high - low
    per_param = {}
    all_norm = []
    for j, name in enumerate(param_names):
        diffs = {}
        for qk in ns_q[name]:
            d = abs(ns_q[name][qk] - npe_q[name][qk])
            dn = d / span[j] if span[j] > 0 else d
            diffs[qk] = dn
            all_norm.append(dn)
        per_param[name] = {
            "median_abs_norm": diffs.get("0.5", np.nan),
            "mean_abs_norm": float(np.mean(list(diffs.values()))),
        }
    return {
        "per_param": per_param,
        "mean_abs_norm": float(np.mean(all_norm)) if all_norm else np.nan,
    }


# ==========================================================================
# JSONL resume helpers
# ==========================================================================

def load_done_ids(results_path: Path) -> set[str]:
    """Set of ``spectrum_id`` already present in results.jsonl (resume-skip)."""
    done: set[str] = set()
    results_path = Path(results_path)
    if not results_path.exists():
        return done
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("spectrum_id")
            if sid is not None:
                done.add(str(sid))
    return done


def append_jsonl(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
