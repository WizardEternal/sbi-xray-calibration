"""Gain-marginalized nested-sampling cross-check (reference posterior).

Runs UltraNest on the 6-parameter model (5 physical + a detector-gain nuisance g)
using the EXACT Poisson likelihood folded through the g-shifted EPIC-pn response,
so the g-marginalized NS posterior p(theta_phys | x) = int p(theta_phys, g | x) dg
can be compared against the gain-marginalized NPE flow
(outputs/gain_marg/model_medium), which does the same nuisance integral amortized.

Conventions match outputs/gain_marg/gen_and_train_gainmarg.py EXACTLY:
  - base model tbabs_powerlaw_bb,
  - medium exposure 353.4 s (bright 3534.0 s),
  - physical priors uniform(nh, alpha, kT) + loguniform(pl_norm, bb_norm),
  - gain prior uniform g in [0.95, 1.05],
  - gain applied to the response via responses.gain_shift_obsconf (same semantics
    as gen_and_train and misspec B4),
  - g discretized on N_GAIN_BINS=200 fine bins for batched folding.

Likelihood reuse: the log-likelihood is calibrate.poisson_loglik(counts, lambda(theta))
with lambda folded through gain_shift_obsconf, the same functions the training-data
generator and the IS refinement use. No duplicated physics.

------------------------------------------------------------------------------
PIVOTAL COST FINDING (measured probe2.py, 2026-07-23, OMP=1, this machine)
------------------------------------------------------------------------------
fakeit_for_multiple_parameters costs a FIXED ~110 ms per call, independent of the
row-batch size (M=1 -> 120 ms, M=400 -> 110 ms), and this does NOT drop in steady
state (warming all 200 bins: 114.8 ms/bin). The ~110 ms is per-obsconf JAX
dispatch / integration overhead, not the jaxspec model build (fold_theta which
rebuilds the model is only ~15 ms slower). Consequence for gain-marginalized NS:
Distinct g values do not vectorize: each unique (binned) g inside a likelihood
block costs one ~110 ms fakeit call. The 200-bin gain cache (precomputed once)
caps the fakeit calls per block at 200 and lets a LARGE draw block (ndraw_min set
high so UltraNest hands the vectorized loglike thousands of points at once)
amortize that fixed overhead across many points. This is why the cache + a large
ndraw_min matter for feasibility, and why 6-param gain-marg NS is intrinsically
much costlier per posterior than the 5-param NS baseline (which folds an entire
block through one shared obsconf in a single ~110 ms call).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sbixcal import responses as R
from sbixcal import models as M
from sbixcal import calibrate as C
from jaxspec.data.util import fakeit_for_multiple_parameters

# ----------------------------------------------------------------------------
# config: must mirror gen_and_train_gainmarg.py
# ----------------------------------------------------------------------------
BASE_MODEL = "tbabs_powerlaw_bb"
RESP = "NGC7793_ULX4_PN"
EXPOSURE_MEDIUM = 353.4
EXPOSURE_BRIGHT = 3534.0
GAIN_LO, GAIN_HI = 0.95, 1.05
N_GAIN_BINS = 200                       # 0.1/199 = 5.03e-4 = 0.05% gain resolution

PHYS_PRIORS = {
    "tbabs_1_nh":          {"dist": "uniform",    "low": 0.15,   "high": 0.35},
    "powerlaw_1_alpha":    {"dist": "uniform",    "low": 1.0,    "high": 3.0},
    "powerlaw_1_norm":     {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":   {"dist": "uniform",    "low": 0.3,    "high": 3.0},
    "blackbodyrad_1_norm": {"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}
GAIN_PRIOR = {"gain_g": {"dist": "uniform", "low": GAIN_LO, "high": GAIN_HI}}
PHYS_ORDER = M.MODEL_PARAMS[BASE_MODEL]            # 5 physical
PARAM_ORDER = PHYS_ORDER + ["gain_g"]             # 6, gain last (matches training)
FULL_PRIORS = {**PHYS_PRIORS, **GAIN_PRIOR}

QUANTILES = (0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975)


# ----------------------------------------------------------------------------
# gain-fold cache: model built once, 200 gain-shifted obsconfs precomputed once
# ----------------------------------------------------------------------------
@dataclass
class GainFolder:
    """Persistent jaxspec model + 200 precomputed gain-shifted obsconfs.

    fold(theta6) snaps each row's g to the nearest of N_GAIN_BINS bin centers and
    issues ONE fakeit call per occupied bin (physical params of the rows in that
    bin folded through that bin's precomputed obsconf). Instrumented: tracks the
    number of loglike blocks, total points, and total fakeit calls so the report
    can quote the true per-eval cost and the calls-per-point amortization ratio.
    """
    exposure_s: float
    n_bins: int = N_GAIN_BINS
    pad: int = 64          # fixed fakeit batch size -> JAX compiles ONE shape (see fold)
    # instrumentation
    n_blocks: int = 0
    n_points: int = 0
    n_fakeit_calls: int = 0
    fold_time_s: float = 0.0
    build_time_s: float = field(default=0.0)
    warmup_time_s: float = field(default=0.0)

    def __post_init__(self):
        t0 = time.perf_counter()
        base = R.load_base_obsconf(RESP)
        self._oc = R.scale_exposure(base, self.exposure_s)
        self._model = M.build_model(BASE_MODEL)
        self.centers = np.linspace(GAIN_LO, GAIN_HI, self.n_bins)
        self._ocs = [R.gain_shift_obsconf(self._oc, float(c)) for c in self.centers]
        self.build_time_s = time.perf_counter() - t0

    def warmup(self):
        """Fold a PAD-row dummy through every bin once so all 200 obsconfs are
        JIT-compiled AT THE PAD SHAPE before the NS run. Because fold() always
        calls fakeit with exactly `pad` rows, JAX then reuses one compiled shape
        per obsconf and every steady-state call costs the flat ~110 ms
        (no mid-run recompiles from varying batch sizes, the optimization that matters).
        One-time ~20-25 s. Not counted in fold instrumentation."""
        t0 = time.perf_counter()
        dummy = np.tile(np.array([0.25, 2.0, 1e-3, 1.0, 0.1]), (self.pad, 1)).astype(np.float64)
        params = {p: dummy[:, j] for j, p in enumerate(PHYS_ORDER)}
        for oc in self._ocs:
            fakeit_for_multiple_parameters(oc, self._model, params, rng_key=0, apply_stat=False)
        self.warmup_time_s = time.perf_counter() - t0
        return self.warmup_time_s

    def _fold_padded(self, phys_rows: np.ndarray, obsconf, out, dest_idx):
        """Fold phys_rows (k,5) through obsconf in FIXED-`pad`-row chunks so every
        fakeit call has the same batch shape (no JAX recompile). Writes results
        into out[dest_idx]. Cost is per-call (~110 ms) and flat in batch size, so
        padding short chunks up to `pad` is free."""
        k = phys_rows.shape[0]
        pad = self.pad
        for start in range(0, k, pad):
            chunk = phys_rows[start:start + pad]
            nc = chunk.shape[0]
            if nc < pad:  # pad up to the fixed shape with a repeat of row 0
                chunk = np.vstack([chunk, np.repeat(chunk[:1], pad - nc, axis=0)])
            params = {p: chunk[:, j] for j, p in enumerate(PHYS_ORDER)}
            lam = np.asarray(fakeit_for_multiple_parameters(
                obsconf, self._model, params, rng_key=0, apply_stat=False), dtype=np.float64)
            self.n_fakeit_calls += 1
            out[dest_idx[start:start + nc]] = lam[:nc]

    def fold(self, theta6: np.ndarray) -> np.ndarray:
        """theta6: (M, 6) in PARAM_ORDER. Returns (M, n_channels) lambda (float64).
        Snaps g to the nearest of n_bins precomputed obsconfs; folds each bin's
        rows in fixed-`pad`-row fakeit calls."""
        t0 = time.perf_counter()
        theta6 = np.atleast_2d(np.asarray(theta6, dtype=np.float64))
        phys = theta6[:, :5]
        g = theta6[:, 5]
        idx = np.abs(g[:, None] - self.centers[None, :]).argmin(axis=1)
        out = np.zeros((theta6.shape[0], 102), dtype=np.float64)
        for b in np.unique(idx):
            rows = np.where(idx == b)[0]
            self._fold_padded(phys[rows], self._ocs[int(b)], out, rows)
        self.n_blocks += 1
        self.n_points += theta6.shape[0]
        self.fold_time_s += time.perf_counter() - t0
        return out

    def stats(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "build_time_s": self.build_time_s,
            "warmup_time_s": self.warmup_time_s,
            "n_loglike_blocks": self.n_blocks,
            "n_points": self.n_points,
            "n_fakeit_calls": self.n_fakeit_calls,
            "fold_time_s": self.fold_time_s,
            "fakeit_calls_per_point": (self.n_fakeit_calls / self.n_points
                                       if self.n_points else float("nan")),
            "ms_per_point": (1e3 * self.fold_time_s / self.n_points
                             if self.n_points else float("nan")),
            "ms_per_fakeit_call": (1e3 * self.fold_time_s / self.n_fakeit_calls
                                   if self.n_fakeit_calls else float("nan")),
        }


# ----------------------------------------------------------------------------
# likelihood + prior transform (dist-aware: matches the TRAINING prior exactly)
# ----------------------------------------------------------------------------
def make_loglike(counts: np.ndarray, folder: GainFolder):
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)

    def loglike(theta):
        theta2 = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        lam = folder.fold(theta2)
        ll = C.poisson_loglik(counts, lam)
        ll = np.where(np.isfinite(ll), ll, -1e30)
        return ll if np.ndim(theta) > 1 else float(ll[0])

    return loglike


def make_transform(priors: dict = FULL_PRIORS, param_order=PARAM_ORDER):
    """Unit-cube -> parameter transform respecting each param's prior dist.

    uniform  -> linear in [low, high];
    loguniform -> log-uniform in [low, high] (i.e. 10**(loguniform log-bounds)).
    This MATCHES the distribution the gain-marg flow was TRAINED on
    (pl_norm/bb_norm drawn loguniform, all else uniform), so the NS reference
    targets the same prior the flow assumes, which is the fair reproduction test.
    (Note: the 5-param scripts/ns_bench uses a purely-linear box transform; at
    medium/bright counts the norm posterior is data-dominated so the two agree to
    ~0.04 normalized-quantile, but here we match the training prior exactly.)
    """
    dists, lo, hi = [], [], []
    for p in param_order:
        spec = priors[p]
        dists.append(spec["dist"])
        lo.append(float(spec["low"]))
        hi.append(float(spec["high"]))
    dists = np.array(dists, dtype=object)
    lo = np.asarray(lo); hi = np.asarray(hi)
    is_log = np.array([d == "loguniform" for d in dists])
    loglo = np.where(is_log, np.log10(np.where(is_log, lo, 1.0)), 0.0)
    loghi = np.where(is_log, np.log10(np.where(is_log, hi, 1.0)), 0.0)

    def transform(u):
        u2 = np.atleast_2d(np.asarray(u, dtype=np.float64))
        out = np.empty_like(u2)
        # linear params
        lin = ~is_log
        out[:, lin] = lo[lin][None, :] + u2[:, lin] * (hi[lin] - lo[lin])[None, :]
        # log params
        if is_log.any():
            e = loglo[is_log][None, :] + u2[:, is_log] * (loghi[is_log] - loglo[is_log])[None, :]
            out[:, is_log] = 10.0 ** e
        return out if np.ndim(u) > 1 else out[0]

    return transform


# ----------------------------------------------------------------------------
# NS run on one spectrum
# ----------------------------------------------------------------------------
@dataclass
class NSGMResult:
    quantiles: dict
    means: dict
    stds: dict
    logz: float
    logzerr: float
    n_like_evals: int
    niter: int
    ess: float
    wall_s: float
    n_live: int
    samples: np.ndarray
    param_names: list
    folder_stats: dict


def run_ns_gainmarg(counts, exposure_s=EXPOSURE_MEDIUM, min_num_live_points=400,
                    max_ncalls=None, dlogz=0.5, seed=0, ndraw_min=2000,
                    ndraw_max=65536, show_status=False, folder=None, warmup=True):
    """Run UltraNest on one spectrum's exact 6-param (5 phys + g) Poisson posterior.

    ndraw_min set high (default 2000) forces UltraNest to request large vectorized
    likelihood blocks so the 200-bin gain cache amortizes the ~110 ms/fakeit-call
    fixed overhead. In-memory (log_dir=None), vectorized loglike.
    """
    from ultranest import ReactiveNestedSampler

    np.random.seed(seed)
    if folder is None:
        folder = GainFolder(exposure_s)
    if warmup and folder.warmup_time_s == 0.0:
        folder.warmup()
    loglike = make_loglike(counts, folder)
    transform = make_transform()

    sampler = ReactiveNestedSampler(
        list(PARAM_ORDER), loglike, transform,
        log_dir=None, vectorized=True,
        ndraw_min=int(ndraw_min), ndraw_max=int(ndraw_max),
    )
    t0 = time.perf_counter()
    res = sampler.run(
        min_num_live_points=int(min_num_live_points),
        max_ncalls=(int(max_ncalls) if max_ncalls else None),
        dlogz=float(dlogz), show_status=bool(show_status), viz_callback=False,
    )
    wall = time.perf_counter() - t0

    samples = np.asarray(res["samples"], dtype=np.float64)
    quantiles, means, stds = {}, {}, {}
    for j, name in enumerate(PARAM_ORDER):
        col = samples[:, j]
        quantiles[name] = {f"{q:g}": float(v)
                           for q, v in zip(QUANTILES, np.quantile(col, QUANTILES))}
        means[name] = float(col.mean())
        stds[name] = float(col.std(ddof=1))

    return NSGMResult(
        quantiles=quantiles, means=means, stds=stds,
        logz=float(res["logz"]), logzerr=float(res["logzerr"]),
        n_like_evals=int(res["ncall"]), niter=int(res["niter"]),
        ess=float(res.get("ess", np.nan)), wall_s=float(wall),
        n_live=int(min_num_live_points), samples=samples,
        param_names=list(PARAM_ORDER), folder_stats=folder.stats(),
    )


# ----------------------------------------------------------------------------
# NPE posterior sampling (the gain-marg flow) on the same spectrum
# ----------------------------------------------------------------------------
def sample_npe(model_dir, counts, n_samples=4000, seed=0, device="cpu",
               reject_outside_prior=False):
    """Sample the 6-param gain-marg NPE flow. reject_outside_prior=False mirrors
    eval_gainmarg.py's convention (raw flow samples)."""
    import torch
    from sbixcal import train_npe as tn
    post, info = tn.load_posterior(model_dir, device=device)
    assert list(info["param_names"]) == PARAM_ORDER, info["param_names"]
    x_t = torch.as_tensor(np.asarray(counts, dtype=np.float32), device=device)
    torch.manual_seed(seed)
    with torch.no_grad():
        s = post.sample((n_samples,), x=x_t, show_progress_bars=False,
                        reject_outside_prior=reject_outside_prior)
    return s.detach().cpu().numpy()


# ----------------------------------------------------------------------------
# NS-vs-NPE comparison (g marginalized out = look at all 6 marginals)
# ----------------------------------------------------------------------------
def _interval_overlap(a, b):
    """IoU of two 1-D intervals a=(lo,hi), b=(lo,hi)."""
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def compare_ns_npe(ns_samples, npe_samples, param_names=PARAM_ORDER, run_c2st=True):
    """Per-parameter NS-vs-NPE agreement on the marginalized posteriors.

    Reports, per parameter: mean difference in units of the NS posterior std,
    std ratio (NPE/NS), and 68% / 95% equal-tailed interval IoU. Plus a joint
    C2ST accuracy over all 6 params (0.5 = indistinguishable) if sbi imports.
    """
    ns = np.asarray(ns_samples, dtype=np.float64)
    npe = np.asarray(npe_samples, dtype=np.float64)
    per_param = {}
    for j, name in enumerate(param_names):
        a, b = ns[:, j], npe[:, j]
        ns_mean, ns_std = a.mean(), a.std(ddof=1)
        npe_mean, npe_std = b.mean(), b.std(ddof=1)
        a68 = np.quantile(a, [0.16, 0.84]); b68 = np.quantile(b, [0.16, 0.84])
        a95 = np.quantile(a, [0.025, 0.975]); b95 = np.quantile(b, [0.025, 0.975])
        per_param[name] = {
            "ns_mean": float(ns_mean), "npe_mean": float(npe_mean),
            "ns_std": float(ns_std), "npe_std": float(npe_std),
            "mean_diff_in_ns_std": float((npe_mean - ns_mean) / ns_std) if ns_std > 0 else float("nan"),
            "std_ratio_npe_over_ns": float(npe_std / ns_std) if ns_std > 0 else float("nan"),
            "overlap68_iou": _interval_overlap(a68, b68),
            "overlap95_iou": _interval_overlap(a95, b95),
        }
    out = {"per_param": per_param}
    mds = [abs(per_param[p]["mean_diff_in_ns_std"]) for p in param_names]
    out["max_abs_mean_diff_in_ns_std"] = float(np.nanmax(mds))
    out["mean_overlap68_iou"] = float(np.mean([per_param[p]["overlap68_iou"] for p in param_names]))

    if run_c2st:
        try:
            import torch
            from sbi.utils.metrics import c2st as _c2st
            n = min(ns.shape[0], npe.shape[0])
            rng = np.random.default_rng(0)
            ia = rng.choice(ns.shape[0], n, replace=False)
            ib = rng.choice(npe.shape[0], n, replace=False)
            acc = _c2st(torch.as_tensor(ns[ia], dtype=torch.float32),
                        torch.as_tensor(npe[ib], dtype=torch.float32))
            out["c2st_accuracy"] = float(np.asarray(acc).reshape(-1)[0])
            out["c2st_note"] = "0.5 = NS and NPE posteriors indistinguishable; ->1 = separable"
        except Exception as e:  # pragma: no cover
            out["c2st_error"] = repr(e)
    return out


if __name__ == "__main__":
    # trivial self-test: cache builds, folds, transform + loglike run on 4 points
    f = GainFolder(EXPOSURE_MEDIUM)
    tr = make_transform()
    u = np.random.default_rng(0).uniform(size=(4, 6))
    th = tr(u)
    print("transform sample row0:", th[0])
    lam = f.fold(th)
    print("fold ->", lam.shape, "stats", f.stats())
    counts = np.random.default_rng(1).poisson(np.clip(lam[0], 0, None)).astype(float)
    ll = make_loglike(counts, f)(th)
    print("loglike ->", ll)
