"""Calibration-testing + recalibration suite for sbi-xray-calibration (Phase 3).

Operates on ANY trained checkpoint directory (one flow per count level; see
``train_npe.load_posterior``). Everything is config-driven, seeded, and writes
artifacts with skip-if-exists so figures are regenerable.

What's here
-----------
1. **SBC** (Talts et al. 2018, arXiv:1804.06788) -- via sbi's built-in
   ``sbi.diagnostics.run_sbc`` + ``check_sbc``. N fresh simulations are drawn
   from the SAME prior/simulator the flow was trained for (``simulate.py``), the
   rank statistic per parameter is computed by sbi, and we record the uniformity
   diagnostics sbi returns: KS p-values (``ks_pvals``) and the C2ST-vs-uniform
   accuracy (``c2st_ranks``). Rank histograms are drawn with
   ``sbi.analysis.sbc_rank_plot``.

2. **Expected coverage / TARP** (Lemos et al. 2023, arXiv:2302.03026) -- via
   ``sbi.diagnostics.run_tarp`` + ``check_tarp``. The expected-coverage-vs-nominal
   (ECP vs alpha) curve is saved as npz + figure, and we ALSO compute the simple
   per-parameter empirical coverage of equal-tailed credible intervals vs nominal
   (a direct, interpretable per-parameter coverage curve) for the before/after
   recalibration comparison.

3. **Recalibration, two methods behind one interface** (``recalibrate``):
   (a) **Importance-sampling refinement** with the known Poisson likelihood --
       the Barret & Dupourque Paper III move (A&A 708, A280, 2026,
       arXiv:2512.16709, Sec. 2-3). NPE posterior samples theta ~ q_NPE(.|x) are
       reweighted by

           w(theta) = p_Poisson(x | theta) * p(theta) / q_NPE(theta | x)

       where p_Poisson is the exact Poisson likelihood of the observed counts
       given the noiseless model counts (lambda) from ``simulate.py``, p(theta)
       is the (box-uniform) prior, and q_NPE is the flow's ``log_prob``. We
       compute the effective sample size (ESS) and FLAG low-ESS cases: a low ESS
       is Paper III's own diagnostic that the NPE proposal is a poor match to the
       true posterior (the correction cannot rescue a badly-placed proposal).
   (b) **Conformal / quantile recalibration** of the 1-D marginals (Lemos et al.
       2023 expected-coverage + split-conformal-prediction style; cf. Vovk et al.
       2005, Angelopoulos & Bates 2021). On a held-out calibration set we learn,
       per parameter, the empirical quantile level at which the marginal interval
       attains nominal coverage, then apply that adjustment to new posteriors.
       Deliberately simple and well-documented.

Before/after coverage comparison (npz + figure) is produced by
``coverage_before_after``.

Pure, importable functions only -- the CLI lives in
``scripts/run_calibration.py``. Nothing here touches ``outputs/models`` except to
*read* a checkpoint via ``train_npe.load_posterior``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from scipy.special import gammaln

from sbi.diagnostics import run_sbc, check_sbc, run_tarp, check_tarp
from sbi.analysis import sbc_rank_plot

from . import priors as _priors
from . import train_npe as _tn


# ==========================================================================
# fresh test-set generation (same prior + simulator the flow was trained for)
# ==========================================================================

def make_fresh_test_set(
    base_model: str,
    prior_cfg: dict,
    exposure_s: float,
    n: int,
    seed: int,
    response_name: str | None = None,
    simulate_fn: Callable | None = None,
):
    """Draw ``n`` (theta, x_expected, x_poisson) triples from the SAME prior and
    simulator the flow was trained for, at exposure ``exposure_s``.

    Returns ``(theta, x_poisson, x_expected, param_names)`` with theta and x as
    float32 numpy arrays. ``x_expected`` is the noiseless per-channel model counts
    (lambda) needed for the exact Poisson likelihood in IS-refinement;
    ``x_poisson`` is the realized integer counts (the "observed data").

    ``simulate_fn`` is injectable for tests (a toy simulator); by default the real
    jaxspec-backed ``simulate.simulate_spectra`` is used (imported lazily so this
    module imports without jaxspec for unit tests).
    """
    if simulate_fn is not None:
        return simulate_fn(base_model, prior_cfg, exposure_s, n, seed)

    # lazy import: keep calibrate.py importable without jaxspec (toy tests)
    from . import simulate as _sim
    from . import responses as _responses

    base = _responses.load_base_obsconf(response_name or _responses.EXAMPLE_NAME)
    obsconf = _responses.scale_exposure(base, exposure_s)

    rng = np.random.default_rng(seed)
    theta, x_exp, names = _sim.simulate_spectra(
        base_model, prior_cfg, obsconf, n, rng,
        apply_poisson=False, seed_for_fakeit=seed,
    )
    # realize Poisson counts from the SAME expected counts (independent rng)
    rng_p = np.random.default_rng(seed + 1)
    x_pois = rng_p.poisson(np.clip(x_exp, 0.0, None)).astype(np.float32)
    return (
        np.asarray(theta, dtype=np.float32),
        np.asarray(x_pois, dtype=np.float32),
        np.asarray(x_exp, dtype=np.float32),
        list(names),
    )


# ==========================================================================
# 1. SBC  (sbi.diagnostics.run_sbc / check_sbc; Talts et al. 2018)
# ==========================================================================

@dataclass
class SBCResult:
    ranks: np.ndarray                 # (N, n_params) int
    dap_samples: np.ndarray           # (N, n_params) data-averaged posterior draws
    ks_pvals: np.ndarray              # (n_params,)  KS-uniformity p-values
    c2st_ranks: np.ndarray            # (n_params,)  C2ST-vs-uniform accuracy (~0.5 good)
    c2st_dap: float                   # prior-vs-DAP C2ST (single value)
    num_posterior_samples: int
    param_names: list[str]
    uniformity_stat: str = "ks_pvals (KS test of rank uniformity) + c2st_ranks"


def run_sbc_check(
    posterior,
    prior,
    theta: torch.Tensor,
    x: torch.Tensor,
    param_names: list[str],
    num_posterior_samples: int = 1000,
    seed: int = 0,
    reduce_fns: str = "marginals",
) -> SBCResult:
    """Run sbi's SBC on a posterior over a fresh (theta, x) test set.

    ``reduce_fns="marginals"`` gives one rank per parameter (per-parameter SBC).
    The uniformity statistic recorded is exactly what ``check_sbc`` returns:
    KS p-values (large p => consistent with uniform => well-calibrated) and the
    C2ST accuracy between the ranks and a uniform baseline (~0.5 => good).
    """
    torch.manual_seed(seed)
    theta = torch.as_tensor(theta, dtype=torch.float32)
    x = torch.as_tensor(x, dtype=torch.float32)

    ranks, dap_samples = run_sbc(
        theta, x, posterior,
        num_posterior_samples=num_posterior_samples,
        reduce_fns=reduce_fns,
        show_progress_bar=False,
    )

    prior_samples = prior.sample((theta.shape[0],))
    checks = check_sbc(
        ranks, prior_samples, dap_samples,
        num_posterior_samples=num_posterior_samples,
    )
    return SBCResult(
        ranks=ranks.cpu().numpy(),
        dap_samples=dap_samples.cpu().numpy(),
        ks_pvals=checks["ks_pvals"].cpu().numpy(),
        c2st_ranks=checks["c2st_ranks"].cpu().numpy(),
        c2st_dap=float(checks["c2st_dap"].cpu().numpy().reshape(-1)[0]),
        num_posterior_samples=num_posterior_samples,
        param_names=list(param_names),
    )


def save_sbc_figure(sbc: SBCResult, out_path: Path, plot_type: str = "hist"):
    """Per-parameter SBC rank histograms via sbi's ``sbc_rank_plot``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = sbc_rank_plot(
        ranks=torch.as_tensor(sbc.ranks),
        num_posterior_samples=sbc.num_posterior_samples,
        plot_type=plot_type,
        parameter_labels=sbc.param_names,
    )
    fig.suptitle(
        "SBC rank histograms (flat = calibrated)\n"
        + "  ".join(f"{n}: KS p={p:.3f}" for n, p in
                    zip(sbc.param_names, sbc.ks_pvals)),
        fontsize=9,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ==========================================================================
# 2. Expected coverage / TARP  (sbi.diagnostics.run_tarp / check_tarp; Lemos+23)
# ==========================================================================

@dataclass
class TARPResult:
    ecp: np.ndarray                   # (num_bins,) expected coverage probability
    alpha: np.ndarray                 # (num_bins,) nominal credibility levels
    atc: float                        # area-to-curve (0 = perfect; >0 too wide)
    ks_pval: float                    # KS p-value of ECP-vs-uniform deviation


def run_tarp_check(
    posterior,
    theta: torch.Tensor,
    x: torch.Tensor,
    num_posterior_samples: int = 1000,
    seed: int = 0,
) -> TARPResult:
    """sbi's TARP expected-coverage test (joint over all parameters)."""
    torch.manual_seed(seed)
    theta = torch.as_tensor(theta, dtype=torch.float32)
    x = torch.as_tensor(x, dtype=torch.float32)
    ecp, alpha = run_tarp(
        theta, x, posterior,
        num_posterior_samples=num_posterior_samples,
        show_progress_bar=False,
    )
    atc, ks_pval = check_tarp(ecp, alpha)
    return TARPResult(
        ecp=ecp.cpu().numpy(),
        alpha=alpha.cpu().numpy(),
        atc=float(atc),
        ks_pval=float(ks_pval),
    )


def save_tarp_npz_and_figure(tarp: TARPResult, npz_path: Path, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_path = Path(npz_path)
    fig_path = Path(fig_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, ecp=tarp.ecp, alpha=tarp.alpha,
             atc=tarp.atc, ks_pval=tarp.ks_pval)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(tarp.alpha, tarp.ecp, "C0-", lw=2, label="TARP ECP")
    ax.set_xlabel("nominal credibility level")
    ax.set_ylabel("expected coverage probability")
    ax.set_title(f"TARP expected coverage (joint)\n"
                 f"ATC={tarp.atc:+.4f}  KS p={tarp.ks_pval:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return npz_path, fig_path


# ==========================================================================
# per-parameter empirical coverage of equal-tailed credible intervals
# (direct, interpretable; used for the before/after recalibration comparison)
# ==========================================================================

def empirical_coverage_curve(
    samples_per_obs: list[np.ndarray],
    truths: np.ndarray,
    nominal_levels: np.ndarray,
):
    """Per-parameter empirical coverage of equal-tailed credible intervals.

    ``samples_per_obs`` is a list (length N) of ``(n_samples, n_params)`` posterior
    sample arrays, one per test observation; ``truths`` is ``(N, n_params)``.
    For each nominal level c we form the equal-tailed central-c interval per
    parameter per observation and measure the fraction of observations whose truth
    falls inside.

    Returns ``coverage`` of shape ``(len(nominal_levels), n_params)``.
    """
    n_obs = len(samples_per_obs)
    n_params = truths.shape[1]
    cov = np.zeros((len(nominal_levels), n_params))
    for li, c in enumerate(nominal_levels):
        qlo = (1.0 - c) / 2.0 * 100.0
        qhi = (1.0 + c) / 2.0 * 100.0
        inside = np.zeros((n_obs, n_params), dtype=bool)
        for i, s in enumerate(samples_per_obs):
            lo = np.percentile(s, qlo, axis=0)
            hi = np.percentile(s, qhi, axis=0)
            inside[i] = (truths[i] >= lo) & (truths[i] <= hi)
        cov[li] = inside.mean(axis=0)
    return cov


# ==========================================================================
# 3a. Importance-sampling refinement with the exact Poisson likelihood
#     (Barret & Dupourque Paper III, arXiv:2512.16709, Sec. 2-3)
# ==========================================================================

def poisson_loglik(counts: np.ndarray, model_counts: np.ndarray) -> np.ndarray:
    r"""Exact Poisson log-likelihood, summed over channels.

    For observed integer counts ``x_c`` and noiseless model counts
    ``lambda_c`` (the expected counts from the simulator, ``apply_poisson=False``):

        log p(x | theta) = sum_c [ x_c * log(lambda_c) - lambda_c - log(x_c!) ]

    ``counts``      : (n_channels,) or (M, n_channels) observed counts
    ``model_counts``: (M, n_channels) model (lambda) counts, one row per theta.
    Returns (M,) log-likelihoods. ``lambda_c`` is floored at a tiny epsilon so
    empty model channels with zero observed counts contribute 0 and do not NaN.
    """
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    lam = np.clip(np.asarray(model_counts, dtype=np.float64), 1e-10, None)
    # broadcast counts (1, C) against lam (M, C)
    ll = counts * np.log(lam) - lam - gammaln(counts + 1.0)
    return ll.sum(axis=1)


@dataclass
class ISResult:
    samples: np.ndarray               # (n_samples, n_params) the NPE proposal draws
    log_w: np.ndarray                 # (n_samples,) unnormalized log importance weights
    weights: np.ndarray               # (n_samples,) normalized weights (sum to 1)
    ess: float                        # effective sample size
    ess_frac: float                   # ESS / n_samples
    low_ess: bool                     # flag: ESS fraction below threshold
    n_samples: int


def importance_refine(
    posterior,
    x_obs: np.ndarray,
    model_counts_fn: Callable[[np.ndarray], np.ndarray],
    prior_log_prob: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 2000,
    seed: int = 0,
    low_ess_frac: float = 0.1,
    device: str = "cpu",
) -> ISResult:
    r"""IS-refine one NPE posterior with the exact Poisson likelihood.

    Draw theta_j ~ q_NPE(. | x_obs), then weight by

        w_j = p_Poisson(x_obs | theta_j) * p(theta_j) / q_NPE(theta_j | x_obs)

    in log space (subtract the max before exponentiating for stability).
    ``model_counts_fn(theta)`` maps ``(M, n_params)`` -> ``(M, n_channels)`` noiseless
    model counts (lambda); ``prior_log_prob(theta)`` -> ``(M,)`` log prior density.

    **Low-ESS is Paper III's own NPE-failure diagnostic.** A small effective
    sample size means the NPE proposal q badly mismatches the true posterior, so
    the reweighting is dominated by a few samples and CANNOT be trusted to repair
    the posterior -- the correct response is to flag it (and, in production,
    fall back to sequential NPE or nested sampling), not to report the IS result
    as if it were reliable.
    """
    torch.manual_seed(seed)
    x_t = torch.as_tensor(np.asarray(x_obs, dtype=np.float32), device=device)

    samples_t = posterior.sample(
        (n_samples,), x=x_t, show_progress_bars=False, reject_outside_prior=True,
    )
    samples = samples_t.detach().cpu().numpy()

    # q_NPE(theta | x): the flow's normalized log density at the drawn samples
    with torch.no_grad():
        log_q = posterior.log_prob(
            samples_t, x=x_t, norm_posterior=True,
        ).detach().cpu().numpy()

    log_like = poisson_loglik(x_obs, model_counts_fn(samples))
    log_prior = np.asarray(prior_log_prob(samples), dtype=np.float64)

    log_w = log_like + log_prior - log_q
    log_w_finite = np.where(np.isfinite(log_w), log_w, -np.inf)
    m = np.max(log_w_finite)
    if not np.isfinite(m):
        # everything underflowed: degenerate, uniform fallback weights
        weights = np.full(n_samples, 1.0 / n_samples)
        ess = float(n_samples)
    else:
        w = np.exp(log_w_finite - m)
        s = w.sum()
        weights = w / s if s > 0 else np.full(n_samples, 1.0 / n_samples)
        # Kish effective sample size
        ess = float((weights.sum() ** 2) / np.sum(weights ** 2))

    ess_frac = ess / n_samples
    return ISResult(
        samples=samples,
        log_w=log_w,
        weights=weights,
        ess=ess,
        ess_frac=ess_frac,
        low_ess=bool(ess_frac < low_ess_frac),
        n_samples=n_samples,
    )


def weighted_quantile(values: np.ndarray, q, weights: np.ndarray):
    """Weighted quantile(s) of a 1-D array. ``q`` in [0,1] (scalar or array)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    # midpoint convention for the weighted CDF
    cw_mid = cw - 0.5 * w / w.sum()
    return np.interp(q, cw_mid, v)


def is_refined_quantiles(is_res: ISResult, q):
    """Per-parameter weighted quantiles of the IS-refined posterior.

    ``q`` scalar or array in [0,1]; returns ``(len(q), n_params)`` (or
    ``(n_params,)`` for scalar q)."""
    q_arr = np.atleast_1d(q)
    n_params = is_res.samples.shape[1]
    out = np.zeros((len(q_arr), n_params))
    for p in range(n_params):
        out[:, p] = weighted_quantile(is_res.samples[:, p], q_arr, is_res.weights)
    return out[0] if np.isscalar(q) else out


def is_coverage_curve(
    posterior,
    x_pois: np.ndarray,
    truths: np.ndarray,
    model_counts_fn: Callable[[np.ndarray], np.ndarray],
    prior_log_prob: Callable[[np.ndarray], np.ndarray],
    nominal_levels: np.ndarray,
    n_is_samples: int = 2000,
    seed: int = 0,
    low_ess_frac: float = 0.1,
    device: str = "cpu",
):
    r"""Per-parameter empirical coverage of the **IS-refined** posterior.

    The before/after partner for the IS-refinement (Paper III) recalibration. For
    each test observation ``x_pois[i]`` (with known truth ``truths[i]`` and the
    same response folded by ``model_counts_fn``) we IS-refine the NPE posterior
    with the exact Poisson likelihood, then form **weighted** equal-tailed central
    credible intervals at each nominal level and measure how often the truth falls
    inside. We return both the all-cases coverage and a coverage restricted to the
    cases that PASS the low-ESS gate (Paper III's own failure flag), so the
    write-up can state plainly where IS refinement works and where it fails for
    lack of effective sample size.

    Returns a dict with:
      ``cov_all``       (len(levels), n_params) IS coverage, all cases
      ``cov_okess``     (len(levels), n_params) IS coverage, low-ESS cases dropped
      ``ess``           (N,) Kish ESS per case
      ``ess_frac``      (N,)
      ``low_ess``       (N,) bool
      ``n_low_ess``     int
      ``n_cases``       int
    """
    x_pois = np.asarray(x_pois, dtype=np.float32)
    truths = np.asarray(truths)
    n_obs, n_params = truths.shape
    nominal_levels = np.asarray(nominal_levels)

    inside_all = np.zeros((len(nominal_levels), n_obs, n_params), dtype=bool)
    ess = np.zeros(n_obs)
    ess_frac = np.zeros(n_obs)
    low_ess = np.zeros(n_obs, dtype=bool)

    for i in range(n_obs):
        is_res = importance_refine(
            posterior, x_pois[i], model_counts_fn, prior_log_prob,
            n_samples=n_is_samples, seed=seed + i,
            low_ess_frac=low_ess_frac, device=device)
        ess[i] = is_res.ess
        ess_frac[i] = is_res.ess_frac
        low_ess[i] = is_res.low_ess
        for li, c in enumerate(nominal_levels):
            qlo = (1.0 - c) / 2.0
            qhi = (1.0 + c) / 2.0
            q = is_refined_quantiles(is_res, [qlo, qhi])  # (2, n_params)
            lo, hi = q[0], q[1]
            inside_all[li, i] = (truths[i] >= lo) & (truths[i] <= hi)

    cov_all = inside_all.mean(axis=1)                         # (levels, params)
    ok = ~low_ess
    if ok.any():
        cov_okess = inside_all[:, ok, :].mean(axis=1)
    else:
        cov_okess = np.full((len(nominal_levels), n_params), np.nan)

    return {
        "cov_all": cov_all,
        "cov_okess": cov_okess,
        "ess": ess,
        "ess_frac": ess_frac,
        "low_ess": low_ess,
        "n_low_ess": int(low_ess.sum()),
        "n_cases": int(n_obs),
    }


# ==========================================================================
# 3b. Conformal / quantile recalibration of 1-D marginals
#     (split-conformal / expected-coverage style; Vovk+05, Lemos+23)
# ==========================================================================

@dataclass
class ConformalRecalibrator:
    """Per-parameter quantile recalibration learned on a held-out calibration set.

    For each parameter we collect, on the calibration set, the posterior CDF value
    at the truth -- u_i = F_post,i(theta_true,i) -- which under perfect calibration
    is Uniform(0,1). We store the EMPIRICAL CDF of these u-values, ``G``. To build a
    recalibrated central interval of nominal level ``c`` for a NEW posterior, we map
    the nominal tail quantiles {(1-c)/2, (1+c)/2} through ``G^{-1}`` to the adjusted
    quantile levels that, on the calibration distribution, actually attain that
    coverage, and read those adjusted quantiles off the new posterior samples.

    This is the 1-D-marginal, quantile-level form of split-conformal recalibration
    / the empirical-coverage remap (cf. Vovk et al. 2005; Angelopoulos & Bates
    2021; the expected-coverage diagnostic of Lemos et al. 2023). It is intentionally
    simple: it only rescales marginal credible-interval WIDTHS to fix 1-D coverage;
    it does not touch the joint dependence structure.
    """
    param_names: list[str]
    u_sorted: list  # per-param sorted PIT values (the empirical G)

    @staticmethod
    def _pit_values(samples_per_obs, truths):
        """PIT u_i = fraction of posterior samples below the truth, per param."""
        n_params = truths.shape[1]
        us = [[] for _ in range(n_params)]
        for s, t in zip(samples_per_obs, truths):
            for p in range(n_params):
                us[p].append(float((s[:, p] < t[p]).mean()))
        return [np.sort(np.asarray(u)) for u in us]

    @classmethod
    def fit(cls, samples_per_obs, truths, param_names):
        truths = np.asarray(truths)
        u_sorted = cls._pit_values(samples_per_obs, truths)
        return cls(param_names=list(param_names), u_sorted=u_sorted)

    def _adjust_level(self, p: int, nominal_q: float) -> float:
        """Map a nominal quantile level to the adjusted level via G^{-1}.

        ``G`` is the empirical CDF of PIT values for parameter ``p``. We want the
        sample-quantile level whose calibration-set coverage equals ``nominal_q``;
        that is ``G^{-1}(nominal_q)`` -- the inverse empirical CDF, i.e. the
        ``nominal_q`` empirical quantile of the stored PIT values.
        """
        u = self.u_sorted[p]
        return float(np.clip(np.quantile(u, np.clip(nominal_q, 0.0, 1.0)), 0.0, 1.0))

    def interval(self, samples: np.ndarray, cred: float = 0.90):
        """Recalibrated equal-tailed central interval at level ``cred``.

        ``samples`` is ``(n_samples, n_params)``. Returns ``(lo, hi)`` each
        ``(n_params,)``."""
        n_params = samples.shape[1]
        qlo_nom = (1.0 - cred) / 2.0
        qhi_nom = (1.0 + cred) / 2.0
        lo = np.zeros(n_params)
        hi = np.zeros(n_params)
        for p in range(n_params):
            qlo = self._adjust_level(p, qlo_nom)
            qhi = self._adjust_level(p, qhi_nom)
            lo[p] = np.quantile(samples[:, p], qlo)
            hi[p] = np.quantile(samples[:, p], qhi)
        return lo, hi

    def coverage_curve(self, samples_per_obs, truths, nominal_levels):
        """Empirical per-parameter coverage AFTER recalibration."""
        truths = np.asarray(truths)
        n_obs = len(samples_per_obs)
        n_params = truths.shape[1]
        cov = np.zeros((len(nominal_levels), n_params))
        for li, c in enumerate(nominal_levels):
            inside = np.zeros((n_obs, n_params), dtype=bool)
            for i, s in enumerate(samples_per_obs):
                lo, hi = self.interval(s, cred=c)
                inside[i] = (truths[i] >= lo) & (truths[i] <= hi)
            cov[li] = inside.mean(axis=0)
        return cov

    def to_dict(self):
        return {"param_names": self.param_names,
                "u_sorted": [u.tolist() for u in self.u_sorted]}


# ==========================================================================
# before/after coverage comparison (npz + figure)
# ==========================================================================

def coverage_before_after(
    samples_cal, truths_cal,
    samples_test, truths_test,
    param_names,
    nominal_levels=None,
):
    """Fit conformal recalibration on the calibration set, then compare raw vs
    recalibrated per-parameter empirical coverage on the test set.

    Returns ``(nominal_levels, cov_raw, cov_recal, recalibrator)``.
    """
    if nominal_levels is None:
        nominal_levels = np.linspace(0.05, 0.95, 19)
    nominal_levels = np.asarray(nominal_levels)

    recal = ConformalRecalibrator.fit(samples_cal, truths_cal, param_names)
    cov_raw = empirical_coverage_curve(samples_test, np.asarray(truths_test),
                                       nominal_levels)
    cov_recal = recal.coverage_curve(samples_test, np.asarray(truths_test),
                                     nominal_levels)
    return nominal_levels, cov_raw, cov_recal, recal


def save_coverage_before_after(
    nominal_levels, cov_raw, cov_recal, param_names,
    npz_path: Path, fig_path: Path, title_suffix: str = "",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_path = Path(npz_path)
    fig_path = Path(fig_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, nominal_levels=nominal_levels,
             cov_raw=cov_raw, cov_recal=cov_recal,
             param_names=np.array(param_names))

    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(3.6 * n_params, 3.6),
                             squeeze=False)
    for p in range(n_params):
        ax = axes[0, p]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        ax.plot(nominal_levels, cov_raw[:, p], "C3o-", ms=3, lw=1.4, label="raw NPE")
        ax.plot(nominal_levels, cov_recal[:, p], "C0s-", ms=3, lw=1.4,
                label="recalibrated")
        ax.set_xlabel("nominal level")
        ax.set_ylabel("empirical coverage")
        ax.set_title(param_names[p], fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if p == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Per-parameter coverage: raw vs conformal-recalibrated "
                 f"{title_suffix}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    return npz_path, fig_path


# ==========================================================================
# orchestration: full suite on one checkpoint directory
# ==========================================================================

@dataclass
class CalibrationConfig:
    n_sbc: int = 1000
    n_tarp: int = 1000
    n_cal: int = 400          # calibration set size for conformal fit
    n_test: int = 400         # held-out test set for before/after coverage
    n_posterior_samples: int = 1000
    n_is_samples: int = 2000
    n_is_cases: int = 20      # how many observations to IS-refine (for the ESS report)
    low_ess_frac: float = 0.1
    seed: int = 20260611
    nominal_levels: list = field(default_factory=lambda: list(np.round(
        np.linspace(0.05, 0.95, 19), 4)))


def _prior_box_log_prob(prior_cfg, param_names):
    """Return a callable theta(M,n)->logp(M,) for the box-uniform prior."""
    low, high = _priors.prior_bounds(prior_cfg, param_names)
    low = np.asarray(low); high = np.asarray(high)
    vol = np.prod(high - low)
    const = -np.log(vol)

    def logp(theta):
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        inside = np.all((theta >= low) & (theta <= high), axis=1)
        out = np.full(theta.shape[0], -np.inf)
        out[inside] = const
        return out
    return logp


def sample_posterior_batch(posterior, x, n_samples, seed=0, device="cpu"):
    """Per-observation posterior sample arrays: list of (n_samples, n_params)."""
    torch.manual_seed(seed)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    out = []
    with torch.no_grad():
        for i in range(x_t.shape[0]):
            s = posterior.sample((n_samples,), x=x_t[i],
                                 show_progress_bars=False,
                                 reject_outside_prior=True)
            out.append(s.detach().cpu().numpy())
    return out
