"""Phase-3 unit tests for the calibration suite (calibrate.py).

Run with the repo venv:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_calibrate.py

All tests are tiny, fast, and seeded. They exercise the Phase-3 contract
on GAUSSIAN / Poisson TOY problems where the answer is analytically known:

  * SBC on a tiny NSF flow trained on a linear-Gaussian toy (x = theta + noise):
    sbi's run_sbc ranks are roughly uniform (loose, seeded KS bound), and TARP
    expected coverage is near-diagonal.
  * IS-refinement sanity on a Poisson "spectrum" toy where the exact 1-D
    posterior is computable: a deliberately-perturbed proposal is IS-corrected
    TOWARD the truth, the weights normalize to 1, and ESS is computed.
  * Conformal recalibration on synthetic too-narrow (overconfident) posteriors:
    after recalibration the empirical coverage matches nominal within tolerance.

The Gaussian-toy SBC flow is trained once (module-scoped fixture) and shared by
the SBC and TARP tests to keep the suite under ~30 s.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from sbixcal import calibrate as C


# ==========================================================================
# shared tiny Gaussian-toy flow (trained once for SBC + TARP)
# ==========================================================================

TOY_DIM = 2
TOY_SIGMA = 0.3
TOY_HIGH = 3.0


def _toy_prior():
    from sbi.utils import BoxUniform
    return BoxUniform(low=torch.zeros(TOY_DIM), high=torch.ones(TOY_DIM) * TOY_HIGH)


def _toy_simulate(theta: torch.Tensor, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return theta + TOY_SIGMA * torch.randn(theta.shape, generator=g)


@pytest.fixture(scope="module")
def toy_flow():
    """A tiny NSF flow trained on the linear-Gaussian toy x = theta + N(0,sigma^2).

    The true posterior of this toy is itself a (truncated) Gaussian, so a well-
    trained flow should be SBC-uniform; we keep the flow small and cap epochs to
    stay fast while remaining well-calibrated.
    """
    warnings.filterwarnings("ignore")
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn

    torch.manual_seed(0)
    np.random.seed(0)
    prior = _toy_prior()
    theta = prior.sample((1000,))
    x = _toy_simulate(theta, seed=0)

    est = posterior_nn(model="nsf", hidden_features=30, num_transforms=3, num_bins=6)
    inf = NPE(prior=prior, density_estimator=est, show_progress_bars=False)
    inf.append_simulations(theta, x)
    de = inf.train(training_batch_size=200, stop_after_epochs=15,
                   max_num_epochs=80, show_train_summary=False)
    post = inf.build_posterior(de, prior=prior)
    return post, prior


# ==========================================================================
# 1. SBC on the Gaussian toy: ranks roughly uniform
# ==========================================================================

def test_sbc_ranks_uniform_on_gaussian_toy(toy_flow):
    post, prior = toy_flow
    torch.manual_seed(1)
    theta_t = prior.sample((200,))
    x_t = _toy_simulate(theta_t, seed=1)

    sbc = C.run_sbc_check(post, prior, theta_t, x_t,
                          param_names=["a", "b"],
                          num_posterior_samples=150, seed=1)

    # shapes / bookkeeping
    assert sbc.ranks.shape == (200, TOY_DIM)
    assert sbc.ks_pvals.shape == (TOY_DIM,)
    assert sbc.c2st_ranks.shape == (TOY_DIM,)

    # ranks roughly uniform: KS p-value should NOT reject uniformity hard.
    # Loose, seeded bound (a well-calibrated toy gives p ~ 0.2-0.6 here).
    assert np.all(sbc.ks_pvals > 0.02), sbc.ks_pvals
    # C2ST accuracy of ranks vs uniform should be near 0.5 (not strongly > 0.5)
    assert np.all(sbc.c2st_ranks < 0.75), sbc.c2st_ranks

    # the rank mean should be near the midpoint num_posterior_samples/2
    assert np.all(np.abs(sbc.ranks.mean(0) - 150 / 2) < 25), sbc.ranks.mean(0)


def test_sbc_figure_renders(tmp_path, toy_flow):
    post, prior = toy_flow
    torch.manual_seed(3)
    theta_t = prior.sample((150,))
    x_t = _toy_simulate(theta_t, seed=3)
    sbc = C.run_sbc_check(post, prior, theta_t, x_t, ["a", "b"],
                          num_posterior_samples=120, seed=3)
    out = C.save_sbc_figure(sbc, tmp_path / "sbc.png")
    assert out.exists() and out.stat().st_size > 0


# ==========================================================================
# 2. TARP expected coverage on the Gaussian toy: near-diagonal
# ==========================================================================

def test_tarp_near_diagonal_on_gaussian_toy(tmp_path, toy_flow):
    post, prior = toy_flow
    torch.manual_seed(2)
    theta_t = prior.sample((200,))
    x_t = _toy_simulate(theta_t, seed=2)

    tarp = C.run_tarp_check(post, theta_t, x_t, num_posterior_samples=150, seed=2)
    assert tarp.ecp.shape == tarp.alpha.shape
    # area-to-curve close to 0 for a calibrated flow (loose bound)
    assert abs(tarp.atc) < 0.1, tarp.atc
    # KS p-value should not reject (large p = consistent with uniform ECP)
    assert tarp.ks_pval > 0.05, tarp.ks_pval

    npz, fig = C.save_tarp_npz_and_figure(
        tarp, tmp_path / "tarp.npz", tmp_path / "tarp.png")
    assert npz.exists() and fig.exists()
    d = np.load(npz)
    assert "ecp" in d and "alpha" in d


# ==========================================================================
# 3a. IS-refinement on a Poisson toy with a computable posterior
# ==========================================================================

# A Poisson "spectrum" toy: theta = (a,) in [1,8]; per-channel model counts
# lambda_c = a * template_c; observed counts x_c ~ Poisson(lambda_c). The exact
# Poisson likelihood is C.poisson_loglik, so the 1-D posterior over a (flat prior)
# is computable on a grid.

_TOY_TEMPLATE = np.linspace(1.0, 3.0, 8)
_TOY_A_TRUE = 5.0
_TOY_A_LOW, _TOY_A_HIGH = 1.0, 8.0


def _toy_model_counts(theta):
    a = np.atleast_2d(np.asarray(theta, dtype=np.float64))[:, 0:1]
    return a * _TOY_TEMPLATE[None, :]


def _exact_poisson_posterior_quantiles(x_obs, qs):
    grid = np.linspace(_TOY_A_LOW, _TOY_A_HIGH, 6000)
    ll = C.poisson_loglik(x_obs, _toy_model_counts(grid[:, None]))
    w = np.exp(ll - ll.max())
    cdf = np.cumsum(w) / np.sum(w)
    return np.array([float(grid[np.searchsorted(cdf, q)]) for q in qs])


class _GaussProposal:
    """A stand-in NPE proposal q(a|x) = N(mu, sd^2) with a real log_prob.

    Mimics the DirectPosterior interface used by importance_refine: sample() and
    log_prob(theta, x). Deliberately MISplaced (mu != posterior mean) so the
    IS-correction has something to repair.
    """

    def __init__(self, mu, sd, seed=0):
        self.mu = float(mu)
        self.sd = float(sd)
        self._rng = np.random.default_rng(seed)

    def sample(self, shape, x=None, **kw):
        n = shape[0]
        s = self._rng.normal(self.mu, self.sd, (n, 1))
        return torch.as_tensor(s, dtype=torch.float32)

    def log_prob(self, theta, x=None, **kw):
        th = theta.detach().cpu().numpy()[:, 0]
        lp = -0.5 * ((th - self.mu) / self.sd) ** 2 - np.log(self.sd * np.sqrt(2 * np.pi))
        return torch.as_tensor(lp, dtype=torch.float32)


def test_is_refinement_moves_toward_truth_and_computes_ess():
    rng = np.random.default_rng(0)
    x_obs = rng.poisson(_TOY_A_TRUE * _TOY_TEMPLATE).astype(np.float64)

    exact = _exact_poisson_posterior_quantiles(x_obs, [0.05, 0.5, 0.95])
    exact_med = exact[1]

    # proposal deliberately centered LOW (a~3.4) and a bit wide -> clearly biased,
    # but not so pathological that ESS collapses to a handful.
    prop = _GaussProposal(mu=3.4, sd=1.0, seed=1)
    prior_logp = C._prior_box_log_prob(
        {"a": {"dist": "uniform", "low": _TOY_A_LOW, "high": _TOY_A_HIGH}}, ["a"])

    is_res = C.importance_refine(
        prop, x_obs, _toy_model_counts, prior_logp,
        n_samples=20000, seed=0, low_ess_frac=0.1)

    # weights normalize to 1
    assert np.isclose(is_res.weights.sum(), 1.0, atol=1e-8)
    # ESS is computed and in (0, n_samples]
    assert 0.0 < is_res.ess <= is_res.n_samples
    assert np.isclose(is_res.ess_frac, is_res.ess / is_res.n_samples)

    raw_med = float(np.median(is_res.samples[:, 0]))
    ref = C.is_refined_quantiles(is_res, [0.05, 0.5, 0.95])
    ref_med = float(ref[1, 0])

    # the IS-corrected median is MUCH closer to the exact posterior median than
    # the raw (biased) proposal median.
    assert abs(ref_med - exact_med) < abs(raw_med - exact_med)
    # and lands near the exact posterior median in absolute terms
    assert abs(ref_med - exact_med) < 0.4, (ref_med, exact_med)
    # refined 5/95 bracket the truth
    assert ref[0, 0] < _TOY_A_TRUE < ref[2, 0], ref[:, 0]


def test_is_low_ess_flag_fires_for_pathological_proposal():
    """A badly-placed, over-narrow proposal yields a low ESS fraction; the
    low_ess flag (Paper III's NPE-failure diagnostic) must fire."""
    rng = np.random.default_rng(2)
    x_obs = rng.poisson(_TOY_A_TRUE * _TOY_TEMPLATE).astype(np.float64)
    # proposal far from the truth AND narrow -> tiny overlap -> low ESS
    prop = _GaussProposal(mu=1.5, sd=0.25, seed=5)
    prior_logp = C._prior_box_log_prob(
        {"a": {"dist": "uniform", "low": _TOY_A_LOW, "high": _TOY_A_HIGH}}, ["a"])
    is_res = C.importance_refine(prop, x_obs, _toy_model_counts, prior_logp,
                                 n_samples=20000, seed=0, low_ess_frac=0.1)
    assert is_res.low_ess, is_res.ess_frac
    assert is_res.ess_frac < 0.1


def test_poisson_loglik_matches_scipy():
    from scipy.stats import poisson as sp_poisson
    counts = np.array([3.0, 5.0, 0.0, 12.0])
    lam = np.array([[2.5, 5.5, 0.3, 10.0]])
    got = C.poisson_loglik(counts, lam)[0]
    ref = sp_poisson.logpmf(counts, lam[0]).sum()
    assert np.isclose(got, ref, atol=1e-8), (got, ref)


def test_weighted_quantile_matches_unweighted_for_uniform_weights():
    g = np.random.default_rng(0)
    v = g.normal(size=5000)
    w = np.ones_like(v)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    got = C.weighted_quantile(v, qs, w)
    ref = np.quantile(v, qs)
    assert np.allclose(got, ref, atol=0.02), (got, ref)


# ==========================================================================
# 3b. Conformal recalibration on too-narrow (overconfident) posteriors
# ==========================================================================

def _miscalibrated_set(seed, n_obs=600, n_params=2, n_samp=2000,
                       obs_sd=1.0, post_sd=0.55):
    """Synthetic too-narrow posteriors.

    Truth ~ N(0,1); a noisy estimate m = truth + N(0, obs_sd) is the posterior
    CENTER; the reported posterior is N(m, post_sd^2) with post_sd < obs_sd, i.e.
    OVERCONFIDENT. Because the center is offset from the truth, narrow intervals
    systematically MISS the truth -> empirical coverage below nominal.
    """
    g = np.random.default_rng(seed)
    truths = g.normal(0.0, 1.0, (n_obs, n_params))
    centers = truths + g.normal(0.0, obs_sd, (n_obs, n_params))
    samples = [g.normal(centers[i][None, :], post_sd, (n_samp, n_params))
               for i in range(n_obs)]
    return samples, truths


def test_conformal_recalibration_fixes_coverage():
    s_cal, t_cal = _miscalibrated_set(1)
    s_test, t_test = _miscalibrated_set(2)
    levels = np.linspace(0.1, 0.9, 9)

    cov_raw = C.empirical_coverage_curve(s_test, t_test, levels)
    # raw posteriors are overconfident -> under-cover at the high-nominal end
    assert cov_raw[-1].mean() < 0.8, cov_raw[-1]   # nominal 0.9 under-covered
    raw_dev = np.max(np.abs(cov_raw - levels[:, None]))
    assert raw_dev > 0.1, raw_dev

    recal = C.ConformalRecalibrator.fit(s_cal, t_cal, ["a", "b"])
    cov_recal = recal.coverage_curve(s_test, t_test, levels)
    recal_dev = np.max(np.abs(cov_recal - levels[:, None]))

    # after recalibration, empirical coverage is within tolerance of nominal,
    # and substantially better than raw.
    assert recal_dev < 0.08, recal_dev
    assert recal_dev < raw_dev


def test_coverage_before_after_and_save(tmp_path):
    s_cal, t_cal = _miscalibrated_set(11)
    s_test, t_test = _miscalibrated_set(12)
    levels, cov_raw, cov_recal, recal = C.coverage_before_after(
        s_cal, t_cal, s_test, t_test, ["a", "b"],
        nominal_levels=np.linspace(0.1, 0.9, 9))

    # recalibrated closer to the diagonal than raw, on aggregate
    raw_dev = np.mean(np.abs(cov_raw - levels[:, None]))
    recal_dev = np.mean(np.abs(cov_recal - levels[:, None]))
    assert recal_dev < raw_dev

    npz, fig = C.save_coverage_before_after(
        levels, cov_raw, cov_recal, ["a", "b"],
        tmp_path / "cov.npz", tmp_path / "cov.png", title_suffix="(toy)")
    assert npz.exists() and fig.exists()
    d = np.load(npz, allow_pickle=True)
    assert d["cov_raw"].shape == d["cov_recal"].shape


def test_empirical_coverage_curve_perfectly_calibrated():
    """A correctly-calibrated posterior (center=truth+N(0,1), sd=1) should give
    empirical coverage ~ nominal."""
    s, t = _miscalibrated_set(7, obs_sd=1.0, post_sd=1.0)
    levels = np.array([0.3, 0.5, 0.7, 0.9])
    cov = C.empirical_coverage_curve(s, t, levels)
    # within ~0.05 of nominal for a well-specified Gaussian
    assert np.all(np.abs(cov.mean(axis=1) - levels) < 0.06), cov.mean(axis=1)
