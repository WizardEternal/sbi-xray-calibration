"""Phase 1 unit tests for the spectral simulator and misspecification generators.

Run with the repo venv:
    .venv\\Scripts\\python.exe -m pytest -q

Tests are deliberately small (a few hundred spectra) so the suite runs in
seconds. They check the scientific invariants for the Phase 1 simulator:
  - counts scale linearly with exposure and with norm,
  - B1 adds counts localized near 6.4 keV,
  - B4 shifts spectral features,
  - identical seeds -> identical spectra,
  - prior samples respect their bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxspec.data.util import fakeit_for_multiple_parameters

from sbixcal import models, priors, responses, misspec
from sbixcal import simulate as sim


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_obs():
    return responses.load_base_obsconf()


@pytest.fixture(scope="module")
def channel_energy(base_obs):
    e = np.asarray(base_obs.out_energies)
    return 0.5 * (e[0] + e[1])


DEV_PRIORS = {
    "tbabs_1_nh": {"dist": "uniform", "low": 0.1, "high": 0.3},
    "powerlaw_1_alpha": {"dist": "uniform", "low": 0.5, "high": 2.5},
    "powerlaw_1_norm": {"dist": "loguniform", "low": 1e-4, "high": 1e-2},
}
DEV_ORDER = models.MODEL_PARAMS["tbabs_powerlaw"]


def _expected_counts(obsconf, params):
    model = models.build_model("tbabs_powerlaw")
    return np.asarray(
        fakeit_for_multiple_parameters(obsconf, model, params, apply_stat=False)
    )


# --------------------------------------------------------------------------
# counts scale linearly with exposure
# --------------------------------------------------------------------------

def test_counts_scale_linearly_with_exposure(base_obs):
    params = {
        "tbabs_1_nh": np.array([0.2]),
        "powerlaw_1_alpha": np.array([1.7]),
        "powerlaw_1_norm": np.array([1e-3]),
    }
    e1 = _expected_counts(responses.scale_exposure(base_obs, 1000.0), params).sum()
    e2 = _expected_counts(responses.scale_exposure(base_obs, 5000.0), params).sum()
    # 5x exposure -> 5x counts
    assert e2 / e1 == pytest.approx(5.0, rel=1e-4)


def test_counts_scale_linearly_with_norm(base_obs):
    obsconf = responses.scale_exposure(base_obs, 1000.0)
    base = {
        "tbabs_1_nh": np.array([0.2]),
        "powerlaw_1_alpha": np.array([1.7]),
        "powerlaw_1_norm": np.array([1e-3]),
    }
    hi = dict(base, powerlaw_1_norm=np.array([3e-3]))
    c1 = _expected_counts(obsconf, base).sum()
    c3 = _expected_counts(obsconf, hi).sum()
    # powerlaw counts are linear in norm
    assert c3 / c1 == pytest.approx(3.0, rel=1e-4)


# --------------------------------------------------------------------------
# B1 adds counts localized near 6.4 keV
# --------------------------------------------------------------------------

def test_b1_adds_localized_counts_near_fe_k(base_obs, channel_energy):
    obsconf = responses.scale_exposure(base_obs, 5000.0)
    src = {
        "tbabs_1_nh": np.array([0.2]),
        "powerlaw_1_alpha": np.array([1.7]),
        "powerlaw_1_norm": np.array([1e-3]),
    }
    clean = _expected_counts(obsconf, src)[0]

    model_b1 = models.build_model_b1("tbabs_powerlaw")
    src_b1 = dict(
        src,
        gauss_1_El=np.array([6.4]),
        gauss_1_sigma=np.array([0.05]),
        gauss_1_norm=np.array([5e-4]),
    )
    withline = np.asarray(
        fakeit_for_multiple_parameters(obsconf, model_b1, src_b1, apply_stat=False)
    )[0]

    diff = withline - clean
    near = (channel_energy > 5.8) & (channel_energy < 7.0)
    far = ~near
    # the excess counts should be concentrated in the Fe-K window
    frac_near = diff[near].sum() / diff.sum()
    assert diff.sum() > 0
    assert frac_near > 0.8, f"only {frac_near:.2f} of the added counts are near Fe-K"
    # and the far-from-line bins barely change
    assert abs(diff[far]).sum() < 0.2 * diff.sum()


def test_b1_strength_monotone(base_obs, channel_energy):
    obsconf = responses.scale_exposure(base_obs, 5000.0)
    model_b1 = models.build_model_b1("tbabs_powerlaw")
    src = dict(
        tbabs_1_nh=np.array([0.2]),
        powerlaw_1_alpha=np.array([1.7]),
        powerlaw_1_norm=np.array([1e-3]),
        gauss_1_El=np.array([6.4]),
        gauss_1_sigma=np.array([0.05]),
    )
    totals = []
    for norm in (0.0, 1e-4, 5e-4, 2e-3):
        s = dict(src, gauss_1_norm=np.array([norm]))
        totals.append(
            np.asarray(fakeit_for_multiple_parameters(obsconf, model_b1, s, apply_stat=False)).sum()
        )
    # stronger line -> more total counts, monotonically
    assert all(np.diff(totals) > 0)


# --------------------------------------------------------------------------
# B4 shifts spectral features
# --------------------------------------------------------------------------

def test_b4_shifts_feature(base_obs, channel_energy):
    obsconf = responses.scale_exposure(base_obs, 5000.0)
    model = models.build_model_b1("tbabs_powerlaw")  # use a line so the shift is measurable
    src = dict(
        tbabs_1_nh=np.array([0.2]),
        powerlaw_1_alpha=np.array([1.7]),
        powerlaw_1_norm=np.array([1e-3]),
        gauss_1_El=np.array([6.4]),
        gauss_1_sigma=np.array([0.05]),
        gauss_1_norm=np.array([5e-3]),
    )

    win = (channel_energy > 5.5) & (channel_energy < 7.5)

    def centroid(oc):
        spec = np.asarray(fakeit_for_multiple_parameters(oc, model, src, apply_stat=False))[0]
        w = np.clip(spec[win] - spec[win].min(), 0, None)
        return float(np.sum(channel_energy[win] * w) / np.sum(w))

    c_nom = centroid(obsconf)
    c_up = centroid(responses.gain_shift_obsconf(obsconf, 1.02))   # +2%
    c_dn = centroid(responses.gain_shift_obsconf(obsconf, 0.98))   # -2%

    # +gain pushes the line to lower channel energy, -gain to higher (documented
    # convention), and the shift magnitude is ~2% of 6.4 keV ~ 0.13 keV.
    assert c_up < c_nom < c_dn
    assert abs(abs(c_up - c_nom) - 0.128) < 0.05
    assert abs(abs(c_dn - c_nom) - 0.128) < 0.05


def test_b4_zero_percent_is_nominal(base_obs):
    obsconf = responses.scale_exposure(base_obs, 5000.0)
    model = models.build_model("tbabs_powerlaw")
    src = dict(
        tbabs_1_nh=np.array([0.2]),
        powerlaw_1_alpha=np.array([1.7]),
        powerlaw_1_norm=np.array([1e-3]),
    )
    a = np.asarray(fakeit_for_multiple_parameters(obsconf, model, src, apply_stat=False))
    oc0 = responses.gain_shift_obsconf(obsconf, 1.0)  # 0% gain
    b = np.asarray(fakeit_for_multiple_parameters(oc0, model, src, apply_stat=False))
    np.testing.assert_allclose(a, b, rtol=1e-6)


# --------------------------------------------------------------------------
# identical seeds -> identical spectra
# --------------------------------------------------------------------------

def test_identical_seed_identical_spectra(base_obs):
    obsconf = responses.scale_exposure(base_obs, 2000.0)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    theta_a, x_a, _ = sim.simulate_spectra(
        "tbabs_powerlaw", DEV_PRIORS, obsconf, 200, rng_a,
        apply_poisson=True, seed_for_fakeit=7,
    )
    theta_b, x_b, _ = sim.simulate_spectra(
        "tbabs_powerlaw", DEV_PRIORS, obsconf, 200, rng_b,
        apply_poisson=True, seed_for_fakeit=7,
    )
    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(x_a, x_b)


def test_different_seed_different_spectra(base_obs):
    obsconf = responses.scale_exposure(base_obs, 2000.0)
    _, x_a, _ = sim.simulate_spectra(
        "tbabs_powerlaw", DEV_PRIORS, obsconf, 200,
        np.random.default_rng(1), apply_poisson=True, seed_for_fakeit=7,
    )
    _, x_b, _ = sim.simulate_spectra(
        "tbabs_powerlaw", DEV_PRIORS, obsconf, 200,
        np.random.default_rng(2), apply_poisson=True, seed_for_fakeit=8,
    )
    assert not np.array_equal(x_a, x_b)


# --------------------------------------------------------------------------
# prior samples respect bounds
# --------------------------------------------------------------------------

def test_prior_samples_respect_bounds():
    rng = np.random.default_rng(0)
    s = priors.sample_prior(DEV_PRIORS, DEV_ORDER, 50000, rng)
    assert priors.within_bounds(s, DEV_PRIORS, DEV_ORDER)
    # explicit per-parameter checks
    assert s["tbabs_1_nh"].min() >= 0.1 and s["tbabs_1_nh"].max() <= 0.3
    assert s["powerlaw_1_alpha"].min() >= 0.5 and s["powerlaw_1_alpha"].max() <= 2.5
    assert s["powerlaw_1_norm"].min() >= 1e-4 and s["powerlaw_1_norm"].max() <= 1e-2


def test_loguniform_is_log_uniform():
    rng = np.random.default_rng(0)
    s = priors.sample_prior(
        {"p": {"dist": "loguniform", "low": 1e-4, "high": 1e-2}}, ["p"], 100000, rng
    )
    logs = np.log10(s["p"])
    # uniform in log10 over 2 decades -> roughly flat histogram, mean near midpoint
    assert logs.min() >= -4 - 1e-6 and logs.max() <= -2 + 1e-6
    assert np.mean(logs) == pytest.approx(-3.0, abs=0.05)


# --------------------------------------------------------------------------
# B2 / B3 sanity (generators produce finite, sensible spectra)
# --------------------------------------------------------------------------

def test_b2_covering_fraction_changes_counts(base_obs):
    obsconf = responses.scale_exposure(base_obs, 5000.0)
    rng = np.random.default_rng(0)
    totals = {}
    for f in (1.0, 0.5):
        model, params = misspec._params_b2("tbabs_powerlaw", DEV_PRIORS, 300, rng, f, {})
        x = np.asarray(fakeit_for_multiple_parameters(obsconf, model, params, apply_stat=False))
        totals[f] = np.median(x.sum(axis=1))
    # partial covering (f=0.5) leaks unabsorbed flux -> more counts than full (f=1)
    assert totals[0.5] > totals[1.0]


def test_b3_brems_continuum_finite_and_curved(base_obs, channel_energy):
    obsconf = responses.scale_exposure(base_obs, 5000.0)
    rng = np.random.default_rng(0)
    model, params = misspec._params_b3(
        "tbabs_powerlaw", DEV_PRIORS, 200, rng, 1.5, {"use_diskbb": False}
    )
    x = np.asarray(fakeit_for_multiple_parameters(obsconf, model, params, apply_stat=False))
    assert np.isfinite(x).all()
    assert (x.sum(axis=1) > 0).all()
    # a kT=1.5 keV brems should be soft: more counts below 2 keV than above 5 keV
    soft = channel_energy < 2.0
    hard = channel_energy > 5.0
    assert x[:, soft].sum() > x[:, hard].sum()
