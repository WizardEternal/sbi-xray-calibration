"""Phase-4 unit tests for the misspecification detectors (detect.py).

Run with the repo venv:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_detect.py

All tests are seeded and use a TINY throwaway flow trained inside the module
fixture (dev Model A = tbabs*powerlaw, 3 params, 1000 sims, <=80 epochs, ~10 s on
CPU). The flow is small but well-trained at a high count level, so an OBVIOUS
misspecification (a strong 6.4 keV Fe-K line that several-folds the counts) is
cleanly separated from clean Model-A spectra by all three detectors (AUC > 0.9).

Contract exercised (brief Phase-4 deliverable 4):
  * D1, D2, D3 each separate an obvious strong Fe-K line from clean, AUC > 0.9.
  * scores are deterministic given the seed.
  * D1 returns BOTH sub-scores (chi2 + KS-on-cumulative).
  * D2 reference embedding cache roundtrips (same array object reused).
  * the marginal-C2ST machinery (D3) returns a CV accuracy + per-spectrum probs.

NOTE (documented, not a bug): a detector gain shift (B4) is NOT separable even at
an extreme 10% by any of the three detectors (a gain shift preserves spectral
SHAPE; the NPE absorbs it into the continuum). That is a genuine scientific
negative result (Phase 4), so the AUC>0.9 gate is tested with the
strong LINE (obvious AND detectable); B4 is tested only for finite/deterministic
operation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from sbixcal import detect as D
from sbixcal import models as M
from sbixcal import priors as P
from sbixcal import responses as R
from sbixcal import simulate as S
from sbixcal import train_npe as TN
from sbixcal import misspec as MS
from jaxspec.data.util import fakeit_for_multiple_parameters


BASE = "tbabs_powerlaw"
PRIOR_CFG = {
    "tbabs_1_nh": {"dist": "uniform", "low": 0.1, "high": 0.3},
    "powerlaw_1_alpha": {"dist": "uniform", "low": 0.5, "high": 2.5},
    "powerlaw_1_norm": {"dist": "loguniform", "low": 1e-4, "high": 1e-2},
}
EXPOSURE = 3000.0          # bright-ish: ~3k clean counts so an obvious line is clear
LINE_NORM = 1e-2           # a STRONG, unambiguous Fe-K line (several-folds the counts)
N_TEST = 30


@pytest.fixture(scope="module")
def setup():
    """Train a tiny throwaway NSF+CNN flow on dev Model A and build the detector
    simulator context. Returns (posterior, ctx, x_clean, x_b1, x_b4, ref)."""
    warnings.filterwarnings("ignore")
    from sbi.inference import NPE

    order = M.MODEL_PARAMS[BASE]
    oc = R.scale_exposure(R.load_base_obsconf(), EXPOSURE)

    rng = np.random.default_rng(0)
    theta, x, _ = S.simulate_spectra(
        BASE, PRIOR_CFG, oc, 1000, rng, apply_poisson=True, seed_for_fakeit=0
    )

    cfg = {"flow": {"hidden_features": 30, "num_transforms": 3, "num_bins": 8},
           "embedding": {"embed_dim": 12, "mlp_hidden": 32}}
    torch.manual_seed(0)
    np.random.seed(0)
    prior = TN.build_prior(PRIOR_CFG, order)
    est = TN.build_density_estimator(cfg, x.shape[1])
    inf = NPE(prior=prior, density_estimator=est, show_progress_bars=False)
    inf.append_simulations(torch.from_numpy(theta.astype("float32")),
                           torch.from_numpy(x.astype("float32")))
    de = inf.train(training_batch_size=200, stop_after_epochs=15,
                   max_num_epochs=80, show_train_summary=False)
    post = inf.build_posterior(de, prior=prior)

    ctx = D.SimulatorContext(
        base_model=BASE, param_names=order, prior_cfg=PRIOR_CFG,
        exposure_s=EXPOSURE, response_name=R.EXAMPLE_NAME, obsconf=oc,
        n_channels=x.shape[1],
    )

    _, x_clean = ctx.clean_sim(N_TEST, seed=2)

    # OBVIOUS B1: strong Fe-K line
    rng = np.random.default_rng(5)
    samp = P.sample_prior(PRIOR_CFG, order, N_TEST, rng)
    samp["gauss_1_El"] = np.full(N_TEST, 6.4)
    samp["gauss_1_sigma"] = np.full(N_TEST, 0.05)
    samp["gauss_1_norm"] = np.full(N_TEST, LINE_NORM)
    x_b1 = np.asarray(
        fakeit_for_multiple_parameters(oc, M.build_model_b1(BASE), samp,
                                       rng_key=7, apply_stat=True),
        dtype=np.float64,
    )

    # max gain shift (B4) -- for the finite/deterministic test
    x_b4, _, _ = MS.simulate_misspec_population(
        BASE, PRIOR_CFG, oc, "B4", 3.0, N_TEST, seed=9
    )

    ref = D.build_embedding_reference(post, ctx, n_ref=200, seed=0)
    return post, ctx, x_clean, x_b1, x_b4, ref


# ==========================================================================
# 1. each detector separates the obvious strong line from clean, AUC > 0.9
# ==========================================================================

def test_d1_ppc_separates_strong_line(setup):
    post, ctx, x_clean, x_b1, _, _ = setup
    sc = np.array([D.detect_d1_ppc(x, post, ctx, k=80, seed=11) for x in x_clean])
    sm = np.array([D.detect_d1_ppc(x, post, ctx, k=80, seed=11) for x in x_b1])
    _, _, auc = D.roc_auc(sc, sm)
    assert auc > 0.9, f"D1 AUC={auc:.3f} on an obvious strong Fe-K line"


def test_d2_embedding_separates_strong_line(setup):
    post, ctx, x_clean, x_b1, _, ref = setup
    sc = np.array([D.detect_d2_embedding(x, post, ctx, ref) for x in x_clean])
    sm = np.array([D.detect_d2_embedding(x, post, ctx, ref) for x in x_b1])
    _, _, auc = D.roc_auc(sc, sm)
    assert auc > 0.9, f"D2 AUC={auc:.3f} on an obvious strong Fe-K line"


def test_d3_marginal_c2st_separates_strong_line(setup):
    post, ctx, x_clean, x_b1, _, _ = setup
    res = D.detect_d3_c2st_cell(post, ctx, x_clean, x_b1, seed=13)
    _, _, auc = D.roc_auc(res.clean_proba, res.mis_proba)
    assert auc > 0.9, f"D3 AUC={auc:.3f} on an obvious strong Fe-K line"
    # CV accuracy is a sane C2ST statistic (>0.5, the misspec IS distinguishable)
    assert res.cv_accuracy > 0.7
    assert res.clean_proba.shape[0] == N_TEST
    assert res.mis_proba.shape[0] == N_TEST


# ==========================================================================
# 2. determinism: same seed -> identical scores
# ==========================================================================

def test_d1_deterministic(setup):
    post, ctx, x_clean, _, _, _ = setup
    a = D.detect_d1_ppc(x_clean[0], post, ctx, k=60, seed=3)
    b = D.detect_d1_ppc(x_clean[0], post, ctx, k=60, seed=3)
    assert a == b


def test_d2_deterministic(setup):
    post, ctx, x_clean, _, _, ref = setup
    a = D.detect_d2_embedding(x_clean[1], post, ctx, ref)
    b = D.detect_d2_embedding(x_clean[1], post, ctx, ref)
    assert a == b


def test_d3_deterministic(setup):
    post, ctx, x_clean, x_b1, _, _ = setup
    r1 = D.detect_d3_c2st_cell(post, ctx, x_clean, x_b1, seed=13)
    r2 = D.detect_d3_c2st_cell(post, ctx, x_clean, x_b1, seed=13)
    assert r1.cv_accuracy == r2.cv_accuracy
    assert np.array_equal(r1.mis_proba, r2.mis_proba)


# ==========================================================================
# 3. D1 returns BOTH sub-scores (chi2 + KS-on-cumulative)
# ==========================================================================

def test_d1_returns_both_subscores(setup):
    post, ctx, x_clean, _, _, _ = setup
    score, parts = D.detect_d1_ppc(x_clean[0], post, ctx, k=80, seed=4,
                                   return_parts=True)
    assert "d1_chi2" in parts and "d1_ks" in parts
    assert 0.0 <= parts["d1_chi2"] <= 1.0
    assert 0.0 <= parts["d1_ks"] <= 1.0
    # combined score is the max of the two sub-scores (suspicious if EITHER fails)
    assert score == pytest.approx(max(parts["d1_chi2"], parts["d1_ks"]))


# ==========================================================================
# 4. D2 reference embedding cache roundtrip
# ==========================================================================

def test_d2_reference_cache_roundtrip(setup):
    post, ctx, _, _, _, _ = setup
    # first build populates the ctx cache; second build returns the SAME embeddings
    r1 = D.build_embedding_reference(post, ctx, n_ref=64, seed=42)
    key = ("embed", 64, 42)
    assert key in ctx._ref_embed_cache
    cached = ctx._ref_embed_cache[key]
    r2 = D.build_embedding_reference(post, ctx, n_ref=64, seed=42)
    # the cached reference embeddings are reused (identical), not recomputed
    assert np.array_equal(r1.ref, r2.ref)
    assert np.array_equal(r1.ref, cached)
    # and the fitted Mahalanobis machinery is consistent
    assert np.allclose(r1.mean, r2.mean)


# ==========================================================================
# 5. D2 sub-scores: kNN is primary, Mahalanobis returned as secondary
# ==========================================================================

def test_d2_returns_knn_and_mahalanobis(setup):
    post, ctx, x_clean, _, _, ref = setup
    score, parts = D.detect_d2_embedding(x_clean[0], post, ctx, ref,
                                         return_parts=True)
    assert "d2_knn" in parts and "d2_mahalanobis" in parts
    # primary score is the kNN distance
    assert score == pytest.approx(parts["d2_knn"])
    assert parts["d2_knn"] >= 0.0 and parts["d2_mahalanobis"] >= 0.0


# ==========================================================================
# 6. all detectors run finite + deterministic on a gain shift (B4).
#    (B4 is NOT separable -- a documented negative result -- so we do not assert
#     AUC>0.9 here, only finite/deterministic operation.)
# ==========================================================================

def test_detectors_run_on_gain_shift(setup):
    post, ctx, x_clean, _, x_b4, ref = setup
    s1 = D.detect_d1_ppc(x_b4[0], post, ctx, k=60, seed=5)
    s2 = D.detect_d2_embedding(x_b4[0], post, ctx, ref)
    res = D.detect_d3_c2st_cell(post, ctx, x_clean, x_b4, seed=6)
    assert np.isfinite(s1) and np.isfinite(s2) and np.isfinite(res.cv_accuracy)
    # deterministic
    assert s1 == D.detect_d1_ppc(x_b4[0], post, ctx, k=60, seed=5)


# ==========================================================================
# 7. roc_auc identities (pure function, no flow needed)
# ==========================================================================

def test_roc_auc_perfect_and_chance():
    # perfectly separated: misspec all above clean -> AUC 1.0
    _, _, auc = D.roc_auc(np.array([0.0, 0.1, 0.2]), np.array([0.8, 0.9, 1.0]))
    assert auc == pytest.approx(1.0)
    # identical distributions -> AUC ~0.5
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    b = rng.normal(size=400)
    _, _, auc2 = D.roc_auc(a, b)
    assert 0.4 < auc2 < 0.6


def test_score_d3_raises_per_spectrum():
    # D3 has no per-spectrum score via the unified interface (by design)
    with pytest.raises(ValueError):
        D.score(np.zeros(102), None, None, "D3")
