"""Phase-2 unit tests for NPE training (train_npe.py).

Run with the repo venv:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_train_npe.py

These are deliberately small (1k sims, 2 epochs) so the suite runs in seconds.
They check the Phase-2 contract from the task brief:
  - the 1-D CNN embedding net forward produces the configured embed_dim,
  - a short training run on 1k sims completes and yields finite loss curves,
  - a checkpoint round-trips: save -> cold load_posterior -> sample, and
    sampling is deterministic in eval mode (same seed -> identical samples),
  - the cold-loaded posterior reproduces the in-memory trained one bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sbixcal import train_npe as tn


# --------------------------------------------------------------------------
# helpers / fixtures
# --------------------------------------------------------------------------

DEV_CFG = {
    "name": "test_dev",
    "base_model": "tbabs_powerlaw",
    "seed": 0,
    "priors": {
        "tbabs_1_nh": {"dist": "uniform", "low": 0.1, "high": 0.3},
        "powerlaw_1_alpha": {"dist": "uniform", "low": 0.5, "high": 2.5},
        "powerlaw_1_norm": {"dist": "loguniform", "low": 1e-4, "high": 1e-2},
    },
    "embedding": {"embed_dim": 16, "conv_channels": [16, 32], "kernel_size": 5,
                  "mlp_hidden": 64},
    "flow": {"hidden_features": 24, "num_transforms": 3, "num_bins": 8},
    "train": {"batch_size": 100, "learning_rate": 5e-4, "validation_fraction": 0.1,
              "stop_after_epochs": 100, "max_num_epochs": 2, "show_progress": False},
}
PARAM_ORDER = tn._models.MODEL_PARAMS["tbabs_powerlaw"]


def _fake_data(n=1000, n_channels=102, seed=0):
    """Cheap synthetic (theta, x) -- a smooth nonlinear map so the flow has
    signal, plus Poisson-like noise. Avoids touching jaxspec for a fast test."""
    g = np.random.default_rng(seed)
    low, high = tn._priors.prior_bounds(DEV_CFG["priors"], PARAM_ORDER)
    theta = g.uniform(low, high, size=(n, len(PARAM_ORDER)))
    base = np.linspace(0.3, 10.0, n_channels)[None, :]
    # crude "spectrum": power-law-ish * absorption-ish, scaled by norm
    nh = theta[:, [0]]
    gamma = theta[:, [1]]
    norm = theta[:, [2]]
    rate = norm * 1e4 * base ** (-gamma) * np.exp(-nh * 0.5 / base)
    x = g.poisson(np.clip(rate, 0, None)).astype(np.float32)
    return (torch.as_tensor(theta, dtype=torch.float32),
            torch.as_tensor(x, dtype=torch.float32))


# --------------------------------------------------------------------------
# embedding net forward shape
# --------------------------------------------------------------------------

def test_embedding_forward_shape():
    net = tn.build_embedding_net(DEV_CFG, n_channels=102)
    x = torch.rand(13, 102) * 100
    out = net(x)
    assert out.shape == (13, 16)
    # embed_dim configurable
    cfg2 = dict(DEV_CFG, embedding=dict(DEV_CFG["embedding"], embed_dim=20))
    net2 = tn.build_embedding_net(cfg2, n_channels=102)
    assert net2(x).shape == (13, 20)


def test_embedding_param_count_reasonable():
    net = tn.build_embedding_net(DEV_CFG, n_channels=102)
    n = sum(p.numel() for p in net.parameters())
    # "~tens of k params"
    assert 5_000 < n < 500_000, n


def test_embedding_log1p_handles_zeros_and_large():
    net = tn.build_embedding_net(DEV_CFG, n_channels=102)
    x = torch.cat([torch.zeros(2, 102), torch.full((2, 102), 1e5)], dim=0)
    out = net(x)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# 2-epoch training on 1k sims runs
# --------------------------------------------------------------------------

def test_train_two_epochs_runs():
    theta, x = _fake_data(n=1000, seed=1)
    prior = tn.build_prior(DEV_CFG["priors"], PARAM_ORDER, device="cpu")
    de, inf, summary = tn.train_one_flow(theta, x, prior, DEV_CFG, device="cpu", seed=0)
    tl = summary["training_loss"]
    vl = summary["validation_loss"]
    assert len(tl) >= 1 and len(vl) >= 1
    assert np.all(np.isfinite(tl)) and np.all(np.isfinite(vl))
    # the flow can produce samples conditioned on a spectrum
    from sbi.inference import NPE
    post = NPE(prior=prior, density_estimator="nsf", show_progress_bars=False
               ).build_posterior(de, prior=prior)
    s = post.sample((10,), x=x[0], show_progress_bars=False)
    assert s.shape == (10, len(PARAM_ORDER))


# --------------------------------------------------------------------------
# checkpoint save -> cold load -> sample roundtrip is deterministic
# --------------------------------------------------------------------------

def test_checkpoint_roundtrip_deterministic(tmp_path):
    theta, x = _fake_data(n=1000, seed=2)
    prior = tn.build_prior(DEV_CFG["priors"], PARAM_ORDER, device="cpu")
    de, inf, summary = tn.train_one_flow(theta, x, prior, DEV_CFG, device="cpu", seed=0)

    meta = {"exposure_s": 649.1, "median_total_counts": float(np.median(x.sum(1))),
            "n": x.shape[0]}
    out_dir = tmp_path / "ckpt"
    tn.save_checkpoint(out_dir, de, summary, DEV_CFG, PARAM_ORDER, x.shape[1], meta)

    # all expected artifacts written
    for fn in ("flow_state.pt", "arch.json", "training_loss.npy",
               "validation_loss.npy", "summary.json"):
        assert (out_dir / fn).exists(), fn

    # cold load (no training data)
    post, info = tn.load_posterior(out_dir, device="cpu")
    assert info["param_names"] == PARAM_ORDER
    xo = x[0]

    # same seed -> identical samples (eval-mode determinism)
    torch.manual_seed(123)
    s1 = post.sample((40,), x=xo, show_progress_bars=False)
    torch.manual_seed(123)
    s2 = post.sample((40,), x=xo, show_progress_bars=False)
    assert torch.equal(s1, s2)


# --------------------------------------------------------------------------
# validation helpers: credible interval + coverage fraction
# --------------------------------------------------------------------------

def test_credible_interval_recovers_known_quantiles():
    # n_params=2; column 0 ~ U[0,1], column 1 ~ U[10,20]; large N so the
    # empirical 5/95 percentiles match the analytic ones closely.
    g = np.random.default_rng(0)
    n = 200_000
    s = np.stack([g.uniform(0, 1, n), g.uniform(10, 20, n)], axis=1)
    lo, hi, med = tn.credible_interval(s, cred=0.90)
    assert np.allclose(lo, [0.05, 10.5], atol=0.01)
    assert np.allclose(hi, [0.95, 19.5], atol=0.01)
    assert np.allclose(med, [0.5, 15.0], atol=0.01)


def test_coverage_fraction_per_param_and_joint():
    # 4 test cases, 2 params. Hand-built intervals so coverage is exact.
    truth = np.array([[1.0, 1.0],
                      [1.0, 1.0],
                      [1.0, 1.0],
                      [1.0, 1.0]])
    lo = np.array([[0.0, 0.0],     # both in
                   [0.0, 2.0],     # p0 in, p1 out (lo>truth)
                   [2.0, 0.0],     # p0 out, p1 in
                   [0.0, 0.0]])    # both in
    hi = np.array([[2.0, 2.0],
                   [2.0, 3.0],
                   [3.0, 2.0],
                   [2.0, 2.0]])
    per_param, joint = tn.coverage_fraction(truth, lo, hi)
    # p0 inside in cases 0,1,3 -> 3/4 ; p1 inside in 0,2,3 -> 3/4
    assert np.allclose(per_param, [0.75, 0.75])
    # joint (both in): cases 0 and 3 -> 2/4
    assert joint == 0.5


def test_cold_load_matches_inmemory(tmp_path):
    """The cold-loaded flow reproduces the in-memory trained flow exactly."""
    theta, x = _fake_data(n=1000, seed=3)
    prior = tn.build_prior(DEV_CFG["priors"], PARAM_ORDER, device="cpu")
    de, inf, summary = tn.train_one_flow(theta, x, prior, DEV_CFG, device="cpu", seed=0)
    de.eval()

    meta = {"exposure_s": 649.1, "median_total_counts": float(np.median(x.sum(1))),
            "n": x.shape[0]}
    out_dir = tmp_path / "ckpt2"
    tn.save_checkpoint(out_dir, de, summary, DEV_CFG, PARAM_ORDER, x.shape[1], meta)

    post_mem = inf.build_posterior(de, prior=prior)
    post_cold, _ = tn.load_posterior(out_dir, device="cpu")
    xo = x[1]

    torch.manual_seed(7)
    a = post_mem.sample((30,), x=xo, show_progress_bars=False)
    torch.manual_seed(7)
    b = post_cold.sample((30,), x=xo, show_progress_bars=False)
    assert torch.allclose(a, b, atol=1e-5)
