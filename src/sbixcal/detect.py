"""Misspecification detectors for sbi-xray-calibration (Phase 4).

Three detectors behind ONE interface

    score(spectrum, posterior, simulator_ctx, *, cfg, seed) -> float

where a HIGHER score means MORE suspicious (more likely the observed spectrum is
misspecified relative to the well-specified Model A the flow was trained on). All
three are config-driven and operate on ANY trained checkpoint via
``train_npe.load_posterior`` (the flow carries its own CNN embedding net and the
prior/base-model/exposure needed to rebuild the simulator context).

Framing: this is a systematic
misspecification-detection benchmark for X-ray spectral SBI. The individual
detector ideas are not new; they are adapted from the general-SBI
misspecification literature and benchmarked here on X-ray spectra. Each
detector's docstring cites its methodological ancestor.

Detectors
---------
D1  **Posterior predictive check (PPC).** Draw K theta ~ posterior(.|x_obs),
    fold each through the SAME response to get noiseless model counts lambda_k,
    Poisson-realize K replicate spectra. Two discrepancy statistics:
      (a) chi2-like statistic on binned counts using the replicate-ensemble
          variance (so the null distribution is the replicate ensemble itself);
      (b) a KS-like distance between the observed and replicate CUMULATIVE count
          spectra -- the direct descendant of Buchner et al. (2014, A&A 564,
          A125) QQ-plot model-discovery methodology (cumulative-counts QQ plot).
    Score = a tail-probability-style statistic: the fraction of replicates whose
    own discrepancy is *less* extreme than the observed one (PPP-complement),
    so ~1 means the observed spectrum sits in the far tail of its own posterior
    predictive (suspicious) and ~0.5 means it is typical.

D2  **Embedding-space OOD.** Push spectra through the flow's trained CNN
    embedding net (the learned summary statistic). Reference distribution =
    embeddings of N_ref clean Model-A simulations at the SAME count level. Score
    = regularized Mahalanobis distance of the observed embedding from the clean
    reference cloud (primary), with a k-NN mean-distance variant also returned.
    Methodological ancestor: Schmitt et al. (2023, arXiv:2112.08866; IJCV
    extension 2024, arXiv:2406.03154), who proposed MMD / distance tests in the
    learned summary-statistic (embedding) space for SBI misspecification.

    **Scoping vs Schmitt.** Our D2 operates on the flow's
    *posterior-trained, un-regularized* CNN embedding -- a NEAR-SUFFICIENT summary
    learned for inference, NOT Schmitt's MMD-regularized, deliberately OVERCOMPLETE
    summary network trained specifically to surface misspecification. Schmitt's
    Eq. 12-13 give the consequence: a misspecification that PRESERVES the summary
    distribution is *provably invisible* to any test in that summary space. That is
    exactly why a sub-percent gain shift (B4) -- which the NPE folds into the
    continuum parameters, leaving the near-sufficient summary distribution
    essentially unchanged -- evades D2. So a D2 non-detection here scopes to the
    near-sufficient embedding; an MMD-regularized overcomplete summary space
    (Schmitt+23/24) is the natural next attempt, not refuted by this benchmark.

D3  **Simplified MARGINAL C2ST.** A single classifier per benchmark cell, trained
    with stratified k-fold CV on the EMBEDDING features to distinguish the
    clean-population from the misspecified-population. The cell statistic is the CV
    accuracy (0.5 = indistinguishable populations = undetectable misspecification;
    ->1 = perfectly separable); per-spectrum out-of-fold class-1 probabilities give
    each spectrum a suspicion score for ROC. It is labelled "simplified marginal"
    because the full *conditional* C2ST (Lopez-Paz & Oquab 2017; SBI-misspec use
    in the sbi toolkit and Schmitt+23/24) trains a fresh classifier per spectrum on
    its posterior-predictive replicates. We found that conditional form pathological
    against overconfident NPE posteriors (tight replicate clusters are trivially
    separable from the broad clean cloud for clean and misspec alike -- see
    RESULTS.md Phase 4), so we take the marginal population two-sample test instead.

Everything here is pure / importable; the benchmark CLI lives in
``scripts/run_detect_benchmark.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import train_npe as _tn


# ==========================================================================
# simulator context: everything a detector needs to (re)simulate
# ==========================================================================

@dataclass
class SimulatorContext:
    """Bundle of the simulator pieces a detector needs, all derived from a
    checkpoint's ``arch.json`` so a detector can run cold from a saved flow.

    Built once per (checkpoint) by :func:`context_from_checkpoint` and reused for
    every spectrum at that count level. Holds the (exposure-scaled) jaxspec
    ObsConfiguration, the base-model name + canonical parameter order, the prior
    config, and a lazily-built clean-reference embedding cache (D2/D3).
    """
    base_model: str
    param_names: list[str]
    prior_cfg: dict
    exposure_s: float
    response_name: str
    obsconf: object                      # jaxspec ObsConfiguration (exposure-scaled)
    n_channels: int
    # caches, keyed by (n_ref, seed) -> np.ndarray (n_ref, embed_dim)
    _ref_embed_cache: dict = field(default_factory=dict)

    def fold(self, theta: np.ndarray) -> np.ndarray:
        """Fold parameter vectors -> noiseless per-channel model counts (lambda).

        ``theta`` is ``(M, n_params)`` in ``param_names`` order (linear units).
        Returns ``(M, n_channels)`` float64. Lazy import keeps this module
        importable without jaxspec for toy unit tests.
        """
        from . import simulate as _sim
        return _sim.fold_theta(self.base_model, self.param_names, theta, self.obsconf)

    def clean_sim(self, n: int, seed: int):
        """Draw ``n`` clean Model-A (theta, Poisson-counts) from the training
        prior at this count level. Returns ``(theta (n,P), x (n,C))`` float arrays.
        """
        from . import simulate as _sim
        from . import priors as _priors
        rng = np.random.default_rng(seed)
        samples = _priors.sample_prior(self.prior_cfg, self.param_names, n, rng)
        theta = np.stack([np.asarray(samples[p]) for p in self.param_names], axis=1)
        lam = self.fold(theta)
        rng_p = np.random.default_rng(seed + 1)
        x = rng_p.poisson(np.clip(lam, 0.0, None)).astype(np.float64)
        return theta.astype(np.float64), x


def context_from_checkpoint(ckpt_dir, response_name: str | None = None):
    """Build a :class:`SimulatorContext` from a trained checkpoint directory.

    Reads ``arch.json`` for the base model, prior, exposure and response, then
    loads + exposure-scales the bundled XMM EPIC-pn ObsConfiguration. Lazy jaxspec
    import (so toy tests can build a context by hand instead).
    """
    from . import responses as _responses

    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "arch.json") as f:
        arch = json.load(f)
    resp_name = response_name or _responses.EXAMPLE_NAME
    base = _responses.load_base_obsconf(resp_name)
    obsconf = _responses.scale_exposure(base, float(arch["exposure_s"]))
    return SimulatorContext(
        base_model=arch["base_model"],
        param_names=list(arch["param_names"]),
        prior_cfg=arch["prior_cfg"],
        exposure_s=float(arch["exposure_s"]),
        response_name=resp_name,
        obsconf=obsconf,
        n_channels=int(arch["n_channels"]),
    )


# ==========================================================================
# embedding extraction (the flow's trained CNN summary statistic)
# ==========================================================================

def get_embedding_net(posterior):
    """Return the trained CNN embedding net living inside the flow.

    sbi 0.26 layout: ``DirectPosterior.posterior_estimator`` is an
    ``NFlowsFlow`` whose ``.embedding_net`` is exactly the ``SpectrumCNN`` that
    was trained jointly with the NSF (verified empirically against the production
    checkpoints; also reachable as ``.net._embedding_net``). We take the public
    ``.embedding_net`` attribute and fall back to ``.net._embedding_net``.
    """
    de = getattr(posterior, "posterior_estimator", None)
    if de is None:
        de = getattr(posterior, "_posterior_estimator", None)
    if de is None:
        raise AttributeError("could not locate the density estimator on posterior")
    if hasattr(de, "embedding_net"):
        return de.embedding_net
    if hasattr(de, "net") and hasattr(de.net, "_embedding_net"):
        return de.net._embedding_net
    raise AttributeError("could not locate the embedding net inside the flow")


def embed_spectra(posterior, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Map raw counts spectra ``(N, n_channels)`` -> embeddings ``(N, embed_dim)``.

    The ``SpectrumCNN`` applies its ``log1p`` normalization internally, so raw
    counts go in. Runs under ``no_grad`` in eval mode -> deterministic.
    """
    emb = get_embedding_net(posterior)
    emb.eval()
    x_t = torch.as_tensor(np.atleast_2d(np.asarray(x, dtype=np.float32)), device=device)
    with torch.no_grad():
        e = emb(x_t)
    return e.detach().cpu().numpy()


# ==========================================================================
# posterior-predictive replicate machinery (shared by D1 + D3)
# ==========================================================================

def posterior_predictive_replicates(
    posterior,
    x_obs: np.ndarray,
    ctx: SimulatorContext,
    k: int,
    seed: int = 0,
    device: str = "cpu",
    max_sampling_time: float = 20.0,
):
    """Draw K posterior-predictive replicate spectra for one observed spectrum.

    1. theta_k ~ q_NPE(. | x_obs), k = 1..K  (reject outside prior).
    2. lambda_k = fold(theta_k) through the SAME response (noiseless model counts).
    3. x_rep_k ~ Poisson(lambda_k).

    Returns ``(x_rep (K,C) float, lam (K,C) float, theta (K,P) float)``.

    ``max_sampling_time`` caps the rejection-sampling wall time: for a *strongly
    misspecified* spectrum the flow's posterior mass can fall largely OUTSIDE the
    training prior box, so ``reject_outside_prior=True`` would loop slowly. We pass
    sbi's ``max_sampling_time`` so the draw returns promptly; if rejection cannot
    fill K samples in time we fall back to a single unrejected batch and clip into
    the prior box (a confused-posterior spectrum is exactly the suspicious case the
    detectors should flag, so a clipped draw is acceptable for the replicate null).
    """
    torch.manual_seed(seed)
    x_t = torch.as_tensor(np.asarray(x_obs, dtype=np.float32), device=device)
    try:
        theta_t = posterior.sample(
            (k,), x=x_t, show_progress_bars=False, reject_outside_prior=True,
            max_sampling_time=float(max_sampling_time),
        )
    except (RuntimeError, ValueError, TypeError):
        # fall back to un-rejected draws clipped into the prior box
        theta_t = posterior.sample(
            (k,), x=x_t, show_progress_bars=False, reject_outside_prior=False,
        )
    theta = theta_t.detach().cpu().numpy().astype(np.float64)
    if theta.shape[0] < k:  # rejection timed out before filling K -> top up & clip
        extra = posterior.sample(
            (k - theta.shape[0],), x=x_t, show_progress_bars=False,
            reject_outside_prior=False,
        ).detach().cpu().numpy().astype(np.float64)
        theta = np.vstack([theta, extra])
    from . import priors as _priors
    lo, hi = _priors.prior_bounds(ctx.prior_cfg, ctx.param_names)
    theta = np.clip(theta, lo[None, :], hi[None, :])
    lam = np.asarray(ctx.fold(theta), dtype=np.float64)
    rng = np.random.default_rng(seed + 12345)
    x_rep = rng.poisson(np.clip(lam, 0.0, None)).astype(np.float64)
    return x_rep, lam, theta


# ==========================================================================
# D1: posterior predictive check (PPC)
# ==========================================================================

def _chi2_stat(counts: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """chi2-like statistic sum_c (counts_c - mean_c)^2 / var_c, per row.

    ``counts`` is ``(..., C)``; ``mean``/``var`` are ``(C,)`` (the replicate-ensemble
    moments). Variance is floored to avoid divide-by-zero in empty channels.
    """
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    var = np.clip(np.asarray(var, dtype=np.float64), 1e-6, None)
    return np.sum((counts - mean) ** 2 / var, axis=1)


def _ks_cumulative_stat(counts: np.ndarray, ref_cum_frac: np.ndarray) -> np.ndarray:
    """KS-like sup-distance between a spectrum's cumulative count fraction and a
    reference cumulative fraction (the Buchner QQ-plot descendant).

    For each row we build the cumulative counts along the channel axis, normalize
    to a fraction in [0,1] (the empirical CDF over channel index, weighted by
    counts), and take the sup |F_row(c) - F_ref(c)| over channels.

    ``counts``       : ``(..., C)``
    ``ref_cum_frac`` : ``(C,)`` reference cumulative fraction.
    Returns ``(...,)``.
    """
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    tot = counts.sum(axis=1, keepdims=True)
    tot = np.where(tot <= 0, 1.0, tot)
    cum = np.cumsum(counts, axis=1) / tot
    return np.max(np.abs(cum - ref_cum_frac[None, :]), axis=1)


def detect_d1_ppc(
    x_obs: np.ndarray,
    posterior,
    ctx: SimulatorContext,
    k: int = 200,
    seed: int = 0,
    device: str = "cpu",
    return_parts: bool = False,
):
    r"""D1 posterior predictive check.

    Builds K posterior-predictive replicates, then forms two discrepancy
    statistics whose null distribution is the replicate ensemble itself:

      * **chi2** vs the replicate-ensemble mean/variance per channel;
      * **KS-on-cumulative**: sup-distance of the cumulative count spectrum from
        the replicate-mean cumulative spectrum (Buchner et al. 2014 QQ-plot
        descendant).

    For each statistic the posterior-predictive p-value-style score is

        score = #{ T(x_rep_k) < T(x_obs) } / K          (PPP-complement)

    i.e. the fraction of replicates LESS extreme than the observation. ~0.5 means
    the observation is a typical draw from its own posterior predictive (well
    specified); ->1 means it sits in the far tail (misspecified -> suspicious).
    The combined D1 score is the max of the two (a spectrum is suspicious if it
    fails EITHER check). Higher = more suspicious.
    """
    x_obs = np.asarray(x_obs, dtype=np.float64).reshape(-1)
    x_rep, lam, theta = posterior_predictive_replicates(
        posterior, x_obs, ctx, k=k, seed=seed, device=device
    )
    rep_mean = x_rep.mean(axis=0)
    rep_var = x_rep.var(axis=0)

    # chi2 against the replicate ensemble moments
    t_obs_chi2 = _chi2_stat(x_obs, rep_mean, rep_var)[0]
    t_rep_chi2 = _chi2_stat(x_rep, rep_mean, rep_var)
    score_chi2 = float(np.mean(t_rep_chi2 < t_obs_chi2))

    # KS-on-cumulative against the replicate-mean cumulative fraction
    ref_tot = rep_mean.sum()
    ref_cum = np.cumsum(rep_mean) / (ref_tot if ref_tot > 0 else 1.0)
    t_obs_ks = _ks_cumulative_stat(x_obs, ref_cum)[0]
    t_rep_ks = _ks_cumulative_stat(x_rep, ref_cum)
    score_ks = float(np.mean(t_rep_ks < t_obs_ks))

    score = max(score_chi2, score_ks)
    if return_parts:
        return score, {
            "d1_chi2": score_chi2,
            "d1_ks": score_ks,
            "t_obs_chi2": float(t_obs_chi2),
            "t_obs_ks": float(t_obs_ks),
        }
    return score


# ==========================================================================
# D2: embedding-space OOD (Mahalanobis + kNN)  -- Schmitt+23/24
# ==========================================================================

@dataclass
class EmbeddingReference:
    """Clean Model-A reference cloud in embedding space, with the Mahalanobis
    machinery precomputed (mean, regularized inverse covariance) and the raw
    reference embeddings kept for the k-NN variant."""
    mean: np.ndarray                 # (embed_dim,)
    cov_inv: np.ndarray              # (embed_dim, embed_dim) regularized
    ref: np.ndarray                  # (n_ref, embed_dim)
    n_ref: int
    embed_dim: int

    @classmethod
    def fit(cls, ref_embed: np.ndarray, reg: float = 1e-3):
        """Fit from clean reference embeddings. ``reg`` shrinks the covariance
        toward a scaled identity (Ledoit-Wolf-style ridge) so the inverse is
        well-conditioned even when n_ref is small or features are collinear."""
        ref = np.atleast_2d(np.asarray(ref_embed, dtype=np.float64))
        n_ref, d = ref.shape
        mean = ref.mean(axis=0)
        cov = np.cov(ref, rowvar=False)
        cov = np.atleast_2d(cov)
        # ridge toward scaled identity
        trace_mean = np.trace(cov) / d if d > 0 else 1.0
        cov_reg = (1.0 - reg) * cov + reg * trace_mean * np.eye(d)
        cov_inv = np.linalg.pinv(cov_reg)
        return cls(mean=mean, cov_inv=cov_inv, ref=ref, n_ref=int(n_ref),
                   embed_dim=int(d))

    def mahalanobis(self, e: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of each embedding row to the reference mean."""
        e = np.atleast_2d(np.asarray(e, dtype=np.float64))
        diff = e - self.mean[None, :]
        m2 = np.einsum("ij,jk,ik->i", diff, self.cov_inv, diff)
        return np.sqrt(np.clip(m2, 0.0, None))

    def knn_distance(self, e: np.ndarray, kk: int = 5) -> np.ndarray:
        """Mean Euclidean distance to the ``kk`` nearest reference embeddings."""
        e = np.atleast_2d(np.asarray(e, dtype=np.float64))
        # (M, n_ref) pairwise distances
        d2 = ((e[:, None, :] - self.ref[None, :, :]) ** 2).sum(axis=2)
        d = np.sqrt(np.clip(d2, 0.0, None))
        kk = min(kk, self.ref.shape[0])
        nn = np.partition(d, kk - 1, axis=1)[:, :kk]
        return nn.mean(axis=1)


def build_embedding_reference(
    posterior,
    ctx: SimulatorContext,
    n_ref: int = 500,
    seed: int = 0,
    reg: float = 1e-3,
    device: str = "cpu",
) -> EmbeddingReference:
    """Simulate ``n_ref`` clean Model-A spectra at this count level, embed them,
    and fit the reference cloud. Cached on ``ctx`` keyed by (n_ref, seed)."""
    key = ("embed", int(n_ref), int(seed))
    if key in ctx._ref_embed_cache:
        ref_embed = ctx._ref_embed_cache[key]
    else:
        _, x_clean = ctx.clean_sim(n_ref, seed=seed)
        ref_embed = embed_spectra(posterior, x_clean, device=device)
        ctx._ref_embed_cache[key] = ref_embed
    return EmbeddingReference.fit(ref_embed, reg=reg)


def detect_d2_embedding(
    x_obs: np.ndarray,
    posterior,
    ctx: SimulatorContext,
    reference: EmbeddingReference,
    knn_k: int = 5,
    device: str = "cpu",
    return_parts: bool = False,
):
    """D2 embedding-space OOD score. Both the Schmitt+23/24 variants are computed:

      * **k-NN mean distance** to the clean reference cloud (the PRIMARY score);
      * **regularized Mahalanobis** distance to the cloud mean (secondary).

    Higher = more suspicious. The reference is reused across many spectra (build it
    once per count level with :func:`build_embedding_reference`).

    Why k-NN is primary (empirical, see RESULTS.md Phase 4): the clean reference
    cloud is dominated by a single high-variance brightness axis (the log-uniform
    norm prior spans decades of total counts). The Mahalanobis whitening inverts
    that covariance and *amplifies* the low-variance noise directions, washing out
    the misspecification signal; the raw-Euclidean k-NN distance is far more
    sensitive to an embedding displaced off the clean manifold by an unmodeled
    feature. The spec wording ("Mahalanobis distance *and/or* k-NN") permits this;
    both are stored so the benchmark can report either.
    """
    e = embed_spectra(posterior, np.asarray(x_obs).reshape(1, -1), device=device)
    maha = float(reference.mahalanobis(e)[0])
    knn = float(reference.knn_distance(e, kk=knn_k)[0])
    if return_parts:
        return knn, {"d2_knn": knn, "d2_mahalanobis": maha}
    return knn


# ==========================================================================
# D3: simplified MARGINAL C2ST on embedding features
# ==========================================================================
#
# Why "marginal" and not the per-spectrum conditional C2ST: the full conditional
# C2ST (Lopez-Paz & Oquab 2017; used for SBI misspecification by e.g. the sbi
# toolkit and Schmitt+23/24) trains a fresh classifier *per observed spectrum* to
# tell that spectrum's posterior-predictive replicates from clean simulations and
# reports a calibrated per-spectrum statistic. We tried that against the
# production flows and it is PATHOLOGICAL here (RESULTS.md Phase 4): an
# overconfident NPE posterior (the very failure mode Phase 2/3 documents) yields a
# *tight* replicate cluster that is trivially separable from the broad clean
# reference cloud for clean AND misspecified spectra alike, so the per-spectrum
# C2ST carries no misspecification signal. We therefore implement the
# simplified MARGINAL version, and label it as such:
#
#   D3 = a single classifier per benchmark cell, trained to separate the
#   clean-population embeddings from the misspecified-population embeddings, with
#   k-fold cross-validation. The cell-level C2ST statistic is the CV accuracy
#   (0.5 = indistinguishable populations = undetectable misspecification; ->1 =
#   perfectly separable). Per-spectrum class-1 probabilities (held-out via the CV
#   folds) feed the same ROC machinery as D1/D2 so D3 has a per-spectrum score too.
#
# This is a population two-sample test, by construction supervised on the cell's
# clean/misspec labels -- that is exactly the C2ST definition (measure how
# distinguishable two samples are), NOT a blind per-spectrum deployment, and the
# benchmark's job is precisely to report that distinguishability per cell.


def _make_c2st_clf(kind: str, seed: int):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=500, C=0.5, random_state=seed)
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        est = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300,
                            random_state=seed)
    else:
        raise ValueError(f"unknown C2ST classifier kind '{kind}'")
    return make_pipeline(StandardScaler(), est)


@dataclass
class MarginalC2STResult:
    """Output of the simplified marginal C2ST for one benchmark cell.

    ``cv_accuracy`` is the cross-validated clean-vs-misspec classification accuracy
    (the C2ST statistic; 0.5 = indistinguishable). ``clean_proba`` / ``mis_proba``
    are the per-spectrum held-out class-1 (=misspecified) probabilities for the
    clean and misspecified spectra respectively -- these are the per-spectrum D3
    scores fed to :func:`roc_auc`. Higher proba = more suspicious.
    """
    cv_accuracy: float
    clean_proba: np.ndarray
    mis_proba: np.ndarray
    n_clean: int
    n_mis: int


def marginal_c2st(
    clean_embed: np.ndarray,
    mis_embed: np.ndarray,
    kind: str = "logreg",
    n_splits: int = 5,
    seed: int = 0,
) -> MarginalC2STResult:
    """Simplified marginal C2ST between a clean-population and a
    misspecified-population embedding sample.

    Fits a classifier with stratified k-fold CV; returns the CV accuracy (C2ST
    statistic) and the held-out (out-of-fold) class-1 probability for every
    spectrum, so each spectrum gets a per-spectrum suspicion score with no
    train/test leakage. ``n_splits`` is clamped to the smaller class size.
    """
    from sklearn.model_selection import StratifiedKFold

    clean_embed = np.atleast_2d(np.asarray(clean_embed, dtype=np.float64))
    mis_embed = np.atleast_2d(np.asarray(mis_embed, dtype=np.float64))
    n_c, n_m = clean_embed.shape[0], mis_embed.shape[0]
    X = np.vstack([clean_embed, mis_embed])
    y = np.concatenate([np.zeros(n_c), np.ones(n_m)])

    n_splits = int(max(2, min(n_splits, n_c, n_m)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(X.shape[0], np.nan)
    accs = []
    for tr, te in skf.split(X, y):
        clf = _make_c2st_clf(kind, seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        classes = list(clf.classes_) if hasattr(clf, "classes_") \
            else list(clf.steps[-1][1].classes_)
        idx = classes.index(1.0) if 1.0 in classes else (p.shape[1] - 1)
        oof[te] = p[:, idx]
        accs.append(float(((p[:, idx] >= 0.5).astype(float) == y[te]).mean()))

    cv_accuracy = float(np.mean(accs))
    clean_proba = oof[:n_c]
    mis_proba = oof[n_c:]
    return MarginalC2STResult(
        cv_accuracy=cv_accuracy, clean_proba=clean_proba, mis_proba=mis_proba,
        n_clean=n_c, n_mis=n_m,
    )


def detect_d3_c2st_cell(
    posterior,
    ctx: SimulatorContext,
    x_clean: np.ndarray,
    x_mis: np.ndarray,
    kind: str = "logreg",
    n_splits: int = 5,
    seed: int = 0,
    device: str = "cpu",
) -> MarginalC2STResult:
    """Cell-level D3: embed the clean and misspecified populations, run the
    marginal C2ST, return its result (CV accuracy + per-spectrum held-out
    probabilities). This is the form the benchmark calls (one classifier per cell,
    not one per spectrum)."""
    ec = embed_spectra(posterior, np.atleast_2d(x_clean), device=device)
    em = embed_spectra(posterior, np.atleast_2d(x_mis), device=device)
    return marginal_c2st(ec, em, kind=kind, n_splits=n_splits, seed=seed)


# ==========================================================================
# unified interface
# ==========================================================================

DETECTORS = ("D1", "D2", "D3")


def score(
    spectrum: np.ndarray,
    posterior,
    simulator_ctx: SimulatorContext,
    detector: str,
    *,
    reference: EmbeddingReference | None = None,
    reference_embed: np.ndarray | None = None,
    cfg: dict | None = None,
    seed: int = 0,
    device: str = "cpu",
    return_parts: bool = False,
):
    """Unified PER-SPECTRUM detector interface for ``detector`` in {"D1","D2"}.

    Higher score => more suspicious. ``cfg`` (optional) supplies per-detector
    hyperparameters; sensible defaults are used otherwise. For D2 the caller should
    pass the prebuilt clean ``reference`` so it is fit ONCE per count level, not per
    spectrum (the benchmark does this).

    **D3 is NOT a per-spectrum detector** -- it is the simplified MARGINAL C2ST,
    which operates on whole clean/misspecified POPULATIONS per benchmark cell. Call
    :func:`detect_d3_c2st_cell` (clean+misspec embeddings -> CV accuracy +
    per-spectrum held-out probabilities) instead; calling ``score(..., "D3")``
    raises, by design, to prevent the leaky per-spectrum misuse.
    """
    cfg = cfg or {}
    if detector == "D1":
        d1 = cfg.get("d1", {})
        return detect_d1_ppc(
            spectrum, posterior, simulator_ctx,
            k=int(d1.get("k", 200)), seed=seed, device=device,
            return_parts=return_parts,
        )
    if detector == "D2":
        d2 = cfg.get("d2", {})
        if reference is None:
            reference = build_embedding_reference(
                posterior, simulator_ctx,
                n_ref=int(d2.get("n_ref", 500)), seed=seed,
                reg=float(d2.get("reg", 1e-3)), device=device,
            )
        return detect_d2_embedding(
            spectrum, posterior, simulator_ctx, reference,
            knn_k=int(d2.get("knn_k", 5)), device=device,
            return_parts=return_parts,
        )
    if detector == "D3":
        raise ValueError(
            "D3 is the simplified MARGINAL C2ST (a population two-sample test); it "
            "has no single-spectrum score. Use detect_d3_c2st_cell(posterior, ctx, "
            "x_clean, x_mis) per benchmark cell."
        )
    raise ValueError(f"unknown detector '{detector}'. known: {DETECTORS}")


# ==========================================================================
# ROC / AUC analysis (used by the benchmark + reusable in tests)
# ==========================================================================

def roc_auc(clean_scores: np.ndarray, misspec_scores: np.ndarray):
    """ROC curve + AUC for a detector, given clean (negative) and misspecified
    (positive) scores. Higher score = more suspicious = more "positive".

    Returns ``(fpr, tpr, auc)``. AUC is computed via the Mann-Whitney U identity
    (mean rank of positives), which is exact and tie-aware and needs no sklearn.
    """
    clean = np.asarray(clean_scores, dtype=np.float64)
    miss = np.asarray(misspec_scores, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    miss = miss[np.isfinite(miss)]
    n_neg, n_pos = clean.size, miss.size
    if n_neg == 0 or n_pos == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    # AUC via Mann-Whitney U with tie correction (rank of positives)
    allv = np.concatenate([miss, clean])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(counts.size)
    np.add.at(sum_ranks, inv, ranks)
    avg_ranks = sum_ranks / counts
    ranks = avg_ranks[inv]
    sum_pos = ranks[:n_pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # ROC curve by sweeping thresholds
    thresh = np.unique(allv)[::-1]
    tpr = np.array([(miss >= t).mean() for t in thresh])
    fpr = np.array([(clean >= t).mean() for t in thresh])
    # prepend (0,0), append (1,1)
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    return fpr, tpr, float(auc)
