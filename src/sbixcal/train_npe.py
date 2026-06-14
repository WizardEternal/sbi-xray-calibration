"""NPE training for sbi-xray-calibration (Phase 2).

Trains a Neural Posterior Estimator (NPE) with a Neural Spline Flow (NSF) and a
small 1-D CNN embedding net over the binned EPIC-pn counts spectrum, on the
Phase-1 simulated datasets (data/sim/*.npz). Config-driven; one trained flow
PER count level (NOT amortized over exposure -- see the config headers and
RESULTS.md: amortizing over exposure is a confound for a calibration analysis,
and fixed-exposure flows match Barret & Dupourque's setup).

Key design points
------------------
* Embedding net (`SpectrumCNN`): input is the ~102-channel counts spectrum.
  Counts are normalized with ``log1p`` (counts span ~0 to thousands and are
  Poisson; log1p compresses the dynamic range, is defined at 0, and keeps the
  shot-noise structure roughly homoscedastic). 2 conv layers + an MLP head,
  ~tens of k params.
* Flow: NSF via ``posterior_nn(model="nsf", embedding_net=cnn)``.
* Prior: BoxUniform built from the YAML prior bounds (log-uniform params get
  *linear* box bounds -- the flow models theta in linear units, matching how the
  npz stores theta; this is fine for a bounded NPE and matches Phase-1 storage).
* Checkpoints are cold-loadable: ``load_posterior(<dir>)`` rebuilds the exact
  architecture from the saved config, loads the flow state_dict, and returns a
  ready-to-sample DirectPosterior. Loading needs no training data.

CLI lives in ``scripts/run_train_npe.py``.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import yaml

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from . import models as _models
from . import priors as _priors


# ==========================================================================
# paths
# ==========================================================================

def _repo_root() -> Path:
    # src/sbixcal/train_npe.py -> repo root is three parents up
    return Path(__file__).resolve().parents[2]


def _sim_path(name: str) -> Path:
    return _repo_root() / "data" / "sim" / f"{name}.npz"


def _models_dir() -> Path:
    return _repo_root() / "outputs" / "models"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ==========================================================================
# embedding net: 1-D CNN over the counts spectrum
# ==========================================================================

class SpectrumCNN(nn.Module):
    """Small 1-D CNN embedding for a binned counts spectrum.

    Forward expects ``x`` of shape ``(batch, n_channels)`` (raw counts). Inside,
    counts are normalized with ``log1p`` and treated as a single-channel 1-D
    signal of length ``n_channels``. Two conv blocks (Conv1d + ReLU + pooling)
    followed by an MLP head produce an ``embed_dim`` feature vector that the NSF
    flow conditions on.

    The ``log1p`` normalization is applied *inside* the module so it travels with
    the saved weights -- a cold-loaded posterior normalizes inputs identically
    without the caller having to remember to do it.
    """

    def __init__(
        self,
        n_channels: int = 102,
        embed_dim: int = 16,
        conv_channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 5,
        mlp_hidden: int = 64,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.embed_dim = int(embed_dim)

        pad = kernel_size // 2
        layers = []
        in_ch = 1
        length = self.n_channels
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
            length = length // 2
        self.conv = nn.Sequential(*layers)
        self._flat = in_ch * length
        self.head = nn.Sequential(
            nn.Linear(self._flat, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, self.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels) raw counts. sbi may pass a leading sample dim;
        # flatten everything but the last (channel) axis to a batch, run, restore.
        orig_shape = x.shape
        x = x.reshape(-1, self.n_channels)
        x = torch.log1p(torch.clamp(x, min=0.0))   # Poisson counts -> log1p
        x = x.unsqueeze(1)                          # (b, 1, n_channels)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.head(x)
        if len(orig_shape) > 2:
            x = x.reshape(*orig_shape[:-1], self.embed_dim)
        return x


def build_embedding_net(cfg: dict, n_channels: int) -> SpectrumCNN:
    emb = cfg.get("embedding", {}) or {}
    return SpectrumCNN(
        n_channels=n_channels,
        embed_dim=int(emb.get("embed_dim", 16)),
        conv_channels=tuple(emb.get("conv_channels", (16, 32))),
        kernel_size=int(emb.get("kernel_size", 5)),
        mlp_hidden=int(emb.get("mlp_hidden", 64)),
    )


# ==========================================================================
# prior + dataset
# ==========================================================================

def build_prior(prior_cfg: dict, param_order, device: str = "cpu") -> BoxUniform:
    """A BoxUniform over the *linear* prior bounds, in ``param_order``.

    theta is stored in the npz in linear units (the loguniform params are
    sampled in log space but stored linearly), so the box must be the linear
    [low, high]. The flow's z-scoring handles the scale difference.
    """
    low, high = _priors.prior_bounds(prior_cfg, param_order)
    return BoxUniform(
        low=torch.as_tensor(low, dtype=torch.float32, device=device),
        high=torch.as_tensor(high, dtype=torch.float32, device=device),
    )


def load_dataset(name: str, max_n: int | None = None):
    """Load a Phase-1 npz -> (theta, x, param_names, meta).

    theta: (n, n_params) float32; x: (n, n_channels) float32 counts.
    If ``max_n`` is given, the first ``max_n`` rows are returned (the rows are
    already in a deterministic, seed-derived order, so a prefix is a valid
    fixed-seed subsample for learning curves).
    """
    path = _sim_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset '{name}' not found at {path}. Generate it first with "
            f"`python -m sbixcal.simulate --config <sim config>`."
        )
    d = np.load(path, allow_pickle=True)
    theta = np.asarray(d["theta"], dtype=np.float32)
    x = np.asarray(d["x"], dtype=np.float32)
    if max_n is not None:
        theta = theta[:max_n]
        x = x[:max_n]
    param_names = [str(p) for p in d["param_names"]]
    meta = {
        "exposure_s": float(d["exposure_s"]),
        "seed": int(d["seed"]),
        "median_total_counts": float(d.get("median_total_counts", np.median(x.sum(1)))),
        "n": int(theta.shape[0]),
    }
    return (
        torch.from_numpy(theta),
        torch.from_numpy(x),
        param_names,
        meta,
    )


# ==========================================================================
# flow builder
# ==========================================================================

def build_density_estimator(cfg: dict, n_channels: int):
    """Return a `posterior_nn` builder (NSF flow + CNN embedding) from config.

    ``z_score_x="none"``: the SpectrumCNN already applies ``log1p`` normalization
    internally, so sbi's outer z-scoring of the *raw* heavy-tailed Poisson counts
    is both redundant and harmful (raw counts have extreme outliers under the
    log-uniform norm prior, which the z-scorer warns about and which causes
    precision loss). theta IS z-scored (``z_score_theta="independent"``, the
    default) -- that is cheap and helps the flow.
    """
    flow = cfg.get("flow", {}) or {}
    embedding_net = build_embedding_net(cfg, n_channels)
    return posterior_nn(
        model="nsf",
        embedding_net=embedding_net,
        z_score_x="none",
        hidden_features=int(flow.get("hidden_features", 50)),
        num_transforms=int(flow.get("num_transforms", 5)),
        num_bins=int(flow.get("num_bins", 10)),
    )


# ==========================================================================
# training
# ==========================================================================

def train_one_flow(
    theta: torch.Tensor,
    x: torch.Tensor,
    prior: BoxUniform,
    cfg: dict,
    device: str,
    seed: int = 0,
):
    """Train a single NPE+NSF flow with a CNN embedding net.

    Returns ``(density_estimator, inference, summary)`` where ``summary`` is the
    sbi training summary (training/validation loss curves etc.).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_channels = x.shape[1]
    train_cfg = cfg.get("train", {}) or {}

    estimator = build_density_estimator(cfg, n_channels)
    inference = NPE(
        prior=prior,
        density_estimator=estimator,
        device=device,
        show_progress_bars=bool(train_cfg.get("show_progress", False)),
    )
    inference.append_simulations(theta.to(device), x.to(device))

    de = inference.train(
        training_batch_size=int(train_cfg.get("batch_size", 200)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-4)),
        validation_fraction=float(train_cfg.get("validation_fraction", 0.1)),
        stop_after_epochs=int(train_cfg.get("stop_after_epochs", 20)),
        max_num_epochs=int(train_cfg.get("max_num_epochs", 2**31 - 1)),
        show_train_summary=False,
    )
    return de, inference, dict(inference.summary)


# ==========================================================================
# checkpoint save / load
# ==========================================================================

def save_checkpoint(
    out_dir: Path,
    density_estimator,
    summary: dict,
    cfg: dict,
    param_names: list[str],
    n_channels: int,
    meta: dict,
    config_src_path: str | None = None,
):
    """Save flow state + config + loss curves to ``out_dir`` so the posterior is
    reloadable cold (no training data needed).

    Writes:
      flow_state.pt        torch state_dict of the density estimator
      arch.json            everything load_posterior needs to rebuild the net
      training_loss.npy    per-epoch training loss
      validation_loss.npy  per-epoch validation loss
      summary.json         full sbi summary + run metadata
      config.yaml          a copy of the source config (if given)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(density_estimator.state_dict(), out_dir / "flow_state.pt")

    arch = {
        "param_names": list(param_names),
        "n_channels": int(n_channels),
        "n_params": len(param_names),
        "flow": cfg.get("flow", {}) or {},
        "embedding": cfg.get("embedding", {}) or {},
        "prior_cfg": cfg["priors"],
        "base_model": cfg["base_model"],
        "exposure_s": meta.get("exposure_s"),
        "median_total_counts": meta.get("median_total_counts"),
    }
    with open(out_dir / "arch.json", "w") as f:
        json.dump(arch, f, indent=2)

    tl = np.asarray(summary.get("training_loss", []), dtype=np.float64)
    vl = np.asarray(summary.get("validation_loss", []), dtype=np.float64)
    np.save(out_dir / "training_loss.npy", tl)
    np.save(out_dir / "validation_loss.npy", vl)

    summary_out = {
        "epochs_trained": summary.get("epochs_trained"),
        "best_validation_loss": summary.get("best_validation_loss"),
        "final_training_loss": float(tl[-1]) if tl.size else None,
        "final_validation_loss": float(vl[-1]) if vl.size else None,
        "n_train": meta.get("n"),
        "exposure_s": meta.get("exposure_s"),
        "median_total_counts": meta.get("median_total_counts"),
        "param_names": list(param_names),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    if config_src_path and Path(config_src_path).exists():
        shutil.copy(config_src_path, out_dir / "config.yaml")

    return out_dir


def load_posterior(model_dir: str | Path, device: str = "cpu"):
    """Cold-load a trained flow from ``model_dir`` and return a DirectPosterior.

    Rebuilds the exact architecture from ``arch.json``, loads ``flow_state.pt``,
    and wraps it in a DirectPosterior. No training data required. The returned
    posterior is in ``eval()`` mode (so same-seed sampling is deterministic).

    Returns ``(posterior, info)`` where ``info`` carries param_names, n_channels,
    exposure, etc.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "arch.json", "r") as f:
        arch = json.load(f)

    n_channels = int(arch["n_channels"])
    param_names = list(arch["param_names"])
    n_params = len(param_names)

    # rebuild the builder with the same architecture
    cfg = {"flow": arch.get("flow", {}), "embedding": arch.get("embedding", {})}
    estimator_builder = build_density_estimator(cfg, n_channels)

    # the builder constructs the net from a sample (theta, x) batch; the shapes
    # (and z-score stats) are then overwritten by the loaded state_dict, so a
    # tiny dummy batch of the right shape is all that's needed.
    dummy_theta = torch.zeros(2, n_params, dtype=torch.float32)
    dummy_x = torch.zeros(2, n_channels, dtype=torch.float32)
    de = estimator_builder(dummy_theta, dummy_x)
    state = torch.load(model_dir / "flow_state.pt", map_location=device)
    de.load_state_dict(state)
    de.to(device)
    de.eval()

    prior = build_prior(arch["prior_cfg"], param_names, device=device)
    inference = NPE(prior=prior, density_estimator="nsf", device=device,
                    show_progress_bars=False)
    posterior = inference.build_posterior(de, prior=prior)

    info = {
        "param_names": param_names,
        "n_channels": n_channels,
        "exposure_s": arch.get("exposure_s"),
        "median_total_counts": arch.get("median_total_counts"),
        "base_model": arch.get("base_model"),
    }
    return posterior, info


# ==========================================================================
# validation helpers (pure functions; used by scripts/run_validate_npe.py)
# ==========================================================================

def credible_interval(samples: np.ndarray, cred: float = 0.90):
    """Per-parameter equal-tailed credible interval from posterior samples.

    ``samples`` is ``(n_samples, n_params)``. Returns ``(lo, hi, median)`` each
    ``(n_params,)`` arrays: the ``cred`` equal-tailed interval and the median.
    """
    qlo = (1.0 - cred) / 2.0 * 100.0
    qhi = (1.0 + cred) / 2.0 * 100.0
    lo = np.percentile(samples, qlo, axis=0)
    hi = np.percentile(samples, qhi, axis=0)
    med = np.median(samples, axis=0)
    return lo, hi, med


def coverage_fraction(truth: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Fraction of test cases whose truth lies inside [lo, hi].

    All arrays are ``(n_test, n_params)``. Returns ``(per_param, joint)`` where
    ``per_param`` is ``(n_params,)`` (marginal coverage) and ``joint`` is the
    scalar fraction with the truth inside the interval for ALL parameters.
    """
    inside = (truth >= lo) & (truth <= hi)
    return inside.mean(axis=0), float(inside.all(axis=1).mean())


# ==========================================================================
# top-level run: one flow per count level
# ==========================================================================

def run_training(cfg: dict, config_src_path: str | None = None,
                 device: str | None = None, force: bool = False,
                 only_level: str | None = None):
    """Train one flow per count level named in the config. Skip-if-exists per
    level unless ``force``. Returns a dict {level_name: out_dir}."""
    device = device or cfg.get("device", "cpu")
    base_model = cfg["base_model"]
    param_order = _models.MODEL_PARAMS[base_model]
    seed = int(cfg.get("seed", 0))

    out_roots = {}
    for level in cfg["levels"]:
        lname = level["name"]
        if only_level and lname != only_level:
            continue
        run_name = f"{cfg['name']}_{lname}"
        out_dir = _models_dir() / run_name
        if (out_dir / "flow_state.pt").exists() and not force:
            print(f"[skip] {run_name} (flow_state.pt exists)")
            out_roots[lname] = out_dir
            continue

        dataset_name = level["dataset"]
        max_n = level.get("n_train")
        theta, x, names, meta = load_dataset(dataset_name, max_n=max_n)
        if names != param_order:
            raise ValueError(
                f"param order mismatch for {dataset_name}: {names} != {param_order}"
            )
        prior = build_prior(cfg["priors"], param_order, device=device)

        t0 = time.perf_counter()
        de, _, summary = train_one_flow(
            theta, x, prior, cfg, device=device, seed=seed + level.get("seed_offset", 0)
        )
        dt = time.perf_counter() - t0
        meta["train_wall_s"] = dt

        save_checkpoint(out_dir, de, summary, cfg, param_order, x.shape[1], meta,
                        config_src_path=config_src_path)
        tl = summary.get("training_loss", [])
        vl = summary.get("validation_loss", [])
        print(f"[done] {run_name}: n={meta['n']} epochs={summary.get('epochs_trained')} "
              f"final_val={vl[-1] if vl else float('nan'):.4f} "
              f"best_val={float(np.min(vl)) if vl else float('nan'):.4f} "
              f"wall={dt:.1f}s -> {out_dir}")
        out_roots[lname] = out_dir
    return out_roots
