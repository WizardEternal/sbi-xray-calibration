"""Config-driven spectral simulator for sbi-xray-calibration (Phase 1).

Generates Poisson-sampled X-ray spectra for the well-specified Model A (dev /
production) at three exposure levels, from a YAML config + global seed, and
saves to data/sim/<name>.npz with skip-if-exists (crash-resumable).

CLI:
    python -m sbixcal.simulate --config configs/sim_modelA_dev.yaml
    python -m sbixcal.simulate --config configs/sim_modelA_dev.yaml --calibrate-exposure
    python -m sbixcal.simulate --config configs/sim_modelA_dev.yaml --level bright
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from jaxspec.data.util import fakeit_for_multiple_parameters

from . import models as _models
from . import priors as _priors
from . import responses as _responses


# --------------------------------------------------------------------------
# config IO
# --------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _repo_root() -> Path:
    # src/sbixcal/simulate.py -> repo root is three parents up
    return Path(__file__).resolve().parents[2]


def _resolve_data_path(name: str) -> Path:
    return _repo_root() / "data" / "sim" / f"{name}.npz"


# --------------------------------------------------------------------------
# core simulation
# --------------------------------------------------------------------------

def simulate_spectra(
    base_model_name: str,
    prior_cfg: dict,
    obsconf,
    n: int,
    rng: np.random.Generator,
    apply_poisson: bool = True,
    seed_for_fakeit: int = 0,
):
    """Draw `n` parameter sets from the prior and fold them through `obsconf`.

    Returns (theta, x, param_order) where theta is (n, n_params) in param_order
    and x is (n, n_channels) of (Poisson-sampled) counts.
    """
    param_order = _models.MODEL_PARAMS[base_model_name]
    model = _models.build_model(base_model_name)

    samples = _priors.sample_prior(prior_cfg, param_order, n, rng)
    # fakeit wants each value as an array; shape (n,) is fine (1 batch dim).
    spectra = fakeit_for_multiple_parameters(
        obsconf, model, samples, rng_key=seed_for_fakeit, apply_stat=apply_poisson
    )
    x = np.asarray(spectra)
    theta = np.stack([np.asarray(samples[p]) for p in param_order], axis=1)
    return theta, x, param_order


def fold_theta(base_model_name, param_order, theta, obsconf):
    """Fold an explicit array of parameter vectors through ``obsconf`` to get the
    NOISELESS per-channel model counts (lambda).

    Unlike ``simulate_spectra`` (which draws theta from a prior), this takes theta
    directly -- used by the Phase-3 importance-sampling refinement, where the
    proposal samples must be folded through the SAME response to evaluate the
    exact Poisson likelihood p(x | theta) = Poisson(x; lambda(theta)).

    ``theta`` is ``(M, n_params)`` in ``param_order`` (linear units). Returns
    ``(M, n_channels)`` float64 expected counts. apply_stat=False => no Poisson
    noise (we want the rate lambda, not a realization).
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    model = _models.build_model(base_model_name)
    params = {p: theta[:, j] for j, p in enumerate(param_order)}
    spectra = fakeit_for_multiple_parameters(
        obsconf, model, params, rng_key=0, apply_stat=False
    )
    return np.asarray(spectra, dtype=np.float64)


def _fold_expected_counts(base_model_name, prior_cfg, obsconf, n, rng):
    """Median total *expected* (noiseless) counts over the prior - used for
    exposure calibration."""
    _, x, _ = simulate_spectra(
        base_model_name, prior_cfg, obsconf, n, rng, apply_poisson=False
    )
    return np.median(x.sum(axis=1))


# --------------------------------------------------------------------------
# exposure calibration
# --------------------------------------------------------------------------

def calibrate_exposures(config: dict, n_probe: int = 20000):
    """Find, for each requested count target, the exposure (seconds) giving that
    median total expected counts over the prior. Counts scale linearly with
    exposure, so one probe exposure + a ratio nails it (we verify with a second
    fold).

    Returns dict {level_name: {target_counts, exposure_s, achieved_median}}.
    """
    base_model_name = config["base_model"]
    prior_cfg = config["priors"]
    base = _responses.load_base_obsconf(config.get("response", _responses.EXAMPLE_NAME))

    probe_exposure = float(config.get("calibration", {}).get("probe_exposure_s", 10000.0))
    rng = np.random.default_rng(config["seed"])

    oc_probe = _responses.scale_exposure(base, probe_exposure)
    median_at_probe = _fold_expected_counts(base_model_name, prior_cfg, oc_probe, n_probe, rng)

    results = {}
    for level in config["levels"]:
        name = level["name"]
        target = float(level["target_counts"])
        # linear scaling: exposure_needed = probe * target / median_at_probe
        exposure = probe_exposure * target / median_at_probe
        # verify
        rng2 = np.random.default_rng(config["seed"] + 1)
        oc = _responses.scale_exposure(base, exposure)
        achieved = _fold_expected_counts(base_model_name, prior_cfg, oc, n_probe, rng2)
        results[name] = {
            "target_counts": target,
            "exposure_s": float(exposure),
            "achieved_median": float(achieved),
        }
    return results


# --------------------------------------------------------------------------
# dataset generation (skip-if-exists)
# --------------------------------------------------------------------------

def generate_level(config: dict, level: dict, force: bool = False) -> Path:
    """Generate one exposure level's dataset and save to data/sim/. Skips if the
    .npz already exists (crash-resumable) unless force=True."""
    base_model_name = config["base_model"]
    prior_cfg = config["priors"]
    name = f"{config['name']}_{level['name']}"
    out_path = _resolve_data_path(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} exists")
        return out_path

    base = _responses.load_base_obsconf(config.get("response", _responses.EXAMPLE_NAME))
    exposure = float(level["exposure_s"])
    obsconf = _responses.scale_exposure(base, exposure)

    n = int(level.get("n", config.get("n", 10000)))
    # deterministic, level-specific seeds derived from the global seed
    level_seed = config["seed"] + level.get("seed_offset", 0)
    rng = np.random.default_rng(level_seed)

    theta, x, param_order = simulate_spectra(
        base_model_name, prior_cfg, obsconf, n, rng,
        apply_poisson=True, seed_for_fakeit=level_seed,
    )

    e_out = np.asarray(base.out_energies)
    np.savez_compressed(
        out_path,
        theta=theta,
        x=x,
        param_names=np.array(param_order),
        e_min=e_out[0],
        e_max=e_out[1],
        exposure_s=exposure,
        base_model=base_model_name,
        seed=level_seed,
        median_total_counts=float(np.median(x.sum(axis=1))),
    )
    print(f"[done] {out_path.name}: theta{theta.shape} x{x.shape} "
          f"median_counts={np.median(x.sum(axis=1)):.0f}")
    return out_path


def generate_all(config: dict, force: bool = False, only_level: str | None = None):
    paths = []
    for level in config["levels"]:
        if only_level and level["name"] != only_level:
            continue
        paths.append(generate_level(config, level, force=force))
    return paths


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Config-driven X-ray spectral simulator")
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default=None, help="generate only this level name")
    ap.add_argument("--force", action="store_true", help="regenerate even if npz exists")
    ap.add_argument("--calibrate-exposure", action="store_true",
                    help="print exposures for the target counts and exit (no datasets)")
    args = ap.parse_args(argv)

    config = load_config(args.config)

    if args.calibrate_exposure:
        res = calibrate_exposures(config)
        print(f"Exposure calibration for '{config['name']}' "
              f"(base={config['base_model']}, response={config.get('response', _responses.EXAMPLE_NAME)}):")
        for name, r in res.items():
            print(f"  {name:8s}: target ~{r['target_counts']:.0f} counts "
                  f"-> exposure {r['exposure_s']:.1f} s "
                  f"(achieved median {r['achieved_median']:.0f})")
        return 0

    generate_all(config, force=args.force, only_level=args.level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
