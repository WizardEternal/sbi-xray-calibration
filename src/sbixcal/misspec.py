"""Misspecification generators B1-B4 (Phase 1).

Each family takes a base model + its priors and produces spectra that deviate
from the well-specified Model A along a configurable strength grid:

  B1  unmodeled narrow Gaussian (Fe-K) line at 6.4 keV; grid = line norm
      (equivalent-width proxy), from negligible to strong.
  B2  Tbpcf partial-covering absorber replacing tbabs; grid = covering fraction f.
  B3  continuum-family swap: powerlaw -> custom thermal bremsstrahlung (default)
      or Diskbb; grid = continuum temperature kT/Tin.
  B4  detector gain shift via response energy-grid rescale; grid = gain percent.

The "nuisance" source parameters (shared with Model A) are drawn from the same
priors so each misspecified dataset is comparable to the clean one at the same
exposure level. Strength = 0 (or gain = 0%) recovers the clean Model A and is a
useful control.

CLI:
    python -m sbixcal.misspec --config configs/misspec_modelA_dev.yaml
    python -m sbixcal.misspec --config configs/misspec_modelA_dev.yaml --family B1
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import yaml

from jaxspec.data.util import fakeit_for_multiple_parameters

from . import models as _models
from . import priors as _priors
from . import responses as _responses
from .simulate import load_config, _repo_root


def _data_path(name: str) -> Path:
    return _repo_root() / "data" / "sim" / f"{name}.npz"


def _stable_hash(s: str, mod: int = 100000) -> int:
    """Deterministic, cross-process-stable hash of a string in [0, mod).

    Python's built-in ``hash()`` of ``str`` is salted per process
    (``PYTHONHASHSEED`` randomization), so using it for RNG seeding silently
    breaks reproducibility across runs/sessions. We hash with sha1 and take the
    digest mod ``mod`` instead, so a given (family, strength-label) always maps to
    the same seed offset regardless of process.
    """
    digest = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


def _base_nuisance(base_model_name, prior_cfg, n, rng):
    """Draw the shared Model-A parameters (the nuisance source params)."""
    order = _models.MODEL_PARAMS[base_model_name]
    return _priors.sample_prior(prior_cfg, order, n, rng), order


# --------------------------------------------------------------------------
# per-family parameter assembly
# --------------------------------------------------------------------------

def _params_b1(base_model_name, prior_cfg, n, rng, strength, fixed):
    """B1: base params + Gaussian line. strength = line norm."""
    src, _ = _base_nuisance(base_model_name, prior_cfg, n, rng)
    src["gauss_1_El"] = np.full(n, fixed.get("line_energy_kev", 6.4))
    src["gauss_1_sigma"] = np.full(n, fixed.get("line_sigma_kev", 0.05))
    src["gauss_1_norm"] = np.full(n, strength)
    model = _models.build_model_b1(base_model_name)
    return model, src


def _params_b2(base_model_name, prior_cfg, n, rng, strength, fixed):
    """B2: Tbpcf partial covering. strength = covering fraction f in [0,1].
    Reuse the tbabs N_H prior for the tbpcf N_H column."""
    src, order = _base_nuisance(base_model_name, prior_cfg, n, rng)
    # tbabs_1_nh -> tbpcf_1_nh, drop tbabs key
    nh = src.pop("tbabs_1_nh")
    src["tbpcf_1_nh"] = nh
    src["tbpcf_1_f"] = np.full(n, strength)
    model = _models.build_model_b2(base_model_name)
    return model, src


def _params_b3(base_model_name, prior_cfg, n, rng, strength, fixed):
    """B3: continuum swap. strength = continuum temperature (kT or Tin, keV).
    The powerlaw is replaced; we keep the same log-uniform norm prior for the
    new continuum so total flux stays comparable."""
    src, order = _base_nuisance(base_model_name, prior_cfg, n, rng)
    use_diskbb = bool(fixed.get("use_diskbb", False))
    # remove powerlaw params, keep nh and (for prod) blackbody params
    pl_norm = src.pop("powerlaw_1_norm")
    src.pop("powerlaw_1_alpha")
    if use_diskbb:
        src["diskbb_1_Tin"] = np.full(n, strength)
        src["diskbb_1_norm"] = pl_norm
    else:
        src["brems_1_kT"] = np.full(n, strength)
        src["brems_1_norm"] = pl_norm
    model = _models.build_model_b3(base_model_name, use_diskbb=use_diskbb)
    return model, src


def _params_b4(base_model_name, prior_cfg, n, rng, strength, fixed):
    """B4: gain shift. strength = gain percent (e.g. 2.0 -> gain 1.02). The model
    is the CLEAN base model; the misspecification lives in the response."""
    src, _ = _base_nuisance(base_model_name, prior_cfg, n, rng)
    model = _models.build_model(base_model_name)
    return model, src  # gain applied to obsconf by caller


FAMILIES = {"B1": _params_b1, "B2": _params_b2, "B3": _params_b3, "B4": _params_b4}


# --------------------------------------------------------------------------
# in-memory population generation (used by the Phase-4 detect benchmark)
# --------------------------------------------------------------------------

def simulate_misspec_population(
    base_model_name: str,
    prior_cfg: dict,
    obsconf,
    family: str,
    strength: float,
    n: int,
    seed: int,
    fixed: dict | None = None,
):
    """Generate ``n`` misspecified spectra for one (family, strength) IN MEMORY.

    Mirrors :func:`generate_family_point` exactly (same per-family parameter
    assembly, same B4 gain-shift-on-response path) but returns arrays instead of
    writing an npz -- the detection benchmark needs to draw fresh misspecified test
    populations on the fly without disk round-trips.

    ``obsconf`` is the EXPOSURE-SCALED ObsConfiguration for the count level (the
    caller scales it from the checkpoint's exposure). For B4 the gain shift is
    applied here to a copy, leaving the caller's obsconf untouched.

    Returns ``(x (n, C) float64 Poisson counts, theta (n, P_base) float64 base
    Model-A params, param_names list)``. ``theta`` holds the shared Model-A
    parameters that the clean inference targets (e.g. Γ for the B1 ΔΓ-bias
    consequence analysis); columns absent for a family (e.g. the swapped continuum
    in B3) are simply not present.
    """
    fixed = fixed or {}
    sval = float(strength)
    rng = np.random.default_rng(int(seed))
    model, params = FAMILIES[family](base_model_name, prior_cfg, n, rng, sval, fixed)

    oc = obsconf
    if family == "B4":
        oc = _responses.gain_shift_obsconf(obsconf, 1.0 + sval / 100.0)

    x = np.asarray(
        fakeit_for_multiple_parameters(oc, model, params, rng_key=int(seed), apply_stat=True),
        dtype=np.float64,
    )
    order = _models.MODEL_PARAMS[base_model_name]
    present = [p for p in order if p in params]
    theta = np.stack([np.asarray(params[p], dtype=np.float64) for p in present], axis=1) \
        if present else np.empty((n, 0))
    return x, theta, present


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def generate_family_point(config, family, level, strength, force=False):
    """Generate one (family, exposure-level, strength) dataset -> npz, skip-if-exists."""
    base_model_name = config["base_model"]
    prior_cfg = config["priors"]
    fam_cfg = config["families"][family]
    fixed = fam_cfg.get("fixed", {})

    sval = float(strength)
    slabel = f"{sval:g}".replace("-", "m").replace(".", "p")
    name = f"{config['name']}_{family}_{level['name']}_s{slabel}"
    out_path = _data_path(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[skip] {out_path.name}")
        return out_path

    base = _responses.load_base_obsconf(config.get("response", _responses.EXAMPLE_NAME))
    obsconf = _responses.scale_exposure(base, float(level["exposure_s"]))

    n = int(level.get("n", config.get("n", 5000)))
    # NOTE: use a stable sha1-based hash, NOT Python's built-in
    # hash(), which is per-process salted (PYTHONHASHSEED) and would make this seed
    # -- and hence the generated dataset -- non-reproducible across runs/sessions.
    seed = config["seed"] + level.get("seed_offset", 0) + _stable_hash(family + slabel)
    seed = int(seed) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    model, params = FAMILIES[family](base_model_name, prior_cfg, n, rng, sval, fixed)

    if family == "B4":
        obsconf = _responses.gain_shift_obsconf(obsconf, 1.0 + sval / 100.0)

    x = np.asarray(
        fakeit_for_multiple_parameters(obsconf, model, params, rng_key=seed, apply_stat=True)
    )
    # store base Model-A theta for downstream comparison (the params the clean
    # inference would target); only the shared columns.
    order = _models.MODEL_PARAMS[base_model_name]
    theta = np.stack([np.asarray(params[p]) for p in order if p in params], axis=1) \
        if all(p in params for p in order) else None

    e_out = np.asarray(base.out_energies)
    np.savez_compressed(
        out_path,
        x=x,
        theta=theta if theta is not None else np.empty((n, 0)),
        param_names=np.array([p for p in order if p in params]),
        family=family,
        strength=sval,
        exposure_s=float(level["exposure_s"]),
        e_min=e_out[0],
        e_max=e_out[1],
        seed=seed,
        median_total_counts=float(np.median(x.sum(axis=1))),
    )
    print(f"[done] {out_path.name}: x{x.shape} median_counts={np.median(x.sum(axis=1)):.0f}")
    return out_path


def generate_all(config, only_family=None, force=False):
    paths = []
    for family, fam_cfg in config["families"].items():
        if only_family and family != only_family:
            continue
        grid = fam_cfg["strength_grid"]
        for level in config["levels"]:
            for strength in grid:
                paths.append(generate_family_point(config, family, level, strength, force=force))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Misspecification generators B1-B4")
    ap.add_argument("--config", required=True)
    ap.add_argument("--family", default=None, choices=list(FAMILIES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    generate_all(config, only_family=args.family, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
