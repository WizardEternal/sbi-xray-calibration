"""Misspecification generators B1-B4.

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

Populations are generated in memory via ``simulate_misspec_population``; there
is no on-disk generation CLI.
"""

from __future__ import annotations

import hashlib

import numpy as np

from jaxspec.data.util import fakeit_for_multiple_parameters

from . import models as _models
from . import priors as _priors
from . import responses as _responses


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


# per-family parameter assembly

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
    is the clean base model; the misspecification lives in the response."""
    src, _ = _base_nuisance(base_model_name, prior_cfg, n, rng)
    model = _models.build_model(base_model_name)
    return model, src  # gain applied to obsconf by caller


FAMILIES = {"B1": _params_b1, "B2": _params_b2, "B3": _params_b3, "B4": _params_b4}


# in-memory population generation (used by the detect benchmark)

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
    """Generate ``n`` misspecified spectra for one (family, strength) in memory.

    Same per-family parameter assembly and B4 gain-shift-on-response path as the
    other families in this module -- the detection benchmark needs to draw fresh
    misspecified test populations on the fly without disk round-trips.

    ``obsconf`` is the exposure-scaled ObsConfiguration for the count level (the
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
