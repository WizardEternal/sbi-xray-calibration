"""Config-driven prior sampling.

A prior config is a dict mapping each parameter name to a spec:

    powerlaw_1_alpha: {dist: uniform,     low: 0.5, high: 2.5}
    powerlaw_1_norm:  {dist: loguniform,  low: 0.01, high: 100}

`uniform` samples linearly in [low, high]; `loguniform` samples uniformly in
log10 between low and high (low/high are given in linear units). Bounds are
inclusive at the config level; samples always respect [low, high].
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def sample_prior(
    prior_cfg: Mapping[str, Mapping],
    param_order,
    n: int,
    rng: np.random.Generator,
):
    """Draw `n` samples for each parameter named in `param_order`.

    Returns a dict {param_name: ndarray(n,)} suitable (after reshaping) for
    `fakeit_for_multiple_parameters`.
    """
    out = {}
    for name in param_order:
        spec = prior_cfg[name]
        dist = spec["dist"]
        low = float(spec["low"])
        high = float(spec["high"])
        if dist == "uniform":
            out[name] = rng.uniform(low, high, size=n)
        elif dist == "loguniform":
            if low <= 0 or high <= 0:
                raise ValueError(f"loguniform bounds must be positive for '{name}'")
            out[name] = 10.0 ** rng.uniform(np.log10(low), np.log10(high), size=n)
        else:
            raise ValueError(f"unknown dist '{dist}' for '{name}'")
    return out


def prior_bounds(prior_cfg: Mapping[str, Mapping], param_order):
    """Return (low, high) arrays in linear units, in `param_order`."""
    lows = np.array([float(prior_cfg[p]["low"]) for p in param_order])
    highs = np.array([float(prior_cfg[p]["high"]) for p in param_order])
    return lows, highs


def within_bounds(samples: Mapping[str, np.ndarray], prior_cfg, param_order):
    """True iff every sample of every parameter lies within its [low, high]."""
    for name in param_order:
        spec = prior_cfg[name]
        lo, hi = float(spec["low"]), float(spec["high"])
        s = np.asarray(samples[name])
        # tiny tolerance for float round-off at the edges
        tol = 1e-9 * (abs(hi) + abs(lo) + 1.0)
        if np.any(s < lo - tol) or np.any(s > hi + tol):
            return False
    return True
