"""Spectral model factory for sbi-xray-calibration.

Defines the well-specified "Model A" source models (dev: tbabs*powerlaw;
production: tbabs*(powerlaw+blackbody)) and the misspecified variants used by the
B1-B4 misspecification families, plus a custom analytic thermal-bremsstrahlung
component (B3).

Parameter naming follows jaxspec's `<component>_<index>_<param>` convention, e.g.
`tbabs_1_nh`, `powerlaw_1_alpha`, `powerlaw_1_norm`. These are exactly the keys
expected by `fakeit_for_multiple_parameters`.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from jaxspec.model.abc import AdditiveComponent
from jaxspec.model.additive import Powerlaw, Blackbodyrad, Gauss, Diskbb
from jaxspec.model.multiplicative import Tbabs, Tbpcf


class Brems(AdditiveComponent):
    r"""Gaunt-factor-free thermal bremsstrahlung continuum (custom component).

    $$\mathcal{M}(E) = K \, E^{-1} \exp(-E / kT)$$

    The free-free continuum shape with the slowly varying O(1) Gaunt factor
    dropped. This is an intentional, documented approximation: the component's
    purpose is to be a *different continuum family* (B3 misspecification), not a
    calibrated plasma model (the B3 wrong-continuum decision).

    Parameters:
        kT   (keV): temperature / exponential cutoff scale
        norm      : normalization
    """

    def __init__(self):
        self.kT = nnx.Param(2.0)
        self.norm = nnx.Param(1e-3)

    def continuum(self, energy):
        return self.norm * jnp.exp(-energy / self.kT) / energy


# Canonical parameter order per model name. The simulator and priors iterate
# these so dict construction is deterministic and bound-checkable.
MODEL_PARAMS = {
    # Model A dev: tbabs * powerlaw (3 params) - Barret & Dupourque 2024 Model 1
    "tbabs_powerlaw": ["tbabs_1_nh", "powerlaw_1_alpha", "powerlaw_1_norm"],
    # Model A production: tbabs * (powerlaw + blackbody) (5 params) - their Model 2
    "tbabs_powerlaw_bb": [
        "tbabs_1_nh",
        "powerlaw_1_alpha",
        "powerlaw_1_norm",
        "blackbodyrad_1_kT",
        "blackbodyrad_1_norm",
    ],
}


def build_model(name: str):
    """Return a jaxspec SpectralModel for a base (well-specified) model name."""
    if name == "tbabs_powerlaw":
        return Tbabs() * Powerlaw()
    if name == "tbabs_powerlaw_bb":
        return Tbabs() * (Powerlaw() + Blackbodyrad())
    raise ValueError(f"unknown base model '{name}'. known: {list(MODEL_PARAMS)}")


# --- Misspecified model builders (B1-B3). B4 is a response operation, not a
# model change, so it lives in responses.py. ---------------------------------


def build_model_b1(base_name: str):
    """B1: base model + an unmodeled narrow Gaussian (Fe-K) line.

    Adds `gauss_1_El`, `gauss_1_sigma`, `gauss_1_norm`. The line energy/width are
    fixed by the misspec config; only `gauss_1_norm` (-> equivalent width) varies
    along the strength grid.
    """
    if base_name == "tbabs_powerlaw":
        return Tbabs() * (Powerlaw() + Gauss())
    if base_name == "tbabs_powerlaw_bb":
        return Tbabs() * (Powerlaw() + Blackbodyrad() + Gauss())
    raise ValueError(f"B1 not defined for base '{base_name}'")


def build_model_b2(base_name: str):
    """B2: replace tbabs with Tbpcf (partial-covering absorber).

    Adds covering fraction `tbpcf_1_f` (grid var) and `tbpcf_1_nh`.
    """
    if base_name == "tbabs_powerlaw":
        return Tbpcf() * Powerlaw()
    if base_name == "tbabs_powerlaw_bb":
        return Tbpcf() * (Powerlaw() + Blackbodyrad())
    raise ValueError(f"B2 not defined for base '{base_name}'")


def build_model_b3(base_name: str, use_diskbb: bool = False):
    """B3: continuum-family swap. powerlaw -> custom thermal bremsstrahlung
    (default) or Diskbb (documented fallback).

    The "strength" of this misspecification is not a single parameter; instead a
    grid of continuum temperatures (kT / Tin) is swept to span shapes from
    powerlaw-like (high kT) to strongly curved (low kT).
    """
    cont = Diskbb() if use_diskbb else Brems()
    if base_name == "tbabs_powerlaw":
        return Tbabs() * cont
    if base_name == "tbabs_powerlaw_bb":
        return Tbabs() * (cont + Blackbodyrad())
    raise ValueError(f"B3 not defined for base '{base_name}'")
