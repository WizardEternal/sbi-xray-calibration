"""Phase 0 prototype: B3 custom analytic thermal-bremsstrahlung component.

jaxspec has NO apec/mekal/bremss (verified). For the B3 "wrong continuum
family" misspecification we try a custom additive component first. A
Gaunt-factor-free thermal bremsstrahlung emissivity is, up to constants,

    dL/dE ~ E^-1 * exp(-E / kT)        (photons / keV ; E, kT in keV)

(the classic free-free continuum shape; the Gaunt factor is a slowly varying
O(1) logarithmic term we drop, which is an acceptable, documented approximation
for a *misspecification template* whose job is to be a different continuum
shape, not a calibrated plasma model).

This script proves the custom component (a) instantiates, (b) folds through the
EPIC-pn response via the standard fakeit path, and (c) produces a continuum that
is visibly a different shape from a powerlaw (the scientific point of B3).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
import jax.numpy as jnp

from jaxspec.model.abc import AdditiveComponent
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs
from jaxspec.data.util import load_example_obsconf, fakeit_for_multiple_parameters


class Brems(AdditiveComponent):
    r"""Gaunt-factor-free thermal bremsstrahlung continuum.

    $$\mathcal{M}(E) = K \, E^{-1} \exp(-E / kT)$$

    Parameters:
        kT   (keV): plasma temperature (exponential cutoff scale)
        norm      : normalization
    """

    def __init__(self):
        self.kT = nnx.Param(2.0)
        self.norm = nnx.Param(1e-3)

    def continuum(self, energy):
        return self.norm * jnp.exp(-energy / self.kT) / energy


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
        "diagnostics",
    )
    os.makedirs(out_dir, exist_ok=True)

    obs = load_example_obsconf("NGC7793_ULX4_PN")

    brems_model = Tbabs() * Brems()
    pl_model = Tbabs() * Powerlaw()

    # Match rough flux: brems at kT=2 keV vs powerlaw Gamma=1.7
    brems_p = {
        "tbabs_1_nh": np.array([0.2]),
        "brems_1_kT": np.array([2.0]),
        "brems_1_norm": np.array([1e-2]),
    }
    pl_p = {
        "tbabs_1_nh": np.array([0.2]),
        "powerlaw_1_alpha": np.array([1.7]),
        "powerlaw_1_norm": np.array([1e-2]),
    }

    b = np.asarray(fakeit_for_multiple_parameters(obs, brems_model, brems_p, apply_stat=False)).ravel()
    p = np.asarray(fakeit_for_multiple_parameters(obs, pl_model, pl_p, apply_stat=False)).ravel()

    e_out = np.asarray(obs.out_energies)
    e_mid = 0.5 * (e_out[0] + e_out[1])

    print(f"Brems folded OK: total expected counts = {b.sum():.1f}, finite={np.isfinite(b).all()}")
    print(f"Powerlaw folded OK: total expected counts = {p.sum():.1f}")
    print(f"Continuum shape differs (ratio brems/pl at low vs high E): "
          f"{b[3]/p[3]:.3f} vs {b[-5]/p[-5]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(e_mid, b, where="mid", label="custom thermal brems (kT=2 keV)", color="C1")
    ax.step(e_mid, p, where="mid", label="powerlaw (Gamma=1.7)", color="k")
    ax.set_xlabel("channel energy [keV]")
    ax.set_ylabel("expected counts / channel")
    ax.set_yscale("log")
    ax.set_ylim(1e-1, None)
    ax.legend()
    ax.set_title("B3 prototype: custom bremsstrahlung vs powerlaw continuum (EPIC-pn)")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "brems_check.png")
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")
    print("VERDICT: custom AdditiveComponent works end-to-end -> B3 uses custom brems.")


if __name__ == "__main__":
    main()
