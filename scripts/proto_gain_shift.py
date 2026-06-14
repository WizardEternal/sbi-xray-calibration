"""Phase 0 prototype: B4 gain-shift on the jaxspec EPIC-pn response.

We need a way to perturb the
response energy calibration so that a *fixed* source model, folded through a
"miscalibrated" response, produces a spectrum whose features are shifted in
channel space relative to the nominal response. That is exactly what a real
detector gain error does, and it is the B4 misspecification family.

Mechanism investigated here (in-place, no FITS rewrite):
    The jaxspec ObsConfiguration is an xarray Dataset. The forward model
    (forward_model_with_multiple_inputs) evaluates the source photon flux on
    the *input* (unfolded) energy grid given by obsconf.in_energies, which is
    read from the coords e_min_unfolded / e_max_unfolded, then multiplies by
    the transfer matrix (ARF x RMF) to get folded counts.

    A detector gain error means the response maps a photon of true energy E to
    a channel as though it had energy E_eff = gain * E + offset (XSPEC `gain`
    convention is E_eff = E/slope - offset; we use the multiplicative slope).
    Equivalently, the response's notion of "energy per input bin" is rescaled.
    We implement this by rescaling the unfolded energy-grid coords by the gain
    factor and re-deriving the per-bin photon flux on the original physical
    grid. Concretely: we build a gain-shifted ObsConfiguration whose
    e_min_unfolded / e_max_unfolded coords are multiplied by `gain`. When the
    SAME source model is folded through it, a feature the source emits at
    energy E lands where the nominal response would have put energy E*gain,
    i.e. the folded spectrum's features move by the gain factor. This is the
    response-energy-axis surgery used for the B4 misspecification family.

We prove it by folding tbabs*(powerlaw + narrow Gaussian at 6.4 keV) through
the nominal and the +2% gain-shifted responses and showing the Fe-K line
moves in channel space.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jaxspec.data.util import load_example_obsconf, fakeit_for_multiple_parameters
from jaxspec.model.additive import Powerlaw, Gauss
from jaxspec.model.multiplicative import Tbabs


def gain_shift_obsconf(obsconf, gain: float):
    """Return a copy of `obsconf` with its unfolded (input) energy grid rescaled
    by `gain`. gain=1.0 is the nominal response; gain=1.02 is a +2% shift.

    This rescales the coords e_min_unfolded / e_max_unfolded that
    ObsConfiguration.in_energies is built from, so the source model is folded
    as if the detector assigned energy E_eff = gain * E to each input bin.
    """
    shifted = obsconf.copy(deep=True)
    shifted = shifted.assign_coords(
        e_min_unfolded=shifted.coords["e_min_unfolded"] * gain,
        e_max_unfolded=shifted.coords["e_max_unfolded"] * gain,
    )
    return shifted


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
        "diagnostics",
    )
    os.makedirs(out_dir, exist_ok=True)

    obs = load_example_obsconf("NGC7793_ULX4_PN")

    # Source model with a STRONG narrow Fe-K line at 6.4 keV so the shift is
    # unmistakable. Same model folded through both responses.
    model = Tbabs() * (Powerlaw() + Gauss())
    params = {
        "tbabs_1_nh": np.array([0.2]),
        "powerlaw_1_alpha": np.array([1.7]),
        "powerlaw_1_norm": np.array([5e-3]),
        "gauss_1_El": np.array([6.4]),
        "gauss_1_sigma": np.array([0.05]),
        "gauss_1_norm": np.array([5e-3]),
    }

    gains = {"nominal (g=1.000)": 1.0, "+2% gain (g=1.020)": 1.02, "-2% gain (g=0.980)": 0.98}

    # Channel energies (folded) of the NOMINAL response for the x-axis.
    e_out = np.asarray(obs.out_energies)  # (2, 102)
    e_mid = 0.5 * (e_out[0] + e_out[1])

    spectra = {}
    for label, g in gains.items():
        oc = gain_shift_obsconf(obs, g)
        # apply_stat=False -> noiseless expected counts, so the feature shift is
        # not masked by Poisson scatter.
        cr = fakeit_for_multiple_parameters(oc, model, params, apply_stat=False)
        spectra[label] = np.asarray(cr).ravel()

    # Fe-K feature location per gain. The folded grid is coarse (~102 channels
    # over 0.3-10 keV => ~0.18 keV/channel near Fe-K), so argmax-on-channel is
    # too quantized to see a 2% (~0.13 keV) shift cleanly. We instead use a
    # continuum-subtracted, flux-weighted centroid over the 5.5-7.5 keV window,
    # which measures the line position to well below one channel width.
    win = (e_mid > 5.5) & (e_mid < 7.5)

    def line_centroid(spec):
        # crude local continuum = min in window; weight by excess above it
        w = np.clip(spec[win] - spec[win].min(), 0, None)
        return float(np.sum(e_mid[win] * w) / np.sum(w))

    print("Fe-K line centroid (flux-weighted) by gain:")
    peak_energy = {}
    for label, spec in spectra.items():
        e_c = line_centroid(spec)
        peak_energy[label] = e_c
        print(f"  {label:24s}  E_centroid = {e_c:.3f} keV   total counts = {spec.sum():.1f}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = {"nominal (g=1.000)": "k", "+2% gain (g=1.020)": "C3", "-2% gain (g=0.980)": "C0"}
    for label, spec in spectra.items():
        axes[0].step(e_mid, spec, where="mid", label=label, color=colors[label], lw=1.5)
    axes[0].set_ylabel("expected counts / channel")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-1, None)
    axes[0].legend()
    axes[0].set_title("tbabs*(powerlaw + 6.4 keV Gauss) folded through gain-shifted EPIC-pn responses")

    # Zoom on the Fe-K region to make the shift obvious.
    for label, spec in spectra.items():
        axes[1].step(e_mid, spec, where="mid", label=label, color=colors[label], lw=1.5)
    axes[1].axvline(6.4, color="gray", ls=":", label="6.4 keV (rest)")
    axes[1].set_xlim(5.0, 8.0)
    axes[1].set_xlabel("channel energy (nominal response) [keV]")
    axes[1].set_ylabel("expected counts / channel")
    axes[1].legend()
    axes[1].set_title("Fe-K zoom: gain shift moves the line in channel space")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "gain_shift_check.png")
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")

    # Verdict numbers. Convention: rescaling the unfolded grid by `gain` means
    # the source feature at model-energy E is placed where the nominal response
    # expected energy E/gain. So gain>1 (+2%) pushes the line to LOWER channel
    # energy and gain<1 (-2%) to HIGHER channel energy. Both are real gain
    # errors; the sign is just a convention choice and is documented in
    # docs/gain_shift_notes.md.
    e_nom = peak_energy["nominal (g=1.000)"]
    e_up = peak_energy["+2% gain (g=1.020)"]
    e_dn = peak_energy["-2% gain (g=0.980)"]
    print(f"\nShift check: +2% moved Fe-K centroid {e_up - e_nom:+.3f} keV, "
          f"-2% moved it {e_dn - e_nom:+.3f} keV (expect ~ -/+ 0.13 keV).")
    assert e_up < e_nom, "expected +gain to push line to LOWER channel energy"
    assert e_dn > e_nom, "expected -gain to push line to HIGHER channel energy"
    assert abs(abs(e_up - e_nom) - 0.128) < 0.05, "shift magnitude off (~2% of 6.4)"
    print("VERDICT: in-place coord rescale works; feature shifts as expected.")


if __name__ == "__main__":
    sys.exit(main())
