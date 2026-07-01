"""Response / ObsConfiguration handling: loading, exposure control, B4 gain shift.

The base response is the bundled real XMM-Newton EPIC-pn observation
`NGC7793_ULX4_PN` (Quintin et al. 2021), 102 grouped folded channels over the
~0.5-10 keV band (this is the single response used throughout).
"""

from __future__ import annotations

import functools
import os

from jaxspec.data.util import load_example_obsconf

EXAMPLE_NAME = "NGC7793_ULX4_PN"

# Second instrument for the article extension: a real NICER XTI response
# (on-axis ARF nixtiaveonaxis20170601v005 + RMF nixtiref20170601v003, public
# NICER CALDB). Sourced into data/nicer/; loaded as a zero-count mock obsconf,
# since the benchmark only needs the response (counts come from fakeit). The
# 0.3-10 keV band gives 969 folded channels. Exposure is set downstream by
# scale_exposure / the count-target calibrator, exactly as for the XMM path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NICER_RMF = os.path.join(_REPO_ROOT, "data", "nicer", "nicer.rmf")
NICER_ARF = os.path.join(_REPO_ROOT, "data", "nicer", "nicer.arf")
NICER_BAND_KEV = (0.3, 10.0)


@functools.lru_cache(maxsize=4)
def load_base_obsconf(name: str = EXAMPLE_NAME):
    """Load (and cache) the base ObsConfiguration. Cached across calls so we pay
    the FITS-read cost once per process.

    `name="NICER"` builds the second-instrument response from the local OGIP
    RMF/ARF; any other name falls through to jaxspec's bundled XMM examples.
    """
    if name.upper() == "NICER":
        from jaxspec.data import Instrument, ObsConfiguration

        inst = Instrument.from_ogip_file(NICER_RMF, NICER_ARF)
        return ObsConfiguration.mock_from_instrument(
            inst, exposure=1.0,
            low_energy=NICER_BAND_KEV[0], high_energy=NICER_BAND_KEV[1],
        )
    return load_example_obsconf(name)


def scale_exposure(obsconf, target_exposure: float):
    """Return a copy of `obsconf` with effective exposure set to
    `target_exposure` (seconds).

    The example obsconf folds exposure into the transfer matrix (verified:
    setting the `exposure` data_var alone does NOT change counts). We instead
    rescale the transfer matrix by target/native, which scales expected counts
    exactly linearly while preserving the realistic 102-channel grouping.
    """
    native = float(obsconf.exposure)
    factor = target_exposure / native
    oc = obsconf.copy(deep=True)
    oc["transfer_matrix"] = oc["transfer_matrix"] * factor
    # keep the bookkeeping consistent
    oc["exposure"] = oc["exposure"] * 0 + target_exposure
    return oc


def gain_shift_obsconf(obsconf, gain: float):
    """B4: return a copy of `obsconf` whose response energy calibration is
    perturbed by multiplicative `gain` (1.0 = nominal).

    Implemented in-place (no RMF FITS rewrite) by rescaling the unfolded
    (input) energy-grid coords that `in_energies` -- and hence the forward
    model's per-bin photon-flux evaluation -- is built from. A source feature
    at model-energy E is folded as if it sat at E/gain in the nominal response,
    so gain>1 shifts features to lower channel energy and gain<1 to higher.
    See docs/gain_shift_notes.md.
    """
    oc = obsconf.copy(deep=True)
    oc = oc.assign_coords(
        e_min_unfolded=oc.coords["e_min_unfolded"] * gain,
        e_max_unfolded=oc.coords["e_max_unfolded"] * gain,
    )
    return oc
