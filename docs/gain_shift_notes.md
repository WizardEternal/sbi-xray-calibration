# B4 gain-shift implementation notes

How the B4 gain-shift misspecification is implemented: an in-place rescale of
the response's unfolded energy grid, with no RMF FITS rewrite. These notes
record the jaxspec internals the mechanism relies on, the sign convention, and
the centroid check that verifies the shift lands where it should.

## What a gain shift is

A detector gain error is a miscalibration of the energy scale: an event of true
photon energy `E` is recorded as though it had energy `E_eff`. The XSPEC `gain`
command uses `E_eff = E/slope - offset`; here we use a pure multiplicative slope
`gain`, i.e. `E_eff = gain * E`. Folding a *fixed* source model through a
gain-shifted response moves every spectral feature (lines, edges, the continuum
rollover) in channel space without changing the source. That is exactly the B4
misspecification: the inference assumes the nominal response, but the data were
generated with a slightly wrong one.

## jaxspec internals (verified by reading the source)

- `ObsConfiguration` is an `xarray.Dataset` subclass. Its `data_vars` are
  `transfer_matrix` (sparse COO, shape 102 folded x 2067 unfolded), `area`
  (ARF, 2067), `exposure`, `folded_counts`, etc.
- `forward_model_with_multiple_inputs` evaluates the source model with
  `model.photon_flux(p, e_low, e_high)` on the **unfolded (input) energy grid**
  `obsconf.in_energies` (shape `(2, 2067)`), then does
  `transfer_matrix @ photon_flux` to get folded counts in 102 channels.
- `ObsConfiguration.in_energies` is **derived from the coords**
  `e_min_unfolded` / `e_max_unfolded` (confirmed by reading the property
  source). So those coords ARE the response's input energy axis.

## The mechanism that worked (in-place, no FITS surgery)

```python
def gain_shift_obsconf(obsconf, gain):
    shifted = obsconf.copy(deep=True)
    return shifted.assign_coords(
        e_min_unfolded=shifted.coords["e_min_unfolded"] * gain,
        e_max_unfolded=shifted.coords["e_max_unfolded"] * gain,
    )
```

Rescaling the unfolded-grid coords by `gain` means the source model is evaluated
as if input bin `i` (which the transfer matrix maps to its nominal channels)
sits at energy `gain * E_i`. A source feature at model-energy `E` therefore
lands where the nominal response expected energy `E/gain`.

**Key API detail:** `ObsConfiguration` is an xarray Dataset, so
`obsconf.assign_coords(e_min_unfolded=..., e_max_unfolded=...)` (after a
`copy(deep=True)`) is all that is required, and the transfer matrix and ARF are
left untouched and the existing `fakeit_for_multiple_parameters` /
`forward_model_with_multiple_inputs` path picks up the shifted grid for free.

### Sign convention (documented, arbitrary)

With this multiplicative convention:
- `gain > 1` (e.g. +2%) pushes features to **lower** channel energy.
- `gain < 1` (e.g. -2%) pushes features to **higher** channel energy.

Both are genuine gain errors; the sign is just which way we label the slope. The
simulator config exposes a signed percent grid and the centre (0%) is the
nominal response.

## Proof (outputs/diagnostics/gain_shift_check.png)

Folded tbabs*(powerlaw + narrow Gauss at 6.4 keV) through nominal, +2%, and -2%
responses. Flux-weighted Fe-K centroid over 5.5-7.5 keV:

| gain  | Fe-K centroid | shift   |
|-------|---------------|---------|
| 1.000 | 6.401 keV     | 0       |
| 1.020 | 6.274 keV     | -0.127 keV (= -2.0%) |
| 0.980 | 6.530 keV     | +0.129 keV (= +2.0%) |

The shift magnitude matches the gain factor to <1% and the line visibly moves in
the zoom panel. Verdict assertions in the script pass.

## What did NOT need to be done

- No writing of a gain-shifted RMF FITS file + reload via
  `Instrument.from_ogip_file` (the documented fallback). The in-place coord
  rescale is exact, fast, and needs no temp files, so the FITS-rewrite path is
  left unused. (`Instrument.from_ogip_file` remains available if a future need
  for a genuinely re-binned RMF arises.)

## Caveat

The example EPIC-pn response is binned to only 102 folded channels (~0.18 keV
per channel near Fe-K), so a 2% shift is a fraction of a channel and
`argmax`-on-channel is too coarse to see it; the **flux-weighted centroid** is
the right diagnostic and is what the simulator's B4 test uses. A finer-binned
response would show the shift channel-by-channel, but the physics (and the
forward model) is identical.
