# sigma_eff gain-information screen

This directory holds the per-spectrum Fisher gain-information screen over the two
500-draw simulated populations:

- `sigma_eff_bright.json`, `sigma_eff_bright.csv`
- `sigma_eff_medium.json`, `sigma_eff_medium.csv`

Both come from `scripts/screen_sigma_eff.py`:

```
python scripts/screen_sigma_eff.py --level bright
python scripts/screen_sigma_eff.py --level medium
```

500 spectra each, 12.8 s bright and 11.4 s medium. The run is deterministic and
repeats bit-exact. Each row carries the spectrum index, total counts, soft-band
counts below 1.2 keV, the five true parameters, `sigma_eff` marginal and
conditional, `I`, `||b||^2`, `R`, the predicted posterior sd and shrink, the
prior-edge margins, and a flag for the spectra that already have a nested
sampling run.

Population seed is 20300611. The script re-checks the ten-spectrum counts gate
(3853/384/429/291/599/5204/25067/10496/6829/54509) at startup and prints pass or
fail. It passes 4/4 bright and 6/6 medium.

## What sigma_eff is

At a spectrum's true theta and its level's exposure, with `mu0` the expected
counts per channel and `W = diag(1/sqrt(mu0))` the Poisson whitening:

```
b         = W [mu(g=1.03) - mu(g=0.97)] / (2 x 0.03)
A         = W dmu/dtheta            (102 x 5)
I         = ||P_A b||^2 / ||b||^2
sigma_eff = 1 / sqrt((1 - I) ||b||^2)
```

`P_A` is the projector onto the span of `A` from a reduced QR, so `I` is the
fraction of the gain signature the five physical parameters absorb. `sigma_eff`
is in units of the gain factor `g`, dimensionless, and the prior on `g` is
U[0.95, 1.05] with standard deviation 0.0289.

The shift is symmetric and `Delta g` is 0.03. Both matter. A one-sided shift of
the same size gives 0.041 / 0.056 / 0.046 for i416 / i394 / i8, which is 12 to 17
per cent off. The form above is algebraically the same as `Delta g / sqrt(R)`
with `R` the residual power, and it equals `sqrt([F^-1]_gg)` of the 6x6 Fisher
matrix. The QR projection and an explicitly inverted 6x6 Fisher agree to a
relative 1e-14 on seven spectra.

## Reproduction of the seven recorded values

Seven per-spectrum values were on record from an earlier calculation whose
scripts no longer exist, so this script is the committed replacement. It
reproduces all seven:

| spectrum | level | recorded | this script | difference |
|---|---|---|---|---|
| i416 | bright | 0.0366 | 0.0374 | +2.4% |
| i394 | bright | 0.0500 | 0.0506 | +1.3% |
| i8 | bright | 0.0390 | 0.0385 | -1.3% |
| i238 | bright | 0.1970 | 0.2032 | +3.2% |
| i22 | medium | 0.4568 | 0.4673 | +2.3% |
| i87 | medium | 0.2026 | 0.2059 | +1.6% |
| i482 | medium | 0.1228 | 0.1262 | +2.8% |

Worst case 3.2 per cent. Six of the seven come out high by 1.3 to 3.2 per cent,
so it is a consistent small offset and not scatter. I could not pin it down. The
Jacobian relative step moves everything by under 1 per cent, whitening by
observed counts instead of `mu0` moves the seven in both directions and is no
better overall, and the shift size is settled below. Two things I did not test: a
different QR rank tolerance, and an average over several shifts.

`h = 0.03` is the only step that lands all seven inside 4 per cent. At h = 0.020
the errors run -1.8 to +14.0 per cent, at 0.025 +10.7 to +15.8, at 0.030 -0.1 to
+3.2, at 0.040 -10.8 to +1.9, at 0.050 -5.2 to +7.8. The step dependence is
jagged, not smooth, because the projected residual is small and jaxspec folds in
float32.

## Population numbers

Bright, n = 500: `sigma_eff` median 0.0947, IQR [0.0534, 0.1648], range [0.0244,
0.4268], median total counts 10542. The fraction below 0.0289, the prior sd, is
0.008; below 0.04 it is 0.106, below 0.05 it is 0.224, and below 0.10 it is
0.528.

Medium, n = 500: `sigma_eff` median 0.2997, IQR [0.1686, 0.5202], range [0.0772,
1.3505]. Not one spectrum of 500 falls below 0.06 and only 2.2 per cent fall
below 0.10, so the medium level carries essentially no gain information anywhere
in the prior.

The four bright spectra that have a nested sampling run:

| idx | counts | soft counts (<1.2 keV) | sigma_eff marginal | sigma_eff conditional | shrink_pred |
|---|---|---|---|---|---|
| i8 | 25067 | 13677 | 0.0390 | 0.00698 | 0.804 |
| i238 | 10496 | 613 | 0.2032 | 0.00865 | 0.990 |
| i394 | 6829 | 3213 | 0.0506 | 0.01447 | 0.869 |
| i416 | 54509 | 14116 | 0.0374 | 0.00685 | 0.792 |

## Caveats for reuse

`sigma_eff` is ordinal here and should stay ordinal. Combined with the `g` prior
in quadrature it predicts posterior shrinks of 0.792 / 0.804 / 0.869 / 0.990 for
i416 / i8 / i394 / i238, against measured nested sampling shrinks of 0.7313 /
0.8557 / 0.7901 / 1.0050. It gets the most informative spectrum and the null
right but swaps i8 against i394, and the widths are off in both directions. A
Gaussian quadrature of a local Fisher width with a bounded uniform prior is
crude, and the exact posterior is not Gaussian. The coarser statement still
holds, since the population spans 0.024 to 0.43 and the three informative spectra
all sit in the bottom decile.

The estimator carries float32 noise that depends on how many spectra are folded
in one call. Over batch sizes 4 to 500 at a fixed spectrum, i8 has CV 0.74 per
cent (0.03816 to 0.03898) and the other three sit at or below 0.08 per cent. A
one-row fold is worse, and i8 then comes out at 0.0401, 4 per cent above the
batch mean. The script always folds the whole population in one call. Repeating a
run at a fixed batch size is bit-exact.

Soft-band counts take channels whose centre is below 1.2 keV, settable with
`--soft-band-kev`. An earlier convention took roughly one channel more and gave
i394 3248 against 3213 here, and i238 645 against 613. Nothing else depends on
the column.

Two adjacent numbers do not reproduce under this definition, and both are worth
knowing before the screen is reused.

The paper's "a Fisher estimate for the medium population predicts a shrink of
0.987" does not come back. This medium population gives median `shrink_pred`
0.9954 and mean 0.9906, and a 200-draw prior sample at seed 20260611 gives 0.9957
and 0.9906. Reaching 0.987 needs a population `sigma_eff` near 0.177, though the
median here is 0.2997. The substance is unharmed, because 0.991 to 0.995 says
"essentially no shrink" at least as strongly as 0.987 does, but the third digit
is not reproducible from this definition. The +2 per cent method offset cannot
explain the gap: at shrink 0.99, a 2 per cent error in `sigma_eff` propagates to
0.04 per cent in shrink.

A recorded "about 48 per cent of bright prior draws carry likelihood information"
does not reproduce either. The fraction below shrink 0.86 is 21.2 per cent here,
and 22.0 per cent on the 200-draw sample. The fraction below shrink 0.95 is 46.0
per cent, and 43.5 per cent on that sample, which is most likely what the 48 per
cent actually was. That number is not in the paper.
