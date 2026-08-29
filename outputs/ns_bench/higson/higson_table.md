# Higson thread-wise bootstrap: results

Every per-run Higson sigma below is computed from the replayed UltraNest birth-death tree with a global (Higson 2018 Algorithm 2) thread resample. The sigma each run's DONE.json stored instead reads points.hdf5 column 0 (`Lmin`, a refill-batch stamp) as a per-point birth contour and resamples threads within start-contour groups, which lowers every sigma by a median 16.5% (range 8.8-24.4%). It is carried in the last column for reference.

Jobs with DONE.json: 13/13

## Per-run: Higson thread bootstrap vs UltraNest error components

| job | kind | counts | wall_s | logZ | UN logzerr | UN bs | UN tail | Higson sigma | n_threads | reconstr-logZ delta | sigma from DONE.json | sigma/sqrt(H/400) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1_bright_idx3 | block | 1420 | 1751 | -1615.87 | 0.418 | 0.418 | 0.010 | 0.215 | 400 | +0.026 | 0.181 | 0.975 |
| B1_medium_idx2 | block | 265 | 1362 | -303.50 | 0.253 | 0.253 | 0.010 | 0.181 | 400 | +0.017 | 0.151 | 1.021 |
| clean_bright_idx9 | block | 1675 | 2992 | -250.97 | 0.143 | 0.143 | 0.010 | 0.165 | 400 | +0.019 | 0.125 | 0.891 |
| gain_pair0_clean | gain_clean | 1403 | 3652 | -288.29 | 0.175 | 0.175 | 0.010 | 0.137 | 400 | +0.011 | 0.125 | 0.981 |
| gain_pair0_gain | gain_gain | 1416 | 3527 | -282.50 | 0.135 | 0.135 | 0.010 | 0.135 | 400 | +0.010 | 0.117 | 0.992 |
| gain_pair10_clean | gain_clean | 3659 | 1524 | -326.69 | 0.159 | 0.159 | 0.010 | 0.133 | 400 | +0.011 | 0.122 | 0.956 |
| gain_pair10_gain | gain_gain | 3700 | 2044 | -326.24 | 0.250 | 0.250 | 0.010 | 0.143 | 400 | +0.011 | 0.114 | 1.031 |
| gain_pair6_clean | gain_clean | 526 | 3322 | -211.39 | 0.203 | 0.203 | 0.010 | 0.165 | 400 | +0.015 | 0.128 | 0.989 |
| gain_pair6_gain | gain_gain | 521 | 3056 | -212.63 | 0.351 | 0.351 | 0.010 | 0.172 | 400 | +0.016 | 0.131 | 0.999 |
| gain_pair8_clean | gain_clean | 90 | 2743 | -121.42 | 0.223 | 0.223 | 0.010 | 0.141 | 400 | +0.010 | 0.121 | 1.031 |
| gain_pair8_gain | gain_gain | 90 | 2914 | -121.82 | 0.182 | 0.182 | 0.010 | 0.144 | 400 | +0.010 | 0.119 | 1.052 |
| gain_pair9_clean | gain_clean | 2128 | 732 | -309.52 | 0.249 | 0.249 | 0.010 | 0.166 | 400 | +0.014 | 0.128 | 1.037 |
| gain_pair9_gain | gain_gain | 2187 | 827 | -311.05 | 0.129 | 0.128 | 0.010 | 0.139 | 400 | +0.014 | 0.119 | 0.876 |

### Consistency checks

- reconstructed nlive profile identical to chains/run.txt on all runs: True
- runs with exactly 400 threads: 13/13
- max |reconstructed logZ - UltraNest logz| = 0.0255 nats
- per-run sigma: median 0.1436, range 0.1335-0.2150 nats
- sigma / sqrt(H/400) (UltraNest logzerr_single): median 0.992, range 0.876-1.052
- median replayed/DONE.json sigma ratio = 1.198

## Committed paired gain-null reference

- n_pairs = 12, mean dlogZ = +0.3345, empirical paired SD = 4.746, SEM = 1.370
- UltraNest per-run logzerr mean = 0.215 (range 0.132-0.322)
- NS floor on the mean from UltraNest logzerr (all 12 pairs) = 0.0894

## NS-sampling floor on the paired mean (from measured Higson sigmas)

- gain pairs measured: 5
- typical per-pair diff floor (Higson) = 0.209 nats (UltraNest 0.304)
- NS floor on the mean over 12 pairs (Higson, extrapolated) = 0.0604 nats (UltraNest 0.0879)
- mean dlogZ = +0.3345; empirical SEM = 1.370
- result: null-consistent: the NS-sampling floor is far below both the mean and the empirical SEM; the observed scatter is dominated by real spectrum-to-spectrum variation, not NS sampling noise.

## B1 line-detection threat check

| job | level | counts | Higson sigma | residual median | CI halfwidth | sigma/|resid| | sigma/CIhalf |
|---|---|---:|---:|---:|---:|---:|---:|
| B1_bright_idx3 | bright | 1420 | 0.215 | -892.1 | 301.7 | 2.41e-04 | 7.13e-04 |
| B1_medium_idx2 | medium | 265 | 0.181 | -67.3 | 22.9 | 2.68e-03 | 7.89e-03 |

## Block reproduction (exactly-attached rows)

| job | counts | committed logZ | rerun logZ | delta rerun-committed | Higson sigma |
|---|---:|---:|---:|---:|---:|
| B1_bright_idx3 | 1420 | -1615.87 | -1615.87 | -0.00 | 0.215 |
| B1_medium_idx2 | 265 | -303.50 | -303.50 | -0.00 | 0.181 |
| clean_bright_idx9 | 1675 | -250.97 | -250.97 | -0.00 | 0.165 |
