# PT2 pair-4 check and reused-pair per-channel checkability (2026-09-05)

Read-only pass. No paper file edited, no NS run, no git commit. Repo branch
`_release`, interpreter `.venv/Scripts/python.exe`, run from repo root with
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.

Helper script (committed alongside this report, not run as part of any
pipeline): `outputs/ns_bench/revision_2026-09-02/pair4_check.py`. It imports
`sbixcal.responses`, `sbixcal.simulate`, `sbixcal.models`, `sbixcal.priors`
directly and reproduces, verbatim, the constants and the draw block of
`scripts/paired_ns_gain_check.py` (`BASE_MODEL`, `RESP`, `EXPOSURE=353.4`,
`GAIN=1.03`, `N=12`, `THETA_SEED=20260630`, `POISSON_SEED_BASE=1000`,
`PRIOR_CFG`), i.e. the same seed convention the campaign used: theta block
drawn once from `np.random.default_rng(20260630)` via `P.sample_prior`, then
per pair `i` the clean and gain lambdas are folded through
`R.scale_exposure(base, 353.4)` and `R.gain_shift_obsconf(clean_oc, 1.03)`,
and `data_clean`/`data_gain` are both drawn from
`np.random.default_rng(1000 + i).poisson(...)` (same Poisson seed for both,
common random numbers). This is the same in-process arithmetic
`paired_ns_gain_check.py --dry-run` performs; `pair4_check.py` was written as
a script instead of a CLI wrapper so it could report per-channel diagnostics
(`--dry-run` itself only prints total counts).

## Check A: pair 4

**Command**

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/Scripts/python.exe outputs/ns_bench/revision_2026-09-02/pair4_check.py
```

**Verbatim output (pair 4 and pair 0 sections)**

```
[cfg] exposure=353.4 gain=1.03 n=12 theta_seed=20260630 poisson_seed_base=1000 resp=NGC7793_ULX4_PN model=tbabs_powerlaw_bb
=== pair 4 ===
  n_channels          = 102
  counts_clean total  = 159
  counts_gain  total  = 159
  n channels differing (data) = 0
  max |data_gain - data_clean| (per channel) = 0.0
  sum |data_gain - data_clean| over channels = 0.0
  lambda_clean sum = 153.232047  lambda_gain sum = 154.552709
  max |lambda_gain - lambda_clean| (per channel, model mean) = 0.084810
  sum |lambda_gain - lambda_clean| over channels (model mean) = 2.428407
  max lambda_clean (peak channel rate) = 2.130893

=== pair 0 ===
  n_channels          = 102
  counts_clean total  = 1403
  counts_gain  total  = 1416
  n channels differing (data) = 68
  max |data_gain - data_clean| (per channel) = 15.0
  sum |data_gain - data_clean| over channels = 225.0
  lambda_clean sum = 1382.792675  lambda_gain sum = 1389.067451
  max |lambda_gain - lambda_clean| (per channel, model mean) = 0.834660
  sum |lambda_gain - lambda_clean| over channels (model mean) = 15.577750
  max lambda_clean (peak channel rate) = 24.695885
```

**1. Draw comparison.** Pair 4: 159/159 total counts, matches the recorded
`counts_clean`/`counts_gain`. 102 channels, **0 of 102 differ**, max
|difference| = 0, summed |difference| = 0 — the two 102-length count vectors
are bit-identical. Pair 0 (control): 1403/1416 total counts, matches the
recorded values and the runbook's stated gate value; 68 of 102 channels
differ, max |difference| = 15 counts in one channel, summed |difference| =
225 counts — the draw path clearly does distinguish clean from gain when
counts are high enough, so the identity at pair 4 is not an artifact of a
broken gain application.

**2. Why pair 4 is identical.** Gain used: `GAIN = 1.03` (3% shift, matching
the paired script's default and `new12_counts.json`'s `"gain": 1.03`). The
gain shift does move the *model mean*: `lambda_gain` sum (154.55) vs
`lambda_clean` sum (153.23), and the largest single-channel expected-count
change is **0.0848 counts** (peak channel rate itself is only 2.13
counts/channel at this exposure/theta). So the shift is real but small: at
159 total counts spread over 102 channels, per-channel lambda is mostly
sub-1, and a 0.08-count shift in the Poisson rate essentially never moves the
inverse-CDF draw across an integer threshold for the *same* underlying random
seed (common-random-number pairing: `np.random.default_rng(1000+i)` used
independently for each vector, not literally the same draw call, but at these
tiny rates the quantile function is so flat between successive integers that
a fixed seed reproduces the same integer both times). Compare pair 0: peak
channel rate is 24.7 counts/channel and the largest per-channel lambda shift
is 0.83 counts — an order of magnitude larger absolute perturbation on a
higher-rate channel, which is enough to occasionally flip a threshold (68
channels do). So pair 4's identical clean/gain data, and hence its
bit-identical NS run and `d_paired = 0.0000` exactly, is a genuine
low-count degeneracy of the common-random-number draw, not a script bug —
consistent with `PT2_RESULT.md`'s existing note on this pair. No further
explanation is needed since the vectors are provably identical, not merely
NS-degenerate on non-identical inputs.

**3. Ultranest store comparison for pair 4 (md5).**

```
chains/equal_weighted_post.txt          clean=ddc6f53155ce715836b3eb507f5c3e03  gain=ddc6f53155ce715836b3eb507f5c3e03  [SAME]
chains/run.txt                          clean=9bc85533ef12b864c720dd84fd82d00d  gain=9bc85533ef12b864c720dd84fd82d00d  [SAME]
chains/weighted_post.txt                clean=dceeca0ef97b631f9b096ff09fb53f65  gain=dceeca0ef97b631f9b096ff09fb53f65  [SAME]
chains/weighted_post_untransformed.txt  clean=1f5f5d2a2d4a2ca81b0a0bb0de2b85b3  gain=1f5f5d2a2d4a2ca81b0a0bb0de2b85b3  [SAME]
info/post_summary.csv                   clean=7b846eeb8cfe2db8c0c4e427e1e923ba  gain=7b846eeb8cfe2db8c0c4e427e1e923ba  [SAME]
info/results.json                       clean=001f1c05a9715ee5cd1dbd4da299a7c8  gain=001f1c05a9715ee5cd1dbd4da299a7c8  [SAME]
results/points.hdf5                     clean=d64de9ac0f9fe6af2ea1eaa33dca59f2  gain=d64de9ac0f9fe6af2ea1eaa33dca59f2  [SAME]
```

Every store file that exists in both directories is byte-identical. One
asymmetry: `pair4_clean/debug.log` exists (1.35 MB) but `pair4_gain/debug.log`
does not — this is an UltraNest logging artifact (verbose log written only
once per process in some code paths), not a data or results file, and does
not change the verdict: given identical input data and identical
`seed=i` NS seed, UltraNest reproducing byte-identical output is expected,
not surprising.

## Check B: the five reused pairs (0, 6, 8, 9, 10)

Searched `outputs/ns_bench/higson/runs/gain_pair{0,6,8,9,10}_{clean,gain}/`
(`chains/`, `info/`, `results/`, `extra/`, `plots/`, `debug.log`,
`DONE.json`), `outputs/ns_bench/higson/logs/gain_pair*.log`, and
`outputs/ns_bench/paired_gain_check.jsonl` (2026-07-23) for a per-channel data
vector.

- `results/points.hdf5` per run holds one dataset, `points`, shape
  `(n_samples, 13)` — UltraNest's internal live/dead-point array in
  parameter (u-space/transformed) coordinates plus bookkeeping columns, **not**
  the input count data; root attrs hold only `{"ncalls": ...}`.
- `info/results.json` and `DONE.json` hold posterior summaries, `logz`,
  `ncall`, `niter`, and (in `DONE.json`) a scalar `"counts": 1403` — the
  *total*, never a per-channel array.
- `extra/` and `plots/` are empty for these runs.
- `outputs/ns_bench/higson/logs/gain_pair0_clean.log` logs
  `reconstructed spectrum counts_sum=1403` — again the total only.
- `outputs/ns_bench/paired_gain_check.jsonl` (the July run this script's
  docstring says is not reproducible with the current draw) has keys `i,
  counts_clean, counts_gain, logz_clean, logzerr_clean, logz_gain,
  logzerr_gain, d_paired, wall_s` — no per-channel field, and irrelevant
  anyway since it predates the current draw code.
- No `.npz` file exists anywhere under `outputs/ns_bench/` and no file
  matching `*data*` exists under `outputs/ns_bench/higson/`.

**Per-channel identity is not checkable; only total counts were matched.**
Fresh draw (same helper script, same seeds) vs the values recorded in
`new12_counts.json` / `medium/pairs_merged.jsonl` for all five reused pairs:

| pair | counts_clean (fresh / recorded) | counts_gain (fresh / recorded) |
|---:|---|---|
| 0  | 1403 / 1403 | 1416 / 1416 |
| 6  | 526 / 526   | 521 / 521   |
| 8  | 90 / 90     | 90 / 90     |
| 9  | 2128 / 2128 | 2187 / 2187 |
| 10 | 3659 / 3659 | 3700 / 3700 |

All ten totals match exactly. That is the extent of what is verifiable from
disk: the paper's "reproduces exactly" can be supported for total counts only
(as `new12_counts.json`'s gate already checked); it cannot currently be
supported at per-channel resolution because no per-channel data vector was
ever persisted for the five higson-reused runs.

## Check C: git status (read-only)

**Commands**

```
git ls-files --error-unmatch scripts/paired_ns_gain_check.py scripts/paired_higson_analyze.py scripts/analyze_count_regression.py scripts/import_higson_pairs.py
git status --short -- scripts/paired_ns_gain_check.py scripts/paired_higson_analyze.py scripts/analyze_count_regression.py scripts/import_higson_pairs.py
```

**Verbatim output**

```
scripts/paired_ns_gain_check.py
error: pathspec 'scripts/paired_higson_analyze.py' did not match any file(s) known to git
Did you forget to 'git add'?
scripts/analyze_count_regression.py
error: pathspec 'scripts/import_higson_pairs.py' did not match any file(s) known to git
Did you forget to 'git add'?

 M scripts/paired_ns_gain_check.py
?? scripts/import_higson_pairs.py
?? scripts/paired_higson_analyze.py
```

| file | tracked? | working-tree state |
|---|---|---|
| `scripts/paired_ns_gain_check.py` | tracked | **modified** (unstaged) vs HEAD |
| `scripts/paired_higson_analyze.py` | **untracked** | new file, not in git |
| `scripts/analyze_count_regression.py` | tracked | clean, no diff vs HEAD |
| `scripts/import_higson_pairs.py` | **untracked** | new file, not in git |

## Verdict (three lines)

- **A:** Pair 4's clean and gain draws are bit-identical — yes, identical,
  because at 159 total counts/102 channels the 3% gain shift moves the model
  mean by at most 0.085 counts/channel (peak rate 2.13), too small to flip any
  Poisson threshold under the shared `default_rng(1000+i)` seed; pair 0
  (peak rate 24.7, max lambda shift 0.83) shows the same draw path does
  differ 68/102 channels when counts are high enough, and the two UltraNest
  stores for pair 4 are md5-identical on every shared file, consistent with
  identical inputs and identical `seed=i`.
- **B:** Per-channel identity for the five reused pairs (0, 6, 8, 9, 10) is
  **not checkable** from anything on disk — `points.hdf5` stores NS
  parameter-space samples not input data, `DONE.json`/`results.json`/logs
  store only the total `counts`, and no `.npz` or data-array sidecar exists
  anywhere under `outputs/ns_bench/`; only total counts were matched (all 10
  totals reproduce exactly: 1403/1416, 526/521, 90/90, 2128/2187, 3659/3700).
- **C:** `paired_ns_gain_check.py` and `analyze_count_regression.py` are
  tracked (the former modified, unstaged; the latter clean);
  `paired_higson_analyze.py` and `import_higson_pairs.py` are both untracked
  (new, not in git).
