# Sub-percent paired Gamma-bias probe (2026-09-08)

## Purpose

Measure the paired Gamma-bias point estimate (and its per-source scatter,
`bias_std`, emitted for free) at sub-percent gain amplitudes, for Section 8
of the ApJ manuscript (`\subsection{Realistic amplitude and limits}`,
`main.tex` L361). Requested by an external review received 2026-09-08
(`P\journal_revision_apj\REVIEW_EXTERNAL_2026-09-08.md` §1), scoped read-only
in `P\journal_revision_apj\REVIEW2_COMPUTE_SCOPING.md` (ITEM 1), and launched
here. This directory supersedes nothing; it is a NEW sibling of
`outputs/gain_marg/subpercent_bias_2026-09-02/` (prep-only, no compute run,
see that dir's `PREP.md`).

**Bugfix note (2026-09-08, read before touching these scripts again)**: the
first launch attempt crashed all 10 cells with `KeyError: 1.03` — a
pre-existing bug in `make_paired_population()` in BOTH
`outputs/gain_marg/eval_gainmarg_paired.py` and `..._bright.py` (the `return`
line hardcoded `x[1.03]`/`lam[1.03]` instead of reading the actual `--gain`
value back out of the dict). It was invisible at the default `--gain 1.03`
and had never been exercised at any other gain before this probe. Fixed
(minimal, both scripts) alongside the `--seed` flag added in the same
session. Full crash trace, fix, and verification:
`P\journal_revision_apj\REVIEW2_LAUNCH_REPORT.md` STEP 3/4.

## Date

Launched 2026-09-08 (IST). Branch `apj-artifacts-2026-09-08` throughout;
no `git add`/`commit`/`checkout`/`reset`/`stash` performed by this campaign.

## Exact commands (probe: single seed, amplitudes 0.001-0.01)

```bash
cd "<repo>/sbi-xray-calibration"
PY="./.venv/Scripts/python.exe"
BASE="outputs/gain_marg/subpercent_bias_2026-09-08"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

# medium, for amp in 0.001 0.002 0.003 0.005 0.01 (g = 1+amp):
"$PY" outputs/gain_marg/eval_gainmarg_paired.py \
  --gain <g> --n-test 500 --out "$BASE/medium/g<amp>" \
  > "$BASE/medium/g<amp>/run.log" 2>&1

# bright, same amplitudes:
"$PY" outputs/gain_marg/eval_gainmarg_paired_bright.py \
  --gain <g> --n-test 500 --out "$BASE/bright/g<amp>" \
  > "$BASE/bright/g<amp>/run.log" 2>&1
```

Driven by `run_probe_medium.sh` / `run_probe_bright.sh` in this directory,
each looping over the 5 amplitudes sequentially (one job per level, two
jobs concurrent = 8 of 16 cores), launched detached so they survive the
orchestrating shell's exit. See `run_phase2.sh` (not launched by this
campaign) for the seed-averaging follow-up.

## Convention (matches the shipped 3% design exactly, `main.tex` L326)

- N = 500 same-parameter (theta) pairs per run: ONE draw of the paired
  thetas from the physical prior, each folded through the response at
  g=1.00 (clean arm) AND at the shifted g (gain arm).
- Common Poisson random numbers (CRN): the same per-spectrum seed reseeds
  the Poisson draw for both arms of a pair, so the pair differs only by the
  gain-shift response fold, not by independent shot noise.
- Default seed family: `SEED = 20260611` (module constant in both
  production scripts), with `SEED_THETA = SEED+60000`,
  `SEED_POISSON_BASE = SEED+70000`, and four `SEED_SAMPLE` offsets for the
  four (flow, case) sampling draws — unchanged from the shipped 3% runs.
  This probe uses the default seed only (no seed averaging); the `--seed`
  flag added in STEP 4 below (this campaign) makes seed-averaging possible
  for a phase-2 follow-up without touching this probe's own results.
- Posterior-mean Gamma is NOT the reported statistic here — the paired
  headline (`results.paired.gamma_bias_delta.{fixed,gain_marg}.mean`) is the
  mean of the PAIRED per-source difference, using the posterior **median**
  Gamma per spectrum per arm (`metrics()`, `med[:, idx] - truth[:, idx]`,
  then the paired delta is `med_gain[:,GAMMA_I] - med_clean[:,GAMMA_I]`,
  `paired_stats()` computes mean/sd/se/ci95 of that array, `se = sd/sqrt(n)`).
  (Correction to the STEP 1 instruction's shorthand: the scoping report
  ITEM 1(a) confirms `bias_mean`/`bias_median` in `cases.*.gamma` use
  posterior median; the paired delta field uses the same per-spectrum
  median, differenced. No posterior-mean field is used in the shipped
  headline statistic.)
- `g = 1 + amp`, amp in {0.001, 0.002, 0.003, 0.005, 0.01} (0.1%-1.0%).
- N_SAMPLES = 1000 posterior draws per spectrum (module default, matches
  shipped runs; NOT the adversary sweep's N_SAMP=3000).
- `reject_outside_prior=True` with an 8s per-spectrum cap, falling back to
  unrejected + top-up, then clip-to-box before percentiles (unchanged from
  the production scripts, see their module docstrings).

## Directory layout

```
subpercent_bias_2026-09-08/
  README.md                  (this file)
  run_probe_medium.sh        (STEP 2, launched STEP 3)
  run_probe_bright.sh        (STEP 2, launched STEP 3)
  probe_medium.progress      (DONE lines, written by run_probe_medium.sh)
  probe_bright.progress
  probe_medium.launcher.log / .err   (PowerShell Start-Process redirect)
  probe_bright.launcher.log / .err
  PIDS.txt                   (STEP 3, PIDs + start time)
  run_phase2.sh               (STEP 5, NOT launched — supervisor decides after probe scatter is known)
  run_regression_test.sh      (STEP 5, NOT launched — validates the --seed edit is a no-op at the default)
  medium/g<amp>/paired_gain_bias_medium.json + .png + run.log
  bright/g<amp>/paired_gain_bias_bright.json + .png + run.log
```
