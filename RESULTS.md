# Results log

Quantitative findings recorded as they appear. Each entry is reproducible from
config + seed with the repo `.venv`.

---

## GO/NO-GO robustness correction (2026-06-12), the count-regime claim is retracted

This entry supersedes the earlier "calibration degrades / NPE becomes
over-confident at high counts" trend claim (recorded below in the Phase-2 and
Phase-3 sections, left in place and annotated as superseded). A GO/NO-GO
robustness pass (reseeding the bright production flow under three new seeds and
retraining one variant with the epoch cap lifted) shows the over-confidence was
a single-flow training artifact. It is not a property of the high-count regime.
The story is now:

> One production flow, which passed every recovery-quality check (mean Pearson r
> 0.88, all 5 posteriors shrinking monotonically with counts), was silently
> miscalibrated: coverage deviation 0.113, SBC collapsed (KS p ≈ 0 on 4/5
> params). It had hit its 150-epoch training-budget cap with a train/val gap
> (−14.91 / −13.36). Three reseeds and an uncapped retrain on the same data are
> near-calibrated (raw coverage deviation **0.014–0.033**, SBC uniform). The trust
> toolkit caught the bad flow (SBC + coverage), conformal repaired it
> (0.113 → 0.026), and the IS low-ESS diagnostic fired on exactly that flow
> (97% low-ESS). Calibration of a single trained flow cannot be assumed from its
> recovery metrics; every flow you intend to deploy must be validated directly.

### The evidence (`outputs/gonogo/summary.jsonl`)

All five bright-level flows train on the SAME ~10 000-count production data; they
differ only in the training seed and (for `uncapped`) the epoch cap. Coverage
deviation = mean |empirical − nominal| over the 5 parameters and 12 nominal
levels; SBC `ks_p_min` = the smallest per-parameter rank-uniformity KS p-value
(small ⇒ ranks non-uniform ⇒ miscalibrated).

| variant | epochs (cap) | median counts | raw cov dev | conformal cov dev | SBC ks_p_min | verdict |
|---|---|---|---|---|---|---|
| **production (orig.)** | **150 / 150 (cap hit)** | 9 982 | **0.113** | 0.026 | **≈ 0 (4/5 fail)** | **silently miscalibrated** |
| reseed 101 | 83 / 150 | 10 096 | 0.033 | 0.027 | 0.084 (pass) | near-calibrated |
| reseed 202 | 151 / 150 (cap hit) | 9 953 | 0.031 | 0.019 | 1.2e−8 (1 param) | near-calibrated (coverage), 1-param SBC flag |
| reseed 303 | 116 / 150 | 10 022 | 0.022 | 0.020 | 0.016 (pass) | near-calibrated |
| **uncapped** | **162 / 400 (converged)** | 9 982 | **0.014** | 0.022 | 0.028 (pass) | **calibrated, SBC uniform** |

The original production flow is the outlier (dev 0.113); every robustness variant
lands at dev 0.014–0.033, a 3–8× improvement from more (or differently-seeded)
training. The cleanest control is `uncapped`: same data, cap lifted from 150 to
400, converged at 162 epochs, dev 0.014 and SBC uniform. That isolates the cause.
The original flow's over-confidence was an undertraining / epoch-cap artifact, and
the high-count regime was not the driver. (Seed 202 also hit the 150-cap and
keeps one ∪-shaped SBC param while its coverage is already fine, consistent with
the cap being the driver.)

The result here is about a single flow, not a count-regime law: a miscalibrated
flow can pass recovery checks and be caught only by the calibration suite. SBC and
coverage flagged it, conformal recalibration repaired it (0.113 → 0.026), and the
importance-sampling low-ESS diagnostic fired on exactly that flow (Phase-3, 97%
low-ESS at bright). Recovery quality does not certify
calibration; run SBC + a coverage test on every flow you ship. The detection
results are seed-robust and unaffected (spot check on the seed-101 flow: B1
strongest line D1 AUC **0.995**, B4 3% gain D1 AUC **0.582** ≈ chance, the same
qualitative story as the production flow).

---

## Robustness corrections (2026-06-12)

A self-review flagged several claims as framed too strongly or comparing
things that are not apples-to-apples. The findings and fixes applied (all numbers
below recomputed from committed artifacts):

- **B1 (blocker, D3 not apples-to-apples).** D3/marginal-C2ST is a supervised
  two-sample **population-separability** statistic (its per-spectrum score is a
  supervised classifier's OOF class-1 probability on *labeled* clean-vs-misspec
  populations), and it is not a per-spectrum unlabeled novelty detector like D1/D2.
  **Fix:** detection tables now segregate D3 into its own column group,
  footnote-labelled as such; added the recomputed **D3 control-cell floor** (on the
  weakest B1 cells, norm 5e-6, D3 cv-accuracy ≈ **0.66** while AUC ≈ **0.43–0.54**,
  at/below chance); added the summary sentence comparing the two detector classes;
  the money-plot panel (b) now plots D1/D2 as the per-spectrum ROC curves and
  overlays D3 dotted-grey as "population separability."
- **Over-generalized high-count claim.** The high-count over-confidence is now stated
  as **conditional** on our setup (wide log-uniform norm prior, single-round
  amortized NPE, training-budget-capped flow), contrasted with Barret & Dupourqué I's
  restricted-prior recovery; contribution line = "we add the rank-based SBC they
  omit, under a harder setup." Result kept (∪-shaped SBC ranks verified).
- **M2 (TARP wrong lesson).** Recomputed from `outputs/calibration/bright/tarp.npz`:
  signed ATC = **−0.0015** cancels (curve bows above the diagonal at α<0.5, below at
  α>0.5); **abs-area = 0.053, max|ECP−α| = 0.102 at α≈0.19** (vs abs-area 0.005/0.012
  at faint/medium). Lesson changed to "a SIGNED area summary cancels and hides it,
  so read the curve / use abs-area or KS." Added committed figure
  `outputs/diagnostics/tarp_bright_curve.png`.
- **M3 (D2 vs Schmitt scoping).** D2 = the posterior-trained, un-regularized,
  near-sufficient embedding variant, distinct from Schmitt+23/24's MMD-regularized overcomplete
  summary; cite their Eq. 12–13 (a misspecification preserving the summary
  distribution is provably invisible) as why a gain shift evades a near-sufficient
  embedding; B4 negative result scoped to "these three detectors."
- **Minors.** m1: ΔΓ "+0.20 softer" is bright-only (faint −0.07, scatter-dominated).
  m2: money-plot panel (a) coverage = mean over the 5 marginals (not joint). m4: IS
  2000-draw budget "consistent with" Paper III's 200k–400k sequentially-refined
  framework, not implied-observed. m5: NS-table note, capped rows
  (`n_like_evals == cap`) are non-converged, logZ is a lower bound, never compare
  logZ across capped/uncapped; live run uses `--max-ncalls 120000` (CLI override of
  the config's 400000). m6: fixed latent `hash()` seeding in
  `misspec.generate_family_point` (per-process salted → non-reproducible) with a
  stable sha1 helper.

Commits are path-scoped: B1 (one), M1+M2+M3+m4 (one), minors m1/m2/m5/m6 + wrap-up
(one). Full pytest green throughout.

---

## Phase 0: prototypes / de-risking (2026-06-11)

### B4 gain-shift mechanism (highest-risk item): SOLVED in-place

- Mechanism: rescale the `ObsConfiguration` unfolded energy-grid coords
  (`e_min_unfolded` / `e_max_unfolded`) by a `gain` factor via
  `obsconf.copy(deep=True).assign_coords(...)`. No RMF FITS rewrite needed.
- Proof (`outputs/diagnostics/gain_shift_check.png`, `scripts/proto_gain_shift.py`):
  tbabs*(powerlaw + 6.4 keV Gauss), flux-weighted Fe-K centroid:
  - nominal  → 6.401 keV
  - +2% gain → 6.274 keV  (shift -0.127 keV = -2.0%)
  - -2% gain → 6.530 keV  (shift +0.129 keV = +2.0%)
- Shift magnitude matches the gain factor to <1%. Full write-up:
  `docs/gain_shift_notes.md`. Risk R1 closed.

### jaxspec fakeit throughput (CPU jax, tbabs*powerlaw, EPIC-pn 102-ch)

(`scripts/proto_benchmark.py`, post-JIT-warmup, includes device->host sync)

| N spectra | wall time | rate |
|-----------|-----------|------|
| 1,000     | 0.77 s    | ~1,300 spec/s (includes fixed overhead) |
| 10,000    | 0.66 s    | ~15,300 spec/s (steady state) |
| 100,000   | ~6.6 s (extrapolated) | |
| 200,000   | ~13 s (extrapolated)  | |

Conclusion: CPU simulation is NOT a bottleneck for 1e5-2e5 training spectra
(risk R3 closed). WSL2/GPU-jax unnecessary.

### B3 continuum-swap decision: CUSTOM BREMSSTRAHLUNG (not Diskbb)

- Tried the custom `AdditiveComponent` path first.
  A Gaunt-factor-free thermal-bremsstrahlung emissivity
  `M(E) = K * E^-1 * exp(-E/kT)` was implemented as a ~10-line jaxspec
  component (`scripts/proto_brems.py`, later promoted into `src/sbixcal`).
- It instantiates, folds through EPIC-pn via the standard
  `fakeit_for_multiple_parameters` path, and produces a finite continuum with a
  visibly different shape from a powerlaw (brems/powerlaw count ratio runs
  0.54 -> 0.14 from low to high energy; `outputs/diagnostics/brems_check.png`).
- The custom component did NOT fight back, so **B3 uses the custom brems
  component**, not the Diskbb fallback. The dropped Gaunt factor (slowly varying
  O(1) log term) is an acceptable, documented approximation for a
  misspecification *template* (the point is a wrong continuum family, not a
  calibrated plasma model). Risk R4 closed via option (i).

### NPE device micro-benchmark (default NSF flow, 2k toy sims, 30 epochs)

| device | wall time |
|--------|-----------|
| CPU    | 4.43 s    |
| GPU (RTX 4050) | 4.84 s |

Conclusion: for the small default flow, CPU is as fast / slightly faster than
GPU, matching the sbi-docs expectation that GPU gives no speedup for small NSF
flows. **Default training device for plain flows = CPU; GPU reserved for the
Phase-2 1D-CNN embedding net / large batches.** Either device meets the compute
budget.

---

## Phase 1: simulation infrastructure (2026-06-11)

### Priors (Barret & Dupourque 2024, A&A 686, A133, Table 1)

- Dev model (Model 1, tbabs*powerlaw): N_H ~ U[0.1,0.3] (1e22), Gamma ~ U[0.5,2.5]
  taken EXACTLY from B&D24 Tab.1. NormPL kept log-uniform (their convention) but
  the 4-decade window shifted from [0.01,100] to **[1e-4,1e-2]**.
- Prod model (Model 2, tbabs*(powerlaw+bbodyrad)): N_H ~ U[0.15,0.35],
  Gamma ~ U[1,3], kTbb ~ U[0.3,3.0] EXACT from B&D24 Tab.1; NormPL shifted
  [0.1,10] -> [1e-4,1e-2] and NormBB shifted [100,1000] -> [1e-2,1] (log-uniform).
- Reason for the norm shift: B&D24's norm ranges are tuned to their NICER
  response; the bundled EPIC-pn response here has ~10x larger effective area, so
  the raw ranges give 1e4-1e8 counts (sub-second exposures). Shifting the
  log-uniform windows down (same widths/convention) keeps the count regimes at
  realistic ks-scale exposures. B&D24 themselves restrict this prior for the
  same count-control reason (their Sect. 3.2). Documented in the configs and
  README.

### Exposure calibration (empirical, deterministic n=20000 probe)

Counts scale linearly with exposure (verified: rescaling the transfer matrix by
exposure/native gives exact linear count scaling at fixed 102-channel grouping).
Median total expected counts over the prior was matched to 100 / 1000 / 10000.

| model       | level  | exposure (s) | achieved median counts |
|-------------|--------|--------------|------------------------|
| modelA_dev  | faint  | 64.9         | 97 (gen: 98)           |
| modelA_dev  | medium | 649.1        | 968 (gen: 952)         |
| modelA_dev  | bright | 6491.4       | 9684 (gen: 9944)       |
| modelA_prod | faint  | 35.3         | 98                     |
| modelA_prod | medium | 353.4        | 983                    |
| modelA_prod | bright | 3534.0       | 9828                   |

(The log-uniform norm makes the per-spectrum count distribution heavy-tailed;
the MEDIAN is the stable, well-defined target and is what is calibrated. Median
is stable to ~4% across seeds at n=20000.)

### Misspecification families (configs/misspec_modelA_dev.yaml)

All four generate finite, sensible spectra (smoke-tested + unit-tested):
- B1 Fe-K Gauss at 6.4 keV, line-norm grid [0, 5e-6, 2e-5, 8e-5, 3e-4]
  (0 = control). Added counts are >80% localized in the 5.8-7.0 keV window
  (test asserts this).
- B2 Tbpcf partial covering, f-grid [1.0, 0.9, 0.7, 0.5, 0.3]; lower f leaks
  unabsorbed flux -> higher counts (test asserts monotonicity).
- B3 custom thermal bremsstrahlung (NOT Diskbb), kT-grid [10,5,3,1.5,1.0] keV;
  low kT is soft/curved (test asserts soft>hard at kT=1.5).
- B4 gain shift, percent-grid [0,0.5,1,2,3] (0 = control); shifts Fe-K centroid
  by the gain factor (test asserts +/-2% -> -/+0.13 keV).

### Datasets

`data/sim/<name>.npz`, skip-if-exists (crash-resumable), gitignored. Each npz
carries theta, x (Poisson counts), param_names, e_min/e_max, exposure_s, seed,
median_total_counts. modelA_dev_{faint,medium,bright} generated and verified.

**Seeding fix.** `misspec.generate_family_point` previously folded
Python's built-in `hash(family + slabel)` into the per-dataset RNG seed. Python
salts `str` hashing per process (`PYTHONHASHSEED` randomization), so that seed
(and hence the generated misspecified dataset) was **not reproducible across runs
or sessions**. Replaced with a `_stable_hash` helper (sha1 of the string, taken
mod 100000), verified identical across processes and `PYTHONHASHSEED` values. One-
line fix; the on-disk datasets are gitignored and regenerated from config + seed,
so no committed artifact changes, and the *reproducibility guarantee* is now
real.

### Tests

`tests/test_simulate.py`: **12 passed in ~9.8 s** (repo venv). Covers linear
count scaling (exposure + norm), B1 localization + monotonicity, B4 feature
shift + 0%-is-nominal, identical-seed reproducibility, prior-bound + log-uniform
correctness, B2/B3 sanity.

---

## Phase 2: NPE training (2026-06-11)

### Architecture

- **Embedding net** (`SpectrumCNN`, `src/sbixcal/train_npe.py`): 1-D CNN over the
  102-channel EPIC-pn counts spectrum. 2 conv blocks (Conv1d k=5 → ReLU →
  MaxPool, channels 1→16→32) + MLP head → embed_dim (16 dev / 20 prod).
  **~55k params** (dev). Counts are normalized with **`log1p`** *inside* the
  module (Poisson counts span 0→thousands under the log-uniform norm prior;
  log1p compresses the dynamic range, is defined at 0, and keeps shot noise
  roughly homoscedastic). Because the CNN normalizes internally, sbi's outer
  `z_score_x` is set to **`none`** (z-scoring the raw heavy-tailed counts caused
  precision-loss warnings and helped nothing); theta is still z-scored.
- **Flow:** NSF via `posterior_nn(model="nsf", embedding_net=cnn)`,
  hidden_features=50, num_transforms=5, num_bins=10.
- **One flow per count level** (3 per model), NOT amortized over exposure:
  amortizing would let the net trade information across count regimes, a
  confound for a calibration analysis (coverage is exposure-dependent). Fixed-
  exposure flows match Barret & Dupourqué (2024). Documented in the config
  headers.
- **Checkpoints** (`outputs/models/<name>/`): `flow_state.pt`, `arch.json`,
  `training_loss.npy`, `validation_loss.npy`, `summary.json`, `config.yaml`.
  Cold-loadable with `train_npe.load_posterior(<dir>)`, which rebuilds the exact
  architecture from `arch.json`, loads the state dict, returns a ready
  DirectPosterior. No training data needed; verified bit-identical to the
  in-memory flow (test).

### Device choice: CPU (CNN-embedding training too)

`scripts/bench_cnn_device.py`, real NSF+CNN stack, N=10k, 15 fixed epochs,
modelA_dev_medium, post-warmup:

| device | total | s/epoch |
|--------|-------|---------|
| CPU    | 22.7 s | 1.42 |
| GPU (RTX 4050) | 48.6 s | 3.04 |

**CPU is ~2× faster than GPU even WITH the 1-D CNN embedding.** The model is
tiny (~55k params), batch 200, 102-channel input, so kernel-launch + host↔device
transfer overhead dominates the trivially small convs. This extends the Phase-0
plain-flow finding to the CNN case. **All Phase-2 training runs on CPU.** (GPU
remains available via `--device cuda`; it is just slower here.)

### Production flows: trained (CPU, 50k sims/level, NSF+CNN, patience 20, cap 150 ep)

`outputs/models/train_npe_prod_{faint,medium,bright}/` (each: `flow_state.pt`,
`arch.json`, `summary.json`, `config.yaml`, loss curves). Per-flow training
summary read from each checkpoint's `summary.json`; wall-clock is the elapsed
training time of each level on CPU (from the run log / checkpoint timestamps).

| level | ~counts | exposure (s) | epochs (early-stop) | best val loss | final train / final val | wall-clock |
|-------|---------|--------------|---------------------|---------------|--------------------------|-----------|
| faint  | 98    | 35.3   | 60  | −10.077 | −10.189 / −10.035 | ~7 min  |
| medium | 986   | 353.4  | 132 | −13.086 | −13.086 / −12.966 | ~18 min |
| bright | 9982  | 3534.0 | 151 (cap) | −16.336 | −14.907 / −13.361 | ~23 min |

Notes: validation loss (negative log-prob of the NSF) deepens monotonically with
count level (more counts ⇒ sharper learnable posteriors, as expected). The
**bright** flow hits the 150-epoch cap with a final-train/final-val gap
(−14.91 vs −13.36) that the faint/medium flows do not show: it is starting to
**over-fit the training set**, the first hint of the over-confidence that the
recovery + Phase-3 coverage numbers below confirm. Early stopping (patience 20)
fired before the cap for faint and medium.

### Production flows: Phase-2 sanity validation (`scripts/run_validate_npe.py`)

200 fresh disjoint sims per level (not the rigorous SBC/TARP pass, which is
Phase 3). Recovery = Pearson r of truth vs posterior median; cov90 = fraction of
truths inside the per-parameter 90% equal-tailed credible interval (a cheap
coverage proxy); widths as a fraction of the prior width. Figures:
`outputs/diagnostics/npe_recovery_{faint,medium,bright}.png`; numbers archived in
`outputs/diagnostics/validation_prod_summary.json`.

**Recovery (Pearson r) + joint 90% coverage:**

| level | ~counts | r(N_H) | r(Γ) | r(normPL) | r(kT_bb) | r(normBB) | mean r | joint 90% cov |
|-------|---------|--------|------|-----------|----------|-----------|--------|---------------|
| faint  | 98   | 0.289 | 0.670 | 0.986 | 0.789 | 0.719 | 0.690 | 0.59 |
| medium | 986  | 0.511 | 0.793 | 0.993 | 0.897 | 0.792 | 0.797 | 0.60 |
| bright | 9982 | 0.718 | 0.841 | 0.996 | 0.925 | 0.898 | 0.876 | 0.41 |

Recovery improves monotonically with counts (mean r 0.69 → 0.80 → 0.88); the
power-law norm is recovered nearly perfectly everywhere (r ≈ 0.99), N_H is the
hardest parameter (r 0.29 at faint, the absorption being weakly constrained at
~100 counts). **The joint 90% coverage is already below nominal at every
level and is WORST at bright (0.59 / 0.60 / 0.41)**, so the flows grow
*over-confident* as counts grow. This is the raw-NPE
under-coverage finding; Phase 3 quantifies it with SBC/TARP and shows
which recalibration repairs it.

**Posterior width vs count level (median 90%-CI as a fraction of prior width):**

| param | faint (~98) | medium (~986) | bright (~9982) | monotone shrink? |
|-------|-------------|---------------|----------------|------------------|
| N_H    | 0.879 | 0.813 | 0.669 | yes |
| Γ      | 0.734 | 0.460 | 0.309 | yes |
| normPL | 0.107 | 0.051 | 0.047 | yes |
| kT_bb  | 0.566 | 0.310 | 0.227 | yes |
| normBB | 0.438 | 0.329 | 0.155 | yes |

**Every parameter's posterior shrinks monotonically with count level** (verdict:
YES, all 5/5), so the flows learn (widths narrow as data informs). The shrinkage
combined with the *falling* coverage is the failure mode of interest:
the posteriors get narrower faster than they get more accurate, so coverage
degrades. Phase 3 is where that is named and (partly) fixed.

**Phase 2 closed:** production flows trained + validated; recovery monotone in
counts, widths monotone in counts, raw-NPE under-coverage carried into Phase 3.
**[Correction 2026-06-12:** the bright under-coverage seen here
turned out to be a single-flow epoch-cap artifact, and the count-regime trend was
the wrong reading (see the GO/NO-GO correction at the top). The recovery-monotone /
width-monotone findings are unaffected; what changed is that "the bright flow
over-fits because it's the high-count regime" became "*this* bright flow over-fit
because it hit the epoch cap, and recovery quality did not warn us." A
recovery-passing flow was silently miscalibrated.**]**

---

## Phase 3: calibration testing + recalibration (2026-06-11): CLOSED

The Phase-3 suite is **implemented, tested, verified end-to-end, and run on the
three production flows**. The calibration results (the count-regime trend,
the recalibration before/after, and the IS low-ESS diagnostic) are in the
**"Production calibration results"** subsection below; the code/test description
that follows it is retained for provenance. Nothing here reads or writes
production checkpoints beyond cold-loading a finished one.

### Production calibration results (the count-regime story)

Run: `scripts/run_calibration.py --config configs/calibration.yaml` against
`outputs/models/train_npe_prod_{faint,medium,bright}/` (Model A production,
tbabs·(powerlaw+blackbody), 5 params). Artifacts per level in
`outputs/calibration/<level>/` (sbc/tarp/coverage/is npz + figures +
`summary.json`). Money plot: `outputs/diagnostics/coverage_money_panel.png`
(`scripts/make_coverage_money_panel.py`).

> **[SUPERSEDED 2026-06-12, see "GO/NO-GO robustness correction" at the top of
> this file.]** The "degrades at high counts" framing below was retracted by the
> robustness pass: the over-confidence is a single-flow epoch-cap/undertraining
> artifact (reseeds + an uncapped retrain on the same data give dev 0.014–0.033),
> and it is not a count-regime law. The numbers in this section are correct *for
> the one capped production flow*; the **generalization to "the high-count regime"**
> is the part that is wrong, and it is corrected above. The result that holds is
> that the trust toolkit caught this single bad flow. The text below is kept
> verbatim for provenance.

**Finding (RETRACTED as a regime claim, see above), for the single
capped production flow, calibration is good at low/medium counts and degrades at
high counts** (the opposite of the naive "low-count regime is worst" guess; the
count-regime trend the original spec asks us to report).
At ~100 and ~1000 counts the raw NPE is already near-diagonal (mean |emp−nominal|
≈ 0.010 / 0.013, TARP near-perfect, SBC rank-uniformity passing on most params).
At ~10000 counts the flow becomes over-confident: raw coverage falls to
0.36 / 0.51 / 0.76 at nominal 50 / 68 / 90 (mean |emp−nominal| = 0.113), SBC KS
p-values collapse to ~0 on 4/5 params, and conformal recalibration repairs it to
0.46 / 0.64 / 0.88 (deviation 0.113 → 0.026). This is the over-confidence first
seen in the Phase-2 validation (joint-90% coverage 0.59/0.60/**0.41**) now named
rigorously: the bright flow hit the 150-epoch cap and over-fit (train/val gap),
and at high counts the posteriors narrow faster than they stay accurate.

**Scope of this claim (SUPERSEDED by the GO/NO-GO correction at
the top), the over-confidence is conditional on this setup and is not a universal
NPE-at-high-counts law.** An earlier revision already weakened the claim to
"conditional on this setup"; the later GO/NO-GO pass went further and retracted
the count-regime trend entirely (reseeds + an uncapped retrain on the same data
calibrate fine, dev 0.014–0.033), so the driver is the **(iii) training-budget
cap** below, and not (i)/(ii). This paragraph records the original M1
scoping; the operative conclusion is the correction at the top of the file. Three
setup choices were thought to drive it:
(i) a deliberately wide **2-decade log-uniform norm prior within each count level**
(a heavy-tailed brightness axis to amortize over), (ii) **single-round amortized
NPE** (no sequential proposal refinement), and (iii) a **training-budget-capped
flow** (the bright run hit the 150-epoch cap with a train/val gap, Phase-2:
−14.91 / −13.36, while faint/medium early-stopped before the cap and stay
calibrated). **Barret & Dupourqué I (arXiv:2401.06061, §3.2–3.3) recover excellently
at 10⁴–10⁵ counts using a deliberately RESTRICTED prior, and never report a
rank-based SBC test.** Our contribution line is therefore precise: *we add the
rank-based calibration test they omit, under a deliberately harder wide-prior
single-round setup, and find over-confidence conditional on this setup.* The
result itself is real (∪-shaped SBC ranks at bright,
verified); only the generality is scoped.

#### SBC (Talts+18) + TARP (Lemos+23)

`sbi.diagnostics.{run_sbc,check_sbc,run_tarp,check_tarp}`, N=1000 fresh sims from
the same prior/simulator each flow was trained for. SBC: per-parameter rank KS
p-value (large = uniform ranks = calibrated) and C2ST-of-ranks (≈0.5 = good).
TARP: joint area-to-curve ATC (0 = perfect) + KS p-value.

| level | ~counts | SBC KS p (per param: N_H, Γ, normPL, kT, normBB) | C2ST(ranks) mean | TARP ATC | TARP KS p |
|-------|---------|---------------------------------------------------|------------------|----------|-----------|
| faint  | 98   | 0.657, 0.034, 0.109, 0.930, 0.976 | 0.575 | +0.0003 | 1.000 |
| medium | 986  | 0.856, 0.193, 0.323, 0.856, 0.452 | 0.569 | +0.0072 | 1.000 |
| bright | 9982 | **0.0, 1e−21, 0.0, 2e−14, 6e−40** | 0.645 | −0.0015 | 0.589 |

faint/medium SBC ranks are consistent with uniform (all KS p > 0.03, C2ST≈0.55–0.58);
**bright SBC fails hard on 4/5 params** (KS p ≈ 0). The rank histograms are
∪-shaped (truth in the tails of the posterior too often), the textbook
over-confidence signature.

**TARP, read the curve and do not rely on the signed-area summary alone.** The bright TARP signed area-to-curve is **ATC ≈ −0.002**
(recomputed from `outputs/calibration/bright/tarp.npz`: −0.0015), which looks
benign and would, read alone, falsely pass the bright flow. That signed area
cancels: the bright ECP curve bows above the diagonal at α < 0.5 and below it at
α > 0.5 (mean ECP−α = **+0.080** for α<0.5, **−0.004** for α>0.5), so the two
lobes nearly annihilate in a signed integral. The unsigned summary catches the
over-confidence cleanly: the bright curve's **abs-area is 0.053** and its
**max|ECP−α| = 0.102 at α ≈ 0.19**, versus abs-areas of only **0.005 / 0.012 at
faint / medium**. TARP does catch the over-confidence if you read the curve
(or use abs-area / a KS-style statistic); the earlier framing that "joint TARP is
insufficient" was imprecise, since it is the signed ATC scalar alone that
hides it, while the per-parameter SBC and the per-parameter credible-interval
coverage below independently flag it. The committed figure
`outputs/diagnostics/tarp_bright_curve.png` (built by `make_support_figs.py`)
shows the curve with the |ECP−α| area shaded.

#### Coverage at nominal {50, 68, 90}: raw → conformal → IS-refined

Per-parameter empirical equal-tailed credible-interval coverage, **mean over the
5 parameters**, at the three standard nominal levels. `raw` and `conformal` are on
the conformal before/after test set; `IS-refined` / `IS-refined (ok-ESS)` are on
the separate IS-coverage test set (raw on that set agrees with `raw` to
sampling noise). "ok-ESS" drops the cases the low-ESS flag trips.

| level | nom | raw NPE | conformal | IS-refined (all) | IS-refined (ok-ESS) |
|-------|-----|---------|-----------|------------------|---------------------|
| faint  | 50 | 0.487 | 0.491 | 0.449 | 0.451 |
| faint  | 68 | 0.664 | 0.674 | 0.615 | 0.619 |
| faint  | 90 | 0.884 | 0.886 | 0.844 | 0.844 |
| medium | 50 | 0.484 | 0.501 | 0.457 | 0.453 |
| medium | 68 | 0.653 | 0.657 | 0.608 | 0.611 |
| medium | 90 | 0.892 | 0.890 | 0.839 | 0.839 |
| bright | 50 | **0.358** | 0.459 | 0.371 | 0.440 |
| bright | 68 | **0.515** | 0.638 | 0.543 | 0.720 |
| bright | 90 | **0.758** | 0.882 | 0.759 | 0.960 |

Faint/medium need no repair (raw already ≈ nominal). At bright, conformal is the
effective fix across all three levels (0.36/0.51/0.76 → 0.46/0.64/0.88).
IS-refinement on all cases barely moves bright (0.37/0.54/0.76) because almost
every case is low-ESS (next table); restricted to the few ok-ESS cases it
over-corrects (0.44/0.72/0.96) on a tiny, selection-biased subsample, which is
why the low-ESS flag matters.

#### Coverage deviation (mean |emp−nominal| over all levels & params) + low-ESS

| level | ~counts | raw dev | conformal dev | IS-all dev | IS low-ESS (cov test set) | low-ESS frac |
|-------|---------|---------|---------------|------------|---------------------------|--------------|
| faint  | 98   | 0.010 | 0.007 | 0.046 | 10 / 150  | 0.07 |
| medium | 986  | 0.013 | 0.010 | 0.041 | 18 / 150  | 0.12 |
| bright | 9982 | 0.113 | 0.026 | 0.100 | **145 / 150** | **0.97** |

#### IS low-ESS is the expected diagnostic at high counts (Paper III relation)

The IS-refinement reuses the **exact Poisson likelihood** (the same one the
Phase-5 NS benchmark will use), reweighting NPE draws by
`w = p_Poisson(x|θ)·p(θ)/q_NPE(θ|x)` and computing the Kish ESS (Barret &
Dupourqué Paper III, arXiv:2512.16709 §2–3). The low-ESS **fraction climbs with
counts: 7% → 12% → 97%**; at bright the median ESS fraction is ~0.009 (≈18 of
2000 draws effective). The flag is doing its job here. A sharp, over-confident proposal `q_NPE` that
is also slightly misplaced relative to the true (now very narrow) high-count
posterior gives importance weights dominated by a handful of samples, so
single-round amortized IS cannot rescue it, and the flag says so. Paper III avoids
this regime by running sequential NPE rounds that keep `q` close to the posterior
(ESS stays usable); ours is deliberately single-round amortized, so at high counts
a near-total low-ESS is the expected diagnostic signal. The practical conclusion at
bright counts is to trust the conformal recalibration (or fall back to NS) over the
starved IS refinement. This reproduces the Paper III IS
machinery within its scope, together with a statement of where single-round
amortization breaks it.

**Budget note.** Our IS uses a **2000-draw** per-spectrum budget,
which understates the achievable ESS relative to Paper III's **200k–400k**
likelihood evaluations on sequentially-refined proposals. Our low-ESS result is
consistent with the Paper III framework (their sequential rounds keep `q` near the
posterior and keep ESS usable; our single-round amortized `q` does not). It is not
an effect they report or that is implied-observed by them; Paper III does not run our
single-round 2000-draw configuration. The claim is that our setup trips the same
diagnostic flag Paper III defines, and not that Paper III observed this.

#### Money plot

`outputs/diagnostics/coverage_money_panel.png` (dpi 220, Okabe-Ito colorblind-safe,
distinct markers+linestyles): three panels (faint / medium / bright), coverage-vs-
nominal, raw NPE (blue ●) vs the better recalibration per level (vermilion ■,
conformal at every level, labelled in-panel with its deviation), diagonal
reference. faint/medium sit on the diagonal; bright shows the raw curve sagging
below the diagonal and conformal pulling it back.

---

### Phase-3 code/test provenance (retained)


The Phase-3 suite was **implemented, tested, and verified end-to-end** before the
production flows finished training. Nothing here reads or writes production
checkpoints beyond cold-loading a finished one.

### Deliverables

- `src/sbixcal/calibrate.py`: config-driven suite operating on ANY checkpoint
  directory (cold-loaded via `train_npe.load_posterior`):
  - **SBC** (Talts et al. 2018) via sbi 0.26.1 `sbi.diagnostics.run_sbc` +
    `check_sbc`; rank histograms via `sbi.analysis.sbc_rank_plot`. Uniformity
    statistic recorded = the KS p-values (`ks_pvals`) **and** the C2ST-of-ranks
    accuracy (`c2st_ranks`) that `check_sbc` returns. Fresh sims drawn from the
    SAME prior+simulator (`simulate.py`) the flow was trained for.
  - **Expected coverage / TARP** (Lemos et al. 2023) via
    `sbi.diagnostics.run_tarp` + `check_tarp` (ATC + KS p-value); the ECP-vs-alpha
    curve saved as npz + figure. Plus a direct per-parameter empirical
    credible-interval coverage curve for the before/after comparison.
  - **Recalibration, two methods, one interface**:
    (a) **IS-refinement** with the exact Poisson likelihood (Barret & Dupourqué
        Paper III, arXiv:2512.16709 §2–3): `w = p_Poisson(x|θ)·p(θ)/q_NPE(θ|x)`,
        `q` from the flow's `log_prob`, λ(θ) from folding θ through the same
        response (`simulate.fold_theta`). Kish **ESS** computed; **low-ESS cases
        flagged** as Paper III's own NPE-failure diagnostic.
    (b) **Conformal / quantile recalibration** of 1-D marginals (split-conformal /
        expected-coverage remap of PIT quantiles; Vovk+05, Lemos+23) learned on a
        held-out calibration set.
  - Before/after coverage comparison saved as npz + figure, per level.
- `scripts/run_calibration.py --config configs/calibration.yaml [--checkpoint <dir>]
  [--level <name>]`: full suite → `outputs/calibration/<level>/` with
  skip-if-exists; every figure regenerable from config+seed. Reads the prior,
  base model, exposure and response from the checkpoint's `arch.json`.
- `tests/test_calibrate.py`: **10 tests, all green** (~45 s).

### sbi 0.26.1 diagnostic APIs used (exact names)

`sbi.diagnostics.run_sbc`, `sbi.diagnostics.check_sbc` (returns `ks_pvals`,
`c2st_ranks`, `c2st_dap`), `sbi.diagnostics.run_tarp`, `sbi.diagnostics.check_tarp`
(returns `atc`, KS p-value), and `sbi.analysis.sbc_rank_plot`. No API gaps; all
four diagnostics ship in 0.26.1 and were used directly without reimplementation.

### Test outcomes (Gaussian / Poisson toys, seeded)

- **SBC** on a tiny NSF flow trained on the linear-Gaussian toy `x = θ + N(0,σ²)`:
  rank KS p-values ≈ 0.22 / 0.49 (> 0.02 loose bound → roughly uniform); C2ST of
  ranks ≈ 0.59 (≈ 0.5). TARP ATC ≈ 0.001, KS p = 1.0 (near-diagonal coverage).
- **IS-refinement** on a Poisson "spectrum" toy with a computable 1-D posterior
  (λ_c = a·template_c): a deliberately low-biased proposal (median a ≈ 3.4) is
  IS-corrected to within < 0.4 of the exact posterior median (a ≈ 5.5); weights
  normalize to 1; ESS computed; the refined 5/95 bracket the truth. A separate
  pathological narrow/off proposal correctly trips the low-ESS flag (ESS frac
  < 0.1). `poisson_loglik` matches `scipy.stats.poisson.logpmf` to 1e-8.
- **Conformal recalibration** on synthetic too-narrow (overconfident) Gaussian
  posteriors: raw nominal-0.9 coverage ≈ 0.64 (under-covers, max |emp−nominal|
  ≈ 0.29); after recalibration max |emp−nominal| < 0.08.

### End-to-end smoke check (throwaway flow, since deleted)

A tiny 800-sim / 60-epoch dev flow (deliberately undertrained, removed after the
check) ran the full suite cleanly: all 8 artifacts written, skip-if-exists
confirmed. Its (expected-to-be-poor) numbers (SBC KS p ≈ 0, conformal coverage
deviation 0.062 → 0.033, IS low-ESS in 28/30 cases) confirm every code path
fires and the diagnostics correctly flag a miscalibrated flow. Production
numbers will be filled in once the Phase-2 flows finish.

---

## Phase 4: misspecification-detection benchmark (2026-06-11)

A systematic misspecification-detection **benchmark for X-ray spectral
SBI**. The detector *ideas* are adapted from the
general-SBI misspec literature (Buchner+14 QQ-plots, Schmitt+23/24 embedding
tests, Lopez-Paz & Oquab C2ST) and benchmarked here on X-ray spectra; the
detectors themselves are not claimed as novel.

### Three detectors (`src/sbixcal/detect.py`)

- **D1 PPC** (per-spectrum). Draw θ~q_NPE(·|x), fold through the SAME EPIC-pn
  response, Poisson-realize K replicates; two discrepancies vs the replicate
  ensemble, (a) χ²-like on binned counts, (b) KS-like on the **cumulative** count
  spectrum (the Buchner+14 QQ-plot descendant). Both sub-scores kept; combined
  score = max (suspicious if EITHER check fails). p-value-style via the replicate
  distribution.
- **D2 embedding-OOD** (per-spectrum; Schmitt+23/24). Mahalanobis AND k-NN
  distance of the observed embedding (the flow's trained CNN summary) from the
  clean training-set embedding cloud (cached once per flow). **k-NN is the primary
  score**: the clean cloud is dominated by one huge-variance brightness axis
  (log-uniform norm spans decades of counts), so Mahalanobis whitening inverts
  that axis and *amplifies noise*, washing out the misspec signal; raw-Euclidean
  k-NN is far more sensitive (k-NN ~0.84 vs Mahalanobis ~0.55 on a strong line at
  bright). The spec's "Mahalanobis *and/or* k-NN" permits this; both are stored.
- **D3 simplified MARGINAL C2ST** (per-cell population test). A single classifier
  per benchmark cell, stratified-k-fold-CV, separating clean-population from
  misspec-population embeddings; cell statistic = CV accuracy (0.5 = undetectable),
  per-spectrum out-of-fold class-1 probabilities feed the ROC. **The per-spectrum
  *conditional* C2ST was found
  pathological** against overconfident NPE posteriors: a tight posterior-
  predictive replicate cluster is trivially separable from the broad clean cloud
  for clean AND misspecified spectra alike (AUC fell *below* 0.5), so it carries no
  misspec signal. We therefore ship the simplified marginal version and label it as
  such. (This pathology is itself a finding:
  the overconfidence isolated in Phase 2/3 actively breaks the conditional C2ST.)

### Harness (`scripts/run_detect_benchmark.py`, `scripts/analyze_detect.py`)

Grid: 4 B-families × strength grid × 3 count levels × 3 detectors. Per-(family,
strength,level) cell: N_misspec misspecified test spectra + a shared per-level
pool of N_clean clean Model-A spectra (the common ROC negative class). Per-spectrum
scores → `outputs/detect/scores.jsonl`; per-cell AUC → `results.jsonl`; both keyed
(family,strength,level,detector) and **skip-if-present on restart** (kill/resume
safe, verified). Figures/tables regenerated separately by `analyze_detect.py`.
Writes ONLY to `outputs/detect/` (gitignored), never `outputs/calibration/`.

### Pilot AUC table (N_test=30, strongest strength/family, all levels)

ROC AUC, clean vs misspecified (0.5 = chance; bold-worthy ≳0.9). Pilot ran in
**122 s** for 36 cells.

| level  | family (strength)        | D1 PPC | D2 emb-OOD | D3 marg-C2ST |
|--------|--------------------------|--------|------------|--------------|
| faint  | B1 line (norm 3e-4)      | 0.777  | 0.596      | 0.766        |
| faint  | B2 pcf (f=0.3)           | 0.571  | 0.659      | 0.704        |
| faint  | B3 brems (kT=1.5)        | 0.496  | 0.433      | 0.597        |
| faint  | B4 gain (3%)             | 0.557  | 0.437      | 0.599        |
| medium | B1 line (norm 3e-4)      | 0.992  | 0.794      | 0.927        |
| medium | B2 pcf (f=0.3)           | 0.734  | 0.863      | 0.920        |
| medium | B3 brems (kT=1.5)        | 0.581  | 0.514      | 0.759        |
| medium | B4 gain (3%)             | 0.576  | 0.459      | 0.613        |
| bright | B1 line (norm 3e-4)      | 0.999  | 0.797      | 0.877        |
| bright | B2 pcf (f=0.3)           | 0.819  | 0.888      | 0.931        |
| bright | B3 brems (kT=1.5)        | 0.450  | 0.627      | 0.711        |
| bright | B4 gain (3%)             | 0.450  | 0.372      | 0.472        |

### Early signal: which detector catches which family (including negatives)

- **B1 (unmodeled Fe-K line): caught, and D1 PPC owns it.** D1 AUC 0.78→0.99→1.00
  with counts (the line adds localized counts the posterior-predictive cannot
  reproduce). D3 close behind (0.77→0.93→0.88). At ~100 counts even a line is only
  marginally detectable (D1 0.78), since the **shot noise at faint swamps the line**.
- **B2 (partial covering): caught by the EMBEDDING detectors (D2/D3), with D1
  lagging.** D2 0.66→0.86→0.89, D3 0.70→0.92→0.93, while D1 lags (0.57→0.73→0.82). A covering
  fraction reshapes the whole soft continuum (a *global* distortion), which the
  embedding cloud separates better than the channel-wise PPC.
- **B3 (wrong continuum family, brems vs powerlaw): WEAK across the board, best by
  D3.** D3 0.60→0.76→0.71; D1/D2 hover near chance. At kT=1.5 the brems continuum
  is close enough to an absorbed powerlaw that the 5-param Model A *absorbs the
  difference* into N_H/Γ/blackbody, a silent continuum-family
  misspecification. (The full grid's lower kT may separate better; pending.)
- **B4 (detector gain shift): NOT detectable by any detector at any level.** Every
  AUC ≈ 0.37–0.61, around or below chance, and it does NOT improve with counts (a
  separate check: even a 10% gain stays at D1≈0.37). **A gain shift preserves
  spectral SHAPE (it only slides the energy axis a fraction of a percent) and
  the NPE folds it into the continuum parameters.** The negative
  result: *gain miscalibration is silently invisible to all three trust scores.*
- General trend: detectability rises with counts for B1/B2/B3 (more counts → the
  PPC/embedding resolve the distortion) but NOT for B4.

### Consequence: B1 silent-failure cost (ΔΓ bias of the undetected line)

NPE posterior-median Γ minus the clean-truth Γ that generated the line-contaminated
spectrum (pilot, line norm 3e-4, N=30):

| level  | ⟨Γ_truth⟩ | ⟨Γ̂⟩  | ⟨ΔΓ⟩   | median ΔΓ | ⟨|ΔΓ|⟩ |
|--------|-----------|-------|--------|-----------|--------|
| faint  | 1.930     | 1.986 | +0.056 | +0.079    | 0.398  |
| medium | 1.930     | 2.053 | +0.123 | +0.161    | 0.363  |
| bright | 1.930     | 2.170 | +0.240 | +0.339    | 0.479  |

The unmodeled 6.4 keV line biases the inferred photon index **softer** (Γ↑), and
the *signed* bias grows with counts (+0.06→+0.12→+0.24): at faint the line is
weak-but-detectable yet barely shifts Γ; at bright the line is loud (D1
AUC 0.999, so it IS flagged here) and drags Γ by +0.24. The dangerous regime is
the middle: at medium counts the line is only ~0.99 detectable by D1 but already
biases Γ by +0.12. (B4's ΔΓ is the clearer undetected-and-biased case and is
recorded for the full grid; B2/B3 have no powerlaw-Γ truth by construction, since
B3 swaps the powerlaw out and B2 keeps it.)

### Tests (`tests/test_detect.py`): 12 passed; full suite **42 passed**

Tiny throwaway flow (dev Model A, 1000 sims, ≤80 epochs, ~10 s CPU). Each of
D1/D2/D3 separates an OBVIOUS strong Fe-K line (norm 1e-2, several-folds the
counts) from clean with **AUC > 0.9** (measured D1 1.00 / D2 0.95 / D3 0.96);
scores deterministic given seed; D1 both sub-scores present; D2 kNN-primary +
Mahalanobis + reference-cache roundtrip; D3 CV-accuracy + per-spectrum probs;
roc_auc identities. B4 is excluded from the AUC>0.9 gate (a documented genuine
non-detection) and tested only for finite/deterministic operation.

### Full-grid run decision

Pilot extrapolation (per-cell wall times, D1-at-bright dominates at ~16 s/cell):
full grid = 144 cells (4 strengths × pilot's 36) at N_misspec=100 / N_clean=200
≈ **~25–35 min**, under the ~45 min self-run threshold → **full run launched
now**. Resume command (idempotent; skips finished cells):

    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe scripts\run_detect_benchmark.py --config configs\detect.yaml
    .venv\Scripts\python.exe scripts\analyze_detect.py --config configs\detect.yaml

Full-grid AUC table + ΔΓ-bias across the whole strength grid land in
`outputs/detect/auc_table.md` / `consequence.md` once the run completes; the
pilot already fixes the qualitative story above (B1→D1, B2→D2/D3, B3 weak,
B4 invisible).

---

## Phase 4: FULL-GRID detection benchmark (2026-06-11): CLOSED

The full 144-cell ROC grid (4 families × 4-point strength grids × 3 count levels
× 3 detectors) **completed**: `outputs/detect/{results,scores}.jsonl` (144 cells,
43,200 per-spectrum scores), `auc_table.md`, `auc_heatmap.png`, the consequence
table `consequence.md`/`consequence.jsonl`, and the README figures
`outputs/diagnostics/detector_auc_grid.png` (3 detectors × 4 families heatmap per
level) + `outputs/diagnostics/dgamma_silent_failure.png`. Every number below is
read straight from those artifacts. N_misspec=100,
N_clean=200 per cell; AUC = ROC of the clean pool (negative) vs the misspecified
test set (positive); 0.5 = chance.

### Best AUC per family per level (across the strength grid + best detector)

**Apples-to-apples caveat.** The three detectors are NOT on the same
axis. **D1/D2 are per-spectrum *unlabeled* novelty scores** (one spectrum → a
suspicion score). **D3/marginal-C2ST is a *population separability* statistic**: it
trains a *supervised* classifier on each cell's *labeled* clean-vs-misspec
embedding populations, and its "per-spectrum score" is that supervised
classifier's out-of-fold class-1 probability. It measures how distinguishable the
two populations are, and not whether one unlabeled spectrum is trustworthy. So D3 is
shown in a **separate column group below**, is never counted as a per-spectrum
"win", and carries a **non-0.5 control-cell floor** (next sub-point). The
table, for each (level, family) the strongest-detectable cell, split by detector
class:

**Per-spectrum detectors (D1/D2), the deployable trust scores:**

| level  | B1 Fe-K line        | B2 partial-cover     | B3 brems continuum  | B4 gain shift        |
|--------|---------------------|----------------------|---------------------|----------------------|
| faint  | **0.757** (D1, 3e-4)| 0.673 (D2, f=0.3)    | 0.565 (D1) ≈chance  | 0.581 (D1, 1%) ≈chance |
| medium | **0.972** (D1, 3e-4)| 0.830 (D2, f=0.3)    | 0.536 (D1) ≈chance  | 0.513 (D2, 2%) ≈chance |
| bright | **0.970** (D1, 3e-4)| 0.843 (D2, f=0.3)    | 0.662 (D2, kT=1.5)  | 0.515 (D2) ≈chance   |

**Population separability (D3 marginal-C2ST, a supervised two-sample statistic and
not a per-spectrum trust score; read against its ~0.66 cv-accuracy / ~0.5-AUC
control floor):**

| level  | B1 (D3) | B2 (D3) | B3 (D3) | B4 (D3) |
|--------|---------|---------|---------|---------|
| faint  | 0.800   | 0.807   | 0.709   | 0.533 (≈floor) |
| medium | 0.917   | 0.935   | 0.771   | 0.537 (≈floor) |
| bright | 0.893   | 0.957   | 0.808   | 0.532 (≈floor) |

**D3 control-cell floor, recomputed from `outputs/detect/results.jsonl`:**
on cells with no real misspecification (the weakest B1 line, norm 5e-6, a
negligible perturbation) D3 reports **cv-accuracy ≈ 0.66** (0.663/0.667/0.663 at
faint/medium/bright) while its **ROC AUC sits at 0.437/0.326/0.541 (at/below
chance)**. The same ~0.66 cv-accuracy floor holds across *all* near-control cells
(weakest grid point of every family: 0.63–0.76 cv-accuracy, AUC near chance). That
gap (high cv-accuracy, chance AUC) shows that D3's cv-accuracy is a
population-separability null offset and not evidence any individual spectrum is
flagged. Every D3 "win" above must be read against that ~0.66 / ~0.5 null.

**Summary across the two detector classes:** *given labeled populations, the embedding space
separates B2/B3 (D3); given a single unlabeled spectrum, only the PPC flags lines
(D1) and nothing flags gain shifts.*

### Which detector owns which family (verified, including the negatives)

- **B1 (unmodeled Fe-K line): the per-spectrum PPC (D1) owns it at medium/bright.**
  D1 AUC on the strongest line (norm 3e-4) climbs 0.757 → **0.972 → 0.970** with
  counts; at the mid line (8e-5) D1 already reaches 0.773/0.893 at medium/bright.
  At **faint** the best per-spectrum score is D1 (0.757); D3's 0.800 is the
  *population separability* number, read against its ~0.66 cv-accuracy floor, and is not a
  per-spectrum win. The line adds *localized* counts in the 5.8–7.0 keV window the
  posterior-predictive cannot reproduce, which is what the channel-wise PPC
  (D1) is built to see. **D1 owns B1 among the deployable per-spectrum detectors.**
- **B2 (partial covering): among per-spectrum detectors the embedding-OOD (D2)
  gets the lift, and the population test (D3) separates it best of all.** D2 climbs
  0.673 → 0.830 → 0.843 with counts while D1 lags (0.592 → 0.639 → 0.778); the D3
  *population* statistic reaches 0.807 → 0.935 → 0.957, which is supervised
  two-sample separability and not a per-spectrum trust score. A covering fraction
  reshapes the *whole* soft continuum (a global distortion), which the embedding
  sees better than the channel-wise PPC. **D2 owns B2 among the per-spectrum
  detectors**, and D2 is the embedding-OOD prior-art detector, so this is
  the point at which it earns its place.
- **B3 (wrong continuum family, brems vs powerlaw): only the D3 *population* test
  gets meaningful lift, and per-spectrum detectors stay near chance.** D3 reaches
  0.71/0.77/0.81 at its best kT (population separability); the per-spectrum D1/D2
  hover near chance (best 0.54–0.66, D1 even dips *below* 0.5 at low kT/bright,
  0.438). At the kT where brems looks most powerlaw-like the 5-parameter Model A
  simply **absorbs the continuum difference into N_H/Γ/blackbody**, a
  silent continuum-family misspecification to any per-spectrum trust score. (Note
  the B3 README caveat: the brems template is Gaunt-factor-free, an analytic
  approximation, see Limitations.)
- **B4 (detector gain shift): NOT detectable by ANY detector at ANY level or
  strength. The gain-shift negative result.** Across all 36 B4 cells (3 detectors ×
  3 levels × 4 strengths) the AUC range is **0.430–0.581, mean 0.497**, dead at
  chance, and it does **not** improve with counts or with stronger gain (a 3%
  shift is no more detectable than 0.5%). Even the D3 *population* test is at its
  ~0.66 cv-accuracy control floor here (AUC 0.53). A gain shift preserves spectral
  *shape* (it only slides the energy axis by a fraction of a percent) and the NPE
  folds that slide into the continuum parameters. **This directly CONTRADICTS the
  original spec's expectation that the embedding-OOD detector (D2) would catch
  gain shifts: it does not (D2 on B4 = 0.48–0.52 at every level). Gain
  miscalibration is silently invisible to all three trust scores.** Gain-shift
  detection remains an open problem here.
- **Among the per-spectrum detectors, D2 earns its keep on B2.** D1 owns B1, D2
  owns B2 (D1 lags there), and both are at/below chance on B3 and B4. D2 is the
  literature's embedding-OOD detector benchmarked here on X-ray
  spectra; the verdict is that the channel-wise PPC (D1, on B1) and the embedding
  OOD (D2, on B2) split the per-spectrum-detectable families between them. (D3 is
  not on this per-spectrum axis; it answers the population-separability question.)

### D2 vs Schmitt: scope of the embedding negative result

Our D2 is the **posterior-trained-embedding** variant: it scores OOD distance in
the flow's *un-regularized, NEAR-SUFFICIENT* CNN summary (the embedding the NPE
learned for inference). It is distinct from Schmitt+23/24's **MMD-regularized,
deliberately OVERCOMPLETE** summary network, which is trained specifically to make
misspecification detectable. This distinction is the reason the gain-shift
negative result has the scope it does: Schmitt's **Eq. 12–13** prove that a
misspecification which *preserves the summary distribution* is **provably
invisible** to any test in that summary space. A sub-percent B4 gain shift is
such a case (the NPE folds it into the continuum parameters, leaving the
near-sufficient summary distribution essentially unchanged), so it evades a
near-sufficient embedding *by construction*. **The scope of the B4 negative result
is therefore "undetectable by these three detectors,"
and the natural next attempt is an MMD-regularized overcomplete summary space
(Schmitt+23/24) trained to surface such shifts**, an approach not refuted
here and not yet tried. The README "Gain-shift family is undetectable" sentence
inherits this scoping.

### On the B2 trend ("inversion"): checked, and it is not anomalous

We examined a possible B2 AUC "inversion (AUC decreases with covering
fraction)". **Verified directly from the grid: there is no pathological
inversion.** When B2 is ordered by *physical misspecification strength* (covering
fraction f decreasing from 0.9 → 0.3 = an increasingly large unabsorbed leak),
AUC rises **monotonically** at every level and for every detector (e.g. D3 medium:
f=0.9→0.7→0.5→0.3 gives 0.626 → 0.808 → 0.880 → 0.935). The one sense in which
"AUC decreases with covering fraction" holds is the trivial, correct one:
**as f → 1 the misspecification vanishes** (full covering = ordinary tbabs
absorption), so AUC falls to chance. Physically, a small (f≈0.9) partial-covering
leak is **near-degenerate with a small downward adjustment of N_H**, so the NPE
absorbs the weak leak into the absorption column, leaving almost no residual for
any detector to flag (D3 0.61–0.63 at f=0.9 across levels). The f≈0.9
near-chance behaviour is the expected "absorbed-into-N_H" degeneracy. (B1 shows
the same benign pattern: weak lines, norm ≤ 2e-5, sit
at chance; AUC rises monotonically with line norm.)

### Consequence: ΔΓ bias of the silently-misspecified line (B1)

Posterior-median Γ minus the clean-truth Γ that generated the line-contaminated
spectrum, at the **strongest** line (norm 3e-4), full grid (N=100):

| level  | ⟨Γ_truth⟩ | ⟨Γ̂⟩  | ⟨ΔΓ⟩   | median ΔΓ | ⟨|ΔΓ|⟩ |
|--------|-----------|-------|--------|-----------|--------|
| faint  | 2.050     | 1.978 | −0.071 | −0.082    | 0.372  |
| medium | 2.007     | 2.102 | +0.095 | +0.098    | 0.307  |
| bright | 2.007     | 2.208 | **+0.201** | **+0.257** | **0.460** |

The unmodeled 6.4 keV line drags the inferred photon index, and the *signed* bias
grows with counts: at faint the line is shot-noise-buried (the small signed bias
is −0.07, scatter-dominated, ⟨|ΔΓ|⟩ 0.37); at bright the line is loud
and pulls Γ **softer by +0.20 on average (median +0.26, mean |ΔΓ| 0.46)**. The
danger is the *combination* with the detection grid: at bright the strongest line
is flagged (D1 AUC 0.97), but at medium the same line biases Γ by +0.10 while only
just being detectable, and B4 gain shifts bias the continuum invisibly at every
level. **Pairing the two grids: the misspecifications that are
hardest to detect (B4 gain, weak B1/B2) are the ones that bias a
downstream Γ/N_H measurement without tripping any trust score.**

### Detector implementation note (labelling)

`detect.py`'s D3 is the **simplified MARGINAL C2ST** (a single
stratified-k-fold-CV classifier per benchmark cell, separating the clean
embedding population from the misspecified embedding population; cell statistic =
CV accuracy, per-spectrum out-of-fold class-1 probability feeds the ROC). It is
**not** the per-spectrum conditional C2ST; that variant was found pathological
against the over-confident NPE posteriors (a tight posterior-predictive replicate
cluster is trivially separable from the broad clean cloud for clean *and*
misspecified spectra alike, so its AUC carried no misspec signal). The README
labels the marginal version as such.

---

## Phase 5: nested-sampling benchmark (2026-06-11): HARNESS BUILT + PILOT

> **[Status update 2026-06-12: the full run is now COMPLETE.** The pilot text
> below (harness, the `max_ncalls` finding, the extrapolation, the "not run here"
> note) is retained for provenance; the finished 76-spectrum results, the
> speed-vs-trust table, the NS-evidence-vs-detector cross-check, and the
> capped-row / rejection-timeout accounting are in the **"Phase 5: nested-sampling
> benchmark (2026-06-12): COMPLETE"** section at the end of this file.]**

The speed-vs-trust benchmark: UltraNest (`ReactiveNestedSampler`, in-memory
`log_dir=None`, ultranest 4.5.0) on the **EXACT SAME** Poisson likelihood the
Phase-3 IS-refinement uses, vs the amortized NPE, on a config-driven subsample.

### Deliverables

- `src/sbixcal/ns_bench.py`: the harness. The NS log-likelihood is literally
  `calibrate.poisson_loglik(counts, simulate.fold_theta(theta))` (reuse, **not**
  re-implementation; a unit test asserts NS and the IS-likelihood agree on logL
  to 1e-9), and the prior is the SAME box (`priors.prior_bounds`) the flow's
  BoxUniform uses, wrapped as a unit-cube→box transform. `vectorized=True`,
  `log_dir=None` (so the UltraNest/h5py Windows file-locking quirk
  cannot bite). Per spectrum it records NS quantiles / logZ / n_like_evals /
  wall-clock and NPE quantiles / sampling wall-clock.
- `configs/ns_bench.yaml`: ~95-spectrum subsample: clean Model-A × {faint,
  medium, bright} (25 each) + B1 Fe-K line (strong, medium+bright) + B4 gain
  shift (medium+bright). A per-spectrum `max_ncalls` cap (see finding below) is
  set in the config.
- `scripts/run_ns_benchmark.py`: append-resumable JSONL keyed by `spectrum_id`
  (`outputs/ns_bench/results.jsonl`, resume-skip verified). B4 spectra are
  *generated* with a gain-shifted response but BOTH inferences (NS and NPE) use
  the NOMINAL response + well-specified Model A, and that mismatch *is* the
  misspecification.
- `scripts/analyze_ns_bench.py`: the speed-vs-agreement table (NS s/spec vs NPE
  ms/spec, quantile agreement) + the NS-evidence-flag-vs-detector-AUC cross-check
  (reads `outputs/detect/results.jsonl` read-only; cells the detector grid hasn't
  produced show "pending", and the whole cross-check is labelled a STUB until the
  detector grid finishes). → `outputs/ns_bench/analysis.md`.
- `tests/test_ns_bench.py`: **5 tests, all green**: NS recovers truth inside the
  95% interval on a bright spectrum (seeded, 80 live points); NS == IS-likelihood
  logL at sampled θ (exact-reuse check); box transform spans the prior; JSONL
  resume-skip; quantile-agreement zero for identical inputs. **Full suite: 47
  passed** (42 prior + 5 new).

### Pilot: measured per-spectrum NS wall-clock

One fully-converged (uncapped) clean faint spectrum + a fast capped probe at all
three levels (`scripts/ns_pilot_timing.py`, `outputs/ns_bench/pilot_timing.json`):

| level | ~counts | NS evals (to converge) | NS wall | NPE wall | NS/NPE speedup | q-agreement (mean \|Δq\|/prior-width) |
|-------|---------|------------------------|---------|----------|----------------|----------------------------------------|
| faint  | ~98   | 25,496 (uncapped)      | 232.9 s | 46 ms | ~5000× | 0.033 |
| faint  | ~98   | 8,012 (capped probe)   | 42.1 s  | 39 ms | n/a | 0.077 |
| medium | ~986  | 8,007 (capped probe)   | 57.3 s  | 32 ms | n/a | 0.046 |
| bright | ~9982 | 8,014 (capped probe)   | 23.1 s  | 33 ms | n/a | 0.097 |

Per-eval wall-clock: faint 5.3 / medium 7.2 / bright 2.9 ms/eval. NS quantiles
**agree with the NPE quantiles to ~0.03–0.10 of the prior width**, i.e. on a
*well-calibrated* faint NPE posterior NS validates the amortized flow to ~3% of
the prior range (the speed-vs-trust point: the flow is ~5000× faster and, where
it is calibrated, posterior-equivalent to NS). The capped probe's NS 90% interval
contained truth on all 5 params at medium (1.00) and 4/5 at faint (0.80); at
bright the 8k cap truncated *before* convergence (truth-in-90 = 0.00), which is
exactly the next finding.

### Pilot FINDING: NS cost has large variance; a `max_ncalls` cap is required

The per-spectrum NS cost is **highly variable**: the uncapped clean faint
spectrum converged at ~25.5k evals (233 s), but a *second* faint draw (high
log-uniform norm → unexpectedly high counts → very tight Poisson posterior)
stalled for >20 min without converging when run uncapped. **This is the central
operational finding for Phase 5:** because the log-uniform `norm` prior makes some
"faint"-exposure draws high-count, the Poisson posterior can be extremely tight
and slow for UltraNest to localize. The harness therefore **requires a
`max_ncalls` cap** (set to 400,000 in the config); `ns.n_like_evals == cap` flags
a non-converged spectrum that the analysis records explicitly instead of waiting
indefinitely. (At high counts the IS-refinement low-ESS flag, Phase 3, and the NS
slow-convergence are the *same* underlying fact: a very sharp posterior the
single-round amortized flow approximates imperfectly.)

### Extrapolation to the full ~95-spectrum run

Using the measured per-eval rates and the faint convergence budget (~25.5k evals)
as the representative per-spectrum eval count (bright needs *more* evals to
converge, so its block is a lower bound; see the truncation above):

- faint block (25 spectra): ~0.9 h
- medium block (35 spectra): ~1.8 h
- bright block (35 spectra): ~0.7 h (lower bound; budget more)
- **total: ~3.4 h single-process, realistically ~3–5 h** (NPE side ~4 s, negligible).

### Exact full-run command

The runner is append-resumable (kill/resume safe), writes ONLY to
`outputs/ns_bench/`, and reads checkpoints in `outputs/models/`. Run:

    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe scripts\run_ns_benchmark.py --config configs\ns_bench.yaml
    .venv\Scripts\python.exe scripts\analyze_ns_bench.py --config configs\ns_bench.yaml

(The single faint `results.jsonl` row from the pilot is valid and will be
resume-skipped.) The analysis prints the speed-vs-agreement table and the
NS-flag-vs-detector cross-check; the detector cross-check fills in automatically
once `outputs/detect/results.jsonl` holds the matching cells. **Not run here**
(build and pilot only here; the full run is launched separately).

### Reading the NS table: caps, logZ, and the `--max-ncalls` override

A note to bolt onto the NS speed-vs-trust table when it lands, so logZ is never
misread:

- **Capped rows are NON-CONVERGED.** A row with `ns.n_like_evals == max_ncalls`
  hit the per-spectrum evaluation cap *before* UltraNest's stopping criterion, so
  the run is **not converged**. The harness records the cap value, so the flag is
  `n_like_evals == cap`.
- **For a capped row, `logZ` is a LOWER BOUND**, not the converged evidence
  (UltraNest's `logZ` accumulates as the live set shrinks; truncating early stops
  that accumulation short). The posterior quantiles from a capped row can still be
  *indicative*, but the evidence is not final.
- **Never compare `logZ` across a capped and an uncapped row** (nor across two
  rows capped at different evaluation budgets); it is comparing a lower bound to a
  converged value (or two differently-truncated lower bounds). Evidence
  comparisons are only valid between *converged* (uncapped) rows.
- **The live run uses `--max-ncalls 120000`** (a CLI override of the config's
  `max_ncalls: 400000`, via `scripts/run_ns_benchmark.py --max-ncalls 120000`).
  120k bounds each spectrum's wall-clock for a practical ~95-spectrum run on light
  compute while still letting the great majority of faint/medium spectra converge
  (the pilot converged faint at ~25.5k evals); the cap exists for the heavy-tailed
  high-count tail cases. The analysis must read the cap from the per-spectrum
  `max_ncalls` field, NOT assume the config's 400000.

---

## Phase 5: nested-sampling benchmark (2026-06-12): COMPLETE

The full run finished: **76 spectra** (56 clean Model-A across the three count
levels + B1 Fe-K line and B4 gain spectra at medium/bright), append-resumed into
`outputs/ns_bench/results.jsonl`; analysis in `outputs/ns_bench/analysis.md` +
`analysis_summary.json`. Every number below is read straight from those artifacts.
The raw `outputs/ns_bench/*` outputs (`results.jsonl`, `analysis.md`,
`analysis_summary.json`) are gitignored and not committed; they regenerate from
`scripts/run_ns_benchmark.py` followed by `scripts/analyze_ns_bench.py`.

**Budget trim.** The clean subsample was trimmed from 25/level to
**medium 16, bright 15** (faint kept at its completed 25) under a wall-clock
budget; the config comment (`configs/ns_bench.yaml`, the `subsample:` block) states
it: *"medians/quantile-agreement stabilize by n~15 per level; the trim is a
compute-budget choice, recorded in RESULTS."* The B1/B4
misspec cells are n=6/n=4. Total 76 = 25+16+15 clean + 6+6 B1 + 4+4 B4.

### 1. Speed vs agreement (clean Model-A spine)

NS is **~9 000–13 000× slower per spectrum** than the amortized NPE, agreeing with
the flow's quantiles to **0.04–0.10 of the prior width** where the NPE is
calibrated. This is the measured speed-vs-trust trade.

| level | ~counts | n | NS s/spec | NS n_like_evals | NPE ms/spec | NS/NPE speedup | q-agreement (mean \|Δq\|/width) |
|---|---|---|---|---|---|---|---|
| faint  | 47   | 25 | 1065.5 | 49 878 | 83  | 12 802× | 0.068 |
| medium | 540  | 16 |  941.2 | 52 463 | 106 |  8 864× | 0.037 |
| bright | 7910 | 15 | 1751.4 | 95 386 | 177 |  9 882× | 0.100 |

NS 90%-interval truth recovery (clean, mean over params): faint 0.87 / medium 0.91
/ bright 0.85, a NS-side coverage sanity check, all near nominal.

### 2. Capped (non-converged) rows + the NPE rejection-timeout flag

- **Capped rows: 12 of 76** carry `ns.n_like_evals ≥ 120 000` (the live
  `--max-ncalls 120000` cap), so they hit the evaluation cap before UltraNest's
  stopping criterion and are **non-converged**. Breakdown: **11 clean** (2 faint,
  2 medium, **7 bright**) + **1 B4-bright**. This is the heavy-tail
  effect: the log-uniform `norm` prior makes some draws high-count, giving very
  tight Poisson posteriors slow for NS to localize. **A capped row's `logZ` is a
  LOWER BOUND** and is never compared against a converged `logZ`; the evidence
  flag in §3 uses level-matched clean baselines and the qualitative ΔlogZ sign is
  robust to this, but the bright clean baseline in particular carries capped rows,
  so its absolute `logZ` is a bound. The speed/agreement
  aggregates in §1 are unaffected (they are wall-clock + quantiles, not evidence).
- **NPE rejection-timeout flag: 0 rows tripped it in this run.** Sampling the flow
  with `reject_outside_prior=True` can stall when a *misspecified* spectrum pushes
  flow mass outside the prior box (worst observed on B4-bright, ~1% acceptance);
  the harness (`ns_bench.py:run_npe_one`) bounds this at 120 s and on timeout falls
  back to raw flow samples, **flagging the row** (`rejection_timeout=True`). The
  flag is a trust signal, since flow mass leaking outside the prior on a misspecified
  spectrum is itself evidence the model is wrong. **In the committed run no spectrum
  exceeded the 120 s bound, so the count is 0**;
  the guard exists for the heavy mass-leak cases and is exercised by the unit test,
  not by this particular subsample.

### 3. NS evidence flags what the per-spectrum trust scores cannot

Evidence flag = `mean(logZ_misspec − logZ_clean)` at the same level (more negative
⇒ the well-specified Model A fits the misspecified spectrum worse ⇒ flagged), read
against the Phase-4 detector AUC for the matching (family, strength, level) cell:

| family | strength | level | n | NS ΔlogZ (mis − clean) | D1 AUC | D2 AUC | D3 AUC | reading |
|---|---|---|---|---|---|---|---|---|
| B1 Fe-K line | 3e-4 | bright | 6 | **−886.5** | 0.970 | 0.810 | 0.893 | NS **and** detectors both flag |
| B1 Fe-K line | 3e-4 | medium | 6 | **−81.8**  | 0.972 | 0.846 | 0.917 | both flag |
| B4 gain shift | 3% | medium | 4 | **−7.8**  | 0.482 | 0.477 | 0.456 | **NS flags; all 3 detectors at chance** |
| B4 gain shift | 3% | bright | 4 | **−1.9**  | 0.474 | 0.484 | 0.446 | **NS flags weakly; detectors at chance** |

The B4 contrast is the key result. A 3% gain shift is invisible to all three
per-spectrum trust scores (AUC 0.45–0.48, at/below chance, since it preserves spectral
shape and the NPE folds it into the continuum parameters), yet nested sampling's
evidence still penalizes it (ΔlogZ −7.8 medium, −1.9 weakly bright). That is the
gap between a posterior-only novelty score (asks "is this draw far from the clean
cloud?") and a goodness-of-fit / evidence test (asks "does any setting of this
model explain the data?"); the latter is the methodology of **Buchner+14** (BXA
evidence comparison, the QQ/cumulative model-discovery lineage) and **Buchner &
Boorman 2023** (the canonical X-ray model-checking best-practice chapter). The
operational conclusion is that an evidence-based check (NS / BXA) catches an energy-scale
calibration error that the cheap amortized-SBI trust scores have a structural blind
spot for, which is the case for keeping evidence in the loop alongside a fast NPE
workhorse. At the loud end the two agree: on the strong B1 line both NS evidence
and D1 fire, ΔlogZ −886 / AUC 0.97. The two methods coincide where the
misspecification is obvious and split on the silent failure mode.

The B4 evidence signal is modest in absolute logZ (−1.9 to −7.8) versus B1's
−82/−886, consistent with a gain shift being a subtle misspecification. Its
sign is the right one and it is non-zero where every per-spectrum
detector is exactly at chance. NS at ~10⁴× the cost is not a deployable
per-spectrum screen; the specific claim is that the evidence carries information about
the gain shift that the posterior-only scores do not.
