# sigma_eff gain-information screen: re-derivation and committed script

2026-09-02. Every number below marked **[mine]** is my own computation in this
session, from `scripts/screen_sigma_eff.py` and the probe scripts described in
section 7. Numbers marked **[recorded]** are quoted from existing project files
with file:line. Nothing here has been through the rule-#2 multi-agent protocol
yet; it is a single-operator re-derivation plus internal cross-checks.

Headline: **the paper's triple reproduces.** The definition is recoverable, the
screen now exists as a committed script, and it runs over the whole 500-draw
population in 13 seconds.

---

## 1. What the paper says, and what was missing

`arxiv_successor/main.tex` L246 (via `deliberation/wf_audit_referee_2026-08-29.md:45`):

> "The finite-shift Fisher error is $\sigma_{\rm eff} = \Delta g/\sqrt{R}$, with
> $R$ the residual power a gain shift $\Delta g$ leaves after the other five
> parameters absorb what they can. For the three it reads 0.037, 0.050, and
> 0.039 against a prior standard deviation of 0.0289."

The audit had already established that nothing committed produced those numbers,
`deliberation/merge_stage5_audit_fixes_log_2026-08-29.md:253-262`:

> "### 4.3 R8 sigma_eff - NOT COMMITTED, so ordinal-only
> `outputs/mechanism/` (which holds `run_residual_power.py`, defining R) is
> git-UNTRACKED. The per-spectrum values 0.037/0.050/0.039 were produced by a
> 2026-08-12 adversary sub-agent in a temp scratchpad outside both repos; that
> directory survives but every script in it is gone"

That is now fixed. `scripts/screen_sigma_eff.py` is the committed replacement.

---

## 2. Task 1: the definition, the mapping, the recorded intermediates

### 2a. Mapping of the triple to spectra [recorded]

Unambiguous, from two places.

`deliberation/E7_smallset_report_final.md:81-82`:

> "(all bright: sigma_eff 0.037 / 0.050 / 0.039 for i416/i394/i8 vs prior std 0.0289)"

and the fuller table the paper's sentence was drawn from,
`deliberation/e7_verify_reports_2026-08-12/adv_external.md:33-40`, whose
second-to-last column is headed "sigma(g) physics gives":

| sid | NS shrink | sigma(g) needed | **sigma(g) physics gives = sigma_eff** | ratio |
|---|---|---|---|---|
| medium_i22 | 0.723 | 0.0228 | **0.4568** | 20.0x |
| bright_i416 | 0.731 | 0.0232 | **0.0366** | 1.6x |
| bright_i394 | 0.790 | 0.0269 | **0.0500** | 1.9x |
| bright_i8 | 0.856 | 0.0331 | **0.0390** | 1.2x |
| medium_i482 | 0.978 | 0.0863 | **0.1228** | 1.4x |
| medium_i87 | 0.978 | 0.0873 | **0.2026** | 2.3x |

Plus `adv_external.md:23`, giving a seventh value:

> "i238 has 10496 counts of which only **645 lie below 1.2 keV** (6.1% soft;
> BB-dominated, kT=2.20) -> sigma_eff(g)=0.197"

So: **i416 = 0.037, i394 = 0.050, i8 = 0.039.** The paper prints them in that
order and the order is correct. Note it is not sorted: i416 (the most
informative) is first, then i394 (the least of the three), then i8.

### 2b. The definition [recorded]

The decision entry the adversary left, `adv_external.md:113`:

> "1. **Chose**: measure gain information with the finite-shift E6 residual power
> `R(dg)` and `sigma_eff = dg/sqrt(R)`, not the finite-difference `dmu/dg`
> Fisher. **Alternatives**: derivative Fisher only; Richardson extrapolation;
> float64 folds. **Kind**: tradeoff. **Cheap to test**: yes, both computed - the
> derivative varies 1.2-2.1x over steps while the finite version varies 10-25%
> over shifts and reproduces E6 exactly; rankings identical either way."

Same choice in `deliberation/decision_digest_2026-08-28.md:26`:

> "Chose: gain information measured by the finite-shift E6 residual power
> (sigma_eff = dg/sqrt(R))"

`R` is defined in `outputs/mechanism/run_residual_power.py:1-2` (untracked, still
on disk):

> "E6b: residual Fisher power R = (1-I) * ||b||^2"

with `I` from `outputs/mechanism/run_mechanism_index.py`: whiten by the Poisson
Fisher metric `W = diag(1/sqrt(mu))`, set `A = W J` over the **five physical
parameters** and `b = W (mu_misspec - mu)`, and take
`I = ||P_A b||^2 / ||b||^2` with `P_A` from a reduced QR. That five-parameter
projection is what the paper's "after the other five parameters absorb what they
can" refers to. `Delta g` is 0.03, from
`run_mechanism_index.py: FAMILY_STRENGTH["B4"]["gain_pct"] = 3.0`; the paper
never prints it, which the audit noted.

### 2c. Other recorded intermediates [recorded]

- Population seed **20300611**, not 20260651 (`adv_external.md:8`), and the
  ten-spectrum counts gate `3853/384/429/291/599/5204/25067/10496/6829/54509`.
- `R` at i22's own theta = 0.00366, `I` = 0.9993 (`adv_external.md:29`). This
  one is inconsistent with its own sigma_eff: `0.03/sqrt(0.00366) = 0.496`, not
  the 0.4568 in the table. The two are quoted as separate statistics in the
  correlation list at `adv_external.md:47-49` ("Spearman(NS shrink, sigma_eff)"
  and "Spearman(NS shrink, E6 R at own theta)" have different values, +0.588 vs
  +0.523), so the R = 0.00366 is the one-sided E6-family R and the sigma_eff is
  not built from it directly. Section 3 shows the symmetric shift is what matches.
- Cross-validation of the adversary's machinery against E6, `adv_external.md:11`:
  bright median I 0.9759243, median R 0.079270, median ||b||^2 5.2999.

**[mine]** No formula, script, or seed for sigma_eff beyond the above survives
anywhere in either repo. I grepped both repos for `sigma_eff`, the sigma-eff
unicode form, `sqrt(R)` and the literal values; every hit is a restatement of the
result, never a construction.

---

## 3. Task 2: re-derivation from first principles

### 3a. Setup [mine]

I regenerated the populations myself from
`misspec.simulate_misspec_population(tbabs_powerlaw_bb, PHYS_PRIORS, obsconf,
"B4", 0.0, 500, 20300611)`, the same call `run_ns_smallset.make_spectrum` uses.
The counts gate passes exactly:

- bright: i8 **25067**, i238 **10496**, i394 **6829**, i416 **54509**
- medium: i22 **3853**, i87 **384**, i91 **429**, i95 **291**, i197 **599**, i482 **5204**

All ten match the recorded gate to the unit. The `ns_smallset` JSONs do not store
the counts array, so regeneration is the only route, and it is exact.

I then computed, at each spectrum's true theta and its level's exposure, the
Poisson Fisher information on `g` with the five physical parameters projected
out, using my own implementation (QR projector, `mu0` whitening, central
differences with box-clipped steps).

### 3b. Which definition reproduces [mine]

The four bright spectra, in the order i8 / i238 / i394 / i416, targets
0.0390 / 0.1970 / 0.0500 / 0.0366:

| variant | i8 | i238 | i394 | i416 |
|---|---|---|---|---|
| one-sided shift `g = 1.03`, `0.03/sqrt(R)` | 0.0458 | 0.2063 | 0.0564 | 0.0411 |
| one-sided shift `g = 0.97` | 0.0323 | 0.1870 | 0.0436 | 0.0321 |
| **symmetric shift, h = 0.03, marginal** | **0.0390** | **0.2034** | **0.0507** | **0.0374** |
| symmetric shift, h = 0.01, marginal | 0.0350 | 0.1734 | 0.0461 | 0.0331 |
| symmetric shift, h = 0.003, marginal | 0.0282 | 0.1447 | 0.0405 | 0.0278 |
| conditional (other five fixed), h = 0.03 | 0.0070 | 0.0087 | 0.0145 | 0.0069 |
| combined with the g prior in quadrature | 0.0232 | 0.0285 | 0.0250 | 0.0229 |

The winner is the **symmetric finite shift at h = Delta_g = 0.03, marginal over
the five physics parameters, whitened by `mu0`**:

```
b         = W [mu(g=1.03) - mu(g=0.97)] / (2 x 0.03),   W = diag(1/sqrt(mu0))
A         = W dmu/dtheta   (102 x 5)
I         = ||P_A b||^2 / ||b||^2
sigma_eff = 1 / sqrt((1 - I) ||b||^2)
```

This is algebraically identical to the paper's `Delta g / sqrt(R)`: with
`b_paper = W [mu(1+dg) - mu(1-dg)]/2 = dg * b`, `R_paper = dg^2 R`, so
`dg/sqrt(R_paper) = 1/sqrt(R)`. It is also identical to `sqrt([F^-1]_gg)` of the
6x6 Fisher matrix. **[mine]** I verified that identity numerically: the QR
projection and an explicitly inverted 6x6 Fisher agree to a relative 1e-14 on
seven spectra. The conditional variant is `1/sqrt(F_gg)`, five to thirty times
smaller, and does not reproduce anything.

The **conditional** and **prior-combined** rows are both far from the recorded
values, so neither is what the original used. Only the marginal one matches.

### 3c. Full reproduction, all seven recorded values [mine]

| spectrum | level | recorded | mine | difference |
|---|---|---|---|---|
| i416 | bright | 0.0366 | 0.0374 | +2.4% |
| i394 | bright | 0.0500 | 0.0506 | +1.3% |
| i8 | bright | 0.0390 | 0.0385 | -1.3% |
| i238 | bright | 0.1970 | 0.2032 | +3.2% |
| i22 | medium | 0.4568 | 0.4673 | +2.3% |
| i87 | medium | 0.2026 | 0.2059 | +1.6% |
| i482 | medium | 0.1228 | 0.1262 | +2.8% |

Worst case 3.2 per cent, well inside the 10 per cent the task set. The
reproduction is real: seven independent values, six of the seven high by 1.3 to
3.2 per cent, which is a consistent small offset rather than a scatter.

### 3d. Why h = 0.03 and not something else [mine]

I scanned the shift, holding everything else fixed. Error against the recorded
value, per cent:

| h | i22 | i87 | i482 | i8 | i238 | i394 | i416 |
|---|---|---|---|---|---|---|---|
| 0.020 | -1.8 | +10.3 | +7.9 | +14.0 | +0.1 | +11.1 | +8.8 |
| 0.025 | +10.7 | +14.4 | +14.8 | +15.8 | +12.6 | +14.5 | +15.5 |
| **0.030** | **+2.2** | **+1.6** | **+2.7** | **-0.1** | **+3.2** | **+1.4** | **+2.2** |
| 0.040 | +1.9 | -3.0 | -1.8 | -10.8 | +1.8 | -3.9 | -3.5 |
| 0.050 | +7.8 | +0.3 | +2.0 | -5.2 | +6.8 | -0.4 | +0.9 |

h = 0.03 is the only step that lands all seven inside 4 per cent. It is also the
step E6 committed to independently. The step dependence is jagged rather than
smooth because the projected residual is small and jaxspec folds in float32; the
adversary saw the same thing and wrote it as "the finite version varies 10-25%
over shifts". Changing the Jacobian's own relative step from 1e-3 to 1e-2 moves
everything by under 1 per cent, so that knob does not matter.

### 3e. The estimator's own numerical noise [mine]

Because the gain derivative is a small residual after a five-parameter
projection, and jaxspec folds in float32, sigma_eff depends slightly on how many
spectra are folded in one call. Over batch sizes 4 to 500, holding the spectrum
fixed:

| spectrum | mean | sd | CV | min | max |
|---|---|---|---|---|---|
| i8 | 0.03850 | 0.00028 | 0.74% | 0.03816 | 0.03898 |
| i238 | 0.20323 | 0.00009 | 0.04% | 0.20313 | 0.20336 |
| i394 | 0.05065 | 0.00002 | 0.03% | 0.05063 | 0.05068 |
| i416 | 0.03747 | 0.00003 | 0.08% | 0.03742 | 0.03751 |

A one-row fold is worse: i8 comes out at 0.0401, 4 per cent above the batch mean.
Repeating a run at a fixed batch size is bit-exact (max relative difference 0.0
over two 500-row runs). The script therefore always folds the whole population in
one call.

Consequence, stated plainly: **the residual +2 per cent offset against the
recorded values is bigger than this noise for i238, i416 and i482, so it is a
genuine small difference in how the original was computed, not a rounding
artifact.** I could not pin it down. Candidates ruled out: the Jacobian step
(under 1 per cent), whitening by observed counts instead of `mu0` (moves the
seven in both directions, no better overall), and the shift size (section 3d).
Candidates I could not test: a different QR rank tolerance, or an average over
several shifts. It does not matter for any use the paper makes of the number.

### 3f. Shrink prediction versus the measured NS shrinks [mine]

Combining the likelihood width with the U[0.95, 1.05] prior (sd 0.0288675) in
quadrature, `shrink_pred = 1/sqrt(1 + (sigma_prior/sigma_eff)^2)`:

| spectrum | sigma_eff | predicted shrink | measured NS shrink | rank predicted | rank measured |
|---|---|---|---|---|---|
| i416 | 0.0374 | 0.792 | 0.7313 | 1 | 1 |
| i8 | 0.0385 | 0.804 | 0.8557 | 2 | 3 |
| i394 | 0.0506 | 0.869 | 0.7901 | 3 | 2 |
| i238 | 0.2032 | 0.990 | 1.0050 | 4 | 4 |

**The ordering does not fully match.** i8 and i394 swap. The screen gets the
most-informative spectrum (i416) and the null (i238) right and puts the middle
two in the wrong order. Predicted shrinks are also off on all three informative
spectra in both directions (0.79 vs 0.73, 0.80 vs 0.86, 0.87 vs 0.79), which is
expected: a Gaussian quadrature combination of a local Fisher width with a
bounded uniform prior is a crude approximation, and the exact posterior is not
Gaussian.

This is not new. The round-2 referee found the same thing,
`deliberation/round2_referee_2026-08-28.md:246-248` [recorded]:

> "5.2: sigma_eff = 0.037 to 0.050 exceeds the prior sd 0.0289 ...
> 1/sqrt(1/0.0289^2 + 1/sigma_eff^2), gives predicted shrinks 0.79, 0.87, 0.80
> against the observed"

Those are the same three numbers I get, to the digit. The paper's response was to
scope sigma_eff as ordinal, and that scoping is correct and necessary: **as a
width prediction it fails, and even ordinally it gets one of three pairs wrong.**
The wider claim survives because the population spans 0.024 to 0.43 and the three
informative spectra all sit in the bottom decile, which is a much coarser
statement than ordering them against each other.

---

## 4. Task 3: the committed script and what it says about the population

`scripts/screen_sigma_eff.py`. Docstring states the definition, the seeds, the
paper sentence it backs, and the reproduction result. Argparse takes `--level`,
`--n-pop`, `--seed-eval-base`, `--strength`, `--delta-g`, `--rel-step`,
`--soft-band-kev`, `--out`. Deterministic. Writes JSON and CSV with per-spectrum
index, total counts, soft-band counts and fraction, the five true parameters,
`sigma_eff_marginal`, `sigma_eff_conditional`, `I`, `||b||^2`, `R`, the predicted
posterior sd and shrink, the prior-edge margins, and an `already_run_ns` flag. It
re-checks the recorded counts gate at startup and prints pass/fail.

Runtime **12.8 s** bright, **11.4 s** medium, for 500 spectra each. Counts gate
4/4 and 6/6 pass.

### 4a. The bright population, n = 500 [mine]

sigma_eff (marginal): **median 0.0947**, IQR **[0.0534, 0.1648]**, range
**[0.0244, 0.4268]**. Median total counts 10542.

Fraction of the 500 below each candidate threshold:

| threshold | 0.02 | 0.0289 | 0.04 | 0.05 | 0.06 | 0.08 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|---|---|---|---|---|
| fraction | 0.000 | 0.008 | 0.106 | 0.224 | 0.298 | 0.426 | 0.528 | 0.698 | 0.842 |

The three information-carrying spectra sit at 0.0374 / 0.0385 / 0.0506, so they
are in roughly the most informative 10 to 22 per cent of the bright population.
Only 0.8 per cent of the population has a likelihood narrower than the prior.

The four already-run bright spectra:

| idx | counts | soft counts (<1.2 keV) | sigma_eff marginal | sigma_eff conditional | shrink_pred |
|---|---|---|---|---|---|
| i8 | 25067 | 13677 | 0.0390 | 0.00698 | 0.804 |
| i238 | 10496 | 613 | 0.2032 | 0.00865 | 0.990 |
| i394 | 6829 | 3213 | 0.0506 | 0.01447 | 0.869 |
| i416 | 54509 | 14116 | 0.0374 | 0.00685 | 0.792 |

(These sigma_eff are the 500-row values the script writes. i8 reads 0.0390 at
n=500 and 0.0385 as the batch-mean of section 3c; that is the float32 spread.)

My soft counts differ slightly from the adversary's: i394 3213 against a recorded
3248, i238 613 against 645. I take channels whose **centre** is below 1.2 keV;
the recorded numbers include roughly one more channel. Diagnostic column, nothing
depends on it, but the definition is now explicit and settable with
`--soft-band-kev`.

### 4b. The medium population, n = 500 [mine]

sigma_eff median **0.2997**, IQR [0.1686, 0.5202], range [0.0772, 1.3505]. Not
one spectrum of 500 falls below 0.06, and only 2.2 per cent below 0.10. The
medium level carries essentially no gain information anywhere in the prior, which
is the paper's claim and it holds strongly.

---

## 5. Task 4: proposed additional bright spectra

### 5a. Exclusions [mine]

Margins are measured in the coordinate the prior is uniform in, so log-space for
the two normalizations, as a fraction of the prior range.

- **kT within 10 per cent of the 3.0 ceiling.** Read as 10 per cent of the value
  (kT >= 2.7): **58 of 500**. Read as 10 per cent of the prior range
  (kT >= 2.73): 50 of 500. I use the stricter, kT >= 2.7.
- **Any other prior edge within 5 per cent of the prior range**: **199 of 500**.
  For reference, 1 - 0.9^5 = 41 per cent is what an unstructured prior gives for
  five parameters and a two-sided 5 per cent margin, and 199/500 = 39.8 per cent,
  so the rule is behaving as designed and not removing anything unusual.
- **Union removed: 237 of 500 (47.4 per cent). Survivors: 263.**

Worth flagging: **i8 and i416 are themselves excluded by these rules.** i8 has
kT = 0.362, which is 0.023 of the prior range above the kT floor; i416 has
PL_norm at 0.995 of its log-prior range, 0.005 from the ceiling, and BB_norm at
0.943. That last one is the known i416 rail (E7 calls the raw Gamma +2.80 sigma
there a rail artifact). So the exclusion rules, applied honestly, would have
removed two of the paper's three headline spectra. That is an argument for the
rules, and it means the new runs will be cleaner than the existing ones.

### 5b. The proposed ten [mine]

Ranked by sigma_eff among survivors, excluding the four already run. All ten are
more informative than i416, the most informative spectrum run so far.

| idx | counts | soft counts | soft % | sigma_eff | shrink_pred | nH | Gamma | PL_norm | kT | BB_norm | min edge margin | kT ceiling margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| i29 | 18954 | 9845 | 52 | 0.0298 | 0.718 | 0.300 | 2.638 | 6.72e-03 | 0.961 | 6.00e-02 | 0.086 | 0.755 |
| i45 | 34539 | 12442 | 36 | 0.0307 | 0.729 | 0.279 | 2.308 | 7.93e-03 | 1.870 | 2.63e-01 | 0.050 | 0.418 |
| i483 | 20577 | 12402 | 60 | 0.0308 | 0.730 | 0.227 | 2.897 | 6.53e-03 | 0.681 | 4.94e-01 | 0.051 | 0.859 |
| i262 | 29034 | 11213 | 39 | 0.0318 | 0.741 | 0.288 | 2.170 | 7.80e-03 | 1.822 | 1.02e-01 | 0.054 | 0.436 |
| i456 | 19060 | 9584 | 50 | 0.0324 | 0.746 | 0.275 | 2.642 | 6.04e-03 | 1.693 | 4.87e-02 | 0.110 | 0.484 |
| i53 | 15994 | 8073 | 50 | 0.0334 | 0.756 | 0.298 | 2.612 | 5.53e-03 | 1.797 | 2.21e-02 | 0.129 | 0.446 |
| i209 | 50865 | 7816 | 15 | 0.0347 | 0.769 | 0.313 | 2.893 | 4.41e-03 | 2.677 | 5.07e-01 | 0.054 | 0.120 |
| i128 | 19332 | 10380 | 54 | 0.0352 | 0.774 | 0.232 | 2.558 | 5.95e-03 | 1.741 | 2.02e-02 | 0.113 | 0.466 |
| i373 | 24455 | 9849 | 40 | 0.0358 | 0.778 | 0.267 | 2.028 | 6.76e-03 | 1.160 | 4.99e-02 | 0.085 | 0.682 |
| i437 | 51399 | 9605 | 19 | 0.0361 | 0.781 | 0.312 | 1.964 | 6.49e-03 | 2.103 | 6.24e-01 | 0.094 | 0.332 |

Next two if twelve are wanted: i228 (0.0363) and i435 (0.0363). Full ranked
survivor list in `selection_bright.json`.

Caveats I would not hide from a referee:

1. **The within-set ordering is not meaningful.** The ten span sigma_eff 0.0298
   to 0.0361, a 21 per cent range, against a +2 per cent method systematic and a
   demonstrated failure to order i8 against i394 correctly. Treat this as a set
   of ten comparably informative spectra, not a ranked ladder.
2. **The set is homogeneous by construction.** Nine of ten have Gamma above 1.96
   and all have 16k to 51k counts against a population median of 10.5k. That is
   the mechanism working as understood (gain information lives in absorbed
   soft-band counts, so steep and bright wins), but these ten test one corner of
   the prior. If the point is to answer "three is too few", ten more from the
   same corner answers it; if the point is to map where the flow starts losing
   gain information, a stratified draw across sigma_eff deciles is the better
   design. Logged as an open choice, not decided here.
3. i209 has a kT ceiling margin of 0.120, which clears the 0.10 rule but not by
   much. Swap it for i228 if you want more headroom.

---

## 6. Two adjacent numbers that do not reproduce [mine]

Both found while checking population statistics. Neither is part of the assigned
task; both are recorded because one of them is in the shipped paper.

### 6a. "predicted shrink 0.987 for the medium population" - SHIPPED, does not reproduce to the digit

`arxiv_successor/main.tex:340`: "a Fisher estimate for the medium population
predicts a shrink of 0.987", repeated at L374.

My medium population (n=500, seed 20300611): **median shrink_pred 0.9954, mean
0.9906**. On the E6 draw instead (n=200, `sample_prior`, seed 20260611):
**median 0.9957, mean 0.9906**. Neither statistic on either population gives
0.987. To get 0.987 you need a population sigma_eff near 0.177; my medium median
is 0.2997.

The claim's substance is unharmed. The point is that the medium population
predicts essentially no shrink, and 0.991 to 0.995 says that at least as strongly
as 0.987 does. But the third digit is not reproducible from this definition, and
the paper prints it as a specific Fisher result. My +2 per cent method systematic
cannot explain the gap: at shrink 0.99 a 2 per cent error in sigma_eff propagates
to 0.04 per cent in shrink.

### 6b. "~48% of prior draws" at bright - E7 report only, does not reproduce

`deliberation/E7_smallset_report_final.md:89`: "at bright, the likelihood carries
information for ~48% of prior draws", from `adv_external.md:105`: "48% of the
bright prior population predicted below shrink 0.86".

My value: **21.2 per cent** below shrink 0.86 (22.0 per cent on the E6 draw). The
fraction below shrink **0.95** is **46.0 per cent** (43.5 per cent on the E6
draw), which is almost certainly what the 48 per cent actually was. So the
statistic looks like it was computed against a 0.95 threshold and written up
against 0.86.

I confirmed **48 per cent does not appear in the shipped paper** (no matching
token in `arxiv_successor/main.tex`), so nothing public is affected. The E7 final
report is the citable source for E7, so the line should be corrected there.

---

## 7. What I ran

- `scripts/screen_sigma_eff.py --level bright` and `--level medium` (committed).
- Probe scripts in the session scratchpad, not committed, all superseded by the
  above: population regeneration and counts gate; the definition scan of section
  3b; the step scan of section 3d; the batch-size scatter of section 3e; the 6x6
  Fisher identity cross-check; the selection of section 5.
- Outputs in this directory: `sigma_eff_bright.{json,csv}`,
  `sigma_eff_medium.{json,csv}`, `selection_bright.json`, `PROGRESS.md`.
- No git operations. No files in `paper-sbi-xray-article/` touched.

Internal cross-checks passed: counts gate 10/10 exact; QR projection equals the
explicit 6x6 Fisher inverse to 1e-14; repeat runs bit-exact; the medium and
bright populations independently reproduce their recorded sigma_eff values under
the same definition.

Not done: no independent re-derivation by another operator, no adversary pass.
Per rule #2 this is a single-operator result and should get the multi-agent
treatment before anything from it is quoted in the paper.

---

## 8. Verdict on the paper sentence

**The triple reproduces.** The sentence as printed is defensible and now backed
by a committed script. Two things worth changing if the sentence is ever
revisited:

1. **State Delta g = 0.03 and say the shift is symmetric.** The audit already
   noted Delta g is only inferable from context. Without "symmetric" the
   one-sided reading gives 0.041 / 0.056 / 0.046, which is 12 to 17 per cent off
   the printed values, so a referee reimplementing it from the sentence alone
   would not reproduce it.
2. **Keep the ordinal scoping and do not weaken it.** It is doing real work. The
   quadrature shrink prediction misorders i8 against i394 and is off on all
   three; section 3f shows exactly how.

No change is required to the printed digits.

---

## 9. Decision log entries

Non-trivial choices made in this work, for `DECISIONS.md`:

- **Chose** the marginal Fisher width, `sqrt([F^-1]_gg)` over the 6x6 including
  g. **Alternatives**: conditional `1/sqrt(F_gg)` with the five physics
  parameters held fixed; the prior-combined posterior width.
- **Chose** the symmetric finite shift `[mu(1+dg) - mu(1-dg)]/(2 dg)`.
  **Alternatives**: the one-sided `+dg` shift the E6 B4 family literally applies;
  the one-sided `-dg` shift; Richardson extrapolation to dg -> 0.
- **Chose** dg = 0.03. **Alternatives**: anything in 0.02 to 0.05; a converged
  derivative.
- **Chose** to whiten by `mu0(theta_true)`, the model expectation.
  **Alternatives**: whiten by the observed Poisson counts the NS run actually saw.
- **Chose** to keep E6's Jacobian relative step 1e-3 and its 0.5 bound margin.
  **Alternatives**: 1e-2; an analytic Jacobian.
- **Chose** prior-edge margins measured in the sampling coordinate, so log-space
  for the two loguniform normalizations. **Alternatives**: linear space for all
  five.
- **Chose** to read "kT within 10 per cent of the ceiling" as kT >= 2.7, 10 per
  cent of the value. **Alternatives**: 10 per cent of the prior range, kT >= 2.73.
- **Chose** soft band = channels whose centre is below 1.2 keV. **Alternatives**:
  channels whose upper edge is below 1.2 keV; the recorded convention, which
  includes about one more channel.
- **Chose** ten additional spectra, ranked purely by sigma_eff. **Alternatives**:
  eight; twelve; a stratified draw across sigma_eff deciles to map the threshold
  rather than pile up in the informative corner.
- **Chose** to report the +2 per cent residual offset against the recorded values
  as unexplained. **Alternatives**: tune a free knob until it matches.
