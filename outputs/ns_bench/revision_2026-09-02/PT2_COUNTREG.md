# Point 2: count-ratio regression, old vs new 12 pairs

## Script identification

`scripts/analyze_count_regression.py` (git-tracked: `git log --all --oneline -- scripts/analyze_count_regression.py`
-> commit `4106a3e "Add successor-paper analysis artifacts and scripts"`; `git ls-files --error-unmatch scripts/analyze_count_regression.py` succeeds).

**Exact specification (from the script's own docstring, confirmed by re-running it):**

- "count ratio" = **natural log** of `counts_gain / counts_clean` per pair, NOT a plain
  percent ratio. The regression is `d_paired = a + b * log(counts_gain / counts_clean)`,
  fit by OLS over the 12 pairs.
- `d_paired` (the paired log-evidence difference) is the response; `log(counts_gain/counts_clean)`
  is the sole regressor.
- The intercept `a` is what main.tex L324 quotes as "+0.03 +/- 0.87 (p=0.97)": the
  count-adjusted gain effect once the count-ratio slope is removed, with its OLS
  standard error and a two-sided t/normal-approx p-value against zero.
- The slope `b` (nats per e-fold of count ratio) gets its own two-sided **permutation**
  p-value (200,000 shuffles, seed 0) — this is a different p from the intercept's and is
  NOT the "p=0.97" in the text.
- The separate median/spread sentence ("+0.6 per cent, -10 to +3 per cent") is a
  **different, simpler statistic**: the plain percent ratio `(counts_gain/counts_clean - 1) x 100`,
  computed directly (not printed by this script; documented in
  `paper-sbi-xray-article/deliberation/merge_stage5_audit_fixes_log_2026-08-29.md` section 4.4
  as "recomputed by me" from the same committed file, not via the regression).

Command (repo venv, from repo root, unmodified script):
```
.venv\Scripts\python.exe scripts\analyze_count_regression.py
```
This hardcodes `outputs/ns_bench/paired_gain_check.jsonl` as input, so a parameterized
copy was made to point at the new file (see below).

## GATE: run on the retired `outputs/ns_bench/paired_gain_check.jsonl` — PASS

Verbatim output of the unmodified committed script:
```
pairs: n=12
log count ratio: mean -0.0030, sd 0.0373, range -0.1090 to +0.0321
paired difference: mean +0.3345, sd 4.7462

slope      = -101.6 +/- 24.2 nats per e-fold (t = -4.20, dof 10)
R^2        = 0.638
perm p     = 0.0021  (200000 permutations, seed 0)
intercept  = +0.03 +/- 0.87 (normal-approx p = 0.97)
resid sd   = 3.00 nats

clean logZ-counts trend (from count_controlled.json): slope -116.9, intercept 91.1, n_clean 56
```
This matches the paper's "+0.03 +/- 0.87 (p=0.97)" and the provenance note's
"+0.0324 +/- 0.8679" to printed precision. Separately, plain percent ratio on the same
12 rows: median +0.603% (paper "+0.6 per cent"), min -10.326% / max +3.265%
(paper "-10 to +3"). **GATE PASSES on both the regression and the descriptive statistic.**

Note for the record: the retired file's regression on the *plain percent ratio* instead of
the log ratio gives a **different, non-matching** pair of numbers — intercept +0.0860 +/-
0.8635 (p=0.923), slope -1.0583 +/- 0.2507 (p=0.0018) — confirming the paper's printed
"+0.03 +/- 0.87 (p=0.97)" is specifically the **natural-log-ratio** specification, not the
percent-linear one.

## NEW data: `outputs/ns_bench/revision_2026-09-02/medium/pairs_merged.jsonl` (12 pairs)

Column names (`counts_clean`, `counts_gain`, `d_paired`) are identical to the old file, so
no field mapping was needed. Only the hardcoded input *path* differed, so a parameterized
copy of the script was saved as `outputs/ns_bench/revision_2026-09-02/count_regression_new12.py`
(the one-line change — reading the path from `sys.argv[1]` instead of a hardcoded constant —
is commented at the top of the file). Run on **both** files to confirm the copy changes
nothing else:

```
.venv\Scripts\python.exe outputs\ns_bench\revision_2026-09-02\count_regression_new12.py outputs\ns_bench\paired_gain_check.jsonl
.venv\Scripts\python.exe outputs\ns_bench\revision_2026-09-02\count_regression_new12.py outputs\ns_bench\revision_2026-09-02\medium\pairs_merged.jsonl
```

Copy on the OLD file reproduces the gate byte-for-byte (same numbers as above). Copy on the
**NEW** file:
```
input file: outputs\ns_bench\revision_2026-09-02\medium\pairs_merged.jsonl
pairs: n=12
log count ratio: mean +0.0005, sd 0.0249, range -0.0580 to +0.0273
paired difference: mean +0.1231, sd 3.2979

slope      = -24.1 +/- 41.2 nats per e-fold (t = -0.58, dof 10)
R^2        = 0.033
perm p     = 0.5731  (200000 permutations, seed 0)
intercept  = +0.14 +/- 0.98 (normal-approx p = 0.89)
resid sd   = 3.40 nats

clean logZ-counts trend (from count_controlled.json, unchanged reference): slope -116.9, intercept 91.1, n_clean 56
```

## Independent cross-check (fresh code, `scipy.stats.linregress`, not copied from the
script above): `outputs/ns_bench/revision_2026-09-02/independent_crosscheck.py`

```
=== outputs/ns_bench/paired_gain_check.jsonl (n=12) ===
percent ratio: median +0.603%  min -10.326%  max +3.265%
[log-ratio spec, matches committed script] intercept a=+0.0324 SE=0.8679 p=0.9709   slope b=-101.6002 SE=24.2142 p=0.0018  R2=0.6378
[percent-linear spec, alternate/NOT the paper's stat] intercept a=+0.0860 SE=0.8635 p=0.9226   slope b=-1.0583 SE=0.2507 p=0.0018  R2=0.6406
plain mean +/- SEM of d_paired: +0.3345 +/- 1.3701

=== outputs/ns_bench/revision_2026-09-02/medium/pairs_merged.jsonl (n=12) ===
percent ratio: median +0.275%  min -5.640%  max +2.773%
[log-ratio spec, matches committed script] intercept a=+0.1354 SE=0.9821 p=0.8931   slope b=-24.0957 SE=41.2222 p=0.5718  R2=0.0330
[percent-linear spec, alternate/NOT the paper's stat] intercept a=+0.1428 SE=0.9818 p=0.8872   slope b=-0.2485 SE=0.4171 p=0.5645  R2=0.0343
plain mean +/- SEM of d_paired: +0.1231 +/- 0.9520
```

Agreement: `scipy.stats.linregress` (log-ratio spec) reproduces the committed script's
old-file numbers to 4 decimal places (intercept 0.0324/0.8679/0.9709, slope
-101.6002/24.2142) and the new-file numbers to 2-3 significant figures (intercept
0.1354->rounds to 0.14, matches script's 0.14; slope -24.0957 vs script's -24.1). The
plain mean +/- SEM of d_paired on the new 12 is +0.1231 +/- 0.9520, matching the task
brief's expected value exactly, and is close to (though not identical to) the log-ratio
intercept +0.14 +/- 0.98 — consistent with the near-zero R^2 (0.033) on the new data,
i.e. the count ratio explains almost none of the new paired variance, unlike the old data
(R^2 0.638).

## What main.tex L324 should now say (PROPOSAL, at the paper's existing precision)

> "Across the twelve pairs the shift moves total counts by a median of **+0.3** per cent,
> with a per-pair spread from **-6** to **+3** per cent... regressing the paired
> differences on the count ratio leaves **+0.14 +/- 0.98** (**p=0.89**)."

(Old sentence for comparison: median +0.6 per cent, spread -10 to +3 per cent; intercept
+0.03 +/- 0.87, p=0.97.)

## Surprise worth flagging

The new 12 pairs are noticeably better count-matched than the retired set: log-ratio sd
dropped from 0.0373 to 0.0249 (~33% smaller spread), and the count-ratio regression's R^2
collapsed from 0.638 to 0.033 — on the new data the count ratio explains almost none of
the paired variance (slope not even nominally significant: perm p=0.57, vs p=0.002 on the
old set). The intercept is still reported at the same precision for continuity with the old
sentence's format, but with R^2 this low the "regressing out the count ratio" framing is
much weaker evidence of anything on the new data than it was on the old — the intercept
(+0.14 +/- 0.98) and the plain unconditional mean +/- SEM (+0.12 +/- 0.95) are now nearly
the same number, because there is little count-ratio trend left to regress out. Whether
that framing sentence should even survive as written, versus just quoting the plain mean
+/- SEM directly, is a paper-editorial call outside this task's scope — flagging it rather
than deciding it.
