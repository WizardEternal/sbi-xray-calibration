# Point 2 result: twelve medium paired-NS gain-null pairs (2026-09-03)

Analysis only. No new compute launched by this pass; all 12 rows (7 new +
5 Higson-reused) were already on disk. This file runs the gate, runs
`scripts/paired_higson_analyze.py` on the 12-pair merge and on the two
subsets, and lists what the paper text must change. No tracked file was
edited; `git status --porcelain -- outputs/ns_bench/revision_2026-09-02/`
returns nothing (the whole directory is untracked, per `.gitignore`).

---

## 1. Gate

Criteria: `counts_clean`/`counts_gain` match `new12_counts.json` for that
pair index; `ncall_clean`/`ncall_gain` < 400000 (the `max_ncalls` cap);
`logz`/`logzerr` present for both clean and gain; `exposure == 353.4`;
`theta_seed == 20260630`. Source: `medium/pairs_merged.jsonl` (cat of
`pairs_reused.jsonl` + `pair_1_2.jsonl` + `pair_3_4.jsonl` + `pair_4.jsonl`
+ `pair_5.jsonl` + `pair_7.jsonl` + `pair_11.jsonl`, verified 12 unique
indices 0-11, no duplicates).

**Result: 12/12 PASS. No CAP-HIT. No stale/duplicate row for pair 11**
(only `pair_11.jsonl`'s single row exists, timestamped 14:23, log
`medium/logs/PT2_UNIT_11.log`; the 10:01 `FAILED_1254` marker has no
corresponding jsonl row anywhere in the merge — confirmed by index-count
check below).

| pair | src | counts clean (row/exp) | counts gain (row/exp) | ncall clean | ncall gain | logz clean | logz gain | logzerr clean | logzerr gain | d_paired = logz_gain − logz_clean |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | higson | 1403/1403 | 1416/1416 | 274441 | 136498 | −288.2856 | −282.4966 | 0.1751 | 0.1349 | +5.7890 |
| 1 | new | 160/160 | 155/155 | 30787 | 31976 | −132.6582 | −128.3105 | 0.1489 | 0.2133 | +4.3477 |
| 2 | new | 2162/2162 | 2170/2170 | 38680 | 35975 | −316.9449 | −322.8155 | 0.2374 | 0.2877 | −5.8705 |
| 3 | new | 1222/1222 | 1255/1255 | 36529 | 36252 | −273.5615 | −273.4209 | 0.2030 | 0.3374 | +0.1405 |
| 4 | new | 159/159 | 159/159 | 327417 | 327417 | −165.7303 | −165.7303 | 0.1222 | 0.1222 | +0.0000 |
| 5 | new | 1454/1454 | 1372/1372 | 36151 | 40966 | −279.5473 | −280.3293 | 0.2341 | 0.3747 | −0.7821 |
| 6 | higson | 526/526 | 521/521 | 85664 | 73607 | −211.3907 | −212.6321 | 0.2032 | 0.3510 | −1.2414 |
| 7 | new | 1616/1616 | 1658/1658 | 51750 | 45295 | −281.5180 | −284.9668 | 0.3069 | 0.1848 | −3.4489 |
| 8 | higson | 90/90 | 90/90 | 52269 | 52448 | −121.4168 | −121.8236 | 0.2234 | 0.1823 | −0.4067 |
| 9 | higson | 2128/2128 | 2187/2187 | 46391 | 49601 | −309.5236 | −311.0516 | 0.2492 | 0.1288 | −1.5280 |
| 10 | higson | 3659/3659 | 3700/3700 | 82444 | 123158 | −326.6919 | −326.2359 | 0.1592 | 0.2502 | +0.4560 |
| 11 | new | 1666/1666 | 1669/1669 | 188125 | 213947 | −278.8285 | −274.8074 | 0.1309 | 0.1856 | +4.0211 |

All 12 rows: `exposure = 353.4`, `theta_seed = 20260630`, `min_num_live_points
= 400`, `dlogz = 0.5`, `max_ncalls = 400000`. Highest ncall of the 24 runs is
pair 4 clean/gain at 327417 (82% of cap) — well under the cap, not a
CAP-HIT.

**Flagged, not a gate failure — pair 4 (i=4, the new draw's lowest-count
pair) has bit-identical clean and gain results**: same counts (159/159),
same `ncall` (327417/327417), same `logz` (−165.73030761357592 both) and
same `logzerr` (0.12224436807816288 both), giving `d_paired = 0.0` exactly.
Traced to the script itself
(`scripts/paired_ns_gain_check.py`): `data_clean` and `data_gain` are drawn
with the *same* Poisson seed (`POISSON_SEED_BASE + i`) from `lam_clean` and
`lam_gain`, and at 159 total counts over 102 channels the 3% gain shift
apparently perturbs no channel's rounded rate enough to change a single
Poisson draw at that seed; the two NS runs then also share `seed=i`, so
identical data gives a bit-identical UltraNest run. This is a genuine
degenerate output of a correctly-running script, not a bug — the most
literal possible instance of "no evidence penalty from the gain shift" — and
it is included in the statistic below as `d_paired = 0`, not excluded.

---

## 2. Analyzer output, all 12 pairs (verbatim)

Command:

```powershell
.venv\Scripts\python.exe scripts\paired_higson_analyze.py `
    --jsonl outputs\ns_bench\revision_2026-09-02\medium\pairs_merged.jsonl `
    --log-dir-root outputs\ns_bench\revision_2026-09-02\medium\runs `
    --label medium_2026-09-02 `
    --out-dir outputs\ns_bench\revision_2026-09-02\medium
```

```
[in] 12 pairs from ['outputs\\ns_bench\\revision_2026-09-02\\medium\\pairs_merged.jsonl']
[higson] pair 1 clean sigma=0.1400 threads=400 nlive_exact=True [1.5s]
[higson] pair 1 gain  sigma=0.1473 threads=400 nlive_exact=True [1.0s]
[higson] pair 2 clean sigma=0.1579 threads=400 nlive_exact=True [1.0s]
[higson] pair 2 gain  sigma=0.1543 threads=400 nlive_exact=True [1.1s]
[higson] pair 3 clean sigma=0.1520 threads=400 nlive_exact=True [1.1s]
[higson] pair 3 gain  sigma=0.1460 threads=400 nlive_exact=True [1.1s]
[higson] pair 4 clean sigma=0.1463 threads=400 nlive_exact=True [1.2s]
[higson] pair 4 gain  sigma=0.1463 threads=400 nlive_exact=True [1.1s]
[higson] pair 5 clean sigma=0.1574 threads=400 nlive_exact=True [1.1s]
[higson] pair 5 gain  sigma=0.1612 threads=400 nlive_exact=True [1.1s]
[higson] pair 7 clean sigma=0.1695 threads=400 nlive_exact=True [1.2s]
[higson] pair 7 gain  sigma=0.1747 threads=400 nlive_exact=True [1.1s]
[higson] pair11 clean sigma=0.1403 threads=400 nlive_exact=True [1.0s]
[higson] pair11 gain  sigma=0.1491 threads=400 nlive_exact=True [1.0s]
# Paired gain-null + Higson floor: medium_2026-09-02

Generated 2026-09-03T14:44:05.

## Paired statistic

- n = 12 pairs; mean d_paired = +0.1231 nats, SD 3.2979, SEM 0.9520
- 95% t CI [-1.9724, +2.2185], t = 0.1293 (df 11), two-sided p = 0.899488

## NS-sampling floor

- per-run Higson sigma over 24 runs: median 0.14684 nats, range 0.13347 to 0.17471
- pairs with a point store: 12/12  (extrapolation needed: False)
- mean per-pair difference variance = 0.045703 nats^2; typical per-pair floor 0.2138 nats
- floor on the paired mean = 0.061713 nats (direct 0.061713)
- SEM / floor = 15.43
- floor from UltraNest's own logzerr = 0.092425 nats

## Variance attribution

- NS share of the paired variance: 0.420 per cent (Higson), 0.942 per cent (UltraNest logzerr) -> 0.42 to 0.94 per cent
- deleting ALL NS variance moves p from 0.899488 to 0.899277 (Higson) / 0.899014 (UltraNest)
```

(Full per-pair / per-run tables and JSON: `medium/paired_higson_results.md`,
`medium/paired_higson_results.json`. Exact values used below:
`mean_d_paired=0.12305681974682514`, `sem=0.9520355875197093`,
`p_two_sided=0.899487829222581`,
`floor_on_mean_higson_direct=0.061713490719756764`,
`sem_over_floor_higson=15.426701300092358`,
`ns_share_range_percent=[0.4201978818515818, 0.9424886772302998]`,
`p_deleting_all_ns_variance_higson=0.899277273135537`,
`p_deleting_all_ns_variance_ultranest=0.899013705656694`.)

**Direct vs extrapolated floor now agree** (`floor_on_mean_higson_direct ==
floor_on_mean_higson_extrapolated == 0.061713490719756764`,
`extrapolation_needed: false`), because all 12 pairs now carry a point
store — this retires the "extrapolated from five proxy pairs" caveat
(§4 below).

---

## 3. Subsets: 7 new pairs alone vs 5 Higson pairs alone

Command (new 7, same flags, `--jsonl` pointed at the 7-row merge, `--out-dir
medium/new7_only`):

```
n=7  mean d_paired = -0.2275 nats, SD 3.6901, SEM 1.3947
t = -0.1631 (df 6), two-sided p = 0.875808
per-run Higson sigma over 14 runs: median 0.15055, range 0.13999-0.17471
floor on the paired mean = 0.081971 nats (direct, 7/7 point stores)
SEM / floor = 17.01
NS share: 0.345% (Higson), 0.805% (UltraNest) -> 0.35 to 0.80 per cent
p deflated: 0.875808 -> 0.875595 (Higson) / 0.875311 (UltraNest)
```

Command (5 Higson, `--jsonl medium/pairs_reused.jsonl`, `--out-dir
medium/higson5_only`):

```
n=5  mean d_paired = +0.6138 nats, SD 2.9949, SEM 1.3394
t = 0.4583 (df 4), two-sided p = 0.670562
per-run Higson sigma over 10 runs: median 0.14166, range 0.13347-0.17170
floor on the paired mean = 0.093635 nats (direct, 5/5 point stores)
SEM / floor = 14.30
NS share: 0.489% (Higson), 1.033% (UltraNest) -> 0.49 to 1.03 per cent
p deflated: 0.670562 -> 0.669820 (Higson) / 0.668989 (UltraNest)
```

**Do the two subsets agree?** Not in sign of the mean, though both are
null-consistent on their own (p=0.88 and p=0.67) and both 95% CIs comfortably
straddle 0. The 5 Higson pairs alone sit at +0.614 (the same figure
CAMPAIGN_PLAN §1 flagged before the 7 new pairs ran); the 7 new pairs alone
sit at −0.227, moving the combined 12-pair mean down to +0.123 — closer to
zero than either subset and than the shipped +0.334. No |t| > 2 in either
subset or in the combined 12; nothing here breaks the null. The two subsets'
SEMs (1.39 vs 1.34) and per-run sigma medians (0.151 vs 0.142) are
consistent with each other and with the combined 24-run median (0.147); nothing
suggests the 7 new pairs are a different measurement population from the 5
reused ones.

---

## 4. Shipped vs new — comparison table

Shipped values from `--validate-shipped` run fresh today (reproduces the
committed 12-pair set — `outputs/ns_bench/paired_gain_check.jsonl`, the
2026-07-01 unreproducible jsonl — plus the 5 Higson point stores; output
`paired_higson_validation.json`/`.md`). New values from §2 above
(`medium/paired_higson_results.json`).

| quantity | shipped (file : field) | new 12-pair (file : field) | changes? |
|---|---|---|---|
| mean d_paired | +0.3344784737589137 (`paired_higson_validation.json : paired_statistic.mean_d_paired`) | +0.12305681974682514 (`medium/paired_higson_results.json : paired_statistic.mean_d_paired`) | yes, smaller magnitude, still ≈0 |
| SEM | 1.3701221698877286 (`...: paired_statistic.sem`) | 0.9520355875197093 (`...: paired_statistic.sem`) | yes, tighter (narrower count spread in the new draw) |
| p (two-sided) | 0.8116289751447194 (`...: paired_statistic.p_two_sided`) | 0.899487829222581 (`...: paired_statistic.p_two_sided`) | yes, more null-consistent |
| Higson floor on mean | 0.060440903873954885 nats (`...: paired_null_floor.floor_on_mean_higson_extrapolated`, extrapolated from 5/12) | 0.061713490719756764 nats (`...: paired_null_floor.floor_on_mean_higson_direct`, direct, 12/12) | ~+2%, no longer an extrapolation |
| SEM / floor | 22.66879020778741 (`...: paired_null_floor.sem_over_floor_higson`) | 15.426701300092358 (`...: paired_null_floor.sem_over_floor_higson`) | yes, ratio shrinks (SEM shrank faster than the floor grew) but stays >>1 |
| per-run sigma median (paired-NS runs only) | 0.14166 nats, range 0.13347-0.17170, n=10 (`paired_higson_validation.md`, "NS-sampling floor") | 0.14683941892481783 nats, range 0.13347-0.17471, n=24 (`medium/paired_higson_results.json : per_run_sigma_summary`) | yes, median +4%, range narrows at the top vs the 13-run figure below |
| NS share of paired variance | 0.19460 to 0.42554 per cent (`...: variance_attribution.ns_share_range_percent`) | 0.42020 to 0.94249 per cent (`medium/paired_higson_results.json : variance_attribution.ns_share_range_percent`) | yes, roughly doubles (SEM/floor ratio fell, so NS is proportionally more of the variance even though absolutely small) |
| p after deleting all NS variance | 0.8116289751447194 -> 0.8114493584751092 (Higson) / 0.8112355367399986 (UltraNest) | 0.899487829222581 -> 0.899277273135537 (Higson) / 0.899013705656694 (UltraNest) | yes, tracks the new p |

**Not replaced by this campaign — a different population.** The paper's
"13 fresh nested-sampling reruns" sentence (median 0.144, range 0.133-0.215)
draws on `outputs/ns_bench/higson/higson_results.json` (`n_jobs_done: 13`),
which is the 10 gain-pair runs (pairs 0,6,8,9,10, clean+gain) **plus 3
runs outside the paired gain-null set**: `B1_bright_idx3` (sigma 0.2150),
`B1_medium_idx2` (sigma 0.1806), `clean_bright_idx9` (sigma 0.1652) — used
for the B1 line-detection error budget and a block-reproduction check, not
for the paired statistic. Those 3 are untouched by this campaign. The new
12-pair, 24-run sigma set (median 0.14684, range 0.13347-0.17471) is a
narrower population that excludes them; its top end (0.175) sits below the
13-run figure's top end (0.215) only because that outlier (`B1_bright_idx3`,
a bright block run) isn't in the paired set at all. The "line detections are
safe" sentence (0.79%/0.07% of the medium/bright confidence half-widths)
uses exactly those 2 non-paired runs and is also untouched.

---

## 5. What the paper text must change

File: `paper-sbi-xray-article/journal_revision/main.tex` (same text in
`arxiv_successor/` and `rasti_paper/`). **Not edited by this task** —
reference only, current line numbers in the journal_revision copy.

- **L36 (abstract), L46, L299, L313, L318, L376** — every instance of
  `+0.33 \pm 1.37` (nats) and `$p=0.812$` / `0.8116` must become the new
  12-pair figures: **+0.12 ± 0.95 nats, p ≈ 0.899**. Abstract (L36) is at
  249/250 words per the standing hard cap — the replacement phrase must be
  the same length or shorter; "+0.33 ± 1.37" -> "+0.12 ± 0.95" is
  character-neutral.
- **L46** — "puts the gain's evidence signal at $+0.3 \pm 1.4$ nats,
  consistent with zero" -> "$+0.1 \pm 1.0$ nats, consistent with zero" (or
  the fuller-precision form used elsewhere).
- **L299 / L313 (`tab:matrix` cell)** — `$+0.33 \pm 1.37$ (paired)` ->
  `$+0.12 \pm 0.95$ (paired)`. The same-paragraph sentence "All twelve pairs
  sit at the medium exposure, so the evidence scheme has no gain measurement
  at bright" is still true (this campaign never ran a bright paired NS pair)
  and needs no change on its own account — only on account of Point 3's
  bright arm, if/when that ships.
- **L318, the floor paragraph, in full**:
  - "13 fresh nested-sampling reruns... median 0.144 nats, range 0.133 to
    0.215" — technically still correct as written (that 13-run set is
    untouched by this campaign); **editorial choice, not a correction**:
    either leave as is, or, if the sentence is meant to describe specifically
    the paired-run population, replace with the 24-run figure (median 0.147,
    range 0.133-0.175). Recommend leaving as is and not conflating the two
    populations — flagging for Karan's call, not deciding it here.
  - "The floor on the paired statistic is 0.060 nats, and that one is an
    extrapolation. The committed paired spectra cannot be reconstructed...
    transferred it to the mean of all twelve. That holds only as far as
    those five stand in..." — **delete the whole extrapolation caveat**. All
    twelve pairs now carry a point store; floor is measured directly at
    0.0617 nats, `extrapolation_needed: false`.
  - "The empirical SEM of the paired statistic is 1.370 nats, 22.7 times the
    floor" -> "0.952 nats, 15.4 times the floor". Still >>1: sampling noise
    is not why the null holds.
  - "nested-sampling share of the paired variance is 0.20 to 0.43 per cent"
    -> **0.42 to 0.94 per cent** (this is the new-campaign number; note the
    stale pre-fix leftover the campaign also caught, next bullet).
  - "deleting all of it moves the $p$-value from 0.8116 to between 0.8114
    and 0.8112" -> "from 0.899 to between 0.8993 and 0.8990" (both
    directions of deflation, Higson then UltraNest).
- **Independent of this campaign — a stale pre-fix number already present in
  L318**: even on the *shipped* (unchanged) numbers, "0.13 to 0.43 per cent"
  and "0.8116 to 0.8115" are wrong on their own terms — the 0.13 end and
  0.811506 come from the pre-2026-08-14-fix `Vbar=0.030028`/`floor=0.050023`
  estimator, not the corrected one the paper otherwise uses
  (`Vbar=0.043837`/`floor=0.060441`). The corrected shipped figures are
  **0.195 to 0.426 per cent** and **0.8116 -> 0.8114** (not 0.13/0.8115).
  This must be fixed whether or not the new 12-pair numbers ship (confirmed
  again today by `--validate-shipped`'s own NOTE line).
- **L376 (conclusions bullet 1)** — "The paired evidence is $+0.33 \pm 1.37$
  nats at medium exposure, with a sampling floor 22.7 times below that error
  bar, extrapolated from five proxy pairs." -> "+0.12 ± 0.95 nats..., floor
  15.4 times below..." and **delete "extrapolated from five proxy pairs"**
  (no longer true — all twelve are measured).

**No conclusion reverses.** Gain null-consistent: yes, more so (p 0.899 vs
0.812). Floor still far below the SEM: yes (15.4x vs 22.7x, both >>1).
6.4 keV line detections unaffected: yes (they use 2 runs outside this
campaign's set, both untouched). The 7-new-pairs-alone and 5-Higson-alone
subsets individually null-consistent too (p=0.876, p=0.671), so this is not
an artifact of one subset dominating the mean.

---

## Exact commands run (this session, in order)

```powershell
.venv\Scripts\python.exe scripts\import_higson_pairs.py `
    --out outputs\ns_bench\revision_2026-09-02\medium\pairs_reused.jsonl

# (bash) merge all 12 rows, ASCII/LF, no BOM
cat pairs_reused.jsonl pair_1_2.jsonl pair_3_4.jsonl pair_4.jsonl pair_5.jsonl pair_7.jsonl pair_11.jsonl > pairs_merged.jsonl

.venv\Scripts\python.exe scripts\paired_higson_analyze.py `
    --jsonl outputs\ns_bench\revision_2026-09-02\medium\pairs_merged.jsonl `
    --log-dir-root outputs\ns_bench\revision_2026-09-02\medium\runs `
    --label medium_2026-09-02 `
    --out-dir outputs\ns_bench\revision_2026-09-02\medium

# 7-new-only subset
.venv\Scripts\python.exe scripts\paired_higson_analyze.py `
    --jsonl outputs\ns_bench\revision_2026-09-02\medium\new7_only\pairs_new7.jsonl `
    --log-dir-root outputs\ns_bench\revision_2026-09-02\medium\runs `
    --label medium_new7_2026-09-02 `
    --out-dir outputs\ns_bench\revision_2026-09-02\medium\new7_only

# 5-Higson-only subset
.venv\Scripts\python.exe scripts\paired_higson_analyze.py `
    --jsonl outputs\ns_bench\revision_2026-09-02\medium\pairs_reused.jsonl `
    --log-dir-root outputs\ns_bench\revision_2026-09-02\medium\runs `
    --label medium_higson5_2026-09-02 `
    --out-dir outputs\ns_bench\revision_2026-09-02\medium\higson5_only

# shipped-number validation, re-run fresh for this report
.venv\Scripts\python.exe scripts\paired_higson_analyze.py --validate-shipped
```

Non-trivial choices made in this pass (no compute launched, so nothing for
`DECISIONS.md` — these are analysis-plumbing choices, not scientific ones):

1. Built `new7_only/pairs_new7.jsonl` and used `medium/pairs_reused.jsonl`
   directly (rather than writing a separate `higson5_only` jsonl copy) for
   the two subset runs in §3, since the analyzer's `--jsonl` takes the file
   as-is and no merge was needed for either single-source subset.
2. Used `cat` (bash) rather than PowerShell `Get-Content | Set-Content
   -Encoding ASCII` for the 12-way merge in §2 — checked first that none of
   the 7 source jsonls carry a BOM (`xxd` on the first bytes), so the
   ASCII-safety concern the runbook flags for PowerShell's default UTF-16
   `>` does not apply to a bash `cat`.
3. Read `paper-sbi-xray-article/journal_revision/main.tex` (a sibling repo,
   outside the campaign dir) to get the exact current sentences and line
   numbers for §5 — read-only, not edited.
