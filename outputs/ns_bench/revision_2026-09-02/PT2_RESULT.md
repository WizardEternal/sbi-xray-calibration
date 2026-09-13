# Twelve medium paired nested-sampling gain-null pairs

This directory holds a 12-pair paired nested-sampling gain-null set at medium
exposure, and the analysis of it. Seven pairs were run here, and their UltraNest
point stores are under `medium/runs/pair{1,2,3,4,5,7,11}_{clean,gain}`. The other
five are the 2026-07 Higson pairs (indices 0, 6, 8, 9, 10), reused in place from
`outputs/ns_bench/higson/runs/` and imported as rows by
`scripts/import_higson_pairs.py`.

Files:

- `medium/pairs_merged.jsonl`, the 12 rows, one per pair
- `medium/pairs_reused.jsonl`, the 5 reused rows alone
- `medium/paired_higson_results.json` and `.md`, the analyzer output
- `medium/paired_higson_results_recompute.jsonl`, the per-run checkpoint
- `new12_counts.json`, the counts gate
- `count_regression_new12.py`, `pair4_check.py`, `independent_crosscheck.py`
- `PT2_COUNTREG.md`, `PT2_PAIR4_CHECK.md`

The analysis came from `scripts/paired_higson_analyze.py`:

```
python scripts/paired_higson_analyze.py \
    --jsonl outputs/ns_bench/revision_2026-09-02/medium/pairs_merged.jsonl \
    --log-dir-root outputs/ns_bench/revision_2026-09-02/medium/runs \
    --label medium_2026-09-02 \
    --out-dir outputs/ns_bench/revision_2026-09-02/medium
```

## Gate

All 12 rows pass: counts match `new12_counts.json` per pair index, `ncall` stays
under the 400000 `max_ncalls` cap, `logz` and `logzerr` are present for both
arms, `exposure = 353.4`, `theta_seed = 20260630`, `min_num_live_points = 400`,
`dlogz = 0.5`. The highest `ncall` of the 24 runs is pair 4 at 327417, 82 per
cent of the cap.

| pair | src | counts clean | counts gain | ncall clean | ncall gain | logz clean | logz gain | logzerr clean | logzerr gain | d_paired |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | higson | 1403 | 1416 | 274441 | 136498 | -288.2856 | -282.4966 | 0.1751 | 0.1349 | +5.7890 |
| 1 | new | 160 | 155 | 30787 | 31976 | -132.6582 | -128.3105 | 0.1489 | 0.2133 | +4.3477 |
| 2 | new | 2162 | 2170 | 38680 | 35975 | -316.9449 | -322.8155 | 0.2374 | 0.2877 | -5.8705 |
| 3 | new | 1222 | 1255 | 36529 | 36252 | -273.5615 | -273.4209 | 0.2030 | 0.3374 | +0.1405 |
| 4 | new | 159 | 159 | 327417 | 327417 | -165.7303 | -165.7303 | 0.1222 | 0.1222 | +0.0000 |
| 5 | new | 1454 | 1372 | 36151 | 40966 | -279.5473 | -280.3293 | 0.2341 | 0.3747 | -0.7821 |
| 6 | higson | 526 | 521 | 85664 | 73607 | -211.3907 | -212.6321 | 0.2032 | 0.3510 | -1.2414 |
| 7 | new | 1616 | 1658 | 51750 | 45295 | -281.5180 | -284.9668 | 0.3069 | 0.1848 | -3.4489 |
| 8 | higson | 90 | 90 | 52269 | 52448 | -121.4168 | -121.8236 | 0.2234 | 0.1823 | -0.4067 |
| 9 | higson | 2128 | 2187 | 46391 | 49601 | -309.5236 | -311.0516 | 0.2492 | 0.1288 | -1.5280 |
| 10 | higson | 3659 | 3700 | 82444 | 123158 | -326.6919 | -326.2359 | 0.1592 | 0.2502 | +0.4560 |
| 11 | new | 1666 | 1669 | 188125 | 213947 | -278.8285 | -274.8074 | 0.1309 | 0.1856 | +4.0211 |

`d_paired` is `logz_gain - logz_clean`, in nats.

## Result

n = 12 pairs. Mean `d_paired` = +0.1231 nats, SD 3.2979, SEM 0.9520, 95 per cent
t CI [-1.9724, +2.2185], t = 0.1293 on 11 df, two-sided p = 0.899488. The gain
shift leaves no detectable evidence penalty.

Per-run Higson sigma over the 24 runs: median 0.14684 nats, range 0.13347 to
0.17471. All 12 pairs carry a point store, so the floor is measured directly and
is not extrapolated. Mean per-pair difference variance is 0.045703 nats^2, the
typical per-pair floor is 0.2138 nats, and the floor on the paired mean is
0.061713 nats. SEM over floor is 15.43. Taking UltraNest's own `logzerr` instead
gives a floor of 0.092425 nats.

Nested sampling accounts for 0.420 per cent of the paired variance on the Higson
estimate and 0.942 per cent on the `logzerr` estimate, so 0.42 to 0.94 per cent.
Deleting all of it moves p from 0.899488 to 0.899277 (Higson) or 0.899014
(UltraNest). Sampling noise is not why the null holds.

Full precision, as written in `medium/paired_higson_results.json`:
`mean_d_paired = 0.12305681974682514`, `sem = 0.9520355875197093`,
`p_two_sided = 0.899487829222581`,
`floor_on_mean_higson_direct = 0.061713490719756764`,
`sem_over_floor_higson = 15.426701300092358`,
`ns_share_range_percent = [0.4201978818515818, 0.9424886772302998]`.

## The two subsets

Running the analyzer on each source separately, with the same flags:

| subset | n | mean d_paired (nats) | SD | SEM | t | p | floor on mean (nats) | SEM/floor | NS share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7 new pairs | 7 | -0.2275 | 3.6901 | 1.3947 | -0.1631 (6 df) | 0.875808 | 0.081971 | 17.01 | 0.35 to 0.80 % |
| 5 reused pairs | 5 | +0.6138 | 2.9949 | 1.3394 | +0.4583 (4 df) | 0.670562 | 0.093635 | 14.30 | 0.49 to 1.03 % |

The two means do not agree in sign, though each is null-consistent on its own and
both 95 per cent CIs straddle zero comfortably. The 5 reused pairs sit at +0.614
and the 7 new ones at -0.227, which pulls the combined 12-pair mean to +0.123,
closer to zero than either subset. No |t| above 2 anywhere. The two SEMs (1.39
against 1.34) and per-run sigma medians (0.151 against 0.142) are consistent with
each other and with the combined 24-run median of 0.147, so the 7 new pairs do
not look like a different measurement population from the 5 reused ones.

## Against the earlier 12-pair set

The earlier set comes back from `python scripts/paired_higson_analyze.py
--validate-shipped`, which reads the 2026-07-01 committed jsonl plus the 5 Higson
point stores.

| quantity | earlier set | this 12-pair set |
|---|---|---|
| mean d_paired | +0.3344784737589137 nats | +0.12305681974682514 nats |
| SEM | 1.3701221698877286 | 0.9520355875197093 |
| p (two-sided) | 0.8116289751447194 | 0.899487829222581 |
| Higson floor on mean | 0.060440903873954885 nats, extrapolated from 5 of 12 | 0.061713490719756764 nats, direct, 12 of 12 |
| SEM / floor | 22.67 | 15.43 |
| per-run sigma median | 0.14166 nats, range 0.13347 to 0.17170, n=10 | 0.14684 nats, range 0.13347 to 0.17471, n=24 |
| NS share of paired variance | 0.195 to 0.426 per cent | 0.420 to 0.942 per cent |

The mean moves closer to zero, the SEM tightens because the new draw has a
narrower count spread, and the floor is now measured instead of extrapolated. The
SEM-over-floor ratio falls from 22.7 to 15.4 because the SEM shrank faster than
the floor grew, and it is still far above 1. No conclusion moves.

## Caveats for reuse

Pair 4 has bit-identical clean and gain results: same counts (159/159), same
`ncall` (327417 both), same `logz` (-165.73030761357592 both) and same `logzerr`
(0.12224436807816288 both), so `d_paired` is exactly 0. This traces to
`scripts/paired_ns_gain_check.py`, which draws `data_clean` and `data_gain` with
the same Poisson seed (`POISSON_SEED_BASE + i`) from `lam_clean` and `lam_gain`.
At 159 total counts over 102 channels the 3 per cent gain shift perturbs no
channel's rate enough to change a single draw at that seed, and the two runs then
share `seed=i`, so identical data gives a bit-identical UltraNest run. It is a
degenerate output of a correctly running script. It enters the statistic as
`d_paired = 0` and is not dropped. `PT2_PAIR4_CHECK.md` has the detail.

The 24-run sigma population here is not the same population as the 13-run set in
`outputs/ns_bench/higson/higson_results.json` (median 0.144 nats, range 0.133 to
0.215). That one is these 10 reused gain-pair runs plus 3 runs outside the paired
gain-null set: `B1_bright_idx3` (sigma 0.2150), `B1_medium_idx2` (0.1806) and
`clean_bright_idx9` (0.1652), which serve the B1 line-detection error budget and
a block-reproduction check. Nothing here touches those 3. The 24-run top end of
0.175 sits below the 13-run top end of 0.215 only because `B1_bright_idx3`, a
bright block run, is not in the paired set at all.

One number in the earlier set is stale on its own terms, independent of anything
measured here. Its paired variance share "0.13 to 0.43 per cent" and deflated p
"0.8116 to 0.8115" come from the pre-2026-08-14 estimator, `Vbar = 0.030028` with
floor 0.050023. Under the corrected estimator (`Vbar = 0.043837`, floor 0.060441)
the same earlier set gives 0.195 to 0.426 per cent and 0.8116 to 0.8114. The
`--validate-shipped` run prints a note to this effect. No conclusion moves, and
the digits are simply stale.

All twelve pairs sit at medium exposure, so this set says nothing about bright.
