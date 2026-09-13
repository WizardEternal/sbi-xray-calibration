# Bright re-evaluation on the uncapped checkpoint

This directory holds the bright arm of the benchmark re-run against
`outputs/models/gonogo_uncapped_bright` instead of the production checkpoint
`outputs/models/train_npe_prod_bright`. It exists to answer whether the bright
results depend on the production flow having been trained under a step cap.

What is here:

- `is_reweight/is_ess_sweep_results.json`, `is_reweight/canonical_auc_results.json`
- `detect/results.jsonl` (48 rows), `detect/consequence.jsonl` (4 B1 rows)
- `gain_marg/paired_gain_bias_bright.json`
- `gain_marg/seed_runs/result_fixed_bright.json`, `cases_fixed_bright.npz`
- `COMPARISON.md`, the production-against-uncapped tables

Only the checkpoint changes. `arch.json` is md5-identical between the two
(`85d9a1d139721cf8d9e1a62c367e4110`), so the prior box, base model, exposure
(3534 s), response and channel count (102) are bit-for-bit the same, and only
`flow_state.pt` differs. Seeds, population sizes, families, detectors and the
gain-marginalized flow are all untouched.

## The three runs

Run 1, importance-sampling and ESS AUCs. The two steps are sequential:
`run_canonical_auc.py` does not recompute seed set 0, it re-reads
`is_ess_sweep_results.json` from its own `--out-dir`. Running step 2 first would
fail, or silently mix the uncapped seed set 1 with the production seed set 0.

```
python outputs/is_reweight/run_is_ess_sweep.py \
    --level bright --train-run gonogo_uncapped \
    --out-dir outputs/uncapped_bright/is_reweight

python outputs/is_reweight/run_canonical_auc.py \
    --level bright --train-run gonogo_uncapped \
    --out-dir outputs/uncapped_bright/is_reweight
```

349 s then 146 s. `is_ess_sweep_results.json` records `config.train_run` as
`gonogo_uncapped`, so its provenance is self-evident. Production comparison
values are `outputs/is_reweight/is_ess_sweep_results.json` `sweep.bright.aucs`
(B1 0.8203, B4 0.5427, B2 0.7087, B3 0.4509 on `auc_ess_eff`) and
`outputs/is_reweight/canonical_auc_results.json` `bright` (B4 seed_set0 0.5427,
seed_set1 0.4960; B1 seed_set0 0.8203, seed_set1 0.7949).

Run 2, detection AUCs D1/D2/D3. The 48 bright rows are 4 families x 4 strengths x
3 detectors from `configs/detect.yaml` at `n_clean=200`, `n_misspec=100`.

```
python scripts/run_detect_benchmark.py \
    --config configs/detect.yaml \
    --level bright --train-run gonogo_uncapped \
    --out-dir outputs/uncapped_bright/detect
```

The log's first line reads `=== level bright (ckpt gonogo_uncapped_bright) ===`,
which is the cheapest confirmation that the override took. Compare against the 48
`level == "bright"` rows of `outputs/detect/results.jsonl`.

Run 3, paired gain bias.

```
python outputs/gain_marg/eval_gainmarg_paired_bright.py \
    --fixed-dir outputs/models/gonogo_uncapped_bright \
    --out outputs/uncapped_bright/gain_marg
```

155 s. Compare against `outputs/gain_marg/paired_gain_bias_bright.json`
`paired.gamma_bias_delta.fixed.mean` = +0.018222.

## What `--fixed-dir` does and does not repoint

`--fixed-dir` repoints only the 5-parameter fixed-response flow. The
gain-marginalized flow stays `outputs/gain_marg/model_bright`, which the uncapped
retrain does not replace. A smoke run at `--n-test 8` shows it: the `fixed` arm
moved from +0.02318 to +0.03863 while the `gain_marg` arm was identical to all
printed digits (+0.04417) in both.

So the +0.019 +/- 0.005 that `summarize_seed_scatter.py` reports is the
gain-marginalized column (mean +0.019414, sd with ddof=1 0.004805, n=4). All four
of its sources use `model_bright`, so that number cannot move under this retrain.
What moves is the fixed-response column, mean +0.017948, sd 0.005493. The shipped
+0.019 is invariant to the capped-against-uncapped question by construction. It
was not re-measured and found to agree.

## The four evaluation seeds

They are four different scripts, not four seeds of one script. Two of the four
are reproducible from committed code:

| # | source file | producer | committed |
|---|---|---|---|
| 1 | `outputs/gain_marg/paired_gain_bias_bright.json` | `eval_gainmarg_paired_bright.py`, posterior mean, 1000 draws | yes |
| 2 | `seed_runs/seed_11.json` | `outputs/gain_marg/verify_e4_2026-07-24/seed11/eval_mine.py` | no |
| 3 | `seed_runs/seed_137.json` | `outputs/gain_marg/verify_e4_2026-07-24/seed137/eval_stage.py` | no |
| 4 | `seed_runs/seed_20260724_gainmarg_bright.json` | `seed_runs/eval_bright_seed.py`, posterior median, 3000 draws | yes |

Only #4 has a CLI:

```
python outputs/gain_marg/seed_runs/eval_bright_seed.py \
    --level bright --flow fixed \
    --fixed-dir outputs/models/gonogo_uncapped_bright \
    --out outputs/uncapped_bright/gain_marg/seed_runs
```

About 135 s at 3.7 to 4.9 cases/s. Compare to
`outputs/gain_marg/seed_runs/seed_20260724_fixed_bright.json` `gamma_bias_mean`
= +0.021554.

Scripts #2 and #3 have the production path hardcoded and no CLI, so re-running
them means editing them first. Neither was re-run.

## Caveats for reuse

Do not run `--flow gainmarg` against `--fixed-dir`. The flag is ignored for the
gain-marginalized arm by design, so it recomputes the identical committed number
at full cost.

`cases_<tag>.npz` is a resume checkpoint. Delete it first if you want a clean
restart instead of a resume.

Run 2 wall-clock is not stable across checkpoints, so do not read a change in it
as a failure. D1 rejection-samples with a `max_sampling_time` cap, and the single
cell `B1|0.0003|bright|D1` accounts for 404.6 s of the production bright total of
772.5 s, because badly misspecified spectra push the flow's mass outside the prior
box. A better-converged flow can leak less and finish much faster, or it can stall
differently. That total also excludes the one-off per-level state build (posterior
load, 200 clean spectra, the 500-embedding D2 reference, cached D1 and D2 clean
scores), which measured 15.8 s at `n_clean=30` with D1 at about 0.29 s per
spectrum.

`detect/results.jsonl` rows do not record `train_run`, so provenance for run 2 is
the directory name and this file. Adding a provenance key would have changed the
schema of the committed production output, so it was left alone.
