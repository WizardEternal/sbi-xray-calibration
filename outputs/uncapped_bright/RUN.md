# Bright re-runs against `gonogo_uncapped_bright` — launch sheet

Launched and completed 2026-09-02, 17:05-17:34 IST (production-vs-uncapped comparison written 21:39 the same day). Committed here.

**What changes.** Only `outputs/models/train_npe_prod_bright` ->
`outputs/models/gonogo_uncapped_bright`. `arch.json` is md5-identical between the
two checkpoints (`85d9a1d139721cf8d9e1a62c367e4110`), so the prior box, base
model, exposure (3534 s), response and channel count (102) are bit-for-bit the
same. The only thing that differs is `flow_state.pt`. Everything else — seeds,
population sizes, families, detectors, the gain-marginalized flow — is untouched.

**Interpreter** (Git Bash, from anywhere; all paths absolute):

```bash
REPO="$(git rev-parse --show-toplevel)"
PY="$REPO/.venv/Scripts/python.exe"
```

`nproc` = **16**. Set `OMP_NUM_THREADS=4` on every job (a past session crashed
this laptop with thread exhaustion, 0x5AA). Three concurrent jobs at 4 threads
each = 12 threads, which is safe.

---

## Run 1 — IS / ESS AUCs  (`outputs/uncapped_bright/is_reweight/`)

Two steps, and they are **sequential**: `run_canonical_auc.py` does not recompute
seed set 0, it re-reads `is_ess_sweep_results.json` **from its own `--out-dir`**.
Running step 2 without step 1 first would either fail or, worse, silently mix the
uncapped seed-set-1 with the production seed-set-0.

```bash
mkdir -p "$REPO/outputs/uncapped_bright/is_reweight"
cd "$REPO"
export OMP_NUM_THREADS=4

# step 1 (~349 s)
nohup "$PY" outputs/is_reweight/run_is_ess_sweep.py \
    --level bright \
    --train-run gonogo_uncapped \
    --out-dir outputs/uncapped_bright/is_reweight \
    > outputs/uncapped_bright/is_reweight/sweep_bright.log 2>&1 &

# step 2 — ONLY after step 1 prints DONE (~146 s)
nohup "$PY" outputs/is_reweight/run_canonical_auc.py \
    --level bright \
    --train-run gonogo_uncapped \
    --out-dir outputs/uncapped_bright/is_reweight \
    > outputs/uncapped_bright/is_reweight/canonical_auc_bright.log 2>&1 &
```

One-liner that chains them safely:

```bash
nohup bash -c 'cd "'"$REPO"'" && export OMP_NUM_THREADS=4 && \
  "'"$PY"'" outputs/is_reweight/run_is_ess_sweep.py --level bright --train-run gonogo_uncapped --out-dir outputs/uncapped_bright/is_reweight && \
  "'"$PY"'" outputs/is_reweight/run_canonical_auc.py --level bright --train-run gonogo_uncapped --out-dir outputs/uncapped_bright/is_reweight' \
  > "$REPO/outputs/uncapped_bright/is_reweight/run1.log" 2>&1 &
```

**Expected wall:** 349 s + 146 s = **~500 s (~8 min)**. (Production timings:
`is_ess_sweep_results.json` `wall_s_by_level.bright` = 348.97 s;
`canonical_auc_bright.log` = "DONE bright in 146s".)

**Done looks like:**
- step 1 log last line: `DONE in ~349s (this process) -> .../is_ess_sweep_results.json`
- step 2 log last line: `DONE bright in ~146s -> .../canonical_auc_results.json`
- files: `is_ess_sweep_results.json` (with `config.train_run` == `"gonogo_uncapped"` —
  this script records the flow it used, so provenance is self-evident),
  `canonical_auc_results.json` (top-level key `bright` only),
  `weights_bright.png`, `ess_khat_bright.png`.

**Compare against:** `outputs/is_reweight/is_ess_sweep_results.json`
`sweep.bright.aucs` (B1 0.8203 / B4 0.5427 / B2 0.7087 / B3 0.4509 on
`auc_ess_eff`) and `outputs/is_reweight/canonical_auc_results.json` `bright`
(B4 seed_set0 0.5427, seed_set1 0.4960; B1 seed_set0 0.8203, seed_set1 0.7949).

---

## Run 2 — detection AUCs D1/D2/D3  (`outputs/uncapped_bright/detect/`)

The 48 committed bright rows are exactly 4 families x 4 strengths x 3 detectors
from `configs/detect.yaml` at `n_clean=200`, `n_misspec=100`. `--level bright`
selects them; the fresh `--out-dir` means an empty `results.jsonl`, so the
crash-resume logic skips nothing.

```bash
mkdir -p "$REPO/outputs/uncapped_bright/detect"
cd "$REPO"
export OMP_NUM_THREADS=4
nohup "$PY" scripts/run_detect_benchmark.py \
    --config configs/detect.yaml \
    --level bright \
    --train-run gonogo_uncapped \
    --out-dir outputs/uncapped_bright/detect \
    > outputs/uncapped_bright/detect/detect_bright.log 2>&1 &
```

**Expected wall:** the committed bright cells sum to **772.5 s**, and that number
excludes the one-off per-level state build (posterior load, 200 clean spectra,
500-embedding D2 reference, cached D1/D2 clean scores). Measured at `n_clean=30`
that build took 15.8 s and D1 costs ~0.29 s/spectrum, so at `n_clean=200` expect
**~60-90 s** on top: budget **~850-900 s (~15 min)**.

Caveat worth stating up front: this timing is not stable across checkpoints. D1
rejection-samples with a `max_sampling_time` cap, and one single cell
(`B1|0.0003|bright|D1`) accounts for 404.6 s of the 772.5 s because badly
misspecified spectra push the flow's mass outside the prior box. A
better-converged flow can leak less and finish much faster, or stall differently.
Do not read a wall-clock change as a failure.

**Done looks like:** log last line
`ALL DONE in <N>s -> .../outputs/uncapped_bright/detect/results.jsonl`, and

```bash
wc -l < "$REPO/outputs/uncapped_bright/detect/results.jsonl"   # must be 48
```

plus `scores.jsonl` and `consequence.jsonl` (4 B1 rows) in the same directory.
The log's first line must read `=== level bright (ckpt gonogo_uncapped_bright) ===`
— that is the cheapest confirmation the override took effect.

**Compare against:** the 48 `level == "bright"` rows of `outputs/detect/results.jsonl`.

Note: `results.jsonl` rows do **not** record `train_run`, so provenance for this
one is the directory name and this file. (Adding a provenance key was deliberately
not done — it would change the schema of the committed production output.)

---

## Run 3 — paired gain bias  (`outputs/uncapped_bright/gain_marg/`)

```bash
mkdir -p "$REPO/outputs/uncapped_bright/gain_marg"
cd "$REPO"
export OMP_NUM_THREADS=4
nohup "$PY" outputs/gain_marg/eval_gainmarg_paired_bright.py \
    --fixed-dir outputs/models/gonogo_uncapped_bright \
    --out outputs/uncapped_bright/gain_marg \
    > outputs/uncapped_bright/gain_marg/paired_bright.log 2>&1 &
```

**Expected wall:** **155 s** (`paired_gain_bias_bright.json` `config.wall_s_total`).

**Done looks like:** log last two lines `[fig] .../paired_gain_bias_bright.png`
and `[json] .../paired_gain_bias_bright.json`, both under
`outputs/uncapped_bright/gain_marg/`.

**Compare against:** `outputs/gain_marg/paired_gain_bias_bright.json`
`paired.gamma_bias_delta.fixed.mean` = +0.018222.

### READ THIS BEFORE INTERPRETING RUN 3

`--fixed-dir` repoints **only** the 5-parameter fixed-response flow. The
gain-marginalized flow stays `outputs/gain_marg/model_bright`, which the uncapped
retrain does not replace. Verified in smoke: at `--n-test 8` the `fixed` arm moved
from +0.02318 (prod) to +0.03863 (uncapped) while the `gain_marg` arm was
identical to all printed digits (+0.04417) in both runs.

Consequence for the shipped number: the "+0.019 +/- 0.005" that
`summarize_seed_scatter.py` reports is the **gain-marginalized** column
(mean +0.019414, sd(ddof=1) 0.004805, n=4 — re-derived today). Every one of its
four sources uses `model_bright`, so **that number cannot move under this
retrain**. What moves is the fixed-response column, mean +0.017948, sd 0.005493.
If the referee point is about the shipped +0.019, the honest answer is that it is
invariant to the capped-vs-uncapped question by construction, not that it was
re-measured and agreed.

### The four evaluation seeds

They are four **different scripts**, not four seeds of one script. Only two of the
four are reproducible from committed code:

| # | source file | producer | committed? |
|---|---|---|---|
| 1 | `outputs/gain_marg/paired_gain_bias_bright.json` | `eval_gainmarg_paired_bright.py` (posterior **mean**, 1000 draws) | yes |
| 2 | `seed_runs/seed_11.json` | `outputs/gain_marg/verify_e4_2026-07-24/seed11/eval_mine.py` | **no** (untracked) |
| 3 | `seed_runs/seed_137.json` | `outputs/gain_marg/verify_e4_2026-07-24/seed137/eval_stage.py` | **no** (untracked) |
| 4 | `seed_runs/seed_20260724_gainmarg_bright.json` | `seed_runs/eval_bright_seed.py` (posterior **median**, 3000 draws) | yes |

Only #4 has a CLI. Command for it (fixed arm; ~135 s, rate ~3.7-4.9 cases/s):

```bash
mkdir -p "$REPO/outputs/uncapped_bright/gain_marg/seed_runs"
cd "$REPO"
export OMP_NUM_THREADS=4
nohup "$PY" outputs/gain_marg/seed_runs/eval_bright_seed.py \
    --level bright --flow fixed \
    --fixed-dir outputs/models/gonogo_uncapped_bright \
    --out outputs/uncapped_bright/gain_marg/seed_runs \
    > outputs/uncapped_bright/gain_marg/seed_runs/seed_fixed_bright.log 2>&1 &
```

Done: `result_fixed_bright.json` + `cases_fixed_bright.npz` in that directory, log
ending in `RESULT {...}` with `gamma_bias_mean`. Compare to
`outputs/gain_marg/seed_runs/seed_20260724_fixed_bright.json` `gamma_bias_mean`
= +0.021554.

Do **not** run `--flow gainmarg` against `--fixed-dir`: that flag is ignored for
the gain-marginalized arm (by design), so it would just recompute the identical
committed number at full cost.

`cases_<tag>.npz` is a **resume checkpoint**. If a run is killed and you want a
clean restart rather than a resume, delete that file first. (A 45 s smoke left one
at `outputs/uncapped_bright/smoke/seed_runs/cases_fixed_bright.npz`; it is in the
smoke tree, not in the run tree, so it cannot be picked up by the commands above.)

Scripts #2 and #3 are untracked E-series files with the production path hardcoded
and no CLI. Re-running them would mean editing untracked scripts. Not done, not
recommended without a decision from Karan.

---

## Concurrency

All three runs read disjoint inputs and write disjoint directories, so **run 1,
run 2 and run 3 can go at the same time**, plus the seed_runs job as a fourth.
With `OMP_NUM_THREADS=4` each that is 16 threads on 16 cores — drop to 3
concurrent if the laptop is doing anything else.

The only ordering constraint anywhere is **inside run 1**: sweep, then canonical.

Wall-clock if all launched together: bounded by run 2 at ~15 min.

## Watching them

```bash
tail -f "$REPO/outputs/uncapped_bright/detect/detect_bright.log"
grep -c . "$REPO/outputs/uncapped_bright/detect/results.jsonl"   # 0 -> 48
```

## Safety re-check after the runs

```bash
cd "$REPO"
stat -c '%Y %s %n' outputs/detect/results.jsonl \
    outputs/gain_marg/paired_gain_bias_bright.json \
    outputs/is_reweight/is_ess_sweep_results.json \
    outputs/is_reweight/canonical_auc_results.json
```

must still print (mtime, size):

```
1781180902 30968    outputs/detect/results.jsonl
1784789840 5142     outputs/gain_marg/paired_gain_bias_bright.json
1784772056 176080   outputs/is_reweight/is_ess_sweep_results.json
1784772383 3384     outputs/is_reweight/canonical_auc_results.json
```

`outputs/uncapped_bright/` is **not** covered by `.gitignore`, so it shows up as
untracked in `git status`. Leave it untracked. `git add -A`, `git clean` and
`git stash` remain banned in this repo (147 untracked E-series files).
