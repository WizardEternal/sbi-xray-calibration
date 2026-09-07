# Bright-arm comparison: paper-quoted / production (`train_npe_prod_bright`) / uncapped rerun (`gonogo_uncapped_bright`)

Mechanical comparison only, generated 2026-09-02. Uncapped flow: 162 epochs, converged
(`outputs/models/gonogo_uncapped_bright/summary.json`, best_validation_loss reached,
not epoch-capped). `arch.json` md5-identical between the two checkpoints (prior box,
exposure 3534s, response, channel count all fixed); only `flow_state.pt` differs.
Flag threshold: |uncapped − production| > 0.05 on an AUC, unless stated otherwise.

Sources:
- Production: `outputs/is_reweight/{is_ess_sweep_results.json,canonical_auc_results.json}`,
  `outputs/detect/results.jsonl` (level==bright), `outputs/detect/consequence.jsonl` (level==bright),
  `outputs/gain_marg/paired_gain_bias_bright.json`, `outputs/gain_marg/seed_runs/seed_20260724_fixed_bright.json`.
- Uncapped: `outputs/uncapped_bright/is_reweight/{is_ess_sweep_results.json,canonical_auc_results.json}`,
  `outputs/uncapped_bright/detect/{results.jsonl,consequence.jsonl}`,
  `outputs/uncapped_bright/gain_marg/paired_gain_bias_bright.json`,
  `outputs/uncapped_bright/gain_marg/seed_runs/result_fixed_bright.json`.
- Paper: `paper-sbi-xray-article/arxiv_successor/main.tex` (Table `tab:auc` ~L80-100 bright row;
  Section 5 `sec:isess` ~L264-296, Table `tab:e1auc`; Table `tab:matrix` ~L303-315; Section 7.1
  `sec:fixbias` ~L330).

## Table 1 — ESS / k-hat AUCs at bright (canonical, both seed sets)

| family | statistic | paper-quoted | production | uncapped | Δ (unc−prod) | flag |
|---|---|---|---|---|---|---|
| B1 | ESS AUC seed0 | 0.820 [0.751,0.882] | 0.8203000 | 0.9308667 | +0.1106 | **FLAG** |
| B1 | ESS AUC seed1 | 0.795 [0.732,0.853] | 0.7949000 | 0.9058000 | +0.1109 | **FLAG** |
| B1 | k-hat AUC (seed0) | 0.809 | 0.8094949 | 0.9553052 | +0.1458 | **FLAG** |
| B4 | ESS AUC seed0 | 0.543 [0.471,0.615] | 0.5426667 | 0.5611333 | +0.0185 | perm p=0.112 |
| B4 | ESS AUC seed1 | 0.496 | 0.4960000 | 0.5194667 | +0.0235 | perm p=0.606 |
| B4 | k-hat AUC (seed0) | 0.540 | 0.5404667 | 0.5476000 | +0.0071 | — |

Note: the paper's quoted B1/B4 ESS AUCs (0.820/0.796, 0.543/0.496) match **production**
exactly to 3 d.p., confirming production is what the paper currently cites. The
canonical_auc_results.json bright block is B1/B4 only in both runs (no B2/B3 canonical
cell exists in either directory). Clean-population reference values also shift:
clean ESS-frac median 0.00951 (prod) → 0.12282 (uncapped, +0.113); clean k-hat median
0.800 (prod) → 0.358 (uncapped, −0.442); clean-control k-hat-frac>0.7 = 0.5 (prod, 6/12
spectra) → 0.0 (uncapped, 0/12) — the uncapped flow's clean bright posteriors no longer
fail the PSIS reliability threshold that the production (capped) flow fails on half its
clean control spectra. Clean median ESS fraction: production 0.0095 vs uncapped 0.123.

**Verdict:** B1 ESS AUC rises by +0.11 on the uncapped flow; B4 stays null-consistent.

Corrected 2026-09-02 by supervisor after reading both canonical_auc_results.json files directly.

## Table 2 — Detection AUCs at bright, per (family, strength, detector)

48/48 cells present in both runs (4 families × 4 strengths × D1/D2/D3). Full row-by-row
diff computed; only cells with |Δ|>0.05 listed, all in D1:

| family | strength | detector | production | uncapped | Δ |
|---|---|---|---|---|---|
| B1 | 5e-6 | D1 | 0.495175 | 0.549200 | +0.0540 |
| B1 | 2e-5 | D1 | 0.674575 | 0.743500 | +0.0689 |
| B1 | 8e-5 | D1 | 0.892700 | 0.974500 | +0.0818 |
| B2 | 0.3  | D1 | 0.778300 | 0.832775 | +0.0545 |
| B2 | 0.5  | D1 | 0.667350 | 0.749375 | +0.0820 |
| B2 | 0.7  | D1 | 0.623850 | 0.687175 | +0.0633 |
| B3 | 1.5  | D1 | 0.485325 | 0.628225 | +0.1429 |
| B3 | 3.0  | D1 | 0.481200 | 0.559900 | +0.0787 |
| B4 | 0.5  | D1 | 0.464200 | 0.532575 | +0.0684 |
| B4 | 3.0  | D1 | 0.474325 | 0.539025 | +0.0647 |

Remaining 38 cells all |Δ|≤0.05; D2 and D3 cells never flag. Every flag is a D1
increase (uncapped never moves D1 down by >0.05); no D2 or D3 cell moves by >0.05.

**Best-over-strength-grid per family (paper Table 1 bright row / Table `tab:matrix`):**

| family | metric | paper | production | uncapped | Δ |
|---|---|---|---|---|---|
| B1 | best D1/D2 | 0.97 (D1) | 0.9702 (D1) | 0.9998 (D1) | +0.0297 |
| B1 | best D3 | 0.89 | 0.8929 | 0.8934 | +0.0005 |
| B2 | best D1/D2 | 0.84 (D2) | 0.8434 (D2) | 0.8481 (D2) | +0.0046 |
| B2 | best D3 | 0.96 | 0.9566 | 0.9567 | +0.0001 |
| B3 | best D1/D2 | 0.66 (D2) | 0.6618 (D2) | 0.6625 (D2) | +0.0007 |
| B3 | best D3 | 0.81 | 0.8079 | 0.8121 | +0.0043 |
| B4 | best D1/D2 | 0.51 | 0.5148 (D2) | 0.5390 (D1) | +0.0242 (winner flips D2→D1) |
| B4 | best D3 | 0.53 | 0.5321 | 0.5333 | +0.0012 |
| B1 bright, s=3e-4 (Table `tab:matrix`) | D1 | 0.970 | 0.9702 | 0.9998 | +0.0296 |
| B1 bright, s=3e-4 | D2 | 0.810 | 0.8100 | 0.8034 | −0.0066 |
| B4 bright, s=3% (Table `tab:matrix`) | D1 | 0.474 | 0.4743 | 0.5390 | **+0.0647 FLAG** |
| B4 bright, s=3% | D2 | 0.484 | 0.4840 | 0.4854 | +0.0014 |

## Consequence.jsonl (D1 posterior-predictive Γ bias, B1, bright, 4 strengths)

| strength | production dΓ_bias_mean (std) | uncapped dΓ_bias_mean (std) | Δ mean |
|---|---|---|---|
| 5e-6 | −0.06820 (0.2991) | −0.03106 (0.2435) | +0.0371 |
| 2e-5 | −0.10902 (0.4110) | −0.08034 (0.2899) | +0.0287 |
| 8e-5 | +0.00217 (0.4143) | +0.04747 (0.3323) | +0.0453 |
| 3e-4 | +0.20101 (0.5328) | +0.22552 (0.4804) | +0.0245 |

No mean-Δ flags (>0.05). Note bias_std shrinks at every strength in the uncapped run
(tighter posteriors), consistent with the flow now being trained to convergence rather
than capped.

## Table 3 — Paired gain bias at bright (fixed flow: prod=capped / uncapped=converged; gain-marg flow unchanged, `model_bright`)

| quantity | production fixed | uncapped fixed | gain-marg (unchanged) | paper-quoted |
|---|---|---|---|---|
| Γ paired bias, mean±se | +0.018222±0.002855 | +0.017913±0.003771 | +0.019491±0.003945 (prod) / +0.019716±0.003944 (unc) | fixed +0.018±0.006 (medium); bright seed-avg +0.019±0.005 |
| log10norm paired bias, mean±se | +0.006276±0.001053 | +0.003870±0.001072 | +0.003135±0.001020 (prod) / +0.003037±0.001013 (unc) | fixed +0.0063±0.0011 → gain-marg +0.0031±0.0010 |

**PL-norm "halving" check:** production fixed/gain-marg ratio = 0.0063/0.0031 ≈ 2.0×
(the paper's "halves"). Uncapped-fixed vs the same (unchanged) gain-marg value:
0.0039/0.0031 ≈ 1.25×. The gap shrinks from Δ=0.0032 (production) to Δ=0.0008 (uncapped) —
does **not** survive at the same magnitude when the fixed flow is the converged one;
most of the apparent norm-bias reduction tracks flow training state, not marginalization.

**Γ bias "~+0.018" check:** production fixed +0.0182, uncapped fixed +0.0179 — both
consistent with the paper's "~+0.018" and with each other (Δ=0.0003, well inside se).
Gain-marg Γ bias also stays close (+0.0195 prod / +0.0197 unc). Marginalization does not
move Γ bias in either flow-training state.

## Table 4 — Seed run: gamma_bias_mean, fixed flow, bright, gain=1.03

| | production (seed_20260724) | uncapped (result_fixed_bright) | Δ |
|---|---|---|---|
| gamma_bias_mean | +0.021554 (se 0.003031, z=7.11) | +0.023763 (se 0.003666, z=6.48) | +0.002209 |
| log10norm_bias_mean | +0.004395 (se 0.000920, z=4.78) | +0.001762 (se 0.000914, z=1.93) | −0.002633 |

Γ bias matches within ~0.7 se (not flagged). log10norm bias drops by more than either se
individually but the two CIs still overlap; z drops from 4.78σ (significant) to 1.93σ
(marginal) — direction consistent with the Table 3 norm-bias finding above.

## Table 5 — Uncapped flow's clean-data (unshifted) Γ bias, `paired_gain_bias_bright.json` → `cases.clean.fixed.gamma`

| | production (capped fixed flow) | uncapped (converged fixed flow) | gain-marg (unchanged, both runs) | Δ (unc−prod, fixed) |
|---|---|---|---|---|
| bias_mean | −0.087148 | −0.017755 | −0.019335 (prod) / −0.019246 (unc) | **+0.069393 FLAG** |
| bias_std | 0.341969 | 0.275672 | 0.276589 (prod) / 0.277168 (unc) | −0.066297 |
| coverage90 | 0.726 | 0.874 | 0.878 (prod) / 0.876 (unc) | **+0.148 FLAG** |

The capped production fixed flow's clean-data Γ bias/coverage were far off the (unchanged)
gain-marg flow's; the uncapped fixed flow's clean bias/coverage nearly match the gain-marg
flow's on both metrics.
