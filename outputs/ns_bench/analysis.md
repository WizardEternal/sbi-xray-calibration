# Phase-5 NS-vs-NPE benchmark: analysis

Spectra analyzed: 76 (56 clean, 20 misspecified). Detector cross-check: **READY** (144 detector cells available).

## 1. Speed vs agreement (clean Model-A spine)

| level | ~counts | n | NS s/spec | NS n_like_evals | NPE ms/spec | NS/NPE speedup | q-agreement (mean |dq|/width) |
|---|---|---|---|---|---|---|---|
| faint | 47 | 25 | 1065.5 | 49878 | 83 | 12802x | 0.068 |
| medium | 540 | 16 | 941.2 | 52463 | 106 | 8864x | 0.037 |
| bright | 7910 | 15 | 1751.4 | 95386 | 177 | 9882x | 0.100 |

## 2. NS truth recovery (clean; 90% interval coverage proxy)

| level | n | NS 90% interval contains truth (mean over params) |
|---|---|---|
| faint | 25 | 0.87 |
| medium | 16 | 0.91 |
| bright | 15 | 0.85 |

## 3. NS misspecification flags vs Phase-4 detector AUC

NS evidence flag = mean(logZ_misspec - logZ_clean) at the same level (more negative => the well-specified Model A fits the misspecified spectrum worse => flagged). Detector AUCs are read read-only from outputs/detect/results.jsonl; cells the (still-running) detector grid has not produced yet show as pending.

| family | strength | level | n | mean d-logZ (mis - clean) | D1 AUC | D2 AUC | D3 AUC | detector status |
|---|---|---|---|---|---|---|---|---|
| B1 | 0.0003 | bright | 6 | -886.5 | 0.970 | 0.810 | 0.893 | ready |
| B1 | 0.0003 | medium | 6 | -81.8 | 0.972 | 0.846 | 0.917 | ready |
| B4 | 3 | bright | 4 | -1.9 | 0.474 | 0.484 | 0.446 | ready |
| B4 | 3 | medium | 4 | -7.8 | 0.482 | 0.477 | 0.456 | ready |
