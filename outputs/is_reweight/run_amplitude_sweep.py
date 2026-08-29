r"""Gain-amplitude control at the medium count level.

Distinguishes TRUE BLINDNESS (ESS-efficiency AUC stays flat ~0.5 across gain
amplitude) from a POWER-LIMITED NULL (AUC rises monotonically with gain).

Strength -> gain factor mapping (sbixcal.misspec.simulate_misspec_population,
family B4): gain = 1.0 + strength/100, so
    strength 1.0  -> 1% gain  (gain_shift_obsconf(oc, 1.01))
    strength 3.0  -> 3% gain  (gain_shift_obsconf(oc, 1.03))  <- committed grid point
    strength 5.0  -> 5% gain  (gain_shift_obsconf(oc, 1.05))
    strength 10.0 -> 10% gain (gain_shift_obsconf(oc, 1.10))

Negative class: reuses the medium clean_cell from the committed main-sweep
JSON (n=150, ess_frac, seeds 80000-81999 range), with no recomputation.
Positive class: fresh B4 draw per amplitude (n=100), own disjoint seed block,
budget=6000 (same IS budget as the main sweep), same is_reweight() call used
everywhere else in this directory.

Uses only the run_is_ess_sweep.py functions, with no method changes.

    set OMP_NUM_THREADS=4
    .venv\Scripts\python.exe outputs\is_reweight\run_amplitude_sweep.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_is_ess_sweep as R  # noqa: E402
from run_canonical_auc import bootstrap_auc_ci, permutation_p  # noqa: E402

ROOT = R.ROOT
OUT = R.OUT
N_MIS = 100
N_BUDGET = 6000
LEVEL = "medium"
BLOCK_IDX = 1
STRENGTHS = [1.0, 3.0, 5.0, 10.0]  # gain percent
SEED_BASE = 950000  # disjoint from main sweep (60000-89999) and canonical_auc (900000-909999)


def main():
    t0 = time.time()
    print(f"===== AMPLITUDE SWEEP: {LEVEL}, B4 gain =====", flush=True)
    lev = R.Level(LEVEL)

    with open(OUT / "is_ess_sweep_results.json") as f:
        sweep = json.load(f)
    clean = np.asarray(sweep["sweep"][LEVEL]["cells"]["clean"]["ess_frac"], dtype=np.float64)
    print(f"  reused clean cell: n={clean.size} median_ess_frac={np.nanmedian(clean):.4f}", flush=True)

    results = {"level": LEVEL, "clean_n": int(clean.size),
               "clean_ess_frac_median": float(np.nanmedian(clean)),
               "n_mis": N_MIS, "n_budget": N_BUDGET, "points": {}}

    for si, strength in enumerate(STRENGTHS):
        print(f"  strength={strength:g}% ...", flush=True)
        xm, _ = R.gen_misspec(lev, "B4", strength, N_MIS, seed=SEED_BASE + si * 1000)
        mis = np.array([
            R.is_reweight(lev, xm[i], N_BUDGET, seed=SEED_BASE + 1000 + si * 1000 + i)["ess_frac"]
            for i in range(N_MIS)
        ])
        _, _, auc = R.roc_auc(-clean, -mis)
        lo, hi, _ = bootstrap_auc_ci(clean, mis, n_boot=3000, seed=SEED_BASE + 2000 + si)
        auc_chk, pval = permutation_p(clean, mis, n_perm=3000, seed=SEED_BASE + 3000 + si)
        results["points"][f"{strength:g}pct"] = {
            "gain_factor": 1.0 + strength / 100.0,
            "auc_ess_eff": float(auc), "ci95_lo": lo, "ci95_hi": hi,
            "perm_p_vs_0p5": pval,
            "ess_frac_median_mis": float(np.nanmedian(mis)),
        }
        print(f"    AUC={auc:.4f} 95%CI=[{lo:.4f},{hi:.4f}] perm_p={pval:.4f} "
              f"eff_mis_med={np.nanmedian(mis):.4f} (clean {np.nanmedian(clean):.4f})", flush=True)
        # incremental crash-safe save
        results["wall_s_partial"] = time.time() - t0
        with open(OUT / "amplitude_sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    results["wall_s"] = time.time() - t0

    # figure: AUC vs gain amplitude with CI band
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = STRENGTHS
    aucs = [results["points"][f"{s:g}pct"]["auc_ess_eff"] for s in xs]
    los = [results["points"][f"{s:g}pct"]["ci95_lo"] for s in xs]
    his = [results["points"][f"{s:g}pct"]["ci95_hi"] for s in xs]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(xs, aucs, "o-", color="C0", label="ESS-eff AUC (B4 vs clean)")
    ax.fill_between(xs, los, his, color="C0", alpha=0.25, label="bootstrap 95% CI")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="AUC=0.5 (null)")
    ax.set_xlabel("B4 gain shift amplitude (%)")
    ax.set_ylabel("ESS-efficiency ROC AUC")
    ax.set_title(f"{LEVEL}: gain-amplitude sweep (flat=blind, rising=power-limited)")
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "amplitude_sweep_medium.png", dpi=130)
    plt.close(fig)

    with open(OUT / "amplitude_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"DONE in {results['wall_s']:.0f}s -> {OUT/'amplitude_sweep_results.json'}", flush=True)


if __name__ == "__main__":
    main()
