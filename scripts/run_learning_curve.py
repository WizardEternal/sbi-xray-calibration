"""Phase-2 learning curve: DEV model, middle (medium) count level.

Trains the dev flow at N in {10k, 25k, 50k} on the medium dev dataset (prefixes
of the 50k training set -- deterministic seed order, so a prefix is a valid
fixed-seed subsample). For each N records:
  * best validation loss (the flow's own model-selection metric), and
  * a quick posterior-quality proxy: mean per-parameter Pearson correlation
    between the true theta and the posterior median on a fresh held-out test set
    (200 sims drawn from the same generator at a disjoint seed).

Outputs:
  outputs/diagnostics/learning_curve_dev.png
  prints a table for RESULTS.md (also the 50k-vs-100k verdict inputs).

Usage:
    .venv\\Scripts\\python.exe scripts\\run_learning_curve.py --config configs\\train_npe_dev.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbixcal import train_npe as tn
from sbixcal import responses as _responses
from sbixcal import simulate as _sim


def make_test_set(cfg, level_exposure_s, n_test=200, seed=999):
    """Fresh test sims from the same dev generator at a disjoint seed."""
    base_model = cfg["base_model"]
    base = _responses.load_base_obsconf()
    obsconf = _responses.scale_exposure(base, level_exposure_s)
    rng = np.random.default_rng(seed)
    theta, x, names = _sim.simulate_spectra(
        base_model, cfg["priors"], obsconf, n_test, rng,
        apply_poisson=True, seed_for_fakeit=seed,
    )
    return (torch.as_tensor(theta, dtype=torch.float32),
            torch.as_tensor(x, dtype=torch.float32), names)


def posterior_quality(de, prior, cfg, theta_test, x_test, n_samples=150, seed=0):
    """Mean per-parameter Pearson r between true theta and posterior median."""
    from sbi.inference import NPE
    inf = NPE(prior=prior, density_estimator="nsf", device="cpu", show_progress_bars=False)
    post = inf.build_posterior(de, prior=prior)
    de.eval()
    n_params = theta_test.shape[1]
    medians = np.zeros((theta_test.shape[0], n_params))
    torch.manual_seed(seed)
    with torch.no_grad():
        for i in range(theta_test.shape[0]):
            # reject_outside_prior=False: medians only; avoids slow rejection
            # loops when an underfit flow puts mass just outside the box. Box is
            # wide vs the posterior, so the median is unaffected in practice.
            s = post.sample((n_samples,), x=x_test[i], show_progress_bars=False,
                            reject_outside_prior=False)
            medians[i] = s.median(0).values.detach().cpu().numpy()
    true = theta_test.cpu().numpy()
    rs = []
    for p in range(n_params):
        r = np.corrcoef(true[:, p], medians[:, p])[0, 1]
        rs.append(float(r))
    return rs, float(np.mean(rs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_npe_dev.yaml")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    cfg = tn.load_config(args.config)
    lc = cfg["learning_curve"]
    dataset = lc["dataset"]
    grid = lc["n_train_grid"]
    param_order = tn._models.MODEL_PARAMS[cfg["base_model"]]

    # full dataset + its exposure (for the matched test set)
    _, _, _, meta = tn.load_dataset(dataset)
    exposure_s = meta["exposure_s"]
    theta_test, x_test, names = make_test_set(cfg, exposure_s, n_test=200, seed=20260611 + 999)
    prior = tn.build_prior(cfg["priors"], param_order, device=args.device)

    rows = []
    for n in grid:
        theta, x, _, m = tn.load_dataset(dataset, max_n=n)
        t0 = time.perf_counter()
        de, _, summary = tn.train_one_flow(theta, x, prior, cfg, device=args.device, seed=cfg["seed"])
        dt = time.perf_counter() - t0
        vl = summary["validation_loss"]
        best_val = float(np.min(vl))
        final_val = float(vl[-1])
        rs, mean_r = posterior_quality(de, prior, cfg, theta_test, x_test)
        rows.append({"n": n, "best_val": best_val, "final_val": final_val,
                     "rs": rs, "mean_r": mean_r, "epochs": summary["epochs_trained"],
                     "wall_s": dt})
        print(f"  N={n:>6d}: best_val={best_val:7.3f} mean_r={mean_r:.3f} "
              f"per-param r={['%.3f' % v for v in rs]} epochs={summary['epochs_trained']} "
              f"wall={dt:.1f}s")

    # ---- plot ----
    out = Path(tn._repo_root()) / "outputs" / "diagnostics" / "learning_curve_dev.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ns = [r["n"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ns, [r["best_val"] for r in rows], "o-", color="C0")
    ax1.set_xlabel("training-set size N")
    ax1.set_ylabel("best validation loss (-log prob)")
    ax1.set_title("Validation loss vs N (dev, medium)")
    ax1.set_xscale("log")
    ax1.grid(alpha=0.3)

    for p, pname in enumerate(param_order):
        ax2.plot(ns, [r["rs"][p] for r in rows], "o-", label=pname)
    ax2.plot(ns, [r["mean_r"] for r in rows], "k--", lw=2, label="mean")
    ax2.set_xlabel("training-set size N")
    ax2.set_ylabel("Pearson r (true vs posterior median)")
    ax2.set_title("Recovery quality vs N (200 fresh test sims)")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")

    # persist the underlying numbers next to the PNG (reproducible record)
    json_out = out.with_name("learning_curve_dev.json")
    with open(json_out, "w") as f:
        json.dump({
            "config": args.config,
            "base_model": cfg["base_model"],
            "dataset": dataset,
            "seed": cfg["seed"],
            "param_order": param_order,
            "n_test": 200,
            "rows": rows,
        }, f, indent=2)
    print(f"saved {json_out}")

    print("\n=== SUMMARY (for RESULTS.md) ===")
    print("| N | best_val | mean_r | per-param r | epochs | wall (s) |")
    print("|---|----------|--------|-------------|--------|----------|")
    for r in rows:
        pr = ", ".join("%.3f" % v for v in r["rs"])
        print(f"| {r['n']} | {r['best_val']:.3f} | {r['mean_r']:.3f} | {pr} | "
              f"{r['epochs']} | {r['wall_s']:.1f} |")
    # verdict helper
    if len(rows) >= 2:
        dv = rows[-2]["best_val"] - rows[-1]["best_val"]
        dr = rows[-1]["mean_r"] - rows[-2]["mean_r"]
        print(f"\n25k->50k: dval={dv:+.3f}, d(mean_r)={dr:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
