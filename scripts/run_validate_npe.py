"""Phase-2 sanity validation of trained flows (NOT full SBC -- that's Phase 3).

For each trained flow named in a train config this script:
  * draws fresh test sims from the same generator (disjoint seed),
  * samples the posterior per test spectrum and computes the per-parameter
    posterior median + the 90% equal-tailed credible interval,
  * checks recovery: truth-vs-median scatter + Pearson r / R^2 per parameter,
  * checks calibration cheaply: the fraction of test spectra whose truth falls
    inside the 90% credible interval (target ~0.90; this is a quick coverage
    proxy, the rigorous SBC/TARP pass is Phase 3),
  * highlights N_HIGHLIGHT (default 5) random test spectra and reports, per
    spectrum, whether the truth is inside the 90% CR for every parameter
    (the "truth inside the 90% credible region most of the time" check),
  * records the median 90%-CI width per parameter (absolute and as a fraction of
    the prior width) so widths can be compared ACROSS count levels.

Outputs one figure per count level:
  outputs/diagnostics/npe_recovery_<level>.png
and, after all levels, a posterior-width-vs-count-level MONOTONICITY table +
verdict (widths must shrink as counts grow).

Usage (repo venv):
    .venv\\Scripts\\python.exe scripts\\run_validate_npe.py --config configs\\train_npe_prod.yaml
    .venv\\Scripts\\python.exe scripts\\run_validate_npe.py --config configs\\train_npe_prod.yaml --level medium
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbixcal import train_npe as tn
from sbixcal import priors as _priors
from sbixcal import responses as _responses
from sbixcal import simulate as _sim


CRED = 0.90              # credible level for the coverage / width checks
N_HIGHLIGHT = 5          # random spectra for the per-spectrum 90%-CR readout


def fresh_test_set(cfg, exposure_s, n_test, seed):
    base_model = cfg["base_model"]
    base = _responses.load_base_obsconf()
    obsconf = _responses.scale_exposure(base, exposure_s)
    rng = np.random.default_rng(seed)
    theta, x, names = _sim.simulate_spectra(
        base_model, cfg["priors"], obsconf, n_test, rng,
        apply_poisson=True, seed_for_fakeit=seed,
    )
    return theta, x, names


def validate_level(cfg, level, n_test=200, n_samples=400, seed_base=20260611):
    lname = level["name"]
    run_name = f"{cfg['name']}_{lname}"
    model_dir = tn._models_dir() / run_name
    post, info = tn.load_posterior(model_dir, device="cpu")
    param_names = info["param_names"]
    exposure_s = info["exposure_s"]
    param_order = tn._models.MODEL_PARAMS[cfg["base_model"]]
    prior_low, prior_high = _priors.prior_bounds(cfg["priors"], param_order)
    prior_width = prior_high - prior_low

    # deterministic per-level seed (Python's hash() is salted per process, which made
    # the test set and the recovery metrics non-reproducible run to run).
    seed = seed_base + 7000 + (zlib.crc32(lname.encode()) % 1000)
    theta, x, names = fresh_test_set(cfg, exposure_s, n_test, seed)
    assert names == param_names, (names, param_names)

    x_t = torch.as_tensor(x, dtype=torch.float32)
    n_params = theta.shape[1]
    medians = np.zeros((n_test, n_params))
    lo90 = np.zeros((n_test, n_params))   # 5th percentile
    hi90 = np.zeros((n_test, n_params))   # 95th percentile
    ranks = np.zeros((n_test, n_params), dtype=int)

    torch.manual_seed(seed)
    with torch.no_grad():
        for i in range(n_test):
            s = post.sample((n_samples,), x=x_t[i], show_progress_bars=False,
                            reject_outside_prior=False).detach().cpu().numpy()
            lo90[i], hi90[i], medians[i] = tn.credible_interval(s, CRED)
            ranks[i] = (s < theta[i]).sum(axis=0)

    # ---- recovery metrics ----
    rs, r2s = [], []
    for p in range(n_params):
        r = np.corrcoef(theta[:, p], medians[:, p])[0, 1]
        ss_res = np.sum((theta[:, p] - medians[:, p]) ** 2)
        ss_tot = np.sum((theta[:, p] - theta[:, p].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        rs.append(float(r))
        r2s.append(float(r2))

    # ---- 90% credible-region coverage (per parameter + all-params-jointly) ----
    inside = (theta >= lo90) & (theta <= hi90)               # (n_test, n_params)
    cover_per_param, cover_joint = tn.coverage_fraction(theta, lo90, hi90)

    # ---- posterior widths (for cross-level monotonicity) ----
    width = hi90 - lo90                                       # (n_test, n_params)
    med_width = np.median(width, axis=0)                     # per-param median 90% width
    med_width_frac = med_width / prior_width                 # as fraction of prior width

    # ---- highlight N_HIGHLIGHT random spectra: truth-in-90%-CR readout ----
    hrng = np.random.default_rng(seed + 12345)
    hidx = hrng.choice(n_test, size=min(N_HIGHLIGHT, n_test), replace=False)
    highlights = []
    for j in hidx:
        per_p_in = inside[j].tolist()
        highlights.append({"idx": int(j), "all_in": bool(inside[j].all()),
                           "per_param_in": per_p_in})

    # ---- plot: top row recovery scatter, bottom row rank histograms ----
    out = Path(tn._repo_root()) / "outputs" / "diagnostics" / f"npe_recovery_{lname}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, n_params, figsize=(3.2 * n_params, 6.6))
    if n_params == 1:
        axes = axes.reshape(2, 1)
    for p in range(n_params):
        ax = axes[0, p]
        # 90% CI error bars on the recovery scatter
        yerr = np.vstack([medians[:, p] - lo90[:, p], hi90[:, p] - medians[:, p]])
        ax.errorbar(theta[:, p], medians[:, p], yerr=yerr, fmt="o", ms=3,
                    alpha=0.35, color="C0", ecolor="C0", elinewidth=0.5, capsize=0)
        # highlight the N_HIGHLIGHT spectra in red/green
        for h in highlights:
            j = h["idx"]
            col = "tab:green" if inside[j, p] else "tab:red"
            ax.errorbar(theta[j, p], medians[j, p],
                        yerr=[[medians[j, p] - lo90[j, p]], [hi90[j, p] - medians[j, p]]],
                        fmt="s", ms=6, color=col, ecolor=col, elinewidth=1.4,
                        capsize=2, zorder=5)
        lo = min(theta[:, p].min(), medians[:, p].min())
        hi = max(theta[:, p].max(), medians[:, p].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"true {param_names[p]}")
        ax.set_ylabel("posterior median (90% CI)")
        ax.set_title(f"{param_names[p]}\nr={rs[p]:.3f} R2={r2s[p]:.3f} "
                     f"cov90={cover_per_param[p]:.2f}", fontsize=9)
        ax.grid(alpha=0.3)
        # mini rank histogram
        axh = axes[1, p]
        nbins = 10
        axh.hist(ranks[:, p], bins=nbins, range=(0, n_samples),
                 color="C1", alpha=0.8, edgecolor="k", linewidth=0.4)
        axh.axhline(n_test / nbins, color="k", ls="--", lw=1)
        axh.set_xlabel(f"rank of truth ({param_names[p]})")
        axh.set_ylabel("count")
        axh.set_title("mini rank hist (visual)", fontsize=8)
    fig.suptitle(f"{cfg['name']} {lname} (~{info['median_total_counts']:.0f} counts): "
                 f"recovery on {n_test} sims, mean r={np.mean(rs):.3f}, "
                 f"joint 90% coverage={cover_joint:.2f}\n"
                 f"squares = {len(highlights)} highlighted spectra "
                 f"(green=truth in 90% CI, red=outside)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # ---- console summary ----
    print(f"[{lname}] median_counts~{info['median_total_counts']:.0f}  "
          f"mean_r={np.mean(rs):.3f}  joint90_cov={cover_joint:.2f}")
    for p in range(n_params):
        print(f"    {param_names[p]:20s} r={rs[p]:.3f} R2={r2s[p]:.3f} "
              f"cov90={cover_per_param[p]:.2f} "
              f"med90width={med_width[p]:.4g} ({med_width_frac[p]:.3f} of prior)")
    n_all_in = sum(h["all_in"] for h in highlights)
    print(f"    {len(highlights)} highlighted spectra: truth inside 90% CR for "
          f"ALL params in {n_all_in}/{len(highlights)}")
    print(f"    saved {out}")

    return {
        "level": lname,
        "param_names": param_names,
        "median_counts": float(info["median_total_counts"]),
        "r": rs, "r2": r2s,
        "mean_r": float(np.mean(rs)),
        "cover_per_param": cover_per_param.tolist(),
        "cover_joint": float(cover_joint),
        "med_width": med_width.tolist(),
        "med_width_frac": med_width_frac.tolist(),
        "highlights_all_in": int(n_all_in),
        "highlights_total": int(len(highlights)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default=None)
    ap.add_argument("--n-test", type=int, default=200)
    args = ap.parse_args(argv)

    cfg = tn.load_config(args.config)
    results = []
    for level in cfg["levels"]:
        if args.level and level["name"] != args.level:
            continue
        results.append(validate_level(cfg, level, n_test=args.n_test))

    # ---- recovery + coverage table ----
    print("\n=== RECOVERY / COVERAGE ===")
    pn = results[0]["param_names"]
    header = "| level | ~counts | " + " | ".join(f"r({p})" for p in pn) + \
             " | mean r | joint 90% cov |"
    print(header)
    print("|" + "---|" * (len(pn) + 4))
    for r in results:
        cells = " | ".join("%.3f" % v for v in r["r"])
        print(f"| {r['level']} | {r['median_counts']:.0f} | {cells} | "
              f"{r['mean_r']:.3f} | {r['cover_joint']:.2f} |")

    # ---- posterior-width vs count-level monotonicity ----
    if args.level is None and len(results) >= 2:
        # order levels by count
        order = sorted(range(len(results)), key=lambda i: results[i]["median_counts"])
        print("\n=== POSTERIOR-WIDTH vs COUNT-LEVEL ===")
        print("median 90%-CI width as a FRACTION of the prior width "
              "(should shrink as counts grow):")
        wh = "| param | " + " | ".join(
            f"{results[i]['level']} (~{results[i]['median_counts']:.0f})" for i in order) \
            + " | monotone shrink? |"
        print(wh)
        print("|" + "---|" * (len(order) + 2))
        all_monotone = True
        for p, pname in enumerate(pn):
            fracs = [results[i]["med_width_frac"][p] for i in order]
            mono = all(fracs[k] >= fracs[k + 1] for k in range(len(fracs) - 1))
            all_monotone = all_monotone and mono
            cells = " | ".join("%.3f" % f for f in fracs)
            print(f"| {pname} | {cells} | {'yes' if mono else 'NO'} |")
        print(f"\nAll parameters' posterior widths shrink monotonically with "
              f"count level: {'YES' if all_monotone else 'NO'}")

        import json
        summary = {
            "what": "Phase-2 sanity validation of the production flows "
                    "(scripts/run_validate_npe.py, deterministic per-level seed). Recovery "
                    "Pearson r, 90% credible-interval coverage, median 90%-CI width as a "
                    "fraction of the prior width. NOT the rigorous SBC/TARP pass (Phase 3).",
            "config": args.config, "base_model": cfg["base_model"],
            "n_test": args.n_test, "cred": CRED, "param_names": pn,
            "levels": {r["level"]: {
                "median_total_counts": int(r["median_counts"]),
                "r": [round(v, 3) for v in r["r"]],
                "r2": [round(v, 3) for v in r["r2"]],
                "mean_r": round(r["mean_r"], 3),
                "cover_per_param": r["cover_per_param"],
                "cover_joint": r["cover_joint"],
                "med_width_frac_of_prior": [round(v, 3) for v in r["med_width_frac"]],
            } for r in results},
            "monotone_width_shrink_all_params": bool(all_monotone),
        }
        sp = Path(tn._repo_root()) / "outputs" / "diagnostics" / "validation_prod_summary.json"
        sp.write_text(json.dumps(summary, indent=2))
        print(f"[written] {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
