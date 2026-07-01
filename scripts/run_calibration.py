"""Run the full Phase-3 calibration suite against a trained checkpoint dir.

Usage (repo venv):
    .venv\\Scripts\\python.exe scripts\\run_calibration.py --config configs\\calibration.yaml
    .venv\\Scripts\\python.exe scripts\\run_calibration.py --config configs\\calibration.yaml --level medium
    .venv\\Scripts\\python.exe scripts\\run_calibration.py --config configs\\calibration.yaml --checkpoint outputs\\models\\train_npe_prod_faint --level faint

For each evaluated count level the suite produces, in outputs/calibration/<level>/:
    sbc_ranks.png                 SBC rank histograms (sbi.analysis.sbc_rank_plot)
    sbc.npz                       ranks + KS p-values + C2ST-of-ranks (uniformity stats)
    tarp.png / tarp.npz           TARP expected-coverage curve + ATC + KS p-value
    coverage_before_after.png/npz raw-vs-conformal-recalibrated per-param coverage
    is_refinement.npz             per-case ESS + low-ESS flags (Paper III diagnostic)
    summary.json                  all headline numbers for this level

Skip-if-exists per artifact (delete the level folder or pass --force to redo).
The flow, prior, base model, exposure and response are all read from the
checkpoint's arch.json -- the fresh test sims use exactly the prior/simulator the
flow was trained for (the correct SBC setup).

This script READS a checkpoint directory; it never writes into outputs/models.
Set OMP_NUM_THREADS=2 when the machine is busy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from sbixcal import calibrate as C
from sbixcal import train_npe as _tn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cov_at(nominal, cov, target):
    """Coverage (mean over parameters) at the nominal level closest to ``target``.

    ``cov`` is (n_levels, n_params); returns a float (mean over params at the
    nearest available nominal level) plus the per-param vector."""
    nominal = np.asarray(nominal)
    j = int(np.argmin(np.abs(nominal - target)))
    per_param = np.asarray(cov)[j]
    return float(np.mean(per_param)), per_param.tolist(), float(nominal[j])


def _calib_dir(level: str) -> Path:
    # Namespaced output root so a second response (e.g. NICER) does not collide
    # with / silently skip on the XMM outputs. Set SBIXCAL_CALIB_OUT to redirect.
    import os
    base = os.environ.get("SBIXCAL_CALIB_OUT")
    root = Path(base) if base else _repo_root() / "outputs" / "calibration"
    return root / level


def _checkpoint_for_level(train_run: str, level: str) -> Path:
    return _repo_root() / "outputs" / "models" / f"{train_run}_{level}"


def run_one_checkpoint(ckpt_dir: Path, level: str, cfg: dict,
                       force: bool = False, device: str = "cpu") -> dict | None:
    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / "arch.json").exists():
        print(f"[skip] {level}: no checkpoint at {ckpt_dir}")
        return None

    out_dir = _calib_dir(level)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"[skip] {level}: {summary_path} exists (use --force to redo)")
        with open(summary_path) as f:
            return json.load(f)

    # ---- cold-load the flow + the exact training prior/model/exposure ----
    post, info = _tn.load_posterior(ckpt_dir, device=device)
    with open(ckpt_dir / "arch.json") as f:
        arch = json.load(f)
    prior_cfg = arch["prior_cfg"]
    base_model = arch["base_model"]
    exposure_s = float(arch["exposure_s"])
    param_names = info["param_names"]
    response = cfg.get("response")

    prior = _tn.build_prior(prior_cfg, param_names, device=device)
    nominal = np.asarray(cfg.get("nominal_levels",
                                 list(np.round(np.linspace(0.05, 0.95, 19), 4))))
    base_seed = int(cfg.get("seed", 0))

    print(f"\n=== calibrating {level}: {ckpt_dir.name} "
          f"(~{info['median_total_counts']:.0f} counts) ===")

    # ====================================================================
    # 1. SBC  (fresh sims from the SAME prior/simulator the flow was trained for)
    # ====================================================================
    n_sbc = int(cfg.get("n_sbc", 1000))
    n_post = int(cfg.get("n_posterior_samples", 1000))
    theta_sbc, x_sbc, _, names = C.make_fresh_test_set(
        base_model, prior_cfg, exposure_s, n_sbc,
        seed=base_seed + 101, response_name=response)
    assert names == param_names, (names, param_names)

    sbc = C.run_sbc_check(post, prior, theta_sbc, x_sbc, param_names,
                          num_posterior_samples=n_post, seed=base_seed + 1)
    C.save_sbc_figure(sbc, out_dir / "sbc_ranks.png")
    np.savez(out_dir / "sbc.npz",
             ranks=sbc.ranks, ks_pvals=sbc.ks_pvals, c2st_ranks=sbc.c2st_ranks,
             c2st_dap=sbc.c2st_dap, param_names=np.array(param_names),
             num_posterior_samples=sbc.num_posterior_samples)
    print(f"  SBC: KS p-vals {np.round(sbc.ks_pvals, 3).tolist()}  "
          f"C2ST(ranks) {np.round(sbc.c2st_ranks, 3).tolist()}  "
          f"[stat: {sbc.uniformity_stat}]")

    # ====================================================================
    # 2. TARP expected coverage (joint)
    # ====================================================================
    n_tarp = int(cfg.get("n_tarp", n_sbc))
    theta_tp, x_tp, _, _ = C.make_fresh_test_set(
        base_model, prior_cfg, exposure_s, n_tarp,
        seed=base_seed + 202, response_name=response)
    tarp = C.run_tarp_check(post, theta_tp, x_tp,
                            num_posterior_samples=n_post, seed=base_seed + 2)
    C.save_tarp_npz_and_figure(tarp, out_dir / "tarp.npz", out_dir / "tarp.png")
    print(f"  TARP: ATC={tarp.atc:+.4f}  KS p={tarp.ks_pval:.3f}")

    # ====================================================================
    # 3b. conformal recalibration + before/after per-parameter coverage
    # ====================================================================
    n_cal = int(cfg.get("n_cal", 400))
    n_test = int(cfg.get("n_test", 400))
    n_cps = int(cfg.get("coverage_posterior_samples", n_post))

    th_cal, x_cal, _, _ = C.make_fresh_test_set(
        base_model, prior_cfg, exposure_s, n_cal,
        seed=base_seed + 303, response_name=response)
    th_test, x_test, _, _ = C.make_fresh_test_set(
        base_model, prior_cfg, exposure_s, n_test,
        seed=base_seed + 404, response_name=response)

    s_cal = C.sample_posterior_batch(post, x_cal, n_cps, seed=base_seed + 31, device=device)
    s_test = C.sample_posterior_batch(post, x_test, n_cps, seed=base_seed + 41, device=device)

    nlv, cov_raw, cov_recal, recal = C.coverage_before_after(
        s_cal, th_cal, s_test, th_test, param_names, nominal_levels=nominal)
    C.save_coverage_before_after(
        nlv, cov_raw, cov_recal, param_names,
        out_dir / "coverage_before_after.npz",
        out_dir / "coverage_before_after.png",
        title_suffix=f"({level}, ~{info['median_total_counts']:.0f} counts)")

    raw_dev = float(np.mean(np.abs(cov_raw - nlv[:, None])))
    recal_dev = float(np.mean(np.abs(cov_recal - nlv[:, None])))
    print(f"  Coverage (mean |emp-nominal|): raw={raw_dev:.3f} -> "
          f"recalibrated={recal_dev:.3f}")

    # ====================================================================
    # 3a. importance-sampling refinement (Paper III) -- ESS report
    # ====================================================================
    n_is_cases = int(cfg.get("n_is_cases", 20))
    n_is = int(cfg.get("n_is_samples", 2000))
    low_ess_frac = float(cfg.get("low_ess_frac", 0.1))
    prior_logp = C._prior_box_log_prob(prior_cfg, param_names)

    th_is, x_is_pois, x_is_exp, _ = C.make_fresh_test_set(
        base_model, prior_cfg, exposure_s, n_is_cases,
        seed=base_seed + 505, response_name=response)

    # model_counts_fn for IS: fold proposal thetas through the SAME response to get
    # noiseless model counts (lambda) for the exact Poisson likelihood.
    from sbixcal import responses as _responses
    from sbixcal import simulate as _sim
    base_oc = _responses.load_base_obsconf(response or _responses.EXAMPLE_NAME)
    is_obsconf = _responses.scale_exposure(base_oc, exposure_s)

    def model_counts_fn(theta_arr):
        # theta_arr (M, n_params) -> (M, n_channels) noiseless model counts
        return _sim.fold_theta(base_model, param_names, theta_arr, is_obsconf)

    ess_list, essfrac_list, lowess_list = [], [], []
    is_med_shift = []   # |refined median - raw median| per param, averaged
    for i in range(n_is_cases):
        is_res = C.importance_refine(
            post, x_is_pois[i], model_counts_fn, prior_logp,
            n_samples=n_is, seed=base_seed + 5000 + i,
            low_ess_frac=low_ess_frac, device=device)
        ess_list.append(is_res.ess)
        essfrac_list.append(is_res.ess_frac)
        lowess_list.append(is_res.low_ess)
        raw_med = np.median(is_res.samples, axis=0)
        ref_med = C.is_refined_quantiles(is_res, [0.5])[0]
        is_med_shift.append(float(np.mean(np.abs(ref_med - raw_med))))

    ess_arr = np.array(ess_list); essfrac_arr = np.array(essfrac_list)
    n_low = int(np.sum(lowess_list))
    np.savez(out_dir / "is_refinement.npz",
             ess=ess_arr, ess_frac=essfrac_arr,
             low_ess=np.array(lowess_list), median_shift=np.array(is_med_shift),
             low_ess_frac_threshold=low_ess_frac)
    print(f"  IS-refine ({n_is_cases} cases, {n_is} draws): "
          f"median ESS={np.median(ess_arr):.0f} "
          f"(frac {np.median(essfrac_arr):.3f}); "
          f"{n_low}/{n_is_cases} flagged low-ESS (<{low_ess_frac:.0%})")

    # ====================================================================
    # 3a-cov. IS-refinement BEFORE/AFTER coverage (the recalibration partner
    #         of the conformal before/after above). Run on a held-out IS-coverage
    #         test set so raw, conformal and IS coverage are directly comparable.
    # ====================================================================
    n_is_cov = int(cfg.get("n_is_cov", 150))
    th_isc, x_isc_pois, x_isc_exp, _ = C.make_fresh_test_set(
        base_model, prior_cfg, exposure_s, n_is_cov,
        seed=base_seed + 606, response_name=response)
    # raw (un-refined) per-parameter coverage on the SAME observations:
    s_isc = C.sample_posterior_batch(post, x_isc_pois, n_cps,
                                     seed=base_seed + 61, device=device)
    cov_raw_isc = C.empirical_coverage_curve(s_isc, np.asarray(th_isc), nominal)
    is_cov = C.is_coverage_curve(
        post, x_isc_pois, th_isc, model_counts_fn, prior_logp, nominal,
        n_is_samples=n_is, seed=base_seed + 6000,
        low_ess_frac=low_ess_frac, device=device)
    np.savez(out_dir / "is_coverage.npz",
             nominal_levels=nominal,
             cov_raw=cov_raw_isc, cov_is_all=is_cov["cov_all"],
             cov_is_okess=is_cov["cov_okess"],
             ess=is_cov["ess"], ess_frac=is_cov["ess_frac"],
             low_ess=is_cov["low_ess"], param_names=np.array(param_names),
             n_low_ess=is_cov["n_low_ess"], n_cases=is_cov["n_cases"])
    is_raw_dev = float(np.mean(np.abs(cov_raw_isc - nominal[:, None])))
    is_all_dev = float(np.mean(np.abs(is_cov["cov_all"] - nominal[:, None])))
    print(f"  IS coverage ({n_is_cov} cases): mean |emp-nominal| "
          f"raw={is_raw_dev:.3f} -> IS-refined={is_all_dev:.3f}; "
          f"{is_cov['n_low_ess']}/{is_cov['n_cases']} low-ESS "
          f"(frac {is_cov['n_low_ess']/is_cov['n_cases']:.2f})")

    # ====================================================================
    # summary
    # ====================================================================
    out_summary = {
        "level": level,
        # repo-relative, forward-slash form: summary.json is a committed
        # artifact and must not record an absolute local path
        "checkpoint": Path(os.path.relpath(ckpt_dir, _repo_root())).as_posix(),
        "median_total_counts": float(info["median_total_counts"]),
        "param_names": param_names,
        "sbc": {
            "ks_pvals": sbc.ks_pvals.tolist(),
            "c2st_ranks": sbc.c2st_ranks.tolist(),
            "c2st_dap": sbc.c2st_dap,
            "uniformity_stat": sbc.uniformity_stat,
            "n_sbc": n_sbc,
        },
        "tarp": {"atc": tarp.atc, "ks_pval": tarp.ks_pval, "n_tarp": n_tarp},
        "coverage": {
            "raw_mean_abs_dev": raw_dev,
            "recal_mean_abs_dev": recal_dev,
            "nominal_levels": nlv.tolist(),
            # headline coverage (mean over parameters) at the standard nominal
            # levels, raw NPE vs the two recalibrations. The IS numbers come
            # from the separate IS-coverage test set (cov_raw_isc is raw NPE on
            # the SAME observations, so raw_is and raw_conformal differ only by
            # test-set sampling noise -- both report raw-NPE under-coverage).
            "at_levels": {
                f"{int(round(t*100))}": {
                    "nominal_used": _cov_at(nlv, cov_raw, t)[2],
                    "raw": _cov_at(nlv, cov_raw, t)[0],
                    "conformal": _cov_at(nlv, cov_recal, t)[0],
                    "raw_is_testset": _cov_at(nominal, cov_raw_isc, t)[0],
                    "is_refined": _cov_at(nominal, is_cov["cov_all"], t)[0],
                    "is_refined_okess": _cov_at(nominal, is_cov["cov_okess"], t)[0],
                }
                for t in (0.5, 0.68, 0.9)
            },
        },
        "is_refinement": {
            "median_ess": float(np.median(ess_arr)),
            "median_ess_frac": float(np.median(essfrac_arr)),
            "n_low_ess": n_low,
            "n_cases": n_is_cases,
            "low_ess_frac_threshold": low_ess_frac,
            "mean_median_shift": float(np.mean(is_med_shift)),
            # ESS distribution + low-ESS fraction on the IS-coverage test set
            # (larger N than the ESS-report cases above):
            "cov_testset_median_ess": float(np.median(is_cov["ess"])),
            "cov_testset_median_ess_frac": float(np.median(is_cov["ess_frac"])),
            "cov_testset_n_low_ess": is_cov["n_low_ess"],
            "cov_testset_n_cases": is_cov["n_cases"],
            "cov_testset_low_ess_fraction": float(
                is_cov["n_low_ess"] / is_cov["n_cases"]),
            "raw_mean_abs_dev": is_raw_dev,
            "is_mean_abs_dev": is_all_dev,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"  -> {out_dir}")
    return out_summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-3 calibration suite")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="evaluate exactly this checkpoint dir (level from --level "
                         "or the dir's trailing name)")
    ap.add_argument("--level", default=None, help="restrict to this level name")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    cfg = _tn.load_config(args.config)

    results = []
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        level = args.level or ckpt.name.split("_")[-1]
        r = run_one_checkpoint(ckpt, level, cfg, force=args.force, device=args.device)
        if r:
            results.append(r)
    else:
        train_run = cfg["train_run"]
        for level in cfg["levels"]:
            if args.level and level != args.level:
                continue
            ckpt = _checkpoint_for_level(train_run, level)
            r = run_one_checkpoint(ckpt, level, cfg, force=args.force, device=args.device)
            if r:
                results.append(r)

    if not results:
        print("\nNo checkpoints evaluated. Train flows first "
              "(scripts/run_train_npe.py) or pass --checkpoint.")
        return 0

    # ---- cross-level headline table ----
    print("\n=== CALIBRATION SUMMARY ===")
    print("| level | ~counts | SBC KS p (min) | TARP ATC | cov dev raw->recal | low-ESS |")
    print("|---|---|---|---|---|---|")
    for r in results:
        ks_min = min(r["sbc"]["ks_pvals"])
        print(f"| {r['level']} | {r['median_total_counts']:.0f} | "
              f"{ks_min:.3f} | {r['tarp']['atc']:+.4f} | "
              f"{r['coverage']['raw_mean_abs_dev']:.3f}->{r['coverage']['recal_mean_abs_dev']:.3f} | "
              f"{r['is_refinement']['cov_testset_n_low_ess']}/"
              f"{r['is_refinement']['cov_testset_n_cases']} |")

    # ---- headline coverage-at-nominal table (raw vs IS vs conformal) ----
    print("\n=== COVERAGE @ NOMINAL (mean over params; for the money plot) ===")
    print("| level | ~counts | nom | raw NPE | IS-refined | conformal | low-ESS frac |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        al = r["coverage"]["at_levels"]
        lef = r["is_refinement"]["cov_testset_low_ess_fraction"]
        for t in ("50", "68", "90"):
            d = al[t]
            print(f"| {r['level']} | {r['median_total_counts']:.0f} | {t} | "
                  f"{d['raw']:.2f} | {d['is_refined']:.2f} | {d['conformal']:.2f} | "
                  f"{lef:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
