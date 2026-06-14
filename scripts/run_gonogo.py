"""GO/NO-GO robustness pack runner -- the cheap decisive test of whether the
primary bright-level NPE over-confidence is robust or a single-run fluke.

Per variant (gonogo_seed101 / seed202 / seed303 / uncapped) this script, ALL
crash-resumable at every stage:

  1. generates the bright training dataset  (skip-if-exists; the uncapped variant
     REUSES the production bright dataset, same draws -- see its config);
  2. trains the bright NPE+NSF flow          (skip-if-checkpoint-exists);
  3. runs the GO/NO-GO calibration suite     (SBC N=500 + coverage at the three
     nominal levels {50,68,90}); artifacts -> outputs/gonogo/<variant>/;
  4. appends a one-line summary               (variant, cov@50/68/90, dev, SBC KS
     p min) to outputs/gonogo/summary.jsonl.

With --detect-spot it ALSO recomputes a 2-cell detector spot-check against the
seed101 flow (B1 bright strongest line ~0.97 expected; B4 bright 3% ~0.50
expected; N=100/class) and appends those AUCs to the summary.

Usage (repo venv, OMP_NUM_THREADS=4):
    .venv\\Scripts\\python.exe scripts\\run_gonogo.py --config configs\\gonogo_seed101.yaml
    .venv\\Scripts\\python.exe scripts\\run_gonogo.py --config configs\\gonogo_seed101.yaml --detect-spot
    .venv\\Scripts\\python.exe scripts\\run_gonogo.py --config configs\\gonogo_uncapped.yaml

Writes ONLY to data/sim/ (training datasets), outputs/models/gonogo_*/
(checkpoints) and outputs/gonogo/ (calibration artifacts + summary). It never
touches outputs/calibration/, outputs/detect/ or outputs/ns_bench/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sbixcal import calibrate as C
from sbixcal import detect as D
from sbixcal import misspec as MS
from sbixcal import simulate as _sim
from sbixcal import train_npe as _tn


# ==========================================================================
# paths
# ==========================================================================

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sim_dir() -> Path:
    d = _repo_root() / "data" / "sim"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _models_dir() -> Path:
    return _repo_root() / "outputs" / "models"


def _gonogo_dir() -> Path:
    d = _repo_root() / "outputs" / "gonogo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _summary_path() -> Path:
    return _gonogo_dir() / "summary.jsonl"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ==========================================================================
# stage 1: bright training dataset (skip-if-exists)
# ==========================================================================

def ensure_dataset(cfg: dict) -> tuple[str, Path]:
    """Ensure the bright training dataset exists; return (dataset_name, path).

    For a FULL-RESEED variant we generate a fresh bright dataset under the
    variant's own seed (new simulation draws). For the UNCAPPED variant we REUSE
    the production bright dataset (same draws as production -- the mechanism test
    changes only the epoch budget, not the data)."""
    variant = cfg["variant"]
    base_model = cfg["base_model"]
    response = cfg.get("response")
    bright = cfg["bright"]
    n = int(bright["n_train"])
    exposure_s = float(bright["exposure_s"])

    if cfg.get("reuse_prod_bright_dataset"):
        name = "modelA_prod_train_bright"
        path = _sim_dir() / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{variant} reuses the production bright dataset {path}, but it is "
                "missing. Generate it first: python -m sbixcal.simulate "
                "--config configs/sim_modelA_prod_train.yaml --level bright"
            )
        print(f"[stage1] {variant}: reusing production bright dataset {name} "
              "(same draws as production)")
        return name, path

    # full reseed -> a fresh bright dataset under this variant's seed.
    name = f"{variant}_bright"
    path = _sim_dir() / f"{name}.npz"
    if path.exists():
        print(f"[stage1] {variant}: dataset {name} exists (skip)")
        return name, path

    print(f"[stage1] {variant}: generating fresh bright dataset {name} "
          f"(n={n}, seed={cfg['seed']}) ...")
    level = {"name": "bright", "exposure_s": exposure_s, "n": n,
             "seed_offset": 0}
    sim_cfg = {
        "name": variant,
        "base_model": base_model,
        "response": response,
        "seed": int(cfg["seed"]),
        "n": n,
        "priors": cfg["priors"],
        "levels": [level],
    }
    out = _sim.generate_level(sim_cfg, level, force=False)
    return name, out


# ==========================================================================
# stage 2: train the bright flow (skip-if-checkpoint-exists)
# ==========================================================================

def ensure_checkpoint(cfg: dict, dataset_name: str) -> Path:
    """Train the bright flow into outputs/models/<variant>_bright/, skipping if a
    checkpoint already exists. Returns the checkpoint dir."""
    variant = cfg["variant"]
    run_name = f"{variant}_bright"
    out_dir = _models_dir() / run_name
    if (out_dir / "flow_state.pt").exists():
        print(f"[stage2] {variant}: checkpoint {run_name} exists (skip)")
        return out_dir

    # bright seed_offset: full-reseed variants use 0 (their global seed IS the
    # reseed); the uncapped variant uses the production bright offset so its torch
    # init matches the production bright flow exactly.
    seed_offset = int(cfg.get("bright_seed_offset", 0))
    n_train = int(cfg["bright"]["n_train"])

    theta, x, names, meta = _tn.load_dataset(dataset_name, max_n=n_train)
    param_order = list(names)
    prior = _tn.build_prior(cfg["priors"], param_order, device=cfg.get("device", "cpu"))

    # build a train-style cfg dict for train_one_flow / save_checkpoint reuse.
    train_cfg = {
        "name": variant,
        "base_model": cfg["base_model"],
        "priors": cfg["priors"],
        "embedding": cfg["embedding"],
        "flow": cfg["flow"],
        "train": cfg["train"],
    }

    print(f"[stage2] {variant}: training bright flow "
          f"(n={n_train}, seed={cfg['seed']}+{seed_offset}, "
          f"cap={cfg['train']['max_num_epochs']}, patience={cfg['train']['stop_after_epochs']}) ...")
    t0 = time.perf_counter()
    de, _, summary = _tn.train_one_flow(
        theta, x, prior, train_cfg, device=cfg.get("device", "cpu"),
        seed=int(cfg["seed"]) + seed_offset,
    )
    dt = time.perf_counter() - t0
    meta["train_wall_s"] = dt

    _tn.save_checkpoint(out_dir, de, summary, train_cfg, param_order,
                        x.shape[1], meta)
    vl = summary.get("validation_loss", [])
    tl = summary.get("training_loss", [])
    print(f"[stage2] {variant}: epochs={summary.get('epochs_trained')} "
          f"final_train={tl[-1] if tl else float('nan'):.3f} "
          f"final_val={vl[-1] if vl else float('nan'):.3f} "
          f"best_val={float(np.min(vl)) if vl else float('nan'):.3f} "
          f"wall={dt:.0f}s -> {out_dir}")
    return out_dir


# ==========================================================================
# stage 3: GO/NO-GO calibration suite (SBC N=500 + coverage @ {50,68,90})
# ==========================================================================

def _cov_at(nominal, cov, target):
    """Mean-over-params coverage at the nominal level nearest ``target``."""
    nominal = np.asarray(nominal)
    j = int(np.argmin(np.abs(nominal - target)))
    return float(np.mean(np.asarray(cov)[j])), float(nominal[j])


def run_calibration(cfg: dict, ckpt_dir: Path, force: bool = False) -> dict:
    """Run the GO/NO-GO calibration suite against the variant's checkpoint and
    write outputs/gonogo/<variant>/calibration.json (skip-if-exists). Returns the
    calibration result dict."""
    variant = cfg["variant"]
    device = cfg.get("device", "cpu")
    out_dir = _gonogo_dir() / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_path = out_dir / "calibration.json"
    if calib_path.exists() and not force:
        print(f"[stage3] {variant}: {calib_path.name} exists (skip)")
        with open(calib_path) as f:
            return json.load(f)

    ccfg = cfg.get("calibration", {})
    response = cfg.get("response")
    base_seed = int(cfg["seed"])

    post, info = _tn.load_posterior(ckpt_dir, device=device)
    with open(ckpt_dir / "arch.json") as f:
        arch = json.load(f)
    prior_cfg = arch["prior_cfg"]
    base_model = arch["base_model"]
    exposure_s = float(arch["exposure_s"])
    param_names = info["param_names"]
    prior = _tn.build_prior(prior_cfg, param_names, device=device)

    nominal = np.asarray(ccfg.get(
        "nominal_levels",
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95]))
    n_sbc = int(ccfg.get("n_sbc", 500))
    n_post = int(ccfg.get("n_posterior_samples", 1000))
    n_cal = int(ccfg.get("n_cal", 400))
    n_test = int(ccfg.get("n_test", 400))
    n_cps = int(ccfg.get("coverage_posterior_samples", n_post))

    print(f"[stage3] {variant}: calibrating {ckpt_dir.name} "
          f"(~{info['median_total_counts']:.0f} counts; SBC N={n_sbc}) ...")

    # ---- SBC (fresh sims from the SAME prior/simulator the flow trained on) ----
    theta_sbc, x_sbc, _, names = C.make_fresh_test_set(
        base_model, prior_cfg, exposure_s, n_sbc,
        seed=base_seed + 101, response_name=response)
    assert names == param_names, (names, param_names)
    sbc = C.run_sbc_check(post, prior, theta_sbc, x_sbc, param_names,
                          num_posterior_samples=n_post, seed=base_seed + 1)
    C.save_sbc_figure(sbc, out_dir / "sbc_ranks.png")
    np.savez(out_dir / "sbc.npz",
             ranks=sbc.ranks, ks_pvals=sbc.ks_pvals, c2st_ranks=sbc.c2st_ranks,
             c2st_dap=sbc.c2st_dap, param_names=np.array(param_names))
    ks_min = float(np.min(sbc.ks_pvals))
    print(f"  SBC: KS p-vals {np.round(sbc.ks_pvals, 4).tolist()} "
          f"(min {ks_min:.2e})  C2ST(ranks) {np.round(sbc.c2st_ranks, 3).tolist()}")

    # ---- coverage at nominal {50,68,90}: raw + conformal before/after ----
    th_cal, x_cal, _, _ = C.make_fresh_test_set(
        base_model, prior_cfg, exposure_s, n_cal,
        seed=base_seed + 303, response_name=response)
    th_test, x_test, _, _ = C.make_fresh_test_set(
        base_model, prior_cfg, exposure_s, n_test,
        seed=base_seed + 404, response_name=response)
    s_cal = C.sample_posterior_batch(post, x_cal, n_cps, seed=base_seed + 31, device=device)
    s_test = C.sample_posterior_batch(post, x_test, n_cps, seed=base_seed + 41, device=device)

    nlv, cov_raw, cov_recal, _ = C.coverage_before_after(
        s_cal, th_cal, s_test, th_test, param_names, nominal_levels=nominal)
    C.save_coverage_before_after(
        nlv, cov_raw, cov_recal, param_names,
        out_dir / "coverage_before_after.npz",
        out_dir / "coverage_before_after.png",
        title_suffix=f"({variant}, ~{info['median_total_counts']:.0f} counts)")

    raw_dev = float(np.mean(np.abs(cov_raw - nlv[:, None])))
    recal_dev = float(np.mean(np.abs(cov_recal - nlv[:, None])))

    cov_at = {}
    for t in (0.5, 0.68, 0.9):
        raw_t, used = _cov_at(nlv, cov_raw, t)
        recal_t, _ = _cov_at(nlv, cov_recal, t)
        cov_at[str(int(round(t * 100)))] = {
            "nominal_used": used, "raw": raw_t, "conformal": recal_t}
    print(f"  Coverage @50/68/90 (raw): "
          f"{cov_at['50']['raw']:.3f}/{cov_at['68']['raw']:.3f}/{cov_at['90']['raw']:.3f}  "
          f"dev raw={raw_dev:.3f} -> conformal={recal_dev:.3f}")

    # epoch info from the checkpoint summary (for the cap mechanism read). sbi
    # stores epochs_trained / best_validation_loss as 1-element lists; coerce to
    # plain scalars so the summary line and verdict table read cleanly.
    with open(ckpt_dir / "summary.json") as f:
        ck_summary = json.load(f)

    def _scalar(v):
        if isinstance(v, (list, tuple)):
            return v[-1] if v else None
        return v

    result = {
        "variant": variant,
        "checkpoint": str(ckpt_dir),
        "median_total_counts": float(info["median_total_counts"]),
        "param_names": param_names,
        "epochs_trained": _scalar(ck_summary.get("epochs_trained")),
        "max_num_epochs": int(cfg["train"]["max_num_epochs"]),
        "final_training_loss": ck_summary.get("final_training_loss"),
        "final_validation_loss": ck_summary.get("final_validation_loss"),
        "best_validation_loss": _scalar(ck_summary.get("best_validation_loss")),
        "sbc": {
            "ks_pvals": sbc.ks_pvals.tolist(),
            "ks_p_min": ks_min,
            "c2st_ranks": sbc.c2st_ranks.tolist(),
            "n_sbc": n_sbc,
        },
        "coverage": {
            "raw_mean_abs_dev": raw_dev,
            "conformal_mean_abs_dev": recal_dev,
            "at_levels": cov_at,
        },
    }
    with open(calib_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> {out_dir}")
    return result


# ==========================================================================
# stage 4: append the one-line summary
# ==========================================================================

def _summary_has(variant: str, kind: str) -> bool:
    """True if a summary.jsonl row with this (variant, kind) already exists."""
    p = _summary_path()
    if not p.exists():
        return False
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("variant") == variant and r.get("kind") == kind:
                return True
    return False


def append_summary_line(cal: dict):
    """Append the one-line calibration summary (variant, cov@50/68/90, dev, SBC KS
    p min) to outputs/gonogo/summary.jsonl. Skip-if-already-present."""
    variant = cal["variant"]
    if _summary_has(variant, "calibration"):
        print(f"[stage4] {variant}: calibration summary line exists (skip)")
        return
    cov = cal["coverage"]["at_levels"]
    row = {
        "variant": variant,
        "kind": "calibration",
        "median_total_counts": cal["median_total_counts"],
        "epochs_trained": cal["epochs_trained"],
        "max_num_epochs": cal["max_num_epochs"],
        "cov50": cov["50"]["raw"],
        "cov68": cov["68"]["raw"],
        "cov90": cov["90"]["raw"],
        "cov_dev_raw": cal["coverage"]["raw_mean_abs_dev"],
        "cov_dev_conformal": cal["coverage"]["conformal_mean_abs_dev"],
        "sbc_ks_p_min": cal["sbc"]["ks_p_min"],
    }
    with open(_summary_path(), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[stage4] {variant}: appended calibration summary line")


# ==========================================================================
# detector spot-check (--detect-spot): 2 cells with the seed101 flow
# ==========================================================================

# the two spot cells: (family, strength, expected AUC) for the seed101 bright flow.
SPOT_CELLS = [
    ("B1", 3.0e-4, {"line_energy_kev": 6.4, "line_sigma_kev": 0.05}, 0.97),
    ("B4", 3.0,    {},                                              0.50),
]


def run_detect_spot(spot_ckpt: Path, n_per_class: int = 100,
                    device: str = "cpu", force: bool = False):
    """Recompute D1 (PPC) AUC for the two spot cells with the seed101 flow:
    B1 bright strongest line (norm 3e-4, expect ~0.97) and B4 bright 3% gain
    (expect ~0.50). N=n_per_class per class. Appends each AUC to summary.jsonl."""
    if _summary_has("gonogo_seed101", "detect_spot") and not force:
        print("[spot] detector spot-check rows exist (skip)")
        return
    if not (spot_ckpt / "arch.json").exists():
        print(f"[spot] no seed101 checkpoint at {spot_ckpt}; train it first (skip)")
        return

    print(f"[spot] detector spot-check with {spot_ckpt.name} "
          f"(N={n_per_class}/class, D1 PPC) ...")
    post, info = _tn.load_posterior(spot_ckpt, device=device)
    ctx = D.context_from_checkpoint(spot_ckpt)
    seed = 101

    # shared clean negative pool (N per class), the ROC negative class.
    _, x_clean = ctx.clean_sim(n_per_class, seed=seed + 1)
    d1_clean = np.array([
        D.detect_d1_ppc(x, post, ctx, k=200, seed=seed + 2, device=device)
        for x in x_clean
    ])

    rows = []
    for family, strength, fixed, expect in SPOT_CELLS:
        # reuse the benchmark's deterministic per-(family,strength) misspec seed so
        # the spot population matches the full-grid convention.
        import hashlib
        h = hashlib.sha1(f"{family}|{float(strength):g}".encode()).hexdigest()
        cell_seed = (int(seed) + int(h[:8], 16)) % (2**31 - 1)
        x_mis, _, _ = MS.simulate_misspec_population(
            ctx.base_model, ctx.prior_cfg, ctx.obsconf,
            family, strength, n_per_class, seed=cell_seed, fixed=fixed)
        d1_mis = np.array([
            D.detect_d1_ppc(x, post, ctx, k=200, seed=seed + 3, device=device)
            for x in x_mis
        ])
        _, _, auc = D.roc_auc(d1_clean, d1_mis)
        print(f"  {family} bright (strength {strength:g}): D1 AUC={auc:.3f} "
              f"(expected ~{expect:.2f})")
        rows.append({
            "variant": "gonogo_seed101",
            "kind": "detect_spot",
            "family": family,
            "strength": float(strength),
            "detector": "D1",
            "auc": float(auc),
            "expected_auc": expect,
            "n_per_class": n_per_class,
        })

    with open(_summary_path(), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[spot] appended {len(rows)} detector spot-check rows")


# ==========================================================================
# driver
# ==========================================================================

def run_variant(cfg: dict, detect_spot: bool = False, force: bool = False):
    variant = cfg["variant"]
    print(f"\n========== {variant} ==========")
    dataset_name, _ = ensure_dataset(cfg)
    ckpt_dir = ensure_checkpoint(cfg, dataset_name)
    cal = run_calibration(cfg, ckpt_dir, force=force)
    append_summary_line(cal)
    if detect_spot:
        # the spot-check always uses the seed101 flow, regardless of which variant
        # config was passed.
        spot_ckpt = _models_dir() / "gonogo_seed101_bright"
        run_detect_spot(spot_ckpt, n_per_class=100,
                        device=cfg.get("device", "cpu"), force=force)
    return cal


def main(argv=None):
    ap = argparse.ArgumentParser(description="GO/NO-GO robustness pack runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--detect-spot", action="store_true",
                    help="also run the 2-cell detector spot-check with the seed101 flow")
    ap.add_argument("--force", action="store_true",
                    help="recompute calibration / spot-check even if present")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_variant(cfg, detect_spot=args.detect_spot, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
