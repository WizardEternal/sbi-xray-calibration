"""Phase-4 misspecification-detection ROC benchmark.

Usage (repo venv; cap compute with OMP_NUM_THREADS):
    .venv\\Scripts\\python.exe scripts\\run_detect_benchmark.py --config configs\\detect.yaml
    .venv\\Scripts\\python.exe scripts\\run_detect_benchmark.py --config configs\\detect.yaml --pilot
    .venv\\Scripts\\python.exe scripts\\run_detect_benchmark.py --config configs\\detect.yaml --level medium --family B1

Grid: 4 B-families x strength grid x 3 count levels x 3 detectors. For each
(family, strength, level) cell we draw n_misspec misspecified test spectra and
reuse a shared per-level pool of n_clean CLEAN Model-A spectra as the common
negative class, then score every spectrum with each detector and compute the
clean-vs-misspecified ROC AUC.

Detectors (see src/sbixcal/detect.py):
  D1  PPC               -- per-spectrum (chi2 + KS sub-scores)
  D2  embedding OOD     -- per-spectrum (kNN primary, Mahalanobis secondary)
  D3  marginal C2ST     -- per-CELL population two-sample test (CV accuracy +
                           out-of-fold per-spectrum probabilities for the ROC)

Outputs (outputs/detect/, gitignored):
  scores.jsonl   one row per (family,strength,level,detector,spectrum) score
  results.jsonl  one row per (family,strength,level,detector) cell  -> AUC etc.
  consequence.jsonl  per (B1 family, strength, level) NPE Gamma-bias vs clean truth

Crash-resumability: every cell's results.jsonl row is keyed
(family,strength,level,detector); on restart, cells already present are SKIPPED.
scores.jsonl is appended in lock-step. Safe to kill and rerun. Figures are
regenerated separately by scripts/analyze_detect.py.

This script READS checkpoints in outputs/models/ and WRITES only to
outputs/detect/. It never touches outputs/calibration/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from sbixcal import detect as D
from sbixcal import misspec as MS
from sbixcal import train_npe as _tn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _out_dir() -> Path:
    d = _repo_root() / "outputs" / "detect"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_for_level(train_run: str, level: str) -> Path:
    return _repo_root() / "outputs" / "models" / f"{train_run}_{level}"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# JSONL helpers (crash-resumable: key -> skip if present)
# --------------------------------------------------------------------------

def _cell_key(family, strength, level, detector) -> str:
    return f"{family}|{float(strength):g}|{level}|{detector}"


def _stable_cell_seed(seed: int, family: str, strength: float) -> int:
    """Deterministic per-(family,strength) misspec-draw seed (process-independent,
    so crash-resume reproduces the SAME misspecified population)."""
    h = hashlib.sha1(f"{family}|{float(strength):g}".encode()).hexdigest()
    return (int(seed) + int(h[:8], 16)) % (2**31 - 1)


def load_done_cells(results_path: Path) -> set[str]:
    """Set of (family,strength,level,detector) keys already in results.jsonl."""
    done = set()
    if not results_path.exists():
        return done
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(_cell_key(r["family"], r["strength"], r["level"], r["detector"]))
    return done


def append_jsonl(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------
# per-level state: posterior + ctx + clean pool + D2 reference (built once)
# --------------------------------------------------------------------------

class LevelState:
    """Everything reused across all cells at one count level: the cold-loaded
    posterior, the simulator context, the shared clean negative pool (counts +
    Gamma truth + embeddings), and the D2 clean reference cloud."""

    def __init__(self, ckpt_dir: Path, level: str, cfg: dict, n_clean: int,
                 device: str = "cpu"):
        self.level = level
        self.device = device
        self.posterior, self.info = _tn.load_posterior(ckpt_dir, device=device)
        self.ctx = D.context_from_checkpoint(ckpt_dir)
        self.param_names = list(self.ctx.param_names)
        self.gamma_idx = self.param_names.index("powerlaw_1_alpha")
        self.seed = int(cfg["seed"])

        # shared clean negative pool (drawn once per level, reused for every cell)
        self.theta_clean, self.x_clean = self.ctx.clean_sim(
            n_clean, seed=self.seed + 1
        )
        self.gamma_clean = self.theta_clean[:, self.gamma_idx]

        # D2 reference cloud (clean embeddings; fit once per level)
        d2 = cfg.get("d2", {})
        self.d2_ref = D.build_embedding_reference(
            self.posterior, self.ctx,
            n_ref=int(d2.get("n_ref", 500)), seed=self.seed,
            reg=float(d2.get("reg", 1e-3)), device=device,
        )

        # cache clean per-spectrum scores for D1/D2 (independent of the cell)
        d1k = int(cfg.get("d1", {}).get("k", 200))
        self._d1_clean = np.array([
            D.detect_d1_ppc(x, self.posterior, self.ctx, k=d1k,
                            seed=self.seed + 2, device=device)
            for x in self.x_clean
        ])
        knn_k = int(d2.get("knn_k", 5))
        self._d2_clean = np.array([
            D.detect_d2_embedding(x, self.posterior, self.ctx, self.d2_ref,
                                  knn_k=knn_k, device=device)
            for x in self.x_clean
        ])

    def d1_clean(self):
        return self._d1_clean

    def d2_clean(self):
        return self._d2_clean


# --------------------------------------------------------------------------
# Gamma-bias consequence (B1 silent-failure cost)
# --------------------------------------------------------------------------

def posterior_median_gamma(state: LevelState, x: np.ndarray,
                           n_samples: int = 200, seed: int = 0) -> np.ndarray:
    """NPE posterior-median Gamma for each spectrum in ``x`` (N, C).

    Returns ``(N,)``. Uses un-rejected sampling for speed/robustness on
    misspecified spectra (the median of the in-box marginal is what a user reads
    off the posterior); the median is then clipped to the prior box.
    """
    from sbixcal import priors as _priors
    lo, hi = _priors.prior_bounds(state.ctx.prior_cfg, state.param_names)
    out = np.empty(x.shape[0])
    for i, xi in enumerate(x):
        torch.manual_seed(seed + i)
        xt = torch.as_tensor(np.asarray(xi, dtype=np.float32), device=state.device)
        s = state.posterior.sample((n_samples,), x=xt, show_progress_bars=False,
                                   reject_outside_prior=False)
        g = s[:, state.gamma_idx].detach().cpu().numpy()
        out[i] = float(np.clip(np.median(g), lo[state.gamma_idx], hi[state.gamma_idx]))
    return out


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

def run_cell(state: LevelState, family: str, strength: float, fixed: dict,
             cfg: dict, n_misspec: int, detectors, scores_path: Path,
             results_path: Path, consequence_path: Path, device: str = "cpu"):
    """Run every requested detector for one (family, strength, level) cell and
    append per-spectrum scores + per-cell AUC rows. Skips detectors already done."""
    level = state.level
    seed = state.seed

    # families share the per-family/strength seed so misspec draws are reproducible.
    # Use a DETERMINISTIC hash (Python's built-in hash() is salted per process via
    # PYTHONHASHSEED, which would redraw a different misspec population on a
    # crash-resume and desync the per-spectrum scores already written for other
    # detectors of the same cell).
    cell_seed = _stable_cell_seed(seed, family, strength)
    x_mis, theta_mis, mis_names = MS.simulate_misspec_population(
        state.ctx.base_model, state.ctx.prior_cfg, state.ctx.obsconf,
        family, strength, n_misspec, seed=cell_seed, fixed=fixed,
    )

    d1k = int(cfg.get("d1", {}).get("k", 200))
    knn_k = int(cfg.get("d2", {}).get("knn_k", 5))
    d3cfg = cfg.get("d3", {})

    done = load_done_cells(results_path)

    for det in detectors:
        key = _cell_key(family, strength, level, det)
        if key in done:
            print(f"[skip] {key}")
            continue
        t0 = time.time()

        if det == "D1":
            clean = state.d1_clean()
            mis = np.array([
                D.detect_d1_ppc(x, state.posterior, state.ctx, k=d1k,
                                seed=seed + 3, device=device) for x in x_mis
            ])
        elif det == "D2":
            clean = state.d2_clean()
            mis = np.array([
                D.detect_d2_embedding(x, state.posterior, state.ctx, state.d2_ref,
                                      knn_k=knn_k, device=device) for x in x_mis
            ])
        elif det == "D3":
            res = D.detect_d3_c2st_cell(
                state.posterior, state.ctx, state.x_clean, x_mis,
                kind=d3cfg.get("kind", "logreg"),
                n_splits=int(d3cfg.get("n_splits", 5)), seed=seed + 4,
                device=device,
            )
            clean = res.clean_proba
            mis = res.mis_proba
        else:
            raise ValueError(f"unknown detector '{det}'")

        fpr, tpr, auc = D.roc_auc(clean, mis)
        dt = time.time() - t0

        # per-spectrum scores
        for j, sc in enumerate(np.asarray(clean)):
            append_jsonl(scores_path, {
                "family": family, "strength": float(strength), "level": level,
                "detector": det, "kind": "clean", "idx": j, "score": float(sc),
            })
        for j, sc in enumerate(np.asarray(mis)):
            append_jsonl(scores_path, {
                "family": family, "strength": float(strength), "level": level,
                "detector": det, "kind": "misspec", "idx": j, "score": float(sc),
            })

        row = {
            "family": family, "strength": float(strength), "level": level,
            "detector": det, "auc": float(auc),
            "n_clean": int(np.size(clean)), "n_misspec": int(np.size(mis)),
            "clean_mean": float(np.nanmean(clean)), "misspec_mean": float(np.nanmean(mis)),
            "wall_s": round(dt, 2),
        }
        if det == "D3":
            row["cv_accuracy"] = float(res.cv_accuracy)
        append_jsonl(results_path, row)
        print(f"[done] {key}: AUC={auc:.3f} "
              f"({row['clean_mean']:.3f}->{row['misspec_mean']:.3f}) [{dt:.1f}s]")

    # ---- B1 Gamma-bias consequence (silent-failure cost) -------------------
    if family == "B1" and "powerlaw_1_alpha" in mis_names:
        cons_key = f"B1|{float(strength):g}|{level}"
        already = False
        if consequence_path.exists():
            with open(consequence_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if f"{r['family']}|{float(r['strength']):g}|{r['level']}" == cons_key:
                        already = True
                        break
        if not already:
            gi = mis_names.index("powerlaw_1_alpha")
            gamma_truth = theta_mis[:, gi]
            gamma_hat = posterior_median_gamma(state, x_mis, n_samples=200,
                                               seed=seed + 5)
            bias = gamma_hat - gamma_truth
            append_jsonl(consequence_path, {
                "family": "B1", "strength": float(strength), "level": level,
                "n": int(x_mis.shape[0]),
                "gamma_truth_mean": float(np.mean(gamma_truth)),
                "gamma_hat_mean": float(np.mean(gamma_hat)),
                "dGamma_bias_mean": float(np.mean(bias)),
                "dGamma_bias_median": float(np.median(bias)),
                "dGamma_bias_std": float(np.std(bias)),
                "abs_bias_mean": float(np.mean(np.abs(bias))),
            })
            print(f"[cons] {cons_key}: <dGamma>={np.mean(bias):+.3f} "
                  f"(|.|={np.mean(np.abs(bias)):.3f})")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_benchmark(cfg: dict, pilot: bool = False, only_level: str | None = None,
                  only_family: str | None = None, device: str = "cpu"):
    out = _out_dir()
    suffix = "_pilot" if pilot else ""
    scores_path = out / f"scores{suffix}.jsonl"
    results_path = out / f"results{suffix}.jsonl"
    consequence_path = out / f"consequence{suffix}.jsonl"

    levels = cfg["levels"]
    if only_level:
        levels = [l for l in levels if l == only_level]
    detectors = cfg.get("detectors", ["D1", "D2", "D3"])
    train_run = cfg["train_run"]

    if pilot:
        n_misspec = int(cfg["pilot"]["n_misspec"])
        n_clean = int(cfg["pilot"]["n_clean"])
        s_index = int(cfg["pilot"].get("strength_index", -1))
    else:
        n_misspec = int(cfg["n_misspec"])
        n_clean = int(cfg["n_clean"])
        s_index = None

    fam_items = list(cfg["families"].items())
    if only_family:
        fam_items = [(f, c) for f, c in fam_items if f == only_family]

    t_all = time.time()
    for level in levels:
        ckpt = _checkpoint_for_level(train_run, level)
        if not (ckpt / "arch.json").exists():
            print(f"[skip-level] {level}: no checkpoint at {ckpt}")
            continue
        print(f"\n=== level {level} (ckpt {ckpt.name}) ===")
        t_lvl = time.time()
        state = LevelState(ckpt, level, cfg, n_clean=n_clean, device=device)
        print(f"  level state ready ({time.time()-t_lvl:.1f}s; "
              f"n_clean={n_clean}, clean counts median "
              f"{int(np.median(state.x_clean.sum(1)))})")

        for family, fam_cfg in fam_items:
            fixed = fam_cfg.get("fixed", {})
            grid = list(fam_cfg["strength_grid"])
            if pilot:
                grid = [grid[s_index]]
            for strength in grid:
                run_cell(state, family, float(strength), fixed, cfg,
                         n_misspec, detectors, scores_path, results_path,
                         consequence_path, device=device)

    print(f"\nALL DONE in {time.time()-t_all:.1f}s -> {results_path}")


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Phase-4 misspec-detection ROC benchmark")
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot", action="store_true",
                    help="small pilot: 1 strength/family, pilot sample sizes")
    ap.add_argument("--level", default=None, help="restrict to one count level")
    ap.add_argument("--family", default=None, help="restrict to one B-family")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_benchmark(cfg, pilot=args.pilot, only_level=args.level,
                  only_family=args.family, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
