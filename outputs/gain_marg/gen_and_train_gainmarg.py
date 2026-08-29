"""Gain marginalization in amortized NPE.

Adds a detector-gain nuisance parameter g (6th param) to the production
tbabs*(powerlaw+blackbody) model, draws it per-simulation, applies it to the
EPIC-pn response via responses.gain_shift_obsconf, and retrains the medium-count
NPE flow on the gain-augmented simulations. At inference the 5 physical params
are read off by marginalizing g out; amortized SBI does the nuisance integral
for free per spectrum (Alsing & Wandelt 2019 nuisance-hardened SBI).

Gain prior: uniform g in [0.95, 1.05] (+/-5%). Justification (input audit):
  - XMM EPIC-pn absolute energy-scale accuracy ~12.5 eV, i.e. ~0.6-1.2% at the
    ~1-2 keV band where most counts sit;
  - read2014 (Read, Guainazzi, Sembay) EPIC pn-vs-MOS energy-scale discrepancy
    0-8%, energy dependent;
  - the injected worst-case shift used throughout is 3%.
  +/-5% uniform is the honest "gain known no better than 5%" envelope: it covers
  the ~1% nominal systematic and brackets the 3% injected test with margin, while
  staying inside the read2014 0-8% cross-instrument spread. Wider prior => wider
  marginal posterior (the honest cost), quantified in the report.

Matches the committed train_npe_prod medium config EXACTLY except for the extra
gain dimension (same 50k train size, same NSF+CNN arch, same optimizer, same
seed, same 353.4 s / ~986-count medium exposure, CPU to match the baseline).

Run (repo venv):
    .venv\\Scripts\\python.exe outputs\\gain_marg\\gen_and_train_gainmarg.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from sbixcal import responses as R
from sbixcal import models as M
from sbixcal import priors as P
from sbixcal import train_npe as tn
from jaxspec.data.util import fakeit_for_multiple_parameters

# ----------------------------------------------------------------------------
# fixed config, mirroring configs/train_npe_prod.yaml (medium) + gain nuisance
# ----------------------------------------------------------------------------
SEED = 20260611
SEED_OFFSET = 2                 # medium level's offset in the prod config
EXPOSURE_S = 353.4              # prod medium exposure (~986 counts)
N_TRAIN = 50000
BASE_MODEL = "tbabs_powerlaw_bb"
DEVICE = "cpu"                  # match the committed baseline flow
GAIN_LO, GAIN_HI = 0.95, 1.05  # +/-5% uniform gain prior
N_GAIN_BINS = 200              # discretization of g for batched folding (0.05% res)

PHYS_PRIORS = {
    "tbabs_1_nh":         {"dist": "uniform",    "low": 0.15,  "high": 0.35},
    "powerlaw_1_alpha":   {"dist": "uniform",    "low": 1.0,   "high": 3.0},
    "powerlaw_1_norm":    {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":  {"dist": "uniform",    "low": 0.3,   "high": 3.0},
    "blackbodyrad_1_norm": {"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}
GAIN_PRIOR = {"gain_g": {"dist": "uniform", "low": GAIN_LO, "high": GAIN_HI}}
PHYS_ORDER = M.MODEL_PARAMS[BASE_MODEL]           # 5 physical
PARAM_ORDER = PHYS_ORDER + ["gain_g"]             # 6 with gain last

EMBED = {"embed_dim": 20, "conv_channels": [16, 32], "kernel_size": 5, "mlp_hidden": 64}
FLOW = {"hidden_features": 50, "num_transforms": 5, "num_bins": 10}
TRAIN = {"batch_size": 200, "learning_rate": 5.0e-4, "validation_fraction": 0.1,
         "stop_after_epochs": 20, "max_num_epochs": 150, "show_progress": False}

ROOT = Path(__file__).resolve().parents[2]
SIM_PATH = ROOT / "data" / "sim" / "modelA_prod_gainmarg_medium.npz"
MODEL_DIR = ROOT / "outputs" / "gain_marg" / "model_medium"


# ----------------------------------------------------------------------------
# gain-augmented simulation (g drawn per-sim, applied via gain_shift_obsconf)
# ----------------------------------------------------------------------------
def simulate_gain_augmented(n, exposure_s, seed):
    """Draw n physical params + a per-sim gain g, fold each through its own
    gain-shifted EPIC-pn response. g is drawn on a fine discrete grid so all sims
    sharing a bin fold through one obsconf (fakeit takes a single obsconf); the
    stored theta uses the exact bin-center g, so theta and the folded spectrum are
    perfectly consistent. Returns theta6 (n,6) and x (n,102) Poisson counts."""
    base = R.load_base_obsconf("NGC7793_ULX4_PN")
    oc = R.scale_exposure(base, exposure_s)
    model = M.build_model(BASE_MODEL)
    rng = np.random.default_rng(seed)

    phys = P.sample_prior(PHYS_PRIORS, PHYS_ORDER, n, rng)          # dict of (n,)
    centers = np.linspace(GAIN_LO, GAIN_HI, N_GAIN_BINS)
    g_idx = rng.integers(0, N_GAIN_BINS, size=n)
    g_val = centers[g_idx]

    n_ch = int(np.asarray(oc.out_energies).shape[-1]) if hasattr(oc, "out_energies") else 102
    x = np.zeros((n, 102), dtype=np.float64)
    for b in np.unique(g_idx):
        mask = g_idx == b
        idx = np.where(mask)[0]
        ocg = R.gain_shift_obsconf(oc, float(centers[b]))
        sub = {p: phys[p][idx] for p in PHYS_ORDER}
        xb = fakeit_for_multiple_parameters(ocg, model, sub, rng_key=int(b), apply_stat=True)
        x[idx] = np.asarray(xb, dtype=np.float64)

    theta = np.stack([phys[p] for p in PHYS_ORDER] + [g_val], axis=1)
    return theta.astype(np.float32), x.astype(np.float32)


def get_training_data():
    if SIM_PATH.exists():
        d = np.load(SIM_PATH)
        print(f"[skip-sim] {SIM_PATH.name} exists, theta{d['theta'].shape}")
        return d["theta"].astype(np.float32), d["x"].astype(np.float32)
    print(f"[sim] generating {N_TRAIN} gain-augmented spectra (medium, {EXPOSURE_S}s)...")
    t0 = time.perf_counter()
    theta, x = simulate_gain_augmented(N_TRAIN, EXPOSURE_S, SEED + SEED_OFFSET)
    dt = time.perf_counter() - t0
    SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SIM_PATH, theta=theta, x=x,
                        param_names=np.array(PARAM_ORDER),
                        exposure_s=EXPOSURE_S, seed=SEED + SEED_OFFSET,
                        median_total_counts=float(np.median(x.sum(1))))
    print(f"[sim done] {dt:.1f}s theta{theta.shape} x{x.shape} "
          f"median_counts={np.median(x.sum(1)):.0f} -> {SIM_PATH.name}")
    return theta, x


# ----------------------------------------------------------------------------
# train the 6-param flow (arch/optimizer identical to prod medium)
# ----------------------------------------------------------------------------
def main():
    theta_np, x_np = get_training_data()
    theta = torch.from_numpy(theta_np)
    x = torch.from_numpy(x_np)

    full_priors = {**PHYS_PRIORS, **GAIN_PRIOR}
    prior = tn.build_prior(full_priors, PARAM_ORDER, device=DEVICE)

    cfg = {"flow": FLOW, "embedding": EMBED, "train": TRAIN,
           "priors": full_priors, "base_model": BASE_MODEL}

    print(f"[train] 6-param gain-marginalized flow on {theta.shape[0]} sims, "
          f"device={DEVICE}...")
    t0 = time.perf_counter()
    de, _, summary = tn.train_one_flow(theta, x, prior, cfg, device=DEVICE,
                                       seed=SEED + SEED_OFFSET)
    dt = time.perf_counter() - t0

    meta = {"exposure_s": EXPOSURE_S, "median_total_counts": float(np.median(x_np.sum(1))),
            "n": int(theta.shape[0]), "train_wall_s": dt}
    tn.save_checkpoint(MODEL_DIR, de, summary, cfg, PARAM_ORDER, x.shape[1], meta)
    vl = summary.get("validation_loss", [])
    ep = summary.get("epochs_trained")
    print(f"[train done] epochs={ep} best_val={float(np.min(vl)) if vl else float('nan'):.4f} "
          f"wall={dt:.1f}s ({dt/60:.1f} min) -> {MODEL_DIR}")
    # persist timing prominently
    (MODEL_DIR / "timing.json").write_text(json.dumps(
        {"train_wall_s": dt, "train_wall_min": dt / 60.0, "epochs": ep,
         "n_train": int(theta.shape[0]), "device": DEVICE,
         "best_validation_loss": float(np.min(vl)) if vl else None}, indent=2))
    print(f"[timing] {dt/60:.2f} min written to {MODEL_DIR/'timing.json'}")


if __name__ == "__main__":
    main()
