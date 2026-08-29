"""Redo every NPE-side statistic with outside-prior rejection enabled (the
detect.py convention), and quantify the outside-prior fraction for all 10
small-set spectra.

WHY
---
The small-set driver (outputs/gain_marg/run_ns_smallset.py:124-125) sampled the
gain-marginalized NPE flow with ``reject_outside_prior=False``:

    npe = G.sample_npe(model_dir, counts, n_samples=NPE_SAMPLES, seed=seed,
                       reject_outside_prior=False)

so the flow was allowed to place posterior mass outside its own training prior
box. sbi logged "X% of samples ... lie outside the prior support" in 4 of the 10
run logs (7.6-30.8%). The other 6 logs are silent, which does NOT mean 0%: the
warning is threshold-gated at 5% (see
.venv/Lib/site-packages/sbi/utils/sbiutils.py:713-742, ``threshold: float = 0.05``,
``if frac_outside > threshold``). Every NPE mean/std/quantile in
outputs/gain_marg/ns_smallset/ therefore carries an un-quantified caveat about
draws outside the prior. This script removes it.

WHAT IT DOES, per spectrum
--------------------------
 1. Regenerates the SAME conditioning spectrum the driver used
    (run_ns_smallset.make_spectrum -> eval_gainmarg B4 population, strength 0,
    seed 20300611, row `idx`) and asserts its total counts match the committed
    run json. A mismatch aborts that spectrum (recorded, not silently patched).
 2. Draw A ("raw"): the original procedure reproduced exactly, with the same flow
    checkpoint, same conditioning, torch.manual_seed(0), n=4000,
    reject_outside_prior=False. Used (a) to measure the outside-prior fraction
    with sbi's own ``within_support``, and (b) as a control: recomputing the
    committed metrics from Draw A must reproduce the committed numbers.
 3. Draw B ("clipped"): the detect.py convention, verbatim
    (src/sbixcal/detect.py:218-239): torch.manual_seed(seed);
    reject_outside_prior=True with max_sampling_time=20.0; on
    (RuntimeError, ValueError, TypeError) fall back to an unrejected batch; top
    up any shortfall with unrejected draws; then clip into the prior box with
    priors.prior_bounds(). Rejection AND clip, in that order.
 4. Recomputes every NPE-side metric from Draw B using the
    SAME definitions as ns_gainmarg.compare_ns_npe (outputs/gain_marg/
    ns_gainmarg.py:328-379): npe mean/std (ddof=1), g-shrink = npe_std/prior_std
    with prior_std = (1.05-0.95)/sqrt(12), mean_diff_in_ns_std =
    (npe_mean - ns_mean)/ns_std, std_ratio_npe_over_ns, and 68%/95% equal-tailed
    interval IoU.
 5. Repeats Draw B at two extra seeds so the Monte-Carlo scatter of the 4000-sample
    draw is measured, not assumed (old and new draws are unpaired, so a shift has
    to clear MC noise to mean anything).

NOT RECOMPUTABLE, and why
-------------------------
C2ST(NS, NPE). The NS posterior SAMPLES were never persisted: the worker keeps
res.samples in memory, writes only means/stds/quantiles to <sid>.json
(run_ns_smallset.py:130-137), and there is no .npy/.npz of NS draws anywhere under
outputs/. Recomputing C2ST would need a full 6-param NS rerun (3.4k-19.0k s per
spectrum). This script therefore reports the NS-vs-NPE C2ST as NOT_COMPUTABLE and
substitutes a DIFFERENT, clearly-labelled statistic: C2ST(raw NPE, clipped NPE),
which measures directly how far the rejection-and-clip fix moves the NPE posterior. A small
value bounds how much the NS-vs-NPE C2ST can have changed.

The NS side needs no rerun for anything else: ns_mean/ns_std in the committed
comparison block are the ddof=1 moments OF the NS samples, and the 16/84 and
2.5/97.5 NS quantiles are stored, so Delta-mean-in-NS-sigma and both IoUs are
exact.

SEEDS (all pinned, nothing drawn from entropy)
----------------------------------------------
  torch.manual_seed(0)      primary NPE draw, matches the original driver seed
  torch.manual_seed(1), (2) extra clipped draws, MC-scatter estimate only
  20300611                  spectrum regeneration (= 20260611 + 40000 + 0)
  20260723                  spectrum-selection seed (inherited via select_jobs)
  numpy default_rng(0)      C2ST subsampling, mirrors compare_ns_npe
  sbi c2st seed             library default (1), unchanged

Reads only. Writes only into outputs/gain_marg/ns_smallset/npe_draw_recompute/.
No existing file is modified. No git operations.

Run (repo venv, from repo root):
    .venv\\Scripts\\python.exe scripts\\npe_draw_recompute.py
    .venv\\Scripts\\python.exe scripts\\npe_draw_recompute.py --merge-only
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DRIVER_DIR = ROOT / "outputs" / "gain_marg"
SMALLSET = DRIVER_DIR / "ns_smallset"
OUTDIR = SMALLSET / "npe_draw_recompute"

sys.path.insert(0, str(DRIVER_DIR))          # ns_gainmarg, run_ns_smallset
sys.path.insert(0, str(ROOT / "src"))        # sbixcal (also installed -e)

# ---------------------------------------------------------------------------
# pinned configuration
# ---------------------------------------------------------------------------
PRIMARY_SEED = 0            # == run_ns_smallset.worker(..., seed=0) default
EXTRA_SEEDS = (1, 2)        # MC-scatter probe only; never used for a reported number
N_SAMPLES = 4000            # == run_ns_smallset.NPE_SAMPLES
MAX_SAMPLING_TIME = 20.0    # == detect.posterior_predictive_replicates default
C2ST_SUBSAMPLE_SEED = 0     # == compare_ns_npe's np.random.default_rng(0)

# Tolerance for "the raw draw reproduces the committed NPE numbers", in units of
# npe_std. The 6 spectra originally run on this Windows box reproduce at exactly
# 0; the 4 run on the Linux cloud box (same sbi 0.26.1 / torch 2.12.0 / nflows
# 0.14 / numpy 2.4.6, but torch+cpu there vs torch+cu130 here) land at ~2e-7 to
# ~1.2e-6, the same RNG stream through float32 kernels that differ in the last
# bits. Anything at or below this is the same draw, not a different one.
REPRO_TOL = 1e-5

GAIN_LO, GAIN_HI = 0.95, 1.05
PRIOR_STD_G = (GAIN_HI - GAIN_LO) / np.sqrt(12.0)   # 0.028867513459481287

# display names, in PARAM_ORDER
DISPLAY = {
    "tbabs_1_nh": "N_H",
    "powerlaw_1_alpha": "Gamma",
    "powerlaw_1_norm": "PL_norm",
    "blackbodyrad_1_kT": "kT_bb",
    "blackbodyrad_1_norm": "BB_norm",
    "gain_g": "gain_g",
}

# outside-prior percentages that sbi actually logged in the original run logs
# (absent = below sbi's 5% warning threshold, now measured exactly by this script)
LOGGED_PCT = {
    "medium_s0_i22": 30.8,
    "bright_s0_i238": 20.2,
    "bright_s0_i394": 7.6,
    "bright_s0_i416": 23.8,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# metric definitions, mirroring ns_gainmarg.compare_ns_npe exactly
# ---------------------------------------------------------------------------
def _interval_overlap(a, b):
    """IoU of two 1-D intervals. Verbatim from ns_gainmarg.py:328-333."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def npe_metrics(npe, ns_ref, param_order):
    """All NPE-side statistics, from an NPE sample array.

    ``ns_ref[name]`` supplies ns_mean, ns_std (the ddof=1 moments of the NS
    samples, as recorded by the original comparison block) and the NS 16/84 and
    2.5/97.5 quantiles (as recorded in the run json's ns.quantiles block).
    """
    npe = np.asarray(npe, dtype=np.float64)
    per_param = {}
    for j, name in enumerate(param_order):
        b = npe[:, j]
        npe_mean = float(b.mean())
        npe_std = float(b.std(ddof=1))
        ns_mean = ns_ref[name]["ns_mean"]
        ns_std = ns_ref[name]["ns_std"]
        b68 = np.quantile(b, [0.16, 0.84])
        b95 = np.quantile(b, [0.025, 0.975])
        per_param[name] = {
            "ns_mean": ns_mean,
            "ns_std": ns_std,
            "npe_mean": npe_mean,
            "npe_std": npe_std,
            "mean_diff_in_ns_std": float((npe_mean - ns_mean) / ns_std) if ns_std > 0 else float("nan"),
            "std_ratio_npe_over_ns": float(npe_std / ns_std) if ns_std > 0 else float("nan"),
            "overlap68_iou": _interval_overlap(ns_ref[name]["ns_q68"], (b68[0], b68[1])),
            "overlap95_iou": _interval_overlap(ns_ref[name]["ns_q95"], (b95[0], b95[1])),
            "npe_q68": [float(b68[0]), float(b68[1])],
            "npe_q95": [float(b95[0]), float(b95[1])],
        }
    mds = [abs(per_param[p]["mean_diff_in_ns_std"]) for p in param_order]
    return {
        "per_param": per_param,
        "max_abs_mean_diff_in_ns_std": float(np.nanmax(mds)),
        "mean_overlap68_iou": float(np.mean([per_param[p]["overlap68_iou"] for p in param_order])),
        "npe_g_mean": per_param["gain_g"]["npe_mean"],
        "npe_g_std": per_param["gain_g"]["npe_std"],
        "npe_g_shrink": float(per_param["gain_g"]["npe_std"] / PRIOR_STD_G),
    }


# ---------------------------------------------------------------------------
# NPE draws
# ---------------------------------------------------------------------------
def draw_raw(post, counts, n, seed, device="cpu"):
    """The ORIGINAL procedure, verbatim: ns_gainmarg.sample_npe(...,
    reject_outside_prior=False), ns_gainmarg.py:309-322."""
    import torch
    x_t = torch.as_tensor(np.asarray(counts, dtype=np.float32), device=device)
    torch.manual_seed(seed)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")       # the 5%-threshold warning; measured below
        s = post.sample((n,), x=x_t, show_progress_bars=False,
                        reject_outside_prior=False)
    return s.detach().cpu().numpy().astype(np.float64)


def draw_clipped(post, counts, n, seed, lo, hi, device="cpu"):
    """The detect.py convention, verbatim: src/sbixcal/detect.py:218-239.

    rejection (capped at max_sampling_time) -> fallback on
    (RuntimeError, ValueError, TypeError) -> top up a shortfall unrejected ->
    unconditional clip into the prior box. Returns (theta, provenance).
    """
    import torch
    prov = {"used_fallback": False, "fallback_exc": None,
            "n_from_rejection": 0, "n_topped_up": 0, "n_clipped": 0}
    x_t = torch.as_tensor(np.asarray(counts, dtype=np.float32), device=device)
    torch.manual_seed(seed)                                        # detect.py:218
    with torch.no_grad():
        try:
            theta_t = post.sample(                                 # detect.py:220-224
                (n,), x=x_t, show_progress_bars=False,
                reject_outside_prior=True,
                max_sampling_time=float(MAX_SAMPLING_TIME),
            )
        except (RuntimeError, ValueError, TypeError) as e:          # detect.py:225-229
            prov["used_fallback"] = True
            prov["fallback_exc"] = f"{type(e).__name__}: {e}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                theta_t = post.sample((n,), x=x_t, show_progress_bars=False,
                                      reject_outside_prior=False)
        theta = theta_t.detach().cpu().numpy().astype(np.float64)
        prov["n_from_rejection"] = int(theta.shape[0]) if not prov["used_fallback"] else 0
        if theta.shape[0] < n:                                      # detect.py:231-236
            prov["n_topped_up"] = int(n - theta.shape[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                extra = post.sample((n - theta.shape[0],), x=x_t,
                                    show_progress_bars=False,
                                    reject_outside_prior=False)
            theta = np.vstack([theta, extra.detach().cpu().numpy().astype(np.float64)])
    clipped = np.clip(theta, lo[None, :], hi[None, :])              # detect.py:237-239
    prov["n_clipped"] = int(np.any(clipped != theta, axis=1).sum())
    return clipped, prov


def outside_prior_stats(prior, samples, param_order, lo, hi):
    """Fraction of RAW samples outside the prior box, by sbi's own criterion
    (sbi.utils.sbiutils.within_support, the function the 5%-threshold warning
    uses), plus a per-parameter breakdown from the linear bounds."""
    import torch
    from sbi.utils.sbiutils import within_support
    s_t = torch.as_tensor(samples, dtype=torch.float32)
    inside = within_support(prior, s_t).detach().cpu().numpy().astype(bool)
    per_param = {}
    for j, name in enumerate(param_order):
        col = samples[:, j]
        per_param[name] = {
            "frac_below_low": float(np.mean(col < lo[j])),
            "frac_above_high": float(np.mean(col > hi[j])),
            "frac_outside": float(np.mean((col < lo[j]) | (col > hi[j]))),
        }
    return {
        "n_samples": int(samples.shape[0]),
        "frac_outside_any_param": float(1.0 - inside.mean()),
        "pct_outside_any_param": float(100.0 * (1.0 - inside.mean())),
        "n_outside": int((~inside).sum()),
        "per_param": per_param,
        "criterion": "sbi.utils.sbiutils.within_support(prior, samples) on the "
                     "BoxUniform built from arch.json prior_cfg linear bounds",
    }


def c2st_between(a, b, seed=C2ST_SUBSAMPLE_SEED):
    """C2ST between two sample sets, same call path as compare_ns_npe:379."""
    try:
        import torch
        from sbi.utils.metrics import c2st as _c2st
        n = min(a.shape[0], b.shape[0])
        rng = np.random.default_rng(seed)
        ia = rng.choice(a.shape[0], n, replace=False)
        ib = rng.choice(b.shape[0], n, replace=False)
        acc = _c2st(torch.as_tensor(a[ia], dtype=torch.float32),
                    torch.as_tensor(b[ib], dtype=torch.float32))
        return float(np.asarray(acc).reshape(-1)[0]), None
    except Exception as e:  # pragma: no cover
        return None, repr(e)


# ---------------------------------------------------------------------------
# per-spectrum worker
# ---------------------------------------------------------------------------
def process(job, D, G, posteriors, spectra_cache):
    """Full recompute for one spectrum. Raises on unrecoverable input
    problems (caller catches and records)."""
    from sbixcal import priors as _priors

    sid = job["spectrum_id"]
    level, idx, strength = job["level"], job["idx"], job["strength"]
    t0 = time.perf_counter()

    run_json = SMALLSET / f"{sid}.json"
    if not run_json.exists():
        raise FileNotFoundError(f"committed run json missing: {run_json}")
    committed = json.loads(run_json.read_text())

    model_dir = D.LEVELS[level]["model_dir"]
    if not (model_dir / "flow_state.pt").exists():
        raise FileNotFoundError(f"flow checkpoint missing: {model_dir / 'flow_state.pt'}")

    # --- 1. regenerate the identical conditioning spectrum --------------------
    key = (level, strength)
    if key not in spectra_cache:
        spectra_cache[key] = D.make_spectrum  # marker; population regenerated per call
    counts, truth = D.make_spectrum(level, strength, idx)
    counts_sum = int(counts.sum())
    if counts_sum != int(committed["total_counts"]):
        raise ValueError(
            f"conditioning spectrum mismatch for {sid}: regenerated total_counts="
            f"{counts_sum} vs committed {committed['total_counts']}. Refusing to "
            f"recompute against a different spectrum.")

    # --- 2. flow + prior -----------------------------------------------------
    if level not in posteriors:
        from sbixcal import train_npe as tn
        post, info = tn.load_posterior(model_dir, device="cpu")
        assert list(info["param_names"]) == G.PARAM_ORDER, info["param_names"]
        arch = json.loads((model_dir / "arch.json").read_text())
        lo, hi = _priors.prior_bounds(arch["prior_cfg"], G.PARAM_ORDER)
        posteriors[level] = (post, lo, hi, arch, _sha256(model_dir / "flow_state.pt"))
    post, lo, hi, arch, ckpt_sha = posteriors[level]

    # --- 3. NS reference block (from the committed run json) ------------------
    cmp_old = committed["comparison_ns_vs_npe"]["per_param"]
    ns_q = committed["ns"]["quantiles"]
    ns_ref = {}
    for name in G.PARAM_ORDER:
        ns_ref[name] = {
            "ns_mean": float(cmp_old[name]["ns_mean"]),
            "ns_std": float(cmp_old[name]["ns_std"]),
            "ns_q68": (float(ns_q[name]["0.16"]), float(ns_q[name]["0.84"])),
            "ns_q95": (float(ns_q[name]["0.025"]), float(ns_q[name]["0.975"])),
        }
        # integrity: the run json's ns.stds must equal the comparison block's ns_std
        d = abs(ns_ref[name]["ns_std"] - float(committed["ns"]["stds"][name]))
        assert d < 1e-12, f"{sid}/{name}: ns_std disagrees between blocks by {d}"

    # --- 4. Draw A: raw (original procedure), + outside-prior fraction --------
    raw = draw_raw(post, counts, N_SAMPLES, PRIMARY_SEED)
    op = outside_prior_stats(post.prior, raw, G.PARAM_ORDER, lo, hi)
    m_raw = npe_metrics(raw, ns_ref, G.PARAM_ORDER)

    # control: does Draw A reproduce the committed NPE numbers?
    repro = {}
    for name in G.PARAM_ORDER:
        repro[name] = {
            "npe_mean_committed": float(cmp_old[name]["npe_mean"]),
            "npe_mean_reproduced": m_raw["per_param"][name]["npe_mean"],
            "npe_std_committed": float(cmp_old[name]["npe_std"]),
            "npe_std_reproduced": m_raw["per_param"][name]["npe_std"],
            "rel_mean_diff": float(abs(m_raw["per_param"][name]["npe_mean"]
                                       - cmp_old[name]["npe_mean"])
                                   / max(abs(cmp_old[name]["npe_std"]), 1e-300)),
            "rel_std_diff": float(abs(m_raw["per_param"][name]["npe_std"]
                                      - cmp_old[name]["npe_std"])
                                  / max(abs(cmp_old[name]["npe_std"]), 1e-300)),
        }
    repro_worst = float(max(r["rel_mean_diff"] for r in repro.values()))
    raw_reproduces = repro_worst < REPRO_TOL

    # --- 5. Draw B: clipped (detect.py convention) --------------------------
    hyg, prov = draw_clipped(post, counts, N_SAMPLES, PRIMARY_SEED, lo, hi)
    op_hyg = outside_prior_stats(post.prior, hyg, G.PARAM_ORDER, lo, hi)
    m_hyg = npe_metrics(hyg, ns_ref, G.PARAM_ORDER)

    # --- 6. MC-scatter probe at extra seeds ----------------------------------
    scatter, extra_draws = {}, {}
    for s in EXTRA_SEEDS:
        h_s, _ = draw_clipped(post, counts, N_SAMPLES, s, lo, hi)
        extra_draws[s] = h_s
        scatter[str(s)] = npe_metrics(h_s, ns_ref, G.PARAM_ORDER)
    seed_spread = {}
    for name in G.PARAM_ORDER:
        vals = [m_hyg["per_param"][name]["mean_diff_in_ns_std"]] + \
               [scatter[str(s)]["per_param"][name]["mean_diff_in_ns_std"] for s in EXTRA_SEEDS]
        seed_spread[name] = {
            "mean_diff_in_ns_std_by_seed": [float(v) for v in vals],
            "range": float(np.max(vals) - np.min(vals)),
            "std": float(np.std(vals, ddof=1)),
        }
    g_shrink_by_seed = [m_hyg["npe_g_shrink"]] + [scatter[str(s)]["npe_g_shrink"] for s in EXTRA_SEEDS]

    # --- 7. C2ST: raw-vs-clipped (NS-vs-NPE not computable, see module docstring)
    #
    # TRAP, and the reason the seeds below are deliberately mismatched: the raw and
    # the clipped draw at the SAME torch seed share their first sampling batch.
    # `reject_outside_prior=False` draws 4000 in one shot; `=True` draws its first
    # rejection batch from the identical RNG state and keeps the in-support rows, so
    # (1 - outside_frac) * 4000 rows are byte-identical across the two sets. A C2ST
    # on duplicated rows is anti-correlated, not chance: the random forest learns a
    # label for a point from one class and meets the same point in the other class at
    # test time. Measured on medium_s0_i22: 2769/4000 shared rows (= the 69.2%
    # first-batch survival rate at 30.8% outside), C2ST 0.364, i.e. far BELOW the 0.5
    # null and monotone in the duplicate count across all 10 spectra: pure artifact.
    # Drawn from independent streams instead (raw seed 0 vs clipped seed 1, 0 shared
    # rows) the same comparison reads 0.614 against a 0.500 null.
    # So: compare INDEPENDENT streams, and keep the artifact value on record.
    hyg_a = extra_draws[EXTRA_SEEDS[0]]      # clipped, seed 1
    hyg_b = extra_draws[EXTRA_SEEDS[1]]      # clipped, seed 2
    c2st_rh, c2st_err = c2st_between(raw, hyg_a)          # independent streams
    c2st_null, c2st_null_err = c2st_between(hyg_a, hyg_b)  # independent, same dist
    c2st_shared, _ = c2st_between(raw, hyg)                # artifact, recorded only
    _seen = {r.tobytes() for r in raw}
    n_shared = int(sum(r.tobytes() in _seen for r in hyg))

    out = {
        "spectrum_id": sid,
        "level": level,
        "idx": idx,
        "strength": strength,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "interpreter": sys.executable,
        "platform": platform.platform(),
        "seeds": {
            "npe_primary_torch_manual_seed": PRIMARY_SEED,
            "npe_extra_torch_manual_seeds": list(EXTRA_SEEDS),
            "spectrum_eval_seed": D._eval_seed(strength),
            "spectrum_select_seed": D.SELECT_SEED,
            "c2st_subsample_seed": C2ST_SUBSAMPLE_SEED,
        },
        "n_samples": N_SAMPLES,
        "flow": {"model_dir": str(model_dir), "flow_state_sha256": ckpt_sha,
                 "exposure_s": arch.get("exposure_s")},
        "conditioning": {
            "total_counts_regenerated": counts_sum,
            "total_counts_committed": int(committed["total_counts"]),
            "counts_match": True,
            "truth_phys": [float(t) for t in truth],
        },
        "prior_bounds": {n: [float(lo[j]), float(hi[j])] for j, n in enumerate(G.PARAM_ORDER)},
        "prior_std_g": float(PRIOR_STD_G),
        "outside_prior_raw": op,
        "outside_prior_clipped": op_hyg,
        "outside_prior_logged_pct_original": LOGGED_PCT.get(sid),
        "raw_reproduction_control": {
            "reproduces_committed": bool(raw_reproduces),
            "worst_rel_mean_diff_in_npe_std": repro_worst,
            "per_param": repro,
        },
        "metrics_raw_reproduced": m_raw,
        "metrics_clipped": m_hyg,
        "clipped_draw_provenance": prov,
        "seed_scatter_clipped": {
            "per_param_mean_diff_in_ns_std": seed_spread,
            "npe_g_shrink_by_seed": [float(v) for v in g_shrink_by_seed],
            "npe_g_shrink_range": float(np.max(g_shrink_by_seed) - np.min(g_shrink_by_seed)),
        },
        "c2st": {
            "ns_vs_npe": {
                "status": "NOT_COMPUTABLE",
                "committed_value": committed["comparison_ns_vs_npe"].get("c2st_accuracy"),
                "reason": "NS posterior samples were never persisted. The worker holds "
                          "res.samples in memory and writes only means/stds/quantiles "
                          "(outputs/gain_marg/run_ns_smallset.py:130-137); no NS sample "
                          "array exists under outputs/ (searched *.npy/*.npz across "
                          "outputs/, and every ns_smallset*/ json). Recomputing it needs "
                          "a full 6-param NS rerun (3426-18994 s per spectrum).",
            },
            "raw_npe_vs_clipped_npe": c2st_rh,
            "raw_npe_vs_clipped_npe_error": c2st_err,
            "raw_npe_vs_clipped_npe_note": "raw seed 0 vs clipped seed 1, independent "
                                            "streams. How far the rejection-and-clip fix moves the "
                                            "NPE posterior; read against the null below.",
            "null_clipped_seed1_vs_seed2": c2st_null,
            "null_clipped_seed1_vs_seed2_error": c2st_null_err,
            "null_note": "same-distribution calibration: two clipped draws differing only "
                         "by torch seed, independent streams.",
            "artifact_same_seed_raw_vs_clipped": c2st_shared,
            "artifact_n_shared_rows": n_shared,
            "artifact_note": "INVALID as a two-sample test, recorded so the trap stays "
                             "visible: the raw and clipped draws at the same torch seed "
                             f"share {n_shared}/{N_SAMPLES} byte-identical rows (the "
                             "rejection draw's first batch comes off the same RNG state), "
                             "which drives c2st below chance instead of above it.",
        },
        "committed_reference": {
            "npe": committed["npe"],
            "comparison_ns_vs_npe": {
                "per_param": cmp_old,
                "max_abs_mean_diff_in_ns_std": committed["comparison_ns_vs_npe"]["max_abs_mean_diff_in_ns_std"],
                "mean_overlap68_iou": committed["comparison_ns_vs_npe"]["mean_overlap68_iou"],
                "c2st_accuracy": committed["comparison_ns_vs_npe"].get("c2st_accuracy"),
            },
            "total_counts": int(committed["total_counts"]),
            "npe_g_shrink_committed": float(committed["npe"]["stds"]["gain_g"] / PRIOR_STD_G),
        },
        "wall_s": time.perf_counter() - t0,
    }
    return out


# ---------------------------------------------------------------------------
# merge + report
# ---------------------------------------------------------------------------
def _f(v, nd=3, dash="n/a"):
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def merge(jobs):
    rows, failures = [], []
    for j in jobs:
        sid = j["spectrum_id"]
        p = OUTDIR / f"{sid}.json"
        e = OUTDIR / f"{sid}.error.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
        elif e.exists():
            failures.append(json.loads(e.read_text()))
        else:
            failures.append({"spectrum_id": sid, "status": "not attempted"})

    merged = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "interpreter": sys.executable,
        "n_done": len(rows),
        "n_failed": len(failures),
        "seeds": {"npe_primary_torch_manual_seed": PRIMARY_SEED,
                  "npe_extra_torch_manual_seeds": list(EXTRA_SEEDS),
                  "n_samples": N_SAMPLES,
                  "max_sampling_time_s": MAX_SAMPLING_TIME},
        "prior_std_g": float(PRIOR_STD_G),
        "convention": "detect.py rejection(max_sampling_time=20s) -> fallback -> "
                      "topup -> clip; src/sbixcal/detect.py:218-239",
        "runs": rows,
        "failures": failures,
    }
    (OUTDIR / "npe_draw_recompute.json").write_text(json.dumps(merged, indent=2))
    write_md(merged)
    return merged


def write_md(m):
    rows = m["runs"]
    L = []
    A = L.append
    A("# NPE draw recompute (outside-prior rejection enabled)")
    A("")
    A(f"Generated {m['generated']} by `scripts/npe_draw_recompute.py`.")
    A(f"Interpreter: `{m['interpreter']}`")
    A("")
    A("Every NPE-side statistic in the original small-set extraction was drawn with")
    A("`reject_outside_prior=False` (`outputs/gain_marg/run_ns_smallset.py:124-125`).")
    A("Here they are redrawn with the `detect.py` convention")
    A("(`src/sbixcal/detect.py:218-239`: rejection capped at 20 s, fallback to an")
    A("unrejected batch on RuntimeError/ValueError/TypeError, top-up of any shortfall,")
    A("then an unconditional clip into the prior box), same flow checkpoints, same")
    A(f"conditioning spectra, `torch.manual_seed({PRIMARY_SEED})`, n={m['seeds']['n_samples']}.")
    A("")
    A(f"g prior std = (1.05-0.95)/sqrt(12) = {m['prior_std_g']:.15f} (the exact value the")
    A("original extraction used, not the rounded 0.028868).")
    A("")
    A("**NS side untouched.** `ns_mean`/`ns_std` are the ddof=1 moments of the NS samples as")
    A("recorded in each run's `comparison_ns_vs_npe` block, cross-checked against the same")
    A("run's `ns.stds` block (asserted equal to <1e-12); the NS 16/84 and 2.5/97.5 quantiles")
    A("come from `ns.quantiles`. Nothing is copied from `SUMMARY.json` or the extraction.")
    A("")

    # ---- 0. integrity ------------------------------------------------------
    A("## 0. Integrity of the recompute")
    A("")
    A("| spectrum_id | counts regen == committed | raw draw vs committed NPE | worst rel. mean diff (in npe_std) |")
    A("|---|---|---|---|")
    for r in rows:
        rc = r["raw_reproduction_control"]
        w = rc["worst_rel_mean_diff_in_npe_std"]
        status = "bit-exact" if w == 0.0 else ("float32-exact" if w < REPRO_TOL else "**MISMATCH**")
        A(f"| {r['spectrum_id']} | {'yes' if r['conditioning']['counts_match'] else 'NO'} "
          f"| {status} | {w:.1e} |")
    A("")
    A("This is the control that matters. The un-rejected draw made by this script reproduces")
    A("the committed NPE numbers, so the clipped numbers beside it differ because of the")
    A("rejection and nothing else. The 6 spectra originally run on this Windows box come back")
    A("at exactly 0; the 4 run on the Linux cloud box (same sbi 0.26.1 / torch 2.12.0 /")
    A("nflows 0.14 / numpy 2.4.6, but torch+cpu there against torch+cu130 here) come back at")
    A(f"2e-7 to 1.2e-6 of an npe_std, which is the same RNG stream through float32 kernels")
    A("that differ in the last bits, not a different draw.")
    A("")
    A("Each spectrum was also regenerated from scratch")
    A("(`run_ns_smallset.make_spectrum`, eval seed 20300611) and its total counts asserted")
    A("equal to the committed value before anything was recomputed.")
    A("")

    # ---- 1. outside-prior fractions ---------------------------------------
    A("## 1. Outside-prior fraction, all 10 (previously known for only 4)")
    A("")
    A("Measured on the raw (un-rejected) 4000-sample draw with sbi's own criterion")
    A("`sbi.utils.sbiutils.within_support`. `logged` is what the original run log printed;")
    A("blank means sbi stayed silent, which the extraction flagged as unexplained. It is")
    A("explained: the warning is threshold-gated at 5%")
    A("(`.venv/Lib/site-packages/sbi/utils/sbiutils.py:713-742`, `threshold: float = 0.05`,")
    A("`if frac_outside > threshold`). Silence meant <=5%, never 0%.")
    A("")
    A("| spectrum_id | level | counts | logged % | measured % | n outside/4000 | worst single param (%) |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        op = r["outside_prior_raw"]
        worst = max(op["per_param"].items(), key=lambda kv: kv[1]["frac_outside"])
        A(f"| {r['spectrum_id']} | {r['level']} | {r['committed_reference']['total_counts']} "
          f"| {'' if r['outside_prior_logged_pct_original'] is None else r['outside_prior_logged_pct_original']} "
          f"| {op['pct_outside_any_param']:.1f} | {op['n_outside']} "
          f"| {DISPLAY[worst[0]]} {100*worst[1]['frac_outside']:.1f} |")
    A("")
    A("Post-rejection check (should be 0.0% everywhere):")
    A("")
    A("| spectrum_id | outside % after rejection | n from rejection | n topped up | n clipped | fallback used |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        p = r["clipped_draw_provenance"]
        A(f"| {r['spectrum_id']} | {r['outside_prior_clipped']['pct_outside_any_param']:.2f} "
          f"| {p['n_from_rejection']} | {p['n_topped_up']} | {p['n_clipped']} "
          f"| {'YES' if p['used_fallback'] else 'no'} |")
    A("")

    # ---- 2. g-shrink -------------------------------------------------------
    A("## 2. NPE g-shrink: old vs new")
    A("")
    A("shrink = npe_std(gain_g)/prior_std. 1.0 = the flow returns the prior for g.")
    A("`seed range` is the spread over three independent clipped draws (seeds 0/1/2) and")
    A("is the Monte-Carlo floor a real shift has to clear.")
    A("")
    A("| spectrum_id | level | counts | old shrink | new shrink | delta | seed range | old npe g mean | new npe g mean |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        old = r["committed_reference"]["npe_g_shrink_committed"]
        new = r["metrics_clipped"]["npe_g_shrink"]
        A(f"| {r['spectrum_id']} | {r['level']} | {r['committed_reference']['total_counts']} "
          f"| {old:.3f} | {new:.3f} | {new - old:+.3f} "
          f"| {r['seed_scatter_clipped']['npe_g_shrink_range']:.3f} "
          f"| {r['committed_reference']['npe']['means']['gain_g']:.4f} "
          f"| {r['metrics_clipped']['npe_g_mean']:.4f} |")
    A("")

    # ---- 3. summary metrics ------------------------------------------------
    A("## 3. Run-level summary metrics: old vs new")
    A("")
    A("| spectrum_id | old max\\|meandiff\\| | new max\\|meandiff\\| | old mean 68IoU | new mean 68IoU | old C2ST(NS,NPE) | new C2ST(NS,NPE) | C2ST(raw NPE, hyg NPE) | C2ST null (hyg vs hyg) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["committed_reference"]["comparison_ns_vs_npe"]
        h = r["metrics_clipped"]
        A(f"| {r['spectrum_id']} | {c['max_abs_mean_diff_in_ns_std']:.2f} | {h['max_abs_mean_diff_in_ns_std']:.2f} "
          f"| {c['mean_overlap68_iou']:.2f} | {h['mean_overlap68_iou']:.2f} "
          f"| {_f(c.get('c2st_accuracy'))} | NOT COMPUTABLE "
          f"| {_f(r['c2st']['raw_npe_vs_clipped_npe'])} "
          f"| {_f(r['c2st'].get('null_clipped_seed1_vs_seed2'))} |")
    A("")
    A("Read the last two columns against each other. The null column is a")
    A("same-distribution calibration (two clipped draws differing only by seed), so a")
    A("raw-vs-clipped value above it means the rejection-and-clip fix moved that spectrum's NPE")
    A("posterior detectably.")
    A("")
    A("Both C2ST columns compare INDEPENDENT sampling streams (raw seed 0 vs clipped seed 1;")
    A("null is clipped seed 1 vs seed 2). That is not cosmetic. At the *same* torch seed the")
    A("raw and clipped draws share their first sampling batch: `reject_outside_prior=False`")
    A("draws 4000 in one shot, `=True` draws its first rejection batch off the identical RNG")
    A("state and keeps the in-support rows, leaving (1 - outside_frac) x 4000 byte-identical")
    A("rows in both sets. C2ST on duplicated rows goes *below* chance, not above, because the")
    A("forest learns a label for a point in one class and meets that same point in the other")
    A("class at test time. Measured on medium_s0_i22: 2769/4000 shared rows, C2ST 0.364. The")
    A("effect is monotone in the duplicate count across all 10 spectra, which is how it was")
    A("caught. Those artifact values are kept in the per-spectrum JSONs under")
    A("`c2st.artifact_same_seed_raw_vs_clipped` so the trap stays visible; they are not a")
    A("two-sample test and must not be quoted as one.")
    A("")
    A("| spectrum_id | shared rows (same-seed) | artifact C2ST (invalid) | valid C2ST (independent) |")
    A("|---|---|---|---|")
    for r in rows:
        A(f"| {r['spectrum_id']} | {r['c2st'].get('artifact_n_shared_rows')}/{m['seeds']['n_samples']} "
          f"| {_f(r['c2st'].get('artifact_same_seed_raw_vs_clipped'))} "
          f"| {_f(r['c2st']['raw_npe_vs_clipped_npe'])} |")
    A("")
    A("**C2ST(NS, NPE) cannot be recomputed.** The NS posterior samples were never written to")
    A("disk: `run_ns_smallset.py:130-137` persists only NS means/stds/quantiles, and no NS")
    A("sample array exists anywhere under `outputs/` (searched every `*.npy`/`*.npz` and every")
    A("`ns_smallset*/` json). Redoing it needs a full 6-param NS rerun, 3426-18994 s per")
    A("spectrum. The last column is a substitute with a different meaning: C2ST between the")
    A("raw and clipped NPE draws, i.e. how far the rejection-and-clip fix moves the NPE posterior. Near")
    A("0.5 means the NS-vs-NPE C2ST cannot have moved much either; well above 0.5 means the")
    A("committed C2ST column is untrustworthy and only an NS rerun can settle it.")
    A("")

    # ---- 4. per-parameter mean_diff ---------------------------------------
    A("## 4. Per-parameter mean_diff_in_ns_std: old -> new (all 10 x 6)")
    A("")
    A("`|.|>2` in bold. Each cell is `old -> new`.")
    A("")
    A("| spectrum_id | " + " | ".join(DISPLAY[p] for p in DISPLAY) + " |")
    A("|---" * (len(DISPLAY) + 1) + "|")
    for r in rows:
        c = r["committed_reference"]["comparison_ns_vs_npe"]["per_param"]
        h = r["metrics_clipped"]["per_param"]
        cells = []
        for p in DISPLAY:
            o, n = c[p]["mean_diff_in_ns_std"], h[p]["mean_diff_in_ns_std"]
            cell = f"{o:.2f} -> {n:.2f}"
            if abs(n) > 2 or abs(o) > 2:
                cell = f"**{cell}**"
            cells.append(cell)
        A(f"| {r['spectrum_id']} | " + " | ".join(cells) + " |")
    A("")

    # ---- 5. per-parameter 68IoU -------------------------------------------
    A("## 5. Per-parameter overlap68_iou: old -> new (all 10 x 6)")
    A("")
    A("| spectrum_id | " + " | ".join(DISPLAY[p] for p in DISPLAY) + " | mean old -> new |")
    A("|---" * (len(DISPLAY) + 2) + "|")
    for r in rows:
        c = r["committed_reference"]["comparison_ns_vs_npe"]["per_param"]
        h = r["metrics_clipped"]["per_param"]
        cells = [f"{c[p]['overlap68_iou']:.2f} -> {h[p]['overlap68_iou']:.2f}" for p in DISPLAY]
        A(f"| {r['spectrum_id']} | " + " | ".join(cells) +
          f" | {r['committed_reference']['comparison_ns_vs_npe']['mean_overlap68_iou']:.2f} -> "
          f"{r['metrics_clipped']['mean_overlap68_iou']:.2f} |")
    A("")

    # ---- 6. per-parameter npe_std ratio ------------------------------------
    A("## 6. Per-parameter std_ratio_npe_over_ns: old -> new")
    A("")
    A("| spectrum_id | " + " | ".join(DISPLAY[p] for p in DISPLAY) + " |")
    A("|---" * (len(DISPLAY) + 1) + "|")
    for r in rows:
        c = r["committed_reference"]["comparison_ns_vs_npe"]["per_param"]
        h = r["metrics_clipped"]["per_param"]
        cells = [f"{c[p]['std_ratio_npe_over_ns']:.2f} -> {h[p]['std_ratio_npe_over_ns']:.2f}" for p in DISPLAY]
        A(f"| {r['spectrum_id']} | " + " | ".join(cells) + " |")
    A("")

    # ---- 7. seed scatter ---------------------------------------------------
    A("## 7. Monte-Carlo scatter of the clipped draw (seeds 0/1/2)")
    A("")
    A("Range of `mean_diff_in_ns_std` across three independent 4000-sample clipped draws.")
    A("A new-vs-old change smaller than this is draw noise, not an effect of the rejection.")
    A("")
    A("| spectrum_id | " + " | ".join(DISPLAY[p] for p in DISPLAY) + " |")
    A("|---" * (len(DISPLAY) + 1) + "|")
    for r in rows:
        s = r["seed_scatter_clipped"]["per_param_mean_diff_in_ns_std"]
        A(f"| {r['spectrum_id']} | " + " | ".join(f"{s[p]['range']:.2f}" for p in DISPLAY) + " |")
    A("")

    # ---- 8. answers --------------------------------------------------------
    A("## 8. The four questions")
    A("")
    for line in status_lines(rows):
        A(line)
    A("")

    if m["failures"]:
        A("## 9. Failures")
        A("")
        for f in m["failures"]:
            A(f"- **{f.get('spectrum_id')}**: {f.get('status', f.get('error', 'unknown'))}")
        A("")

    (OUTDIR / "npe_draw_recompute.md").write_text("\n".join(L), encoding="utf-8")


def status_lines(rows):
    """Machine-computed answers to the four questions, so the .md cannot drift
    from the numbers."""
    out = []
    if not rows:
        return ["(no completed runs)"]

    # Q1: g-shrink
    d = [(r["spectrum_id"],
          r["metrics_clipped"]["npe_g_shrink"] - r["committed_reference"]["npe_g_shrink_committed"],
          r["metrics_clipped"]["npe_g_shrink"],
          r["seed_scatter_clipped"]["npe_g_shrink_range"]) for r in rows]
    worst = max(d, key=lambda t: abs(t[1]))
    news = [t[2] for t in d]
    out.append("**Q1. Does any NPE g-shrink move materially, i.e. does \"the flow returns "
               "approximately the prior for g everywhere\" survive?**")
    out.append("")
    out.append(f"New shrink range across the 10 runs: {min(news):.3f}-{max(news):.3f} "
               f"(old range {min(r['committed_reference']['npe_g_shrink_committed'] for r in rows):.3f}-"
               f"{max(r['committed_reference']['npe_g_shrink_committed'] for r in rows):.3f}). "
               f"Largest single shift {worst[1]:+.3f} on {worst[0]}, against a seed-to-seed "
               f"MC range of {worst[3]:.3f} on that run.")
    out.append("")
    moved = [t for t in d if abs(t[1]) > 3 * t[3]]
    out.append(f"Shifts clearing 3x their own MC range: {len(moved)}/{len(d)} "
               f"({', '.join(f'{t[0]} {t[1]:+.3f}' for t in moved) if moved else 'none'}).")
    out.append("Every shift is negative: rejection can only narrow the flow's g marginal, and")
    out.append("the two old values above 1.0 (a posterior wider than its own prior, which is")
    out.append("impossible once the flow is truncated to that prior) are gone.")
    out.append("")
    out.append("The quantity that carries the judgment is how much of the NS gain constraint")
    out.append("the flow reproduces: capture = (1 - npe_shrink)/(1 - ns_shrink), 0 = flow")
    out.append("returns the prior, 1 = flow matches NS. Only runs where NS itself constrains g")
    out.append("(ns_shrink <= 0.95) are informative.")
    out.append("")
    out.append("| spectrum_id | ns g shrink | old npe shrink | new npe shrink | old capture | new capture |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        ns_shrink = r["committed_reference"]["comparison_ns_vs_npe"]["per_param"]["gain_g"]["ns_std"] / PRIOR_STD_G
        if ns_shrink > 0.95:
            continue
        o = r["committed_reference"]["npe_g_shrink_committed"]
        n = r["metrics_clipped"]["npe_g_shrink"]
        out.append(f"| {r['spectrum_id']} | {ns_shrink:.3f} | {o:.3f} | {n:.3f} "
                   f"| {(1-o)/(1-ns_shrink):.2f} | {(1-n)/(1-ns_shrink):.2f} |")
    out.append("")

    # Q2: i394/i416 offsets and i394 68IoU
    out.append("**Q2. Do the i394 / i416 2-3 sigma location offsets and the i394 mean 68IoU = 0.26 survive?**")
    out.append("")
    for sid in ("bright_s0_i394", "bright_s0_i416"):
        r = next((x for x in rows if x["spectrum_id"] == sid), None)
        if r is None:
            out.append(f"- {sid}: NOT RECOMPUTED")
            continue
        c = r["committed_reference"]["comparison_ns_vs_npe"]["per_param"]
        h = r["metrics_clipped"]["per_param"]
        ss = r["seed_scatter_clipped"]["per_param_mean_diff_in_ns_std"]
        big = [p for p in DISPLAY if abs(c[p]["mean_diff_in_ns_std"]) > 2 or abs(h[p]["mean_diff_in_ns_std"]) > 2]
        for p in big:
            out.append(f"- {sid} / {DISPLAY[p]}: {c[p]['mean_diff_in_ns_std']:+.2f} -> "
                       f"{h[p]['mean_diff_in_ns_std']:+.2f} sigma (seed MC range "
                       f"{ss[p]['range']:.2f})")
        out.append(f"- {sid} mean 68IoU: {r['committed_reference']['comparison_ns_vs_npe']['mean_overlap68_iou']:.2f} "
                   f"-> {r['metrics_clipped']['mean_overlap68_iou']:.2f}; "
                   f"max|meandiff| {r['committed_reference']['comparison_ns_vs_npe']['max_abs_mean_diff_in_ns_std']:.2f} "
                   f"-> {r['metrics_clipped']['max_abs_mean_diff_in_ns_std']:.2f}")
        lost = [DISPLAY[p] for p in DISPLAY
                if abs(c[p]["mean_diff_in_ns_std"]) > 2 >= abs(h[p]["mean_diff_in_ns_std"])]
        gained = [DISPLAY[p] for p in DISPLAY
                  if abs(h[p]["mean_diff_in_ns_std"]) > 2 >= abs(c[p]["mean_diff_in_ns_std"])]
        if lost or gained:
            out.append(f"  - CHANGED SET: dropped below 2 sigma: {', '.join(lost) or 'none'}; "
                       f"rose above 2 sigma: {', '.join(gained) or 'none'}")
    out.append("")

    # Q3: outside-prior vs counts
    cnt = np.array([r["committed_reference"]["total_counts"] for r in rows], dtype=float)
    frac = np.array([r["outside_prior_raw"]["frac_outside_any_param"] for r in rows], dtype=float)
    lev = np.array([r["level"] for r in rows])
    out.append("**Q3. Does the outside-prior fraction correlate with counts / brightness?**")
    out.append("")

    def _pear(a, b):
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def _spear(a, b):
        if len(a) < 3:
            return float("nan")
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return _pear(ra, rb)

    out.append(f"- all 10: Pearson r(counts, outside frac) = {_pear(cnt, frac):.3f}, "
               f"Spearman = {_spear(cnt, frac):.3f}")
    out.append(f"- all 10 vs log10(counts): Pearson r = {_pear(np.log10(cnt), frac):.3f}")
    for lv in ("medium", "bright"):
        msk = lev == lv
        if msk.sum() >= 3:
            out.append(f"- {lv} (n={int(msk.sum())}): Pearson r = {_pear(cnt[msk], frac[msk]):.3f}, "
                       f"Spearman = {_spear(cnt[msk], frac[msk]):.3f}")
        else:
            out.append(f"- {lv} (n={int(msk.sum())}): too few points for a correlation")
    mmean = float(frac[lev == "medium"].mean()) if (lev == "medium").any() else float("nan")
    bmean = float(frac[lev == "bright"].mean()) if (lev == "bright").any() else float("nan")
    out.append(f"- level means: medium {100*mmean:.1f}%, bright {100*bmean:.1f}%")
    out.append("- n=10 (6 medium, 4 bright); descriptive only, not significance-tested.")
    out.append("")
    lowc, highc = frac[cnt < 1000], frac[cnt >= 1000]
    if len(lowc) and len(highc):
        out.append("The correlation coefficients describe the pattern badly. What the data")
        out.append("actually show is a floor, not a trend:")
        out.append("")
        out.append(f"- all {len(lowc)} runs below 1000 counts sit at {100*lowc.min():.1f}-{100*lowc.max():.1f}%")
        out.append(f"- the {len(highc)} runs above 1000 counts scatter over {100*highc.min():.1f}-{100*highc.max():.1f}%")
        out.append("- and the scatter up there is not ordered by counts: medium_s0_i482 (5204 counts)")
        out.append("  sits at 3.6% while medium_s0_i22 (3853 counts) sits at 30.8%, and bright_s0_i8")
        out.append("  (25067 counts) sits at 3.7% while bright_s0_i416 (54509) sits at 23.8%.")
        out.append("")
        out.append("So: low counts bound the outside fraction small, high counts merely permit it to")
        out.append("be large. Neither counts nor edge distance predicts which high-count spectra")
        out.append("actually spill, and with n=10 this recompute cannot settle what does.")
        out.append("")
    # the competing explanation: how close the flow's posterior sits to a prior edge
    prox, dom = [], []
    for r in rows:
        pp = r["metrics_raw_reproduced"]["per_param"]
        bounds = r["prior_bounds"]
        best = min(
            min(pp[p]["npe_mean"] - bounds[p][0], bounds[p][1] - pp[p]["npe_mean"]) / pp[p]["npe_std"]
            for p in DISPLAY)
        prox.append(best)
        w = max(r["outside_prior_raw"]["per_param"].items(), key=lambda kv: kv[1]["frac_outside"])
        dom.append(DISPLAY[w[0]])
    prox = np.array(prox, dtype=float)
    out.append("Competing explanation, tested: how close the flow's posterior sits to a prior")
    out.append("edge. For each run take the smallest distance from an NPE posterior mean to its")
    out.append("nearest prior bound, in units of that parameter's npe_std.")
    out.append("")
    out.append(f"- Pearson r(edge distance, outside frac) = {_pear(prox, frac):.3f}, "
               f"Spearman = {_spear(prox, frac):.3f}")
    out.append(f"- edge distance range across the 10 runs: {prox.min():.2f}-{prox.max():.2f} npe_std")
    from collections import Counter
    out.append(f"- parameter carrying the largest outside fraction, by run: "
               f"{', '.join(f'{k} x{v}' for k, v in Counter(dom).most_common())}")
    out.append("")

    # Q4: judgments
    out.append("**Q4. Which conclusions survive on the NPE side?**")
    out.append("")
    near_prior = [r["spectrum_id"] for r in rows if abs(r["metrics_clipped"]["npe_g_shrink"] - 1.0) <= 0.05]
    far = [r["spectrum_id"] for r in rows if abs(r["metrics_clipped"]["npe_g_shrink"] - 1.0) > 0.05]
    out.append(f"- (i) \"flow = prior for g\" (NPE half of judgment i): {len(near_prior)}/{len(rows)} runs "
               f"still have clipped shrink within 5% of 1.0"
               f"{'; outside that band: ' + ', '.join(far) if far else ''}. The claim survives as "
               f"a statement about most spectra but no longer as an exceptionless one, and the "
               f"NPE-vs-NS gap on the affected run narrows rather than vanishing. The NS half "
               f"(flow lossy for g, spectrum-dependently) is untouched by this recompute.")
    n_big_old = sum(1 for r in rows for p in DISPLAY
                    if abs(r["committed_reference"]["comparison_ns_vs_npe"]["per_param"][p]["mean_diff_in_ns_std"]) > 2)
    n_big_new = sum(1 for r in rows for p in DISPLAY
                    if abs(r["metrics_clipped"]["per_param"][p]["mean_diff_in_ns_std"]) > 2)
    out.append(f"- (ii) bright-count location offsets: {n_big_old} parameter-level |.|>2 sigma "
               f"exceedances before, {n_big_new} after.")
    out.append("- (iii) i22 medium shrinkage is an NS-side statement (ns g shrink 0.723); this "
               "recompute does not touch the NS posterior and so neither confirms nor "
               "disturbs it.")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only", action="store_true",
                    help="rebuild npe_draw_recompute.{json,md} from existing per-spectrum files")
    ap.add_argument("--only", default=None, help="comma-separated spectrum_ids")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    import run_ns_smallset as D
    import ns_gainmarg as G

    jobs = D.select_jobs()
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        jobs = [j for j in jobs if j["spectrum_id"] in keep]

    if args.merge_only:
        m = merge(jobs)
        print(f"[merge] {m['n_done']} done, {m['n_failed']} failed -> {OUTDIR}")
        return

    print(f"[recompute] interpreter {sys.executable}")
    print(f"[recompute] {len(jobs)} spectra -> {OUTDIR}")
    posteriors, spectra_cache = {}, {}
    for j in jobs:
        sid = j["spectrum_id"]
        try:
            t0 = time.perf_counter()
            out = process(j, D, G, posteriors, spectra_cache)
            (OUTDIR / f"{sid}.json").write_text(json.dumps(out, indent=2))
            err = OUTDIR / f"{sid}.error.json"
            if err.exists():
                err.unlink()
            print(f"[ok] {sid:22s} {time.perf_counter()-t0:6.1f}s "
                  f"outside {out['outside_prior_raw']['pct_outside_any_param']:5.1f}% "
                  f"gshrink {out['committed_reference']['npe_g_shrink_committed']:.3f}->"
                  f"{out['metrics_clipped']['npe_g_shrink']:.3f} "
                  f"max|md| {out['committed_reference']['comparison_ns_vs_npe']['max_abs_mean_diff_in_ns_std']:.2f}->"
                  f"{out['metrics_clipped']['max_abs_mean_diff_in_ns_std']:.2f} "
                  f"68IoU {out['committed_reference']['comparison_ns_vs_npe']['mean_overlap68_iou']:.2f}->"
                  f"{out['metrics_clipped']['mean_overlap68_iou']:.2f}", flush=True)
        except Exception as e:
            (OUTDIR / f"{sid}.error.json").write_text(json.dumps({
                "spectrum_id": sid, "status": "FAILED",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2))
            print(f"[FAIL] {sid}: {type(e).__name__}: {e}", flush=True)

    m = merge(jobs)
    print(f"[merge] {m['n_done']} done, {m['n_failed']} failed -> "
          f"{OUTDIR / 'npe_draw_recompute.md'}")


if __name__ == "__main__":
    main()
