"""C2ST between the nested-sampling and the NPE posterior, for the three cases
whose NS samples were persisted.

INPUTS
------
NS equal-weight samples (the ``samples`` array in each npz, already resampled to
equal weight; the ``weighted_points`` / ``weighted_weights`` arrays in the same
npz are the raw importance-weighted NS chain and are not used here):
  outputs/gain_marg/ns_smallset_v2/i22_widebox_samples.npz       (nsteps=24 / default sampler)
  outputs/gain_marg/ns_smallset_v2/i394_slice_samples.npz        (nsteps=24, SliceSampler)
  outputs/gain_marg/ns_smallset_v2/i416_slice_samples.npz        (nsteps=24, SliceSampler)
  outputs/gain_marg/ns_smallset_v2_ns48/i394_slice_samples.npz   (nsteps=48)
  outputs/gain_marg/ns_smallset_v2_ns48/i416_slice_samples.npz   (nsteps=48)
i22_widebox has no nsteps=48 counterpart: it used the default MLFriends sampler
rather than SliceSampler, so nsteps does not apply to it and only one NS variant
exists. The earlier small-set runs (outputs/gain_marg/run_ns_smallset.py) wrote
moments and quantiles only, so no C2ST can be computed from them at all.

NPE flow checkpoints (outputs/gain_marg/run_ns_smallset.py LEVELS mapping):
  i22_widebox            -> outputs/gain_marg/model_medium  (level=medium)
  i394_slice, i416_slice -> outputs/gain_marg/model_bright   (level=bright)

Conditioning spectrum: the ``observed_counts`` array persisted in each case's
npz, byte-identical between the nsteps=24 and nsteps=48 variant of i394 and i416
(same population seed 20300611, same scheme B4 strength 0.0, same idx). The flow
is conditioned on that array directly rather than on a regenerated spectrum, so
it cannot drift from the array NS's likelihood used.

PARAMETER SPACE AND NPE DRAW CONVENTION
---------------------------------------
Full 6-D theta = ns_gainmarg.PARAM_ORDER (5 physical + gain_g), matching the
npz's param_names and each flow's arch.json param_names exactly (checked before
every draw).

NPE draws use the rejection-and-clip convention, verbatim
src/sbixcal/detect.py:218-239 (posterior_predictive_replicates):
    torch.manual_seed(seed)
    try: theta = post.sample((n,), x=x_obs, reject_outside_prior=True,
                              max_sampling_time=20.0)
    except (RuntimeError, ValueError, TypeError):
        theta = post.sample((n,), x=x_obs, reject_outside_prior=False)   # fallback
    if theta.shape[0] < n:
        theta = vstack(theta, post.sample((n-shape[0],), reject_outside_prior=False))  # top-up
    theta = clip(theta, lo, hi)                                          # clip into prior box
Raw draws (reject_outside_prior=False, unclipped) are used nowhere here. They
put 2.2 to 30.8 per cent of their mass where the NS posterior has exactly zero
support, which on its own forces a C2ST of 0.51 to 0.65 carrying no information
about posterior shape.

TWO PRIOR BOXES (i22_widebox only)
----------------------------------
i22_widebox's NS run used a widened physics prior box for inference (kT up to
8.0 keV and the rest, run_ns_v2.py WIDE_PHYS_PRIORS) while the gain prior stayed
U[0.95,1.05]. The trained flow (model_medium) only knows the nominal box
(ns_gainmarg.PHYS_PRIORS), so there is no wide-box NPE to compare against and
the clip step necessarily clips into the nominal box. Measured here: only 3.1%
of the i22_widebox NS samples fall inside the nominal box on all 5 physical
parameters jointly (tbabs_1_nh 22.8% outside, powerlaw_1_alpha 50.6%,
powerlaw_1_norm 72.8%, blackbodyrad_1_kT 48.2%, blackbodyrad_1_norm 0%, gain_g
0%, as expected since only the physics box was widened). A literal full-sample
C2ST is then dominated by that support mismatch, because most NS draws sit where
the flow cannot produce anything, and it reads close to 1.0 however well the two
agree inside the nominal box. i22 is therefore reported two ways:
  (a) "full": literal C2ST(all NS wide-box samples, NPE draw). This measures
      support mismatch, not posterior shape.
  (b) "nominal_box_restricted": C2ST on the 3.1% of NS wide-box samples that
      fall inside the nominal box (n about 203) against a matched-count NPE
      draw. This is a shape check, on low n, over the region both models can
      express.
Neither replaces the other, so both are written out.

METHOD
------
Standard C2ST: balanced two-class classifier (label 0 = NS samples, label 1 =
NPE samples), sklearn Pipeline(StandardScaler -> MLPClassifier), 5-fold
StratifiedKFold cross-validated accuracy (folds fit their own StandardScaler,
no leakage across the CV split). Sample counts are matched between classes by
construction: NPE is drawn with n = n_ns for that case/nsteps variant (never
subsampled down from a larger pool, since sbi's rejection+fallback+top-up
always returns exactly n).

Controls (computed for every case/nsteps variant):
  - null control: C2ST between two disjoint NPE draws (independent torch seeds
    NPE_NULL_SEED_A=11, NPE_NULL_SEED_B=12) of the same conditioning spectrum.
    Same-distribution calibration, so it must sit near 0.5.
  - independent-stream discipline: the main draw (seed 10) and the two null
    draws (11, 12) use different torch seeds, so no sampling batch is shared
    between them. A shared first RNG batch would put duplicate rows in both
    classes and drive the C2ST below chance. That trap does not apply to the
    NS-vs-NPE comparison, since those are different generative processes, but
    the null-vs-null pair uses the same discipline for a like-for-like null.

Pinned seeds (nothing drawn from entropy):
  NPE_MAIN_SEED   = 10   (NPE draw compared against NS)
  NPE_NULL_SEED_A = 11   (null control, draw A)
  NPE_NULL_SEED_B = 12   (null control, draw B)
  CLF_SEED        = 0    (MLPClassifier random_state)
  CV_SEED         = 0    (StratifiedKFold random_state)
  MAX_SAMPLING_TIME = 20.0 s (sbi rejection cap, detect.py default)

Writes only into outputs/gain_marg/ns_smallset_v2/.

Run (repo venv, from repo root):
    .venv\\Scripts\\python.exe outputs\\gain_marg\\ns_smallset_v2\\c2st_real.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # outputs/gain_marg/ns_smallset_v2
GAIN_MARG = HERE.parent                          # outputs/gain_marg
ROOT = HERE.parents[2]                           # repo root
sys.path.insert(0, str(GAIN_MARG))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import ns_gainmarg as G  # noqa: E402

PARAM_ORDER = list(G.PARAM_ORDER)

# ---------------------------------------------------------------------------
# pinned configuration
# ---------------------------------------------------------------------------
NPE_MAIN_SEED = 10
NPE_NULL_SEED_A = 11
NPE_NULL_SEED_B = 12
CLF_SEED = 0
CV_SEED = 0
MAX_SAMPLING_TIME = 20.0
N_CV_FOLDS = 5
MLP_HIDDEN = (64, 64)

CASES = {
    "i22_widebox": {
        "level": "medium",
        "variants": {"ns24": HERE / "i22_widebox_samples.npz"},
        "note": "default MLFriends sampler, wide physics box for NS inference; "
                "flow trained on nominal box only; see the two-prior-boxes "
                "note in the module docstring.",
    },
    "i394_slice": {
        "level": "bright",
        "variants": {
            "ns24": HERE / "i394_slice_samples.npz",
            "ns48": HERE.parent / "ns_smallset_v2_ns48" / "i394_slice_samples.npz",
        },
        "note": "SliceSampler, nominal physics box (matches flow training box).",
    },
    "i416_slice": {
        "level": "bright",
        "variants": {
            "ns24": HERE / "i416_slice_samples.npz",
            "ns48": HERE.parent / "ns_smallset_v2_ns48" / "i416_slice_samples.npz",
        },
        "note": "SliceSampler, nominal physics box (matches flow training box).",
    },
}

MODEL_DIRS = {
    "medium": GAIN_MARG / "model_medium",
    "bright": GAIN_MARG / "model_bright",
}


# ---------------------------------------------------------------------------
# NPE draw, verbatim src/sbixcal/detect.py:218-239 convention
# ---------------------------------------------------------------------------
def clipped_draw(post, counts, n, seed, lo, hi, device="cpu"):
    import torch

    prov = {"used_fallback": False, "fallback_exc": None,
            "n_from_rejection": 0, "n_topped_up": 0, "n_clipped": 0}
    x_t = torch.as_tensor(np.asarray(counts, dtype=np.float32), device=device)
    torch.manual_seed(seed)
    with torch.no_grad():
        try:
            theta_t = post.sample(
                (n,), x=x_t, show_progress_bars=False,
                reject_outside_prior=True, max_sampling_time=float(MAX_SAMPLING_TIME),
            )
        except (RuntimeError, ValueError, TypeError) as e:
            prov["used_fallback"] = True
            prov["fallback_exc"] = f"{type(e).__name__}: {e}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                theta_t = post.sample((n,), x=x_t, show_progress_bars=False,
                                      reject_outside_prior=False)
        theta = theta_t.detach().cpu().numpy().astype(np.float64)
        prov["n_from_rejection"] = int(theta.shape[0]) if not prov["used_fallback"] else 0
        if theta.shape[0] < n:
            prov["n_topped_up"] = int(n - theta.shape[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                extra = post.sample((n - theta.shape[0],), x=x_t, show_progress_bars=False,
                                    reject_outside_prior=False)
            theta = np.vstack([theta, extra.detach().cpu().numpy().astype(np.float64)])
    clipped = np.clip(theta, lo[None, :], hi[None, :])
    prov["n_clipped"] = int(np.any(clipped != theta, axis=1).sum())
    return clipped, prov


# ---------------------------------------------------------------------------
# C2ST core
# ---------------------------------------------------------------------------
def run_c2st(a, b, clf_seed=CLF_SEED, cv_seed=CV_SEED, n_folds=N_CV_FOLDS):
    """5-fold StratifiedKFold CV accuracy of a small MLP separating a vs b.

    a, b: (n, d) arrays, same n (balanced classes by construction upstream).
    StandardScaler is fit INSIDE each CV fold (via Pipeline), not globally, so
    no test-fold information leaks into the scaler.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.shape[0], b.shape[0])
    X = np.concatenate([a[:n], b[:n]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    clf = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu", solver="adam",
            max_iter=1000, early_stopping=True, n_iter_no_change=20,
            random_state=clf_seed,
        )),
    ])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std(ddof=1)),
        "fold_scores": [float(s) for s in scores],
        "n_per_class": int(n),
        "n_folds": int(n_folds),
        "classifier": f"Pipeline(StandardScaler, MLPClassifier(hidden_layer_sizes={MLP_HIDDEN}, "
                      f"activation=relu, solver=adam, max_iter=1000, early_stopping=True, "
                      f"n_iter_no_change=20, random_state={clf_seed}))",
        "cv": f"StratifiedKFold(n_splits={n_folds}, shuffle=True, random_state={cv_seed})",
    }


# ---------------------------------------------------------------------------
# per-case, per-variant worker
# ---------------------------------------------------------------------------
def process_variant(case_name, level, npz_path, posteriors):
    d = np.load(npz_path, allow_pickle=True)
    assert list(d["param_names"]) == PARAM_ORDER, (case_name, list(d["param_names"]))
    ns_samples = np.asarray(d["samples"], dtype=np.float64)
    counts = d["observed_counts"]
    n_ns = ns_samples.shape[0]

    model_dir = MODEL_DIRS[level]
    if level not in posteriors:
        from sbixcal import train_npe as tn
        from sbixcal import priors as _priors
        post, info = tn.load_posterior(str(model_dir), device="cpu")
        assert list(info["param_names"]) == PARAM_ORDER, info["param_names"]
        arch = json.loads((model_dir / "arch.json").read_text())
        lo, hi = _priors.prior_bounds(arch["prior_cfg"], PARAM_ORDER)
        posteriors[level] = (post, lo, hi, arch)
    post, lo, hi, arch = posteriors[level]

    t0 = time.perf_counter()
    npe_main, prov_main = clipped_draw(post, counts, n_ns, NPE_MAIN_SEED, lo, hi)
    npe_null_a, prov_a = clipped_draw(post, counts, n_ns, NPE_NULL_SEED_A, lo, hi)
    npe_null_b, prov_b = clipped_draw(post, counts, n_ns, NPE_NULL_SEED_B, lo, hi)
    draw_wall_s = time.perf_counter() - t0

    real = run_c2st(ns_samples, npe_main)
    null = run_c2st(npe_null_a, npe_null_b)

    out = {
        "case": case_name,
        "level": level,
        "variant_npz": npz_path.relative_to(ROOT).as_posix(),
        "model_dir": model_dir.relative_to(ROOT).as_posix(),
        "flow_exposure_s": arch.get("exposure_s"),
        "conditioning_total_counts": float(np.asarray(counts).sum()),
        "n_ns_samples": int(n_ns),
        "param_order": PARAM_ORDER,
        "c2st_real": real,
        "c2st_null": null,
        "npe_draw_provenance": {
            "main_seed": NPE_MAIN_SEED, "main": prov_main,
            "null_a_seed": NPE_NULL_SEED_A, "null_a": prov_a,
            "null_b_seed": NPE_NULL_SEED_B, "null_b": prov_b,
        },
        "draw_wall_s": draw_wall_s,
    }
    return out, ns_samples, (post, lo, hi)


def i22_nominal_box_restriction(ns_samples, post, lo_nominal, hi_nominal):
    """Secondary, matched-support check for i22_widebox only (see the
    two-prior-boxes note): restrict the NS wide-box samples to the subset that
    falls inside the flow's nominal training box on all 6 parameters, and
    compare that subset against a matched-count NPE draw (nominal box)."""
    inside = np.all((ns_samples >= lo_nominal[None, :]) & (ns_samples <= hi_nominal[None, :]), axis=1)
    n_inside = int(inside.sum())
    frac_inside = float(inside.mean())
    restricted = ns_samples[inside]

    per_param_outside = {}
    for j, name in enumerate(PARAM_ORDER):
        col = ns_samples[:, j]
        per_param_outside[name] = float(np.mean((col < lo_nominal[j]) | (col > hi_nominal[j])))

    if n_inside < 2 * N_CV_FOLDS:
        return {
            "n_inside_nominal_box": n_inside,
            "frac_inside_nominal_box": frac_inside,
            "per_param_frac_outside_nominal_box": per_param_outside,
            "status": "SKIPPED: fewer than 2*n_folds in-box NS samples, CV undefined",
        }

    d = np.load(CASES["i22_widebox"]["variants"]["ns24"])
    counts = d["observed_counts"]
    npe_main, prov_main = clipped_draw(post, counts, n_inside, NPE_MAIN_SEED + 100, lo_nominal, hi_nominal)
    npe_null_a, prov_a = clipped_draw(post, counts, n_inside, NPE_NULL_SEED_A + 100, lo_nominal, hi_nominal)
    npe_null_b, prov_b = clipped_draw(post, counts, n_inside, NPE_NULL_SEED_B + 100, lo_nominal, hi_nominal)

    real = run_c2st(restricted, npe_main)
    null = run_c2st(npe_null_a, npe_null_b)
    return {
        "n_inside_nominal_box": n_inside,
        "frac_inside_nominal_box": frac_inside,
        "per_param_frac_outside_nominal_box": per_param_outside,
        "status": "computed",
        "c2st_real": real,
        "c2st_null": null,
        "npe_draw_provenance": {
            "main_seed": NPE_MAIN_SEED + 100, "main": prov_main,
            "null_a_seed": NPE_NULL_SEED_A + 100, "null_a": prov_a,
            "null_b_seed": NPE_NULL_SEED_B + 100, "null_b": prov_b,
        },
    }


def main():
    print(f"[c2st_real] interpreter {sys.executable}", flush=True)
    posteriors = {}
    results = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "platform": platform.platform(),
        "param_order": PARAM_ORDER,
        "seeds": {
            "npe_main_seed": NPE_MAIN_SEED,
            "npe_null_seed_a": NPE_NULL_SEED_A,
            "npe_null_seed_b": NPE_NULL_SEED_B,
            "clf_seed": CLF_SEED,
            "cv_seed": CV_SEED,
        },
        "classifier_config": {
            "type": "sklearn Pipeline(StandardScaler, MLPClassifier)",
            "hidden_layer_sizes": list(MLP_HIDDEN),
            "activation": "relu", "solver": "adam", "max_iter": 1000,
            "early_stopping": True, "n_iter_no_change": 20,
            "n_cv_folds": N_CV_FOLDS,
        },
        "npe_draw_convention": "rejection-and-clip (src/sbixcal/detect.py:218-239): "
                               "reject_outside_prior=True, max_sampling_time=20.0s, "
                               "exception fallback to unrejected draw, top-up any "
                               "shortfall with unrejected draws, then clip into the "
                               "flow's training prior box.",
        "cases": {},
    }

    for case_name, cfg in CASES.items():
        level = cfg["level"]
        case_out = {"level": level, "note": cfg["note"], "variants": {}}
        i22_ns_samples_for_restriction = None
        i22_post_lo_hi = None
        for variant, npz_path in cfg["variants"].items():
            if not npz_path.exists():
                case_out["variants"][variant] = {"status": f"MISSING: {npz_path}"}
                print(f"[MISSING] {case_name}/{variant}: {npz_path}", flush=True)
                continue
            t0 = time.perf_counter()
            out, ns_samples, plh = process_variant(case_name, level, npz_path, posteriors)
            out["wall_s"] = time.perf_counter() - t0
            case_out["variants"][variant] = out
            print(f"[ok] {case_name}/{variant}: n_ns={out['n_ns_samples']} "
                  f"c2st_real={out['c2st_real']['accuracy_mean']:.4f} "
                  f"c2st_null={out['c2st_null']['accuracy_mean']:.4f} "
                  f"({out['wall_s']:.1f}s)", flush=True)
            if case_name == "i22_widebox":
                i22_ns_samples_for_restriction = ns_samples
                i22_post_lo_hi = plh

        if case_name == "i22_widebox" and i22_ns_samples_for_restriction is not None:
            post, lo, hi = i22_post_lo_hi
            t0 = time.perf_counter()
            restr = i22_nominal_box_restriction(i22_ns_samples_for_restriction, post, lo, hi)
            restr["wall_s"] = time.perf_counter() - t0
            case_out["nominal_box_restricted_secondary_check"] = restr
            if restr.get("status") == "computed":
                print(f"[ok] i22_widebox/nominal_box_restricted: "
                      f"n_inside={restr['n_inside_nominal_box']} "
                      f"({100*restr['frac_inside_nominal_box']:.1f}% of NS draws) "
                      f"c2st_real={restr['c2st_real']['accuracy_mean']:.4f} "
                      f"c2st_null={restr['c2st_null']['accuracy_mean']:.4f} "
                      f"({restr['wall_s']:.1f}s)", flush=True)
            else:
                print(f"[skip] i22_widebox/nominal_box_restricted: {restr['status']}", flush=True)

        results["cases"][case_name] = case_out

    out_path = HERE / "c2st_real_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
