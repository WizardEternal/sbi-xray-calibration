"""Three additional gain-marginalized NS runs.

Reuses outputs/gain_marg/ns_gainmarg.py machinery (GainFolder, make_loglike,
make_transform) but adds the two things the original small-set runs lacked:

  1. Configurable physical prior box. ns_gainmarg.run_ns_gainmarg() hardcodes
     make_transform() (nominal box). Here the box is a per-run config entry, so
     the wide-box test can widen the physics prior while leaving the gain prior
     U[0.95, 1.05] untouched.
  2. An independent sampler: optional ultranest.stepsampler.SliceSampler
     in place of the default MLFriends ellipsoidal region sampler.

It also persists the equal-weight posterior samples, which the original runs did
not, and without which C2ST(NS, NPE) cannot be computed at all.

Runs (each one detached process, `--run <name>`):

  smoke_default  i87 medium, wide box,    default sampler, min_live 50, capped
  smoke_slice    i87 medium, nominal box, slice sampler,   min_live 50, capped
  i22_widebox    i22 medium, WIDE box,    default sampler, min_live 400
  i394_slice     i394 bright, nominal box, SLICE sampler,  min_live 400
  i416_slice     i416 bright, nominal box, SLICE sampler,  min_live 400

Both smokes are WIRING TESTS ONLY (min_live 50 + an ncall cap). Their numbers are
NOT posterior estimates and must never be compared to anything.

Artifacts per run, all in this directory:
    <run>.RUNNING   marker, present only while the process is alive
    <run>.DONE      marker written on success  (or <run>.FAILED with traceback)
    <run>.log       stdout+stderr
    <run>.progress  heartbeat, rewritten ~every 60 s
    <run>.json      config + priors + sampler + logz + moments + quantiles + g-shrink
    <run>_samples.npz  equal-weight posterior samples, one named array per parameter

Usage (repo venv, from repo root):
    .venv\\Scripts\\python.exe outputs\\gain_marg\\ns_smallset_v2\\run_ns_v2.py --run i22_widebox
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # outputs/gain_marg/ns_smallset_v2
GAIN_MARG = HERE.parent                           # outputs/gain_marg
ROOT = HERE.parents[2]                            # repo root
sys.path.insert(0, str(GAIN_MARG))
sys.path.insert(0, str(ROOT))

import ns_gainmarg as G  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
N_POP = 500
SEED_EVAL = 20260611 + 40000      # = 20300611, the eval_gainmarg B4 spectrum seed
STRENGTH = 0.0                    # clean control (scheme B4, strength 0)
SCHEME = "B4"
NS_SEED = 0
NS_DLOGZ = 0.5
NS_NDRAW_MIN = 2000
NS_NDRAW_MAX = 65536

# std of the gain prior U[0.95, 1.05]: width 0.1 -> 0.1/sqrt(12)
GAIN_PRIOR_STD = 0.1 / np.sqrt(12.0)              # 0.028867513459481287

QUANTILES_OUT = (0.025, 0.16, 0.5, 0.84, 0.975)

EXPOSURE = {"medium": G.EXPOSURE_MEDIUM, "bright": G.EXPOSURE_BRIGHT}

# nominal physics box = exactly the training / original-NS box
NOMINAL_PHYS_PRIORS = dict(G.PHYS_PRIORS)

# wide physics box for the truncation test. Gain prior deliberately UNCHANGED.
WIDE_PHYS_PRIORS = {
    "tbabs_1_nh":          {"dist": "uniform",    "low": 0.05,   "high": 0.6},
    "powerlaw_1_alpha":    {"dist": "uniform",    "low": 0.3,    "high": 4.5},
    "powerlaw_1_norm":     {"dist": "loguniform", "low": 1.0e-6, "high": 1.0e-1},
    "blackbodyrad_1_kT":   {"dist": "uniform",    "low": 0.05,   "high": 8.0},
    "blackbodyrad_1_norm": {"dist": "loguniform", "low": 1.0e-4, "high": 5.0},
}

PHYS_BOXES = {"nominal": NOMINAL_PHYS_PRIORS, "wide": WIDE_PHYS_PRIORS}

# default slice-sampler step count: 2 * ndim with ndim = 6 (5 phys + g).
DEFAULT_NSTEPS = 12

WIDEBOX_PREDICTION = (
    "Recorded before the run, 2026-08-14. Prior-box truncation test for medium i22. "
    "The original nominal-box run gave std(gain_g) = 0.0208732, i.e. g-shrink = "
    "std(g)/0.0288675 = 0.7230, so the g posterior is 28% narrower than its prior. "
    "i22's true kT is 2.80998 keV, sitting 0.19 keV below the nominal prior ceiling "
    "kT <= 3.0, so the exact blackbody kT-proportional-to-g degeneracy direction runs "
    "into the wall of the box. Hypothesis: the apparent g constraint is that wall and "
    "not the data. Prediction if the hypothesis is true: with kT allowed up to 8.0, "
    "and the other physics bounds widened too, the degeneracy is free to extend, so "
    "g-shrink moves substantially toward 1.0 (g posterior about equal to g prior). "
    "Prediction if false: g-shrink stays near 0.72 and the constraint is genuinely "
    "data-driven."
)

RUNS = {
    "smoke_default": {
        "level": "medium", "idx": 87, "expect_counts": 384,
        "phys_box": "wide", "sampler": "default",
        "min_live": 50, "max_ncalls": 6000, "smoke": True,
        "purpose": "WIRING TEST ONLY (wide-box + default-sampler path). Numbers meaningless.",
    },
    "smoke_slice": {
        "level": "medium", "idx": 87, "expect_counts": 384,
        "phys_box": "nominal", "sampler": "slice",
        "min_live": 50, "max_ncalls": 6000, "smoke": True,
        "purpose": "WIRING TEST + slice-sampler cost calibration. Numbers meaningless.",
    },
    "smoke_slice2": {
        "level": "medium", "idx": 87, "expect_counts": 384,
        "phys_box": "nominal", "sampler": "slice",
        "min_live": 50, "max_ncalls": 1200, "smoke": True,
        "purpose": ("WIRING TEST ONLY: re-runs the slice path against the ADDED "
                    "validity-diagnostics + hardened-json code, which smoke_slice "
                    "predates. Numbers meaningless."),
    },
    "i22_widebox": {
        "level": "medium", "idx": 22, "expect_counts": 3853,
        "phys_box": "wide", "sampler": "default",
        "min_live": 400, "max_ncalls": None, "smoke": False,
        "purpose": "Prior-box truncation test for the i22 g-shrink.",
        "prediction": WIDEBOX_PREDICTION,
    },
    "i394_slice": {
        "level": "bright", "idx": 394, "expect_counts": 6829,
        "phys_box": "nominal", "sampler": "slice",
        "min_live": 400, "max_ncalls": None, "smoke": False,
        "purpose": ("Independent-sampler NS realization for bright i394. The earlier "
                    "bright runs used exactly one sampler (MLFriends ellipsoidal), so "
                    "no sampler-independent check existed for these spectra."),
    },
    "i416_slice": {
        "level": "bright", "idx": 416, "expect_counts": 54509,
        "phys_box": "nominal", "sampler": "slice",
        "min_live": 400, "max_ncalls": None, "smoke": False,
        "purpose": ("Independent-sampler NS realization for bright i416 (54509 counts, "
                    "the expensive one)."),
    },
}


# ---------------------------------------------------------------------------
# spectrum regeneration (identical convention to run_ns_smallset.make_spectrum)
# ---------------------------------------------------------------------------
def make_spectrum(level: str, idx: int):
    from sbixcal import responses as R
    from sbixcal.misspec import simulate_misspec_population
    oc = R.scale_exposure(R.load_base_obsconf(G.RESP), EXPOSURE[level])
    x, th, present = simulate_misspec_population(
        G.BASE_MODEL, G.PHYS_PRIORS, oc, SCHEME, float(STRENGTH), N_POP, SEED_EVAL)
    assert present == G.PHYS_ORDER, present
    return np.asarray(x)[idx].astype(np.float64), np.asarray(th)[idx].astype(np.float64)


# ---------------------------------------------------------------------------
# heartbeat: rewritten from inside the likelihood so a future session can see
# progress without attaching to the process.
# ---------------------------------------------------------------------------
class Heartbeat:
    def __init__(self, path: Path, folder, every_s: float = 60.0):
        self.path, self.folder, self.every_s = path, folder, every_s
        self.t0 = time.perf_counter()
        self.last = 0.0

    def tick(self):
        now = time.perf_counter()
        if now - self.last < self.every_s:
            return
        self.last = now
        st = self.folder.stats()
        try:
            self.path.write_text(json.dumps({
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_s": round(now - self.t0, 1),
                "elapsed_h": round((now - self.t0) / 3600.0, 3),
                "n_loglike_blocks": st["n_loglike_blocks"],
                "n_points": st["n_points"],
                "n_fakeit_calls": st["n_fakeit_calls"],
                "ms_per_point": round(st["ms_per_point"], 1),
            }, indent=2))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# the NS run
# ---------------------------------------------------------------------------
def run_one(name: str, nsteps: int = DEFAULT_NSTEPS, resumable: bool = False):
    cfg = RUNS[name]
    HERE.mkdir(parents=True, exist_ok=True)
    running = HERE / f"{name}.RUNNING"
    done = HERE / f"{name}.DONE"
    failed = HERE / f"{name}.FAILED"
    progress = HERE / f"{name}.progress"
    out_json = HERE / f"{name}.json"
    out_npz = HERE / f"{name}_samples.npz"

    t_all = time.perf_counter()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    running.write_text(json.dumps({"pid": os.getpid(), "started": started, "run": name}, indent=2))
    if failed.exists():
        failed.unlink()

    try:
        level, idx = cfg["level"], cfg["idx"]
        print(f"[{name}] pid {os.getpid()} start {started}", flush=True)

        # ---- hard sanity gate: total counts must match the committed spectrum ----
        counts, truth = make_spectrum(level, idx)
        total = int(counts.sum())
        if total != cfg["expect_counts"]:
            raise AssertionError(
                f"COUNTS GATE FAILED for {name}: regenerated total counts {total} != "
                f"committed {cfg['expect_counts']}. Refusing to run: the spectrum is "
                f"not the one the original runs measured.")
        print(f"[{name}] counts gate OK: {total} counts (level={level}, idx={idx})", flush=True)
        print(f"[{name}] truth_phys = {truth.tolist()}", flush=True)

        phys = PHYS_BOXES[cfg["phys_box"]]
        priors = {**phys, **G.GAIN_PRIOR}
        for p in G.PARAM_ORDER:
            assert p in priors, f"prior missing for {p}"
        assert priors["gain_g"]["low"] == 0.95 and priors["gain_g"]["high"] == 1.05, \
            "gain prior must stay U[0.95, 1.05]"

        folder = G.GainFolder(EXPOSURE[level])
        wt = folder.warmup()
        print(f"[{name}] GainFolder built {folder.build_time_s:.1f}s, warmup {wt:.1f}s "
              f"({folder.n_bins} gain bins)", flush=True)

        base_loglike = G.make_loglike(counts, folder)
        hb = Heartbeat(progress, folder)

        def loglike(theta):
            hb.tick()
            return base_loglike(theta)

        transform = G.make_transform(priors=priors, param_order=G.PARAM_ORDER)

        from ultranest import ReactiveNestedSampler
        np.random.seed(NS_SEED)
        if resumable:
            log_dir = str(HERE / f"{name}_ns_logdir")
            sampler = ReactiveNestedSampler(
                list(G.PARAM_ORDER), loglike, transform,
                log_dir=log_dir, resume='resume', vectorized=True,
                ndraw_min=NS_NDRAW_MIN, ndraw_max=NS_NDRAW_MAX,
            )
        else:
            log_dir = None
            sampler = ReactiveNestedSampler(
                list(G.PARAM_ORDER), loglike, transform,
                log_dir=None, vectorized=True,               # log_dir=None: Windows h5py dodge
                ndraw_min=NS_NDRAW_MIN, ndraw_max=NS_NDRAW_MAX,
            )

        sampler_cfg = {"kind": "default_mlfriends_ellipsoidal"}
        if cfg["sampler"] == "slice":
            import ultranest.stepsampler as SS
            sampler.stepsampler = SS.SliceSampler(
                nsteps=int(nsteps),
                generate_direction=SS.generate_mixture_random_direction,
                adaptive_nsteps=False,
                region_filter=False,
            )
            sampler_cfg = {
                "kind": "ultranest.stepsampler.SliceSampler",
                "nsteps": int(nsteps),
                "generate_direction": "generate_mixture_random_direction",
                "adaptive_nsteps": False,
                "region_filter": False,
            }
        print(f"[{name}] sampler = {sampler_cfg}", flush=True)

        t0 = time.perf_counter()
        res = sampler.run(
            min_num_live_points=int(cfg["min_live"]),
            max_ncalls=(int(cfg["max_ncalls"]) if cfg["max_ncalls"] else None),
            dlogz=float(NS_DLOGZ), show_status=False, viz_callback=False,
        )
        wall = time.perf_counter() - t0

        samples = np.asarray(res["samples"], dtype=np.float64)
        means, stds, quants = {}, {}, {}
        for j, p in enumerate(G.PARAM_ORDER):
            col = samples[:, j]
            means[p] = float(col.mean())
            stds[p] = float(col.std(ddof=1))
            quants[p] = {f"{q:g}": float(v)
                         for q, v in zip(QUANTILES_OUT, np.quantile(col, QUANTILES_OUT))}

        g_shrink = float(stds["gain_g"] / GAIN_PRIOR_STD)

        # ---- persist equal-weight samples, named columns ----
        npz = {p: samples[:, j] for j, p in enumerate(G.PARAM_ORDER)}
        npz["samples"] = samples
        npz["param_names"] = np.array(list(G.PARAM_ORDER))
        npz["observed_counts"] = counts
        npz["truth_phys"] = truth
        ws = res.get("weighted_samples")
        if isinstance(ws, dict):
            for k in ("points", "weights", "logl"):
                if k in ws:
                    npz[f"weighted_{k}"] = np.asarray(ws[k])
        np.savez_compressed(out_npz, **npz)
        print(f"[{name}] wrote {samples.shape[0]} equal-weight samples -> {out_npz.name}",
              flush=True)

        # every JSON-safe scalar ultranest reported (covers achieved dlogz proxies)
        scal = {}
        for k, v in res.items():
            if isinstance(v, (bool, str)):
                scal[k] = v
            elif isinstance(v, (int, float, np.floating, np.integer)):
                scal[k] = float(v)

        # ---- VALIDITY DIAGNOSTICS -------------------------------------------
        # For a step sampler these are the whole ballgame: too-few nsteps means the
        # chain does not travel far enough between accepted points, which biases the
        # posterior narrow, i.e. it would spuriously confirm a small g-shrink.
        # The Mann-Whitney-U insertion-order test is ultranest's detector for exactly
        # that failure (a run that under-explores shows non-uniform insertion order).
        # Wrapped whole: a bug in a DIAGNOSTIC must never destroy a 12-hour run.
        # (The samples npz is already on disk above, before this point, by design.)
        diag = {}
        try:
            mww = res.get("insertion_order_MWW_test")
            if isinstance(mww, dict):
                diag["insertion_order_MWW_test"] = {
                    k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                        else str(v)) for k, v in mww.items()}
                diag["insertion_order_note"] = (
                    "ultranest flags a run when the MWW independent-sample check fails; "
                    "'converged'/large p = insertion order uniform = sampling looked "
                    "unbiased. A FAIL here invalidates the run; rerun with 2x nsteps.")
            if cfg["sampler"] == "slice":
                try:
                    info = sampler.stepsampler.get_info_dict()
                    diag["stepsampler_info"] = json.loads(json.dumps(info, default=str))
                except Exception as e:
                    diag["stepsampler_info_error"] = repr(e)
                try:
                    diag["stepsampler_mean_jump_distance"] = float(
                        sampler.stepsampler.mean_jump_distance)
                    diag["stepsampler_far_enough_fraction"] = float(
                        sampler.stepsampler.far_enough_fraction)
                except Exception as e:
                    diag["stepsampler_jump_stats_error"] = repr(e)
            diag["evals_per_iteration"] = (float(res["ncall"]) / float(res["niter"])
                                           if res["niter"] else float("nan"))
        except Exception as e:
            diag["diagnostics_error"] = repr(e)

        out = {
            "run": name,
            "purpose": cfg["purpose"],
            "smoke_wiring_test_only": bool(cfg["smoke"]),
            "started": started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": platform.python_version(),
            "pid": os.getpid(),
            "spectrum": {
                "level": level, "idx": idx, "scheme": SCHEME, "strength": STRENGTH,
                "population": N_POP, "spectrum_eval_seed": SEED_EVAL,
                "exposure_s": EXPOSURE[level],
                "total_counts": total, "expected_counts": cfg["expect_counts"],
                "counts_gate": "PASS", "truth_phys": truth.tolist(),
                "phys_order": list(G.PHYS_ORDER),
            },
            "config": {
                "phys_box": cfg["phys_box"],
                "priors": priors,
                "gain_prior_std": GAIN_PRIOR_STD,
                "sampler": sampler_cfg,
                "min_live": cfg["min_live"], "dlogz_target": NS_DLOGZ,
                "max_ncalls": cfg["max_ncalls"],
                "n_gain_bins": G.N_GAIN_BINS,
                "ndraw_min": NS_NDRAW_MIN, "ndraw_max": NS_NDRAW_MAX,
                "ns_seed": NS_SEED, "log_dir": log_dir,
            },
            "ns": {
                "logz": float(res["logz"]), "logzerr": float(res["logzerr"]),
                "achieved_dlogz_proxy": float(res.get("logzerr_tail", res["logzerr"])),
                "ess": float(res.get("ess", np.nan)),
                "n_like_evals": int(res["ncall"]), "niter": int(res["niter"]),
                "n_equal_weight_samples": int(samples.shape[0]),
                "wall_s": float(wall), "wall_h": float(wall / 3600.0),
                "means": means, "stds": stds, "quantiles": quants,
                "gain_g_std": stds["gain_g"],
                "g_shrink_std_over_prior_std": g_shrink,
                "folder_stats": folder.stats(),
                "ultranest_scalars": scal,
                "diagnostics": diag,
            },
            "wall_total_s": time.perf_counter() - t_all,
            "samples_npz": out_npz.name,
        }
        if "prediction" in cfg:
            out["prediction_recorded_before_run"] = cfg["prediction"]
        try:
            out_json.write_text(json.dumps(out, indent=2))
        except TypeError:  # last-resort: never lose the result to a serialization nit
            out_json.write_text(json.dumps(out, indent=2, default=str))

        done.write_text(json.dumps({
            "run": name, "finished": out["finished"],
            "wall_h": round(out["wall_total_s"] / 3600.0, 3),
            "g_shrink": g_shrink, "logz": out["ns"]["logz"],
        }, indent=2))
        if running.exists():
            running.unlink()
        print(f"[{name}] DONE wall {out['wall_total_s'] / 3600.0:.2f} h  "
              f"logZ {out['ns']['logz']:.2f} +- {out['ns']['logzerr']:.2f}  "
              f"std(g) {stds['gain_g']:.6f}  g-shrink {g_shrink:.4f}  "
              f"ncall {out['ns']['n_like_evals']}  niter {out['ns']['niter']}", flush=True)
        return out

    except BaseException:
        tb = traceback.format_exc()
        failed.write_text(f"run={name}\nfailed={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                          f"elapsed_s={time.perf_counter() - t_all:.0f}\n\n{tb}")
        if running.exists():
            running.unlink()
        print(f"[{name}] FAILED\n{tb}", flush=True)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--nsteps", type=int, default=DEFAULT_NSTEPS,
                    help="SliceSampler nsteps (slice runs only); default 12 = 2*ndim")
    ap.add_argument("--resumable", action="store_true", default=False,
                    help="use a persistent UltraNest log_dir + resume='resume' so a "
                         "killed/crashed run can be restarted from its point store "
                         "instead of from scratch. Default off (committed behavior).")
    args = ap.parse_args()
    run_one(args.run, nsteps=args.nsteps, resumable=args.resumable)


if __name__ == "__main__":
    main()
