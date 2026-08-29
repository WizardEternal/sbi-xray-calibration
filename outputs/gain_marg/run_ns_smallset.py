"""Small-set driver: gain-marginalized NS cross-check on 6 medium + 4 bright
spectra, one detached process per spectrum, resumable via .done markers, with a
MANIFEST and a merge/analysis step.

  * seeded spectrum selection      -> select_jobs()
  * per-spectrum worker (full NS)   -> `--worker ...`
  * detached process scheduler      -> `--launch [--max-concurrent N]`
  * dry-run (plan only, no launch)  -> `--dry-run`
  * merge/analysis                  -> `--merge`

`--dry-run` prints the plan, writes the MANIFEST, and spawns nothing. The full
6-param NS budget is min_live=400, dlogz=0.5, uncapped ncalls (the reference
posterior); each spectrum costs roughly 10-70 min depending on counts and
machine load.

Bright dependency: the bright jobs need outputs/gain_marg/model_bright (the bright
gain-marg NPE flow). If it is absent the bright workers write an error marker and
the driver reports it; medium jobs are independent.

Run examples (repo venv, from repo root):
    .venv\\Scripts\\python.exe outputs\\gain_marg\\run_ns_smallset.py --dry-run
    .venv\\Scripts\\python.exe outputs\\gain_marg\\run_ns_smallset.py --launch --max-concurrent 3
    .venv\\Scripts\\python.exe outputs\\gain_marg\\run_ns_smallset.py --merge
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_gainmarg as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "outputs" / "gain_marg" / "ns_smallset"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
THIS = str(Path(__file__).resolve())

# selection config -------------------------------------------------------------
SELECT_SEED = 20260723
N_MEDIUM = 6
N_BRIGHT = 4
N_POP = 500                       # eval_gainmarg population size per (level,strength)
SEED_EVAL_BASE = 20260611 + 40000  # eval_gainmarg SEED + 40000
# clean control (strength 0) reproduction test by default; flip STRENGTH to 3.0
# (injected g=1.03) to test the marginalization under misspecification.
STRENGTH = 0.0

LEVELS = {
    "medium": {"exposure_s": G.EXPOSURE_MEDIUM, "model_dir": ROOT / "outputs" / "gain_marg" / "model_medium"},
    "bright": {"exposure_s": G.EXPOSURE_BRIGHT, "model_dir": ROOT / "outputs" / "gain_marg" / "model_bright"},
}

# full reference-NS budget -----------------------------------------------------
NS_MIN_LIVE = 400
NS_DLOGZ = 0.5
NS_NDRAW_MIN = 2000
NPE_SAMPLES = 4000


def _eval_seed(strength: float) -> int:
    return SEED_EVAL_BASE + int(strength * 10)


def select_jobs():
    """Deterministic (level, strength, idx) job list. Seeded index choice within
    each level's eval_gainmarg population; medium and bright drawn independently."""
    rng = np.random.default_rng(SELECT_SEED)
    med_idx = sorted(rng.choice(N_POP, size=N_MEDIUM, replace=False).tolist())
    bri_idx = sorted(rng.choice(N_POP, size=N_BRIGHT, replace=False).tolist())
    jobs = []
    for idx in med_idx:
        jobs.append({"level": "medium", "strength": STRENGTH, "idx": int(idx)})
    for idx in bri_idx:
        jobs.append({"level": "bright", "strength": STRENGTH, "idx": int(idx)})
    for j in jobs:
        j["spectrum_id"] = f"{j['level']}_s{j['strength']:g}_i{j['idx']}"
    return jobs


def make_spectrum(level: str, strength: float, idx: int):
    """Regenerate ONE spectrum deterministically (eval_gainmarg B4 convention)."""
    from sbixcal import responses as R
    from sbixcal.misspec import simulate_misspec_population
    oc = R.scale_exposure(R.load_base_obsconf(G.RESP), LEVELS[level]["exposure_s"])
    x, th, present = simulate_misspec_population(
        G.BASE_MODEL, G.PHYS_PRIORS, oc, "B4", float(strength), N_POP, _eval_seed(strength))
    assert present == G.PHYS_ORDER, present
    return np.asarray(x)[idx].astype(np.float64), np.asarray(th)[idx].astype(np.float64)


# ---------------------------------------------------------------------------
# worker: one spectrum, full NS + NPE + comparison
# ---------------------------------------------------------------------------
def worker(level, strength, idx, seed=0):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sid = f"{level}_s{strength:g}_i{idx}"
    res_path = OUTDIR / f"{sid}.json"
    done_path = OUTDIR / f"{sid}.done"
    err_path = OUTDIR / f"{sid}.error"
    model_dir = LEVELS[level]["model_dir"]
    t_all = time.perf_counter()
    try:
        if not model_dir.exists():
            raise FileNotFoundError(f"NPE flow missing: {model_dir} (train it before the {level} jobs)")
        counts, truth = make_spectrum(level, strength, idx)
        folder = G.GainFolder(LEVELS[level]["exposure_s"])
        folder.warmup()
        res = G.run_ns_gainmarg(counts, exposure_s=LEVELS[level]["exposure_s"],
                                min_num_live_points=NS_MIN_LIVE, max_ncalls=None,
                                dlogz=NS_DLOGZ, ndraw_min=NS_NDRAW_MIN, seed=seed,
                                folder=folder, warmup=False)
        npe = G.sample_npe(model_dir, counts, n_samples=NPE_SAMPLES, seed=seed,
                           reject_outside_prior=False)
        cmp = G.compare_ns_npe(res.samples, npe, run_c2st=True)
        out = {
            "spectrum_id": sid, "level": level, "strength": strength, "idx": idx,
            "total_counts": int(counts.sum()), "truth_phys": truth.tolist(),
            "ns": {"logz": res.logz, "logzerr": res.logzerr,
                   "n_like_evals": res.n_like_evals, "niter": res.niter,
                   "wall_s": res.wall_s, "means": res.means, "stds": res.stds,
                   "quantiles": res.quantiles, "folder_stats": res.folder_stats},
            "npe": {"n_samples": int(npe.shape[0]),
                    "means": {p: float(npe[:, j].mean()) for j, p in enumerate(G.PARAM_ORDER)},
                    "stds": {p: float(npe[:, j].std(ddof=1)) for j, p in enumerate(G.PARAM_ORDER)}},
            "comparison_ns_vs_npe": cmp,
            "wall_total_s": time.perf_counter() - t_all,
        }
        res_path.write_text(json.dumps(out, indent=2))
        done_path.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        if err_path.exists():
            err_path.unlink()
        print(f"[worker done] {sid} wall {out['wall_total_s']:.0f}s "
              f"max|meandiff| {cmp['max_abs_mean_diff_in_ns_std']:.2f} sigma "
              f"C2ST {cmp.get('c2st_accuracy', 'NA')}", flush=True)
    except Exception as e:  # persist the failure so the driver can report it
        err_path.write_text(f"{type(e).__name__}: {e}")
        print(f"[worker ERROR] {sid}: {e}", flush=True)
        raise


# ---------------------------------------------------------------------------
# scheduler + MANIFEST
# ---------------------------------------------------------------------------
def _status(sid):
    if (OUTDIR / f"{sid}.done").exists():
        return "done"
    if (OUTDIR / f"{sid}.error").exists():
        return "error"
    return "pending"


def write_manifest(jobs, extra=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    man = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "select_seed": SELECT_SEED, "strength": STRENGTH,
        "ns_budget": {"min_live": NS_MIN_LIVE, "dlogz": NS_DLOGZ,
                      "ndraw_min": NS_NDRAW_MIN, "max_ncalls": None,
                      "n_gain_bins": G.N_GAIN_BINS, "npe_samples": NPE_SAMPLES},
        "levels": {k: {"exposure_s": v["exposure_s"],
                          "model_dir": v["model_dir"].relative_to(ROOT).as_posix(),
                       "model_exists": v["model_dir"].exists()} for k, v in LEVELS.items()},
        "jobs": [{**j, "status": _status(j["spectrum_id"])} for j in jobs],
    }
    if extra:
        man.update(extra)
    (OUTDIR / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    return man


def worker_cmd(job):
    return [PY, "-u", THIS, "--worker", "--level", job["level"],
            "--strength", str(job["strength"]), "--idx", str(job["idx"])]


def launch(max_concurrent=3, dry_run=False, poll_s=20):
    jobs = select_jobs()
    man = write_manifest(jobs)
    pending = [j for j in jobs if _status(j["spectrum_id"]) == "pending"]
    print(f"[driver] {len(jobs)} jobs, {len(pending)} pending, "
          f"max_concurrent={max_concurrent}")
    print(f"[driver] MANIFEST -> {OUTDIR / 'MANIFEST.json'}")
    for j in jobs:
        tag = "" if LEVELS[j["level"]]["model_dir"].exists() else "  [MODEL MISSING]"
        print(f"    {j['spectrum_id']:24s} status={_status(j['spectrum_id'])}{tag}")
    print("\n[driver] exact per-worker command (one detached process each):")
    print("    " + " ".join(worker_cmd(jobs[0])))
    if dry_run:
        print("\n[dry-run] no processes spawned. To launch:")
        print(f"    {PY} {THIS} --launch --max-concurrent {max_concurrent}")
        return man

    running = {}  # sid -> Popen
    OUTDIR.mkdir(parents=True, exist_ok=True)
    queue = list(pending)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
    while queue or running:
        # reap
        for sid in list(running):
            if running[sid].poll() is not None:
                running.pop(sid)
        # fill slots
        while queue and len(running) < max_concurrent:
            j = queue.pop(0)
            sid = j["spectrum_id"]
            if _status(sid) == "done":
                continue
            log = open(OUTDIR / f"{sid}.log", "w")
            p = subprocess.Popen(worker_cmd(j), stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), creationflags=flags)
            running[sid] = p
            print(f"[driver] launched {sid} (pid {p.pid})", flush=True)
        write_manifest(jobs)
        if not queue and not running:
            break
        time.sleep(poll_s)
    man = write_manifest(jobs, extra={"finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print("[driver] all jobs settled. Statuses:",
          {j["spectrum_id"]: _status(j["spectrum_id"]) for j in jobs})
    return man


def merge():
    jobs = select_jobs()
    rows, summary = [], {}
    for j in jobs:
        sid = j["spectrum_id"]
        p = OUTDIR / f"{sid}.json"
        if not p.exists():
            summary[sid] = _status(sid)
            continue
        d = json.loads(p.read_text())
        c = d["comparison_ns_vs_npe"]
        rows.append({
            "spectrum_id": sid, "level": d["level"], "counts": d["total_counts"],
            "ns_wall_s": d["ns"]["wall_s"], "ns_evals": d["ns"]["n_like_evals"],
            "max_abs_mean_diff_in_ns_std": c["max_abs_mean_diff_in_ns_std"],
            "mean_overlap68_iou": c["mean_overlap68_iou"],
            "c2st": c.get("c2st_accuracy"),
        })
        summary[sid] = "done"
    out = {"n_done": len(rows), "rows": rows,
           "status": summary,
           "aggregate": {
               "max_abs_mean_diff_in_ns_std_worst": float(np.max([r["max_abs_mean_diff_in_ns_std"] for r in rows])) if rows else None,
               "mean_overlap68_iou_mean": float(np.mean([r["mean_overlap68_iou"] for r in rows])) if rows else None,
               "c2st_mean": float(np.mean([r["c2st"] for r in rows if r["c2st"] is not None])) if rows else None,
           }}
    (OUTDIR / "SUMMARY.json").write_text(json.dumps(out, indent=2))
    print(f"[merge] {len(rows)} done -> {OUTDIR / 'SUMMARY.json'}")
    for r in rows:
        print(f"  {r['spectrum_id']:24s} counts {r['counts']:6d} wall {r['ns_wall_s']:6.0f}s "
              f"evals {r['ns_evals']:6d}  max|meandiff| {r['max_abs_mean_diff_in_ns_std']:.2f}  "
              f"68IoU {r['mean_overlap68_iou']:.2f}  C2ST {r['c2st']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--level", choices=list(LEVELS))
    ap.add_argument("--strength", type=float, default=STRENGTH)
    ap.add_argument("--idx", type=int)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--max-concurrent", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        assert args.level is not None and args.idx is not None, "--worker needs --level and --idx"
        worker(args.level, args.strength, args.idx)
    elif args.merge:
        merge()
    elif args.launch or args.dry_run:
        launch(max_concurrent=args.max_concurrent, dry_run=args.dry_run)
    else:
        ap.error("pick one of --worker / --launch / --dry-run / --merge")


if __name__ == "__main__":
    main()
