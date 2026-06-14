"""Phase-5 unit tests for the nested-sampling benchmark (ns_bench.py).

Run with the repo venv:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_ns_bench.py

Seeded, reduced-live-point tests covering the Phase-5 contract:

  * NS on one BRIGHT spectrum recovers the truth inside the 95% credible interval.
    The likelihood is the EXACT Poisson likelihood of the observed counts, so at
    high counts UltraNest must bracket the generating parameters -- a direct
    correctness check on the reused likelihood + box transform.
  * NS and the IS-refinement likelihood agree on logL at sampled theta to machine
    precision (the EXACT-REUSE check: ns_bench.make_poisson_loglike is literally
    calibrate.poisson_loglik(counts, fold_theta(theta)), the same function the
    Phase-3 IS-refinement calls).
  * JSONL resume-skip logic: load_done_ids returns exactly the ids already
    written, so a resumed run skips them.

All NS runs use a small min_num_live_points + an n_like_evals cap so the suite
stays fast (a few seconds per NS run on the 3-param dev model).
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from sbixcal import ns_bench as NB
from sbixcal import calibrate as C
from sbixcal import models as M
from sbixcal import priors as P
from sbixcal import responses as R
from sbixcal import simulate as S


# --------------------------------------------------------------------------
# import the CLI runner module (scripts/run_ns_benchmark.py) for the
# cross-spectrum --workers parallelism tests
# --------------------------------------------------------------------------

def _load_runner():
    """Import scripts/run_ns_benchmark.py as a module (it is not on the package
    path). Registered in sys.modules under a stable name so multiprocessing's spawn
    workers can re-import the worker function by qualified name."""
    repo = Path(__file__).resolve().parents[1]
    scripts_dir = repo / "scripts"
    # put scripts/ on sys.path so spawned (spawn-start) workers can re-import the
    # runner by the bare name `run_ns_benchmark` to unpickle the worker function.
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    name = "run_ns_benchmark"
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


RUNNER = _load_runner()


BASE = "tbabs_powerlaw"
PARAM_NAMES = M.MODEL_PARAMS[BASE]
PRIOR_CFG = {
    "tbabs_1_nh": {"dist": "uniform", "low": 0.1, "high": 0.3},
    "powerlaw_1_alpha": {"dist": "uniform", "low": 0.5, "high": 2.5},
    "powerlaw_1_norm": {"dist": "loguniform", "low": 1e-4, "high": 1e-2},
}
# bright exposure -> ~10k counts so the Poisson posterior is tight enough to test
# truth recovery cheaply.
EXPOSURE = 6000.0


@pytest.fixture(scope="module")
def bright_spectrum():
    """One BRIGHT dev-Model-A spectrum from a known truth, plus the clean
    exposure-scaled obsconf and a model_counts_fn that folds theta through it."""
    warnings.filterwarnings("ignore")
    oc = R.scale_exposure(R.load_base_obsconf(), EXPOSURE)

    # a single fixed truth comfortably inside the prior box
    truth = np.array([0.2, 1.6, 3.0e-3])  # N_H, Gamma, normPL

    def model_counts_fn(theta_arr):
        return S.fold_theta(BASE, PARAM_NAMES, theta_arr, oc)

    lam = model_counts_fn(truth[None, :])[0]
    rng = np.random.default_rng(123)
    counts = rng.poisson(np.clip(lam, 0.0, None)).astype(np.float64)
    return truth, counts, model_counts_fn


def test_ns_recovers_truth_bright(bright_spectrum):
    """NS 95% interval (5-95% quantiles) brackets the generating truth on a bright
    spectrum (seeded, reduced live points)."""
    truth, counts, model_counts_fn = bright_spectrum
    res = NB.run_ns_one(
        counts, model_counts_fn, PRIOR_CFG, PARAM_NAMES,
        min_num_live_points=80, max_ncalls=40000, dlogz=0.5, seed=0,
    )
    # every parameter's truth inside its [5%, 95%] NS interval
    for j, name in enumerate(PARAM_NAMES):
        lo = res.quantiles[name]["0.05"]
        hi = res.quantiles[name]["0.95"]
        assert lo <= truth[j] <= hi, (
            f"{name}: truth {truth[j]:.4g} outside NS 90% [{lo:.4g}, {hi:.4g}]")
    # sane logZ + positive eval count
    assert np.isfinite(res.logz)
    assert res.n_like_evals > 0


def test_ns_likelihood_is_exact_is_likelihood(bright_spectrum):
    """ns_bench.make_poisson_loglike == calibrate.poisson_loglik(counts,
    fold_theta(theta)) at sampled theta (the exact-reuse check)."""
    truth, counts, model_counts_fn = bright_spectrum
    loglike = NB.make_poisson_loglike(counts, model_counts_fn)

    rng = np.random.default_rng(7)
    # draw a block of theta inside the box and compare both routes
    low, high = P.prior_bounds(PRIOR_CFG, PARAM_NAMES)
    theta = low[None, :] + rng.random((16, len(PARAM_NAMES))) * (high - low)[None, :]

    ll_nb = np.asarray(loglike(theta), dtype=np.float64)  # ns_bench route (vectorized)
    ll_is = C.poisson_loglik(counts, model_counts_fn(theta))  # IS-refinement route
    assert np.allclose(ll_nb, ll_is, rtol=0, atol=1e-9)

    # scalar path agrees too (single theta vector)
    ll_scalar = loglike(theta[0])
    assert np.isclose(ll_scalar, ll_is[0], atol=1e-9)


def test_box_transform_maps_unit_cube_to_prior(bright_spectrum):
    """The unit-cube -> box transform spans exactly the prior bounds."""
    transform = NB.make_box_transform(PRIOR_CFG, PARAM_NAMES)
    low, high = P.prior_bounds(PRIOR_CFG, PARAM_NAMES)
    u = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    out = transform(u)
    assert np.allclose(out[0], low)
    assert np.allclose(out[1], high)
    assert np.allclose(out[2], 0.5 * (low + high))


def test_jsonl_resume_skip(tmp_path):
    """load_done_ids returns exactly the spectrum_ids already written; a resumed
    run would skip them."""
    path = tmp_path / "results.jsonl"
    assert NB.load_done_ids(path) == set()  # missing file -> empty

    NB.append_jsonl(path, {"spectrum_id": "clean|bright|clean|0|0", "x": 1})
    NB.append_jsonl(path, {"spectrum_id": "B1|medium|B1_s0.0003|3|2", "x": 2})
    # a malformed line must be skipped gracefully
    with open(path, "a") as f:
        f.write("not json\n")
    NB.append_jsonl(path, {"spectrum_id": "clean|faint|clean|1|5", "x": 3})

    done = NB.load_done_ids(path)
    assert done == {
        "clean|bright|clean|0|0",
        "B1|medium|B1_s0.0003|3|2",
        "clean|faint|clean|1|5",
    }


def test_quantile_agreement_zero_for_identical(bright_spectrum):
    """Identical NS/NPE quantile dicts -> zero agreement distance."""
    q = {name: {"0.5": 1.0, "0.05": 0.5, "0.95": 1.5} for name in PARAM_NAMES}
    low, high = P.prior_bounds(PRIOR_CFG, PARAM_NAMES)
    agree = NB.quantile_agreement(q, q, PARAM_NAMES, low, high)
    assert agree["mean_abs_norm"] == 0.0


# ==========================================================================
# cross-spectrum --workers parallelism (the harness contract)
#
# These build ONE tiny real checkpoint (2-epoch flow on cheap fake data) and run
# run_ns_benchmark.run_subsample on a 4-spectrum mini-grid with a 2-live-point /
# few-eval NS config, asserting:
#   * --workers 2 produces the SAME SET of row keys as serial (--workers 1),
#   * resume-skip works after deleting one row,
#   * an injected failing spectrum yields a keyed error row and does NOT abort.
# The PARENT is the only writer in both paths; workers return row dicts.
# ==========================================================================

TRAIN_RUN = "ns_bench_test_run"
TEST_LEVEL = "medium"
# tiny exposure -> few counts -> the 2-live-point NS run is cheap.
TEST_EXPOSURE = 1500.0

_NS_TEST_CFG = {
    "name": "ns_bench_test",
    "base_model": "tbabs_powerlaw",
    "seed": 0,
    "priors": dict(PRIOR_CFG),
    "embedding": {"embed_dim": 16, "conv_channels": [16, 32], "kernel_size": 5,
                  "mlp_hidden": 64},
    "flow": {"hidden_features": 24, "num_transforms": 3, "num_bins": 8},
    "train": {"batch_size": 100, "learning_rate": 5e-4, "validation_fraction": 0.1,
              "stop_after_epochs": 100, "max_num_epochs": 2, "show_progress": False},
}


def _fake_train_data(oc, n=800, seed=0):
    """Cheap (theta, x): draw theta from the prior box, fold a crude analytic
    spectrum (102 channels matching the real response), Poisson-noise it. Avoids a
    jaxspec simulation for the throwaway training set; the flow only needs to be a
    valid, loadable density estimator for this harness test."""
    import torch
    g = np.random.default_rng(seed)
    low, high = P.prior_bounds(PRIOR_CFG, PARAM_NAMES)
    theta = g.uniform(low, high, size=(n, len(PARAM_NAMES)))
    base = np.linspace(0.3, 10.0, 102)[None, :]
    nh, gamma, norm = theta[:, [0]], theta[:, [1]], theta[:, [2]]
    rate = norm * 1e4 * base ** (-gamma) * np.exp(-nh * 0.5 / base)
    x = g.poisson(np.clip(rate, 0, None)).astype(np.float32)
    return (torch.as_tensor(theta, dtype=torch.float32),
            torch.as_tensor(x, dtype=torch.float32))


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """Train + save a tiny 2-epoch flow checkpoint to a temp models dir, laid out
    as outputs/models/<TRAIN_RUN>_<level>/ so LevelNS can cold-load it. Returns the
    checkpoint dir."""
    warnings.filterwarnings("ignore")
    from sbixcal import train_npe as tn
    import torch

    oc = R.scale_exposure(R.load_base_obsconf(), TEST_EXPOSURE)
    theta, x = _fake_train_data(oc, n=800, seed=11)
    prior = tn.build_prior(_NS_TEST_CFG["priors"], PARAM_NAMES, device="cpu")
    de, inf, summary = tn.train_one_flow(theta, x, prior, _NS_TEST_CFG,
                                         device="cpu", seed=0)
    de.eval()
    meta = {"exposure_s": TEST_EXPOSURE,
            "median_total_counts": float(np.median(x.sum(1).numpy())),
            "n": int(x.shape[0])}
    ckpt = tmp_path_factory.mktemp("models") / f"{TRAIN_RUN}_{TEST_LEVEL}"
    tn.save_checkpoint(ckpt, de, summary, _NS_TEST_CFG, PARAM_NAMES,
                       int(x.shape[1]), meta)
    return Path(ckpt)


def _mini_cfg():
    """A 4-spectrum mini-grid config with a 2-live-point / few-eval NS setting."""
    return {
        "name": "ns_bench_test",
        "train_run": TRAIN_RUN,
        "response": R.EXAMPLE_NAME,
        "seed": 20260611,
        "ns": {"min_num_live_points": 2, "dlogz": 0.5, "max_ncalls": 600},
        "npe": {"n_samples": 64},
        "subsample": [{"family": "clean", "level": TEST_LEVEL, "n": 4}],
    }


def _patch_runner(monkeypatch, ckpt_dir, out_dir):
    """Point the runner at the temp checkpoint (per level) and a temp results dir,
    so the test never reads/writes outputs/ns_bench/ or outputs/models/."""
    monkeypatch.setattr(RUNNER, "_checkpoint_for_level",
                        lambda train_run, level: Path(ckpt_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(RUNNER, "_out_dir", lambda: out_dir)
    return out_dir / "results.jsonl"


def _row_keys(results_path):
    return NB.load_done_ids(results_path)


def test_workers_match_serial_keyset(tiny_ckpt, tmp_path, monkeypatch):
    """--workers 2 produces the SAME SET of spectrum_id keys as serial on the
    4-spectrum mini-grid (parent-only writes; workers return rows)."""
    cfg = _mini_cfg()

    serial_path = _patch_runner(monkeypatch, tiny_ckpt, tmp_path / "serial")
    n_serial = RUNNER.run_subsample(cfg, workers=1)
    serial_keys = _row_keys(serial_path)
    assert n_serial == 4
    assert len(serial_keys) == 4

    par_path = _patch_runner(monkeypatch, tiny_ckpt, tmp_path / "parallel")
    n_par = RUNNER.run_subsample(cfg, workers=2)
    par_keys = _row_keys(par_path)
    assert n_par == 4
    assert par_keys == serial_keys

    # every parallel row is a real result (no error rows), and the parent wrote
    # exactly one line per spectrum.
    rows = [l for l in par_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 4
    for line in rows:
        import json
        r = json.loads(line)
        assert "error" not in r
        assert "ns" in r and "npe" in r and "agreement" in r


def test_resume_skip_after_deleting_one_row(tiny_ckpt, tmp_path, monkeypatch):
    """After a full --workers 2 run, deleting one results row and rerunning
    recomputes exactly that one spectrum (resume-skip on the rest)."""
    import json
    cfg = _mini_cfg()
    results_path = _patch_runner(monkeypatch, tiny_ckpt, tmp_path / "resume")

    RUNNER.run_subsample(cfg, workers=2)
    keys_full = _row_keys(results_path)
    assert len(keys_full) == 4

    # drop the last row and rerun; only the dropped key should be recomputed.
    lines = [l for l in results_path.read_text().splitlines() if l.strip()]
    dropped = json.loads(lines[-1])["spectrum_id"]
    results_path.write_text("\n".join(lines[:-1]) + "\n")
    assert dropped not in _row_keys(results_path)

    n_new = RUNNER.run_subsample(cfg, workers=2)
    assert n_new == 1                      # exactly the dropped spectrum
    assert _row_keys(results_path) == keys_full  # full key set restored


def test_injected_failure_yields_error_row_and_continues(tiny_ckpt, tmp_path,
                                                         monkeypatch):
    """A failing spectrum (its task points at a non-existent checkpoint, so the
    worker's LevelNS build raises) yields a keyed ERROR ROW and does NOT abort the
    run: the other 3 spectra still complete."""
    import json
    cfg = _mini_cfg()
    results_path = _patch_runner(monkeypatch, tiny_ckpt, tmp_path / "fault")

    real_build = RUNNER.build_tasks

    def build_tasks_with_one_bad(*args, **kwargs):
        tasks = real_build(*args, **kwargs)
        assert tasks, "expected a non-empty task list"
        # sabotage exactly one task: point it at a checkpoint that does not exist
        # so reconstructing LevelNS inside the worker raises -> error row.
        tasks[0] = dict(tasks[0],
                        ckpt_dir=str(tmp_path / "does_not_exist_ckpt"))
        build_tasks_with_one_bad.bad_id = tasks[0]["spectrum_id"]
        return tasks

    monkeypatch.setattr(RUNNER, "build_tasks", build_tasks_with_one_bad)

    n_run = RUNNER.run_subsample(cfg, workers=2)
    assert n_run == 4  # all 4 produced a row (3 results + 1 error), run not aborted

    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 4
    err = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    assert len(err) == 1 and len(ok) == 3
    bad = err[0]
    assert bad["spectrum_id"] == build_tasks_with_one_bad.bad_id
    assert isinstance(bad["error"], str) and bad["error"]  # carries the exc string
    # resume treats the error row as "done" (keyed), so a rerun does not retry it.
    assert bad["spectrum_id"] in _row_keys(results_path)


def test_analyze_skips_error_rows_with_count(tmp_path, monkeypatch):
    """analyze_ns_bench partitions out keyed error rows and reports the count,
    keeping them out of every table."""
    import json

    repo = Path(__file__).resolve().parents[1]
    scripts_dir = repo / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    analyze = importlib.import_module("analyze_ns_bench")

    # build the layout analyze expects: <repo_root>/outputs/ns_bench/results.jsonl
    outputs = tmp_path / "outputs" / "ns_bench"
    outputs.mkdir(parents=True)
    monkeypatch.setattr(analyze, "_repo_root", lambda: tmp_path)

    qd = {n: {f"{q:g}": 0.0 for q in NB.QUANTILES} for n in PARAM_NAMES}
    good = {
        "spectrum_id": "clean|medium|clean|0|0", "family": "clean",
        "level": "medium", "strength_label": "clean", "n_counts": 100,
        "param_names": PARAM_NAMES, "truth": [0.2, 1.6, 3e-3],
        "ns": {"quantiles": qd, "logz": -10.0, "logzerr": 0.1,
               "n_like_evals": 500, "niter": 50, "ess": 40.0, "wall_s": 1.0,
               "n_live": 2},
        "npe": {"quantiles": qd, "sample_wall_s": 0.01, "n_samples": 64},
        "agreement": {"per_param": {}, "mean_abs_norm": 0.0},
    }
    bad = {"spectrum_id": "clean|medium|clean|0|1", "family": "clean",
           "level": "medium", "strength_label": "clean",
           "error": "RuntimeError: boom"}
    with open(outputs / "results.jsonl", "w") as f:
        f.write(json.dumps(good) + "\n")
        f.write(json.dumps(bad) + "\n")

    rc = analyze.main(["--config", str(repo / "configs" / "ns_bench.yaml")])
    assert rc == 0
    summary = json.loads((outputs / "analysis_summary.json").read_text())
    assert summary["n_error_rows"] == 1
    assert summary["n_spectra"] == 1  # error row excluded from the analyzed set
