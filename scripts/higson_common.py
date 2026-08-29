r"""Shared code for the thread-wise bootstrap: job manifest, exact spectrum
reconstruction, thread identification and the Higson (2018) thread-wise
bootstrap over logZ.

Every job reuses the EXACT spectrum/seed/config of a committed shipped run so each
Higson error bar attaches to a real row:

 * gain-null pairs  -> scripts/paired_ns_gain_check.py (medium, exposure 353.4,
   theta rng=default_rng(20260630), Poisson rng=default_rng(1000+i), NS seed=i,
   clean model folded through the nominal response; the gain member folds the SAME
   clean model through gain-shifted data).
 * B1 / clean rows  -> scripts/run_ns_benchmark.py draw_block (config ns_bench.yaml,
   global seed 20260611, block_seed = (seed + 1000*(block_idx+1)) % (2**31-1), NS
   seed = seed + within-block-index). prior_cfg/base_model/exposure/param_names are
   read from the level checkpoint's arch.json (exactly as LevelNS does).

NS settings for every run match the shipped runs: min_num_live_points=400, dlogz=0.5,
max_ncalls=400000. The ONLY change vs the committed runs is log_dir=<isolated dir>
(committed runs used log_dir=None), so UltraNest writes its point store and the
thread structure can be recovered.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = "tbabs_powerlaw_bb"
RESP = "NGC7793_ULX4_PN"
TRAIN_RUN = "train_npe_prod"
GLOBAL_SEED = 20260611          # configs/ns_bench.yaml seed
GAIN_EXPOSURE = 353.4           # medium, matches paired_ns_gain_check.py
GAIN = 1.03
PAIRED_PRIOR_CFG = {
    "tbabs_1_nh":         {"dist": "uniform",    "low": 0.15,   "high": 0.35},
    "powerlaw_1_alpha":   {"dist": "uniform",    "low": 1.0,    "high": 3.0},
    "powerlaw_1_norm":    {"dist": "loguniform", "low": 1.0e-4, "high": 1.0e-2},
    "blackbodyrad_1_kT":  {"dist": "uniform",    "low": 0.3,    "high": 3.0},
    "blackbodyrad_1_norm":{"dist": "loguniform", "low": 1.0e-2, "high": 1.0},
}
NS_KW = dict(min_num_live_points=400, dlogz=0.5, max_ncalls=400000)

# config subsample block order (configs/ns_bench.yaml) -> block_idx
BLOCK_IDX = {
    ("clean", "faint"): 0, ("clean", "medium"): 1, ("clean", "bright"): 2,
    ("B1", "medium"): 3, ("B1", "bright"): 4, ("B4", "medium"): 5, ("B4", "bright"): 6,
}
BLOCK_N = {
    ("clean", "faint"): 25, ("clean", "medium"): 16, ("clean", "bright"): 15,
    ("B1", "medium"): 6, ("B1", "bright"): 6, ("B4", "medium"): 4, ("B4", "bright"): 4,
}
LEVEL_EXPOSURE = {"faint": 35.3, "medium": 353.4, "bright": 3534.0}


# ==========================================================================
# JOB MANIFEST (13 runs, hard cap 16)
# ==========================================================================
# CAVEAT on the gain pairs (found 2026-07-23, input audit):
#   scripts/paired_ns_gain_check.py has mtime 2026-07-01 20:04, AFTER it wrote
#   outputs/ns_bench/paired_gain_check.jsonl (mtime 19:00). The current script does
#   NOT reproduce the committed paired-gain counts (i=0 gives 1403 vs committed 468;
#   ratios are non-uniform, so the theta draws themselves differ). No param-order
#   permutation, N, exposure, or sampling-layout hypothesis reproduces them, and no
#   theta/spectrum array was saved. So the EXACT committed paired-gain spectra are
#   unrecoverable. The block rows (B1/clean, from run_ns_benchmark.py via frozen
#   arch.json) DO reconstruct byte-exactly and ARE attached.
#   -> The gain-pair runs below are VALID medium-exposure clean+3%-gain draws from
#      the current script (structurally identical experiment), used to measure the
#      per-run Higson sampling sigma (the floor), which depends on n_live=400 and the
#      count level, not the exact theta. They are NOT attached to specific committed
#      pair indices; committed_logz is None and no reproduction delta is reported.
# Selection: 5 pairs spanning current-script clean counts to cover the floor-vs-count
#   trend: i=8 (90), i=6 (526), i=0 (1403), i=9 (2128), i=10 (3659).
# B1 medium idx2 (n=265, committed logz -303.50); B1 bright idx3 (n=1420, -1615.87);
# clean bright idx9 (n=1675, -250.97). These three reconstruct byte-exactly.
GAIN_PAIR_IDS = [8, 6, 0, 9, 10]

def build_jobs():
    jobs = []
    for i in GAIN_PAIR_IDS:
        jobs.append({"job_id": f"gain_pair{i}_clean", "kind": "gain_clean", "pair_i": i,
                     "ns_seed": i, "committed_logz": None,
                     "attaches_to": "medium-exposure clean draw (count regime, NOT exact committed row)"})
        jobs.append({"job_id": f"gain_pair{i}_gain", "kind": "gain_gain", "pair_i": i,
                     "ns_seed": i, "committed_logz": None,
                     "attaches_to": "medium-exposure 3%-gain draw (count regime, NOT exact committed row)"})
    jobs.append({"job_id": "B1_medium_idx2", "kind": "block", "family": "B1", "level": "medium",
                 "strength": 3.0e-4, "block_i": 2, "ns_seed": GLOBAL_SEED + 2,
                 "committed_logz": -303.50, "attaches_to": "B1|medium|B1_s0.0003|3|2"})
    jobs.append({"job_id": "B1_bright_idx3", "kind": "block", "family": "B1", "level": "bright",
                 "strength": 3.0e-4, "block_i": 3, "ns_seed": GLOBAL_SEED + 3,
                 "committed_logz": -1615.87, "attaches_to": "B1|bright|B1_s0.0003|4|3"})
    jobs.append({"job_id": "clean_bright_idx9", "kind": "block", "family": "clean", "level": "bright",
                 "strength": None, "block_i": 9, "ns_seed": GLOBAL_SEED + 9,
                 "committed_logz": -250.97, "attaches_to": "clean|bright|clean|2|9"})
    return jobs


# committed values for reproduction-delta reporting
COMMITTED_GAIN = {  # pair_i -> (logz_clean, logz_gain, d_paired, counts_clean, counts_gain)
    0: (-223.81712408779174, -222.25534126908562, 1.5617828187061207, 468, 468),
    1: (-227.8065161504817, -227.50495281548433, 0.3015633349973825, 505, 515),
    5: (-348.99263433502176, -352.76456184069156, -3.771927505669794, 6187, 6389),
    8: (-179.71089427507792, -169.3059568102523, 10.404937464825622, 368, 330),
    11: (-296.2787908309418, -297.339467532389, -1.0606767014472211, 1836, 1850),
}
# expected total counts for block jobs (from committed results.jsonl n_counts)
COMMITTED_BLOCK_COUNTS = {"B1_medium_idx2": 265, "B1_bright_idx3": 1420, "clean_bright_idx9": 1675}


# ==========================================================================
# spectrum reconstruction
# ==========================================================================

def _obsconf(exposure):
    from sbixcal import responses as R
    return R.scale_exposure(R.load_base_obsconf(RESP), exposure)


def reconstruct_gain_pair(pair_i):
    """Reconstruct (data_clean, data_gain, model_counts_fn, prior_cfg, param_names)
    for paired-gain index pair_i, byte-identically to scripts/paired_ns_gain_check.py."""
    from sbixcal import simulate as S, models as M, priors as P
    from sbixcal import responses as R
    param_order = M.MODEL_PARAMS[BASE_MODEL]
    clean_oc = _obsconf(GAIN_EXPOSURE)
    gain_oc = R.gain_shift_obsconf(clean_oc, GAIN)
    rng = np.random.default_rng(20260630)
    samples = P.sample_prior(PAIRED_PRIOR_CFG, param_order, 12, rng)
    theta = np.stack([np.asarray(samples[p]) for p in param_order], axis=1)
    th = theta[pair_i:pair_i + 1]
    lam_clean = np.asarray(S.fold_theta(BASE_MODEL, param_order, th, clean_oc))[0]
    lam_gain = np.asarray(S.fold_theta(BASE_MODEL, param_order, th, gain_oc))[0]
    data_clean = np.random.default_rng(1000 + pair_i).poisson(np.maximum(lam_clean, 0)).astype(float)
    data_gain = np.random.default_rng(1000 + pair_i).poisson(np.maximum(lam_gain, 0)).astype(float)

    def model_counts_fn(theta_arr):
        return S.fold_theta(BASE_MODEL, param_order, theta_arr, clean_oc)
    return data_clean, data_gain, model_counts_fn, PAIRED_PRIOR_CFG, param_order


def _arch(level):
    with open(ROOT / "outputs" / "models" / f"{TRAIN_RUN}_{level}" / "arch.json") as f:
        return json.load(f)


def reconstruct_block(family, level, block_i, strength):
    """Reconstruct (counts, model_counts_fn, prior_cfg, param_names) for one config
    subsample spectrum, byte-identically to scripts/run_ns_benchmark.py draw_block.
    prior_cfg/base_model/exposure/param_names read from the level checkpoint arch.json."""
    from sbixcal import simulate as _sim, misspec as _MS
    arch = _arch(level)
    prior_cfg = arch["prior_cfg"]
    base_model = arch["base_model"]
    param_names = list(arch["param_names"])
    exposure = float(arch["exposure_s"])
    obsconf = _obsconf(exposure)
    bidx = BLOCK_IDX[(family, level)]
    n = BLOCK_N[(family, level)]
    block_seed = (GLOBAL_SEED + 1000 * (bidx + 1)) % (2**31 - 1)

    if family == "clean":
        rng = np.random.default_rng(block_seed)
        _, x_exp, _ = _sim.simulate_spectra(base_model, prior_cfg, obsconf, n, rng,
                                            apply_poisson=False, seed_for_fakeit=block_seed)
        rng_p = np.random.default_rng(block_seed + 1)
        x = rng_p.poisson(np.clip(x_exp, 0.0, None)).astype(np.float64)
    else:
        x, _, _ = _MS.simulate_misspec_population(base_model, prior_cfg, obsconf,
                                                  family, float(strength), n, seed=block_seed, fixed={})
        x = np.asarray(x, dtype=np.float64)
    counts = x[block_i]

    def model_counts_fn(theta_arr):
        return _sim.fold_theta(base_model, param_names, theta_arr, obsconf)
    return counts, model_counts_fn, prior_cfg, param_names


# ==========================================================================
# thread identification + Higson thread-wise bootstrap
# ==========================================================================

def logz_from_birth(logl, birth):
    """Combined-run logZ from per-point (logl, birth_logl) via the standard
    nlive/volume estimator. nlive_i = (#births < logl_i) - (#deaths < logl_i).
    Returns (logZ, nlive_array) with arrays sorted by logl."""
    logl = np.asarray(logl, float)
    birth = np.asarray(birth, float)
    order = np.argsort(logl, kind="mergesort")
    logl_s = logl[order]
    birth_s = birth[order]
    n = len(logl_s)
    birth_sorted = np.sort(birth_s)
    n_born = np.searchsorted(birth_sorted, logl_s, side="left")   # births strictly < logl_i
    n_dead = np.arange(n)                                         # deaths strictly < logl_i (sorted)
    nlive = np.maximum(n_born - n_dead, 1)
    t = nlive / (nlive + 1.0)
    logX = np.concatenate([[0.0], np.cumsum(np.log(t))])
    dX = np.exp(logX[:-1]) - np.exp(logX[1:])
    dX = np.maximum(dX, 0.0)
    m = dX > 0
    logw = np.log(dX[m]) + logl_s[m]
    logZ = float(np.logaddexp.reduce(logw)) if m.any() else float("-inf")
    return logZ, nlive


def reconstruct_tree(run_dir):
    """Recover the true per-point birth contour of an UltraNest run
    by replaying its refill-batch consumption. Do NOT use column 0 (`Lmin`) of
    results/points.hdf5 as a birth contour; see the module docstring.

    UltraNest stores a whole refill batch stamped with the one Lmin current when the
    batch was drawn (integrator.py ~L1935), then consumes it FIFO over the following
    iterations, keeping a member only while ``likes[ib] > Lmin`` at the CURRENT
    contour (~L1941). So: walk the ordered death sequence in chains/run.txt and, for
    every death that was replaced, pop the next unused point-store row (creation
    order) whose logl clears that death contour. The replacement inherits the dying
    point's thread, which yields exactly n_live threads for a static run.

    Returns a dict with the reconstructed run (logl / birth / thread_id, all sorted
    by logl, i.e. in death order) plus the validation diagnostics.
    """
    import h5py
    run_dir = Path(run_dir)
    run = np.loadtxt(run_dir / "chains" / "run.txt", skiprows=1)
    nlive_ref = np.asarray(run[:, 3], float)
    logl_dead = np.asarray(run[:, 4], float)
    n_dead = len(logl_dead)
    if not np.all(np.diff(logl_dead) > 0):
        raise ValueError(f"{run_dir.name}: death sequence in run.txt is not strictly increasing")

    with h5py.File(str(run_dir / "results" / "points.hdf5"), "r") as f:
        pts = f["points"][:]
    lmin_store = np.asarray(pts[:, 0], float)
    logl_store = np.asarray(pts[:, 1], float)

    # the initial live points are the store prefix stamped Lmin = -inf
    is_init = ~np.isfinite(lmin_store)
    n_init = int(is_init.sum())
    if not (is_init[:n_init].all() and not is_init[n_init:].any()):
        raise ValueError(f"{run_dir.name}: initial live points are not a store prefix")
    if n_init != int(nlive_ref[0]):
        raise ValueError(f"{run_dir.name}: n_init={n_init} != run.txt nlive[0]={nlive_ref[0]}")

    # a static run replaces every death until the live set starts draining, i.e.
    # exactly (n_dead - n_init) replacements; cross-checked against the nlive column.
    n_repl = n_dead - n_init
    n_repl_nlive = int((nlive_ref[1:] == nlive_ref[:-1]).sum())
    if n_repl != n_repl_nlive:
        raise ValueError(f"{run_dir.name}: replacement count mismatch "
                         f"{n_repl} (n_dead-n_init) vs {n_repl_nlive} (nlive column)")

    live = {}                       # logl -> [(thread_id, birth_contour), ...]
    for tid in range(n_init):
        live.setdefault(float(logl_store[tid]), []).append((tid, -np.inf))

    out_logl = np.empty(n_dead, float)
    out_birth = np.empty(n_dead, float)
    out_tid = np.empty(n_dead, np.int64)
    used = np.zeros(len(logl_store), bool)
    used[:n_init] = True
    ptr = n_init                    # forward-only pointer into the refill store
    n_skipped = 0
    for i in range(n_dead):
        d = float(logl_dead[i])
        stack = live.get(d)
        if not stack:
            raise ValueError(f"{run_dir.name}: death logl {d!r} not in the live set at iter {i}")
        tid, b = stack.pop()
        if not stack:
            del live[d]
        out_logl[i] = d
        out_birth[i] = b
        out_tid[i] = tid
        if i < n_repl:
            while ptr < len(logl_store) and (used[ptr] or not (logl_store[ptr] > d)):
                if not used[ptr]:
                    n_skipped += 1      # batch member stale at the current contour
                used[ptr] = True
                ptr += 1
            if ptr >= len(logl_store):
                raise ValueError(f"{run_dir.name}: point store exhausted at iter {i}")
            used[ptr] = True
            live.setdefault(float(logl_store[ptr]), []).append((tid, d))
            ptr += 1

    if not (out_logl > out_birth).all():
        raise ValueError(f"{run_dir.name}: a point is not above its birth contour")
    _, nlive_rec = logz_from_birth(out_logl, out_birth)
    n_tail = int(len(logl_store) - n_init - (used[n_init:]).sum())
    return {
        "logl": out_logl,
        "birth": out_birth,
        "thread_id": out_tid,
        "n_threads": n_init,
        "nlive": nlive_rec,
        "nlive_ref": nlive_ref,
        "nlive_exact": bool(np.array_equal(nlive_rec, nlive_ref.astype(nlive_rec.dtype))),
        "nlive_n_mismatch": int((nlive_rec != nlive_ref).sum()),
        "nlive_max_abs_diff": float(np.abs(nlive_rec - nlive_ref).max()),
        "n_init": n_init,
        "n_dead": n_dead,
        "n_store": int(len(logl_store)),
        "n_store_unused": int(n_skipped + n_tail),
        "n_store_skipped_stale": int(n_skipped),
        "n_store_tail": n_tail,
        "lmin_store": lmin_store,
        "logl_store": logl_store,
    }


def assign_threads(logl, birth):
    """Greedy stack assignment of points to threads from (logl, birth). A point born
    at contour b continues the thread whose last point died at exactly b; else it
    starts a new thread (initial live points born at birth=-inf each start one).
    Returns (order, logl_s, birth_s, thread_labels, thread_start) sorted by logl.

    NOTE (2026-08-14): feed this the output of ``reconstruct_tree()``. Feeding it the
    raw ``points.hdf5`` Lmin column produces thousands of phantom single-point
    threads (that was bug 1)."""
    from collections import defaultdict
    logl = np.asarray(logl, float)
    birth = np.asarray(birth, float)
    order = np.argsort(logl, kind="mergesort")
    logl_s = logl[order]
    birth_s = birth[order]
    n = len(logl_s)
    labels = np.full(n, -1, dtype=np.int64)
    ends = defaultdict(list)        # end-contour -> stack of thread ids waiting
    thread_start = []
    next_tid = 0
    for i in range(n):
        b = float(birth_s[i])
        waiting = ends.get(b)
        if waiting:
            tid = waiting.pop()
        else:
            tid = next_tid
            next_tid += 1
            thread_start.append(b)
        labels[i] = tid
        ends[float(logl_s[i])].append(tid)
    return order, logl_s, birth_s, labels, np.asarray(thread_start, float)


def higson_bootstrap(logl, birth, n_resamples=400, seed=0, labels=None):
    """Higson (2018) thread-wise bootstrap std on logZ.

    Identify threads, then resample ALL n threads globally WITH REPLACEMENT,
    recombine, recompute logZ per resample. Returns a dict with the point-estimate
    logZ, the resample std (the sampling-error floor), the resample logZ summary and
    the thread count.

    The resample is global, not stratified within start-contour
    groups. Higson, Handley, Hobson & Lasenby 2018 (arXiv:1703.09701) Algorithm 2 is
    global ("create a list of n threads by sampling x with replacement"); nestcheck's
    only stratified option (`ninit_sep`) uses at most 2 strata, never one stratum per
    distinct start contour. On the phantom-Lmin tree the old scheme froze 10-31% of
    the points at zero variance and suppressed sigma. For a static run reconstructed
    with ``reconstruct_tree`` every thread starts at -inf, so there is a single group
    and the two schemes are numerically identical; the fix is about correctness.

    ``labels`` optionally supplies the per-point thread id (e.g. ``thread_id`` from
    ``reconstruct_tree``, already in logl order). Default None recomputes them with
    ``assign_threads``."""
    logl = np.asarray(logl, float)
    birth = np.asarray(birth, float)
    if labels is None:
        order, logl_s, birth_s, labels, thread_start = assign_threads(logl, birth)
    else:
        order = np.argsort(logl, kind="mergesort")
        logl_s = logl[order]
        birth_s = birth[order]
        labels = np.asarray(labels)[order]
        # thread start contour = birth of each thread's lowest-logl point
        uniq = np.unique(labels)
        remap = np.zeros(int(uniq.max()) + 1, np.int64)
        remap[uniq] = np.arange(len(uniq))
        labels = remap[labels]
        thread_start = np.full(len(uniq), np.nan)
        for t in range(len(uniq)):
            thread_start[t] = birth_s[labels == t][0]
    nthreads = int(labels.max()) + 1
    # points per thread
    thread_logl = [logl_s[labels == t] for t in range(nthreads)]
    thread_birth = [birth_s[labels == t] for t in range(nthreads)]
    n_start_groups = len(set(float(s) for s in thread_start))

    logZ0, nlive_direct = logz_from_birth(logl_s, birth_s)

    rng = np.random.default_rng(seed)
    resamples = np.empty(n_resamples, float)
    for r in range(n_resamples):
        # Algorithm 2: n threads drawn with replacement from all n threads
        sel = rng.integers(0, nthreads, nthreads)
        lr = np.concatenate([thread_logl[t] for t in sel])
        br = np.concatenate([thread_birth[t] for t in sel])
        resamples[r], _ = logz_from_birth(lr, br)
    return {
        "logZ_reconstructed": logZ0,
        "higson_sigma": float(resamples.std(ddof=1)),
        "higson_logZ_mean": float(resamples.mean()),
        "higson_logZ_p16_p84": [float(np.percentile(resamples, 16)),
                                 float(np.percentile(resamples, 84))],
        "n_threads": nthreads,
        "n_start_groups": n_start_groups,
        "n_points": int(len(logl_s)),
        "n_resamples": int(n_resamples),
    }
