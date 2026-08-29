"""Paired common-random-number evaluation of the gain-shift photon-index bias,
written independently of eval_gainmarg_paired*.py, which it does not import.
Only the repo's forward fold (fold_theta), the response machinery and the flow
loader (load_posterior) are reused.

It differs from eval_gainmarg_paired_bright.py in two ways that matter when the
numbers are compared: the point estimate is the posterior MEDIAN rather than the
mean, and N_SAMP is 3000 rather than 1000. The estimand is the same, the paired
(gain minus clean) Gamma offset over N_THETA=500 common-parameter pairs.

One (level, flow) combination per process. Per-case checkpoint -> resumable.
"""
import sys, os, json, argparse, time
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
from sbixcal import responses as R, simulate as S, priors as P, train_npe as TN

OUT = os.path.join(REPO, "outputs", "gain_marg", "seed_runs")

BASE_SEED = 20260724
GAIN = 1.03
N_THETA = 500
N_SAMP = 3000

PRIOR5 = {
 "tbabs_1_nh": {"dist":"uniform","low":0.15,"high":0.35},
 "powerlaw_1_alpha": {"dist":"uniform","low":1.0,"high":3.0},
 "powerlaw_1_norm": {"dist":"loguniform","low":1.0e-4,"high":1.0e-2},
 "blackbodyrad_1_kT": {"dist":"uniform","low":0.3,"high":3.0},
 "blackbodyrad_1_norm": {"dist":"loguniform","low":1.0e-2,"high":1.0},
}
ORDER5 = ["tbabs_1_nh","powerlaw_1_alpha","powerlaw_1_norm","blackbodyrad_1_kT","blackbodyrad_1_norm"]
EXPO = {"medium":353.4, "bright":3534.0}
GAMMA_IDX = 1   # powerlaw_1_alpha in both 5- and 6-param order
NORM_IDX  = 2   # powerlaw_1_norm
G_IDX     = 5   # gain_g (6-param flow only)

def prior_bounds(n_params):
    lo = np.array([PRIOR5[p]["low"] for p in ORDER5], float)
    hi = np.array([PRIOR5[p]["high"] for p in ORDER5], float)
    if n_params == 6:
        lo = np.append(lo, 0.95); hi = np.append(hi, 1.05)
    return lo, hi

def sample_post(post, x_np, n_params, lo, hi, seed):
    """Sample N_SAMP with rejection to prior box + clip (the detect.py convention).
    Same torch seed used by caller for clean/gain of a theta, i.e. common random
    numbers on the sampler."""
    xt = torch.as_tensor(np.asarray(x_np, np.float32))
    torch.manual_seed(seed)
    try:
        s = post.sample((N_SAMP,), x=xt, show_progress_bars=False,
                        reject_outside_prior=True, max_sampling_time=25.0)
    except (RuntimeError, ValueError, TypeError):
        s = post.sample((N_SAMP,), x=xt, show_progress_bars=False,
                        reject_outside_prior=False)
    s = s.detach().cpu().numpy().astype(np.float64)
    if s.shape[0] < N_SAMP:  # rejection timed out -> top up unrejected
        extra = post.sample((N_SAMP - s.shape[0],), x=xt, show_progress_bars=False,
                            reject_outside_prior=False).detach().cpu().numpy().astype(np.float64)
        s = np.vstack([s, extra])
    s = np.clip(s, lo[None,:], hi[None,:])   # clip defense
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["medium","bright"])
    ap.add_argument("--flow", required=True, choices=["fixed","gainmarg"])
    args = ap.parse_args()
    level, flow = args.level, args.flow
    tag = f"{flow}_{level}"
    ckpt = os.path.join(OUT, f"cases_{tag}.npz")

    # --- responses ---
    base = R.load_base_obsconf("NGC7793_ULX4_PN")
    oc_c = R.scale_exposure(base, EXPO[level])
    oc_g = R.gain_shift_obsconf(oc_c, GAIN)

    # --- theta (5-param, from training prior; seed fixed) ---
    rng = np.random.default_rng(BASE_SEED)
    samp = P.sample_prior(PRIOR5, ORDER5, N_THETA, rng)
    theta = np.stack([samp[p] for p in ORDER5], axis=1)  # (N,5)

    # --- fold (noiseless expected counts), batched ---
    lam_c = np.asarray(S.fold_theta("tbabs_powerlaw_bb", ORDER5, theta, oc_c), float)
    lam_g = np.asarray(S.fold_theta("tbabs_powerlaw_bb", ORDER5, theta, oc_g), float)

    # --- CRN Poisson realizations: shared per-theta seed for clean & gain ---
    pois_seed = BASE_SEED + 100000
    x_c = np.zeros_like(lam_c); x_g = np.zeros_like(lam_g)
    for i in range(N_THETA):
        x_c[i] = np.random.default_rng(pois_seed + i).poisson(np.clip(lam_c[i],0,None))
        x_g[i] = np.random.default_rng(pois_seed + i).poisson(np.clip(lam_g[i],0,None))

    # --- flow ---
    mdir = {"fixed":{"medium":"outputs/models/train_npe_prod_medium",
                      "bright":"outputs/models/train_npe_prod_bright"},
            "gainmarg":{"medium":"outputs/gain_marg/model_medium",
                         "bright":"outputs/gain_marg/model_bright"}}[flow][level]
    post, info = TN.load_posterior(os.path.join(REPO, mdir))
    n_params = len(info["param_names"])
    lo, hi = prior_bounds(n_params)

    # per-case result arrays
    med_g_c = np.full(N_THETA, np.nan); med_g_g = np.full(N_THETA, np.nan)  # Gamma median clean/gain
    sig_g_c = np.full(N_THETA, np.nan); sig_g_g = np.full(N_THETA, np.nan)  # Gamma std
    med_ln_c = np.full(N_THETA, np.nan); med_ln_g = np.full(N_THETA, np.nan) # log10 norm median
    gm_med = np.full(N_THETA, np.nan); gm_lo = np.full(N_THETA, np.nan); gm_hi = np.full(N_THETA, np.nan)  # g marginal on GAIN spectrum

    # resume
    done0 = 0
    if os.path.exists(ckpt):
        d = np.load(ckpt)
        for k,v in [("med_g_c",med_g_c),("med_g_g",med_g_g),("sig_g_c",sig_g_c),("sig_g_g",sig_g_g),
                    ("med_ln_c",med_ln_c),("med_ln_g",med_ln_g),("gm_med",gm_med),("gm_lo",gm_lo),("gm_hi",gm_hi)]:
            if k in d: v[:] = d[k]
        done0 = int(d["done"]) if "done" in d else 0
        print(f"[resume] {tag} from case {done0}", flush=True)

    samp_seed = BASE_SEED + 7
    t0 = time.time()
    for i in range(done0, N_THETA):
        s_c = sample_post(post, x_c[i], n_params, lo, hi, samp_seed + i)  # CRN seed
        s_g = sample_post(post, x_g[i], n_params, lo, hi, samp_seed + i)
        med_g_c[i] = np.median(s_c[:,GAMMA_IDX]); med_g_g[i] = np.median(s_g[:,GAMMA_IDX])
        sig_g_c[i] = np.std(s_c[:,GAMMA_IDX]);   sig_g_g[i] = np.std(s_g[:,GAMMA_IDX])
        med_ln_c[i] = np.median(np.log10(s_c[:,NORM_IDX]))
        med_ln_g[i] = np.median(np.log10(s_g[:,NORM_IDX]))
        if n_params == 6:
            gcol = s_g[:,G_IDX]  # g-marginal evaluated on the injected g=1.03 spectrum
            gm_med[i] = np.median(gcol)
            gm_lo[i] = np.percentile(gcol,5); gm_hi[i] = np.percentile(gcol,95)  # 90% eq-tailed
        if (i+1) % 25 == 0 or i == N_THETA-1:
            np.savez(ckpt, med_g_c=med_g_c, med_g_g=med_g_g, sig_g_c=sig_g_c, sig_g_g=sig_g_g,
                     med_ln_c=med_ln_c, med_ln_g=med_ln_g, gm_med=gm_med, gm_lo=gm_lo, gm_hi=gm_hi,
                     done=i+1)
            el = time.time()-t0
            print(f"[{tag}] {i+1}/{N_THETA}  {el:.0f}s  rate={(i+1-done0)/el:.2f}/s", flush=True)

    # ---- stats ----
    def paired(a_clean, a_gain):
        d = a_gain - a_clean
        m = float(np.mean(d)); se = float(np.std(d, ddof=1)/np.sqrt(len(d)))
        return m, se, (m/se if se>0 else float("nan"))
    gm_bias = paired(med_g_c, med_g_g)
    ln_bias = paired(med_ln_c, med_ln_g)
    sigma_gamma_clean = float(np.mean(sig_g_c))
    sigma_gamma_gain  = float(np.mean(sig_g_g))
    res = {
        "tag": tag, "level": level, "flow": flow, "N": N_THETA, "N_samp": N_SAMP, "gain": GAIN,
        "gamma_bias_mean": gm_bias[0], "gamma_bias_se": gm_bias[1], "gamma_bias_z": gm_bias[2],
        "log10norm_bias_mean": ln_bias[0], "log10norm_bias_se": ln_bias[1], "log10norm_bias_z": ln_bias[2],
        "sigma_gamma_clean_mean": sigma_gamma_clean, "sigma_gamma_gain_mean": sigma_gamma_gain,
        "gamma_bias_in_sigma": gm_bias[0]/sigma_gamma_clean,
    }
    if flow == "gainmarg":
        prior_w = 0.10
        gm_width = gm_hi - gm_lo
        excl = ((gm_lo > 1.0) | (gm_hi < 1.0))  # 90% interval excludes g=1
        res.update({
            "g_marg_median_mean": float(np.mean(gm_med)),
            "g_marg_median_median": float(np.median(gm_med)),
            "g_marg_width_mean": float(np.mean(gm_width)),
            "g_marg_width_over_prior": float(np.mean(gm_width)/prior_w),
            "g_marg_frac_exclude_g1": float(np.mean(excl)),
        })
    with open(os.path.join(OUT, f"result_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("RESULT", json.dumps(res, indent=2), flush=True)

if __name__ == "__main__":
    main()
