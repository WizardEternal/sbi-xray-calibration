"""ADVERSARY: the 'single NPE draw' caveat. BOTH computations used exactly ONE flow
draw seed (primary torch seed 10; re-derivation torch seed 8675309). If the flow
draw carries appreciable seed variance, neither number has a quoted uncertainty
that covers it, and 'stable across sampler settings' is measuring only the NS side
(the NPE side is a shared constant in the primary, since torch.manual_seed(10) is
re-set before every variant's draw).

Draws the flow with 8 independent torch seeds per spectrum and recomputes:
  - C2ST vs the ns24 NS sample, PRIMARY classifier convention (MLP(64,64), plain 5-fold)
  - the flow's g width / prior width (the 'capture' input)
"""
import json, os, sys, time, pathlib, warnings
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[4]  # repo root (was a hardcoded absolute path)
GM = ROOT/"outputs"/"gain_marg"; NS24 = GM/"ns_smallset_v2"; NS48 = GM/"ns_smallset_v2_ns48"
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(GM)); sys.path.insert(0, str(ROOT))
from sbixcal import train_npe as tn
from sbixcal import priors as _priors

P = ["tbabs_1_nh","powerlaw_1_alpha","powerlaw_1_norm","blackbodyrad_1_kT","blackbodyrad_1_norm","gain_g"]
LOGC = [2,4]; PRIOR_G_SD = 0.1/np.sqrt(12)
OUT = pathlib.Path("x08_npe_seed_results.json")
SEEDS = [10, 8675309, 1, 2, 3, 4, 5, 6]

post, info = tn.load_posterior(str(GM/"model_bright"), device="cpu")
arch = json.loads((GM/"model_bright"/"arch.json").read_text())
lo, hi = _priors.prior_bounds(arch["prior_cfg"], P)
print("prior box lo", lo, "hi", hi, flush=True)
print("arch median_total_counts:", arch.get("median_total_counts"), "exposure_s:", arch.get("exposure_s"), flush=True)

def feat(S):
    S = S.copy()
    for c in LOGC: S[:,c] = np.log10(np.clip(S[:,c], 1e-30, None))
    return S

def c2st_primary(A, B, seed_cv=0):
    n = min(len(A), len(B))
    X = np.vstack([feat(A[:n]), feat(B[:n])]); y = np.r_[np.zeros(n,int), np.ones(n,int)]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed_cv); acc=[]
    for k,(tr,te) in enumerate(cv.split(X,y)):
        m = Pipeline([("sc",StandardScaler()),("m",MLPClassifier((64,64),activation="relu",solver="adam",
              max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=0))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); m.fit(X[tr],y[tr])
        acc.append(float((m.predict(X[te])==y[te]).mean()))
    return float(np.mean(acc)), [round(a,5) for a in acc]

res = json.loads(OUT.read_text()) if OUT.exists() else {"runs":[], "meta":{"seeds":SEEDS,
        "note":"raw 20000 draws per seed, then two conventions: in-box filter, and clip-into-box"}}
done = {r["key"] for r in res["runs"]}
def save():
    t=OUT.with_suffix(".tmp"); t.write_text(json.dumps(res,indent=1)); os.replace(t,OUT)

for stem,tag in [("i394_slice","i394"),("i416_slice","i416")]:
    d = np.load(NS24/f"{stem}_samples.npz", allow_pickle=True)
    A = np.asarray(d["samples"], np.float64); x = np.asarray(d["observed_counts"], np.float64)
    ns_g_sd = A[:,5].std(ddof=1)
    for sd in SEEDS:
        key=f"{tag}|seed{sd}"
        if key in done: continue
        t0=time.time(); torch.manual_seed(sd)
        with torch.no_grad():
            raw = post.sample((20000,), x=torch.as_tensor(x, dtype=torch.float32),
                              show_progress_bars=False, reject_outside_prior=False)
        raw = np.asarray(raw.cpu().numpy(), np.float64)
        m_in = np.all((raw>=lo[None,:])&(raw<=hi[None,:]),axis=1)
        B_f = raw[m_in]                                  # in-prior filter (D-028 / re-derive)
        B_c = np.clip(raw, lo[None,:], hi[None,:])       # clip-into-box (raw+clip, no rejection)
        n = len(A)
        rng = np.random.default_rng(9)
        Bf = B_f[rng.choice(len(B_f), min(n,len(B_f)), replace=False)]
        Bc = B_c[rng.choice(len(B_c), n, replace=False)]
        cf,_ = c2st_primary(A, Bf); cc,_ = c2st_primary(A, Bc)
        cr,_ = c2st_primary(A, raw[rng.choice(len(raw), n, replace=False)])
        e = {"key":key,"spectrum":tag,"torch_seed":sd,"n_ns":int(n),
             "frac_out_of_box":float((~m_in).mean()),
             "c2st_filter":cf, "c2st_clip":cc, "c2st_raw":cr,
             "flow_g_sd_filter":float(B_f[:,5].std(ddof=1)),
             "flow_g_sd_raw":float(raw[:,5].std(ddof=1)),
             "flow_g_over_prior_filter":float(B_f[:,5].std(ddof=1)/PRIOR_G_SD),
             "flow_g_over_prior_raw":float(raw[:,5].std(ddof=1)/PRIOR_G_SD),
             "ns_g_over_prior":float(ns_g_sd/PRIOR_G_SD), "wall_s":round(time.time()-t0,1)}
        print(f"{key}: out={e['frac_out_of_box']:.4f} C2ST filt={cf:.4f} clip={cc:.4f} raw={cr:.4f} "
              f"flow_g/prior filt={e['flow_g_over_prior_filter']:.4f} raw={e['flow_g_over_prior_raw']:.4f} "
              f"NS_g/prior={e['ns_g_over_prior']:.4f} ({e['wall_s']}s)", flush=True)
        res["runs"].append(e); save()
print("DONE", flush=True)
