"""Which parameter dimensions carry the 6-D C2ST between the NS and the NPE
posterior, for the two bright spectra whose NS samples were persisted.

The 6-D number is quoted next to a statement about gain information, so it
matters whether it is driven by the gain dimension or by width mismatch in nH,
Gamma and the power-law normalization. The marginal Cohen's d already points at
the physical parameters (i394: nH -1.90, Gamma -1.69, PLn -1.83, g -0.71; i416:
PLn -1.40, g +0.17), so the C2ST is recomputed on the full 6-D space, on the 5
physical dimensions with g dropped, on g alone, and on each parameter alone.

It also checks the direction of the duplicate-row bias: does dropping duplicate
NS atoms move the NS sample toward the wider flow?

Run (repo venv, from repo root):
    .venv\\Scripts\\python.exe outputs\\gain_marg\\c2st_gain_ablation\\c2st_dimension_ablation.py
"""
import json, os, sys, time, pathlib, warnings
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[3]
GM = ROOT/"outputs"/"gain_marg"; NS24 = GM/"ns_smallset_v2"
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(GM)); sys.path.insert(0,str(ROOT))
from sbixcal import train_npe as tn
from sbixcal import priors as _priors
P=["tbabs_1_nh","powerlaw_1_alpha","powerlaw_1_norm","blackbodyrad_1_kT","blackbodyrad_1_norm","gain_g"]
LOGC=[2,4]; OUT=pathlib.Path(__file__).resolve().parent/"c2st_dimension_ablation_results.json"
post,info=tn.load_posterior(str(GM/"model_bright"),device="cpu")
arch=json.loads((GM/"model_bright"/"arch.json").read_text())
lo,hi=_priors.prior_bounds(arch["prior_cfg"],P)

def feat(S):
    S=S.copy()
    for c in LOGC: S[:,c]=np.log10(np.clip(S[:,c],1e-30,None))
    return S

def c2st(A,B,cols,cvmode="plain",seed_cv=0):
    n=min(len(A),len(B))
    X=np.vstack([feat(A[:n])[:,cols],feat(B[:n])[:,cols]]); y=np.r_[np.zeros(n,int),np.ones(n,int)]
    if cvmode=="plain":
        sp=StratifiedKFold(5,shuffle=True,random_state=seed_cv).split(X,y)
    else:
        _,g=np.unique(np.vstack([feat(A[:n]),feat(B[:n])]),axis=0,return_inverse=True)
        sp=StratifiedGroupKFold(5,shuffle=True,random_state=seed_cv).split(X,y,groups=g)
    acc=[]
    for k,(tr,te) in enumerate(sp):
        m=Pipeline([("sc",StandardScaler()),("m",MLPClassifier((64,64),activation="relu",solver="adam",
            max_iter=1000,early_stopping=True,n_iter_no_change=20,random_state=0))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); m.fit(X[tr],y[tr])
        acc.append(float((m.predict(X[te])==y[te]).mean()))
    return float(np.mean(acc)),[round(a,5) for a in acc]

res=json.loads(OUT.read_text()) if OUT.exists() else {"runs":[],"moments":{}}
done={r["key"] for r in res["runs"]}
def save():
    t=OUT.with_suffix(".tmp"); t.write_text(json.dumps(res,indent=1)); os.replace(t,OUT)

SETS=[("full6",[0,1,2,3,4,5]),("no_g",[0,1,2,3,4]),("g_only",[5])]+[(f"only_{p}",[i]) for i,p in enumerate(P)]

for stem,tag in [("i394_slice","i394"),("i416_slice","i416")]:
    d=np.load(NS24/f"{stem}_samples.npz",allow_pickle=True)
    A=np.asarray(d["samples"],np.float64); x=np.asarray(d["observed_counts"],np.float64)
    # PRIMARY convention: sbi rejection sampling, torch seed 10 (n_clipped was 0, so == in-box truncation)
    torch.manual_seed(10)
    with torch.no_grad():
        B=post.sample((len(A),),x=torch.as_tensor(x,dtype=torch.float32),show_progress_bars=False,
                      reject_outside_prior=True,max_sampling_time=20.0)
    B=np.asarray(B.cpu().numpy(),np.float64)
    nclip=int(np.any(np.clip(B,lo[None,:],hi[None,:])!=B,axis=1).sum())
    print(f"[{tag}] NS n={len(A)}  NPE n={len(B)}  n_outside_box={nclip}",flush=True)
    # --- dedup bias direction: weighted (posterior) vs unique-atom (dedup) moments vs flow
    Ad=np.unique(A,axis=0); fA,fAd,fB=feat(A),feat(Ad),feat(B)
    res["moments"][tag]={p:{"ns_eq_sd":float(fA[:,i].std(ddof=1)),
                            "ns_dedup_sd":float(fAd[:,i].std(ddof=1)),
                            "flow_sd":float(fB[:,i].std(ddof=1)),
                            "dedup_over_eq":float(fAd[:,i].std(ddof=1)/fA[:,i].std(ddof=1)),
                            "flow_over_eq":float(fB[:,i].std(ddof=1)/fA[:,i].std(ddof=1)),
                            "ns_eq_mean":float(fA[:,i].mean()),"ns_dedup_mean":float(fAd[:,i].mean()),
                            "flow_mean":float(fB[:,i].mean())} for i,p in enumerate(P)}
    save()
    for name,cols in SETS:
        key=f"{tag}|{name}"
        if key in done: continue
        t0=time.time(); m,f=c2st(A,B,cols)
        e={"key":key,"spectrum":tag,"set":name,"cols":cols,"c2st_plain":m,"folds":f,
           "n_per_class":int(min(len(A),len(B))),"wall_s":round(time.time()-t0,1)}
        if name in ("full6","no_g","g_only"):
            mg,_=c2st(A,B,cols,cvmode="group"); e["c2st_group"]=mg
        print(f"  {key:22s} plain={m:.4f}"+(f" group={e.get('c2st_group'):.4f}" if 'c2st_group' in e else "")+f" ({e['wall_s']}s)",flush=True)
        res["runs"].append(e); save()
print("DONE",flush=True)
