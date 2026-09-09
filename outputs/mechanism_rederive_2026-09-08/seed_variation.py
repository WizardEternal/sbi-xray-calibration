import os,sys,json,time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="2"
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, yaml
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
sys.path.insert(0,str(REPO/"outputs"/"mechanism_rederive_2026-09-08"))
from jaxspec.data.util import fakeit_for_multiple_parameters
from sbixcal import misspec as M, models as MO, priors as P, responses as RS
BASE="tbabs_powerlaw_bb"; ORD=MO.MODEL_PARAMS[BASE]
LOG=[2,4]; CW=np.array([.2,2.,2.,2.7,2.]); REL=1e-3
TM={"B1":{"tbabs_1_nh":0,"powerlaw_1_alpha":1,"powerlaw_1_norm":2,"blackbodyrad_1_kT":3,"blackbodyrad_1_norm":4},
    "B2":{"tbpcf_1_nh":0,"powerlaw_1_alpha":1,"powerlaw_1_norm":2,"blackbodyrad_1_kT":3,"blackbodyrad_1_norm":4},
    "B3":{"tbabs_1_nh":0,"brems_1_norm":2,"blackbodyrad_1_kT":3,"blackbodyrad_1_norm":4},
    "B4":{"tbabs_1_nh":0,"powerlaw_1_alpha":1,"powerlaw_1_norm":2,"blackbodyrad_1_kT":3,"blackbodyrad_1_norm":4}}
SIM=yaml.safe_load(open(REPO/"configs"/"sim_modelA_prod.yaml")); DET=yaml.safe_load(open(REPO/"configs"/"detect.yaml"))
PC=SIM["priors"]; STR={f:float(DET["families"][f]["strength_grid"][-1]) for f in TM}
FX={f:dict(DET["families"][f].get("fixed") or {}) for f in TM}
base=RS.load_base_obsconf("NGC7793_ULX4_PN"); MA=MO.build_model(BASE)
def fold(m,pr,oc,ch=400):
    n=len(next(iter(pr.values()))); o=[]
    for s in range(0,n,ch):
        sl=slice(s,min(s+ch,n)); o.append(np.asarray(fakeit_for_multiple_parameters(oc,m,{k:np.asarray(v[sl],float) for k,v in pr.items()},rng_key=0,apply_stat=False),dtype=np.float64))
    return np.concatenate(o,0)
def c2t(c):
    t=np.array(c,float,copy=True); t[:,LOG]=10.0**t[:,LOG]; return t
def t2c(t):
    c=np.array(t,float,copy=True); c[:,LOG]=np.log10(c[:,LOG]); return c
def jac(th,oc):
    c0=t2c(th); n,p=c0.shape; h=REL*CW; bl=[th]
    for j in range(p):
        for s in (1.,-1.):
            cj=c0.copy(); cj[:,j]+=s*h[j]; bl.append(c2t(cj))
    big=np.concatenate(bl,0); mu=fold(MA,{q:big[:,k] for k,q in enumerate(ORD)},oc)
    J=np.empty((n,mu.shape[1],p))
    for j in range(p): J[:,:,j]=(mu[n+2*j*n:n+(2*j+1)*n]-mu[n+(2*j+1)*n:n+(2*j+2)*n])/(2*h[j])
    return mu[:n],J
def mf(f,th,oc):
    n=th.shape[0]; m,pr=M.FAMILIES[f](BASE,PC,n,np.random.default_rng(0),STR[f],FX[f])
    for k,col in TM[f].items(): pr[k]=th[:,col].copy()
    return fold(m,pr,RS.gain_shift_obsconf(oc,1+STR[f]/100.) if f=="B4" else oc)
res={}
for seed in [20260611,101,202,303,404]:
    rng=np.random.default_rng(seed)
    th=np.stack([P.sample_prior(PC,ORD,200,rng)[q] for q in ORD],1)
    oc=RS.scale_exposure(base,3534.0); mu0,J=jac(th,oc); row={"total_counts_median":float(np.median(mu0.sum(1)))}
    for f in ["B1","B2","B3","B4"]:
        d=mf(f,th,oc)-mu0; n=200; R=np.empty(n); tot=np.empty(n)
        for i in range(n):
            w=1/np.sqrt(mu0[i]); bw=w*d[i]; Jw=w[:,None]*J[i]; q,_=np.linalg.qr(Jw)
            r=bw-q@(q.T@bw); R[i]=r@r; tot[i]=bw@bw
        row[f]={"R_median":float(np.median(R)),"I_median":float(np.median(1-R/tot)),"bnorm2_median":float(np.median(tot))}
    res[seed]=row
    print(seed,"cts",round(row["total_counts_median"],1),{f:round(row[f]["R_median"],5) for f in ["B1","B2","B3","B4"]},flush=True)
json.dump(res,open(REPO/"outputs"/"mechanism_rederive_2026-09-08"/"seed_variation.json","w"),indent=1)
