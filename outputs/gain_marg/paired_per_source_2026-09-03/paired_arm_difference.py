"""Paired standard error of the arm difference on the paper's statistic (2026-09-03).

Statistic per spectrum i and arm a in {fixed, gainmarg}: delta_a,i = bias_a(gain case)_i - bias_a(clean case)_i
(the induced shift). Arm difference D_i = delta_fixed,i - delta_gainmarg,i over the same 500 CRN spectra.
Inputs: the *_per_source.npz written by eval_gainmarg_paired{,_bright}.py --save-per-source.
Run: python paired_arm_difference.py   (from this directory)
"""
import numpy as np, glob, json
out = {}
def stats(a, b):
    D = a - b; n = len(D); se = D.std(ddof=1) / np.sqrt(n)
    return dict(n=int(n), mean=float(D.mean()), sd=float(D.std(ddof=1)), se_paired=float(se), t=float(D.mean() / se),
                se_naive=float(np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)), corr=float(np.corrcoef(a, b)[0, 1]),
                mean_a=float(a.mean()), mean_b=float(b.mean()))
z = {d: np.load(glob.glob(f"{d}/*_per_source.npz")[0]) for d in ["bright_uncapped", "bright_production", "medium"]}
delta = lambda zz, arm, stat: zz[f"{arm}_gain_{stat}_bias"] - zz[f"{arm}_clean_{stat}_bias"]
for d, zz in z.items():
    for stat in ["gamma", "log10norm"]:
        out[f"{d}/{stat}/fixed_minus_gainmarg"] = stats(delta(zz, "fixed", stat), delta(zz, "gainmarg", stat))
assert np.array_equal(z["bright_uncapped"]["theta"], z["bright_production"]["theta"])
for stat in ["gamma", "log10norm"]:
    out[f"bright/{stat}/fixed_capped_minus_fixed_uncapped"] = stats(delta(z["bright_production"], "fixed", stat), delta(z["bright_uncapped"], "fixed", stat))
# fixed-effect combination of the normalization arm difference over the two independent levels
b, m = out["bright_uncapped/log10norm/fixed_minus_gainmarg"], out["medium/log10norm/fixed_minus_gainmarg"]
w = np.array([1 / b["se_paired"] ** 2, 1 / m["se_paired"] ** 2]); mu = np.array([b["mean"], m["mean"]])
out["combined/log10norm/fixed_minus_gainmarg"] = dict(mean=float((w * mu).sum() / w.sum()), se=float(1 / np.sqrt(w.sum())), t=float((w * mu).sum() / np.sqrt(w.sum())))
for k, v in out.items():
    print(f"{k:55s} mean {v['mean']:+.6f}  se_paired {v.get('se_paired', v.get('se')):.6f}  t {v['t']:+.2f}" + (f"  (naive se {v['se_naive']:.6f}, corr {v['corr']:.3f})" if 'se_naive' in v else ""))
json.dump(out, open("paired_arm_difference.json", "w"), indent=1)
