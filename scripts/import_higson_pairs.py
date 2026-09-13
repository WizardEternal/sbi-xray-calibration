r"""Turn the 2026-07 Higson gain-pair runs into paired-jsonl rows.

The 2026-07-23 Higson batch (scripts/higson_batch.py) ran pairs 8, 6, 0, 9, 10 of
the CURRENT scripts/paired_ns_gain_check.py draw with exactly the new set's NS
settings (min_num_live_points=400, dlogz=0.5, max_ncalls=400000, np.random.seed =
pair index) and an isolated per-run log_dir. Their total counts reproduce the
current draw exactly (1403/1416, 526/521, 90/90, 2128/2187, 3659/3700), so they
ARE five of the new twelve pairs and do not need re-running.

This writes them as paired rows (same schema as paired_ns_gain_check.py, with
log_dir_clean/log_dir_gain pointing at the existing run directories) so
scripts/paired_higson_analyze.py can merge them with the new set's own jsonls.

    python scripts/import_higson_pairs.py \
        --out outputs/ns_bench/revision_2026-09-02/medium/pairs_reused.jsonl
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "ns_bench" / "higson" / "runs"
PAIRS = [0, 6, 8, 9, 10]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", default=",".join(str(i) for i in PAIRS))
    args = ap.parse_args(argv)
    idx = [int(x) for x in str(args.pairs).split(",") if x.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in idx:
        d = {}
        for kind in ("clean", "gain"):
            m = RUNS / f"gain_pair{i}_{kind}" / "DONE.json"
            if not m.exists():
                raise SystemExit(f"[err] missing {m}")
            d[kind] = json.loads(m.read_text())
        c, g = d["clean"], d["gain"]
        rows.append({
            "i": i,
            "counts_clean": c["counts"], "counts_gain": g["counts"],
            "logz_clean": c["ns_logz"], "logzerr_clean": c["ns_logzerr"],
            "logz_gain": g["ns_logz"], "logzerr_gain": g["ns_logzerr"],
            "d_paired": g["ns_logz"] - c["ns_logz"],
            "wall_s": round(c["wall_s"] + g["wall_s"], 1),
            "exposure": 353.4, "gain": 1.03, "n_block": 12, "theta_seed": 20260630,
            "ncall_clean": c["ns_ncall"], "ncall_gain": g["ns_ncall"],
            "max_ncalls": 400000, "min_num_live_points": 400, "dlogz": 0.5,
            "log_dir_clean": f"outputs/ns_bench/higson/runs/gain_pair{i}_clean",
            "log_dir_gain": f"outputs/ns_bench/higson/runs/gain_pair{i}_gain",
            "provenance": "2026-07-23 higson_batch.py run, identical NS settings and "
                          "identical spectrum to the current paired_ns_gain_check.py draw",
        })
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        print(f"[import] pair {r['i']:>2} counts {r['counts_clean']}/{r['counts_gain']} "
              f"d_paired {r['d_paired']:+.4f} ncall {r['ncall_clean']}/{r['ncall_gain']} "
              f"wall {r['wall_s']:.0f}s")
    print(f"[out] {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
