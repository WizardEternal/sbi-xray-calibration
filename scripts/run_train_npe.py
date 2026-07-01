"""Train NPE+NSF flows (one per count level) from a YAML config.

Usage (repo venv):
    .venv\\Scripts\\python.exe scripts\\run_train_npe.py --config configs\\train_npe_prod.yaml
    .venv\\Scripts\\python.exe scripts\\run_train_npe.py --config configs\\train_npe_dev.yaml --level medium
    .venv\\Scripts\\python.exe scripts\\run_train_npe.py --config configs\\train_npe_prod.yaml --device cuda --force

Checkpoints land in outputs/models/<config-name>_<level>/ and are cold-loadable
via sbixcal.train_npe.load_posterior(<dir>).
"""

from __future__ import annotations

import argparse

from sbixcal import train_npe as tn


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train NPE+NSF flows per count level")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="override config device (cpu/cuda)")
    ap.add_argument("--level", default=None, help="train only this level name")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    args = ap.parse_args(argv)

    cfg = tn.load_config(args.config)
    out = tn.run_training(
        cfg,
        config_src_path=args.config,
        device=args.device,
        force=args.force,
        only_level=args.level,
    )
    print("\nTrained flows:")
    for level, d in out.items():
        print(f"  {level:8s} -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
