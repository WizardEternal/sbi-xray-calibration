"""Phase-2 device benchmark: NPE training with the 1-D CNN embedding net, CPU vs GPU.

Phase-0 found that for a PLAIN NSF flow CPU ~ GPU. The question Phase 2 must
answer is whether the CNN EMBEDDING net tips the balance to GPU. We train the
real train_npe stack (NSF + SpectrumCNN) on a fixed dev dataset for a fixed
number of epochs on each device and time it.

Prints a table of the device benchmark.

Usage:
    .venv\\Scripts\\python.exe scripts\\bench_cnn_device.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from sbixcal import train_npe as tn


N = 10000          # representative training size
EPOCHS = 15        # fixed, bounded, comparable across devices
DATASET = "modelA_dev_medium"   # 10k available; middle count regime


def bench(device: str, cfg: dict) -> dict:
    theta, x, names, meta = tn.load_dataset(DATASET, max_n=N)
    prior = tn.build_prior(cfg["priors"], names, device=device)
    # warmup CUDA context (compile kernels) outside the timed region
    if device == "cuda":
        _ = torch.zeros(8, 102, device="cuda").sum().item()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    de, inf, summary = tn.train_one_flow(theta, x, prior, cfg, device=device, seed=0)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ep = summary["epochs_trained"]
    epochs = int(ep[0] if isinstance(ep, (list, tuple)) else ep)
    return {"device": device, "wall_s": dt, "epochs": epochs,
            "s_per_epoch": dt / max(epochs, 1),
            "final_val": summary["validation_loss"][-1]}


def main():
    cfg = tn.load_config("configs/train_npe_dev.yaml")
    cfg["train"]["max_num_epochs"] = EPOCHS
    cfg["train"]["stop_after_epochs"] = 10_000   # disable early stop for a fair fixed-epoch race

    print(f"=== CNN-embedding NPE device benchmark (N={N}, {EPOCHS} epochs, {DATASET}) ===")
    rows = []
    rows.append(bench("cpu", cfg))
    print(f"  CPU : {rows[-1]['wall_s']:6.1f} s  ({rows[-1]['s_per_epoch']:.2f} s/epoch, "
          f"final_val {rows[-1]['final_val']:.3f})")
    if torch.cuda.is_available():
        rows.append(bench("cuda", cfg))
        print(f"  GPU : {rows[-1]['wall_s']:6.1f} s  ({rows[-1]['s_per_epoch']:.2f} s/epoch, "
              f"final_val {rows[-1]['final_val']:.3f})  [{torch.cuda.get_device_name(0)}]")
        cpu, gpu = rows[0]["wall_s"], rows[1]["wall_s"]
        winner = "GPU" if gpu < cpu else "CPU"
        print(f"\n  faster for CNN-embedding training: {winner} "
              f"(cpu {cpu:.1f}s vs gpu {gpu:.1f}s, speedup {cpu/gpu:.2f}x)")
    else:
        print("  CUDA not available")

    print("\n=== SUMMARY ===")
    for r in rows:
        print(f"  {r['device']:4s}: {r['wall_s']:6.1f} s total, "
              f"{r['s_per_epoch']:.2f} s/epoch ({r['epochs']} epochs)")


if __name__ == "__main__":
    main()
