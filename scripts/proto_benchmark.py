"""Phase 0 prototypes 2 & 3: jaxspec fakeit throughput + NPE device micro-benchmark.

(2) Time vectorized fakeit for 10,000 tbabs*powerlaw spectra on CPU jax and
    extrapolate to 100k / 200k.
(3) Train a small default-flow NPE on a 2k toy set on GPU vs CPU and time it.

Prints a compact table of the prototype benchmark.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from jaxspec.data.util import load_example_obsconf, fakeit_for_multiple_parameters
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs


def bench_fakeit():
    obs = load_example_obsconf("NGC7793_ULX4_PN")
    model = Tbabs() * Powerlaw()
    rng = np.random.default_rng(0)

    def draw(n):
        return {
            "tbabs_1_nh": rng.uniform(0.1, 0.3, size=n),
            "powerlaw_1_alpha": rng.uniform(0.5, 2.5, size=n),
            "powerlaw_1_norm": 10 ** rng.uniform(-2, 2, size=n),
        }

    # warm up JIT (compilation happens on first call; exclude from timing)
    _ = fakeit_for_multiple_parameters(obs, model, draw(64), rng_key=1)

    results = {}
    for n in (1000, 10000):
        p = draw(n)
        t0 = time.perf_counter()
        spec = fakeit_for_multiple_parameters(obs, model, p, rng_key=2)
        spec = np.asarray(spec)  # force device->host sync
        dt = time.perf_counter() - t0
        results[n] = dt
        print(f"  fakeit {n:>6d} spectra: {dt:7.3f} s  ({n/dt:9.0f} spec/s)  "
              f"shape={spec.shape}")

    rate = 10000 / results[10000]  # spectra/sec at the 10k scale
    print(f"\n  steady-state rate ~ {rate:.0f} spectra/s")
    for target in (100_000, 200_000):
        print(f"  extrapolated {target:>7d} spectra: ~{target/rate:6.1f} s")
    return results, rate


def make_toy_npe_data(n=2000, n_bins=102, seed=0):
    """Tiny synthetic (theta, x) set just to time NPE training (3 params)."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, 3, generator=g)  # 3 params in [0,1]
    # cheap nonlinear map to a 102-d "spectrum" so the flow has signal to learn
    base = torch.linspace(0, 1, n_bins)
    x = (theta[:, [0]] * torch.exp(-base * (1 + 3 * theta[:, [1]]))
         + theta[:, [2]] * torch.sin(6 * base)) + 0.01 * torch.randn(n, n_bins, generator=g)
    return theta, x


def bench_npe(device):
    from sbi.inference import NPE

    theta, x = make_toy_npe_data()
    theta = theta.to(device)
    x = x.to(device)
    t0 = time.perf_counter()
    inf = NPE(device=device)
    inf.append_simulations(theta, x)
    # cap epochs so the benchmark is bounded and comparable across devices
    inf.train(max_num_epochs=30, show_train_summary=False)
    dt = time.perf_counter() - t0
    print(f"  NPE train (2k sims, 30 epochs) on {device:4s}: {dt:6.2f} s")
    return dt


def main():
    print("=== (2) jaxspec fakeit throughput (CPU jax) ===")
    _, rate = bench_fakeit()

    print("\n=== (3) NPE device micro-benchmark (default flow, 2k toy sims) ===")
    cpu_dt = bench_npe("cpu")
    gpu_dt = None
    if torch.cuda.is_available():
        gpu_dt = bench_npe("cuda")
        winner = "GPU" if gpu_dt < cpu_dt else "CPU"
        print(f"\n  faster device for this small flow: {winner} "
              f"(cpu {cpu_dt:.2f}s vs gpu {gpu_dt:.2f}s)")
    else:
        print("  CUDA not available")

    print("\n=== SUMMARY ===")
    print(f"fakeit rate          : {rate:.0f} spectra/s (CPU jax, tbabs*powerlaw, 102-ch EPIC-pn)")
    print(f"100k spectra         : ~{100_000/rate:.1f} s")
    print(f"200k spectra         : ~{200_000/rate:.1f} s")
    print(f"NPE 2k/30ep CPU      : {cpu_dt:.2f} s")
    if gpu_dt is not None:
        print(f"NPE 2k/30ep GPU      : {gpu_dt:.2f} s")


if __name__ == "__main__":
    main()
