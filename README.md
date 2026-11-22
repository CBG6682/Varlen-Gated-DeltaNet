<h1 align="center">
  <img src="assets/title.svg" alt="Varlen-Gated-DeltaNet" width="720">
</h1>

<p align="center"><em>Fused Variable-Length Gated DeltaNet v2 &amp; Single-Step Decode CUDA/CuTe Kernels for NVIDIA Turing (SM75)</em></p>

<p align="center">
  <img src="https://img.shields.io/github/license/CBG6682/Varlen-Gated-DeltaNet?style=flat&logo=opensourceinitiative&logoColor=white&labelColor=454545&color=76B900" alt="license">
  <img src="https://img.shields.io/github/last-commit/CBG6682/Varlen-Gated-DeltaNet?style=flat&logo=git&logoColor=white&labelColor=454545&color=76B900" alt="last commit">
  <img src="https://img.shields.io/github/languages/top/CBG6682/Varlen-Gated-DeltaNet?style=flat&labelColor=454545&color=76B900" alt="top language">
  <img src="https://img.shields.io/badge/arXiv-2605.22791-B31B1B?style=flat&logo=arxiv&logoColor=white&labelColor=454545" alt="arXiv 2605.22791">
</p>

<p align="center"><em>Built with the tools and technologies:</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-76B900.svg?style=flat&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/CUTLASS%20CuTe-2E7D32.svg?style=flat&logo=nvidia&logoColor=white" alt="CUTLASS CuTe">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/torch.compile-FF6F00.svg?style=flat&logo=pytorch&logoColor=white" alt="torch.compile">
</p>

<div align="center">

| **5.06x** | **9.17x** | **51% ↓** | **13.35** |
|:---:|:---:|:---:|:---:|
| Training speedup vs PyTorch (`L=4096`) | Decode speedup vs PyTorch (`B=1024`) | Peak training VRAM vs NVLabs (`L=4096`) | Peak FP16 TFLOP/s on Tesla T4 |

</div>

<details>
<summary><b>Table of Contents</b></summary>

- [Motivation](#motivation)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running tests and benchmarks](#running-tests-and-benchmarks)
- [Project structure](#project-structure)
- [License](#license)

</details>

---

## Motivation

`deltanet_v2` provides fused variable-length training and single-step (`L = 1`) autoregressive decode CUDA/CuTe kernels for [**Gated DeltaNet v2** (arXiv:2605.22791)](https://arxiv.org/pdf/2605.22791), tuned specifically for **NVIDIA Turing (`SM75`)** GPUs such as Tesla T4 and RTX 20 series cards with native **PyTorch Autograd**, **GQA**, and **`torch.compile` (TorchInductor)** support.

- ⚡ **Turing (`SM75`) native Tensor Core fusion:** Uses `SM75_16x8x8` MMA atoms, swizzled Shared Memory layouts, and register-level epilogues to avoid `SM80+` shared-memory/barrier assumptions.
- 🧠 **Minimal activation footprint:** Recomputes chunk-wise WY activations during backward and reuses intermediate buffers (`Y` -> `S`), cutting peak training VRAM in half compared to open-source Triton baselines.
- 🔗 **End-to-end `torch.compile` integration:** Registers `FakeTensor` metadata with `TORCH_LIBRARY`, supports packed varlen sequences (`cu_seqlens`), and ships a pre-built wheel alongside 1-command source builds.

---

## Benchmarks

> [!NOTE]
> All benchmarks below were measured on **NVIDIA Tesla T4 (16 GB, `SM75`)** in FP16 (`torch.float16`, PyTorch `2.14.1+cu130`, CUDA `13.0`) via [benchmark/demo.ipynb](benchmark/demo.ipynb). All 4 correctness suites (varlen Fwd/Bwd, decode `L=1`, end-to-end `DeltaBlock` + CUDA Graphs, and `torch.compile`) passed before benchmarking.

### Training benchmark

<p align="center">
  <img src="assets/training_benchmark.png" alt="DeltaNet v2 Training Benchmark on NVIDIA Tesla T4 (SM75, FP16)" width="85%">
</p>

**Part A — Pure token mixer vs open-source baselines** (`B = 4`, `d_model = 768`, `H_q = 8`, `H_kv = 8`, `d_k = d_v = 64`, `chunk_size = 64`):

| SeqLen | Method | Fwd (ms) ↓ | Fwd+Bwd (ms) ↓ | TFLOP/s ↑ | Peak VRAM (MB) ↓ | Speedup ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | PyTorch Reference (Unfused) | 3.20 | 10.34 | 3.53 | 164.9 | 1.00x |
| 512 | FLA GatedDeltaNet (v1) | 4.73 | 10.73 | 3.40 | **94.4** | 0.96x |
| 512 | FLA GatedDeltaNet-2 | 2.77 | 8.36 | 4.36 | 155.6 | 1.24x |
| 512 | NVLabs GatedDeltaNet-2 | 2.60 | 8.19 | 4.45 | 162.4 | 1.26x |
| 512 | **DeltaNet v2 (Ours)** | **1.17** | **3.72** | **9.81** | 103.2 | **2.78x** |
| 512 | DeltaNet v2 (Compiled) | 1.60 | 5.39 | 6.76 | 103.3 | 1.92x |
| 1024 | PyTorch Reference (Unfused) | 5.10 | 17.88 | 4.08 | 307.9 | 1.00x |
| 1024 | FLA GatedDeltaNet (v1) | 9.26 | 20.95 | 3.48 | **152.6** | 0.85x |
| 1024 | FLA GatedDeltaNet-2 | 3.46 | 16.32 | 4.47 | 270.7 | 1.10x |
| 1024 | NVLabs GatedDeltaNet-2 | 3.14 | 15.93 | 4.57 | 274.7 | 1.12x |
| 1024 | **DeltaNet v2 (Ours)** | **1.59** | **5.62** | **12.97** | 152.7 | **3.18x** |
| 1024 | DeltaNet v2 (Compiled) | 1.89 | 8.74 | 8.34 | **152.5** | 2.05x |
| 2048 | PyTorch Reference (Unfused) | 8.42 | 40.84 | 3.57 | 566.8 | 1.00x |
| 2048 | FLA GatedDeltaNet (v1) | 18.41 | 41.60 | 3.50 | **253.6** | 0.98x |
| 2048 | FLA GatedDeltaNet-2 | 6.71 | 32.20 | 4.53 | 491.4 | 1.27x |
| 2048 | NVLabs GatedDeltaNet-2 | 6.77 | 32.14 | 4.54 | 499.4 | 1.27x |
| 2048 | DeltaNet v2 (Ours) | 2.97 | 11.22 | 12.99 | 254.8 | 3.64x |
| 2048 | **DeltaNet v2 (Compiled)** | **2.92** | **10.92** | **13.35** | 254.8 | **3.74x** |
| 4096 | PyTorch Reference (Unfused) | 16.88 | 115.24 | 2.53 | 1085.2 | 1.00x |
| 4096 | FLA GatedDeltaNet (v1) | 36.37 | 83.11 | 3.51 | **455.6** | 1.39x |
| 4096 | FLA GatedDeltaNet-2 | 14.00 | 64.72 | 4.50 | 932.9 | 1.78x |
| 4096 | NVLabs GatedDeltaNet-2 | 14.35 | 63.69 | 4.58 | 948.9 | 1.81x |
| 4096 | **DeltaNet v2 (Ours)** | 6.74 | **22.76** | **12.80** | 459.8 | **5.06x** |
| 4096 | DeltaNet v2 (Compiled) | **6.05** | 24.16 | 12.06 | 459.8 | 4.77x |

**Part B — End-to-end `DeltaBlock` training with SwiGLU MLP and `torch.compile`** (`d_model = 768`, `hidden_dim = 3072`, `H_q = 8`, `H_kv = 4`, `L = 1024`):

| Batch (`B`) | Mode | Step (ms) ↓ | Throughput (tok/s) ↑ | Peak VRAM (GB) ↓ | Speedup ↑ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | torch-eager | 20.4 | 200,997 | 0.32 | 1.00x |
| 4 | torch-compiled | 15.5 | 263,571 | 0.24 | 1.31x |
| 4 | kernel-eager | 14.2 | 288,463 | 0.24 | 1.44x |
| 4 | **kernel-compiled** | **13.8** | **297,651** | **0.19** | **1.48x** |
| 8 | torch-eager | 39.1 | 209,554 | 0.61 | 1.00x |
| 8 | torch-compiled | 29.6 | 276,862 | 0.45 | 1.32x |
| 8 | kernel-eager | 28.3 | 289,695 | 0.45 | 1.38x |
| 8 | **kernel-compiled** | **25.6** | **320,186** | **0.34** | **1.53x** |
| 16 | torch-eager | 75.8 | 216,265 | 1.18 | 1.00x |
| 16 | torch-compiled | 58.7 | 279,107 | 0.86 | 1.29x |
| 16 | kernel-eager | 54.7 | 299,667 | 0.86 | 1.39x |
| 16 | **kernel-compiled** | **50.6** | **323,582** | **0.64** | **1.50x** |

### Inference benchmark

<p align="center">
  <img src="assets/inference_benchmark.png" alt="DeltaNet v2 Inference Decode and E2E Rollout Benchmark on NVIDIA Tesla T4 (SM75, FP16)" width="85%">
</p>

**Part A — Single-step decode (`L = 1`) pure token mixer vs open-source baselines** (`d_model = 768`, `H_q = 8`, `H_kv = 8`):

| Batch (`B`) | Method | Latency (ms) ↓ | Throughput (tok/s) ↑ | Memory BW (GB/s) ↑ | Peak VRAM (MB) ↓ | Speedup ↑ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PyTorch Reference (Unfused) | 1.036 | 966 | 5.5 | 27.4 | 1.00x |
| 1 | FLA GatedDeltaNet (v1) | 0.936 | 1,068 | 6.1 | 27.1 | 1.11x |
| 1 | FLA GatedDeltaNet-2 | 1.347 | 742 | 4.2 | 27.1 | 0.77x |
| 1 | NVLabs GatedDeltaNet-2 | 1.255 | 797 | 4.5 | 27.1 | 0.83x |
| 1 | **DeltaNet v2 (Ours)** | **0.296** | **3,374** | **19.1** | **27.0** | **3.49x** |
| 1 | DeltaNet v2 (Compiled) | 0.741 | 1,349 | 7.6 | **27.0** | 1.40x |
| 16 | PyTorch Reference (Unfused) | 1.800 | 8,888 | 4.3 | 36.6 | 1.00x |
| 16 | FLA GatedDeltaNet (v1) | 1.511 | 10,589 | 5.1 | 31.9 | 1.19x |
| 16 | FLA GatedDeltaNet-2 | 2.153 | 7,432 | 3.6 | 32.0 | 0.84x |
| 16 | NVLabs GatedDeltaNet-2 | 2.252 | 7,104 | 3.4 | 32.0 | 0.80x |
| 16 | DeltaNet v2 (Ours) | 0.698 | 22,909 | 11.0 | **30.0** | 2.58x |
| 16 | **DeltaNet v2 (Compiled)** | **0.433** | **36,981** | **17.7** | **30.0** | **4.16x** |
| 64 | PyTorch Reference (Unfused) | 1.137 | 56,275 | 12.4 | 67.2 | 1.00x |
| 64 | FLA GatedDeltaNet (v1) | 0.990 | 64,654 | 14.3 | 47.2 | 1.15x |
| 64 | FLA GatedDeltaNet-2 | 1.416 | 45,205 | 10.0 | 47.4 | 0.80x |
| 64 | NVLabs GatedDeltaNet-2 | 1.389 | 46,078 | 10.2 | 47.4 | 0.82x |
| 64 | **DeltaNet v2 (Ours)** | **0.427** | **150,054** | **33.1** | 39.8 | **2.67x** |
| 64 | DeltaNet v2 (Compiled) | 0.432 | 148,234 | 32.7 | **39.5** | 2.63x |
| 256 | PyTorch Reference (Unfused) | 2.204 | 116,146 | 18.1 | 188.5 | 1.00x |
| 256 | FLA GatedDeltaNet (v1) | 1.482 | 172,760 | 26.9 | 108.2 | 1.49x |
| 256 | FLA GatedDeltaNet-2 | 1.300 | 196,959 | 30.7 | 109.2 | 1.70x |
| 256 | NVLabs GatedDeltaNet-2 | 1.416 | 180,739 | 28.1 | 109.2 | 1.56x |
| 256 | DeltaNet v2 (Ours) | 0.704 | 363,550 | 56.6 | 78.7 | 3.13x |
| 256 | **DeltaNet v2 (Compiled)** | **0.692** | **370,010** | **57.6** | **77.7** | **3.19x** |
| 1024 | PyTorch Reference (Unfused) | 7.946 | 128,869 | 18.0 | 673.4 | 1.00x |
| 1024 | FLA GatedDeltaNet (v1) | 1.453 | 704,553 | 98.3 | 352.3 | 5.47x |
| 1024 | FLA GatedDeltaNet-2 | 1.865 | 549,074 | 76.6 | 356.3 | 4.26x |
| 1024 | NVLabs GatedDeltaNet-2 | 1.922 | 532,861 | 74.4 | 356.3 | 4.13x |
| 1024 | **DeltaNet v2 (Ours)** | **0.867** | **1,181,215** | **164.8** | 234.4 | **9.17x** |
| 1024 | DeltaNet v2 (Compiled) | 0.981 | 1,044,170 | 145.7 | **230.4** | 8.10x |
| 2048 | PyTorch Reference (Unfused) | 14.412 | 142,108 | 19.4 | 1384.1 | 1.00x |
| 2048 | FLA GatedDeltaNet (v1) | 2.893 | 707,854 | 96.9 | 677.9 | 4.98x |
| 2048 | FLA GatedDeltaNet-2 | 3.158 | 648,528 | 88.7 | 685.8 | 4.56x |
| 2048 | NVLabs GatedDeltaNet-2 | 3.173 | 645,405 | 88.3 | 685.8 | 4.54x |
| 2048 | DeltaNet v2 (Ours) | 1.664 | 1,230,466 | 168.4 | 441.9 | 8.66x |
| 2048 | **DeltaNet v2 (Compiled)** | **1.622** | **1,262,259** | **172.7** | **433.9** | **8.88x** |

**Part B — End-to-end `DeltaBlock` inference (`Prefill = 512`, `Gen = 32` steps)** (`d_model = 768`, `hidden_dim = 3072`, `H_q = 8`, `H_kv = 4`):

| Batch (`B`) | Mode | Decode `L=1` (ms) ↓ | Throughput (tok/s) ↑ | E2E Prefill+Gen (ms) ↓ | Peak VRAM (MB) ↓ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | torch-eager | 1.298 | 12,331 | 49.65 | 372.7 |
| 16 | torch-compiled | 0.560 | 28,566 | 28.01 | 148.0 |
| 16 | kernel-eager | 1.065 | 15,027 | **23.44** | 247.3 |
| 16 | **kernel-compiled** | **0.551** | **29,032** | 25.02 | **147.2** |
| 64 | torch-eager | 1.884 | 33,975 | 66.19 | 1414.8 |
| 64 | torch-compiled | **0.568** | **112,715** | 50.25 | **516.0** |
| 64 | kernel-eager | 0.649 | 98,658 | 44.96 | 913.0 |
| 64 | kernel-compiled | 0.575 | 111,309 | **43.83** | 913.0 |
| 256 | torch-eager | 1.372 | 186,611 | 235.97 | 5583.8 |
| 256 | torch-compiled | 0.951 | 269,063 | 193.27 | **1988.8** |
| 256 | kernel-eager | 0.675 | 379,248 | 168.63 | 3575.8 |
| 256 | **kernel-compiled** | **0.608** | **420,961** | **167.58** | 3575.8 |
| 1024 | torch-eager | 4.302 | 238,049 | — | 341.6 |
| 1024 | torch-compiled | 2.157 | 474,652 | — | 164.0 |
| 1024 | kernel-eager | 1.934 | 529,551 | — | 89.0 |
| 1024 | **kernel-compiled** | **1.807** | **566,716** | — | **72.4** |

---

## Installation

### Requirements

- **GPU:** NVIDIA Turing (`SM75`, e.g. Tesla T4, RTX 2060/2070/2080) or compatible CUDA GPU
- **Software:** CUDA Toolkit >= 11.8 (`12.x` / `13.0` supported), PyTorch >= 2.0, `setuptools >= 61.0`, `wheel`, `ninja`
- **CUTLASS:** Pinned as a submodule in `third_party/cutlass` (`setup.py` initializes or shallow-clones CUTLASS automatically if missing)

### Option 1: Install pre-built wheel

A pre-built wheel for `Python 3.13` / `CUDA 13.0` / `PyTorch 2.14` (`SM75`) is included in `wheels/`:

```bash
pip install --no-deps wheels/deltanet_v2-2.0.0-py3-none-any.whl
```

### Option 2: Build from source

```bash
git clone --recursive https://github.com/CBG6682/Varlen-Gated-DeltaNet.git
cd Varlen-Gated-DeltaNet
pip install -r requirements.txt
pip install --no-deps --no-build-isolation -e .
```

To package a wheel into `wheels/`:

```bash
python -m pip wheel --no-deps --no-build-isolation -w wheels .
```

---

## Quick start

### Packed varlen training (`DeltaBlock` + `torch.compile`)

```python
import torch
from deltanet_v2 import DeltaBlock

device = "cuda"
model = DeltaBlock(
    d_model=768,
    num_q_heads=8,
    num_kv_heads=4,
    d_k=64,
    d_v=64,
    chunk_size=64,
    hidden_dim=768 * 4,
).to(device=device, dtype=torch.float16)

compiled_model = torch.compile(model, backend="inductor")

# Packed batch of 2 sequences: lengths [512, 1024] -> total_L = 1536
cu_seqlens = torch.tensor([0, 512, 1536], dtype=torch.int32, device=device)
x = torch.randn(1536, 768, device=device, dtype=torch.float16, requires_grad=True)

out, final_states = compiled_model(x, cu_seqlens=cu_seqlens, use_kernel=True)
out.sum().backward()
```

### Autoregressive decode (`L = 1`)

```python
model.eval()
batch_size = 64
x_next = torch.randn(batch_size, 1, 768, device=device, dtype=torch.float16)
past_states = torch.zeros(batch_size, 4, 64, 64, device=device, dtype=torch.float16)

with torch.no_grad():
    y, past_states = model(x_next, past_states=past_states, use_kernel=True)
```

---

## Running tests and benchmarks

```bash
# 1. Numerical correctness (Varlen Fwd/Bwd, Decode L=1, E2E Handoff, CUDAGraph, torch.compile)
python tests/test_correctness.py

# 2. Training benchmark (Token Mixer vs Baselines & E2E DeltaBlock)
python tests/bench_training.py --save-plot assets/training_benchmark.png

# 3. Inference benchmark (Decode L=1 & E2E Prefill+Generation)
python tests/bench_inference.py --gen-steps 32 --save-plot assets/inference_benchmark.png
```

For an end-to-end Colab/Kaggle workflow covering setup, wheel/source build, correctness verification, benchmarks, and Nsight Compute (`ncu`) kernel profiling, see [benchmark/demo.ipynb](benchmark/demo.ipynb).

---

## Project structure

```text
Varlen-Gated-DeltaNet/
├── include/                         # CUDA/CuTe headers (deltanet_common.cuh, deltanet_ops.cuh)
├── src/                             # Fused CUDA kernels & PyTorch dispatcher bindings
├── deltanet_v2/                     # Python package (ops.py, model.py)
├── tests/                           # Correctness & benchmark suites
├── benchmark/                       # End-to-end Colab/NCU demo notebook (demo.ipynb)
├── assets/                          # Animated title SVG & benchmark plots
├── wheels/                          # Pre-built wheel (.whl)
├── setup.py
├── pyproject.toml
├── requirements.txt
└── LICENSE                          # Apache License 2.0
```

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
