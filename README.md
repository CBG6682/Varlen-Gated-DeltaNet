# DeltaNet v2 (`deltanet_v2`)

High-performance **Variable-Length Gated DeltaNet v2** CUDA/CuTe library for **NVIDIA Turing (`SM75` — Tesla T4 / RTX 20-Series)**, featuring chunk-wise WY state updates, Grouped Query Attention (GQA), fast `L = 1` autoregressive decoding, native **PyTorch Autograd**, and **`torch.compile` (TorchInductor)** support.

Based on NVIDIA's [**Gated DeltaNet v2** (arXiv:2605.22791)](https://arxiv.org/pdf/2605.22791), preserving the exact recurrence formulation while adding GQA and fused Turing kernels.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x%20%7C%2013.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/Target%20GPU-Turing%20SM75%20(Tesla%20T4%20%2F%20RTX%2020--Series)-orange.svg)](https://developer.nvidia.com/cuda-gpus)

---

## Motivation

This project is built to help **students, researchers, and developers** train and deploy linear-attention models efficiently on accessible **Turing (`SM75`)** GPUs such as **Google Colab / Kaggle Tesla T4** and **RTX 20-series** cards:

- **Tailored for Turing `SM75`:** Most open-source linear-attention kernels target Ampere/Hopper (`SM80+`) and hit register/shared-memory limits on `SM75`. DeltaNet v2 is tuned directly for Turing tensor cores and memory hierarchy.
- **Fused Chunk-WY & Decode Pipelines:** Fuses intra-chunk attention, inter-chunk state scan, and single-step (`L = 1`) decode into compact on-chip kernels to cut global memory round-trips and peak activation VRAM.
- **`torch.compile` & 1-Command Setup:** Registers `FakeTensor` metadata with the PyTorch Dispatcher for seamless `torch.compile` integration, with pre-built `.whl` distribution support.

---

## Benchmarks (NVIDIA Tesla T4 — `SM75`)

> **Hardware Scope:** All measurements below were collected on an **NVIDIA Tesla T4 (16 GB VRAM, `SM75`)** using FP16 (`torch.float16`). Other GPU architectures (Ampere, Ada, Hopper) have not been evaluated.

### 1. End-to-End `DeltaBlock` with `torch.compile`

Evaluates the **complete token-mixer block** (`W_proj` projections, DeltaNet recurrence, RMSNorm, gating, `W_out`, and MLP/FFW) with $d_{\text{model}} = 768$, $H_q = 8$, $H_{kv} = 4$, $d_k = d_v = 64$.

![End-to-End Benchmark with torch.compile](assets/benchmark_compiled_e2e.png)

#### Autoregressive Decode (`L = 1`)

| Batch Size | Torch Eager (ms / tok/s) | Torch Compiled (ms / tok/s) | Kernel Compiled (ms / tok/s) | Peak VRAM (Base / Ours) | Speedup vs Eager |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **16** | 10.55 / 1,516 | 3.63 / 4,405 | **3.36 / 4,769** | 0.33 / 0.32 GB | **3.14x** |
| **64** | 9.94 / 6,438 | 4.83 / 13,258 | **3.27 / 19,570** | 0.42 / 0.36 GB | **3.04x** |
| **256** | 29.45 / 8,693 | 14.29 / 17,917 | **7.08 / 36,184** | 0.81 / 0.56 GB | **4.16x** |
| **1024** | 113.56 / 9,017 | 58.30 / 17,565 | **24.65 / 41,544** | 2.34 / 1.36 GB | **4.61x** |

*Max Decode Capacity before OOM:* **13,312** (`kernel-compiled`, **42,258 tok/s**) vs **10,752** (`torch-compiled`, 19,570 tok/s) vs **6,656** (`torch-eager`, 9,152 tok/s) — **2.0x batch capacity** and **4.62x peak throughput**.

#### Training (`Fwd + Bwd`, `L = 1024`)

| Batch Size | Torch Eager | Torch Compiled | Kernel Compiled | Saved Activations (Comp / Ours) |
| :---: | :---: | :---: | :---: | :---: |
| **4** | 411.2 ms (9,961 tok/s) | 209.0 ms (19,600 tok/s) | **194.3 ms (21,083 tok/s)** | 2.36 / **1.62 GB** |
| **8** | 776.0 ms (10,555 tok/s) | 363.0 ms (22,596 tok/s) | **345.0 ms (23,765 tok/s)** | 4.53 / **3.02 GB** |
| **16** | *OOM* | 706.0 ms (23,198 tok/s) | **619.0 ms (26,459 tok/s)** | 8.86 / **5.85 GB** |

---

### 2. Pure Token Mixer vs Open-Source Baselines

Isolates the **recurrence token-mixer kernel** ($d_{\text{model}} = 768$, $H_q = 8$, $H_{kv} = 8$, $d_k = d_v = 64$) against **PyTorch Reference (Unfused)**, **FLA GatedDeltaNet (v1)**, **FLA GatedDeltaNet-2**, and **NVLabs GatedDeltaNet-2**.

![Open-Source Baselines Benchmark Comparison](assets/benchmark_opensource_baselines.png)

#### Autoregressive Decode (`L = 1`, Latency ms / Throughput tok/s)

| Batch | PyTorch Eager | FLA v1 | FLA v2 | NVLabs v2 | DeltaNet v2 (Ours) | Speedup (vs PyTorch / NVLabs) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.980 / 1,020 | 1.139 / 878 | 2.252 / 444 | 1.282 / 780 | **0.281 / 3,557** | **3.49x / 4.56x** |
| **16** | 1.756 / 9,111 | 0.905 / 17,675 | 1.337 / 11,967 | 1.298 / 12,324 | **0.598 / 26,760** | **2.94x / 2.17x** |
| **64** | 1.361 / 47,027 | 0.962 / 66,517 | 2.003 / 31,950 | 1.584 / 40,406 | **0.414 / 154,628** | **3.29x / 3.83x** |
| **256** | 2.207 / 116,011 | 0.924 / 276,984 | 1.371 / 186,721 | 1.465 / 174,755 | **0.437 / 585,870** | **5.05x / 3.35x** |
| **1024** | 7.498 / 136,563 | 1.572 / 651,352 | 1.752 / 584,317 | 1.751 / 584,885 | **0.975 / 1,050,217** | **7.69x / 1.80x** |
| **2048** | 12.906 / 158,688 | 2.872 / 713,035 | 3.138 / 652,713 | 3.139 / 652,358 | **1.665 / 1,229,778** | **7.75x / 1.89x** |

#### Training Pass (`Fwd + Bwd`, `B = 4`, ms)

| SeqLen | Tokens | PyTorch Eager | FLA v1 | FLA v2 | NVLabs v2 | DeltaNet v2 (Ours) | Peak VRAM (Ours / NVLabs) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 2,048 | 9.89 | 11.65 | 8.27 | 8.36 | **3.63** (**2.30x** vs NVLabs) | **104.3** / 164.3 MB |
| **1024** | 4,096 | 17.97 | 20.95 | 15.96 | 15.97 | **5.53** (**2.89x** vs NVLabs) | **154.6** / 275.6 MB |
| **2048** | 8,192 | 40.68 | 41.53 | 31.73 | 31.81 | **11.05** (**2.88x** vs NVLabs) | **255.8** / 500.2 MB |
| **4096** | 16,384 | 115.32 | 82.57 | 63.31 | 63.25 | **22.96** (**2.75x** vs NVLabs) | **459.6** / 949.7 MB |

---

## Installation

### Requirements
- **GPU:** NVIDIA Turing (`SM75`: Tesla T4, RTX 2060/2070/2080, Quadro RTX). Embedded `compute_75` PTX also supports JIT execution on Ampere/Ada/Hopper.
- **OS & Python:** Linux `x86_64`, **Python 3.13**, **PyTorch >= 2.0**, **CUDA Toolkit >= 11.8 / 12.x / 13.0**, `g++ >= 10` (C++20 with automatic C++17 fallback).
- **CUTLASS / CuTe:** Managed as a submodule in `third_party/cutlass` (`>= 4.x`, `<cute/tensor.hpp>`).

### Option 1: Install Pre-built Wheel (Fastest)

```bash
pip install wheels/deltanet_v2-2.0.0-py3-none-any.whl
# Or directly from GitHub Releases:
pip install https://github.com/CBG6682/Varlen-Gated-DeltaNet/releases/download/v2.0.0/deltanet_v2-2.0.0-py3-none-any.whl
```

### Option 2: Build from Source

```bash
git clone --recursive https://github.com/CBG6682/Varlen-Gated-DeltaNet.git
cd Varlen-Gated-DeltaNet
git submodule update --init --recursive third_party/cutlass
pip install -r requirements.txt
MAX_JOBS=2 pip install . --no-build-isolation
```

---

## Quick Start

### 1. Packed Varlen Training + `torch.compile`

```python
import torch
from deltanet_v2 import DeltaBlock

device = "cuda"
model = DeltaBlock(
    d_model=768,
    num_q_heads=8,
    num_kv_heads=4,  # Supports GQA (num_q_heads % num_kv_heads == 0)
    d_k=64,
    d_v=64,
    chunk_size=64,
    hidden_dim=768 * 4,
).to(device=device, dtype=torch.float16)

compiled_model = torch.compile(model, backend="inductor")

# Packed batch of 2 sequences: lengths [512, 1024] -> total_L = 1536
x = torch.randn(1536, 768, device=device, dtype=torch.float16, requires_grad=True)
cu_seqlens = torch.tensor([0, 512, 1536], dtype=torch.int32, device=device)

out, final_states = compiled_model(x, cu_seqlens=cu_seqlens, use_kernel=True)
out.sum().backward()
```

### 2. Autoregressive Decode (`L = 1`)

```python
model.eval()
batch_size = 64
next_token = torch.randn(batch_size, 1, 768, device=device, dtype=torch.float16)
past_states = torch.zeros(batch_size, 4, 64, 64, device=device, dtype=torch.float16)

with torch.no_grad():
    y, past_states = model(next_token, past_states=past_states, use_kernel=True)
```

---

## Running Tests & Benchmarks

```bash
# 1. Numerical correctness (Varlen Fwd/Bwd, Decode L=1, E2E Handoff, CUDAGraph, torch.compile)
python tests/test_correctness.py
python tests/test_correctness.py --quick   # Fast smoke run

# 2. Autoregressive decode benchmark (L = 1)
python -u tests/bench_inference.py --dim 768 --batches 1,16,64,256,1024,2048 --warmup 5 --iters 20

# 3. Training pass benchmark (Fwd + Bwd)
python -u tests/bench_training.py --dim 768 --batch-size 4 --seq-lens 512,1024,2048,4096 --warmup 3 --iters 10
```

For interactive Colab/Kaggle execution and full end-to-end / profiling workflows, see [demo.ipynb](demo.ipynb) and [benchmark/deltanet.ipynb](benchmark/deltanet.ipynb).

---

## Project Structure

```text
Deltanet_v2/
├── include/                    # CUDA/CuTe headers (deltanet_common.cuh, deltanet_ops.cuh)
├── src/                        # Fused CUDA kernels & PyTorch dispatcher bindings
├── deltanet_v2/                # Python package (ops.py, model.py)
├── tests/                      # Correctness suites & inference/training benchmarks
├── benchmark/                  # Reproducible benchmark & profiling notebook (deltanet.ipynb)
├── assets/                     # Benchmark plots
├── wheels/                     # Pre-compiled .whl distribution
├── demo.ipynb                  # Google Colab / Kaggle quick-start notebook
├── setup.py                    # CMake/CUDA build with C++20/17 fallback
├── pyproject.toml
├── requirements.txt
└── LICENSE                     # Apache License 2.0
```

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
