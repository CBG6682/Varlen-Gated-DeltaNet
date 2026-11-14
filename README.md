# DeltaNet v2: Fused Linear Attention CUDA Implementation for Turing GPUs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/Target%20GPU-Turing%20SM75%20(Tesla%20T4%20%2F%20RTX%2020--Series)-orange.svg)](https://developer.nvidia.com/cuda-gpus)

**DeltaNet v2** is a fast, hardware-optimized CUDA implementation of the linear attention mechanism with recurrent chunk-wise state updates (chunk WY representation) and Grouped Query Attention (GQA).

This project takes inspiration from NVIDIA's paper [**Gated DeltaNet v2** (arXiv:2605.22791)](https://arxiv.org/pdf/2605.22791), with an architectural modification to the gating structure: the gates for Key ($k$) and Value ($v$) are decoupled to operate independently, yielding a more effective state update and significantly optimizing autoregressive inference.

---

## 🎯 Motivation

Most existing open-source implementations of DeltaNet v2 and related linear attention variants focus almost exclusively on deep optimizations for high-end enterprise GPUs such as **NVIDIA A100 and H100+**. 

However, free and accessible cloud environments like **Google Colab and Kaggle** predominantly offer **NVIDIA Tesla T4** GPUs (Turing SM75), and many students, independent researchers, and developers work on consumer **RTX 20-series / 20 Super** hardware. 

This repository was created specifically to bridge that gap:
- **Tailored for Turing SM75**: Specifically engineered and tuned for the compute and shared memory constraints of Tesla T4 and RTX 20-series GPUs.
- **Accessible Experimentation**: Allows anyone to train, experiment with, and evaluate linear attention models on free Colab/Kaggle instances or accessible consumer cards without needing an A100 cluster.
- **Fused Kernel Pipelines**: Fuses multi-pass recurrences, normalizations, and epilogues into compact on-chip kernels to minimize global memory round-trips.

> **Note**: *The source code and annotations are provided for reference, study, and experimentation. The implementation was developed utilizing an agentic workflow with strict design rules to ensure mathematical correctness, tight register pressure control, and efficient on-chip memory utilization.*

---

## ⚡ Highlights

- **Up to 4.6x Faster Inference**: 42,258 tokens/s peak decode throughput on Tesla T4 (vs 9,017 tokens/s in eager PyTorch).
- **2.5x Faster Training**: Fused variable-length forward and backward recurrence with exact autograd support.
- **2.0x Higher Batch Capacity**: Reduces peak activation and intermediate memory footprint, doubling max batch size before OOM.
- **`torch.compile` Ready**: Built-in PyTorch Dispatcher integration with fake tensor registrations for seamless `torch.compile` compatibility.
- **1-Command Setup**: Self-contained installation on Google Colab, Kaggle, and local GPU machines without external dependencies.

---

## 📊 Benchmark Results

All benchmarks were evaluated on an **NVIDIA Tesla T4 GPU (16 GB VRAM, SM75)** using **PyTorch 2.x**:

### 1. Autoregressive Inference (Single Decode Step, $L = 1$)

Evaluating generation throughput and latency across varying batch sizes ($d_{\text{model}} = 768$, $H_q = 8$, $H_{kv} = 4$, $d_k = 64$, $d_v = 64$):

| Batch Size | Mode | Latency (ms) | Throughput (tok/s) | Peak VRAM (GB) | Speedup vs Torch Eager |
|:---:|:---|:---:|:---:|:---:|:---:|
| **16** | torch-eager | 10.55 | 1,516 | 0.33 | 1.00x |
| 16 | torch-compiled | 3.63 | 4,405 | 0.32 | 2.90x |
| 16 | **kernel-compiled** | **3.36** | **4,769** | **0.32** | **3.14x** |
| **64** | torch-eager | 9.94 | 6,438 | 0.42 | 1.00x |
| 64 | torch-compiled | 4.83 | 13,258 | 0.38 | 2.06x |
| 64 | **kernel-compiled** | **3.27** | **19,570** | **0.36** | **3.04x** |
| **256** | torch-eager | 29.45 | 8,693 | 0.81 | 1.00x |
| 256 | torch-compiled | 14.29 | 17,917 | 0.62 | 2.06x |
| 256 | **kernel-compiled** | **7.08** | **36,184** | **0.56** | **4.16x** |
| **1024** | torch-eager | 113.56 | 9,017 | 2.34 | 1.00x |
| 1024 | torch-compiled | 58.30 | 17,565 | 1.59 | 1.95x |
| 1024 | **kernel-eager** | **26.11** | **39,215** | **1.36** | **4.35x** |
| 1024 | **kernel-compiled** | **24.65** | **41,544** | **1.36** | **4.61x** |

#### Maximum Batch Size & Peak Throughput (Decode)
| Mode | Max Batch Size (before OOM) | Peak Throughput (tok/s) | Time for 256 tokens (s) |
|:---|:---:|:---:|:---:|
| torch-eager | 6,656 | 9,152 | 186.2s |
| torch-compiled | 10,752 | 19,570 | 140.6s |
| kernel-eager | 12,288 | 40,706 | 77.3s |
| **kernel-compiled** | **13,312 (2.0x capacity)** | **42,258 (4.62x speedup)** | **80.6s** |

---

### 2. Training (Forward + Backward Pass)

Evaluated with sequence length $L = 1024$:

| Batch Size | Mode | Step Time (ms) | Training Throughput (tok/s) | Saved Activations VRAM |
|:---:|:---|:---:|:---:|:---:|
| **4** | torch-eager | 411.2 | 9,961 | 4.46 GB |
| 4 | torch-compiled | 209.0 | 19,600 | 2.36 GB |
| 4 | **kernel-compiled** | **194.3** | **21,083** | **1.62 GB** (2.76x less) |
| **8** | torch-eager | 776.0 | 10,555 | 8.71 GB |
| 8 | torch-compiled | 363.0 | 22,596 | 4.53 GB |
| 8 | **kernel-compiled** | **345.0** | **23,765** | **3.02 GB** (2.88x less) |
| **16** | torch-eager | *OOM* | - | - |
| 16 | torch-compiled | 706.0 | 23,198 | 8.86 GB |
| 16 | **kernel-compiled** | **619.0** | **26,459** | **5.85 GB** |

#### Maximum Training Batch Size
- `torch-eager`: 11
- `torch-compiled`: 22
- `kernel-compiled`: **32** (2.9x capacity vs eager, 1.45x vs compiled)

#### Numerical Precision Verification
- Forward Output Cosine Similarity: `1.000000`
- Backward Gradient Cosine Similarity: `0.999993`
- Relative Error: `3.69e-03`
- Numerical Stability: Zero NaN/Inf occurrences across all sequence lengths.

---

## 📋 Requirements & Supported GPUs

### 1. Supported GPUs (Turing SM75 with Tensor Cores)
The CUDA kernels are specifically engineered for **NVIDIA Turing SM75** hardware:
- **Cloud & Free-tier Environments**:
  - **NVIDIA Tesla T4** (Default free GPU tier on **Google Colab** and **Kaggle**)
- **Consumer Desktops & Laptops (GeForce RTX 20 Series)**:
  - GeForce RTX 2080 Ti
  - GeForce RTX 2080 Super / RTX 2080
  - GeForce RTX 2070 Super / RTX 2070
  - GeForce RTX 2060 Super / RTX 2060
  - GeForce RTX 20-Series Mobile / Max-Q GPUs
- **Professional & Workstation GPUs**:
  - Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000
- *Forward Compatibility*: Embedded `compute_75` PTX allows the binaries to be JIT-executed on newer GPUs (Ampere SM80/SM86, Ada SM89, Hopper SM90).

### 2. Software Requirements (Before Compilation)
Before building and compiling the extension, ensure your system provides:
- **Operating System**: Linux (Ubuntu 20.04/22.04+, Debian, CentOS/RHEL)
- **Python**: `>= 3.8` (recommended `3.10` or `3.11`)
- **PyTorch**: `>= 2.0.0` with CUDA support (e.g. `torch >= 2.0.0+cu118` or `torch >= 2.1.0+cu121`)
- **CUDA Toolkit**: `>= 11.8` or `12.x` (with `nvcc` compiler installed and accessible in `$PATH`)
- **C++ Host Compiler**: `gcc` / `g++ >= 9.0` or `clang` supporting C++17
- **Python Build Dependencies**:
  ```bash
  pip install -r requirements.txt
  # or: pip install torch setuptools wheel ninja
  ```
  *(Installing `ninja` is strongly recommended for fast multi-threaded kernel compilation)*
- **NVIDIA CUTLASS / CuTe**:
  - Requires **CUTLASS >= 3.0** (specifically the header-only `<cute/tensor.hpp>`).
  - **Automatic Handling**: `setup.py` automatically performs a shallow clone of CUTLASS (`git clone --depth 1 https://github.com/NVIDIA/cutlass.git`) into `./cutlass` during build if CUTLASS is not found on your system.
  - **Custom CUTLASS Path (optional)**: If you already have CUTLASS installed locally, set:
    ```bash
    export CUTLASS_PATH=/path/to/cutlass
    ```

---

## 🚀 Quick Start

### 1. Installation

#### Direct installation from GitHub (Google Colab / Kaggle / Local):
```bash
pip install git+https://github.com/CBG6682/Varlen-Gated-DeltaNet.git
```

#### From source:
```bash
git clone --recursive https://github.com/CBG6682/Varlen-Gated-DeltaNet.git
cd Varlen-Gated-DeltaNet
pip install .
```
> **Note**: If NVIDIA CUTLASS is not found on your system, `setup.py` automatically performs a shallow clone of CUTLASS. No manual clone or configuration needed.

---

### 2. Usage Example

```python
import torch
from deltanet_v2 import DeltaBlock

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize DeltaBlock with independent key/value gating
model = DeltaBlock(
    d_model=768,
    num_q_heads=8,
    num_kv_heads=4,  # Supports GQA (num_q_heads % num_kv_heads == 0)
    d_k=64,
    d_v=64,
    chunk_size=64,
    hidden_dim=768 * 4,
).to(device=device, dtype=torch.float16)

# -------------------------------------------------------------
# Training (Variable-length Packed Sequences)
# -------------------------------------------------------------
model.train()

# 2 sequences of lengths [512, 1024] packed into total_L = 1536 tokens
total_L = 1536
x = torch.randn(total_L, 768, device=device, dtype=torch.float16, requires_grad=True)
cu_seqlens = torch.tensor([0, 512, 1536], dtype=torch.int32, device=device)

# Forward pass using fused CUDA kernel
out, final_states = model(x, cu_seqlens=cu_seqlens, use_kernel=True)

# Backward pass
loss = out.sum()
loss.backward()
print("Grad x shape:", x.grad.shape)  # torch.Size([1536, 768])

# -------------------------------------------------------------
# Autoregressive Token Generation (L = 1 decode step)
# -------------------------------------------------------------
model.eval()

batch_size = 64
next_token = torch.randn(batch_size, 1, 768, device=device, dtype=torch.float16)

# Initialize recurrent state: (B, num_kv_heads, d_v, d_k)
past_states = torch.zeros(batch_size, 4, 64, 64, device=device, dtype=torch.float16)

with torch.no_grad():
    y, past_states = model(next_token, past_states=past_states, use_kernel=True)

print("Decoded token shape:", y.shape)            # torch.Size([64, 1, 768])
print("Updated state shape:", past_states.shape)  # torch.Size([64, 4, 64, 64])
```

---

### 3. Accelerated with `torch.compile`

DeltaNet v2 kernels register fake/meta tensors with the PyTorch Dispatcher, enabling full graph capture under `torch.compile`:

```python
compiled_model = torch.compile(model)
out, states = compiled_model(x, cu_seqlens=cu_seqlens, use_kernel=True)
```

---

## 🔬 Benchmarking Against Baselines (NVLabs GatedDeltaNet-2 & FLA)

A comprehensive benchmark suite is provided in [`tests/`](file:///tests) to compare DeltaNet v2 against:
- **NVIDIA Official Baseline**: [`NVlabs/GatedDeltaNet-2`](https://github.com/nvlabs/gateddeltanet-2) (arXiv:2605.22791)
- **Flash Linear Attention**: [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) (`fla.layers.GatedDeltaNet2` & `fla.layers.GatedDeltaNet`)
- **PyTorch Reference**: Unfused eager baseline

The scripts measure both **latency (ms) / throughput (tokens/s)** and **peak VRAM consumption (allocated & reserved MB)**.

### Running in Google Colab (Root `/content`)

Open [`demo.ipynb`](file:///demo.ipynb) in Google Colab or run directly in bash:

```bash
# 1. Clone repository with dedicated third_party/cutlass submodule to /content/Deltanet
git clone --recursive https://github.com/CBG6682/Varlen-Gated-DeltaNet.git /content/Deltanet
cd /content/Deltanet

# 2. Install dependencies & DeltaNet v2
pip install ninja setuptools wheel einops
pip install . --no-build-isolation

# 3. Install baselines
pip install flash-linear-attention
pip install git+https://github.com/nvlabs/gateddeltanet-2.git --no-deps || true

# 4. Benchmark Single-Step Autoregressive Inference (L = 1)
python /content/Deltanet/tests/bench_inference.py --dim 768 --batches 1,16,64,256,1024,2048

# 5. Benchmark Training Pass (Forward + Backward)
python /content/Deltanet/tests/bench_training.py --dim 768 --batch-size 4 --seq-lens 512,1024,2048,4096

# 6. Run All Benchmarks
python /content/Deltanet/tests/bench_all.py
```

---

## 📂 Project Structure

```
Deltanet/
├── include/
│   ├── deltanet_common.cuh       # Math utilities, activations, warp reduction
│   └── deltanet_ops.cuh          # Kernel declarations & host coordinators
├── src/
│   ├── binding.cpp               # PyTorch dispatcher & Python C-extension
│   ├── chunk_wy_fwd.cu           # Chunk WY forward kernel
│   ├── chunk_wy_bwd.cu           # Chunk WY backward kernel
│   ├── chunk_scan_fwd.cu         # Recurrence scan forward kernel
│   ├── chunk_scan_bwd.cu         # Recurrence scan backward kernel
│   ├── output_combine_fwd.cu     # Output combination forward kernel
│   ├── output_combine_bwd.cu     # Output combination backward kernels
│   ├── deltanet.cu               # Forward & backward pipeline coordinators
│   └── deltanet_infer.cu         # Fast single-step (L = 1) inference kernel
├── deltanet_v2/
│   ├── __init__.py               # Package exports
│   ├── ops.py                    # Op dispatcher, fake tensor registrations, autograd
│   └── model.py                  # Full DeltaBlock PyTorch module
├── tests/
│   ├── utils.py                  # CUDA event timers, VRAM tracking, baseline loaders
│   ├── bench_inference.py        # Autoregressive decode step (L = 1) benchmark
│   ├── bench_training.py         # Forward + Backward training pass benchmark
│   ├── bench_all.py              # Consolidated master benchmark runner
│   └── test_correctness.py       # Numerical consistency verification vs PyTorch
├── third_party/
│   └── cutlass/                  # Dedicated NVIDIA CUTLASS / CuTe submodule (>= 3.0)
├── demo.ipynb                    # Colab / Kaggle runner notebook (/content)
├── setup.py                      # Build script with auto-CUTLASS discovery
├── pyproject.toml                # PEP 517/518 build specification
├── requirements.txt              # Build and runtime Python dependencies
├── LICENSE                       # Apache 2.0 License
└── README.md
```

---

## 📜 License

This project is licensed under the **Apache License 2.0**. You are free to use, modify, and distribute this software for personal, academic, and commercial purposes. See [LICENSE](LICENSE) for details.
