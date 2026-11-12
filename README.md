# DeltaNet v2: High-Performance Linear Attention CUDA Kernels

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware](https://img.shields.io/badge/GPU-Turing%20SM75%20(Tesla%20T4)-orange.svg)](https://developer.nvidia.com/cuda-gpus)

**DeltaNet v2** is a fast, hardware-optimized CUDA implementation of the DeltaNet linear attention mechanism featuring recurrent chunk-wise state updates (chunk WY representation), Grouped Query Attention (GQA), and multi-gate control (decay $\alpha$, erase $\beta$, write $W$, and L2 scaling $\gamma$).

Implemented using **NVIDIA CUTLASS / CuTe**, DeltaNet v2 is specifically tailored for **NVIDIA Turing SM75** architecture (e.g. Tesla T4 GPUs on Google Colab and Kaggle) using fused Tensor Core MMA instructions (`SM75_16x8x8_F32F16F16F32_TN`), swizzled shared memory layouts (`Swizzle<3,3,3>`) for zero bank conflicts, register-level epilogues, and tight memory reuse.

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
- Numerical Overflow / Underflow: None (`nan=False`, stable softplus/sigmoid)

---

## 🚀 Quick Start

### 1. Installation

#### Direct installation from GitHub (Google Colab / Kaggle / Local):
```bash
pip install git+https://github.com/CBG6682/Deltanet.git
```

#### From source:
```bash
git clone https://github.com/CBG6682/Deltanet.git
cd Deltanet
pip install -e .
```
> **Note**: If NVIDIA CUTLASS is not found on your system, `setup.py` automatically performs a shallow clone of CUTLASS. No manual clone or configuration needed.

---

### 2. Usage Example

```python
import torch
from deltanet_v2 import DeltaBlock

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize DeltaBlock
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

# Forward pass using high-performance CUDA kernel
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

## 🛠️ Kernel Architecture & CUDA Details

| Stage | Kernel | Tensor Core Instruction | Shared Memory Design |
|:---|:---|:---|:---|
| **WY Chunking** | `kernel_chunk_wy_fwd` | Elementwise & CuTe layout | Stationary norm & scalar caching |
| **Scan Recurrence** | `chunk_scan_fwd` | Batched GEMM recurrence | Chunk transition chaining |
| **Output Combine** | `kernel_output_combine_fwd` | `SM75_16x8x8_F32F16F16F32_TN` | 32 KB SMEM (4 x 8 KB buffers), `Swizzle<3,3,3>`, in-place register epilogue |
| **Decode Step (L=1)** | `kernel_deltanet_infer` | `SM75_16x8x8_F32F16F16F32_TN` | Fused state read-modify-write, multi-head parallel |
| **Backward Pass** | 3-kernel architecture | `LDSM_N`, `LDSM_T` | Recomputation of activations in SMEM, minimizing GMEM traffic |

### Hardware Architecture:
- **Primary Target**: **NVIDIA Turing SM75** (e.g. Tesla T4 on Google Colab / Kaggle, RTX 2060–2080 Ti, Quadro RTX).
- **Microarchitecture Specification**:
  - Kernels hardcode CuTe MMA atoms to **`SM75_16x8x8_F32F16F16F32_TN`** and S2R copy atoms to `SM75_U32x2_LDSM_N`, `SM75_U16x2_LDSM_T`.
  - Block dimensions ($d_k = 64, d_v = 64, \text{chunk\_size} = 64$) and shared memory budgets (32 KB forward, 56.25 KB backward) are strictly tuned to fit within Turing's 64 KB shared memory per-block ceiling.
- **Execution on Newer Architectures (Ampere SM80+, Ada SM89, Hopper SM90)**:
  - Through PTX forward compatibility (`-gencode=arch=compute_75,code=compute_75`), the SM75 instruction sequence can be JIT-executed by the NVIDIA driver on newer GPUs.
  - *Note*: It runs the SM75 $16 \times 8 \times 8$ Tensor Core instruction pipeline and does not use Ampere-specific $16 \times 8 \times 16$ MMA, asynchronous `cp.async`, or Hopper TMA.

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
├── demo.ipynb                    # Interactive tutorial notebook for Colab / Kaggle
├── setup.py                      # Build script with auto-CUTLASS discovery
├── pyproject.toml                # PEP 517/518 build specification
├── LICENSE                       # Apache 2.0 License
└── README.md
```

---

## 📜 License

This project is licensed under the **Apache License 2.0**. You are free to use, modify, and distribute this software for personal, academic, and commercial purposes. See [LICENSE](LICENSE) for details.
