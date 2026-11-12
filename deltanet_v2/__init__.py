"""
DeltaNet v2: High-Performance Linear Attention CUDA Kernels & PyTorch Module.

Engineered with NVIDIA CUTLASS / CuTe specifically targeting Turing SM75 (Tesla T4).
"""

from deltanet_v2.ops import (
    DeltaBlockKernelFn,
    apply_deltablock,
    apply_deltablock_infer,
    custom_ops,
)
from deltanet_v2.model import DeltaBlock

__version__ = "2.0.0"
__all__ = [
    "DeltaBlock",
    "DeltaBlockKernelFn",
    "apply_deltablock",
    "apply_deltablock_infer",
    "custom_ops",
]
