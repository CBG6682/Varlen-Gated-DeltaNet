"""
DeltaNet v2: High-Performance Linear Attention CUDA Kernels & PyTorch Module.

Optimized with NVIDIA CUTLASS / CuTe for Turing SM75, Ampere SM80/SM86, Ada SM89, and Hopper SM90+.
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
