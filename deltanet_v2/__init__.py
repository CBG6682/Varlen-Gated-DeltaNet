"""
DeltaNet v2: High-Performance Linear Attention CUDA Kernels & PyTorch Module.

Engineered with NVIDIA CUTLASS / CuTe specifically targeting Turing SM75 (Tesla T4).
"""

import os
import sys

# Extend package __path__ to also include installed site-packages/dist-packages paths
# when running from a local checkout, allowing discovery of compiled C-extensions (_C*.so).
try:
    import site
    sp_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    if hasattr(site, "getusersitepackages"):
        sp_dirs = list(sp_dirs) + [site.getusersitepackages()]
    for sp in sp_dirs:
        cand = os.path.join(sp, "deltanet_v2")
        if os.path.isdir(cand) and cand not in __path__:
            __path__.append(cand)
except Exception:
    pass

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
