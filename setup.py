import os
import subprocess
import sys
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_cutlass_include():
    """Locate or automatically clone NVIDIA CUTLASS repository."""
    # 1. User-specified CUTLASS path
    env_path = os.environ.get("CUTLASS_PATH")
    if env_path:
        cand = os.path.join(env_path, "include") if not env_path.endswith("include") else env_path
        if os.path.isdir(cand):
            return os.path.abspath(cand)

    # 2. Local cutlass/ directory
    local_cand = os.path.join(THIS_DIR, "cutlass", "include")
    if os.path.isdir(local_cand):
        return os.path.abspath(local_cand)

    # 3. Relative path inside parent development workspace (if applicable)
    parent_cand = os.path.abspath(os.path.join(THIS_DIR, "../../../../cutlass/include"))
    if os.path.isdir(parent_cand):
        return parent_cand

    # 4. Automatically clone shallow CUTLASS repository
    target_dir = os.path.join(THIS_DIR, "cutlass")
    print("CUTLASS not found. Cloning NVIDIA CUTLASS (shallow clone)...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", target_dir],
            check=True,
        )
        if os.path.isdir(os.path.join(target_dir, "include")):
            return os.path.abspath(os.path.join(target_dir, "include"))
    except Exception as e:
        print(f"Warning: Failed to clone CUTLASS automatically ({e}).")

    return os.path.abspath(os.path.join(THIS_DIR, "include"))


def get_cuda_arch_flags():
    """Determine targeted CUDA compute architectures."""
    # If user explicitly defined arch list via environment variable
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []

    # DeltaNet v2 kernels are specifically engineered with CuTe SM75 MMA atoms (16x8x8).
    # We generate native SASS for sm_75 and embed compute_75 PTX for driver forward compatibility.
    return [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_75,code=compute_75",
    ]


cutlass_include = get_cutlass_include()
include_dirs = [
    os.path.join(THIS_DIR, "include"),
    cutlass_include,
]

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
] + get_cuda_arch_flags()

sources = [
    os.path.join(THIS_DIR, "src", s)
    for s in [
        "binding.cpp",
        "chunk_wy_fwd.cu",
        "chunk_wy_bwd.cu",
        "chunk_scan_fwd.cu",
        "chunk_scan_bwd.cu",
        "output_combine_fwd.cu",
        "output_combine_bwd.cu",
        "deltanet.cu",
        "deltanet_infer.cu",
    ]
]

setup(
    name="deltanet_v2",
    version="2.0.0",
    description="High-Performance DeltaNet v2 Linear Attention CUDA Kernels & PyTorch Module",
    author="CBG6682",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="deltanet_v2._C",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
)
