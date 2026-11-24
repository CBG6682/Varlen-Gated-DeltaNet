import os
import shutil
import subprocess
import sys
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Limit maximum parallel compilation jobs to avoid memory exhaustion and crashes
# on resource-constrained environments (e.g. Google Colab free tier with 2 vCPUs).
# CuTe template expansion requires significant host RAM (~1.5-2.0 GB per worker).
if "MAX_JOBS" not in os.environ:
    os.environ["MAX_JOBS"] = "2"
    print(f"[setup.py] Setting default MAX_JOBS={os.environ['MAX_JOBS']} to prevent CPU/RAM exhaustion")


def get_cutlass_include():
    """Locate or automatically clone NVIDIA CUTLASS repository."""
    # 1. Dedicated project-level third_party/cutlass (submodule or vendor directory)
    third_party_cutlass = os.path.join(THIS_DIR, "third_party", "cutlass")
    for cand in [
        os.path.join(third_party_cutlass, "include"),
        third_party_cutlass,
        os.path.join(THIS_DIR, "thirdparty", "cutlass", "include"),
        os.path.join(THIS_DIR, "thirdparty", "cutlass"),
    ]:
        if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
            print(f"[setup.py] Found dedicated CUTLASS in third_party: {cand}")
            return os.path.abspath(cand)

    # Auto-initialize git submodule third_party/cutlass if present but unpopulated
    if os.path.isfile(os.path.join(THIS_DIR, ".gitmodules")):
        try:
            print("[setup.py] Initializing dedicated third_party/cutlass submodule...")
            subprocess.run(
                ["git", "submodule", "update", "--init", "--depth", "1", "third_party/cutlass"],
                cwd=THIS_DIR,
                check=False,
            )
            cand = os.path.join(third_party_cutlass, "include")
            if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
                print(f"[setup.py] Submodule initialized successfully: {cand}")
                return os.path.abspath(cand)
        except Exception as e:
            print(f"[setup.py] Notice: Submodule update skipped ({e})")

    # 2. User-specified CUTLASS path via environment variable
    env_path = os.environ.get("CUTLASS_PATH")
    if env_path:
        for cand in [os.path.join(env_path, "include"), env_path]:
            if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
                print(f"[setup.py] Found CUTLASS from CUTLASS_PATH: {cand}")
                return os.path.abspath(cand)

    # 3. Google Colab standard locations
    for colab_cand in [
        "/content/Deltanet/third_party/cutlass/include",
        "/content/cutlass/include",
        "/content/cutlass",
        "/content/Deltanet/cutlass/include",
    ]:
        if os.path.isfile(os.path.join(colab_cand, "cute", "tensor.hpp")):
            print(f"[setup.py] Found CUTLASS in Colab path: {colab_cand}")
            return os.path.abspath(colab_cand)

    # 4. Local cutlass/ directory inside package
    local_cands = [
        os.path.join(THIS_DIR, "cutlass", "include"),
        os.path.join(THIS_DIR, "cutlass"),
        os.path.join(THIS_DIR, "..", "cutlass", "include"),
    ]
    for cand in local_cands:
        if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
            print(f"[setup.py] Found CUTLASS in local directory: {cand}")
            return os.path.abspath(cand)

    # 5. Automatically shallow-clone CUTLASS directly into dedicated third_party/cutlass
    target_dir = os.path.join(THIS_DIR, "third_party", "cutlass")
    print(f"[setup.py] CUTLASS (<cute/tensor.hpp>) not found. Shallow cloning to dedicated {target_dir}...")
    try:
        if os.path.exists(target_dir) and not os.path.isfile(os.path.join(target_dir, "include", "cute", "tensor.hpp")):
            import shutil
            print(f"[setup.py] Removing unpopulated target directory before clone: {target_dir}")
            shutil.rmtree(target_dir, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", target_dir],
            check=True,
        )
        for cand in [os.path.join(target_dir, "include"), target_dir]:
            if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
                print(f"[setup.py] Successfully cloned CUTLASS: {cand}")
                return os.path.abspath(cand)
    except Exception as e:
        print(f"[setup.py] Warning: Automatic git clone of CUTLASS failed: {e}")

    # Fallback to local include
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


def get_existing_prebuilt():
    """Find pre-compiled CUDA extension binary if present in deltanet_v2/ or extract from wheels/."""
    pkg_dir = os.path.join(THIS_DIR, "deltanet_v2")
    if os.path.isdir(pkg_dir):
        for f in os.listdir(pkg_dir):
            if f.startswith("_C") and f.endswith(".so"):
                return os.path.join(pkg_dir, f)

    # Check wheels/ or root directory for prebuilt wheel containing .so
    import glob
    import zipfile
    wheel_candidates = (
        glob.glob(os.path.join(THIS_DIR, "wheels", "*.whl"))
        + glob.glob(os.path.join(THIS_DIR, "*.whl"))
        + glob.glob(os.path.join(THIS_DIR, "dist", "*.whl"))
    )
    for whl in wheel_candidates:
        try:
            with zipfile.ZipFile(whl, "r") as z:
                for member in z.namelist():
                    if member.startswith("deltanet_v2/_C") and member.endswith(".so"):
                        print(f"[setup.py] Unpacking pre-compiled extension {member} from {whl}...")
                        z.extract(member, THIS_DIR)
                        extracted = os.path.join(THIS_DIR, member)
                        if os.path.isfile(extracted):
                            return extracted
        except Exception as e:
            print(f"[setup.py] Notice: Could not read wheel {whl} ({e})")
    return None


# Compilation mode options:
# 1. FORCE_REBUILD=1 or REBUILD=1: Force recompile from CUDA source and overwrite.
# 2. Existing prebuilt found and no rebuild requested: Use existing binary (fast install).
# 3. No prebuilt found: Default to compiling from source.
force_rebuild = os.environ.get("FORCE_REBUILD", os.environ.get("REBUILD", "0")) in ("1", "true", "True")
existing_prebuilt = get_existing_prebuilt()

if force_rebuild:
    print("[setup.py] FORCE_REBUILD=1 requested: Recompiling CUDA extension from source...")
    pkg_dir = os.path.join(THIS_DIR, "deltanet_v2")
    if os.path.isdir(pkg_dir):
        for fname in os.listdir(pkg_dir):
            if fname.startswith("_C") and fname.endswith(".so"):
                os.remove(os.path.join(pkg_dir, fname))
    build_dir = os.path.join(THIS_DIR, "build")
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)
    should_compile = True
elif existing_prebuilt is not None:
    print(f"[setup.py] Pre-compiled extension found: {os.path.basename(existing_prebuilt)}. Skipping compilation.")
    print("[setup.py] (Set FORCE_REBUILD=1 to recompile from source).")
    should_compile = False
else:
    print("[setup.py] No pre-compiled extension found. Defaulting to compile from source...")
    should_compile = True

ext_modules = []
cmdclass = {}

if should_compile:
    cutlass_include = get_cutlass_include()
    include_dirs = [
        os.path.join(THIS_DIR, "include"),
        cutlass_include,
    ]

    cxx_std = os.environ.get("CXX_STANDARD", "c++20")

    nvcc_flags = [
        "-O3",
        f"-std={cxx_std}",
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
            "block_ops.cu",
            "proj_conv_prep.cu",
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

    ext_modules = [
        CUDAExtension(
            name="deltanet_v2._C",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", f"-std={cxx_std}"],
                "nvcc": nvcc_flags,
            },
        ),
    ]

    class Cpp20WithCpp17FallbackBuildExtension(BuildExtension):
        """Compile with C++20 by default; fall back to C++17 only on build error."""

        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as exc:
                if os.environ.get("CXX_STANDARD") == "c++17":
                    raise
                print(f"[setup.py] C++20 build failed ({exc}). Falling back to C++17...")
                for ext in self.extensions:
                    for key in ("cxx", "nvcc"):
                        if key in ext.extra_compile_args:
                            ext.extra_compile_args[key] = [
                                "-std=c++17" if flag == "-std=c++20" else flag
                                for flag in ext.extra_compile_args[key]
                            ]
                if os.path.isdir(self.build_temp):
                    shutil.rmtree(self.build_temp, ignore_errors=True)
                super().build_extensions()

    cmdclass = {"build_ext": Cpp20WithCpp17FallbackBuildExtension}

setup(
    name="deltanet_v2",
    version="2.0.0",
    description="High-Performance DeltaNet v2 Linear Attention CUDA Kernels & PyTorch Module",
    author="CBG6682",
    packages=find_packages(include=["deltanet_v2", "deltanet_v2.*"]),
    package_data={"deltanet_v2": ["*.so"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.13",
    install_requires=[
        "torch>=2.0.0",
        "ninja",
    ],
)
