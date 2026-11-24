#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

using namespace cute;

// ============================================================================
// Part 1: Device Helpers and Global Kernels
// ============================================================================

// Race-free FP32 block reduction using separate per-warp and output SMEM slots
template <int NUM_THREADS>
__device__ __forceinline__ float block_reduce_sum(
    float val,
    float* s_warp,
    float* s_out
) {
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    val = warp_reduce_sum(val);
    if constexpr (NUM_WARPS == 1) {
        return __shfl_sync(0xffffffff, val, 0);
    } else {
        int lane_id = threadIdx.x & 31;
        int warp_id = threadIdx.x >> 5;
        if (lane_id == 0) {
            s_warp[warp_id] = val;
        }
        __syncthreads();
        if (warp_id == 0) {
            float wv = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
            wv = warp_reduce_sum(wv);
            if (lane_id == 0) {
                *s_out = wv;
            }
        }
        __syncthreads();
        return *s_out;
    }
}

// Single-tile RMSNorm forward kernel (D <= 8192)
template <
    typename T,
    int NUM_THREADS,
    int TILE_D,
    bool IS_TRAINING,
    typename TiledCopyRow
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_rmsnorm_fwd(
    const T* __restrict__ x_ptr,   // (L, D) [FP16] - Input activations
    const T* __restrict__ w_ptr,   // (D,) [FP16] - RMSNorm scale weight
    T*       __restrict__ out_ptr, // (L, D) [FP16] - Output activations
    float*   __restrict__ rstd_ptr,// (L,) [FP32] - Saved rstd (when IS_TRAINING)
    int64_t L,                     // scalar [int64] - Number of tokens
    int64_t D,                     // scalar [int64] - Hidden dimension
    float   eps                    // scalar [float] - Stability epsilon
) {
    // 1. Setup
    int64_t row = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    if (row >= L) return;

    __shared__ float s_warp[(NUM_THREADS / 32) > 1 ? (NUM_THREADS / 32) : 1];
    __shared__ float s_out;

    auto gX   = make_tensor(make_gmem_ptr(x_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gOut = make_tensor(make_gmem_ptr(out_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gW   = make_tensor(make_gmem_ptr(w_ptr),   make_layout(make_shape(Int<1>{}, D), make_stride(D, Int<1>{})));

    auto gX_tile   = local_tile(gX,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
    auto gOut_tile = local_tile(gOut, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
    auto gW_tile   = local_tile(gW,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, Int<0>{}));

    TiledCopyRow tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto tXgX   = thr_copy.partition_S(gX_tile);
    auto tXgW   = thr_copy.partition_S(gW_tile);
    auto tXgOut = thr_copy.partition_D(gOut_tile);

    auto tXrX = make_fragment_like(tXgX);
    auto tXrW = make_fragment_like(tXgW);
    auto tXp  = make_fragment_like<bool>(tXgX);

    const bool is_full = (D == TILE_D);
    if (!is_full) {
        auto cRow   = make_identity_tensor(make_shape(Int<1>{}, Int<TILE_D>{}));
        auto tXcRow = thr_copy.partition_S(cRow);
        #pragma unroll
        for (int m = 0; m < size<1>(tXp); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXp); ++n) {
                bool in_bounds = (static_cast<int64_t>(get<1>(tXcRow(Int<0>{}, m, n))) < D);
                #pragma unroll
                for (int v = 0; v < size<0>(tXp); ++v) {
                    tXp(v, m, n) = in_bounds;
                }
            }
        }
        clear(tXrX);
        clear(tXrW);
    }

    // 2. Load row and accumulate variance
    if (is_full) {
        cute::copy(tiled_copy, tXgX, tXrX);
        cute::copy(tiled_copy, tXgW, tXrW);
    } else {
        cute::copy_if(tiled_copy, tXp, tXgX, tXrX);
        cute::copy_if(tiled_copy, tXp, tXgW, tXrW);
    }

    float sum_sq = 0.0f;
    #pragma unroll
    for (int m = 0; m < size<1>(tXrX); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tXrX); ++n) {
            #pragma unroll
            for (int v = 0; v < size<0>(tXrX); ++v) {
                float xv = half_to_float(tXrX(v, m, n));
                sum_sq += xv * xv;
            }
        }
    }

    float total_sq = block_reduce_sum<NUM_THREADS>(sum_sq, s_warp, &s_out);
    float rstd = rsqrtf(total_sq / static_cast<float>(D) + eps);

    if constexpr (IS_TRAINING) {
        if (tid == 0) {
            rstd_ptr[row] = rstd;
        }
    }

    // 3. Scale and vectorized store
    #pragma unroll
    for (int m = 0; m < size<1>(tXrX); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tXrX); ++n) {
            #pragma unroll
            for (int v = 0; v < size<0>(tXrX); ++v) {
                float out_f = half_to_float(tXrX(v, m, n)) * rstd * half_to_float(tXrW(v, m, n));
                tXrX(v, m, n) = float_to_out<T>(out_f);
            }
        }
    }

    if (is_full) {
        cute::copy(tiled_copy, tXrX, tXgOut);
    } else {
        cute::copy_if(tiled_copy, tXp, tXrX, tXgOut);
    }
}

// Multi-tile RMSNorm forward kernel (D > 8192)
template <
    typename T,
    int NUM_THREADS,
    int TILE_D,
    bool IS_TRAINING,
    typename TiledCopyRow
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_rmsnorm_fwd_large(
    const T* __restrict__ x_ptr,   // (L, D) [FP16] - Input activations
    const T* __restrict__ w_ptr,   // (D,) [FP16] - RMSNorm scale weight
    T*       __restrict__ out_ptr, // (L, D) [FP16] - Output activations
    float*   __restrict__ rstd_ptr,// (L,) [FP32] - Saved rstd (when IS_TRAINING)
    int64_t L,                     // scalar [int64] - Number of tokens
    int64_t D,                     // scalar [int64] - Hidden dimension
    float   eps                    // scalar [float] - Stability epsilon
) {
    // 1. Setup
    int64_t row = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    if (row >= L) return;

    __shared__ float s_warp[(NUM_THREADS / 32) > 1 ? (NUM_THREADS / 32) : 1];
    __shared__ float s_out;

    const int64_t num_tiles = (D + TILE_D - 1) / TILE_D;
    const bool last_full = (D == num_tiles * TILE_D);
    const int64_t last_base = (num_tiles - 1) * TILE_D;

    auto gX   = make_tensor(make_gmem_ptr(x_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gOut = make_tensor(make_gmem_ptr(out_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gW   = make_tensor(make_gmem_ptr(w_ptr),   make_layout(make_shape(Int<1>{}, D), make_stride(D, Int<1>{})));

    TiledCopyRow tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);

    auto gSample = local_tile(gX, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, Int<0>{}));
    auto tXDummy = thr_copy.partition_S(gSample);
    auto tXrX    = make_fragment_like(tXDummy);
    auto tXrW    = make_fragment_like(tXDummy);
    auto tXpLast = make_fragment_like<bool>(tXDummy);

    if (!last_full) {
        auto cTile   = make_identity_tensor(make_shape(Int<1>{}, Int<TILE_D>{}));
        auto tXcTile = thr_copy.partition_S(cTile);
        #pragma unroll
        for (int m = 0; m < size<1>(tXpLast); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXpLast); ++n) {
                bool in_bounds = (last_base + static_cast<int64_t>(get<1>(tXcTile(Int<0>{}, m, n))) < D);
                #pragma unroll
                for (int v = 0; v < size<0>(tXpLast); ++v) {
                    tXpLast(v, m, n) = in_bounds;
                }
            }
        }
    }

    // 2. Pass 1: accumulate row variance
    float sum_sq = 0.0f;
    for (int64_t t = 0; t < num_tiles; ++t) {
        auto gX_tile = local_tile(gX, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
        auto tXgX    = thr_copy.partition_S(gX_tile);
        bool is_partial = (!last_full && t == num_tiles - 1);
        if (is_partial) {
            clear(tXrX);
            cute::copy_if(tiled_copy, tXpLast, tXgX, tXrX);
        } else {
            cute::copy(tiled_copy, tXgX, tXrX);
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tXrX); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXrX); ++n) {
                #pragma unroll
                for (int v = 0; v < size<0>(tXrX); ++v) {
                    float xv = half_to_float(tXrX(v, m, n));
                    sum_sq += xv * xv;
                }
            }
        }
    }

    float total_sq = block_reduce_sum<NUM_THREADS>(sum_sq, s_warp, &s_out);
    float rstd = rsqrtf(total_sq / static_cast<float>(D) + eps);

    if constexpr (IS_TRAINING) {
        if (tid == 0) {
            rstd_ptr[row] = rstd;
        }
    }

    // 3. Pass 2: normalize and store
    for (int64_t t = 0; t < num_tiles; ++t) {
        auto gX_tile   = local_tile(gX,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
        auto gW_tile   = local_tile(gW,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, t));
        auto gOut_tile = local_tile(gOut, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
        auto tXgX      = thr_copy.partition_S(gX_tile);
        auto tXgW      = thr_copy.partition_S(gW_tile);
        auto tXgOut    = thr_copy.partition_D(gOut_tile);

        bool is_partial = (!last_full && t == num_tiles - 1);
        if (is_partial) {
            clear(tXrX);
            clear(tXrW);
            cute::copy_if(tiled_copy, tXpLast, tXgX, tXrX);
            cute::copy_if(tiled_copy, tXpLast, tXgW, tXrW);
        } else {
            cute::copy(tiled_copy, tXgX, tXrX);
            cute::copy(tiled_copy, tXgW, tXrW);
        }

        #pragma unroll
        for (int m = 0; m < size<1>(tXrX); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXrX); ++n) {
                #pragma unroll
                for (int v = 0; v < size<0>(tXrX); ++v) {
                    float out_f = half_to_float(tXrX(v, m, n)) * rstd * half_to_float(tXrW(v, m, n));
                    tXrX(v, m, n) = float_to_out<T>(out_f);
                }
            }
        }

        if (is_partial) {
            cute::copy_if(tiled_copy, tXpLast, tXrX, tXgOut);
        } else {
            cute::copy(tiled_copy, tXrX, tXgOut);
        }
    }
}

// Single-tile persistent RMSNorm backward kernel (D <= 8192)
template <
    typename T,
    int NUM_THREADS,
    int TILE_D,
    typename TiledCopyRow
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_rmsnorm_bwd(
    const T*     __restrict__ grad_out_ptr,    // (L, D) [FP16] - Upstream gradient
    const T*     __restrict__ x_ptr,           // (L, D) [FP16] - Saved input activations
    const T*     __restrict__ w_ptr,           // (D,) [FP16] - RMSNorm scale weight
    const float* __restrict__ rstd_ptr,        // (L,) [FP32] - Saved rstd
    T*           __restrict__ grad_x_ptr,      // (L, D) [FP16] - Output input gradient
    float*       __restrict__ grad_w_fp32_ptr, // (D,) [FP32] - Output weight grad accumulator
    int64_t L,                                 // scalar [int64] - Number of tokens
    int64_t D                                  // scalar [int64] - Hidden dimension
) {
    // 1. Setup
    int tid = threadIdx.x;
    __shared__ float s_warp[(NUM_THREADS / 32) > 1 ? (NUM_THREADS / 32) : 1];
    __shared__ float s_out;

    auto gDY = make_tensor(make_gmem_ptr(grad_out_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gX  = make_tensor(make_gmem_ptr(x_ptr),        make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gDX = make_tensor(make_gmem_ptr(grad_x_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gW  = make_tensor(make_gmem_ptr(w_ptr),        make_layout(make_shape(Int<1>{}, D), make_stride(D, Int<1>{})));

    auto gW_tile = local_tile(gW, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, Int<0>{}));
    TiledCopyRow tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto tXgW     = thr_copy.partition_S(gW_tile);

    auto tXrW  = make_fragment_like(tXgW);
    auto tXrX  = make_fragment_like(tXgW);
    auto tXrDY = make_fragment_like(tXgW);
    auto r_dW  = make_fragment_like<float>(tXgW);
    auto tXp   = make_fragment_like<bool>(tXgW);
    clear(r_dW);

    auto cRow   = make_identity_tensor(make_shape(Int<1>{}, Int<TILE_D>{}));
    auto tXcRow = thr_copy.partition_S(cRow);

    const bool is_full = (D == TILE_D);
    if (!is_full) {
        #pragma unroll
        for (int m = 0; m < size<1>(tXp); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXp); ++n) {
                bool in_bounds = (static_cast<int64_t>(get<1>(tXcRow(Int<0>{}, m, n))) < D);
                #pragma unroll
                for (int v = 0; v < size<0>(tXp); ++v) {
                    tXp(v, m, n) = in_bounds;
                }
            }
        }
        clear(tXrW);
        clear(tXrX);
        clear(tXrDY);
        cute::copy_if(tiled_copy, tXp, tXgW, tXrW);
    } else {
        cute::copy(tiled_copy, tXgW, tXrW);
    }

    const float inv_D = 1.0f / static_cast<float>(D);

    // 2. Persistent row loop
    for (int64_t row = static_cast<int64_t>(blockIdx.x); row < L; row += static_cast<int64_t>(gridDim.x)) {
        float rstd = rstd_ptr[row];

        auto gX_tile  = local_tile(gX,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
        auto gDY_tile = local_tile(gDY, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
        auto gDX_tile = local_tile(gDX, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
        auto tXgX     = thr_copy.partition_S(gX_tile);
        auto tXgDY    = thr_copy.partition_S(gDY_tile);
        auto tXgDX    = thr_copy.partition_D(gDX_tile);

        if (is_full) {
            cute::copy(tiled_copy, tXgX, tXrX);
            cute::copy(tiled_copy, tXgDY, tXrDY);
        } else {
            cute::copy_if(tiled_copy, tXp, tXgX, tXrX);
            cute::copy_if(tiled_copy, tXp, tXgDY, tXrDY);
        }

        float local_dot = 0.0f;
        #pragma unroll
        for (int m = 0; m < size<1>(tXrX); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXrX); ++n) {
                #pragma unroll
                for (int v = 0; v < size<0>(tXrX); ++v) {
                    float x_norm = half_to_float(tXrX(v, m, n)) * rstd;
                    float dy     = half_to_float(tXrDY(v, m, n));
                    float w      = half_to_float(tXrW(v, m, n));
                    local_dot   += (dy * w) * x_norm;
                    r_dW(v, m, n) += dy * x_norm;
                }
            }
        }

        float mean_dot = block_reduce_sum<NUM_THREADS>(local_dot, s_warp, &s_out) * inv_D;

        #pragma unroll
        for (int m = 0; m < size<1>(tXrX); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXrX); ++n) {
                #pragma unroll
                for (int v = 0; v < size<0>(tXrX); ++v) {
                    float x_norm = half_to_float(tXrX(v, m, n)) * rstd;
                    float g      = half_to_float(tXrDY(v, m, n)) * half_to_float(tXrW(v, m, n));
                    float dx     = rstd * (g - x_norm * mean_dot);
                    tXrDY(v, m, n) = float_to_out<T>(dx);
                }
            }
        }

        if (is_full) {
            cute::copy(tiled_copy, tXrDY, tXgDX);
        } else {
            cute::copy_if(tiled_copy, tXp, tXrDY, tXgDX);
        }
    }

    // 3. Flush accumulated dW once per persistent CTA
    #pragma unroll
    for (int m = 0; m < size<1>(r_dW); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(r_dW); ++n) {
            #pragma unroll
            for (int v = 0; v < size<0>(r_dW); ++v) {
                int64_t col = static_cast<int64_t>(get<1>(tXcRow(v, m, n)));
                if (col < D) {
                    atomicAdd(grad_w_fp32_ptr + col, r_dW(v, m, n));
                }
            }
        }
    }
}

// Multi-tile persistent RMSNorm backward kernel (D > 8192)
template <
    typename T,
    int NUM_THREADS,
    int TILE_D,
    typename TiledCopyRow
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_rmsnorm_bwd_large(
    const T*     __restrict__ grad_out_ptr,    // (L, D) [FP16] - Upstream gradient
    const T*     __restrict__ x_ptr,           // (L, D) [FP16] - Saved input activations
    const T*     __restrict__ w_ptr,           // (D,) [FP16] - RMSNorm scale weight
    const float* __restrict__ rstd_ptr,        // (L,) [FP32] - Saved rstd
    T*           __restrict__ grad_x_ptr,      // (L, D) [FP16] - Output input gradient
    float*       __restrict__ grad_w_fp32_ptr, // (D,) [FP32] - Output weight grad accumulator
    int64_t L,                                 // scalar [int64] - Number of tokens
    int64_t D                                  // scalar [int64] - Hidden dimension
) {
    // 1. Setup
    int tid = threadIdx.x;
    __shared__ float s_warp[(NUM_THREADS / 32) > 1 ? (NUM_THREADS / 32) : 1];
    __shared__ float s_out;

    const int64_t num_tiles = (D + TILE_D - 1) / TILE_D;
    const bool last_full = (D == num_tiles * TILE_D);
    const int64_t last_base = (num_tiles - 1) * TILE_D;
    const float inv_D = 1.0f / static_cast<float>(D);

    auto gDY = make_tensor(make_gmem_ptr(grad_out_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gX  = make_tensor(make_gmem_ptr(x_ptr),        make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gDX = make_tensor(make_gmem_ptr(grad_x_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gW  = make_tensor(make_gmem_ptr(w_ptr),        make_layout(make_shape(Int<1>{}, D), make_stride(D, Int<1>{})));

    TiledCopyRow tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto gSample  = local_tile(gX, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, Int<0>{}));
    auto tXDummy  = thr_copy.partition_S(gSample);

    auto tXrX    = make_fragment_like(tXDummy);
    auto tXrDY   = make_fragment_like(tXDummy);
    auto tXrW    = make_fragment_like(tXDummy);
    auto tXpLast = make_fragment_like<bool>(tXDummy);

    auto cTile   = make_identity_tensor(make_shape(Int<1>{}, Int<TILE_D>{}));
    auto tXcTile = thr_copy.partition_S(cTile);

    if (!last_full) {
        #pragma unroll
        for (int m = 0; m < size<1>(tXpLast); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXpLast); ++n) {
                bool in_bounds = (last_base + static_cast<int64_t>(get<1>(tXcTile(Int<0>{}, m, n))) < D);
                #pragma unroll
                for (int v = 0; v < size<0>(tXpLast); ++v) {
                    tXpLast(v, m, n) = in_bounds;
                }
            }
        }
    }

    // 2. Persistent row loop
    for (int64_t row = static_cast<int64_t>(blockIdx.x); row < L; row += static_cast<int64_t>(gridDim.x)) {
        float rstd = rstd_ptr[row];

        float local_dot = 0.0f;
        for (int64_t t = 0; t < num_tiles; ++t) {
            auto gX_tile  = local_tile(gX,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto gDY_tile = local_tile(gDY, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto gW_tile  = local_tile(gW,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, t));
            auto tXgX     = thr_copy.partition_S(gX_tile);
            auto tXgDY    = thr_copy.partition_S(gDY_tile);
            auto tXgW     = thr_copy.partition_S(gW_tile);

            bool is_partial = (!last_full && t == num_tiles - 1);
            if (is_partial) {
                clear(tXrX);
                clear(tXrDY);
                clear(tXrW);
                cute::copy_if(tiled_copy, tXpLast, tXgX, tXrX);
                cute::copy_if(tiled_copy, tXpLast, tXgDY, tXrDY);
                cute::copy_if(tiled_copy, tXpLast, tXgW, tXrW);
            } else {
                cute::copy(tiled_copy, tXgX, tXrX);
                cute::copy(tiled_copy, tXgDY, tXrDY);
                cute::copy(tiled_copy, tXgW, tXrW);
            }

            #pragma unroll
            for (int m = 0; m < size<1>(tXrX); ++m) {
                #pragma unroll
                for (int n = 0; n < size<2>(tXrX); ++n) {
                    #pragma unroll
                    for (int v = 0; v < size<0>(tXrX); ++v) {
                        float x_norm = half_to_float(tXrX(v, m, n)) * rstd;
                        float dy     = half_to_float(tXrDY(v, m, n));
                        float w      = half_to_float(tXrW(v, m, n));
                        local_dot   += (dy * w) * x_norm;
                    }
                }
            }
        }

        float mean_dot = block_reduce_sum<NUM_THREADS>(local_dot, s_warp, &s_out) * inv_D;

        for (int64_t t = 0; t < num_tiles; ++t) {
            auto gX_tile  = local_tile(gX,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto gDY_tile = local_tile(gDY, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto gW_tile  = local_tile(gW,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, t));
            auto gDX_tile = local_tile(gDX, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto tXgX     = thr_copy.partition_S(gX_tile);
            auto tXgDY    = thr_copy.partition_S(gDY_tile);
            auto tXgW     = thr_copy.partition_S(gW_tile);
            auto tXgDX    = thr_copy.partition_D(gDX_tile);

            bool is_partial = (!last_full && t == num_tiles - 1);
            if (is_partial) {
                clear(tXrX);
                clear(tXrDY);
                clear(tXrW);
                cute::copy_if(tiled_copy, tXpLast, tXgX, tXrX);
                cute::copy_if(tiled_copy, tXpLast, tXgDY, tXrDY);
                cute::copy_if(tiled_copy, tXpLast, tXgW, tXrW);
            } else {
                cute::copy(tiled_copy, tXgX, tXrX);
                cute::copy(tiled_copy, tXgDY, tXrDY);
                cute::copy(tiled_copy, tXgW, tXrW);
            }

            int64_t tile_col_base = t * TILE_D;
            #pragma unroll
            for (int m = 0; m < size<1>(tXrX); ++m) {
                #pragma unroll
                for (int n = 0; n < size<2>(tXrX); ++n) {
                    #pragma unroll
                    for (int v = 0; v < size<0>(tXrX); ++v) {
                        float x_norm = half_to_float(tXrX(v, m, n)) * rstd;
                        float dy     = half_to_float(tXrDY(v, m, n));
                        float w      = half_to_float(tXrW(v, m, n));
                        float dx     = rstd * (dy * w - x_norm * mean_dot);
                        tXrDY(v, m, n) = float_to_out<T>(dx);

                        if (!is_partial || tXpLast(v, m, n)) {
                            int64_t col = tile_col_base + static_cast<int64_t>(get<1>(tXcTile(v, m, n)));
                            atomicAdd(grad_w_fp32_ptr + col, dy * x_norm);
                        }
                    }
                }
            }

            if (is_partial) {
                cute::copy_if(tiled_copy, tXpLast, tXrDY, tXgDX);
            } else {
                cute::copy(tiled_copy, tXrDY, tXgDX);
            }
        }
    }
}

// Single-tile fused Residual + RMSNorm forward kernel (D <= 8192)
template <
    typename T,
    int NUM_THREADS,
    int TILE_D,
    bool IS_TRAINING,
    typename TiledCopyRow
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_residual_rmsnorm_fwd(
    const T* __restrict__ x_ptr,     // (L, D) [FP16] - Residual branch input
    const T* __restrict__ res_ptr,   // (L, D) [FP16] - Branch output to add
    const T* __restrict__ w_ptr,     // (D,) [FP16] - RMSNorm scale weight
    T*       __restrict__ h_ptr,     // (L, D) [FP16] - Sum residual h = x + residual_in
    T*       __restrict__ out_ptr,   // (L, D) [FP16] - Normalized output x_norm
    float*   __restrict__ rstd_ptr,  // (L,) [FP32] - Saved rstd (when IS_TRAINING)
    int64_t L,                       // scalar [int64] - Number of tokens
    int64_t D,                       // scalar [int64] - Hidden dimension
    float   eps                      // scalar [float] - Stability epsilon
) {
    // 1. Setup
    int64_t row = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    if (row >= L) return;

    __shared__ float s_warp[(NUM_THREADS / 32) > 1 ? (NUM_THREADS / 32) : 1];
    __shared__ float s_out;

    auto gX   = make_tensor(make_gmem_ptr(x_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gRes = make_tensor(make_gmem_ptr(res_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gH   = make_tensor(make_gmem_ptr(h_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gOut = make_tensor(make_gmem_ptr(out_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gW   = make_tensor(make_gmem_ptr(w_ptr),   make_layout(make_shape(Int<1>{}, D), make_stride(D, Int<1>{})));

    auto gX_tile   = local_tile(gX,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
    auto gRes_tile = local_tile(gRes, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
    auto gH_tile   = local_tile(gH,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
    auto gOut_tile = local_tile(gOut, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
    auto gW_tile   = local_tile(gW,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, Int<0>{}));

    TiledCopyRow tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto tXgX     = thr_copy.partition_S(gX_tile);
    auto tXgRes   = thr_copy.partition_S(gRes_tile);
    auto tXgW     = thr_copy.partition_S(gW_tile);
    auto tXgH     = thr_copy.partition_D(gH_tile);
    auto tXgOut   = thr_copy.partition_D(gOut_tile);

    auto tXrX   = make_fragment_like(tXgX);
    auto tXrRes = make_fragment_like(tXgRes);
    auto tXrW   = make_fragment_like(tXgW);
    auto tXp    = make_fragment_like<bool>(tXgX);

    const bool is_full = (D == TILE_D);
    if (!is_full) {
        auto cRow   = make_identity_tensor(make_shape(Int<1>{}, Int<TILE_D>{}));
        auto tXcRow = thr_copy.partition_S(cRow);
        #pragma unroll
        for (int m = 0; m < size<1>(tXp); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXp); ++n) {
                bool in_bounds = (static_cast<int64_t>(get<1>(tXcRow(Int<0>{}, m, n))) < D);
                #pragma unroll
                for (int v = 0; v < size<0>(tXp); ++v) {
                    tXp(v, m, n) = in_bounds;
                }
            }
        }
        clear(tXrX);
        clear(tXrRes);
        clear(tXrW);
        cute::copy_if(tiled_copy, tXp, tXgX, tXrX);
        cute::copy_if(tiled_copy, tXp, tXgRes, tXrRes);
        cute::copy_if(tiled_copy, tXp, tXgW, tXrW);
    } else {
        cute::copy(tiled_copy, tXgX, tXrX);
        cute::copy(tiled_copy, tXgRes, tXrRes);
        cute::copy(tiled_copy, tXgW, tXrW);
    }

    // 2. Add residual in FP16 pairs and accumulate FP32 variance of h
    float sum_sq = 0.0f;
    #pragma unroll
    for (int m = 0; m < size<1>(tXrX); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tXrX); ++n) {
            #pragma unroll
            for (int v = 0; v < size<0>(tXrX); v += 2) {
                __half2 x_h2 = __halves2half2(to_half_val(tXrX(v, m, n)),   to_half_val(tXrX(v + 1, m, n)));
                __half2 r_h2 = __halves2half2(to_half_val(tXrRes(v, m, n)), to_half_val(tXrRes(v + 1, m, n)));
                __half2 h_h2 = __hadd2(x_h2, r_h2);

                tXrX(v, m, n)     = half_to_out<T>(__low2half(h_h2));
                tXrX(v + 1, m, n) = half_to_out<T>(__high2half(h_h2));

                float h0 = __half2float(__low2half(h_h2));
                float h1 = __half2float(__high2half(h_h2));
                sum_sq += h0 * h0 + h1 * h1;
            }
        }
    }

    float total_sq = block_reduce_sum<NUM_THREADS>(sum_sq, s_warp, &s_out);
    float rstd = rsqrtf(total_sq / static_cast<float>(D) + eps);

    if constexpr (IS_TRAINING) {
        if (tid == 0) {
            rstd_ptr[row] = rstd;
        }
    }

    // 3. Store h and normalized output
    #pragma unroll
    for (int m = 0; m < size<1>(tXrRes); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tXrRes); ++n) {
            #pragma unroll
            for (int v = 0; v < size<0>(tXrRes); ++v) {
                float out_f = half_to_float(tXrX(v, m, n)) * rstd * half_to_float(tXrW(v, m, n));
                tXrRes(v, m, n) = float_to_out<T>(out_f);
            }
        }
    }

    if (is_full) {
        cute::copy(tiled_copy, tXrX, tXgH);
        cute::copy(tiled_copy, tXrRes, tXgOut);
    } else {
        cute::copy_if(tiled_copy, tXp, tXrX, tXgH);
        cute::copy_if(tiled_copy, tXp, tXrRes, tXgOut);
    }
}

// Multi-tile fused Residual + RMSNorm forward kernel (D > 8192)
template <
    typename T,
    int NUM_THREADS,
    int TILE_D,
    bool IS_TRAINING,
    typename TiledCopyRow
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_residual_rmsnorm_fwd_large(
    const T* __restrict__ x_ptr,     // (L, D) [FP16] - Residual branch input
    const T* __restrict__ res_ptr,   // (L, D) [FP16] - Branch output to add
    const T* __restrict__ w_ptr,     // (D,) [FP16] - RMSNorm scale weight
    T*       __restrict__ h_ptr,     // (L, D) [FP16] - Sum residual h = x + residual_in
    T*       __restrict__ out_ptr,   // (L, D) [FP16] - Normalized output x_norm
    float*   __restrict__ rstd_ptr,  // (L,) [FP32] - Saved rstd (when IS_TRAINING)
    int64_t L,                       // scalar [int64] - Number of tokens
    int64_t D,                       // scalar [int64] - Hidden dimension
    float   eps                      // scalar [float] - Stability epsilon
) {
    // 1. Setup
    int64_t row = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    if (row >= L) return;

    __shared__ float s_warp[(NUM_THREADS / 32) > 1 ? (NUM_THREADS / 32) : 1];
    __shared__ float s_out;

    const int64_t num_tiles = (D + TILE_D - 1) / TILE_D;
    const bool last_full = (D == num_tiles * TILE_D);
    const int64_t last_base = (num_tiles - 1) * TILE_D;

    auto gX   = make_tensor(make_gmem_ptr(x_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gRes = make_tensor(make_gmem_ptr(res_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gH   = make_tensor(make_gmem_ptr(h_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gOut = make_tensor(make_gmem_ptr(out_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gW   = make_tensor(make_gmem_ptr(w_ptr),   make_layout(make_shape(Int<1>{}, D), make_stride(D, Int<1>{})));

    TiledCopyRow tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto gSample  = local_tile(gX, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, Int<0>{}));
    auto tXDummy  = thr_copy.partition_S(gSample);

    auto tXrX    = make_fragment_like(tXDummy);
    auto tXrRes  = make_fragment_like(tXDummy);
    auto tXrW    = make_fragment_like(tXDummy);
    auto tXpLast = make_fragment_like<bool>(tXDummy);

    if (!last_full) {
        auto cTile   = make_identity_tensor(make_shape(Int<1>{}, Int<TILE_D>{}));
        auto tXcTile = thr_copy.partition_S(cTile);
        #pragma unroll
        for (int m = 0; m < size<1>(tXpLast); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXpLast); ++n) {
                bool in_bounds = (last_base + static_cast<int64_t>(get<1>(tXcTile(Int<0>{}, m, n))) < D);
                #pragma unroll
                for (int v = 0; v < size<0>(tXpLast); ++v) {
                    tXpLast(v, m, n) = in_bounds;
                }
            }
        }
    }

    // 2. Pass 1: store h and accumulate variance
    float sum_sq = 0.0f;
    for (int64_t t = 0; t < num_tiles; ++t) {
        auto gX_tile   = local_tile(gX,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
        auto gRes_tile = local_tile(gRes, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
        auto gH_tile   = local_tile(gH,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
        auto tXgX      = thr_copy.partition_S(gX_tile);
        auto tXgRes    = thr_copy.partition_S(gRes_tile);
        auto tXgH      = thr_copy.partition_D(gH_tile);

        bool is_partial = (!last_full && t == num_tiles - 1);
        if (is_partial) {
            clear(tXrX);
            clear(tXrRes);
            cute::copy_if(tiled_copy, tXpLast, tXgX, tXrX);
            cute::copy_if(tiled_copy, tXpLast, tXgRes, tXrRes);
        } else {
            cute::copy(tiled_copy, tXgX, tXrX);
            cute::copy(tiled_copy, tXgRes, tXrRes);
        }

        #pragma unroll
        for (int m = 0; m < size<1>(tXrX); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXrX); ++n) {
                #pragma unroll
                for (int v = 0; v < size<0>(tXrX); v += 2) {
                    __half2 x_h2 = __halves2half2(to_half_val(tXrX(v, m, n)),   to_half_val(tXrX(v + 1, m, n)));
                    __half2 r_h2 = __halves2half2(to_half_val(tXrRes(v, m, n)), to_half_val(tXrRes(v + 1, m, n)));
                    __half2 h_h2 = __hadd2(x_h2, r_h2);

                    tXrX(v, m, n)     = half_to_out<T>(__low2half(h_h2));
                    tXrX(v + 1, m, n) = half_to_out<T>(__high2half(h_h2));

                    float h0 = __half2float(__low2half(h_h2));
                    float h1 = __half2float(__high2half(h_h2));
                    sum_sq += h0 * h0 + h1 * h1;
                }
            }
        }

        if (is_partial) {
            cute::copy_if(tiled_copy, tXpLast, tXrX, tXgH);
        } else {
            cute::copy(tiled_copy, tXrX, tXgH);
        }
    }

    float total_sq = block_reduce_sum<NUM_THREADS>(sum_sq, s_warp, &s_out);
    float rstd = rsqrtf(total_sq / static_cast<float>(D) + eps);

    if constexpr (IS_TRAINING) {
        if (tid == 0) {
            rstd_ptr[row] = rstd;
        }
    }

    // 3. Pass 2: normalize h and store x_norm
    for (int64_t t = 0; t < num_tiles; ++t) {
        auto gH_tile   = local_tile(gH,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
        auto gW_tile   = local_tile(gW,   make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, t));
        auto gOut_tile = local_tile(gOut, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
        auto tXgH      = thr_copy.partition_S(gH_tile);
        auto tXgW      = thr_copy.partition_S(gW_tile);
        auto tXgOut    = thr_copy.partition_D(gOut_tile);

        bool is_partial = (!last_full && t == num_tiles - 1);
        if (is_partial) {
            clear(tXrX);
            clear(tXrW);
            cute::copy_if(tiled_copy, tXpLast, tXgH, tXrX);
            cute::copy_if(tiled_copy, tXpLast, tXgW, tXrW);
        } else {
            cute::copy(tiled_copy, tXgH, tXrX);
            cute::copy(tiled_copy, tXgW, tXrW);
        }

        #pragma unroll
        for (int m = 0; m < size<1>(tXrX); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXrX); ++n) {
                #pragma unroll
                for (int v = 0; v < size<0>(tXrX); ++v) {
                    float out_f = half_to_float(tXrX(v, m, n)) * rstd * half_to_float(tXrW(v, m, n));
                    tXrX(v, m, n) = float_to_out<T>(out_f);
                }
            }
        }

        if (is_partial) {
            cute::copy_if(tiled_copy, tXpLast, tXrX, tXgOut);
        } else {
            cute::copy(tiled_copy, tXrX, tXgOut);
        }
    }
}

// Single-tile persistent Residual + RMSNorm backward kernel (D <= 8192)
template <
    typename T,
    int NUM_THREADS,
    int TILE_D,
    typename TiledCopyRow
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_residual_rmsnorm_bwd(
    const T*     __restrict__ grad_norm_ptr,   // (L, D) [FP16] - Gradient from normalized branch
    const T*     __restrict__ grad_h_out_ptr,  // (L, D) [FP16] - Gradient from residual skip branch
    const T*     __restrict__ h_ptr,           // (L, D) [FP16] - Saved residual sum h
    const T*     __restrict__ w_ptr,           // (D,) [FP16] - RMSNorm scale weight
    const float* __restrict__ rstd_ptr,        // (L,) [FP32] - Saved rstd
    T*           __restrict__ grad_h_total_ptr,// (L, D) [FP16] - Output combined grad_h
    float*       __restrict__ grad_w_fp32_ptr, // (D,) [FP32] - Output weight grad accumulator
    int64_t L,                                 // scalar [int64] - Number of tokens
    int64_t D                                  // scalar [int64] - Hidden dimension
) {
    // 1. Setup
    int tid = threadIdx.x;
    __shared__ float s_warp[(NUM_THREADS / 32) > 1 ? (NUM_THREADS / 32) : 1];
    __shared__ float s_out;

    auto gGN = make_tensor(make_gmem_ptr(grad_norm_ptr),    make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gGH = make_tensor(make_gmem_ptr(grad_h_out_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gH  = make_tensor(make_gmem_ptr(h_ptr),            make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gGO = make_tensor(make_gmem_ptr(grad_h_total_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gW  = make_tensor(make_gmem_ptr(w_ptr),            make_layout(make_shape(Int<1>{}, D), make_stride(D, Int<1>{})));

    auto gW_tile = local_tile(gW, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, Int<0>{}));
    TiledCopyRow tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto tXgW     = thr_copy.partition_S(gW_tile);

    auto tXrW        = make_fragment_like(tXgW);
    auto tXrH        = make_fragment_like(tXgW);
    auto tXrGradNorm = make_fragment_like(tXgW);
    auto tXrGradH    = make_fragment_like(tXgW);
    auto r_dW        = make_fragment_like<float>(tXgW);
    auto tXp         = make_fragment_like<bool>(tXgW);
    clear(r_dW);

    auto cRow   = make_identity_tensor(make_shape(Int<1>{}, Int<TILE_D>{}));
    auto tXcRow = thr_copy.partition_S(cRow);

    const bool is_full = (D == TILE_D);
    if (!is_full) {
        #pragma unroll
        for (int m = 0; m < size<1>(tXp); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXp); ++n) {
                bool in_bounds = (static_cast<int64_t>(get<1>(tXcRow(Int<0>{}, m, n))) < D);
                #pragma unroll
                for (int v = 0; v < size<0>(tXp); ++v) {
                    tXp(v, m, n) = in_bounds;
                }
            }
        }
        clear(tXrW);
        clear(tXrH);
        clear(tXrGradNorm);
        clear(tXrGradH);
        cute::copy_if(tiled_copy, tXp, tXgW, tXrW);
    } else {
        cute::copy(tiled_copy, tXgW, tXrW);
    }

    const float inv_D = 1.0f / static_cast<float>(D);

    // 2. Persistent row loop
    for (int64_t row = static_cast<int64_t>(blockIdx.x); row < L; row += static_cast<int64_t>(gridDim.x)) {
        float rstd = rstd_ptr[row];

        auto gH_tile  = local_tile(gH,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
        auto gGN_tile = local_tile(gGN, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
        auto gGH_tile = local_tile(gGH, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));
        auto gGO_tile = local_tile(gGO, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, Int<0>{}));

        auto tXgH  = thr_copy.partition_S(gH_tile);
        auto tXgGN = thr_copy.partition_S(gGN_tile);
        auto tXgGH = thr_copy.partition_S(gGH_tile);
        auto tXgGO = thr_copy.partition_D(gGO_tile);

        if (is_full) {
            cute::copy(tiled_copy, tXgH, tXrH);
            cute::copy(tiled_copy, tXgGN, tXrGradNorm);
            cute::copy(tiled_copy, tXgGH, tXrGradH);
        } else {
            cute::copy_if(tiled_copy, tXp, tXgH, tXrH);
            cute::copy_if(tiled_copy, tXp, tXgGN, tXrGradNorm);
            cute::copy_if(tiled_copy, tXp, tXgGH, tXrGradH);
        }

        float local_dot = 0.0f;
        #pragma unroll
        for (int m = 0; m < size<1>(tXrH); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXrH); ++n) {
                #pragma unroll
                for (int v = 0; v < size<0>(tXrH); ++v) {
                    float h_norm = half_to_float(tXrH(v, m, n)) * rstd;
                    float dy     = half_to_float(tXrGradNorm(v, m, n));
                    float w      = half_to_float(tXrW(v, m, n));
                    local_dot   += (dy * w) * h_norm;
                    r_dW(v, m, n) += dy * h_norm;
                }
            }
        }

        float mean_dot = block_reduce_sum<NUM_THREADS>(local_dot, s_warp, &s_out) * inv_D;

        #pragma unroll
        for (int m = 0; m < size<1>(tXrH); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXrH); ++n) {
                #pragma unroll
                for (int v = 0; v < size<0>(tXrH); ++v) {
                    float h_norm   = half_to_float(tXrH(v, m, n)) * rstd;
                    float g        = half_to_float(tXrGradNorm(v, m, n)) * half_to_float(tXrW(v, m, n));
                    float dx_total = half_to_float(tXrGradH(v, m, n)) + rstd * (g - h_norm * mean_dot);
                    tXrGradH(v, m, n) = float_to_out<T>(dx_total);
                }
            }
        }

        if (is_full) {
            cute::copy(tiled_copy, tXrGradH, tXgGO);
        } else {
            cute::copy_if(tiled_copy, tXp, tXrGradH, tXgGO);
        }
    }

    // 3. Flush accumulated dW once per persistent CTA
    #pragma unroll
    for (int m = 0; m < size<1>(r_dW); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(r_dW); ++n) {
            #pragma unroll
            for (int v = 0; v < size<0>(r_dW); ++v) {
                int64_t col = static_cast<int64_t>(get<1>(tXcRow(v, m, n)));
                if (col < D) {
                    atomicAdd(grad_w_fp32_ptr + col, r_dW(v, m, n));
                }
            }
        }
    }
}

// Multi-tile persistent Residual + RMSNorm backward kernel (D > 8192)
template <
    typename T,
    int NUM_THREADS,
    int TILE_D,
    typename TiledCopyRow
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_residual_rmsnorm_bwd_large(
    const T*     __restrict__ grad_norm_ptr,   // (L, D) [FP16] - Gradient from normalized branch
    const T*     __restrict__ grad_h_out_ptr,  // (L, D) [FP16] - Gradient from residual skip branch
    const T*     __restrict__ h_ptr,           // (L, D) [FP16] - Saved residual sum h
    const T*     __restrict__ w_ptr,           // (D,) [FP16] - RMSNorm scale weight
    const float* __restrict__ rstd_ptr,        // (L,) [FP32] - Saved rstd
    T*           __restrict__ grad_h_total_ptr,// (L, D) [FP16] - Output combined grad_h
    float*       __restrict__ grad_w_fp32_ptr, // (D,) [FP32] - Output weight grad accumulator
    int64_t L,                                 // scalar [int64] - Number of tokens
    int64_t D                                  // scalar [int64] - Hidden dimension
) {
    // 1. Setup
    int tid = threadIdx.x;
    __shared__ float s_warp[(NUM_THREADS / 32) > 1 ? (NUM_THREADS / 32) : 1];
    __shared__ float s_out;

    const int64_t num_tiles = (D + TILE_D - 1) / TILE_D;
    const bool last_full = (D == num_tiles * TILE_D);
    const int64_t last_base = (num_tiles - 1) * TILE_D;
    const float inv_D = 1.0f / static_cast<float>(D);

    auto gGN = make_tensor(make_gmem_ptr(grad_norm_ptr),    make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gGH = make_tensor(make_gmem_ptr(grad_h_out_ptr),   make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gH  = make_tensor(make_gmem_ptr(h_ptr),            make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gGO = make_tensor(make_gmem_ptr(grad_h_total_ptr), make_layout(make_shape(L, D),        make_stride(D, Int<1>{})));
    auto gW  = make_tensor(make_gmem_ptr(w_ptr),            make_layout(make_shape(Int<1>{}, D), make_stride(D, Int<1>{})));

    TiledCopyRow tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto gSample  = local_tile(gH, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, Int<0>{}));
    auto tXDummy  = thr_copy.partition_S(gSample);

    auto tXrH        = make_fragment_like(tXDummy);
    auto tXrGradNorm = make_fragment_like(tXDummy);
    auto tXrGradH    = make_fragment_like(tXDummy);
    auto tXrW        = make_fragment_like(tXDummy);
    auto tXpLast     = make_fragment_like<bool>(tXDummy);

    auto cTile   = make_identity_tensor(make_shape(Int<1>{}, Int<TILE_D>{}));
    auto tXcTile = thr_copy.partition_S(cTile);

    if (!last_full) {
        #pragma unroll
        for (int m = 0; m < size<1>(tXpLast); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXpLast); ++n) {
                bool in_bounds = (last_base + static_cast<int64_t>(get<1>(tXcTile(Int<0>{}, m, n))) < D);
                #pragma unroll
                for (int v = 0; v < size<0>(tXpLast); ++v) {
                    tXpLast(v, m, n) = in_bounds;
                }
            }
        }
    }

    // 2. Persistent row loop
    for (int64_t row = static_cast<int64_t>(blockIdx.x); row < L; row += static_cast<int64_t>(gridDim.x)) {
        float rstd = rstd_ptr[row];

        float local_dot = 0.0f;
        for (int64_t t = 0; t < num_tiles; ++t) {
            auto gH_tile  = local_tile(gH,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto gGN_tile = local_tile(gGN, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto gW_tile  = local_tile(gW,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, t));
            auto tXgH     = thr_copy.partition_S(gH_tile);
            auto tXgGN    = thr_copy.partition_S(gGN_tile);
            auto tXgW     = thr_copy.partition_S(gW_tile);

            bool is_partial = (!last_full && t == num_tiles - 1);
            if (is_partial) {
                clear(tXrH);
                clear(tXrGradNorm);
                clear(tXrW);
                cute::copy_if(tiled_copy, tXpLast, tXgH, tXrH);
                cute::copy_if(tiled_copy, tXpLast, tXgGN, tXrGradNorm);
                cute::copy_if(tiled_copy, tXpLast, tXgW, tXrW);
            } else {
                cute::copy(tiled_copy, tXgH, tXrH);
                cute::copy(tiled_copy, tXgGN, tXrGradNorm);
                cute::copy(tiled_copy, tXgW, tXrW);
            }

            #pragma unroll
            for (int m = 0; m < size<1>(tXrH); ++m) {
                #pragma unroll
                for (int n = 0; n < size<2>(tXrH); ++n) {
                    #pragma unroll
                    for (int v = 0; v < size<0>(tXrH); ++v) {
                        float h_norm = half_to_float(tXrH(v, m, n)) * rstd;
                        float dy     = half_to_float(tXrGradNorm(v, m, n));
                        float w      = half_to_float(tXrW(v, m, n));
                        local_dot   += (dy * w) * h_norm;
                    }
                }
            }
        }

        float mean_dot = block_reduce_sum<NUM_THREADS>(local_dot, s_warp, &s_out) * inv_D;

        for (int64_t t = 0; t < num_tiles; ++t) {
            auto gH_tile  = local_tile(gH,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto gGN_tile = local_tile(gGN, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto gGH_tile = local_tile(gGH, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));
            auto gW_tile  = local_tile(gW,  make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(Int<0>{}, t));
            auto gGO_tile = local_tile(gGO, make_shape(Int<1>{}, Int<TILE_D>{}), make_coord(row, t));

            auto tXgH  = thr_copy.partition_S(gH_tile);
            auto tXgGN = thr_copy.partition_S(gGN_tile);
            auto tXgGH = thr_copy.partition_S(gGH_tile);
            auto tXgW  = thr_copy.partition_S(gW_tile);
            auto tXgGO = thr_copy.partition_D(gGO_tile);

            bool is_partial = (!last_full && t == num_tiles - 1);
            if (is_partial) {
                clear(tXrH);
                clear(tXrGradNorm);
                clear(tXrGradH);
                clear(tXrW);
                cute::copy_if(tiled_copy, tXpLast, tXgH, tXrH);
                cute::copy_if(tiled_copy, tXpLast, tXgGN, tXrGradNorm);
                cute::copy_if(tiled_copy, tXpLast, tXgGH, tXrGradH);
                cute::copy_if(tiled_copy, tXpLast, tXgW, tXrW);
            } else {
                cute::copy(tiled_copy, tXgH, tXrH);
                cute::copy(tiled_copy, tXgGN, tXrGradNorm);
                cute::copy(tiled_copy, tXgGH, tXrGradH);
                cute::copy(tiled_copy, tXgW, tXrW);
            }

            int64_t tile_col_base = t * TILE_D;
            #pragma unroll
            for (int m = 0; m < size<1>(tXrH); ++m) {
                #pragma unroll
                for (int n = 0; n < size<2>(tXrH); ++n) {
                    #pragma unroll
                    for (int v = 0; v < size<0>(tXrH); ++v) {
                        float h_norm   = half_to_float(tXrH(v, m, n)) * rstd;
                        float dy       = half_to_float(tXrGradNorm(v, m, n));
                        float w        = half_to_float(tXrW(v, m, n));
                        float dx_total = half_to_float(tXrGradH(v, m, n)) + rstd * (dy * w - h_norm * mean_dot);
                        tXrGradH(v, m, n) = float_to_out<T>(dx_total);

                        if (!is_partial || tXpLast(v, m, n)) {
                            int64_t col = tile_col_base + static_cast<int64_t>(get<1>(tXcTile(v, m, n)));
                            atomicAdd(grad_w_fp32_ptr + col, dy * h_norm);
                        }
                    }
                }
            }

            if (is_partial) {
                cute::copy_if(tiled_copy, tXpLast, tXrGradH, tXgGO);
            } else {
                cute::copy(tiled_copy, tXrGradH, tXgGO);
            }
        }
    }
}

// SwiGLU forward kernel: act = SiLU(gate) * up from concatenated gate_up
template <
    typename T,
    int NUM_THREADS,
    int TILE_M,
    int TILE_N,
    typename TiledCopy2D
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_swiglu_fwd(
    const T* __restrict__ gate_up_ptr, // (L, 2 * H_mlp) [FP16] - Concatenated gate and up projections
    T*       __restrict__ act_ptr,     // (L, H_mlp) [FP16] - Activated output
    int64_t L,                         // scalar [int64] - Number of tokens
    int64_t H_mlp                      // scalar [int64] - MLP intermediate dimension
) {
    // 1. Setup
    int64_t tile_m = static_cast<int64_t>(blockIdx.x);
    int64_t tile_n = static_cast<int64_t>(blockIdx.y);
    int tid = threadIdx.x;

    int64_t row_base = tile_m * TILE_M;
    int64_t col_base = tile_n * TILE_N;
    if (row_base >= L || col_base >= H_mlp) return;

    const bool is_interior = (row_base + TILE_M <= L) && (col_base + TILE_N <= H_mlp);
    const int64_t stride_gu = H_mlp * 2;

    auto gGate = make_tensor(make_gmem_ptr(gate_up_ptr),         make_layout(make_shape(L, H_mlp), make_stride(stride_gu, Int<1>{})));
    auto gUp   = make_tensor(make_gmem_ptr(gate_up_ptr + H_mlp), make_layout(make_shape(L, H_mlp), make_stride(stride_gu, Int<1>{})));
    auto gAct  = make_tensor(make_gmem_ptr(act_ptr),             make_layout(make_shape(L, H_mlp), make_stride(H_mlp, Int<1>{})));

    auto gGate_tile = local_tile(gGate, make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_coord(tile_m, tile_n));
    auto gUp_tile   = local_tile(gUp,   make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_coord(tile_m, tile_n));
    auto gAct_tile  = local_tile(gAct,  make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_coord(tile_m, tile_n));

    TiledCopy2D tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto tXgGate  = thr_copy.partition_S(gGate_tile);
    auto tXgUp    = thr_copy.partition_S(gUp_tile);
    auto tXgAct   = thr_copy.partition_D(gAct_tile);

    auto tXrGate = make_fragment_like(tXgGate);
    auto tXrUp   = make_fragment_like(tXgUp);
    auto tXp     = make_fragment_like<bool>(tXgGate);

    if (!is_interior) {
        auto cTile   = make_identity_tensor(make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
        auto tXcTile = thr_copy.partition_S(cTile);
        #pragma unroll
        for (int m = 0; m < size<1>(tXp); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXp); ++n) {
                int64_t r = row_base + static_cast<int64_t>(get<0>(tXcTile(Int<0>{}, m, n)));
                int64_t c = col_base + static_cast<int64_t>(get<1>(tXcTile(Int<0>{}, m, n)));
                bool in_bounds = (r < L) && (c < H_mlp);
                #pragma unroll
                for (int v = 0; v < size<0>(tXp); ++v) {
                    tXp(v, m, n) = in_bounds;
                }
            }
        }
        clear(tXrGate);
        clear(tXrUp);
        cute::copy_if(tiled_copy, tXp, tXgGate, tXrGate);
        cute::copy_if(tiled_copy, tXp, tXgUp, tXrUp);
    } else {
        cute::copy(tiled_copy, tXgGate, tXrGate);
        cute::copy(tiled_copy, tXgUp, tXrUp);
    }

    // 2. Evaluate SiLU(gate) * up using half2 intrinsics
    #pragma unroll
    for (int m = 0; m < size<1>(tXrGate); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tXrGate); ++n) {
            #pragma unroll
            for (int v = 0; v < size<0>(tXrGate); v += 2) {
                __half2 g_h2   = __halves2half2(to_half_val(tXrGate(v, m, n)), to_half_val(tXrGate(v + 1, m, n)));
                __half2 u_h2   = __halves2half2(to_half_val(tXrUp(v, m, n)),   to_half_val(tXrUp(v + 1, m, n)));
                __half2 out_h2 = __hmul2(fast_silu_h2(g_h2), u_h2);
                tXrGate(v, m, n)     = half_to_out<T>(__low2half(out_h2));
                tXrGate(v + 1, m, n) = half_to_out<T>(__high2half(out_h2));
            }
        }
    }

    // 3. Vectorized store
    if (is_interior) {
        cute::copy(tiled_copy, tXrGate, tXgAct);
    } else {
        cute::copy_if(tiled_copy, tXp, tXrGate, tXgAct);
    }
}

// SwiGLU backward kernel: computes grad_gate_up from grad_act and saved gate_up
template <
    typename T,
    int NUM_THREADS,
    int TILE_M,
    int TILE_N,
    typename TiledCopy2D
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_swiglu_bwd(
    const T* __restrict__ grad_act_ptr,     // (L, H_mlp) [FP16] - Upstream gradient
    const T* __restrict__ gate_up_ptr,      // (L, 2 * H_mlp) [FP16] - Saved gate_up
    T*       __restrict__ grad_gate_up_ptr, // (L, 2 * H_mlp) [FP16] - Output grad_gate_up
    int64_t L,                              // scalar [int64] - Number of tokens
    int64_t H_mlp                           // scalar [int64] - MLP intermediate dimension
) {
    // 1. Setup
    int64_t tile_m = static_cast<int64_t>(blockIdx.x);
    int64_t tile_n = static_cast<int64_t>(blockIdx.y);
    int tid = threadIdx.x;

    int64_t row_base = tile_m * TILE_M;
    int64_t col_base = tile_n * TILE_N;
    if (row_base >= L || col_base >= H_mlp) return;

    const bool is_interior = (row_base + TILE_M <= L) && (col_base + TILE_N <= H_mlp);
    const int64_t stride_gu = H_mlp * 2;

    auto gDY   = make_tensor(make_gmem_ptr(grad_act_ptr),            make_layout(make_shape(L, H_mlp), make_stride(H_mlp, Int<1>{})));
    auto gGate = make_tensor(make_gmem_ptr(gate_up_ptr),             make_layout(make_shape(L, H_mlp), make_stride(stride_gu, Int<1>{})));
    auto gUp   = make_tensor(make_gmem_ptr(gate_up_ptr + H_mlp),     make_layout(make_shape(L, H_mlp), make_stride(stride_gu, Int<1>{})));
    auto gDG   = make_tensor(make_gmem_ptr(grad_gate_up_ptr),        make_layout(make_shape(L, H_mlp), make_stride(stride_gu, Int<1>{})));
    auto gDU   = make_tensor(make_gmem_ptr(grad_gate_up_ptr + H_mlp),make_layout(make_shape(L, H_mlp), make_stride(stride_gu, Int<1>{})));

    auto gDY_tile   = local_tile(gDY,   make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_coord(tile_m, tile_n));
    auto gGate_tile = local_tile(gGate, make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_coord(tile_m, tile_n));
    auto gUp_tile   = local_tile(gUp,   make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_coord(tile_m, tile_n));
    auto gDG_tile   = local_tile(gDG,   make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_coord(tile_m, tile_n));
    auto gDU_tile   = local_tile(gDU,   make_shape(Int<TILE_M>{}, Int<TILE_N>{}), make_coord(tile_m, tile_n));

    TiledCopy2D tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto tXgDY   = thr_copy.partition_S(gDY_tile);
    auto tXgGate = thr_copy.partition_S(gGate_tile);
    auto tXgUp   = thr_copy.partition_S(gUp_tile);
    auto tXgDG   = thr_copy.partition_D(gDG_tile);
    auto tXgDU   = thr_copy.partition_D(gDU_tile);

    auto tXrDY   = make_fragment_like(tXgDY);
    auto tXrGate = make_fragment_like(tXgGate);
    auto tXrUp   = make_fragment_like(tXgUp);
    auto tXp     = make_fragment_like<bool>(tXgGate);

    if (!is_interior) {
        auto cTile   = make_identity_tensor(make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
        auto tXcTile = thr_copy.partition_S(cTile);
        #pragma unroll
        for (int m = 0; m < size<1>(tXp); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXp); ++n) {
                int64_t r = row_base + static_cast<int64_t>(get<0>(tXcTile(Int<0>{}, m, n)));
                int64_t c = col_base + static_cast<int64_t>(get<1>(tXcTile(Int<0>{}, m, n)));
                bool in_bounds = (r < L) && (c < H_mlp);
                #pragma unroll
                for (int v = 0; v < size<0>(tXp); ++v) {
                    tXp(v, m, n) = in_bounds;
                }
            }
        }
        clear(tXrDY);
        clear(tXrGate);
        clear(tXrUp);
        cute::copy_if(tiled_copy, tXp, tXgDY, tXrDY);
        cute::copy_if(tiled_copy, tXp, tXgGate, tXrGate);
        cute::copy_if(tiled_copy, tXp, tXgUp, tXrUp);
    } else {
        cute::copy(tiled_copy, tXgDY, tXrDY);
        cute::copy(tiled_copy, tXgGate, tXrGate);
        cute::copy(tiled_copy, tXgUp, tXrUp);
    }

    // 2. Compute SwiGLU backward gradients
    #pragma unroll
    for (int m = 0; m < size<1>(tXrGate); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tXrGate); ++n) {
            #pragma unroll
            for (int v = 0; v < size<0>(tXrGate); v += 2) {
                __half2 g_h2 = __halves2half2(to_half_val(tXrGate(v, m, n)), to_half_val(tXrGate(v + 1, m, n)));
                __half2 s_h2 = fast_sigmoid_h2(g_h2);

                float g0  = half_to_float(tXrGate(v, m, n));
                float g1  = half_to_float(tXrGate(v + 1, m, n));
                float u0  = half_to_float(tXrUp(v, m, n));
                float u1  = half_to_float(tXrUp(v + 1, m, n));
                float dy0 = half_to_float(tXrDY(v, m, n));
                float dy1 = half_to_float(tXrDY(v + 1, m, n));
                float s0  = __half2float(__low2half(s_h2));
                float s1  = __half2float(__high2half(s_h2));

                float du0 = dy0 * (g0 * s0);
                float du1 = dy1 * (g1 * s1);
                float dg0 = (dy0 * u0) * (s0 * (1.0f + g0 * (1.0f - s0)));
                float dg1 = (dy1 * u1) * (s1 * (1.0f + g1 * (1.0f - s1)));

                tXrGate(v, m, n)     = float_to_out<T>(dg0);
                tXrGate(v + 1, m, n) = float_to_out<T>(dg1);
                tXrUp(v, m, n)       = float_to_out<T>(du0);
                tXrUp(v + 1, m, n)   = float_to_out<T>(du1);
            }
        }
    }

    // 3. Vectorized stores into split halves of grad_gate_up
    if (is_interior) {
        cute::copy(tiled_copy, tXrGate, tXgDG);
        cute::copy(tiled_copy, tXrUp, tXgDU);
    } else {
        cute::copy_if(tiled_copy, tXp, tXrGate, tXgDG);
        cute::copy_if(tiled_copy, tXp, tXrUp, tXgDU);
    }
}

// Vectorized 128-bit elementwise residual add kernel: y = a + b
template <
    typename T,
    int NUM_THREADS,
    int TILE_ELEMS,
    typename TiledCopy1D
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_residual_add_fwd(
    const T* __restrict__ a_ptr, // (total_elems,) [FP16] - First operand
    const T* __restrict__ b_ptr, // (total_elems,) [FP16] - Second operand
    T*       __restrict__ y_ptr, // (total_elems,) [FP16] - Output sum
    int64_t total_elems          // scalar [int64] - Total number of elements
) {
    // 1. Setup
    int64_t tile_idx = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    int64_t base = tile_idx * TILE_ELEMS;
    if (base >= total_elems) return;

    const bool is_interior = (base + TILE_ELEMS <= total_elems);

    auto gA = make_tensor(make_gmem_ptr(a_ptr), make_layout(make_shape(Int<1>{}, total_elems), make_stride(total_elems, Int<1>{})));
    auto gB = make_tensor(make_gmem_ptr(b_ptr), make_layout(make_shape(Int<1>{}, total_elems), make_stride(total_elems, Int<1>{})));
    auto gY = make_tensor(make_gmem_ptr(y_ptr), make_layout(make_shape(Int<1>{}, total_elems), make_stride(total_elems, Int<1>{})));

    auto gA_tile = local_tile(gA, make_shape(Int<1>{}, Int<TILE_ELEMS>{}), make_coord(Int<0>{}, tile_idx));
    auto gB_tile = local_tile(gB, make_shape(Int<1>{}, Int<TILE_ELEMS>{}), make_coord(Int<0>{}, tile_idx));
    auto gY_tile = local_tile(gY, make_shape(Int<1>{}, Int<TILE_ELEMS>{}), make_coord(Int<0>{}, tile_idx));

    TiledCopy1D tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);
    auto tXgA     = thr_copy.partition_S(gA_tile);
    auto tXgB     = thr_copy.partition_S(gB_tile);
    auto tXgY     = thr_copy.partition_D(gY_tile);

    auto tXrA = make_fragment_like(tXgA);
    auto tXrB = make_fragment_like(tXgB);
    auto tXp  = make_fragment_like<bool>(tXgA);

    if (!is_interior) {
        auto cTile   = make_identity_tensor(make_shape(Int<1>{}, Int<TILE_ELEMS>{}));
        auto tXcTile = thr_copy.partition_S(cTile);
        #pragma unroll
        for (int m = 0; m < size<1>(tXp); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tXp); ++n) {
                bool in_bounds = (base + static_cast<int64_t>(get<1>(tXcTile(Int<0>{}, m, n))) < total_elems);
                #pragma unroll
                for (int v = 0; v < size<0>(tXp); ++v) {
                    tXp(v, m, n) = in_bounds;
                }
            }
        }
        clear(tXrA);
        clear(tXrB);
        cute::copy_if(tiled_copy, tXp, tXgA, tXrA);
        cute::copy_if(tiled_copy, tXp, tXgB, tXrB);
    } else {
        cute::copy(tiled_copy, tXgA, tXrA);
        cute::copy(tiled_copy, tXgB, tXrB);
    }

    // 2. Pairwise FP16 add
    #pragma unroll
    for (int m = 0; m < size<1>(tXrA); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tXrA); ++n) {
            #pragma unroll
            for (int v = 0; v < size<0>(tXrA); v += 2) {
                __half2 a_h2 = __halves2half2(to_half_val(tXrA(v, m, n)), to_half_val(tXrA(v + 1, m, n)));
                __half2 b_h2 = __halves2half2(to_half_val(tXrB(v, m, n)), to_half_val(tXrB(v + 1, m, n)));
                __half2 y_h2 = __hadd2(a_h2, b_h2);
                tXrA(v, m, n)     = half_to_out<T>(__low2half(y_h2));
                tXrA(v + 1, m, n) = half_to_out<T>(__high2half(y_h2));
            }
        }
    }

    // 3. Vectorized store
    if (is_interior) {
        cute::copy(tiled_copy, tXrA, tXgY);
    } else {
        cute::copy_if(tiled_copy, tXp, tXrA, tXgY);
    }
}

// ============================================================================
// Part 2: Host Dispatch Helpers and PyTorch Entry Points
// ============================================================================

namespace block_ops_detail {

template <int NUM_THREADS>
using RowTiledCopyHalf = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint4>, cutlass::half_t>{},
    Layout<Shape<_1, Int<NUM_THREADS>>, Stride<Int<NUM_THREADS>, _1>>{},
    Layout<Shape<_1, _8>, Stride<_8, _1>>{}
));

template <int TILE_M, int TILE_N, int THR_M, int THR_N>
using SwigluTiledCopyHalf = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint4>, cutlass::half_t>{},
    Layout<Shape<Int<THR_M>, Int<THR_N>>, Stride<Int<THR_N>, _1>>{},
    Layout<Shape<_1, _8>, Stride<_8, _1>>{}
));

inline void check_rmsnorm_tensor_2d(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::ScalarType::Half, name, " must be float16");
    TORCH_CHECK(t.dim() == 2, name, " must be a 2D tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline void check_rmsnorm_weight_1d(const at::Tensor& w, int64_t D) {
    TORCH_CHECK(w.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Half, "weight must be float16");
    TORCH_CHECK(w.dim() == 1, "weight must be a 1D tensor");
    TORCH_CHECK(w.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(w.size(0) == D, "weight size mismatch with hidden dimension");
}

template <int NUM_THREADS, int TILE_D, bool IS_TRAINING>
inline void launch_rmsnorm_fwd_tile(
    const cutlass::half_t* x_ptr,
    const cutlass::half_t* w_ptr,
    cutlass::half_t* out_ptr,
    float* rstd_ptr,
    int64_t L,
    int64_t D,
    float eps,
    cudaStream_t stream
) {
    using Element = cutlass::half_t;
    using TiledCopyRow = RowTiledCopyHalf<NUM_THREADS>;
    dim3 grid(static_cast<unsigned int>(L));
    dim3 block(NUM_THREADS);
    kernel_fused_rmsnorm_fwd<Element, NUM_THREADS, TILE_D, IS_TRAINING, TiledCopyRow>
        <<<grid, block, 0, stream>>>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool IS_TRAINING>
inline void dispatch_rmsnorm_fwd(
    const cutlass::half_t* x_ptr,
    const cutlass::half_t* w_ptr,
    cutlass::half_t* out_ptr,
    float* rstd_ptr,
    int64_t L,
    int64_t D,
    float eps,
    cudaStream_t stream
) {
    if (D <= 256) {
        launch_rmsnorm_fwd_tile<32, 256, IS_TRAINING>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 512) {
        launch_rmsnorm_fwd_tile<64, 512, IS_TRAINING>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 1024) {
        launch_rmsnorm_fwd_tile<128, 1024, IS_TRAINING>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 2048) {
        launch_rmsnorm_fwd_tile<256, 2048, IS_TRAINING>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 4096) {
        launch_rmsnorm_fwd_tile<256, 4096, IS_TRAINING>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 8192) {
        launch_rmsnorm_fwd_tile<256, 8192, IS_TRAINING>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else {
        constexpr int NUM_THREADS = 256;
        constexpr int TILE_D = 2048;
        using TiledCopyRow = RowTiledCopyHalf<NUM_THREADS>;
        dim3 grid(static_cast<unsigned int>(L));
        dim3 block(NUM_THREADS);
        kernel_fused_rmsnorm_fwd_large<cutlass::half_t, NUM_THREADS, TILE_D, IS_TRAINING, TiledCopyRow>
            <<<grid, block, 0, stream>>>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <int NUM_THREADS, int TILE_D, bool IS_TRAINING>
inline void launch_residual_rmsnorm_fwd_tile(
    const cutlass::half_t* x_ptr,
    const cutlass::half_t* res_ptr,
    const cutlass::half_t* w_ptr,
    cutlass::half_t* h_ptr,
    cutlass::half_t* out_ptr,
    float* rstd_ptr,
    int64_t L,
    int64_t D,
    float eps,
    cudaStream_t stream
) {
    using Element = cutlass::half_t;
    using TiledCopyRow = RowTiledCopyHalf<NUM_THREADS>;
    dim3 grid(static_cast<unsigned int>(L));
    dim3 block(NUM_THREADS);
    kernel_fused_residual_rmsnorm_fwd<Element, NUM_THREADS, TILE_D, IS_TRAINING, TiledCopyRow>
        <<<grid, block, 0, stream>>>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool IS_TRAINING>
inline void dispatch_residual_rmsnorm_fwd(
    const cutlass::half_t* x_ptr,
    const cutlass::half_t* res_ptr,
    const cutlass::half_t* w_ptr,
    cutlass::half_t* h_ptr,
    cutlass::half_t* out_ptr,
    float* rstd_ptr,
    int64_t L,
    int64_t D,
    float eps,
    cudaStream_t stream
) {
    if (D <= 256) {
        launch_residual_rmsnorm_fwd_tile<32, 256, IS_TRAINING>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 512) {
        launch_residual_rmsnorm_fwd_tile<64, 512, IS_TRAINING>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 1024) {
        launch_residual_rmsnorm_fwd_tile<128, 1024, IS_TRAINING>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 2048) {
        launch_residual_rmsnorm_fwd_tile<256, 2048, IS_TRAINING>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 4096) {
        launch_residual_rmsnorm_fwd_tile<256, 4096, IS_TRAINING>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else if (D <= 8192) {
        launch_residual_rmsnorm_fwd_tile<256, 8192, IS_TRAINING>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, eps, stream);
    } else {
        constexpr int NUM_THREADS = 256;
        constexpr int TILE_D = 2048;
        using TiledCopyRow = RowTiledCopyHalf<NUM_THREADS>;
        dim3 grid(static_cast<unsigned int>(L));
        dim3 block(NUM_THREADS);
        kernel_fused_residual_rmsnorm_fwd_large<cutlass::half_t, NUM_THREADS, TILE_D, IS_TRAINING, TiledCopyRow>
            <<<grid, block, 0, stream>>>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <int NUM_THREADS, int TILE_D>
inline void launch_rmsnorm_bwd_tile(
    const cutlass::half_t* grad_out_ptr,
    const cutlass::half_t* x_ptr,
    const cutlass::half_t* w_ptr,
    const float* rstd_ptr,
    cutlass::half_t* grad_x_ptr,
    float* grad_w_fp32_ptr,
    int64_t L,
    int64_t D,
    cudaStream_t stream
) {
    using TiledCopyRow = RowTiledCopyHalf<NUM_THREADS>;
    unsigned int num_blocks = static_cast<unsigned int>(std::min<int64_t>(L, 160));
    dim3 grid(num_blocks);
    dim3 block(NUM_THREADS);
    kernel_fused_rmsnorm_bwd<cutlass::half_t, NUM_THREADS, TILE_D, TiledCopyRow>
        <<<grid, block, 0, stream>>>(grad_out_ptr, x_ptr, w_ptr, rstd_ptr, grad_x_ptr, grad_w_fp32_ptr, L, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NUM_THREADS, int TILE_D>
inline void launch_residual_rmsnorm_bwd_tile(
    const cutlass::half_t* grad_norm_ptr,
    const cutlass::half_t* grad_h_out_ptr,
    const cutlass::half_t* h_ptr,
    const cutlass::half_t* w_ptr,
    const float* rstd_ptr,
    cutlass::half_t* grad_h_total_ptr,
    float* grad_w_fp32_ptr,
    int64_t L,
    int64_t D,
    cudaStream_t stream
) {
    using TiledCopyRow = RowTiledCopyHalf<NUM_THREADS>;
    unsigned int num_blocks = static_cast<unsigned int>(std::min<int64_t>(L, 160));
    dim3 grid(num_blocks);
    dim3 block(NUM_THREADS);
    kernel_fused_residual_rmsnorm_bwd<cutlass::half_t, NUM_THREADS, TILE_D, TiledCopyRow>
        <<<grid, block, 0, stream>>>(grad_norm_ptr, grad_h_out_ptr, h_ptr, w_ptr, rstd_ptr, grad_h_total_ptr, grad_w_fp32_ptr, L, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace block_ops_detail
using namespace block_ops_detail;

// Outputs: (out: (L, D) [FP16], rstd: (L,) [FP32] or empty when !is_training)
std::tuple<at::Tensor, at::Tensor>
fused_rmsnorm_fwd(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps,
    bool is_training
) {
    check_rmsnorm_tensor_2d(x, "x");
    int64_t L = x.size(0);
    int64_t D = x.size(1);
    TORCH_CHECK(D > 0 && D % 8 == 0, "D must be positive and divisible by 8");
    check_rmsnorm_weight_1d(weight, D);

    c10::cuda::CUDAGuard device_guard(x.device());

    if (L == 0) {
        return std::make_tuple(
            at::empty({0, D}, x.options()),
            at::empty({0}, x.options().dtype(at::kFloat))
        );
    }

    at::Tensor x_a = ensure_aligned16(x);
    at::Tensor w_a = ensure_aligned16(weight);
    at::Tensor out = torch::empty({L, D}, x.options());
    at::Tensor rstd = is_training
        ? torch::empty({L}, x.options().dtype(at::kFloat))
        : at::empty({0}, x.options().dtype(at::kFloat));

    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        const auto* x_ptr = reinterpret_cast<const cutlass::half_t*>(x_a.data_ptr<at::Half>());
        const auto* w_ptr = reinterpret_cast<const cutlass::half_t*>(w_a.data_ptr<at::Half>());
        auto* out_ptr = reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>());
        float* rstd_ptr = is_training ? rstd.data_ptr<float>() : nullptr;

        if (is_training) {
            dispatch_rmsnorm_fwd<true>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, static_cast<float>(eps), stream);
        } else {
            dispatch_rmsnorm_fwd<false>(x_ptr, w_ptr, out_ptr, rstd_ptr, L, D, static_cast<float>(eps), stream);
        }
    }

    return std::make_tuple(out, rstd);
}

// Outputs: (grad_x: (L, D) [FP16], grad_weight: (D,) [FP16])
std::tuple<at::Tensor, at::Tensor>
fused_rmsnorm_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& rstd
) {
    check_rmsnorm_tensor_2d(grad_out, "grad_out");
    check_rmsnorm_tensor_2d(x, "x");
    int64_t L = x.size(0);
    int64_t D = x.size(1);
    TORCH_CHECK(D > 0 && D % 8 == 0, "D must be positive and divisible by 8");
    TORCH_CHECK(grad_out.size(0) == L && grad_out.size(1) == D, "grad_out shape mismatch");
    check_rmsnorm_weight_1d(weight, D);

    TORCH_CHECK(rstd.is_cuda(), "rstd must be a CUDA tensor");
    TORCH_CHECK(rstd.scalar_type() == at::ScalarType::Float, "rstd must be float32");
    TORCH_CHECK(rstd.dim() == 1 && rstd.size(0) == L, "rstd shape mismatch");
    TORCH_CHECK(rstd.is_contiguous(), "rstd must be contiguous");

    c10::cuda::CUDAGuard device_guard(x.device());

    if (L == 0) {
        return std::make_tuple(
            at::empty({0, D}, x.options()),
            at::zeros({D}, weight.options())
        );
    }

    at::Tensor grad_out_a = ensure_aligned16(grad_out);
    at::Tensor x_a = ensure_aligned16(x);
    at::Tensor w_a = ensure_aligned16(weight);

    at::Tensor grad_x = torch::empty({L, D}, x.options());
    at::Tensor grad_w_fp32 = torch::zeros({D}, weight.options().dtype(at::kFloat));

    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        const auto* dy_ptr = reinterpret_cast<const cutlass::half_t*>(grad_out_a.data_ptr<at::Half>());
        const auto* x_ptr = reinterpret_cast<const cutlass::half_t*>(x_a.data_ptr<at::Half>());
        const auto* w_ptr = reinterpret_cast<const cutlass::half_t*>(w_a.data_ptr<at::Half>());
        const float* rstd_ptr = rstd.data_ptr<float>();
        auto* dx_ptr = reinterpret_cast<cutlass::half_t*>(grad_x.data_ptr<at::Half>());
        float* dw_ptr = grad_w_fp32.data_ptr<float>();

        if (D <= 256) {
            launch_rmsnorm_bwd_tile<32, 256>(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr, L, D, stream);
        } else if (D <= 512) {
            launch_rmsnorm_bwd_tile<64, 512>(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr, L, D, stream);
        } else if (D <= 1024) {
            launch_rmsnorm_bwd_tile<128, 1024>(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr, L, D, stream);
        } else if (D <= 2048) {
            launch_rmsnorm_bwd_tile<256, 2048>(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr, L, D, stream);
        } else if (D <= 4096) {
            launch_rmsnorm_bwd_tile<256, 4096>(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr, L, D, stream);
        } else if (D <= 8192) {
            launch_rmsnorm_bwd_tile<256, 8192>(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr, L, D, stream);
        } else {
            constexpr int NUM_THREADS = 256;
            constexpr int TILE_D = 2048;
            using TiledCopyRow = RowTiledCopyHalf<NUM_THREADS>;
            unsigned int num_blocks = static_cast<unsigned int>(std::min<int64_t>(L, 160));
            dim3 grid(num_blocks);
            dim3 block(NUM_THREADS);
            kernel_fused_rmsnorm_bwd_large<cutlass::half_t, NUM_THREADS, TILE_D, TiledCopyRow>
                <<<grid, block, 0, stream>>>(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr, L, D);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    at::Tensor grad_weight = grad_w_fp32.to(at::kHalf);
    grad_w_fp32.reset();

    return std::make_tuple(grad_x, grad_weight);
}

// Outputs: (h: (L, D) [FP16], x_norm: (L, D) [FP16], rstd: (L,) [FP32] or empty when !is_training)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
fused_residual_rmsnorm_fwd(
    const at::Tensor& x,
    const at::Tensor& residual_in,
    const at::Tensor& weight,
    double eps,
    bool is_training
) {
    check_rmsnorm_tensor_2d(x, "x");
    check_rmsnorm_tensor_2d(residual_in, "residual_in");
    int64_t L = x.size(0);
    int64_t D = x.size(1);
    TORCH_CHECK(D > 0 && D % 8 == 0, "D must be positive and divisible by 8");
    TORCH_CHECK(residual_in.size(0) == L && residual_in.size(1) == D, "residual_in shape mismatch");
    check_rmsnorm_weight_1d(weight, D);

    c10::cuda::CUDAGuard device_guard(x.device());

    if (L == 0) {
        return std::make_tuple(
            at::empty({0, D}, x.options()),
            at::empty({0, D}, x.options()),
            at::empty({0}, x.options().dtype(at::kFloat))
        );
    }

    at::Tensor x_a = ensure_aligned16(x);
    at::Tensor res_a = ensure_aligned16(residual_in);
    at::Tensor w_a = ensure_aligned16(weight);

    at::Tensor h = torch::empty({L, D}, x.options());
    at::Tensor x_norm = torch::empty({L, D}, x.options());
    at::Tensor rstd = is_training
        ? torch::empty({L}, x.options().dtype(at::kFloat))
        : at::empty({0}, x.options().dtype(at::kFloat));

    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        const auto* x_ptr = reinterpret_cast<const cutlass::half_t*>(x_a.data_ptr<at::Half>());
        const auto* res_ptr = reinterpret_cast<const cutlass::half_t*>(res_a.data_ptr<at::Half>());
        const auto* w_ptr = reinterpret_cast<const cutlass::half_t*>(w_a.data_ptr<at::Half>());
        auto* h_ptr = reinterpret_cast<cutlass::half_t*>(h.data_ptr<at::Half>());
        auto* out_ptr = reinterpret_cast<cutlass::half_t*>(x_norm.data_ptr<at::Half>());
        float* rstd_ptr = is_training ? rstd.data_ptr<float>() : nullptr;

        if (is_training) {
            dispatch_residual_rmsnorm_fwd<true>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, static_cast<float>(eps), stream);
        } else {
            dispatch_residual_rmsnorm_fwd<false>(x_ptr, res_ptr, w_ptr, h_ptr, out_ptr, rstd_ptr, L, D, static_cast<float>(eps), stream);
        }
    }

    return std::make_tuple(h, x_norm, rstd);
}

// Outputs: (grad_h_total: (L, D) [FP16], grad_weight: (D,) [FP16])
std::tuple<at::Tensor, at::Tensor>
fused_residual_rmsnorm_bwd(
    const at::Tensor& grad_norm_out,
    const at::Tensor& grad_h_out,
    const at::Tensor& h,
    const at::Tensor& weight,
    const at::Tensor& rstd
) {
    check_rmsnorm_tensor_2d(grad_norm_out, "grad_norm_out");
    check_rmsnorm_tensor_2d(grad_h_out, "grad_h_out");
    check_rmsnorm_tensor_2d(h, "h");

    int64_t L = h.size(0);
    int64_t D = h.size(1);
    TORCH_CHECK(D > 0 && D % 8 == 0, "D must be positive and divisible by 8");
    TORCH_CHECK(grad_norm_out.size(0) == L && grad_norm_out.size(1) == D, "grad_norm_out shape mismatch");
    TORCH_CHECK(grad_h_out.size(0) == L && grad_h_out.size(1) == D, "grad_h_out shape mismatch");
    check_rmsnorm_weight_1d(weight, D);

    TORCH_CHECK(rstd.is_cuda(), "rstd must be a CUDA tensor");
    TORCH_CHECK(rstd.scalar_type() == at::ScalarType::Float, "rstd must be float32");
    TORCH_CHECK(rstd.dim() == 1 && rstd.size(0) == L, "rstd shape mismatch");
    TORCH_CHECK(rstd.is_contiguous(), "rstd must be contiguous");

    c10::cuda::CUDAGuard device_guard(h.device());

    if (L == 0) {
        return std::make_tuple(
            at::empty({0, D}, h.options()),
            at::zeros({D}, weight.options())
        );
    }

    at::Tensor gn_a = ensure_aligned16(grad_norm_out);
    at::Tensor gh_a = ensure_aligned16(grad_h_out);
    at::Tensor h_a = ensure_aligned16(h);
    at::Tensor w_a = ensure_aligned16(weight);

    at::Tensor grad_h_total = torch::empty({L, D}, h.options());
    at::Tensor grad_w_fp32 = torch::zeros({D}, weight.options().dtype(at::kFloat));

    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        const auto* gn_ptr = reinterpret_cast<const cutlass::half_t*>(gn_a.data_ptr<at::Half>());
        const auto* gh_ptr = reinterpret_cast<const cutlass::half_t*>(gh_a.data_ptr<at::Half>());
        const auto* h_ptr = reinterpret_cast<const cutlass::half_t*>(h_a.data_ptr<at::Half>());
        const auto* w_ptr = reinterpret_cast<const cutlass::half_t*>(w_a.data_ptr<at::Half>());
        const float* rstd_ptr = rstd.data_ptr<float>();
        auto* go_ptr = reinterpret_cast<cutlass::half_t*>(grad_h_total.data_ptr<at::Half>());
        float* dw_ptr = grad_w_fp32.data_ptr<float>();

        if (D <= 256) {
            launch_residual_rmsnorm_bwd_tile<32, 256>(gn_ptr, gh_ptr, h_ptr, w_ptr, rstd_ptr, go_ptr, dw_ptr, L, D, stream);
        } else if (D <= 512) {
            launch_residual_rmsnorm_bwd_tile<64, 512>(gn_ptr, gh_ptr, h_ptr, w_ptr, rstd_ptr, go_ptr, dw_ptr, L, D, stream);
        } else if (D <= 1024) {
            launch_residual_rmsnorm_bwd_tile<128, 1024>(gn_ptr, gh_ptr, h_ptr, w_ptr, rstd_ptr, go_ptr, dw_ptr, L, D, stream);
        } else if (D <= 2048) {
            launch_residual_rmsnorm_bwd_tile<256, 2048>(gn_ptr, gh_ptr, h_ptr, w_ptr, rstd_ptr, go_ptr, dw_ptr, L, D, stream);
        } else if (D <= 4096) {
            launch_residual_rmsnorm_bwd_tile<256, 4096>(gn_ptr, gh_ptr, h_ptr, w_ptr, rstd_ptr, go_ptr, dw_ptr, L, D, stream);
        } else if (D <= 8192) {
            launch_residual_rmsnorm_bwd_tile<256, 8192>(gn_ptr, gh_ptr, h_ptr, w_ptr, rstd_ptr, go_ptr, dw_ptr, L, D, stream);
        } else {
            constexpr int NUM_THREADS = 256;
            constexpr int TILE_D = 2048;
            using TiledCopyRow = RowTiledCopyHalf<NUM_THREADS>;
            unsigned int num_blocks = static_cast<unsigned int>(std::min<int64_t>(L, 160));
            dim3 grid(num_blocks);
            dim3 block(NUM_THREADS);
            kernel_fused_residual_rmsnorm_bwd_large<cutlass::half_t, NUM_THREADS, TILE_D, TiledCopyRow>
                <<<grid, block, 0, stream>>>(gn_ptr, gh_ptr, h_ptr, w_ptr, rstd_ptr, go_ptr, dw_ptr, L, D);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    at::Tensor grad_weight = grad_w_fp32.to(at::kHalf);
    grad_w_fp32.reset();

    return std::make_tuple(grad_h_total, grad_weight);
}

// Output: act (L, H_mlp) [FP16]
at::Tensor
fused_swiglu_fwd(
    const at::Tensor& gate_up
) {
    check_rmsnorm_tensor_2d(gate_up, "gate_up");
    int64_t L = gate_up.size(0);
    int64_t two_h = gate_up.size(1);
    TORCH_CHECK(two_h > 0 && two_h % 16 == 0, "gate_up.size(1) must be positive and divisible by 16");
    int64_t H_mlp = two_h / 2;

    c10::cuda::CUDAGuard device_guard(gate_up.device());

    if (L == 0) {
        return at::empty({0, H_mlp}, gate_up.options());
    }

    at::Tensor gate_up_a = ensure_aligned16(gate_up);
    at::Tensor act = torch::empty({L, H_mlp}, gate_up.options());

    {
        constexpr int NUM_THREADS = 256;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        const auto* gu_ptr = reinterpret_cast<const cutlass::half_t*>(gate_up_a.data_ptr<at::Half>());
        auto* act_ptr = reinterpret_cast<cutlass::half_t*>(act.data_ptr<at::Half>());

        if (H_mlp <= 256) {
            constexpr int TILE_M = 8;
            constexpr int TILE_N = 256;
            using TiledCopy2D = SwigluTiledCopyHalf<TILE_M, TILE_N, 8, 32>;
            dim3 grid(static_cast<unsigned int>((L + TILE_M - 1) / TILE_M), static_cast<unsigned int>((H_mlp + TILE_N - 1) / TILE_N));
            dim3 block(NUM_THREADS);
            kernel_fused_swiglu_fwd<cutlass::half_t, NUM_THREADS, TILE_M, TILE_N, TiledCopy2D>
                <<<grid, block, 0, stream>>>(gu_ptr, act_ptr, L, H_mlp);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (H_mlp <= 512) {
            constexpr int TILE_M = 4;
            constexpr int TILE_N = 512;
            using TiledCopy2D = SwigluTiledCopyHalf<TILE_M, TILE_N, 4, 64>;
            dim3 grid(static_cast<unsigned int>((L + TILE_M - 1) / TILE_M), static_cast<unsigned int>((H_mlp + TILE_N - 1) / TILE_N));
            dim3 block(NUM_THREADS);
            kernel_fused_swiglu_fwd<cutlass::half_t, NUM_THREADS, TILE_M, TILE_N, TiledCopy2D>
                <<<grid, block, 0, stream>>>(gu_ptr, act_ptr, L, H_mlp);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            constexpr int TILE_M = 2;
            constexpr int TILE_N = 1024;
            using TiledCopy2D = SwigluTiledCopyHalf<TILE_M, TILE_N, 2, 128>;
            dim3 grid(static_cast<unsigned int>((L + TILE_M - 1) / TILE_M), static_cast<unsigned int>((H_mlp + TILE_N - 1) / TILE_N));
            dim3 block(NUM_THREADS);
            kernel_fused_swiglu_fwd<cutlass::half_t, NUM_THREADS, TILE_M, TILE_N, TiledCopy2D>
                <<<grid, block, 0, stream>>>(gu_ptr, act_ptr, L, H_mlp);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    return act;
}

// Output: grad_gate_up (L, 2 * H_mlp) [FP16]
at::Tensor
fused_swiglu_bwd(
    const at::Tensor& grad_act,
    const at::Tensor& gate_up
) {
    check_rmsnorm_tensor_2d(grad_act, "grad_act");
    check_rmsnorm_tensor_2d(gate_up, "gate_up");

    int64_t L = grad_act.size(0);
    int64_t H_mlp = grad_act.size(1);
    TORCH_CHECK(H_mlp > 0 && H_mlp % 8 == 0, "grad_act.size(1) must be positive and divisible by 8");
    TORCH_CHECK(gate_up.size(0) == L && gate_up.size(1) == 2 * H_mlp, "gate_up shape mismatch");

    c10::cuda::CUDAGuard device_guard(gate_up.device());

    if (L == 0) {
        return at::empty({0, 2 * H_mlp}, gate_up.options());
    }

    at::Tensor ga_a = ensure_aligned16(grad_act);
    at::Tensor gu_a = ensure_aligned16(gate_up);
    at::Tensor grad_gate_up = torch::empty({L, 2 * H_mlp}, gate_up.options());

    {
        constexpr int NUM_THREADS = 256;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        const auto* ga_ptr = reinterpret_cast<const cutlass::half_t*>(ga_a.data_ptr<at::Half>());
        const auto* gu_ptr = reinterpret_cast<const cutlass::half_t*>(gu_a.data_ptr<at::Half>());
        auto* dgu_ptr = reinterpret_cast<cutlass::half_t*>(grad_gate_up.data_ptr<at::Half>());

        if (H_mlp <= 256) {
            constexpr int TILE_M = 8;
            constexpr int TILE_N = 256;
            using TiledCopy2D = SwigluTiledCopyHalf<TILE_M, TILE_N, 8, 32>;
            dim3 grid(static_cast<unsigned int>((L + TILE_M - 1) / TILE_M), static_cast<unsigned int>((H_mlp + TILE_N - 1) / TILE_N));
            dim3 block(NUM_THREADS);
            kernel_fused_swiglu_bwd<cutlass::half_t, NUM_THREADS, TILE_M, TILE_N, TiledCopy2D>
                <<<grid, block, 0, stream>>>(ga_ptr, gu_ptr, dgu_ptr, L, H_mlp);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (H_mlp <= 512) {
            constexpr int TILE_M = 4;
            constexpr int TILE_N = 512;
            using TiledCopy2D = SwigluTiledCopyHalf<TILE_M, TILE_N, 4, 64>;
            dim3 grid(static_cast<unsigned int>((L + TILE_M - 1) / TILE_M), static_cast<unsigned int>((H_mlp + TILE_N - 1) / TILE_N));
            dim3 block(NUM_THREADS);
            kernel_fused_swiglu_bwd<cutlass::half_t, NUM_THREADS, TILE_M, TILE_N, TiledCopy2D>
                <<<grid, block, 0, stream>>>(ga_ptr, gu_ptr, dgu_ptr, L, H_mlp);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            constexpr int TILE_M = 2;
            constexpr int TILE_N = 1024;
            using TiledCopy2D = SwigluTiledCopyHalf<TILE_M, TILE_N, 2, 128>;
            dim3 grid(static_cast<unsigned int>((L + TILE_M - 1) / TILE_M), static_cast<unsigned int>((H_mlp + TILE_N - 1) / TILE_N));
            dim3 block(NUM_THREADS);
            kernel_fused_swiglu_bwd<cutlass::half_t, NUM_THREADS, TILE_M, TILE_N, TiledCopy2D>
                <<<grid, block, 0, stream>>>(ga_ptr, gu_ptr, dgu_ptr, L, H_mlp);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    return grad_gate_up;
}

// Output: y (L, D) [FP16]
at::Tensor
fused_residual_add_fwd(
    const at::Tensor& a,
    const at::Tensor& b
) {
    check_rmsnorm_tensor_2d(a, "a");
    check_rmsnorm_tensor_2d(b, "b");
    int64_t L = a.size(0);
    int64_t D = a.size(1);
    TORCH_CHECK(D > 0 && D % 8 == 0, "D must be positive and divisible by 8");
    TORCH_CHECK(b.size(0) == L && b.size(1) == D, "b shape mismatch with a");

    c10::cuda::CUDAGuard device_guard(a.device());

    if (L == 0) {
        return at::empty({0, D}, a.options());
    }

    at::Tensor a_a = ensure_aligned16(a);
    at::Tensor b_a = ensure_aligned16(b);
    at::Tensor y = torch::empty({L, D}, a.options());

    {
        constexpr int NUM_THREADS = 256;
        constexpr int TILE_ELEMS = NUM_THREADS * 8;
        using TiledCopy1D = RowTiledCopyHalf<NUM_THREADS>;
        int64_t total_elems = L * D;
        unsigned int num_blocks = static_cast<unsigned int>((total_elems + TILE_ELEMS - 1) / TILE_ELEMS);
        dim3 grid(num_blocks);
        dim3 block(NUM_THREADS);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        const auto* a_ptr = reinterpret_cast<const cutlass::half_t*>(a_a.data_ptr<at::Half>());
        const auto* b_ptr = reinterpret_cast<const cutlass::half_t*>(b_a.data_ptr<at::Half>());
        auto* y_ptr = reinterpret_cast<cutlass::half_t*>(y.data_ptr<at::Half>());

        kernel_fused_residual_add_fwd<cutlass::half_t, NUM_THREADS, TILE_ELEMS, TiledCopy1D>
            <<<grid, block, 0, stream>>>(a_ptr, b_ptr, y_ptr, total_elems);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return y;
}
