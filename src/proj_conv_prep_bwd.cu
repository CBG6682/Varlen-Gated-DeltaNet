#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <optional>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
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

namespace proj_conv_prep_bwd_detail {

__device__ __forceinline__ float stable_sigmoid_f(float x) {
    float xc = fmaxf(fminf(x, 80.0f), -80.0f);
    return 1.0f / (1.0f + expf(-xc));
}

} // namespace proj_conv_prep_bwd_detail
using namespace proj_conv_prep_bwd_detail;

// Kernel 1a (Bwd): Q/K L2-norm + SiLU VJP -> d_conv_buf and d_conv_weight
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_HEAD,
    bool HAS_PAST_CONV,
    typename TiledCopyHalf
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_conv_act_l2_transpose_bwd(
    const T*       __restrict__ dq_h_ptr,          // (H_q, total_L, 64) [FP16] - Grad q_h
    const T*       __restrict__ dk_h_ptr,          // (H_kv, total_L, 64) [FP16] - Grad k_h
    const T*       __restrict__ dv_h_ptr,          // (H_kv, total_L, 64) [FP16] - Grad v_h
    const T*       __restrict__ proj_ptr,          // (total_L, total_proj_dim) [FP16] - Saved projection
    const T*       __restrict__ conv_w_ptr,        // (4, D_qkv) [FP16] - Conv weights
    const T*       __restrict__ past_conv_ptr,     // (B, 3, D_qkv) [FP16] - Optional past conv cache
    const float*   __restrict__ q_inv_norm_ptr,    // (H_q, total_L) [FP32] - Saved Q inv norm
    const float*   __restrict__ k_inv_norm_ptr,    // (H_kv, total_L) [FP32] - Saved K inv norm
    const int32_t* __restrict__ chunk_offsets,     // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens,  // (num_chunks,) [Int32] - Valid tokens per chunk
    const int32_t* __restrict__ chunk_seq_ids,     // (num_chunks,) [Int32] - Sequence id per chunk
    const int32_t* __restrict__ chunk_local_starts,// (num_chunks,) [Int32] - Local token start in sequence
    T*             __restrict__ d_conv_buf_ptr,    // (H_q + 2*H_kv, total_L, 64) [FP16] - Output d_conv
    float*         __restrict__ d_conv_w_fp32_ptr, // (4, D_qkv) [FP32] - Weight grad accumulator
    int64_t total_proj_dim,                        // scalar [int64] - Stride of proj
    int64_t D_qkv,                                 // scalar [int64] - Total Q+K+V width
    int64_t H_q,                                   // scalar [int64] - Number of query heads
    int64_t H_kv,                                  // scalar [int64] - Number of KV heads
    int64_t total_L,                               // scalar [int64] - Total packed tokens
    int64_t num_chunks,                            // scalar [int64] - Total varlen chunks
    float   scale                                  // scalar [float] - Query scale
) {
    // 1. Setup
    int chunk_idx   = blockIdx.x;
    int head_global = blockIdx.y;
    int tid         = threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    int mode = 2;
    int head_local = head_global - static_cast<int>(H_q + H_kv);
    if (head_global < H_q) {
        mode = 0;
        head_local = head_global;
    } else if (head_global < H_q + H_kv) {
        mode = 1;
        head_local = head_global - static_cast<int>(H_q);
    }

    int32_t chunk_start = chunk_offsets[chunk_idx];
    int32_t valid_len   = chunk_valid_lens[chunk_idx];
    int32_t seq_id      = chunk_seq_ids[chunk_idx];
    int32_t local_start = chunk_local_starts[chunk_idx];

    int64_t col_offset = static_cast<int64_t>(head_global) * D_HEAD;

    // Reuse SMEM of sIn_raw as FP32 reduction scratch after main loop
    union SmemUnion {
        T sIn_raw[72 * D_HEAD];
        float sDW_red[16][D_HEAD];
    };
    __shared__ alignas(16) SmemUnion smem_u;
    __shared__ alignas(16) T sW_raw[16 * D_HEAD];

    using SwizzleOp = Swizzle<3, 3, 3>;
    auto smem_layout_in   = composition(SwizzleOp{}, make_layout(make_shape(Int<72>{},         Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{})));
    auto smem_layout_main = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{})));
    auto smem_layout_w    = composition(SwizzleOp{}, make_layout(make_shape(Int<16>{},         Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{})));

    auto sIn      = make_tensor(make_smem_ptr(smem_u.sIn_raw),              smem_layout_in);
    auto sIn_main = make_tensor(make_smem_ptr(smem_u.sIn_raw + 8 * D_HEAD), smem_layout_main);
    auto sW       = make_tensor(make_smem_ptr(sW_raw),                      smem_layout_w);

    TiledCopyHalf tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);

    auto gW = make_tensor(
        make_gmem_ptr(conv_w_ptr + col_offset),
        make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(D_qkv, Int<1>{}))
    );
    auto gProjMain = make_tensor(
        make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + col_offset),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );

    const T* grad_in_base = (mode == 0) ? dq_h_ptr : ((mode == 1) ? dk_h_ptr : dv_h_ptr);
    auto gGradIn = make_tensor(
        make_gmem_ptr(grad_in_base + (static_cast<int64_t>(head_local) * total_L + chunk_start) * D_HEAD),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto gDConvOut = make_tensor(
        make_gmem_ptr(d_conv_buf_ptr + (static_cast<int64_t>(head_global) * total_L + chunk_start) * D_HEAD),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );

    auto tXgW      = thr_copy.partition_S(gW);
    auto tXsW      = thr_copy.partition_D(sW);
    auto tXgProj   = thr_copy.partition_S(gProjMain);
    auto tXsInMain = thr_copy.partition_D(sIn_main);
    auto tXgGradIn = thr_copy.partition_S(gGradIn);
    auto tXgDCOut  = thr_copy.partition_D(gDConvOut);

    auto tCrW    = make_fragment_like(tXgW);
    auto tCrHalo = make_fragment_like(tXgW);
    auto tCrMain = make_fragment_like(tXgProj);
    auto tCrGrad = make_fragment_like(tXgGradIn);

    auto tPred4  = make_fragment_like<bool>(tXgW);
    auto tPred3  = make_fragment_like<bool>(tXgW);
    auto tPred64 = make_fragment_like<bool>(tXgProj);

    auto cTile16   = make_identity_tensor(make_shape(Int<16>{}, Int<D_HEAD>{}));
    auto tXcTile16 = thr_copy.partition_S(cTile16);
    #pragma unroll
    for (int m = 0; m < size<1>(tPred4); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPred4); ++n) {
            int r = get<0>(tXcTile16(Int<0>{}, m, n));
            #pragma unroll
            for (int v = 0; v < size<0>(tPred4); ++v) {
                tPred4(v, m, n) = (r < 4);
                tPred3(v, m, n) = (r < 3);
            }
        }
    }

    auto cTile64   = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}));
    auto tXcTile64 = thr_copy.partition_S(cTile64);
    #pragma unroll
    for (int m = 0; m < size<1>(tPred64); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPred64); ++n) {
            bool in_bounds = (get<0>(tXcTile64(Int<0>{}, m, n)) < valid_len);
            #pragma unroll
            for (int v = 0; v < size<0>(tPred64); ++v) {
                tPred64(v, m, n) = in_bounds;
            }
        }
    }

    // 2. Load inputs, weights, and upstream gradients
    clear(tCrW);
    cute::copy_if(tiled_copy, tPred4, tXgW, tCrW);
    cute::copy(tiled_copy, tCrW, tXsW);

    clear(tCrHalo);
    if (local_start > 0) {
        auto gHaloPrev = make_tensor(
            make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start - 3) * total_proj_dim + col_offset),
            make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
        );
        cute::copy_if(tiled_copy, tPred3, thr_copy.partition_S(gHaloPrev), tCrHalo);
    } else if constexpr (HAS_PAST_CONV) {
        auto gHaloState = make_tensor(
            make_gmem_ptr(past_conv_ptr + static_cast<int64_t>(seq_id) * 3 * D_qkv + col_offset),
            make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(D_qkv, Int<1>{}))
        );
        cute::copy_if(tiled_copy, tPred3, thr_copy.partition_S(gHaloState), tCrHalo);
    }

    if (row_id < 3) {
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            sIn(5 + row_id, col_base + v) = tCrHalo(v, 0, 0);
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy, tXgProj, tCrMain);
        cute::copy(tiled_copy, tXgGradIn, tCrGrad);
    } else {
        clear(tCrMain);
        clear(tCrGrad);
        cute::copy_if(tiled_copy, tPred64, tXgProj, tCrMain);
        cute::copy_if(tiled_copy, tPred64, tXgGradIn, tCrGrad);
    }
    cute::copy(tiled_copy, tCrMain, tXsInMain);
    __syncthreads();

    // 3. Recompute forward activations, compute VJP, and accumulate d_conv_weight
    float r_dw[4][8];
    #pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            r_dw[tap][v] = 0.0f;
        }
    }

    const float* inv_norm_ptr = (mode == 0) ? q_inv_norm_ptr : k_inv_norm_ptr;

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int local_row = m * 16 + row_id;
        float conv_f[8];
        float sig_f[8];
        float g_f[8];
        float d_silu[8];

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float cf = 0.0f;
            #pragma unroll
            for (int tap = 0; tap < 4; ++tap) {
                cf += half_to_float(sW(tap, col_base + v)) *
                      half_to_float(sIn(5 + local_row + tap, col_base + v));
            }
            conv_f[v] = cf;
            sig_f[v]  = stable_sigmoid_f(cf);
            g_f[v]    = half_to_float(tCrGrad(v, m, 0));
        }

        if (mode < 2) {
            float inv_norm = (local_row < valid_len)
                ? inv_norm_ptr[static_cast<int64_t>(head_local) * total_L + chunk_start + local_row]
                : 0.0f;
            float q_scale = (mode == 0) ? scale : 1.0f;
            float u_norm[8];
            float local_dot = 0.0f;

            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float gv = g_f[v] * q_scale;
                float un = (conv_f[v] * sig_f[v]) * inv_norm;
                g_f[v]    = gv;
                u_norm[v] = un;
                local_dot += gv * un;
            }
            float dot = octet_reduce_sum(local_dot);

            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                d_silu[v] = inv_norm * (g_f[v] - u_norm[v] * dot);
            }
        } else {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                d_silu[v] = g_f[v];
            }
        }

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float dsilu_dconv = sig_f[v] * (1.0f + conv_f[v] * (1.0f - sig_f[v]));
            float dc = (local_row < valid_len) ? (d_silu[v] * dsilu_dconv) : 0.0f;
            tCrGrad(v, m, 0) = float_to_out<T>(dc);

            #pragma unroll
            for (int tap = 0; tap < 4; ++tap) {
                r_dw[tap][v] += dc * half_to_float(sIn(5 + local_row + tap, col_base + v));
            }
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy, tCrGrad, tXgDCOut);
    } else {
        cute::copy_if(tiled_copy, tPred64, tCrGrad, tXgDCOut);
    }

    // 4. Reduce d_conv_weight across block rows via shared memory reuse
    #pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
        __syncthreads();
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            smem_u.sDW_red[row_id][col_base + v] = r_dw[tap][v];
        }
        __syncthreads();
        if (tid < D_HEAD) {
            float col_sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                col_sum += smem_u.sDW_red[r][tid];
            }
            atomicAdd(d_conv_w_fp32_ptr + static_cast<int64_t>(tap) * D_qkv + col_offset + tid, col_sum);
        }
    }
}

// Kernel 1b (Bwd): transpose-conv scatter from d_conv_buf to token-major d_proj_inout and d_past_conv_state
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_HEAD,
    bool OUT_PAST_CONV_GRAD,
    typename TiledCopyHalf
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_conv_dx_transpose_bwd(
    const T*       __restrict__ d_conv_buf_ptr,     // (H_q + 2*H_kv, total_L, 64) [FP16] - Conv output grad
    const T*       __restrict__ conv_w_ptr,         // (4, D_qkv) [FP16] - Conv weights
    const int32_t* __restrict__ chunk_offsets,      // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens,   // (num_chunks,) [Int32] - Valid tokens per chunk
    const int32_t* __restrict__ chunk_seq_ids,      // (num_chunks,) [Int32] - Sequence id per chunk
    const int32_t* __restrict__ chunk_local_starts, // (num_chunks,) [Int32] - Local token start in sequence
    const int32_t* __restrict__ chunk_is_last,      // (num_chunks,) [Int32] - Last chunk flag per sequence
    T*             __restrict__ d_proj_ptr,         // (total_L, total_proj_dim) [FP16] - Output projection grad
    T*             __restrict__ d_past_conv_ptr,    // (B, 3, D_qkv) [FP16] - Output past conv state grad
    int64_t total_proj_dim,                         // scalar [int64] - Stride of d_proj_inout
    int64_t D_qkv,                                  // scalar [int64] - Total Q+K+V width
    int64_t total_L,                                // scalar [int64] - Total packed tokens
    int64_t num_chunks                              // scalar [int64] - Total varlen chunks
) {
    // 1. Setup
    int chunk_idx   = blockIdx.x;
    int head_global = blockIdx.y;
    int tid         = threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    int32_t chunk_start = chunk_offsets[chunk_idx];
    int32_t valid_len   = chunk_valid_lens[chunk_idx];
    int32_t seq_id      = chunk_seq_ids[chunk_idx];
    int32_t local_start = chunk_local_starts[chunk_idx];
    int32_t is_last     = chunk_is_last[chunk_idx];
    int32_t next_valid  = (is_last == 0) ? chunk_valid_lens[chunk_idx + 1] : 0;

    int64_t col_offset = static_cast<int64_t>(head_global) * D_HEAD;

    // Main chunk occupies rows 0..63 and lookahead rows from next chunk occupy rows 64..66
    __shared__ alignas(16) T sDC_raw[72 * D_HEAD];
    __shared__ alignas(16) T sW_raw[16 * D_HEAD];

    using SwizzleOp = Swizzle<3, 3, 3>;
    auto smem_layout_dc   = composition(SwizzleOp{}, make_layout(make_shape(Int<72>{},         Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{})));
    auto smem_layout_main = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{})));
    auto smem_layout_w    = composition(SwizzleOp{}, make_layout(make_shape(Int<16>{},         Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{})));

    auto sDC      = make_tensor(make_smem_ptr(sDC_raw), smem_layout_dc);
    auto sDC_main = make_tensor(make_smem_ptr(sDC_raw), smem_layout_main);
    auto sW       = make_tensor(make_smem_ptr(sW_raw),  smem_layout_w);

    TiledCopyHalf tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);

    auto gW = make_tensor(
        make_gmem_ptr(conv_w_ptr + col_offset),
        make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(D_qkv, Int<1>{}))
    );
    auto gDC = make_tensor(
        make_gmem_ptr(d_conv_buf_ptr + (static_cast<int64_t>(head_global) * total_L + chunk_start) * D_HEAD),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto gDProj = make_tensor(
        make_gmem_ptr(d_proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + col_offset),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );

    auto tXgW     = thr_copy.partition_S(gW);
    auto tXsW     = thr_copy.partition_D(sW);
    auto tXgDC    = thr_copy.partition_S(gDC);
    auto tXsDC    = thr_copy.partition_D(sDC_main);
    auto tXgDProj = thr_copy.partition_D(gDProj);

    auto tCrW    = make_fragment_like(tXgW);
    auto tCrLook = make_fragment_like(tXgW);
    auto tCrMain = make_fragment_like(tXgDC);

    auto tPred4    = make_fragment_like<bool>(tXgW);
    auto tPredLook = make_fragment_like<bool>(tXgW);
    auto tPredPast = make_fragment_like<bool>(tXgW);
    auto tPred64   = make_fragment_like<bool>(tXgDC);

    auto cTile16   = make_identity_tensor(make_shape(Int<16>{}, Int<D_HEAD>{}));
    auto tXcTile16 = thr_copy.partition_S(cTile16);
    #pragma unroll
    for (int m = 0; m < size<1>(tPred4); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPred4); ++n) {
            int r = get<0>(tXcTile16(Int<0>{}, m, n));
            #pragma unroll
            for (int v = 0; v < size<0>(tPred4); ++v) {
                tPred4(v, m, n)    = (r < 4);
                tPredLook(v, m, n) = (r < 3) && (is_last == 0) && (r < next_valid);
                tPredPast(v, m, n) = (r < 3) && (local_start == 0);
            }
        }
    }

    auto cTile64   = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}));
    auto tXcTile64 = thr_copy.partition_S(cTile64);
    #pragma unroll
    for (int m = 0; m < size<1>(tPred64); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPred64); ++n) {
            bool in_bounds = (get<0>(tXcTile64(Int<0>{}, m, n)) < valid_len);
            #pragma unroll
            for (int v = 0; v < size<0>(tPred64); ++v) {
                tPred64(v, m, n) = in_bounds;
            }
        }
    }

    // 2. Load weights, main chunk d_conv, and lookahead d_conv into SMEM
    clear(tCrW);
    cute::copy_if(tiled_copy, tPred4, tXgW, tCrW);
    cute::copy(tiled_copy, tCrW, tXsW);

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy, tXgDC, tCrMain);
    } else {
        clear(tCrMain);
        cute::copy_if(tiled_copy, tPred64, tXgDC, tCrMain);
    }
    cute::copy(tiled_copy, tCrMain, tXsDC);

    clear(tCrLook);
    if (is_last == 0) {
        auto gDCLook = make_tensor(
            make_gmem_ptr(d_conv_buf_ptr + (static_cast<int64_t>(head_global) * total_L + chunk_start + CHUNK_SIZE) * D_HEAD),
            make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
        );
        cute::copy_if(tiled_copy, tPredLook, thr_copy.partition_S(gDCLook), tCrLook);
    }
    if (row_id < 3) {
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            sDC(CHUNK_SIZE + row_id, col_base + v) = tCrLook(v, 0, 0);
        }
    }
    __syncthreads();

    // 3. Emit d_proj_qkv in token-major layout and optional d_past_conv_state
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int local_row = m * 16 + row_id;
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float dp_f = 0.0f;
            #pragma unroll
            for (int tap = 0; tap < 4; ++tap) {
                dp_f += half_to_float(sW(tap, col_base + v)) *
                        half_to_float(sDC(local_row + (3 - tap), col_base + v));
            }
            tCrMain(v, m, 0) = float_to_out<T>(dp_f);
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy, tCrMain, tXgDProj);
    } else {
        cute::copy_if(tiled_copy, tPred64, tCrMain, tXgDProj);
    }

    if constexpr (OUT_PAST_CONV_GRAD) {
        if (local_start == 0 && row_id < 3) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float val = 0.0f;
                if (row_id == 0) {
                    val = half_to_float(sW(0, col_base + v)) * half_to_float(sDC(0, col_base + v));
                } else if (row_id == 1) {
                    val = half_to_float(sW(1, col_base + v)) * half_to_float(sDC(0, col_base + v)) +
                          half_to_float(sW(0, col_base + v)) * half_to_float(sDC(1, col_base + v));
                } else {
                    val = half_to_float(sW(2, col_base + v)) * half_to_float(sDC(0, col_base + v)) +
                          half_to_float(sW(1, col_base + v)) * half_to_float(sDC(1, col_base + v)) +
                          half_to_float(sW(0, col_base + v)) * half_to_float(sDC(2, col_base + v));
                }
                tCrLook(v, 0, 0) = float_to_out<T>(val);
            }
        }
        auto gDPast = make_tensor(
            make_gmem_ptr(d_past_conv_ptr + static_cast<int64_t>(seq_id) * 3 * D_qkv + col_offset),
            make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(D_qkv, Int<1>{}))
        );
        cute::copy_if(tiled_copy, tPredPast, tCrLook, thr_copy.partition_D(gDPast));
    }
}

// Kernel 2 (Bwd): reverse suffix scan of dG_h + gates/decay VJP -> d_proj_inout, d_A_log, d_dt_bias
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_HEAD,
    bool ALLOW_NEG_EIGVAL,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_gates_decay_transpose_bwd(
    const T*       __restrict__ db_h_ptr,          // (H_kv, total_L, 64) [FP16] - Grad b_eff_h
    const T*       __restrict__ dw_h_ptr,          // (H_kv, total_L, 64) [FP16] - Grad w_h
    const float*   __restrict__ dG_h_ptr,          // (H_kv, total_L, 64) [FP32] - Grad G_ln_h (natural-log decay)
    const T*       __restrict__ proj_ptr,          // (total_L, total_proj_dim) [FP16] - Saved projection
    const float*   __restrict__ A_log_ptr,         // (H_kv,) [FP32] - Log-decay rate
    const float*   __restrict__ dt_bias_ptr,       // (H_kv, 64) [FP32] - Decay bias
    const int32_t* __restrict__ chunk_offsets,     // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens,  // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ d_proj_ptr,        // (total_L, total_proj_dim) [FP16] - Output projection grad
    float*         __restrict__ d_A_log_ptr,       // (H_kv,) [FP32] - Output A_log grad
    float*         __restrict__ d_dt_bias_ptr,     // (H_kv, 64) [FP32] - Output dt_bias grad
    int64_t total_proj_dim,                        // scalar [int64] - Stride of proj
    int64_t offset_b,                              // scalar [int64] - Column start of b
    int64_t offset_w,                              // scalar [int64] - Column start of w
    int64_t offset_f,                              // scalar [int64] - Column start of f
    int64_t total_L,                               // scalar [int64] - Total packed tokens
    int64_t num_chunks                             // scalar [int64] - Total varlen chunks
) {
    // 1. Setup
    int chunk_idx = blockIdx.x;
    int h         = blockIdx.y;
    int tid       = threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    int32_t chunk_start = chunk_offsets[chunk_idx];
    int32_t valid_len   = chunk_valid_lens[chunk_idx];

    float neg_exp_A = -expf(A_log_ptr[h]);
    const float* dt_row_ptr = dt_bias_ptr + static_cast<int64_t>(h) * D_HEAD + col_base;
    float4 dt0 = *reinterpret_cast<const float4*>(dt_row_ptr);
    float4 dt1 = *reinterpret_cast<const float4*>(dt_row_ptr + 4);
    float r_dt[8] = {dt0.x, dt0.y, dt0.z, dt0.w, dt1.x, dt1.y, dt1.z, dt1.w};

    __shared__ alignas(16) float sDG_raw[CHUNK_SIZE * D_HEAD];
    __shared__ float s_a_warp[NUM_THREADS / 32];
    using SwizzleF32 = Swizzle<3, 2, 3>;
    auto smem_layout_f32 = composition(
        SwizzleF32{},
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto sDG = make_tensor(make_smem_ptr(sDG_raw), smem_layout_f32);

    auto cTile64 = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}));

    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);
    int64_t head_tok_off = (static_cast<int64_t>(h) * total_L + chunk_start) * D_HEAD;
    auto gDG_in = make_tensor(
        make_gmem_ptr(dG_h_ptr + head_tok_off),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto tXgDG_in = thr_copy_f32.partition_S(gDG_in);
    auto tXsDG    = thr_copy_f32.partition_D(sDG);
    auto tCrDG    = make_fragment_like(tXgDG_in);
    auto tPredF32 = make_fragment_like<bool>(tXgDG_in);
    auto tXcF32   = thr_copy_f32.partition_S(cTile64);
    #pragma unroll
    for (int m = 0; m < size<1>(tPredF32); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPredF32); ++n) {
            bool in_bounds = (get<0>(tXcF32(Int<0>{}, m, n)) < valid_len);
            #pragma unroll
            for (int v = 0; v < size<0>(tPredF32); ++v) {
                tPredF32(v, m, n) = in_bounds;
            }
        }
    }

    TiledCopyHalf tiled_copy_h;
    auto thr_copy_h = tiled_copy_h.get_slice(tid);
    int64_t col_off_h = static_cast<int64_t>(h) * D_HEAD;

    auto gB_in = make_tensor(
        make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_b + col_off_h),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );
    auto gW_in = make_tensor(
        make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_w + col_off_h),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );
    auto gF_in = make_tensor(
        make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_f + col_off_h),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );
    auto gDB_in = make_tensor(
        make_gmem_ptr(db_h_ptr + head_tok_off),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto gDW_in = make_tensor(
        make_gmem_ptr(dw_h_ptr + head_tok_off),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto gDB_out = make_tensor(
        make_gmem_ptr(d_proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_b + col_off_h),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );
    auto gDW_out = make_tensor(
        make_gmem_ptr(d_proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_w + col_off_h),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );
    auto gDF_out = make_tensor(
        make_gmem_ptr(d_proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_f + col_off_h),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );

    auto tXgB  = thr_copy_h.partition_S(gB_in);
    auto tXgW  = thr_copy_h.partition_S(gW_in);
    auto tXgF  = thr_copy_h.partition_S(gF_in);
    auto tXgDB = thr_copy_h.partition_S(gDB_in);
    auto tXgDW = thr_copy_h.partition_S(gDW_in);
    auto tXgDB_out = thr_copy_h.partition_D(gDB_out);
    auto tXgDW_out = thr_copy_h.partition_D(gDW_out);
    auto tXgDF_out = thr_copy_h.partition_D(gDF_out);

    auto tCrB  = make_fragment_like(tXgB);
    auto tCrW  = make_fragment_like(tXgW);
    auto tCrF  = make_fragment_like(tXgF);
    auto tCrDB = make_fragment_like(tXgDB);
    auto tCrDW = make_fragment_like(tXgDW);

    auto tPred64   = make_fragment_like<bool>(tXgB);
    auto tXcTile64 = thr_copy_h.partition_S(cTile64);
    #pragma unroll
    for (int m = 0; m < size<1>(tPred64); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPred64); ++n) {
            bool in_bounds = (get<0>(tXcTile64(Int<0>{}, m, n)) < valid_len);
            #pragma unroll
            for (int v = 0; v < size<0>(tPred64); ++v) {
                tPred64(v, m, n) = in_bounds;
            }
        }
    }

    // 2. Load dG_h and run reverse suffix scan along chunk tokens
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_f32, tXgDG_in, tCrDG);
    } else {
        clear(tCrDG);
        cute::copy_if(tiled_copy_f32, tPredF32, tXgDG_in, tCrDG);
    }
    cute::copy(tiled_copy_f32, tCrDG, tXsDG);
    __syncthreads();

    if (tid < D_HEAD) {
        float acc = 0.0f;
        for (int r = valid_len - 1; r >= 0; --r) {
            acc += sDG(r, tid);
            sDG(r, tid) = acc;
        }
    }
    __syncthreads();

    // 3. Load raw projections and upstream gate grads; compute VJP to d_proj_inout
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, tXgB, tCrB);
        cute::copy(tiled_copy_h, tXgW, tCrW);
        cute::copy(tiled_copy_h, tXgF, tCrF);
        cute::copy(tiled_copy_h, tXgDB, tCrDB);
        cute::copy(tiled_copy_h, tXgDW, tCrDW);
    } else {
        clear(tCrB);
        clear(tCrW);
        clear(tCrF);
        clear(tCrDB);
        clear(tCrDW);
        cute::copy_if(tiled_copy_h, tPred64, tXgB, tCrB);
        cute::copy_if(tiled_copy_h, tPred64, tXgW, tCrW);
        cute::copy_if(tiled_copy_h, tPred64, tXgF, tCrF);
        cute::copy_if(tiled_copy_h, tPred64, tXgDB, tCrDB);
        cute::copy_if(tiled_copy_h, tPred64, tXgDW, tCrDW);
    }

    float r_d_dt[8];
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        r_d_dt[v] = 0.0f;
    }
    float r_dA = 0.0f;
    constexpr float b_scale = ALLOW_NEG_EIGVAL ? 2.0f : 1.0f;

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int local_row = m * 16 + row_id;
        float sb_f[8];
        float sw_f[8];

        #pragma unroll
        for (int v = 0; v < 8; v += 2) {
            __half2 b_h2  = __halves2half2(to_half_val(tCrB(v, m, 0)), to_half_val(tCrB(v + 1, m, 0)));
            __half2 w_h2  = __halves2half2(to_half_val(tCrW(v, m, 0)), to_half_val(tCrW(v + 1, m, 0)));
            __half2 sb_h2 = fast_sigmoid_h2(b_h2);
            __half2 sw_h2 = fast_sigmoid_h2(w_h2);
            sb_f[v]     = __half2float(__low2half(sb_h2));
            sb_f[v + 1] = __half2float(__high2half(sb_h2));
            sw_f[v]     = __half2float(__low2half(sw_h2));
            sw_f[v + 1] = __half2float(__high2half(sw_h2));
        }

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float db_eff = half_to_float(tCrDB(v, m, 0));
            float dw_eff = half_to_float(tCrDW(v, m, 0));
            float db_raw = db_eff * sb_f[v] * (1.0f - sb_f[v]) * b_scale;
            float dw_raw = dw_eff * sw_f[v] * (1.0f - sw_f[v]);

            float u     = half_to_float(tCrF(v, m, 0)) + r_dt[v];
            float sp    = stable_softplus_f(u);
            float sig_u = stable_sigmoid_f(u);
            float dg    = (local_row < valid_len) ? sDG(local_row, col_base + v) : 0.0f;
            float df    = dg * neg_exp_A * sig_u;

            if (local_row < valid_len) {
                r_d_dt[v] += df;
                r_dA      += dg * (neg_exp_A * sp);
            }

            tCrB(v, m, 0) = float_to_out<T>(db_raw);
            tCrW(v, m, 0) = float_to_out<T>(dw_raw);
            tCrF(v, m, 0) = float_to_out<T>(df);
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, tCrB, tXgDB_out);
        cute::copy(tiled_copy_h, tCrW, tXgDW_out);
        cute::copy(tiled_copy_h, tCrF, tXgDF_out);
    } else {
        cute::copy_if(tiled_copy_h, tPred64, tCrB, tXgDB_out);
        cute::copy_if(tiled_copy_h, tPred64, tCrW, tXgDW_out);
        cute::copy_if(tiled_copy_h, tPred64, tCrF, tXgDF_out);
    }

    // 4. Block reduce d_A_log and d_dt_bias before atomic accumulation
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    float dA_warp = warp_reduce_sum(r_dA);
    if (lane_id == 0) {
        s_a_warp[warp_id] = dA_warp;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_a_warp[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            atomicAdd(d_A_log_ptr + h, val);
        }
    }

    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        sDG(row_id, col_base + v) = r_d_dt[v];
    }
    __syncthreads();

    if (tid < D_HEAD) {
        float dt_sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            dt_sum += sDG(r, tid);
        }
        atomicAdd(d_dt_bias_ptr + static_cast<int64_t>(h) * D_HEAD + tid, dt_sum);
    }
}

// ============================================================================
// Part 2: Host Dispatch Helpers and PyTorch Entry Points
// ============================================================================

namespace proj_conv_prep_bwd_detail {

using TiledCopyChunkHalf = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint4>, cutlass::half_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>, Stride<_8, _1>>{}
));

using TiledCopyChunkFloat = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint4>, float>{},
    Layout<Shape<_8, _16>, Stride<_16, _1>>{},
    Layout<Shape<_1, _4>, Stride<_4, _1>>{}
));

struct ProjOffsets {
    int64_t D_q;
    int64_t D_k;
    int64_t D_v;
    int64_t D_out;
    int64_t D_qkv;
    int64_t offset_b;
    int64_t offset_w;
    int64_t offset_f;
    int64_t offset_ogate;
    int64_t total_proj_dim;
};

inline ProjOffsets compute_proj_offsets(
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v
) {
    ProjOffsets po;
    po.D_q            = num_q_heads * d_k;
    po.D_k            = num_kv_heads * d_k;
    po.D_v            = num_kv_heads * d_v;
    po.D_out          = num_q_heads * d_v;
    po.D_qkv          = po.D_q + po.D_k + po.D_v;
    po.offset_b       = po.D_qkv;
    po.offset_w       = po.offset_b + po.D_k;
    po.offset_f       = po.offset_w + po.D_v;
    po.offset_ogate   = po.offset_f + po.D_k;
    po.total_proj_dim = po.offset_ogate + po.D_out;
    return po;
}

inline void validate_common_proj_args(
    const at::Tensor& proj,
    const at::Tensor& conv_weight,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const std::optional<at::Tensor>& past_conv_state,
    const at::Tensor& cu_seq_lens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t conv_size,
    int64_t chunk_size,
    const ProjOffsets& po,
    int64_t batch_size
) {
    TORCH_CHECK(d_k == 64 && d_v == 64, "d_k and d_v must be 64");
    TORCH_CHECK(conv_size == 4, "conv_size must be 4");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(num_kv_heads > 0 && num_q_heads > 0 && (num_q_heads % num_kv_heads == 0),
                "num_q_heads must be positive and divisible by num_kv_heads");

    TORCH_CHECK(proj.is_cuda() && proj.scalar_type() == at::ScalarType::Half, "proj must be CUDA float16");
    TORCH_CHECK(proj.dim() == 2 && proj.is_contiguous(), "proj must be contiguous 2D");
    TORCH_CHECK(proj.size(1) == po.total_proj_dim, "proj column dimension mismatch");

    TORCH_CHECK(conv_weight.is_cuda() && conv_weight.scalar_type() == at::ScalarType::Half, "conv_weight must be CUDA float16");
    TORCH_CHECK(conv_weight.dim() == 2 && conv_weight.is_contiguous(), "conv_weight must be contiguous 2D");
    TORCH_CHECK(conv_weight.size(0) == conv_size && conv_weight.size(1) == po.D_qkv, "conv_weight shape mismatch");

    TORCH_CHECK(A_log.is_cuda() && A_log.scalar_type() == at::ScalarType::Float, "A_log must be CUDA float32");
    TORCH_CHECK(A_log.dim() == 1 && A_log.is_contiguous() && A_log.size(0) == num_kv_heads, "A_log shape mismatch");

    TORCH_CHECK(dt_bias.is_cuda() && dt_bias.scalar_type() == at::ScalarType::Float, "dt_bias must be CUDA float32");
    TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.is_contiguous(), "dt_bias must be contiguous 2D");
    TORCH_CHECK(dt_bias.size(0) == num_kv_heads && dt_bias.size(1) == d_k, "dt_bias shape mismatch");

    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 1,
                "cu_seq_lens must be a 1D CUDA tensor with size >= 1");
    TORCH_CHECK(cu_seq_lens.scalar_type() == at::ScalarType::Int || cu_seq_lens.scalar_type() == at::ScalarType::Long,
                "cu_seq_lens must be int32 or int64");

    if (past_conv_state.has_value()) {
        const auto& pcs = *past_conv_state;
        TORCH_CHECK(pcs.is_cuda() && pcs.scalar_type() == at::ScalarType::Half, "past_conv_state must be CUDA float16");
        TORCH_CHECK(pcs.dim() == 3 && pcs.is_contiguous(), "past_conv_state must be contiguous 3D");
        TORCH_CHECK(pcs.size(0) == batch_size && pcs.size(1) == conv_size - 1 && pcs.size(2) == po.D_qkv,
                    "past_conv_state shape mismatch");
    }
}

} // namespace proj_conv_prep_bwd_detail
using namespace proj_conv_prep_bwd_detail;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
proj_conv_prep_bwd_with_meta(
    const at::Tensor& dq_h,
    const at::Tensor& dk_h,
    const at::Tensor& dv_h,
    const at::Tensor& db_h,
    const at::Tensor& dw_h,
    const at::Tensor& dG_h,
    const at::Tensor& proj,
    const at::Tensor& conv_weight,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const std::optional<at::Tensor>& past_conv_state,
    const at::Tensor& q_inv_norm,
    const at::Tensor& k_inv_norm,
    const at::Tensor& cu_seq_lens,
    const VarlenChunkMeta& meta,
    at::Tensor& d_proj_inout,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t conv_size,
    int64_t chunk_size,
    double scale,
    bool allow_neg_eigval,
    bool has_past_conv_state
) {
    int64_t batch_size = cu_seq_lens.size(0) - 1;
    ProjOffsets po = compute_proj_offsets(num_q_heads, num_kv_heads, d_k, d_v);
    validate_common_proj_args(
        proj, conv_weight, A_log, dt_bias, past_conv_state, cu_seq_lens,
        num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size, po, batch_size
    );

    int64_t total_L = proj.size(0);
    auto check_head3d = [&](const at::Tensor& t, const char* name, int64_t H, int64_t D, at::ScalarType dtype) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype, name, " dtype or device mismatch");
        TORCH_CHECK(t.dim() == 3 && t.is_contiguous(), name, " must be contiguous 3D");
        TORCH_CHECK(t.size(0) == H && t.size(1) == total_L && t.size(2) == D, name, " shape mismatch");
    };
    check_head3d(dq_h, "dq_h", num_q_heads,  d_k, at::ScalarType::Half);
    check_head3d(dk_h, "dk_h", num_kv_heads, d_k, at::ScalarType::Half);
    check_head3d(dv_h, "dv_h", num_kv_heads, d_v, at::ScalarType::Half);
    check_head3d(db_h, "db_h", num_kv_heads, d_k, at::ScalarType::Half);
    check_head3d(dw_h, "dw_h", num_kv_heads, d_v, at::ScalarType::Half);
    check_head3d(dG_h, "dG_h", num_kv_heads, d_k, at::ScalarType::Float);

    TORCH_CHECK(q_inv_norm.is_cuda() && q_inv_norm.scalar_type() == at::ScalarType::Float && q_inv_norm.is_contiguous(),
                "q_inv_norm must be contiguous CUDA float32");
    TORCH_CHECK(q_inv_norm.dim() == 2 && q_inv_norm.size(0) == num_q_heads && q_inv_norm.size(1) == total_L,
                "q_inv_norm shape mismatch");
    TORCH_CHECK(k_inv_norm.is_cuda() && k_inv_norm.scalar_type() == at::ScalarType::Float && k_inv_norm.is_contiguous(),
                "k_inv_norm must be contiguous CUDA float32");
    TORCH_CHECK(k_inv_norm.dim() == 2 && k_inv_norm.size(0) == num_kv_heads && k_inv_norm.size(1) == total_L,
                "k_inv_norm shape mismatch");

    TORCH_CHECK(d_proj_inout.is_cuda() && d_proj_inout.scalar_type() == at::ScalarType::Half && d_proj_inout.is_contiguous(),
                "d_proj_inout must be contiguous CUDA float16");
    TORCH_CHECK(d_proj_inout.dim() == 2 && d_proj_inout.size(0) == total_L && d_proj_inout.size(1) == po.total_proj_dim,
                "d_proj_inout shape mismatch");

    c10::cuda::CUDAGuard device_guard(proj.device());

    auto h_opts = proj.options();
    auto f_opts = proj.options().dtype(at::kFloat);

    at::Tensor d_A_log           = torch::zeros({num_kv_heads}, f_opts);
    at::Tensor d_dt_bias         = torch::zeros({num_kv_heads, d_k}, f_opts);
    at::Tensor d_past_conv_state = has_past_conv_state
        ? torch::zeros({batch_size, conv_size - 1, po.D_qkv}, h_opts)
        : at::empty({0}, h_opts);

    if (total_L == 0 || meta.num_chunks == 0) {
        at::Tensor d_conv_weight = torch::zeros({conv_size, po.D_qkv}, h_opts);
        return std::make_tuple(d_proj_inout, d_conv_weight, d_A_log, d_dt_bias, d_past_conv_state);
    }

    int64_t H_total = num_q_heads + 2 * num_kv_heads;
    at::Tensor d_conv_buf    = torch::empty({H_total, total_L, d_k}, h_opts);
    at::Tensor d_conv_w_fp32 = torch::zeros({conv_size, po.D_qkv}, f_opts);

    at::Tensor dq_a     = ensure_aligned16(dq_h);
    at::Tensor dk_a     = ensure_aligned16(dk_h);
    at::Tensor dv_a     = ensure_aligned16(dv_h);
    at::Tensor db_a     = ensure_aligned16(db_h);
    at::Tensor dw_a     = ensure_aligned16(dw_h);
    at::Tensor dG_a     = ensure_aligned16(dG_h);
    at::Tensor proj_a   = ensure_aligned16(proj);
    at::Tensor conv_w_a = ensure_aligned16(conv_weight);
    at::Tensor A_log_a  = ensure_aligned16(A_log);
    at::Tensor dt_a     = ensure_aligned16(dt_bias);
    at::Tensor d_proj_a = ensure_aligned16(d_proj_inout);

    bool has_past_conv = past_conv_state.has_value();
    at::Tensor past_conv_a = has_past_conv ? ensure_aligned16(*past_conv_state) : at::Tensor();

    const int32_t* c_off  = meta.chunk_offsets.data_ptr<int32_t>();
    const int32_t* c_vlen = meta.chunk_valid_lens.data_ptr<int32_t>();
    const int32_t* c_seq  = meta.chunk_seq_ids.data_ptr<int32_t>();
    const int32_t* c_loc  = meta.chunk_local_starts.data_ptr<int32_t>();
    const int32_t* c_last = meta.chunk_is_last.data_ptr<int32_t>();

    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_SIZE  = 64;
    constexpr int D_HEAD      = 64;

    // 1. Kernel 1a: L2-norm + SiLU VJP -> d_conv_buf and d_conv_w_fp32
    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        dim3 grid(static_cast<unsigned int>(meta.num_chunks), static_cast<unsigned int>(H_total));
        dim3 block(NUM_THREADS);

        const auto* dq_ptr     = reinterpret_cast<const cutlass::half_t*>(dq_a.data_ptr<at::Half>());
        const auto* dk_ptr     = reinterpret_cast<const cutlass::half_t*>(dk_a.data_ptr<at::Half>());
        const auto* dv_ptr     = reinterpret_cast<const cutlass::half_t*>(dv_a.data_ptr<at::Half>());
        const auto* proj_ptr   = reinterpret_cast<const cutlass::half_t*>(proj_a.data_ptr<at::Half>());
        const auto* cw_ptr     = reinterpret_cast<const cutlass::half_t*>(conv_w_a.data_ptr<at::Half>());
        const auto* pcs_ptr    = has_past_conv ? reinterpret_cast<const cutlass::half_t*>(past_conv_a.data_ptr<at::Half>()) : nullptr;
        const float* qi_ptr    = q_inv_norm.data_ptr<float>();
        const float* ki_ptr    = k_inv_norm.data_ptr<float>();
        auto* dcb_ptr          = reinterpret_cast<cutlass::half_t*>(d_conv_buf.data_ptr<at::Half>());
        float* dcw_fp32_ptr    = d_conv_w_fp32.data_ptr<float>();

        if (has_past_conv) {
            kernel_fused_conv_act_l2_transpose_bwd<
                cutlass::half_t, NUM_THREADS, CHUNK_SIZE, D_HEAD, true, TiledCopyChunkHalf
            ><<<grid, block, 0, stream>>>(
                dq_ptr, dk_ptr, dv_ptr, proj_ptr, cw_ptr, pcs_ptr, qi_ptr, ki_ptr,
                c_off, c_vlen, c_seq, c_loc, dcb_ptr, dcw_fp32_ptr,
                po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale)
            );
        } else {
            kernel_fused_conv_act_l2_transpose_bwd<
                cutlass::half_t, NUM_THREADS, CHUNK_SIZE, D_HEAD, false, TiledCopyChunkHalf
            ><<<grid, block, 0, stream>>>(
                dq_ptr, dk_ptr, dv_ptr, proj_ptr, cw_ptr, pcs_ptr, qi_ptr, ki_ptr,
                c_off, c_vlen, c_seq, c_loc, dcb_ptr, dcw_fp32_ptr,
                po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale)
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 2. Kernel 1b: Transpose-conv scatter -> d_proj_inout QKV slice and d_past_conv_state
    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        dim3 grid(static_cast<unsigned int>(meta.num_chunks), static_cast<unsigned int>(H_total));
        dim3 block(NUM_THREADS);

        const auto* dcb_ptr = reinterpret_cast<const cutlass::half_t*>(d_conv_buf.data_ptr<at::Half>());
        const auto* cw_ptr  = reinterpret_cast<const cutlass::half_t*>(conv_w_a.data_ptr<at::Half>());
        auto* dproj_ptr     = reinterpret_cast<cutlass::half_t*>(d_proj_a.data_ptr<at::Half>());
        auto* dpast_ptr     = has_past_conv_state ? reinterpret_cast<cutlass::half_t*>(d_past_conv_state.data_ptr<at::Half>()) : nullptr;

        if (has_past_conv_state) {
            kernel_fused_conv_dx_transpose_bwd<
                cutlass::half_t, NUM_THREADS, CHUNK_SIZE, D_HEAD, true, TiledCopyChunkHalf
            ><<<grid, block, 0, stream>>>(
                dcb_ptr, cw_ptr, c_off, c_vlen, c_seq, c_loc, c_last, dproj_ptr, dpast_ptr,
                po.total_proj_dim, po.D_qkv, total_L, meta.num_chunks
            );
        } else {
            kernel_fused_conv_dx_transpose_bwd<
                cutlass::half_t, NUM_THREADS, CHUNK_SIZE, D_HEAD, false, TiledCopyChunkHalf
            ><<<grid, block, 0, stream>>>(
                dcb_ptr, cw_ptr, c_off, c_vlen, c_seq, c_loc, c_last, dproj_ptr, dpast_ptr,
                po.total_proj_dim, po.D_qkv, total_L, meta.num_chunks
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    d_conv_buf.reset();

    // 3. Kernel 2: Gates and channel-wise decay backward
    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        dim3 grid(static_cast<unsigned int>(meta.num_chunks), static_cast<unsigned int>(num_kv_heads));
        dim3 block(NUM_THREADS);

        const auto* db_ptr   = reinterpret_cast<const cutlass::half_t*>(db_a.data_ptr<at::Half>());
        const auto* dw_ptr   = reinterpret_cast<const cutlass::half_t*>(dw_a.data_ptr<at::Half>());
        const float* dG_ptr  = dG_a.data_ptr<float>();
        const auto* proj_ptr = reinterpret_cast<const cutlass::half_t*>(proj_a.data_ptr<at::Half>());
        const float* A_ptr   = A_log_a.data_ptr<float>();
        const float* dt_ptr  = dt_a.data_ptr<float>();
        auto* dproj_ptr      = reinterpret_cast<cutlass::half_t*>(d_proj_a.data_ptr<at::Half>());
        float* dA_ptr        = d_A_log.data_ptr<float>();
        float* ddt_ptr       = d_dt_bias.data_ptr<float>();

        if (allow_neg_eigval) {
            kernel_fused_gates_decay_transpose_bwd<
                cutlass::half_t, NUM_THREADS, CHUNK_SIZE, D_HEAD, true, TiledCopyChunkHalf, TiledCopyChunkFloat
            ><<<grid, block, 0, stream>>>(
                db_ptr, dw_ptr, dG_ptr, proj_ptr, A_ptr, dt_ptr, c_off, c_vlen,
                dproj_ptr, dA_ptr, ddt_ptr,
                po.total_proj_dim, po.offset_b, po.offset_w, po.offset_f, total_L, meta.num_chunks
            );
        } else {
            kernel_fused_gates_decay_transpose_bwd<
                cutlass::half_t, NUM_THREADS, CHUNK_SIZE, D_HEAD, false, TiledCopyChunkHalf, TiledCopyChunkFloat
            ><<<grid, block, 0, stream>>>(
                db_ptr, dw_ptr, dG_ptr, proj_ptr, A_ptr, dt_ptr, c_off, c_vlen,
                dproj_ptr, dA_ptr, ddt_ptr,
                po.total_proj_dim, po.offset_b, po.offset_w, po.offset_f, total_L, meta.num_chunks
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (!d_proj_a.is_same(d_proj_inout)) {
        // Copy only [0, offset_ogate) to preserve d_o_gate written by output_combine_bwd
        d_proj_inout.slice(1, 0, po.offset_ogate).copy_(d_proj_a.slice(1, 0, po.offset_ogate));
    }

    at::Tensor d_conv_weight = d_conv_w_fp32.to(at::kHalf);
    d_conv_w_fp32.reset();

    return std::make_tuple(d_proj_inout, d_conv_weight, d_A_log, d_dt_bias, d_past_conv_state);
}

// Outputs: (d_proj_inout, d_conv_weight, d_A_log, d_dt_bias, d_past_conv_state)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
proj_conv_prep_bwd(
    const at::Tensor& dq_h,
    const at::Tensor& dk_h,
    const at::Tensor& dv_h,
    const at::Tensor& db_h,
    const at::Tensor& dw_h,
    const at::Tensor& dG_h,
    const at::Tensor& proj,
    const at::Tensor& conv_weight,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const std::optional<at::Tensor>& past_conv_state,
    const at::Tensor& q_inv_norm,
    const at::Tensor& k_inv_norm,
    const at::Tensor& cu_seq_lens,
    at::Tensor& d_proj_inout,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t conv_size,
    int64_t chunk_size,
    double scale,
    bool allow_neg_eigval,
    bool has_past_conv_state
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 1,
                "cu_seq_lens must be a 1D CUDA tensor with size >= 1");
    VarlenChunkMeta meta = compute_varlen_chunk_metadata(cu_seq_lens, chunk_size);
    return proj_conv_prep_bwd_with_meta(
        dq_h, dk_h, dv_h, db_h, dw_h, dG_h, proj, conv_weight, A_log, dt_bias, past_conv_state,
        q_inv_norm, k_inv_norm, cu_seq_lens, meta, d_proj_inout,
        num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size,
        scale, allow_neg_eigval, has_past_conv_state
    );
}
