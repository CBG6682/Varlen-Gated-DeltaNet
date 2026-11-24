#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm75.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

using namespace cute;

// ============================================================================
// Part 1: Global Kernels
// ============================================================================

// Kernel 1: Per-Head RMSNorm * SiLU(o_gate) + head-last transpose backward (4 KB SMEM)
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_V,
    typename TiledCopyHalf
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_out_norm_swiglu_transpose_bwd(
    const T*       __restrict__ gGradOutTok,      // (total_L, H_q * 64) [FP16] - Upstream gradient
    const T*       __restrict__ gO_raw,           // (H_q, total_L, 64) [FP16] - Saved pre-norm output
    const float*   __restrict__ gO_rstd,          // (H_q, total_L) [FP32] - Saved per-head rstd
    const T*       __restrict__ gProj,            // (total_L, total_proj_dim) [FP16] - Unified projection
    const T*       __restrict__ gONormW,          // (64,) [FP16] - Per-head RMSNorm weight
    const int32_t* __restrict__ chunk_offsets,    // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens, // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ gDO_raw,          // (H_q, total_L, 64) [FP16] - Output grad w.r.t. O_raw
    T*             __restrict__ gDProj_inout,     // (total_L, total_proj_dim) [FP16] - Writes o_gate grad slice
    float*         __restrict__ gDONormW_fp32,    // (64,) [FP32] - Accumulated weight gradient
    int64_t num_q_heads,                          // scalar [int64] - Total Query heads
    int64_t num_chunks,                           // scalar [int64] - Total varlen chunks
    int64_t total_L,                              // scalar [int64] - Total packed tokens
    int64_t total_proj_dim,                       // scalar [int64] - Stride of gProj
    int64_t offset_ogate                          // scalar [int64] - Column start of o_gate
) {
    // 1. Setup
    int chunk_idx = blockIdx.x;
    int h_q       = blockIdx.y;
    int tid       = threadIdx.x;

    if (chunk_idx >= num_chunks || h_q >= num_q_heads) return;

    int32_t valid_len   = chunk_valid_lens[chunk_idx];
    int32_t chunk_start = chunk_offsets[chunk_idx];
    if (valid_len <= 0) return;

    int row_id   = tid / 8;
    int col_base = (tid % 8) * 8;

    __shared__ alignas(16) float s_dw[16 * D_V]; // 4 KB

    TiledCopyHalf tiled_copy_h;
    auto thr_copy_h = tiled_copy_h.get_slice(tid);

    int64_t tok_stride  = num_q_heads * D_V;
    int64_t head_tok_off = (static_cast<int64_t>(h_q) * total_L + chunk_start) * D_V;

    auto gGradOut_tile = make_tensor(
        make_gmem_ptr(gGradOutTok + static_cast<int64_t>(chunk_start) * tok_stride + static_cast<int64_t>(h_q) * D_V),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(tok_stride, Int<1>{}))
    );
    auto gORaw_tile = make_tensor(
        make_gmem_ptr(gO_raw + head_tok_off),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{}))
    );
    auto gOGate_tile = make_tensor(
        make_gmem_ptr(gProj + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_ogate + static_cast<int64_t>(h_q) * D_V),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(total_proj_dim, Int<1>{}))
    );
    auto gDORaw_tile = make_tensor(
        make_gmem_ptr(gDO_raw + head_tok_off),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{}))
    );
    auto gDGate_tile = make_tensor(
        make_gmem_ptr(gDProj_inout + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_ogate + static_cast<int64_t>(h_q) * D_V),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(total_proj_dim, Int<1>{}))
    );

    auto tXgGO    = thr_copy_h.partition_S(gGradOut_tile);
    auto tXgORaw  = thr_copy_h.partition_S(gORaw_tile);
    auto tXgOGate = thr_copy_h.partition_S(gOGate_tile);
    auto tXgDORaw = thr_copy_h.partition_D(gDORaw_tile);
    auto tXgDGate = thr_copy_h.partition_D(gDGate_tile);

    auto tCrGO    = make_fragment_like(tXgGO);
    auto tCrORaw  = make_fragment_like(tXgORaw);
    auto tCrOGate = make_fragment_like(tXgOGate);

    auto tPredHalf64 = make_fragment_like<bool>(tXgGO);
    if (valid_len < CHUNK_SIZE) {
        auto cTile64   = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}));
        auto tXcHalf64 = thr_copy_h.partition_S(cTile64);
        #pragma unroll
        for (int m = 0; m < size<1>(tPredHalf64); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredHalf64); ++n) {
                bool in_bounds = (get<0>(tXcHalf64(Int<0>{}, m, n)) < valid_len);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredHalf64); ++v) {
                    tPredHalf64(v, m, n) = in_bounds;
                }
            }
        }
    }

    // Broadcast 1D o_norm_weight vector via L1
    const uint4 ow_u4 = *reinterpret_cast<const uint4*>(gONormW + col_base);
    const __half* ow_h = reinterpret_cast<const __half*>(&ow_u4);
    float r_ow[8];
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        r_ow[v] = __half2float(ow_h[v]);
    }

    // 2. Load tiles, evaluate RMSNorm + SiLU(o_gate) VJP, and store dO_raw & d_o_gate
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, tXgGO,    tCrGO);
        cute::copy(tiled_copy_h, tXgORaw,  tCrORaw);
        cute::copy(tiled_copy_h, tXgOGate, tCrOGate);
    } else {
        clear(tCrGO);
        clear(tCrORaw);
        clear(tCrOGate);
        cute::copy_if(tiled_copy_h, tPredHalf64, tXgGO,    tCrGO);
        cute::copy_if(tiled_copy_h, tPredHalf64, tXgORaw,  tCrORaw);
        cute::copy_if(tiled_copy_h, tPredHalf64, tXgOGate, tCrOGate);
    }

    float r_dw[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int64_t rstd_base = static_cast<int64_t>(h_q) * total_L + chunk_start;

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int   local_row = m * 16 + row_id;
        bool  row_valid = (local_row < valid_len);
        float rstd      = row_valid ? gO_rstd[rstd_base + local_row] : 0.0f;

        float o_norm[8];
        float dy_w[8];
        float local_dot = 0.0f;

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float go    = half_to_float(tCrGO(v, m, Int<0>{}));
            float on    = half_to_float(tCrORaw(v, m, Int<0>{})) * rstd;
            float g_val = half_to_float(tCrOGate(v, m, Int<0>{}));

            float sig = (g_val >= 0.0f)
                ? (1.0f / (1.0f + expf(-g_val)))
                : (expf(g_val) / (1.0f + expf(g_val)));
            float silu_g  = g_val * sig;
            float dsilu_g = sig * (1.0f + g_val * (1.0f - sig));

            float d_ogate = go * (on * r_ow[v]) * dsilu_g;
            tCrOGate(v, m, Int<0>{}) = float_to_out<T>(d_ogate);

            float dy = go * silu_g;
            if (row_valid) {
                r_dw[v] += dy * on;
            }
            float dw_val = dy * r_ow[v];
            o_norm[v] = on;
            dy_w[v]   = dw_val;
            local_dot += dw_val * on;
        }

        float mean_dot = octet_reduce_sum(local_dot) * (1.0f / static_cast<float>(D_V));

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float do_val = rstd * (dy_w[v] - o_norm[v] * mean_dot);
            tCrORaw(v, m, Int<0>{}) = float_to_out<T>(do_val);
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, tCrORaw,  tXgDORaw);
        cute::copy(tiled_copy_h, tCrOGate, tXgDGate);
    } else {
        cute::copy_if(tiled_copy_h, tPredHalf64, tCrORaw,  tXgDORaw);
        cute::copy_if(tiled_copy_h, tPredHalf64, tCrOGate, tXgDGate);
    }

    // 3. Reduce d_o_norm_weight across 16 row groups before single atomicAdd per channel
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        s_dw[row_id * D_V + col_base + v] = r_dw[v];
    }
    __syncthreads();

    if (tid < D_V) {
        float acc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            acc += s_dw[r * D_V + tid];
        }
        atomicAdd(gDONormW_fp32 + tid, acc);
    }
}

// Kernel 2: Inter-chunk gradients (dQ_inter, dG_inter, dS_out [FP32]) and dR_out [FP16] (exact 32 KB SMEM)
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int SUB_CHUNK,
    int D_K,
    int D_V,
    typename MMAAtom,
    typename TiledMMA16,
    typename TiledMMA221,
    typename S2RCopyAtomA_KM,
    typename S2RCopyAtomB_KM,
    typename S2RCopyAtomA_MNM,
    typename S2RCopyAtomB_MNM,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_output_combine_dR_dS_dQinter_bwd(
    const T*       __restrict__ gDO_raw,          // (H_q, total_L, 64) [FP16] - Grad O_raw
    const T*       __restrict__ gQ_h,             // (H_q, total_L, 64) [FP16] - Scaled normalized Query
    const T*       __restrict__ gK_h,             // (H_kv, total_L, 64) [FP16] - Normalized Key
    const float*   __restrict__ gG_log2_h,        // (H_kv, total_L, 64) [FP32] - Chunk cumsum log2 decay
    const float*   __restrict__ gS_list,          // (H_kv, num_chunks, 64, 64) [FP32] - Chunk initial states
    const int32_t* __restrict__ chunk_offsets,    // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens, // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ gDQ_h,            // (H_q, total_L, 64) [FP16] - Output grad Query (inter)
    float*         __restrict__ gDG_ext,          // (H_kv, total_L, 64) [FP32] - Output grad decay (inter)
    float*         __restrict__ gDS_out,          // (H_kv, num_chunks, 64, 64) [FP32] - Output grad S_out
    T*             __restrict__ gDR_out,          // (H_kv, num_chunks, 64, 64) [FP16] - Output grad R_out
    int64_t num_q_heads,                          // scalar [int64] - Total Query heads
    int64_t num_kv_heads,                         // scalar [int64] - Total KV heads
    int64_t num_groups,                           // scalar [int64] - Query heads per KV group
    int64_t num_chunks,                           // scalar [int64] - Total varlen chunks
    int64_t total_L                               // scalar [int64] - Total packed tokens
) {
    // 1. Setup & SMEM Allocation (exact 32.0 KB)
    int curr_chunk = blockIdx.x;
    int h_kv       = blockIdx.y;
    int tid        = threadIdx.x;

    if (curr_chunk >= num_chunks || h_kv >= num_kv_heads) return;

    int32_t valid_len   = chunk_valid_lens[curr_chunk];
    int32_t chunk_start = chunk_offsets[curr_chunk];
    if (valid_len <= 0) return;

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    int last_sub       = (valid_len - 1) / SUB_CHUNK;
    int last_sub_valid = valid_len - last_sub * SUB_CHUNK;

    struct alignas(16) SmemStorageK2 {
        T smem_1[CHUNK_SIZE * D_K];
        T smem_2[CHUNK_SIZE * D_V];
        union {
            struct {
                T buf_3[CHUNK_SIZE * D_K];
                T buf_4[CHUNK_SIZE * D_K];
            } half_pair;
            struct {
                T     q_sub[SUB_CHUNK * D_K];
                T     k_sub[SUB_CHUNK * D_K];
                float G_sub[SUB_CHUNK * D_K];
                T     pad_4[CHUNK_SIZE * D_K];
            } sub_pair;
            float dS_f32[D_V * D_K];
        } smem_34;
    };
    static_assert(sizeof(SmemStorageK2) == 32768, "SmemStorageK2 must be exactly 32 KB (32768 bytes)");
    __shared__ SmemStorageK2 smem;

    using SwizzleHalf = Swizzle<3, 3, 3>;
    using SwizzleF32  = Swizzle<3, 2, 3>;

    auto layout_64x64_km  = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_64x64_mnm = composition(SwizzleHalf{}, make_layout(make_shape(Int<D_K>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_K>{})));
    auto layout_64x64_f32 = composition(SwizzleF32{},  make_layout(make_shape(Int<D_V>{}, Int<D_K>{}),        make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_sub_half  = composition(SwizzleHalf{}, make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}),  make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_sub_f32   = composition(SwizzleF32{},  make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}),  make_stride(Int<D_K>{}, Int<1>{})));

    auto s1_km     = make_tensor(make_smem_ptr(smem.smem_1),                  layout_64x64_km);
    auto s1_mnm    = make_tensor(make_smem_ptr(smem.smem_1),                  layout_64x64_mnm);
    auto s2_km     = make_tensor(make_smem_ptr(smem.smem_2),                  layout_64x64_km);
    auto s2_mnm    = make_tensor(make_smem_ptr(smem.smem_2),                  layout_64x64_mnm);
    auto s3_km     = make_tensor(make_smem_ptr(smem.smem_34.half_pair.buf_3), layout_64x64_km);
    auto s3_mnm    = make_tensor(make_smem_ptr(smem.smem_34.half_pair.buf_3), layout_64x64_mnm);
    auto s4_km     = make_tensor(make_smem_ptr(smem.smem_34.half_pair.buf_4), layout_64x64_km);
    auto sDS_f32   = make_tensor(make_smem_ptr(smem.smem_34.dS_f32),          layout_64x64_f32);
    auto sSubQ     = make_tensor(make_smem_ptr(smem.smem_34.sub_pair.q_sub),  layout_sub_half);
    auto sSubK     = make_tensor(make_smem_ptr(smem.smem_34.sub_pair.k_sub),  layout_sub_half);
    auto sSubG     = make_tensor(make_smem_ptr(smem.smem_34.sub_pair.G_sub),  layout_sub_f32);

    TiledCopyHalf  tiled_copy_h;
    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_h   = tiled_copy_h.get_slice(tid);
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);

    auto tXs1_d    = thr_copy_h.partition_D(s1_km);
    auto tXs2_s    = thr_copy_h.partition_S(s2_km);
    auto tXs2_d    = thr_copy_h.partition_D(s2_km);
    auto tXs3_d    = thr_copy_h.partition_D(s3_km);
    auto tXs4_s    = thr_copy_h.partition_S(s4_km);
    auto tXsDS_s   = thr_copy_f32.partition_S(sDS_f32);
    auto tXsSubQ_d = thr_copy_h.partition_D(sSubQ);
    auto tXsSubK_d = thr_copy_h.partition_D(sSubK);
    auto tXsSubG_d = thr_copy_f32.partition_D(sSubG);

    auto tCrBuf0 = make_fragment_like(tXs1_d);
    auto tCrBuf1 = make_fragment_like(tXs1_d);
    auto tCrSubA = make_fragment_like(tXsSubQ_d);
    auto tCrSubB = make_fragment_like(tXsSubK_d);

    auto cTile64     = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}));
    auto tXcHalf64   = thr_copy_h.partition_S(cTile64);
    auto tXcF32_64   = thr_copy_f32.partition_S(cTile64);
    auto tCrG64      = make_tensor<float>(shape(tXcF32_64));
    auto tPredHalf64 = make_fragment_like<bool>(tCrBuf0);
    auto tPredF32_64 = make_fragment_like<bool>(tCrG64);

    if (valid_len < CHUNK_SIZE) {
        #pragma unroll
        for (int m = 0; m < size<1>(tPredHalf64); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredHalf64); ++n) {
                bool in_bounds = (get<0>(tXcHalf64(Int<0>{}, m, n)) < valid_len);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredHalf64); ++v) {
                    tPredHalf64(v, m, n) = in_bounds;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tPredF32_64); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredF32_64); ++n) {
                bool in_bounds = (get<0>(tXcF32_64(Int<0>{}, m, n)) < valid_len);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredF32_64); ++v) {
                    tPredF32_64(v, m, n) = in_bounds;
                }
            }
        }
    }

    auto cSub16       = make_identity_tensor(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}));
    auto tXcHalfSub   = thr_copy_h.partition_S(cSub16);
    auto tXcF32Sub    = thr_copy_f32.partition_S(cSub16);
    auto tCrGSub      = make_tensor<float>(shape(tXcF32Sub));
    auto tPredHalfSub = make_fragment_like<bool>(tCrSubA);
    auto tPredF32Sub  = make_fragment_like<bool>(tCrGSub);

    if (last_sub_valid < SUB_CHUNK) {
        #pragma unroll
        for (int m = 0; m < size<1>(tPredHalfSub); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredHalfSub); ++n) {
                bool in_bounds = (get<0>(tXcHalfSub(Int<0>{}, m, n)) < last_sub_valid);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredHalfSub); ++v) {
                    tPredHalfSub(v, m, n) = in_bounds;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tPredF32Sub); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredF32Sub); ++n) {
                bool in_bounds = (get<0>(tXcF32Sub(Int<0>{}, m, n)) < last_sub_valid);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredF32Sub); ++v) {
                    tPredF32Sub(v, m, n) = in_bounds;
                }
            }
        }
    }

    auto layout_64x64_gmem = make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
    auto layout_sub_gmem   = make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));

    int64_t kv_tok_base = (static_cast<int64_t>(h_kv) * total_L + chunk_start) * D_K;
    int64_t kv_mat_off  = (static_cast<int64_t>(h_kv) * num_chunks + curr_chunk) * (CHUNK_SIZE * D_K);

    // 2. Prologue: stage stationary S_fp16 into smem_1
    {
        auto gS_in = make_tensor(make_gmem_ptr(gS_list + kv_mat_off), layout_64x64_gmem);
        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gS_in), tCrG64);
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                tCrBuf0(v, m, Int<0>{}) = float_to_out<T>(tCrG64(v, m, Int<0>{}));
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tXs1_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs1_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
        }
    }
    __syncthreads();

    // 3. Loop 1: dQ_inter, dG_inter, and dS_out [FP32]
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t h_q        = static_cast<int64_t>(h_kv) * num_groups + g;
        int64_t q_tok_base = (h_q * total_L + chunk_start) * D_K;

        auto gDO_in = make_tensor(make_gmem_ptr(gDO_raw   + q_tok_base),  layout_64x64_gmem);
        auto gQ_in  = make_tensor(make_gmem_ptr(gQ_h      + q_tok_base),  layout_64x64_gmem);
        auto gG_in  = make_tensor(make_gmem_ptr(gG_log2_h + kv_tok_base), layout_64x64_gmem);

        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gDO_in), tCrBuf0);
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQ_in),  tCrBuf1);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gG_in), tCrG64);
        } else {
            clear(tCrBuf0);
            clear(tCrBuf1);
            clear(tCrG64);
            cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gDO_in), tCrBuf0);
            cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gQ_in),  tCrBuf1);
            cute::copy_if(tiled_copy_f32, tPredF32_64, thr_copy_f32.partition_S(gG_in), tCrG64);
        }

        #pragma unroll
        for (int m = 0; m < size<1>(tXs2_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs2_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
        }

        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int  local_row = m * 16 + row_id;
            bool row_valid = (local_row < valid_len);
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float val = row_valid
                    ? (half_to_float(tCrBuf1(v, m, Int<0>{})) * exp2f(tCrG64(v, m, Int<0>{})))
                    : 0.0f;
                tCrBuf1(v, m, Int<0>{}) = float_to_out<T>(val);
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tXs3_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs3_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf1(Int<0>{}, m, Int<0>{}));
        }
        __syncthreads();

        // GEMM 1 (dQ_gamma = dO_g @ S_fp16) and GEMM 2 (dS_g = dO_g^T @ Q_gamma_g)
        {
            TiledMMA221 tiled_mma_221;
            auto thr_mma_221      = tiled_mma_221.get_slice(tid);
            auto s2r_copy_a221_k  = make_tiled_copy_A(S2RCopyAtomA_KM{},  tiled_mma_221);
            auto s2r_copy_a221_m  = make_tiled_copy_A(S2RCopyAtomA_MNM{}, tiled_mma_221);
            auto s2r_copy_b221_m  = make_tiled_copy_B(S2RCopyAtomB_MNM{}, tiled_mma_221);

            auto s2r_thr_a221_k = s2r_copy_a221_k.get_slice(tid);
            auto s2r_thr_a221_m = s2r_copy_a221_m.get_slice(tid);
            auto s2r_thr_b221_m = s2r_copy_b221_m.get_slice(tid);

            auto tCsA_dO_k = s2r_thr_a221_k.partition_S(s2_km);
            auto tCsB_S_m  = s2r_thr_b221_m.partition_S(s1_mnm);
            auto tCsC_4    = thr_mma_221.partition_C(s4_km);

            auto tCrA         = thr_mma_221.partition_fragment_A(s2_km);
            auto tCrB         = thr_mma_221.partition_fragment_B(s1_mnm);
            auto tCrA_km_view = s2r_thr_a221_k.retile_D(tCrA);
            auto tCrA_mn_view = s2r_thr_a221_m.retile_D(tCrA);
            auto tCrB_mn_view = s2r_thr_b221_m.retile_D(tCrB);
            auto tCrC         = thr_mma_221.make_fragment_C(tCsC_4);
            auto rC_half      = make_tensor_like<T>(tCrC);

            clear(tCrC);
            cute::copy(s2r_copy_a221_k, tCsA_dO_k, tCrA_km_view);
            cute::copy(s2r_copy_b221_m, tCsB_S_m,  tCrB_mn_view);
            cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);

            #pragma unroll
            for (int i = 0; i < size(tCrC); ++i) {
                rC_half(i) = float_to_out<T>(tCrC(i));
            }
            cute::copy(rC_half, tCsC_4);
            __syncthreads();

            cute::copy(tiled_copy_h, tXs4_s, tCrBuf0);
            if (valid_len == CHUNK_SIZE) {
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQ_in),  tCrBuf1);
                cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gG_in), tCrG64);
            } else {
                cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gQ_in),  tCrBuf1);
                cute::copy_if(tiled_copy_f32, tPredF32_64, thr_copy_f32.partition_S(gG_in), tCrG64);
            }

            auto gDG_tile = make_tensor(make_gmem_ptr(gDG_ext + kv_tok_base), layout_64x64_gmem);
            auto gDQ_tile = make_tensor(make_gmem_ptr(gDQ_h   + q_tok_base),  layout_64x64_gmem);

            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    float dq_f = half_to_float(tCrBuf0(v, m, Int<0>{})) * exp2f(tCrG64(v, m, Int<0>{}));
                    tCrBuf0(v, m, Int<0>{}) = float_to_out<T>(dq_f);
                    tCrG64(v, m, Int<0>{})  = dq_f * half_to_float(tCrBuf1(v, m, Int<0>{}));
                }
            }
            if (valid_len == CHUNK_SIZE) {
                cute::copy(tiled_copy_h, tCrBuf0, thr_copy_h.partition_D(gDQ_tile));
            } else {
                cute::copy_if(tiled_copy_h, tPredHalf64, tCrBuf0, thr_copy_h.partition_D(gDQ_tile));
            }

            if (g > 0) {
                auto tCrGPrev = make_fragment_like(tCrG64);
                if (valid_len == CHUNK_SIZE) {
                    cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDG_tile), tCrGPrev);
                } else {
                    clear(tCrGPrev);
                    cute::copy_if(tiled_copy_f32, tPredF32_64, thr_copy_f32.partition_S(gDG_tile), tCrGPrev);
                }
                #pragma unroll
                for (int i = 0; i < size(tCrG64); ++i) {
                    tCrG64(i) += tCrGPrev(i);
                }
            }
            if (valid_len == CHUNK_SIZE) {
                cute::copy(tiled_copy_f32, tCrG64, thr_copy_f32.partition_D(gDG_tile));
            } else {
                cute::copy_if(tiled_copy_f32, tPredF32_64, tCrG64, thr_copy_f32.partition_D(gDG_tile));
            }

            // GEMM 2: dS_g = dO_g^T @ Q_gamma_g
            auto tCsA_dO_m = s2r_thr_a221_m.partition_S(s2_mnm);
            auto tCsB_Qg_m = s2r_thr_b221_m.partition_S(s3_mnm);
            auto tCsC_dS   = thr_mma_221.partition_C(sDS_f32);

            clear(tCrC);
            cute::copy(s2r_copy_a221_m, tCsA_dO_m, tCrA_mn_view);
            cute::copy(s2r_copy_b221_m, tCsB_Qg_m, tCrB_mn_view);
            cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);
            __syncthreads();

            cute::copy(tCrC, tCsC_dS);
            __syncthreads();
        }

        cute::copy(tiled_copy_f32, tXsDS_s, tCrG64);
        auto gDS_tile = make_tensor(make_gmem_ptr(gDS_out + kv_mat_off), layout_64x64_gmem);
        if (g > 0) {
            auto tCrGPrev = make_fragment_like(tCrG64);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDS_tile), tCrGPrev);
            #pragma unroll
            for (int i = 0; i < size(tCrG64); ++i) {
                tCrG64(i) += tCrGPrev(i);
            }
        }
        cute::copy(tiled_copy_f32, tCrG64, thr_copy_f32.partition_D(gDS_tile));
        __syncthreads();
    }

    // 4. Loop 2: recompute Aqk_g and accumulate tCrC_dR += Aqk_g^T @ dO_g in FP32
    {
        TiledMMA221 tiled_mma_221;
        auto thr_mma_221     = tiled_mma_221.get_slice(tid);
        auto s2r_copy_a221_m = make_tiled_copy_A(S2RCopyAtomA_MNM{}, tiled_mma_221);
        auto s2r_copy_b221_m = make_tiled_copy_B(S2RCopyAtomB_MNM{}, tiled_mma_221);
        auto s2r_thr_a221_m  = s2r_copy_a221_m.get_slice(tid);
        auto s2r_thr_b221_m  = s2r_copy_b221_m.get_slice(tid);

        auto tCsA_Aqk = s2r_thr_a221_m.partition_S(s2_mnm);
        auto tCsB_dO  = s2r_thr_b221_m.partition_S(s1_mnm);
        auto tCsC_2   = thr_mma_221.partition_C(s2_km);

        auto tCrA        = thr_mma_221.partition_fragment_A(s2_mnm);
        auto tCrB        = thr_mma_221.partition_fragment_B(s1_mnm);
        auto tCrA_mn_vw  = s2r_thr_a221_m.retile_D(tCrA);
        auto tCrB_mn_vw  = s2r_thr_b221_m.retile_D(tCrB);
        auto tCrC_dR     = thr_mma_221.make_fragment_C(tCsC_2);
        clear(tCrC_dR);

        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t h_q        = static_cast<int64_t>(h_kv) * num_groups + g;
            int64_t q_tok_base = (h_q * total_L + chunk_start) * D_K;

            // Zero smem_2 before assembling Aqk_g
            clear(tCrBuf0);
            #pragma unroll
            for (int m = 0; m < size<1>(tXs2_d); ++m) {
                *reinterpret_cast<uint4*>(&tXs2_d(Int<0>{}, m, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
            }

            // Diagonal 16x16 blocks of Aqk_g (inclusive causal j <= row_id)
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                int sub_start = s * SUB_CHUNK;
                if (sub_start >= valid_len) break;

                int     sub_valid   = (s == last_sub) ? last_sub_valid : SUB_CHUNK;
                bool    is_full_sub = (sub_valid == SUB_CHUNK);
                int64_t q_sub_off   = q_tok_base  + static_cast<int64_t>(sub_start) * D_K;
                int64_t kv_sub_off  = kv_tok_base + static_cast<int64_t>(sub_start) * D_K;

                auto gQs = make_tensor(make_gmem_ptr(gQ_h      + q_sub_off),  layout_sub_gmem);
                auto gKs = make_tensor(make_gmem_ptr(gK_h      + kv_sub_off), layout_sub_gmem);
                auto gGs = make_tensor(make_gmem_ptr(gG_log2_h + kv_sub_off), layout_sub_gmem);

                if (is_full_sub) {
                    cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQs),   tCrSubA);
                    cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKs),   tCrSubB);
                    cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGs), tCrGSub);
                } else {
                    clear(tCrSubA);
                    clear(tCrSubB);
                    clear(tCrGSub);
                    cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gQs),   tCrSubA);
                    cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gKs),   tCrSubB);
                    cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGs), tCrGSub);
                }

                *reinterpret_cast<uint4*>(&tXsSubK_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrSubB(Int<0>{}, Int<0>{}, Int<0>{}));
                #pragma unroll
                for (int x = 0; x < 2; ++x) {
                    *reinterpret_cast<uint4*>(&tXsSubG_d(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{})) =
                        *reinterpret_cast<const uint4*>(&tCrGSub(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{}));
                }
                __syncthreads();

                float q_r[8];
                float g_r[8];
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    q_r[v] = (row_id < sub_valid) ? half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) : 0.0f;
                    g_r[v] = (row_id < sub_valid) ? tCrGSub(v, Int<0>{}, Int<0>{}) : 0.0f;
                }

                for (int j = 0; j < SUB_CHUNK; ++j) {
                    float partial = 0.0f;
                    if (j <= row_id && row_id < sub_valid) {
                        #pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            partial += q_r[v] *
                                       half_to_float(sSubK(j, col_base + v)) *
                                       exp2f(g_r[v] - sSubG(j, col_base + v));
                        }
                    }
                    float a_rj = octet_reduce_sum(partial);
                    if (col_id == 0 && j <= row_id && row_id < sub_valid) {
                        s2_km(sub_start + row_id, sub_start + j) = float_to_out<T>(a_rj);
                    }
                }
                __syncthreads();
            }

            // Off-diagonal 16x16 blocks of Aqk_g
            {
                TiledMMA16 tiled_mma_16;
                auto thr_mma_16   = tiled_mma_16.get_slice(tid & 63);
                auto s2r_copy_a16 = make_tiled_copy_A(S2RCopyAtomA_KM{}, tiled_mma_16);
                auto s2r_copy_b16 = make_tiled_copy_B(S2RCopyAtomB_KM{}, tiled_mma_16);
                auto s2r_thr_a16  = s2r_copy_a16.get_slice(tid & 63);
                auto s2r_thr_b16  = s2r_copy_b16.get_slice(tid & 63);

                auto tCsA_16    = s2r_thr_a16.partition_S(sSubQ);
                auto tCsB_16    = s2r_thr_b16.partition_S(sSubK);
                auto tCrA_16    = thr_mma_16.partition_fragment_A(sSubQ);
                auto tCrB_16    = thr_mma_16.partition_fragment_B(sSubK);
                auto tCrA_16_vw = s2r_thr_a16.retile_D(tCrA_16);
                auto tCrB_16_vw = s2r_thr_b16.retile_D(tCrB_16);
                auto tCsC_16_00 = thr_mma_16.partition_C(
                    local_tile(s2_km, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(Int<0>{}, Int<0>{}))
                );
                auto tCrC_16    = thr_mma_16.make_fragment_C(tCsC_16_00);
                auto rC_half_16 = make_tensor_like<T>(tCrC_16);

                for (int i = 1; i < 4; ++i) {
                    int i_start = i * SUB_CHUNK;
                    if (i_start >= valid_len) break;

                    int     i_valid   = (i == last_sub) ? last_sub_valid : SUB_CHUNK;
                    bool    is_full_i = (i_valid == SUB_CHUNK);
                    int64_t q_i_off   = q_tok_base  + static_cast<int64_t>(i_start) * D_K;
                    int64_t kv_i_off  = kv_tok_base + static_cast<int64_t>(i_start) * D_K;

                    const float* p_g_ref = gG_log2_h + kv_i_off + col_base;
                    float4 gr0 = *reinterpret_cast<const float4*>(p_g_ref);
                    float4 gr1 = *reinterpret_cast<const float4*>(p_g_ref + 4);
                    float r_gref[8] = {gr0.x, gr0.y, gr0.z, gr0.w, gr1.x, gr1.y, gr1.z, gr1.w};

                    auto gQi = make_tensor(make_gmem_ptr(gQ_h      + q_i_off),  layout_sub_gmem);
                    auto gGi = make_tensor(make_gmem_ptr(gG_log2_h + kv_i_off), layout_sub_gmem);

                    if (is_full_i) {
                        cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQi),   tCrSubA);
                        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGi), tCrGSub);
                    } else {
                        clear(tCrSubA);
                        clear(tCrGSub);
                        cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gQi),   tCrSubA);
                        cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGi), tCrGSub);
                    }

                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        float val = (row_id < i_valid)
                            ? (half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) *
                               exp2f(tCrGSub(v, Int<0>{}, Int<0>{}) - r_gref[v]))
                            : 0.0f;
                        tCrSubA(v, Int<0>{}, Int<0>{}) = float_to_out<T>(val);
                    }
                    *reinterpret_cast<uint4*>(&tXsSubQ_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                        *reinterpret_cast<const uint4*>(&tCrSubA(Int<0>{}, Int<0>{}, Int<0>{}));
                    __syncthreads();

                    if (tid < 64) {
                        cute::copy(s2r_copy_a16, tCsA_16, tCrA_16_vw);
                    }

                    for (int j = 0; j < i; ++j) {
                        int64_t kv_j_off = kv_tok_base + static_cast<int64_t>(j * SUB_CHUNK) * D_K;
                        auto gKj = make_tensor(make_gmem_ptr(gK_h      + kv_j_off), layout_sub_gmem);
                        auto gGj = make_tensor(make_gmem_ptr(gG_log2_h + kv_j_off), layout_sub_gmem);

                        cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKj),   tCrSubB);
                        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGj), tCrGSub);

                        #pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            float val = half_to_float(tCrSubB(v, Int<0>{}, Int<0>{})) *
                                        exp2f(r_gref[v] - tCrGSub(v, Int<0>{}, Int<0>{}));
                            tCrSubB(v, Int<0>{}, Int<0>{}) = float_to_out<T>(val);
                        }
                        *reinterpret_cast<uint4*>(&tXsSubK_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                            *reinterpret_cast<const uint4*>(&tCrSubB(Int<0>{}, Int<0>{}, Int<0>{}));
                        __syncthreads();

                        if (tid < 64) {
                            cute::copy(s2r_copy_b16, tCsB_16, tCrB_16_vw);
                            clear(tCrC_16);
                            cute::gemm(MMAAtom{}, tCrA_16, tCrB_16, tCrC_16);

                            #pragma unroll
                            for (int idx = 0; idx < size(tCrC_16); ++idx) {
                                rC_half_16(idx) = float_to_out<T>(tCrC_16(idx));
                            }
                            auto s2_ij = local_tile(s2_km, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(i, j));
                            cute::copy(rC_half_16, thr_mma_16.partition_C(s2_ij));
                        }
                        __syncthreads();
                    }
                }
            }

            // Load dO_g into smem_1 for GEMM 3
            auto gDO_in = make_tensor(make_gmem_ptr(gDO_raw + q_tok_base), layout_64x64_gmem);
            if (valid_len == CHUNK_SIZE) {
                cute::copy(tiled_copy_h, thr_copy_h.partition_S(gDO_in), tCrBuf0);
            } else {
                clear(tCrBuf0);
                cute::copy_if(tiled_copy_h, tPredHalf64, thr_copy_h.partition_S(gDO_in), tCrBuf0);
            }
            #pragma unroll
            for (int m = 0; m < size<1>(tXs1_d); ++m) {
                *reinterpret_cast<uint4*>(&tXs1_d(Int<0>{}, m, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
            }
            __syncthreads();

            // GEMM 3: tCrC_dR += Aqk_g^T @ dO_g
            cute::copy(s2r_copy_a221_m, tCsA_Aqk, tCrA_mn_vw);
            cute::copy(s2r_copy_b221_m, tCsB_dO,  tCrB_mn_vw);
            cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_dR);
            __syncthreads();
        }

        // Stage dR_out [FP16] through swizzled smem_2 then store 128-bit
        auto rC_half_dR = make_tensor_like<T>(tCrC_dR);
        #pragma unroll
        for (int i = 0; i < size(tCrC_dR); ++i) {
            rC_half_dR(i) = float_to_out<T>(tCrC_dR(i));
        }
        cute::copy(rC_half_dR, tCsC_2);
        __syncthreads();

        auto gDR_tile = make_tensor(make_gmem_ptr(gDR_out + kv_mat_off), layout_64x64_gmem);
        cute::copy(tiled_copy_h, tXs2_s, tCrBuf0);
        cute::copy(tiled_copy_h, tCrBuf0, thr_copy_h.partition_D(gDR_tile));
    }
}

// Kernel 3: Intra-chunk dAqk = tril(dO @ R^T) and BC=16 dQ_intra, dK_intra, dG_intra (exact 32 KB SMEM)
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int SUB_CHUNK,
    int D_K,
    int D_V,
    typename MMAAtom,
    typename TiledMMA141,
    typename TiledMMA221,
    typename S2RCopyAtomA_KM,
    typename S2RCopyAtomB_KM,
    typename S2RCopyAtomA_MNM,
    typename S2RCopyAtomB_MNM,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_output_combine_intra_dqk_bwd(
    const T*       __restrict__ gDO_raw,          // (H_q, total_L, 64) [FP16] - Grad O_raw
    const T*       __restrict__ gR_list,          // (H_kv, num_chunks, 64, 64) [FP16] - Saved residual R
    const T*       __restrict__ gQ_h,             // (H_q, total_L, 64) [FP16] - Scaled normalized Query
    const T*       __restrict__ gK_h,             // (H_kv, total_L, 64) [FP16] - Normalized Key
    const float*   __restrict__ gG_log2_h,        // (H_kv, total_L, 64) [FP32] - Chunk cumsum log2 decay
    const int32_t* __restrict__ chunk_offsets,    // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens, // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ gDQ_h,            // (H_q, total_L, 64) [FP16] - Adds dQ_intra in-place
    T*             __restrict__ gDK_ext,          // (H_kv, total_L, 64) [FP16] - Final intra dK_ext
    float*         __restrict__ gDK_fp32,         // (H_kv, total_L, 64) [FP32] - Group dK scratch (nullable)
    float*         __restrict__ gDG_ext,          // (H_kv, total_L, 64) [FP32] - Adds q*dQ - k*dK in-place
    int64_t num_q_heads,                          // scalar [int64] - Total Query heads
    int64_t num_kv_heads,                         // scalar [int64] - Total KV heads
    int64_t num_groups,                           // scalar [int64] - Query heads per KV group
    int64_t num_chunks,                           // scalar [int64] - Total varlen chunks
    int64_t total_L                               // scalar [int64] - Total packed tokens
) {
    // 1. Setup & SMEM Allocation (exact 32.0 KB)
    int curr_chunk = blockIdx.x;
    int h_kv       = blockIdx.y;
    int tid        = threadIdx.x;

    if (curr_chunk >= num_chunks || h_kv >= num_kv_heads) return;

    int32_t valid_len   = chunk_valid_lens[curr_chunk];
    int32_t chunk_start = chunk_offsets[curr_chunk];
    if (valid_len <= 0) return;

    int row_id   = tid / 8;
    int col_base = (tid % 8) * 8;

    int last_sub       = (valid_len - 1) / SUB_CHUNK;
    int last_sub_valid = valid_len - last_sub * SUB_CHUNK;

    struct alignas(16) SmemStorageK3 {
        T smem_1[CHUNK_SIZE * D_V];
        T smem_2[CHUNK_SIZE * CHUNK_SIZE];
        union {
            T dO_buf[CHUNK_SIZE * D_V];
            struct {
                T     a_sub[SUB_CHUNK * D_K];
                T     b_sub[SUB_CHUNK * D_K];
                float G_sub[SUB_CHUNK * D_K];
            } sub;
        } smem_3;
        union {
            T     pad_4[CHUNK_SIZE * D_K];
            float C_sub[SUB_CHUNK * D_K];
        } smem_4;
    };
    static_assert(sizeof(SmemStorageK3) == 32768, "SmemStorageK3 must be exactly 32 KB (32768 bytes)");
    __shared__ SmemStorageK3 smem;

    using SwizzleHalf = Swizzle<3, 3, 3>;
    using SwizzleF32  = Swizzle<3, 2, 3>;

    auto layout_64x64_km  = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_64x64_mnm = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<CHUNK_SIZE>{})));
    auto layout_sub_km    = composition(SwizzleHalf{}, make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_sub_mnm   = composition(SwizzleHalf{}, make_layout(make_shape(Int<D_K>{}, Int<SUB_CHUNK>{}),  make_stride(Int<1>{}, Int<D_K>{})));
    auto layout_sub_f32   = composition(SwizzleF32{},  make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    auto s1_R      = make_tensor(make_smem_ptr(smem.smem_1),           layout_64x64_km);
    auto s2_dAqk   = make_tensor(make_smem_ptr(smem.smem_2),           layout_64x64_km);
    auto s2_dAqk_T = make_tensor(make_smem_ptr(smem.smem_2),           layout_64x64_mnm);
    auto s3_dO     = make_tensor(make_smem_ptr(smem.smem_3.dO_buf),    layout_64x64_km);
    auto sSubA     = make_tensor(make_smem_ptr(smem.smem_3.sub.a_sub), layout_sub_km);
    auto sSubA_mnm = make_tensor(make_smem_ptr(smem.smem_3.sub.a_sub), layout_sub_mnm);
    auto sSubB     = make_tensor(make_smem_ptr(smem.smem_3.sub.b_sub), layout_sub_km);
    auto sSubG     = make_tensor(make_smem_ptr(smem.smem_3.sub.G_sub), layout_sub_f32);
    auto s4_C_sub  = make_tensor(make_smem_ptr(smem.smem_4.C_sub),     layout_sub_f32);

    TiledCopyHalf  tiled_copy_h;
    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_h   = tiled_copy_h.get_slice(tid);
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);

    auto tXs1_d    = thr_copy_h.partition_D(s1_R);
    auto tXs3_dO_d = thr_copy_h.partition_D(s3_dO);
    auto tXsSubA_d = thr_copy_h.partition_D(sSubA);
    auto tXsSubB_d = thr_copy_h.partition_D(sSubB);
    auto tXsSubG_d = thr_copy_f32.partition_D(sSubG);
    auto tXs4Sub_s = thr_copy_f32.partition_S(s4_C_sub);

    auto tCrBuf0 = make_fragment_like(tXs1_d);
    auto tCrSubA = make_fragment_like(tXsSubA_d);
    auto tCrSubB = make_fragment_like(tXsSubB_d);

    auto cTile64     = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}));
    auto tXcHalf64   = thr_copy_h.partition_S(cTile64);
    auto tPredHalf64 = make_fragment_like<bool>(tCrBuf0);

    if (valid_len < CHUNK_SIZE) {
        #pragma unroll
        for (int m = 0; m < size<1>(tPredHalf64); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredHalf64); ++n) {
                bool in_bounds = (get<0>(tXcHalf64(Int<0>{}, m, n)) < valid_len);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredHalf64); ++v) {
                    tPredHalf64(v, m, n) = in_bounds;
                }
            }
        }
    }

    auto cSub16       = make_identity_tensor(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}));
    auto tXcHalfSub   = thr_copy_h.partition_S(cSub16);
    auto tXcF32Sub    = thr_copy_f32.partition_S(cSub16);
    auto tCrGSub      = make_tensor<float>(shape(tXcF32Sub));
    auto tCrGSubAlt   = make_tensor<float>(shape(tXcF32Sub));
    auto tPredHalfSub = make_fragment_like<bool>(tCrSubA);
    auto tPredF32Sub  = make_fragment_like<bool>(tCrGSub);

    if (last_sub_valid < SUB_CHUNK) {
        #pragma unroll
        for (int m = 0; m < size<1>(tPredHalfSub); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredHalfSub); ++n) {
                bool in_bounds = (get<0>(tXcHalfSub(Int<0>{}, m, n)) < last_sub_valid);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredHalfSub); ++v) {
                    tPredHalfSub(v, m, n) = in_bounds;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tPredF32Sub); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredF32Sub); ++n) {
                bool in_bounds = (get<0>(tXcF32Sub(Int<0>{}, m, n)) < last_sub_valid);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredF32Sub); ++v) {
                    tPredF32Sub(v, m, n) = in_bounds;
                }
            }
        }
    }

    auto layout_64x64_gmem = make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
    auto layout_sub_gmem   = make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));

    int64_t kv_tok_base = (static_cast<int64_t>(h_kv) * total_L + chunk_start) * D_K;
    int64_t kv_mat_off  = (static_cast<int64_t>(h_kv) * num_chunks + curr_chunk) * (CHUNK_SIZE * D_V);

    // 2. Prologue: load stationary R_c into smem_1
    {
        auto gR_in = make_tensor(make_gmem_ptr(gR_list + kv_mat_off), layout_64x64_gmem);
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gR_in), tCrBuf0);
        #pragma unroll
        for (int m = 0; m < size<1>(tXs1_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs1_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
        }
    }
    __syncthreads();

    // 3. Sequential GQA loop
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t h_q        = static_cast<int64_t>(h_kv) * num_groups + g;
        int64_t q_tok_base = (h_q * total_L + chunk_start) * D_K;

        // 3.1 GEMM 0: dAqk_g = tril(dO_g @ R_c^T, 0) into smem_2
        {
            auto gDO_in = make_tensor(make_gmem_ptr(gDO_raw + q_tok_base), layout_64x64_gmem);
            if (valid_len == CHUNK_SIZE) {
                cute::copy(tiled_copy_h, thr_copy_h.partition_S(gDO_in), tCrBuf0);
            } else {
                clear(tCrBuf0);
                cute::copy_if(tiled_copy_h, tPredHalf64, thr_copy_h.partition_S(gDO_in), tCrBuf0);
            }
            #pragma unroll
            for (int m = 0; m < size<1>(tXs3_dO_d); ++m) {
                *reinterpret_cast<uint4*>(&tXs3_dO_d(Int<0>{}, m, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
            }
        }
        __syncthreads();

        {
            TiledMMA221 tiled_mma_221;
            auto thr_mma_221   = tiled_mma_221.get_slice(tid);
            auto s2r_copy_a221 = make_tiled_copy_A(S2RCopyAtomA_KM{}, tiled_mma_221);
            auto s2r_copy_b221 = make_tiled_copy_B(S2RCopyAtomB_KM{}, tiled_mma_221);
            auto s2r_thr_a221  = s2r_copy_a221.get_slice(tid);
            auto s2r_thr_b221  = s2r_copy_b221.get_slice(tid);

            auto tCsA_dO = s2r_thr_a221.partition_S(s3_dO);
            auto tCsB_R  = s2r_thr_b221.partition_S(s1_R);
            auto tCsC_2  = thr_mma_221.partition_C(s2_dAqk);
            auto tCcC_2  = thr_mma_221.partition_C(make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{})));

            auto tCrA_221   = thr_mma_221.partition_fragment_A(s3_dO);
            auto tCrB_221   = thr_mma_221.partition_fragment_B(s1_R);
            auto tCrA_vw    = s2r_thr_a221.retile_D(tCrA_221);
            auto tCrB_vw    = s2r_thr_b221.retile_D(tCrB_221);
            auto tCrC_221   = thr_mma_221.make_fragment_C(tCsC_2);
            auto rC_half_2  = make_tensor_like<T>(tCrC_221);

            clear(tCrC_221);
            cute::copy(s2r_copy_a221, tCsA_dO, tCrA_vw);
            cute::copy(s2r_copy_b221, tCsB_R,  tCrB_vw);
            cute::gemm(MMAAtom{}, tCrA_221, tCrB_221, tCrC_221);

            #pragma unroll
            for (int idx = 0; idx < size(tCrC_221); ++idx) {
                auto coord = tCcC_2(idx);
                int r = get<0>(coord);
                int s = get<1>(coord);
                float val = (s <= r && r < valid_len) ? tCrC_221(idx) : 0.0f;
                rC_half_2(idx) = float_to_out<T>(val);
            }
            cute::copy(rC_half_2, tCsC_2);
        }
        __syncthreads();

        // 3.2 Phase A: Sub-chunk dQ_intra (accumulate dQ and +q * dQ_intra)
        for (int i = 0; i <= last_sub; ++i) {
            int     i_start   = i * SUB_CHUNK;
            int     i_valid   = (i == last_sub) ? last_sub_valid : SUB_CHUNK;
            bool    is_full_i = (i_valid == SUB_CHUNK);
            int64_t q_i_off   = q_tok_base  + static_cast<int64_t>(i_start) * D_K;
            int64_t kv_i_off  = kv_tok_base + static_cast<int64_t>(i_start) * D_K;

            auto gKi = make_tensor(make_gmem_ptr(gK_h      + kv_i_off), layout_sub_gmem);
            auto gGi = make_tensor(make_gmem_ptr(gG_log2_h + kv_i_off), layout_sub_gmem);

            if (is_full_i) {
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKi),   tCrSubB);
                cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGi), tCrGSub);
            } else {
                clear(tCrSubB);
                clear(tCrGSub);
                cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gKi),   tCrSubB);
                cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGi), tCrGSub);
            }

            float g_r[8];
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                g_r[v] = (row_id < i_valid) ? tCrGSub(v, Int<0>{}, Int<0>{}) : 0.0f;
            }
            *reinterpret_cast<uint4*>(&tXsSubB_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrSubB(Int<0>{}, Int<0>{}, Int<0>{}));
            #pragma unroll
            for (int x = 0; x < 2; ++x) {
                *reinterpret_cast<uint4*>(&tXsSubG_d(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrGSub(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{}));
            }

            float dq_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            if (i > 0) {
                const float* p_g_ref = gG_log2_h + kv_i_off + col_base;
                float4 gr0 = *reinterpret_cast<const float4*>(p_g_ref);
                float4 gr1 = *reinterpret_cast<const float4*>(p_g_ref + 4);
                float r_gref[8] = {gr0.x, gr0.y, gr0.z, gr0.w, gr1.x, gr1.y, gr1.z, gr1.w};

                TiledMMA141 tiled_mma_141;
                auto thr_mma_141    = tiled_mma_141.get_slice(tid);
                auto s2r_copy_a141  = make_tiled_copy_A(S2RCopyAtomA_KM{},  tiled_mma_141);
                auto s2r_copy_b141  = make_tiled_copy_B(S2RCopyAtomB_MNM{}, tiled_mma_141);
                auto s2r_thr_a141   = s2r_copy_a141.get_slice(tid);
                auto s2r_thr_b141   = s2r_copy_b141.get_slice(tid);

                auto tCsB_K    = s2r_thr_b141.partition_S(sSubA_mnm);
                auto tCsC_141  = thr_mma_141.partition_C(s4_C_sub);
                auto s_dAqk_00 = local_tile(s2_dAqk, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(Int<0>{}, Int<0>{}));

                auto tCrA_141 = thr_mma_141.partition_fragment_A(s_dAqk_00);
                auto tCrB_141 = thr_mma_141.partition_fragment_B(sSubA_mnm);
                auto tCrA_vw  = s2r_thr_a141.retile_D(tCrA_141);
                auto tCrB_vw  = s2r_thr_b141.retile_D(tCrB_141);
                auto tCrC_141 = thr_mma_141.make_fragment_C(tCsC_141);
                clear(tCrC_141);

                for (int j = 0; j < i; ++j) {
                    int64_t kv_j_off = kv_tok_base + static_cast<int64_t>(j * SUB_CHUNK) * D_K;
                    auto gKj = make_tensor(make_gmem_ptr(gK_h      + kv_j_off), layout_sub_gmem);
                    auto gGj = make_tensor(make_gmem_ptr(gG_log2_h + kv_j_off), layout_sub_gmem);

                    cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKj),   tCrSubA);
                    cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGj), tCrGSub);

                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        float val = half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) *
                                    exp2f(r_gref[v] - tCrGSub(v, Int<0>{}, Int<0>{}));
                        tCrSubA(v, Int<0>{}, Int<0>{}) = float_to_out<T>(val);
                    }
                    *reinterpret_cast<uint4*>(&tXsSubA_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                        *reinterpret_cast<const uint4*>(&tCrSubA(Int<0>{}, Int<0>{}, Int<0>{}));
                    __syncthreads();

                    auto s_dAqk_ij = local_tile(s2_dAqk, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(i, j));
                    auto tCsA_ij   = s2r_thr_a141.partition_S(s_dAqk_ij);
                    cute::copy(s2r_copy_a141, tCsA_ij, tCrA_vw);
                    cute::copy(s2r_copy_b141, tCsB_K,  tCrB_vw);
                    cute::gemm(MMAAtom{}, tCrA_141, tCrB_141, tCrC_141);
                    __syncthreads();
                }

                cute::copy(tCrC_141, tCsC_141);
                __syncthreads();

                cute::copy(tiled_copy_f32, tXs4Sub_s, tCrGSub);
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    dq_acc[v] = (row_id < i_valid)
                        ? (tCrGSub(v, Int<0>{}, Int<0>{}) * exp2f(g_r[v] - r_gref[v]))
                        : 0.0f;
                }
            } else {
                __syncthreads();
            }

            // Diagonal (i, i) contribution in FP32
            for (int s = 0; s < SUB_CHUNK; ++s) {
                if (s <= row_id && row_id < i_valid) {
                    float da = half_to_float(s2_dAqk(i_start + row_id, i_start + s));
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        dq_acc[v] += da *
                                     half_to_float(sSubB(s, col_base + v)) *
                                     exp2f(g_r[v] - sSubG(s, col_base + v));
                    }
                }
            }

            // Accumulate dQ_intra -> gDQ_h and +q * dQ_intra -> gDG_ext
            auto gDQ_sub = make_tensor(make_gmem_ptr(gDQ_h     + q_i_off),  layout_sub_gmem);
            auto gQi_sub = make_tensor(make_gmem_ptr(gQ_h      + q_i_off),  layout_sub_gmem);
            auto gDG_sub = make_tensor(make_gmem_ptr(gDG_ext   + kv_i_off), layout_sub_gmem);

            if (is_full_i) {
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gDQ_sub),   tCrSubA);
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQi_sub),   tCrSubB);
                cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDG_sub), tCrGSub);
            } else {
                clear(tCrSubA);
                clear(tCrSubB);
                clear(tCrGSub);
                cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gDQ_sub),   tCrSubA);
                cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gQi_sub),   tCrSubB);
                cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gDG_sub), tCrGSub);
            }

            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                tCrSubA(v, Int<0>{}, Int<0>{}) = float_to_out<T>(
                    half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) + dq_acc[v]
                );
                tCrGSub(v, Int<0>{}, Int<0>{}) += half_to_float(tCrSubB(v, Int<0>{}, Int<0>{})) * dq_acc[v];
            }

            if (is_full_i) {
                cute::copy(tiled_copy_h,   tCrSubA, thr_copy_h.partition_D(gDQ_sub));
                cute::copy(tiled_copy_f32, tCrGSub, thr_copy_f32.partition_D(gDG_sub));
            } else {
                cute::copy_if(tiled_copy_h,   tPredHalfSub, tCrSubA, thr_copy_h.partition_D(gDQ_sub));
                cute::copy_if(tiled_copy_f32, tPredF32Sub,  tCrGSub, thr_copy_f32.partition_D(gDG_sub));
            }
            __syncthreads();
        }

        // 3.3 Phase B: Sub-chunk dK_intra (accumulate -k * dK_intra_g and cross-group dK)
        for (int j = 0; j <= last_sub; ++j) {
            int     j_start   = j * SUB_CHUNK;
            int     j_valid   = (j == last_sub) ? last_sub_valid : SUB_CHUNK;
            bool    is_full_j = (j_valid == SUB_CHUNK);
            int64_t q_j_off   = q_tok_base  + static_cast<int64_t>(j_start) * D_K;
            int64_t kv_j_off  = kv_tok_base + static_cast<int64_t>(j_start) * D_K;

            auto gQj = make_tensor(make_gmem_ptr(gQ_h      + q_j_off),  layout_sub_gmem);
            auto gGj = make_tensor(make_gmem_ptr(gG_log2_h + kv_j_off), layout_sub_gmem);

            if (is_full_j) {
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQj),   tCrSubB);
                cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGj), tCrGSub);
            } else {
                clear(tCrSubB);
                clear(tCrGSub);
                cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gQj),   tCrSubB);
                cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGj), tCrGSub);
            }

            float g_s[8];
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                g_s[v] = (row_id < j_valid) ? tCrGSub(v, Int<0>{}, Int<0>{}) : 0.0f;
            }
            *reinterpret_cast<uint4*>(&tXsSubB_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrSubB(Int<0>{}, Int<0>{}, Int<0>{}));
            #pragma unroll
            for (int x = 0; x < 2; ++x) {
                *reinterpret_cast<uint4*>(&tXsSubG_d(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrGSub(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{}));
            }
            __syncthreads();

            float dk_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            // Diagonal (j, j) contribution in FP32
            for (int r = 0; r < SUB_CHUNK; ++r) {
                if (r >= row_id && r < j_valid && row_id < j_valid) {
                    float da = half_to_float(s2_dAqk(j_start + r, j_start + row_id));
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        dk_acc[v] += da *
                                     half_to_float(sSubB(r, col_base + v)) *
                                     exp2f(sSubG(r, col_base + v) - g_s[v]);
                    }
                }
            }
            __syncthreads();

            // Off-diagonal (i > j) contributions using per-block anchor g_ref = G[i * 16]
            if (j < last_sub) {
                TiledMMA141 tiled_mma_141;
                auto thr_mma_141   = tiled_mma_141.get_slice(tid);
                auto s2r_copy_a141 = make_tiled_copy_A(S2RCopyAtomA_MNM{}, tiled_mma_141);
                auto s2r_copy_b141 = make_tiled_copy_B(S2RCopyAtomB_MNM{}, tiled_mma_141);
                auto s2r_thr_a141  = s2r_copy_a141.get_slice(tid);
                auto s2r_thr_b141  = s2r_copy_b141.get_slice(tid);

                auto tCsB_Q     = s2r_thr_b141.partition_S(sSubA_mnm);
                auto tCsC_141   = thr_mma_141.partition_C(s4_C_sub);
                auto s_dAqkT_00 = local_tile(s2_dAqk_T, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(Int<0>{}, Int<0>{}));

                auto tCrA_141 = thr_mma_141.partition_fragment_A(s_dAqkT_00);
                auto tCrB_141 = thr_mma_141.partition_fragment_B(sSubA_mnm);
                auto tCrA_vw  = s2r_thr_a141.retile_D(tCrA_141);
                auto tCrB_vw  = s2r_thr_b141.retile_D(tCrB_141);
                auto tCrC_141 = thr_mma_141.make_fragment_C(tCsC_141);

                for (int i = j + 1; i <= last_sub; ++i) {
                    int     i_start   = i * SUB_CHUNK;
                    int     i_valid   = (i == last_sub) ? last_sub_valid : SUB_CHUNK;
                    bool    is_full_i = (i_valid == SUB_CHUNK);
                    int64_t q_i_off   = q_tok_base  + static_cast<int64_t>(i_start) * D_K;
                    int64_t kv_i_off  = kv_tok_base + static_cast<int64_t>(i_start) * D_K;

                    const float* p_g_ref = gG_log2_h + kv_i_off + col_base;
                    float4 gr0 = *reinterpret_cast<const float4*>(p_g_ref);
                    float4 gr1 = *reinterpret_cast<const float4*>(p_g_ref + 4);
                    float r_gref[8] = {gr0.x, gr0.y, gr0.z, gr0.w, gr1.x, gr1.y, gr1.z, gr1.w};

                    auto gQi = make_tensor(make_gmem_ptr(gQ_h      + q_i_off),  layout_sub_gmem);
                    auto gGi = make_tensor(make_gmem_ptr(gG_log2_h + kv_i_off), layout_sub_gmem);

                    if (is_full_i) {
                        cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQi),   tCrSubA);
                        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGi), tCrGSub);
                    } else {
                        clear(tCrSubA);
                        clear(tCrGSub);
                        cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gQi),   tCrSubA);
                        cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGi), tCrGSub);
                    }

                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        float val = (row_id < i_valid)
                            ? (half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) *
                               exp2f(tCrGSub(v, Int<0>{}, Int<0>{}) - r_gref[v]))
                            : 0.0f;
                        tCrSubA(v, Int<0>{}, Int<0>{}) = float_to_out<T>(val);
                    }
                    *reinterpret_cast<uint4*>(&tXsSubA_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                        *reinterpret_cast<const uint4*>(&tCrSubA(Int<0>{}, Int<0>{}, Int<0>{}));
                    __syncthreads();

                    auto s_dAqkT_ji = local_tile(s2_dAqk_T, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(j, i));
                    auto tCsA_ji    = s2r_thr_a141.partition_S(s_dAqkT_ji);

                    clear(tCrC_141);
                    cute::copy(s2r_copy_a141, tCsA_ji, tCrA_vw);
                    cute::copy(s2r_copy_b141, tCsB_Q,  tCrB_vw);
                    cute::gemm(MMAAtom{}, tCrA_141, tCrB_141, tCrC_141);

                    cute::copy(tCrC_141, tCsC_141);
                    __syncthreads();

                    cute::copy(tiled_copy_f32, tXs4Sub_s, tCrGSub);
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        if (row_id < j_valid) {
                            dk_acc[v] += tCrGSub(v, Int<0>{}, Int<0>{}) * exp2f(r_gref[v] - g_s[v]);
                        }
                    }
                    __syncthreads();
                }
            }

            // First subtract current group -k * dK_intra_g from gDG_ext, then accumulate dK across GQA
            auto gKj_sub = make_tensor(make_gmem_ptr(gK_h    + kv_j_off), layout_sub_gmem);
            auto gDG_sub = make_tensor(make_gmem_ptr(gDG_ext + kv_j_off), layout_sub_gmem);
            auto gDK_sub = make_tensor(make_gmem_ptr(gDK_ext + kv_j_off), layout_sub_gmem);

            if (is_full_j) {
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKj_sub),   tCrSubA);
                cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDG_sub), tCrGSub);
            } else {
                clear(tCrSubA);
                clear(tCrGSub);
                cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gKj_sub),   tCrSubA);
                cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gDG_sub), tCrGSub);
            }

            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                tCrGSub(v, Int<0>{}, Int<0>{}) -= half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) * dk_acc[v];
            }
            if (is_full_j) {
                cute::copy(tiled_copy_f32, tCrGSub, thr_copy_f32.partition_D(gDG_sub));
            } else {
                cute::copy_if(tiled_copy_f32, tPredF32Sub, tCrGSub, thr_copy_f32.partition_D(gDG_sub));
            }

            if (num_groups == 1) {
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    tCrSubA(v, Int<0>{}, Int<0>{}) = float_to_out<T>(dk_acc[v]);
                }
                if (is_full_j) {
                    cute::copy(tiled_copy_h, tCrSubA, thr_copy_h.partition_D(gDK_sub));
                } else {
                    cute::copy_if(tiled_copy_h, tPredHalfSub, tCrSubA, thr_copy_h.partition_D(gDK_sub));
                }
            } else {
                auto gDK32_sub = make_tensor(make_gmem_ptr(gDK_fp32 + kv_j_off), layout_sub_gmem);
                if (g == 0) {
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        tCrGSub(v, Int<0>{}, Int<0>{}) = dk_acc[v];
                    }
                    if (is_full_j) {
                        cute::copy(tiled_copy_f32, tCrGSub, thr_copy_f32.partition_D(gDK32_sub));
                    } else {
                        cute::copy_if(tiled_copy_f32, tPredF32Sub, tCrGSub, thr_copy_f32.partition_D(gDK32_sub));
                    }
                } else {
                    if (is_full_j) {
                        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDK32_sub), tCrGSubAlt);
                    } else {
                        clear(tCrGSubAlt);
                        cute::copy_if(tiled_copy_f32, tPredF32Sub, thr_copy_f32.partition_S(gDK32_sub), tCrGSubAlt);
                    }
                    if (g + 1 < num_groups) {
                        #pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            tCrGSubAlt(v, Int<0>{}, Int<0>{}) += dk_acc[v];
                        }
                        if (is_full_j) {
                            cute::copy(tiled_copy_f32, tCrGSubAlt, thr_copy_f32.partition_D(gDK32_sub));
                        } else {
                            cute::copy_if(tiled_copy_f32, tPredF32Sub, tCrGSubAlt, thr_copy_f32.partition_D(gDK32_sub));
                        }
                    } else {
                        #pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            tCrSubA(v, Int<0>{}, Int<0>{}) = float_to_out<T>(
                                tCrGSubAlt(v, Int<0>{}, Int<0>{}) + dk_acc[v]
                            );
                        }
                        if (is_full_j) {
                            cute::copy(tiled_copy_h, tCrSubA, thr_copy_h.partition_D(gDK_sub));
                        } else {
                            cute::copy_if(tiled_copy_h, tPredHalfSub, tCrSubA, thr_copy_h.partition_D(gDK_sub));
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Part 2: Host Dispatch Functions
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
output_combine_bwd_with_meta(
    const at::Tensor& grad_out_tok,
    const at::Tensor& O_raw,
    const at::Tensor& o_rstd,
    const at::Tensor& proj,
    const at::Tensor& o_norm_weight,
    const at::Tensor& q_h,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& S_list,
    const at::Tensor& R_list,
    const VarlenChunkMeta& meta,
    at::Tensor& d_proj_inout,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t chunk_size
) {
    TORCH_CHECK(d_k == 64 && d_v == 64, "d_k and d_v must be 64");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(num_kv_heads > 0 && num_q_heads > 0 && (num_q_heads % num_kv_heads == 0),
                "num_q_heads must be positive and divisible by num_kv_heads");

    TORCH_CHECK(grad_out_tok.is_cuda(), "grad_out_tok must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(grad_out_tok.device());

    auto check_t = [&](const at::Tensor& t, const char* name, int64_t expected_dim, at::ScalarType dtype) {
        TORCH_CHECK(t.is_cuda() && t.device() == grad_out_tok.device(), name, " must be on the same CUDA device");
        TORCH_CHECK(t.scalar_type() == dtype, name, " dtype mismatch");
        TORCH_CHECK(t.dim() == expected_dim && t.is_contiguous(), name, " must be contiguous with expected rank");
    };

    check_t(grad_out_tok,  "grad_out_tok",  2, at::ScalarType::Half);
    check_t(O_raw,         "O_raw",         3, at::ScalarType::Half);
    check_t(o_rstd,        "o_rstd",        2, at::ScalarType::Float);
    check_t(proj,          "proj",          2, at::ScalarType::Half);
    check_t(o_norm_weight, "o_norm_weight", 1, at::ScalarType::Half);
    check_t(q_h,           "q_h",           3, at::ScalarType::Half);
    check_t(k_h,           "k_h",           3, at::ScalarType::Half);
    check_t(G_log2_h,      "G_log2_h",      3, at::ScalarType::Float);
    check_t(S_list,        "S_list",        4, at::ScalarType::Float);
    check_t(R_list,        "R_list",        4, at::ScalarType::Half);
    check_t(d_proj_inout,  "d_proj_inout",  2, at::ScalarType::Half);

    int64_t total_L        = q_h.size(1);
    int64_t num_chunks     = meta.num_chunks;
    int64_t num_groups     = num_q_heads / num_kv_heads;
    int64_t offset_ogate   = num_q_heads * d_k + 3 * num_kv_heads * d_k + 2 * num_kv_heads * d_v;
    int64_t total_proj_dim = offset_ogate + num_q_heads * d_v;

    TORCH_CHECK(grad_out_tok.size(0) == total_L && grad_out_tok.size(1) == num_q_heads * d_v,
                "grad_out_tok shape mismatch");
    TORCH_CHECK(O_raw.size(0) == num_q_heads && O_raw.size(1) == total_L && O_raw.size(2) == d_v,
                "O_raw shape mismatch");
    TORCH_CHECK(o_rstd.size(0) == num_q_heads && o_rstd.size(1) == total_L, "o_rstd shape mismatch");
    TORCH_CHECK(proj.size(0) == total_L && proj.size(1) == total_proj_dim, "proj shape mismatch");
    TORCH_CHECK(d_proj_inout.size(0) == total_L && d_proj_inout.size(1) == total_proj_dim,
                "d_proj_inout shape mismatch");
    TORCH_CHECK(o_norm_weight.size(0) == d_v, "o_norm_weight shape mismatch");
    TORCH_CHECK(q_h.size(0) == num_q_heads && q_h.size(2) == d_k, "q_h shape mismatch");
    TORCH_CHECK(k_h.size(0) == num_kv_heads && k_h.size(1) == total_L && k_h.size(2) == d_k, "k_h shape mismatch");
    TORCH_CHECK(G_log2_h.size(0) == num_kv_heads && G_log2_h.size(1) == total_L && G_log2_h.size(2) == d_k,
                "G_log2_h shape mismatch");
    TORCH_CHECK(S_list.size(0) == num_kv_heads && S_list.size(1) == num_chunks &&
                S_list.size(2) == d_v && S_list.size(3) == d_k, "S_list shape mismatch");
    TORCH_CHECK(R_list.size(0) == num_kv_heads && R_list.size(1) == num_chunks &&
                R_list.size(2) == chunk_size && R_list.size(3) == d_v, "R_list shape mismatch");

    auto h_opts = grad_out_tok.options();
    auto f_opts = G_log2_h.options();

    if (num_chunks <= 0 || total_L == 0) {
        at::Tensor dq_h            = at::empty({num_q_heads, total_L, d_k}, h_opts);
        at::Tensor dk_ext          = at::empty({num_kv_heads, total_L, d_k}, h_opts);
        at::Tensor dG_ext          = at::empty({num_kv_heads, total_L, d_k}, f_opts);
        at::Tensor dR_out          = at::empty({num_kv_heads, num_chunks, chunk_size, d_v}, h_opts);
        at::Tensor dS_out          = at::empty({num_kv_heads, num_chunks, d_v, d_k}, f_opts);
        at::Tensor d_o_norm_weight = at::zeros({d_v}, h_opts);
        return std::make_tuple(dq_h, dk_ext, dG_ext, dR_out, dS_out, d_o_norm_weight, d_proj_inout);
    }

    TORCH_CHECK(meta.chunk_offsets.is_cuda() && meta.chunk_offsets.device() == grad_out_tok.device() &&
                meta.chunk_offsets.scalar_type() == at::ScalarType::Int &&
                meta.chunk_offsets.dim() == 1 && meta.chunk_offsets.is_contiguous() &&
                meta.chunk_offsets.size(0) >= num_chunks + 1, "meta.chunk_offsets mismatch");
    TORCH_CHECK(meta.chunk_valid_lens.is_cuda() && meta.chunk_valid_lens.device() == grad_out_tok.device() &&
                meta.chunk_valid_lens.scalar_type() == at::ScalarType::Int &&
                meta.chunk_valid_lens.dim() == 1 && meta.chunk_valid_lens.is_contiguous() &&
                meta.chunk_valid_lens.size(0) >= num_chunks, "meta.chunk_valid_lens mismatch");

    at::Tensor dO_raw          = torch::empty({num_q_heads, total_L, d_v}, h_opts);
    at::Tensor dq_h            = torch::empty({num_q_heads, total_L, d_k}, h_opts);
    at::Tensor dk_ext          = torch::zeros({num_kv_heads, total_L, d_k}, h_opts);
    at::Tensor dG_ext          = torch::zeros({num_kv_heads, total_L, d_k}, f_opts);
    at::Tensor dR_out          = torch::empty({num_kv_heads, num_chunks, chunk_size, d_v}, h_opts);
    at::Tensor dS_out          = torch::empty({num_kv_heads, num_chunks, d_v, d_k}, f_opts);
    at::Tensor d_o_norm_w_fp32 = torch::zeros({d_v}, f_opts);
    at::Tensor dk_fp32         = (num_groups > 1)
        ? torch::empty({num_kv_heads, total_L, d_k}, f_opts)
        : at::Tensor();

    at::Tensor go_a   = ensure_aligned16(grad_out_tok);
    at::Tensor oraw_a = ensure_aligned16(O_raw);
    at::Tensor rstd_a = ensure_aligned16(o_rstd);
    at::Tensor proj_a = ensure_aligned16(proj);
    at::Tensor ow_a   = ensure_aligned16(o_norm_weight);
    at::Tensor q_a    = ensure_aligned16(q_h);
    at::Tensor k_a    = ensure_aligned16(k_h);
    at::Tensor G_a    = ensure_aligned16(G_log2_h);
    at::Tensor S_a    = ensure_aligned16(S_list);
    at::Tensor R_a    = ensure_aligned16(R_list);

    bool dp_aligned = (d_proj_inout.numel() == 0) ||
                      ((reinterpret_cast<uintptr_t>(d_proj_inout.data_ptr()) & 15) == 0);
    at::Tensor d_proj_work = dp_aligned ? d_proj_inout : d_proj_inout.clone();

    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_CONST = 64;
    constexpr int SUB_CONST   = 16;
    constexpr int DK_CONST    = 64;
    constexpr int DV_CONST    = 64;

    using Element     = cutlass::half_t;
    using MMAAtom     = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMA16  = TiledMMA<MMAAtom, Layout<Shape<_1, _2, _1>, Stride<_1, _1, _2>>>;
    using TiledMMA141 = TiledMMA<MMAAtom, Layout<Shape<_1, _4, _1>, Stride<_1, _1, _4>>>;
    using TiledMMA221 = TiledMMA<MMAAtom, Layout<Shape<_2, _2, _1>, Stride<_1, _2, _4>>>;

    using S2RCopyAtomA_KM  = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using S2RCopyAtomB_KM  = Copy_Atom<SM75_U32x1_LDSM_N, Element>;
    using S2RCopyAtomA_MNM = Copy_Atom<SM75_U16x4_LDSM_T, Element>;
    using S2RCopyAtomB_MNM = Copy_Atom<SM75_U16x2_LDSM_T, Element>;

    using TiledCopyHalf = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, Element>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));
    using TiledCopyFloat = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, float>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));

    const Element* p_go     = reinterpret_cast<const Element*>(go_a.data_ptr<at::Half>());
    const Element* p_oraw   = reinterpret_cast<const Element*>(oraw_a.data_ptr<at::Half>());
    const float*   p_rstd   = rstd_a.data_ptr<float>();
    const Element* p_proj   = reinterpret_cast<const Element*>(proj_a.data_ptr<at::Half>());
    const Element* p_ow     = reinterpret_cast<const Element*>(ow_a.data_ptr<at::Half>());
    const Element* p_q      = reinterpret_cast<const Element*>(q_a.data_ptr<at::Half>());
    const Element* p_k      = reinterpret_cast<const Element*>(k_a.data_ptr<at::Half>());
    const float*   p_G      = G_a.data_ptr<float>();
    const float*   p_S      = S_a.data_ptr<float>();
    const Element* p_R      = reinterpret_cast<const Element*>(R_a.data_ptr<at::Half>());
    const int32_t* p_off    = meta.chunk_offsets.data_ptr<int32_t>();
    const int32_t* p_vlen   = meta.chunk_valid_lens.data_ptr<int32_t>();

    Element* p_dO_raw  = reinterpret_cast<Element*>(dO_raw.data_ptr<at::Half>());
    Element* p_dproj   = reinterpret_cast<Element*>(d_proj_work.data_ptr<at::Half>());
    float*   p_dow_f32 = d_o_norm_w_fp32.data_ptr<float>();
    Element* p_dq      = reinterpret_cast<Element*>(dq_h.data_ptr<at::Half>());
    Element* p_dk      = reinterpret_cast<Element*>(dk_ext.data_ptr<at::Half>());
    float*   p_dk_f32  = (num_groups > 1) ? dk_fp32.data_ptr<float>() : nullptr;
    float*   p_dG      = dG_ext.data_ptr<float>();
    float*   p_dS      = dS_out.data_ptr<float>();
    Element* p_dR      = reinterpret_cast<Element*>(dR_out.data_ptr<at::Half>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // 1. Kernel 1: Per-Head RMSNorm * SiLU(o_gate) + transpose backward
    {
        dim3 grid(static_cast<unsigned int>(num_chunks), static_cast<unsigned int>(num_q_heads));
        dim3 block(NUM_THREADS);
        kernel_out_norm_swiglu_transpose_bwd<
            Element, NUM_THREADS, CHUNK_CONST, DV_CONST, TiledCopyHalf
        ><<<grid, block, 0, stream>>>(
            p_go, p_oraw, p_rstd, p_proj, p_ow, p_off, p_vlen,
            p_dO_raw, p_dproj, p_dow_f32,
            num_q_heads, num_chunks, total_L, total_proj_dim, offset_ogate
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 2. Kernel 2: Inter-chunk gradients (dQ_inter, dG_inter, dS_out) & dR_out
    {
        dim3 grid(static_cast<unsigned int>(num_chunks), static_cast<unsigned int>(num_kv_heads));
        dim3 block(NUM_THREADS);
        kernel_output_combine_dR_dS_dQinter_bwd<
            Element, NUM_THREADS, CHUNK_CONST, SUB_CONST, DK_CONST, DV_CONST,
            MMAAtom, TiledMMA16, TiledMMA221,
            S2RCopyAtomA_KM, S2RCopyAtomB_KM, S2RCopyAtomA_MNM, S2RCopyAtomB_MNM,
            TiledCopyHalf, TiledCopyFloat
        ><<<grid, block, 0, stream>>>(
            p_dO_raw, p_q, p_k, p_G, p_S, p_off, p_vlen,
            p_dq, p_dG, p_dS, p_dR,
            num_q_heads, num_kv_heads, num_groups, num_chunks, total_L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 3. Kernel 3: Intra-chunk gradients (dQ_intra, dK_intra, dG_intra)
    {
        dim3 grid(static_cast<unsigned int>(num_chunks), static_cast<unsigned int>(num_kv_heads));
        dim3 block(NUM_THREADS);
        kernel_output_combine_intra_dqk_bwd<
            Element, NUM_THREADS, CHUNK_CONST, SUB_CONST, DK_CONST, DV_CONST,
            MMAAtom, TiledMMA141, TiledMMA221,
            S2RCopyAtomA_KM, S2RCopyAtomB_KM, S2RCopyAtomA_MNM, S2RCopyAtomB_MNM,
            TiledCopyHalf, TiledCopyFloat
        ><<<grid, block, 0, stream>>>(
            p_dO_raw, p_R, p_q, p_k, p_G, p_off, p_vlen,
            p_dq, p_dk, p_dk_f32, p_dG,
            num_q_heads, num_kv_heads, num_groups, num_chunks, total_L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (!dp_aligned) {
        int64_t ogate_cols = num_q_heads * d_v;
        d_proj_inout.narrow(1, offset_ogate, ogate_cols).copy_(
            d_proj_work.narrow(1, offset_ogate, ogate_cols)
        );
    }

    dO_raw.reset();
    if (dk_fp32.defined()) {
        dk_fp32.reset();
    }
    at::Tensor d_o_norm_weight = d_o_norm_w_fp32.to(at::kHalf);
    d_o_norm_w_fp32.reset();

    return std::make_tuple(dq_h, dk_ext, dG_ext, dR_out, dS_out, d_o_norm_weight, d_proj_inout);
}

// Outputs: (dq_h, dk_ext, dG_ext [FP32], dR_out, dS_out [FP32], d_o_norm_weight, d_proj_inout)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
output_combine_bwd(
    const at::Tensor& grad_out_tok,
    const at::Tensor& O_raw,
    const at::Tensor& o_rstd,
    const at::Tensor& proj,
    const at::Tensor& o_norm_weight,
    const at::Tensor& q_h,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& S_list,
    const at::Tensor& R_list,
    const at::Tensor& cu_seq_lens,
    at::Tensor& d_proj_inout,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t chunk_size
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 1,
                "cu_seq_lens must be a 1D CUDA tensor with size >= 1");
    VarlenChunkMeta meta = compute_varlen_chunk_metadata(cu_seq_lens, chunk_size);
    return output_combine_bwd_with_meta(
        grad_out_tok, O_raw, o_rstd, proj, o_norm_weight, q_h, k_h, G_log2_h,
        S_list, R_list, meta, d_proj_inout, num_q_heads, num_kv_heads, d_k, d_v, chunk_size
    );
}
