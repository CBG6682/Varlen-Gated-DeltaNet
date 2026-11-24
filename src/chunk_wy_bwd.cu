#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm75.hpp>
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
// Part 1: Global Kernels
// ============================================================================

// Kernel 1: State tail, dY = -dU @ S, chained gate-aware dA, triangular inverse dT, dZ -> (dv, dw), dE_bar -> (db, dk, dG)
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_K,
    int D_V,
    typename MMAAtom,
    typename TiledMMA221,
    typename S2RCopyAtomA_KM,
    typename S2RCopyAtomB_KM,
    typename S2RCopyAtomA_MNM,
    typename S2RCopyAtomB_MNM,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_chunk_wy_dA_dT_bwd(
    const T*       __restrict__ gDR_total,        // (H_kv, num_chunks, 64, 64) [FP16] - Total grad dR (= dU)
    const T*       __restrict__ gR_list,          // (H_kv, num_chunks, 64, 64) [FP16] - Saved residual R
    const float*   __restrict__ gS_list,          // (H_kv, num_chunks, 64, 64) [FP32] - Chunk initial states
    const float*   __restrict__ gDS_next_list,    // (H_kv, num_chunks, 64, 64) [FP32] - Grad w.r.t. S_{c+1}
    const T*       __restrict__ gA_list,          // (H_kv, num_chunks, 64, 64) [FP16] - Saved WY inverse A
    const T*       __restrict__ gK_h,             // (H_kv, total_L, 64) [FP16] - Normalized Key
    const T*       __restrict__ gV_h,             // (H_kv, total_L, 64) [FP16] - Activated Value
    const T*       __restrict__ gB_eff_h,         // (H_kv, total_L, 64) [FP16] - Effective erase gate
    const T*       __restrict__ gW_h,             // (H_kv, total_L, 64) [FP16] - Write gate
    const float*   __restrict__ gG_log2_h,        // (H_kv, total_L, 64) [FP32] - Chunk cumsum log2 decay
    const T*       __restrict__ gDK_ext,          // (H_kv, total_L, 64) [FP16] - External Key grad
    const float*   __restrict__ gDG_ext,          // (H_kv, total_L, 64) [FP32] - External log-decay grad
    const int32_t* __restrict__ chunk_offsets,    // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens, // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ gDT,              // (H_kv, num_chunks, 64, 64) [FP16] - Output grad dT
    T*             __restrict__ gDK_h,            // (H_kv, total_L, 64) [FP16] - Accumulated Key grad
    T*             __restrict__ gDV_h,            // (H_kv, total_L, 64) [FP16] - Output Value grad
    T*             __restrict__ gDB_h,            // (H_kv, total_L, 64) [FP16] - Accumulated Beta grad
    T*             __restrict__ gDW_h,            // (H_kv, total_L, 64) [FP16] - Output Write gate grad
    float*         __restrict__ gDG_h,            // (H_kv, total_L, 64) [FP32] - Accumulated log-decay grad
    int64_t num_kv_heads,                         // scalar [int64] - Total KV heads
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

    struct alignas(16) SmemStorageK1 {
        union {
            struct {
                T smem_1[CHUNK_SIZE * D_K];
                T smem_2[CHUNK_SIZE * D_K];
            } pair;
            float f32[CHUNK_SIZE * D_K];
        } u12;
        T smem_3[CHUNK_SIZE * D_K];
        union {
            T     smem_4[CHUNK_SIZE * D_K];
            float red[16 * D_K];
        } u4;
    };
    static_assert(sizeof(SmemStorageK1) == 32768, "SmemStorageK1 must be exactly 32 KB (32768 bytes)");
    __shared__ SmemStorageK1 smem;

    using SwizzleHalf = Swizzle<3, 3, 3>;
    using SwizzleF32  = Swizzle<3, 2, 3>;

    auto layout_km  = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_mnm = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<1>{}, Int<CHUNK_SIZE>{})));
    auto layout_f32 = composition(SwizzleF32{},  make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    auto s1_km    = make_tensor(make_smem_ptr(smem.u12.pair.smem_1), layout_km);
    auto s1_mnm   = make_tensor(make_smem_ptr(smem.u12.pair.smem_1), layout_mnm);
    auto s2_km    = make_tensor(make_smem_ptr(smem.u12.pair.smem_2), layout_km);
    auto s2_mnm   = make_tensor(make_smem_ptr(smem.u12.pair.smem_2), layout_mnm);
    auto sF32_12  = make_tensor(make_smem_ptr(smem.u12.f32),         layout_f32);
    auto s3_km    = make_tensor(make_smem_ptr(smem.smem_3),          layout_km);
    auto s3_mnm   = make_tensor(make_smem_ptr(smem.smem_3),          layout_mnm);
    auto s4_km    = make_tensor(make_smem_ptr(smem.u4.smem_4),       layout_km);
    auto s4_mnm   = make_tensor(make_smem_ptr(smem.u4.smem_4),       layout_mnm);

    TiledCopyHalf  tiled_copy_h;
    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_h   = tiled_copy_h.get_slice(tid);
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);

    auto tXs1_s      = thr_copy_h.partition_S(s1_km);
    auto tXs1_d      = thr_copy_h.partition_D(s1_km);
    auto tXs2_s      = thr_copy_h.partition_S(s2_km);
    auto tXs2_d      = thr_copy_h.partition_D(s2_km);
    auto tXs3_d      = thr_copy_h.partition_D(s3_km);
    auto tXs4_d      = thr_copy_h.partition_D(s4_km);
    auto tXsF32_12_s = thr_copy_f32.partition_S(sF32_12);

    auto tCrBuf0 = make_fragment_like(tXs1_d);
    auto tCrBuf1 = make_fragment_like(tXs2_d);
    auto tCrF0   = make_fragment_like(tXsF32_12_s);
    auto tCrF1   = make_fragment_like(tXsF32_12_s);

    auto cTile64     = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}));
    auto tXcHalf64   = thr_copy_h.partition_S(cTile64);
    auto tXcF32_64   = thr_copy_f32.partition_S(cTile64);
    auto tPredHalf64 = make_fragment_like<bool>(tCrBuf0);
    auto tPredF32_64 = make_fragment_like<bool>(tCrF0);

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

    // Shared TiledMMA221 operands and single accumulator across all 64x64 GEMMs
    TiledMMA221 tiled_mma;
    auto thr_mma     = tiled_mma.get_slice(tid);
    auto s2r_copy_ak = make_tiled_copy_A(S2RCopyAtomA_KM{},  tiled_mma);
    auto s2r_copy_bk = make_tiled_copy_B(S2RCopyAtomB_KM{},  tiled_mma);
    auto s2r_copy_am = make_tiled_copy_A(S2RCopyAtomA_MNM{}, tiled_mma);
    auto s2r_copy_bm = make_tiled_copy_B(S2RCopyAtomB_MNM{}, tiled_mma);

    auto s2r_thr_ak = s2r_copy_ak.get_slice(tid);
    auto s2r_thr_bk = s2r_copy_bk.get_slice(tid);
    auto s2r_thr_am = s2r_copy_am.get_slice(tid);
    auto s2r_thr_bm = s2r_copy_bm.get_slice(tid);

    auto tCsA_s1_km = s2r_thr_ak.partition_S(s1_km);
    auto tCsA_s2_km = s2r_thr_ak.partition_S(s2_km);
    auto tCsA_s3_km = s2r_thr_ak.partition_S(s3_km);
    auto tCsA_s4_mn = s2r_thr_am.partition_S(s4_mnm);

    auto tCsB_s1_mn = s2r_thr_bm.partition_S(s1_mnm);
    auto tCsB_s2_km = s2r_thr_bk.partition_S(s2_km);
    auto tCsB_s2_mn = s2r_thr_bm.partition_S(s2_mnm);
    auto tCsB_s3_mn = s2r_thr_bm.partition_S(s3_mnm);
    auto tCsB_s4_km = s2r_thr_bk.partition_S(s4_km);

    auto tCsC_s1     = thr_mma.partition_C(s1_km);
    auto tCsC_s2     = thr_mma.partition_C(s2_km);
    auto tCsC_s3     = thr_mma.partition_C(s3_km);
    auto tCsC_f32_12 = thr_mma.partition_C(sF32_12);
    auto tCcC        = thr_mma.partition_C(make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{})));

    auto tCrA       = thr_mma.partition_fragment_A(s1_km);
    auto tCrB       = thr_mma.partition_fragment_B(s2_km);
    auto tCrA_km_vw = s2r_thr_ak.retile_D(tCrA);
    auto tCrA_mn_vw = s2r_thr_am.retile_D(tCrA);
    auto tCrB_km_vw = s2r_thr_bk.retile_D(tCrB);
    auto tCrB_mn_vw = s2r_thr_bm.retile_D(tCrB);
    auto tCrC       = thr_mma.make_fragment_C(tCsC_s1);
    auto rC_half    = make_tensor_like<T>(tCrC);

    auto layout_64x64_gmem = make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
    int64_t kv_mat_off     = (static_cast<int64_t>(h_kv) * num_chunks + curr_chunk) * (CHUNK_SIZE * D_K);
    int64_t kv_tok_base    = (static_cast<int64_t>(h_kv) * total_L + chunk_start) * D_K;

    auto gDR_in      = make_tensor(make_gmem_ptr(gDR_total     + kv_mat_off),  layout_64x64_gmem);
    auto gR_in       = make_tensor(make_gmem_ptr(gR_list       + kv_mat_off),  layout_64x64_gmem);
    auto gS_in       = make_tensor(make_gmem_ptr(gS_list       + kv_mat_off),  layout_64x64_gmem);
    auto gDS_next_in = make_tensor(make_gmem_ptr(gDS_next_list + kv_mat_off),  layout_64x64_gmem);
    auto gA_in       = make_tensor(make_gmem_ptr(gA_list       + kv_mat_off),  layout_64x64_gmem);
    auto gDT_out     = make_tensor(make_gmem_ptr(gDT           + kv_mat_off),  layout_64x64_gmem);

    auto gK_in       = make_tensor(make_gmem_ptr(gK_h          + kv_tok_base), layout_64x64_gmem);
    auto gV_in       = make_tensor(make_gmem_ptr(gV_h          + kv_tok_base), layout_64x64_gmem);
    auto gB_in       = make_tensor(make_gmem_ptr(gB_eff_h      + kv_tok_base), layout_64x64_gmem);
    auto gW_in       = make_tensor(make_gmem_ptr(gW_h          + kv_tok_base), layout_64x64_gmem);
    auto gG_in       = make_tensor(make_gmem_ptr(gG_log2_h     + kv_tok_base), layout_64x64_gmem);
    auto gDK_ext_in  = make_tensor(make_gmem_ptr(gDK_ext       + kv_tok_base), layout_64x64_gmem);
    auto gDG_ext_in  = make_tensor(make_gmem_ptr(gDG_ext       + kv_tok_base), layout_64x64_gmem);

    auto gDK_out     = make_tensor(make_gmem_ptr(gDK_h         + kv_tok_base), layout_64x64_gmem);
    auto gDV_out     = make_tensor(make_gmem_ptr(gDV_h         + kv_tok_base), layout_64x64_gmem);
    auto gDB_out     = make_tensor(make_gmem_ptr(gDB_h         + kv_tok_base), layout_64x64_gmem);
    auto gDW_out     = make_tensor(make_gmem_ptr(gDW_h         + kv_tok_base), layout_64x64_gmem);
    auto gDG_out     = make_tensor(make_gmem_ptr(gDG_h         + kv_tok_base), layout_64x64_gmem);

    // 2. Stage 1: State tail path (dK_tail = R @ dS_next_fp16, dG_last_state, dk_ext/dG_ext init)
    {
        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gS_in),       tCrF0);
        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDS_next_in), tCrF1);

        float ds_dot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float s_val  = tCrF0(v, m, Int<0>{});
                float ds_val = tCrF1(v, m, Int<0>{});
                ds_dot[v] += ds_val * s_val;
                tCrBuf0(v, m, Int<0>{}) = float_to_out<T>(s_val);
                tCrBuf1(v, m, Int<0>{}) = float_to_out<T>(ds_val);
            }
        }
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            smem.u4.red[row_id * D_K + col_base + v] = ds_dot[v];
        }

        // Keep S_fp16 resident in smem_3 through GEMM 2
        #pragma unroll
        for (int m = 0; m < size<1>(tXs3_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs3_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
            *reinterpret_cast<uint4*>(&tXs2_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf1(Int<0>{}, m, Int<0>{}));
        }

        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_h, thr_copy_h.partition_S(gR_in), tCrBuf0);
        } else {
            clear(tCrBuf0);
            cute::copy_if(tiled_copy_h, tPredHalf64, thr_copy_h.partition_S(gR_in), tCrBuf0);
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tXs1_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs1_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
        }
        __syncthreads();

        // GEMM 1: dK_tail = R @ dS_next_fp16
        clear(tCrC);
        cute::copy(s2r_copy_ak, tCsA_s1_km, tCrA_km_vw);
        cute::copy(s2r_copy_bm, tCsB_s2_mn, tCrB_mn_vw);
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);
        __syncthreads();

        cute::copy(tCrC, tCsC_f32_12);
        __syncthreads();

        const float* p_g_last = gG_log2_h + kv_tok_base + static_cast<int64_t>(valid_len - 1) * D_K + col_base;
        float4 gl0 = *reinterpret_cast<const float4*>(p_g_last);
        float4 gl1 = *reinterpret_cast<const float4*>(p_g_last + 4);
        float g_last_reg[8] = {gl0.x, gl0.y, gl0.z, gl0.w, gl1.x, gl1.y, gl1.z, gl1.w};

        cute::copy(tiled_copy_f32, tXsF32_12_s, tCrF0);
        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gG_in),      tCrF1);
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gK_in),        tCrBuf0);
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gDK_ext_in),   tCrBuf1);
        } else {
            clear(tCrF1);
            clear(tCrBuf0);
            clear(tCrBuf1);
            cute::copy_if(tiled_copy_f32, tPredF32_64, thr_copy_f32.partition_S(gG_in),      tCrF1);
            cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gK_in),        tCrBuf0);
            cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gDK_ext_in),   tCrBuf1);
        }

        float tail_g_sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int  local_row = m * 16 + row_id;
            bool row_valid = (local_row < valid_len);
            bool before_last = (local_row < valid_len - 1);
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float dk_tail = row_valid
                    ? (tCrF0(v, m, Int<0>{}) * exp2f(g_last_reg[v] - tCrF1(v, m, Int<0>{})))
                    : 0.0f;
                tCrBuf1(v, m, Int<0>{}) = float_to_out<T>(half_to_float(tCrBuf1(v, m, Int<0>{})) + dk_tail);

                // Derivatives of G_last and G_r cancel at r == valid_len - 1
                float g_term = before_last ? (dk_tail * half_to_float(tCrBuf0(v, m, Int<0>{}))) : 0.0f;
                tCrF0(v, m, Int<0>{}) = -g_term;
                tail_g_sum[v] += g_term;
            }
        }

        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_h, tCrBuf1, thr_copy_h.partition_D(gDK_out));
        } else {
            cute::copy_if(tiled_copy_h, tPredHalf64, tCrBuf1, thr_copy_h.partition_D(gDK_out));
        }

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            smem.u4.red[row_id * D_K + col_base + v] =
                smem.u4.red[row_id * D_K + col_base + v] * exp2f(g_last_reg[v]) + tail_g_sum[v];
        }
        __syncthreads();

        int last_m      = (valid_len - 1) / 16;
        int last_row_id = (valid_len - 1) % 16;
        if (row_id == last_row_id) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float tot = 0.0f;
                #pragma unroll
                for (int r = 0; r < 16; ++r) {
                    tot += smem.u4.red[r * D_K + col_base + v];
                }
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m == last_m) {
                        tCrF0(v, m, Int<0>{}) += tot;
                    }
                }
            }
        }

        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDG_ext_in), tCrF1);
        } else {
            clear(tCrF1);
            cute::copy_if(tiled_copy_f32, tPredF32_64, thr_copy_f32.partition_S(gDG_ext_in), tCrF1);
        }
        #pragma unroll
        for (int i = 0; i < size(tCrF0); ++i) {
            tCrF0(i) += tCrF1(i);
        }
        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_f32, tCrF0, thr_copy_f32.partition_D(gDG_out));
        } else {
            cute::copy_if(tiled_copy_f32, tPredF32_64, tCrF0, thr_copy_f32.partition_D(gDG_out));
        }
        __syncthreads();
    }

    // 3. Stage 2: Load stationary A into smem_4 and compute dY = -dU @ S_fp16 into smem_3
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gDR_in), tCrBuf0);
    } else {
        clear(tCrBuf0);
        cute::copy_if(tiled_copy_h, tPredHalf64, thr_copy_h.partition_S(gDR_in), tCrBuf0);
    }
    cute::copy(tiled_copy_h, thr_copy_h.partition_S(gA_in), tCrBuf1);

    #pragma unroll
    for (int m = 0; m < size<1>(tXs1_d); ++m) {
        *reinterpret_cast<uint4*>(&tXs1_d(Int<0>{}, m, Int<0>{})) =
            *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
        *reinterpret_cast<uint4*>(&tXs4_d(Int<0>{}, m, Int<0>{})) =
            *reinterpret_cast<const uint4*>(&tCrBuf1(Int<0>{}, m, Int<0>{}));
    }
    __syncthreads();

    clear(tCrC);
    cute::copy(s2r_copy_ak, tCsA_s1_km, tCrA_km_vw);
    cute::copy(s2r_copy_bm, tCsB_s3_mn, tCrB_mn_vw);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < size(tCrC); ++i) {
        rC_half(i) = float_to_out<T>(-tCrC(i));
    }
    cute::copy(rC_half, tCsC_s3);
    __syncthreads();

    // 4. Stage 3: Compute dZ = A^T @ dU -> emit gDV_h, gDW_h, then stage Z into smem_2
    clear(tCrC);
    cute::copy(s2r_copy_am, tCsA_s4_mn, tCrA_mn_vw);
    cute::copy(s2r_copy_bm, tCsB_s1_mn, tCrB_mn_vw);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);

    #pragma unroll
    for (int i = 0; i < size(tCrC); ++i) {
        rC_half(i) = float_to_out<T>(tCrC(i));
    }
    cute::copy(rC_half, tCsC_s2);
    __syncthreads();

    {
        cute::copy(tiled_copy_h, tXs2_s, tCrBuf0);
        auto tCrW   = make_fragment_like(tCrBuf0);
        auto tCrOut = make_fragment_like(tCrBuf0);

        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_h, thr_copy_h.partition_S(gV_in), tCrBuf1);
            cute::copy(tiled_copy_h, thr_copy_h.partition_S(gW_in), tCrW);
        } else {
            clear(tCrBuf1);
            clear(tCrW);
            cute::copy_if(tiled_copy_h, tPredHalf64, thr_copy_h.partition_S(gV_in), tCrBuf1);
            cute::copy_if(tiled_copy_h, tPredHalf64, thr_copy_h.partition_S(gW_in), tCrW);
        }

        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int  local_row = m * 16 + row_id;
            bool row_valid = (local_row < valid_len);
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float dz_f = half_to_float(tCrBuf0(v, m, Int<0>{}));
                float v_f  = half_to_float(tCrBuf1(v, m, Int<0>{}));
                float w_f  = half_to_float(tCrW(v, m, Int<0>{}));
                tCrOut(v, m, Int<0>{})  = float_to_out<T>(dz_f * w_f);
                tCrBuf0(v, m, Int<0>{}) = float_to_out<T>(dz_f * v_f);
                tCrW(v, m, Int<0>{})    = float_to_out<T>(row_valid ? (w_f * v_f) : 0.0f);
            }
        }

        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_h, tCrOut,  thr_copy_h.partition_D(gDV_out));
            cute::copy(tiled_copy_h, tCrBuf0, thr_copy_h.partition_D(gDW_out));
        } else {
            cute::copy_if(tiled_copy_h, tPredHalf64, tCrOut,  thr_copy_h.partition_D(gDV_out));
            cute::copy_if(tiled_copy_h, tPredHalf64, tCrBuf0, thr_copy_h.partition_D(gDW_out));
        }

        // Same-thread read-then-overwrite of smem_2 with Z = w * v
        #pragma unroll
        for (int m = 0; m < size<1>(tXs2_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs2_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrW(Int<0>{}, m, Int<0>{}));
        }
    }
    __syncthreads();

    // 5. Stage 4: Chained gate-aware dA = dU @ Z^T + dY @ E_bar^T in single register accumulator
    clear(tCrC);
    cute::copy(s2r_copy_ak, tCsA_s1_km, tCrA_km_vw);
    cute::copy(s2r_copy_bk, tCsB_s2_km, tCrB_km_vw);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);
    __syncthreads();

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gK_in),   tCrBuf0);
        cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gB_in),   tCrBuf1);
        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gG_in), tCrF0);
    } else {
        clear(tCrBuf0);
        clear(tCrBuf1);
        clear(tCrF0);
        cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gK_in),   tCrBuf0);
        cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gB_in),   tCrBuf1);
        cute::copy_if(tiled_copy_f32, tPredF32_64, thr_copy_f32.partition_S(gG_in), tCrF0);
    }
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int  local_row = m * 16 + row_id;
        bool row_valid = (local_row < valid_len);
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float e_val = row_valid
                ? (exp2f(tCrF0(v, m, Int<0>{})) *
                   half_to_float(tCrBuf1(v, m, Int<0>{})) *
                   half_to_float(tCrBuf0(v, m, Int<0>{})))
                : 0.0f;
            tCrBuf0(v, m, Int<0>{}) = float_to_out<T>(e_val);
        }
    }
    #pragma unroll
    for (int m = 0; m < size<1>(tXs2_d); ++m) {
        *reinterpret_cast<uint4*>(&tXs2_d(Int<0>{}, m, Int<0>{})) =
            *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
    }
    __syncthreads();

    // Accumulate dY @ E_bar^T directly onto live tCrC
    cute::copy(s2r_copy_ak, tCsA_s3_km, tCrA_km_vw);
    cute::copy(s2r_copy_bk, tCsB_s2_km, tCrB_km_vw);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);

    #pragma unroll
    for (int idx = 0; idx < size(tCrC); ++idx) {
        auto coord = tCcC(idx);
        int r = get<0>(coord);
        int s = get<1>(coord);
        float val = (r > s && r < valid_len) ? tCrC(idx) : 0.0f;
        rC_half(idx) = float_to_out<T>(val);
    }
    cute::copy(rC_half, tCsC_s1);
    __syncthreads();

    // 6. Stage 5: Triangular inverse VJP dT = -tril(A^T @ dA @ A^T, -1) -> gDT
    clear(tCrC);
    cute::copy(s2r_copy_am, tCsA_s4_mn, tCrA_mn_vw);
    cute::copy(s2r_copy_bm, tCsB_s1_mn, tCrB_mn_vw);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);

    #pragma unroll
    for (int i = 0; i < size(tCrC); ++i) {
        rC_half(i) = float_to_out<T>(tCrC(i));
    }
    cute::copy(rC_half, tCsC_s2);
    __syncthreads();

    clear(tCrC);
    cute::copy(s2r_copy_ak, tCsA_s2_km, tCrA_km_vw);
    cute::copy(s2r_copy_bk, tCsB_s4_km, tCrB_km_vw);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);

    #pragma unroll
    for (int idx = 0; idx < size(tCrC); ++idx) {
        auto coord = tCcC(idx);
        int r = get<0>(coord);
        int s = get<1>(coord);
        float val = (r > s && r < valid_len) ? (-tCrC(idx)) : 0.0f;
        rC_half(idx) = float_to_out<T>(val);
    }
    cute::copy(rC_half, tCsC_s1);
    __syncthreads();

    cute::copy(tiled_copy_h, tXs1_s, tCrBuf0);
    cute::copy(tiled_copy_h, tCrBuf0, thr_copy_h.partition_D(gDT_out));
    __syncthreads();

    // 7. Stage 6: dE_bar = A^T @ dY (FP32) -> emit gDB_h, accumulate gDK_h and gDG_h
    clear(tCrC);
    cute::copy(s2r_copy_am, tCsA_s4_mn, tCrA_mn_vw);
    cute::copy(s2r_copy_bm, tCsB_s3_mn, tCrB_mn_vw);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);

    cute::copy(tCrC, tCsC_f32_12);
    __syncthreads();

    cute::copy(tiled_copy_f32, tXsF32_12_s, tCrF0);
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gG_in), tCrF1);
        cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gK_in),   tCrBuf0);
        cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gB_in),   tCrBuf1);
    } else {
        clear(tCrF1);
        clear(tCrBuf0);
        clear(tCrBuf1);
        cute::copy_if(tiled_copy_f32, tPredF32_64, thr_copy_f32.partition_S(gG_in), tCrF1);
        cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gK_in),   tCrBuf0);
        cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gB_in),   tCrBuf1);
    }

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float de_eg = tCrF0(v, m, Int<0>{}) * exp2f(tCrF1(v, m, Int<0>{}));
            float kf    = half_to_float(tCrBuf0(v, m, Int<0>{}));
            float bf    = half_to_float(tCrBuf1(v, m, Int<0>{}));
            tCrBuf0(v, m, Int<0>{}) = float_to_out<T>(de_eg * kf);
            tCrF0(v, m, Int<0>{})   = de_eg * bf;
            tCrF1(v, m, Int<0>{})   = de_eg * bf * kf;
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, tCrBuf0, thr_copy_h.partition_D(gDB_out));
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gDK_out), tCrBuf1);
    } else {
        cute::copy_if(tiled_copy_h, tPredHalf64, tCrBuf0, thr_copy_h.partition_D(gDB_out));
        clear(tCrBuf1);
        cute::copy_if(tiled_copy_h, tPredHalf64, thr_copy_h.partition_S(gDK_out), tCrBuf1);
    }

    #pragma unroll
    for (int i = 0; i < size(tCrBuf1); ++i) {
        tCrBuf1(i) = float_to_out<T>(half_to_float(tCrBuf1(i)) + tCrF0(i));
    }
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h,   tCrBuf1, thr_copy_h.partition_D(gDK_out));
        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDG_out), tCrF0);
    } else {
        cute::copy_if(tiled_copy_h,   tPredHalf64, tCrBuf1, thr_copy_h.partition_D(gDK_out));
        clear(tCrF0);
        cute::copy_if(tiled_copy_f32, tPredF32_64, thr_copy_f32.partition_S(gDG_out), tCrF0);
    }

    #pragma unroll
    for (int i = 0; i < size(tCrF0); ++i) {
        tCrF0(i) += tCrF1(i);
    }
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_f32, tCrF0, thr_copy_f32.partition_D(gDG_out));
    } else {
        cute::copy_if(tiled_copy_f32, tPredF32_64, tCrF0, thr_copy_f32.partition_D(gDG_out));
    }
}

// Kernel 2: Propagate dT backwards through T = tril(E_bar @ K_bar^T, -1) in BC=16 sub-chunks
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int SUB_CHUNK,
    int D_K,
    typename MMAAtom,
    typename TiledMMA141,
    typename S2RCopyAtomA_KM,
    typename S2RCopyAtomA_MNM,
    typename S2RCopyAtomB_MNM,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_chunk_wy_intra_T_bwd(
    const T*       __restrict__ gDT,              // (H_kv, num_chunks, 64, 64) [FP16] - Grad w.r.t. T
    const T*       __restrict__ gK_h,             // (H_kv, total_L, 64) [FP16] - Normalized Key
    const T*       __restrict__ gB_eff_h,         // (H_kv, total_L, 64) [FP16] - Effective erase gate
    const float*   __restrict__ gG_log2_h,        // (H_kv, total_L, 64) [FP32] - Chunk cumsum log2 decay
    const int32_t* __restrict__ chunk_offsets,    // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens, // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ gDB_h,            // (H_kv, total_L, 64) [FP16] - In-place Beta grad
    T*             __restrict__ gDK_h,            // (H_kv, total_L, 64) [FP16] - In-place Key grad
    float*         __restrict__ gDG_h,            // (H_kv, total_L, 64) [FP32] - In-place log-decay grad
    int64_t num_kv_heads,                         // scalar [int64] - Total KV heads
    int64_t num_chunks,                           // scalar [int64] - Total varlen chunks
    int64_t total_L                               // scalar [int64] - Total packed tokens
) {
    // 1. Setup & SMEM Allocation (exact 18.0 KB <= 32 KB)
    int curr_chunk = blockIdx.x;
    int h_kv       = blockIdx.y;
    int tid        = threadIdx.x;

    if (curr_chunk >= num_chunks || h_kv >= num_kv_heads) return;

    int32_t valid_len   = chunk_valid_lens[curr_chunk];
    int32_t chunk_start = chunk_offsets[curr_chunk];
    if (valid_len <= 1) return;

    int row_id   = tid / 8;
    int col_base = (tid % 8) * 8;

    int last_sub       = (valid_len - 1) / SUB_CHUNK;
    int last_sub_valid = valid_len - last_sub * SUB_CHUNK;

    struct alignas(16) SmemStorageK2 {
        T sDT[CHUNK_SIZE * CHUNK_SIZE];
        T sSubA[SUB_CHUNK * D_K];
        union {
            float sSubG[SUB_CHUNK * D_K];
            float sC_sub[SUB_CHUNK * D_K];
        } u_gc;
        union {
            T     sSubK[SUB_CHUNK * D_K];
            float sSubE_f32[SUB_CHUNK * D_K];
        } u_ke;
    };
    static_assert(sizeof(SmemStorageK2) == 18432, "SmemStorageK2 must be exactly 18 KB (18432 bytes)");
    __shared__ SmemStorageK2 smem;

    using SwizzleHalf = Swizzle<3, 3, 3>;
    using SwizzleF32  = Swizzle<3, 2, 3>;

    auto layout_64x64_km  = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{})));
    auto layout_64x64_mnm = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<CHUNK_SIZE>{})));
    auto layout_sub_km    = composition(SwizzleHalf{}, make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}),        make_stride(Int<D_K>{},        Int<1>{})));
    auto layout_sub_mnm   = composition(SwizzleHalf{}, make_layout(make_shape(Int<D_K>{},        Int<SUB_CHUNK>{}),  make_stride(Int<1>{},          Int<D_K>{})));
    auto layout_sub_f32   = composition(SwizzleF32{},  make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}),        make_stride(Int<D_K>{},        Int<1>{})));

    auto sDT       = make_tensor(make_smem_ptr(smem.sDT),            layout_64x64_km);
    auto sDT_T     = make_tensor(make_smem_ptr(smem.sDT),            layout_64x64_mnm);
    auto sSubA     = make_tensor(make_smem_ptr(smem.sSubA),          layout_sub_km);
    auto sSubA_mnm = make_tensor(make_smem_ptr(smem.sSubA),          layout_sub_mnm);
    auto sSubG     = make_tensor(make_smem_ptr(smem.u_gc.sSubG),     layout_sub_f32);
    auto sC_sub    = make_tensor(make_smem_ptr(smem.u_gc.sC_sub),    layout_sub_f32);
    auto sSubK     = make_tensor(make_smem_ptr(smem.u_ke.sSubK),     layout_sub_km);
    auto sSubE_f32 = make_tensor(make_smem_ptr(smem.u_ke.sSubE_f32), layout_sub_f32);

    TiledCopyHalf  tiled_copy_h;
    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_h   = tiled_copy_h.get_slice(tid);
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);

    auto tXsDT_d    = thr_copy_h.partition_D(sDT);
    auto tXsSubA_d  = thr_copy_h.partition_D(sSubA);
    auto tXsSubK_s  = thr_copy_h.partition_S(sSubK);
    auto tXsSubK_d  = thr_copy_h.partition_D(sSubK);
    auto tXsSubG_d  = thr_copy_f32.partition_D(sSubG);
    auto tXsC_sub_s = thr_copy_f32.partition_S(sC_sub);
    auto tXsSubE_d  = thr_copy_f32.partition_D(sSubE_f32);

    auto tCrDT      = make_fragment_like(tXsDT_d);
    auto tCrSubA    = make_fragment_like(tXsSubA_d);
    auto tCrSubB    = make_fragment_like(tXsSubA_d);
    auto tCrSubC    = make_fragment_like(tXsSubA_d);
    auto tCrGSub    = make_fragment_like(tXsC_sub_s);
    auto tCrGSubAlt = make_fragment_like(tXsC_sub_s);

    auto cSub16       = make_identity_tensor(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}));
    auto tXcHalfSub   = thr_copy_h.partition_S(cSub16);
    auto tXcF32Sub    = thr_copy_f32.partition_S(cSub16);
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

    // Shared TiledMMA141 operands across all 16x16x64 sub-chunk GEMMs
    TiledMMA141 tiled_mma_141;
    auto thr_mma_141   = tiled_mma_141.get_slice(tid);
    auto s2r_copy_a_km = make_tiled_copy_A(S2RCopyAtomA_KM{},  tiled_mma_141);
    auto s2r_copy_a_mn = make_tiled_copy_A(S2RCopyAtomA_MNM{}, tiled_mma_141);
    auto s2r_copy_b_mn = make_tiled_copy_B(S2RCopyAtomB_MNM{}, tiled_mma_141);
    auto s2r_thr_a_km  = s2r_copy_a_km.get_slice(tid);
    auto s2r_thr_a_mn  = s2r_copy_a_mn.get_slice(tid);
    auto s2r_thr_b_mn  = s2r_copy_b_mn.get_slice(tid);

    auto s_dT_00   = local_tile(sDT, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(Int<0>{}, Int<0>{}));
    auto tCsB_mnm  = s2r_thr_b_mn.partition_S(sSubA_mnm);
    auto tCsC_141  = thr_mma_141.partition_C(sC_sub);

    auto tCrA_141    = thr_mma_141.partition_fragment_A(s_dT_00);
    auto tCrB_141    = thr_mma_141.partition_fragment_B(sSubA_mnm);
    auto tCrA_km_vw  = s2r_thr_a_km.retile_D(tCrA_141);
    auto tCrA_mn_vw  = s2r_thr_a_mn.retile_D(tCrA_141);
    auto tCrB_mn_vw  = s2r_thr_b_mn.retile_D(tCrB_141);
    auto tCrC_141    = thr_mma_141.make_fragment_C(tCsC_141);

    auto layout_64x64_gmem = make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{}));
    auto layout_sub_gmem   = make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}),        make_stride(Int<D_K>{},        Int<1>{}));

    int64_t kv_mat_off  = (static_cast<int64_t>(h_kv) * num_chunks + curr_chunk) * (CHUNK_SIZE * CHUNK_SIZE);
    int64_t kv_tok_base = (static_cast<int64_t>(h_kv) * total_L + chunk_start) * D_K;

    // 2. Prologue: load stationary dT into sDT
    {
        auto gDT_in = make_tensor(make_gmem_ptr(gDT + kv_mat_off), layout_64x64_gmem);
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gDT_in), tCrDT);
        #pragma unroll
        for (int m = 0; m < size<1>(tXsDT_d); ++m) {
            *reinterpret_cast<uint4*>(&tXsDT_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrDT(Int<0>{}, m, Int<0>{}));
        }
    }
    __syncthreads();

    // 3. Phase A: Row sub-chunks i (dE_row -> gDB_h, gDK_h, gDG_h)
    for (int i = 0; i <= last_sub; ++i) {
        int     i_start   = i * SUB_CHUNK;
        int     i_valid   = (i == last_sub) ? last_sub_valid : SUB_CHUNK;
        bool    is_full_i = (i_valid == SUB_CHUNK);
        int64_t kv_i_off  = kv_tok_base + static_cast<int64_t>(i_start) * D_K;

        auto gKi = make_tensor(make_gmem_ptr(gK_h      + kv_i_off), layout_sub_gmem);
        auto gGi = make_tensor(make_gmem_ptr(gG_log2_h + kv_i_off), layout_sub_gmem);

        if (is_full_i) {
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKi),   tCrSubA);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGi), tCrGSub);
        } else {
            clear(tCrSubA);
            clear(tCrGSub);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gKi),   tCrSubA);
            cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGi), tCrGSub);
        }

        float g_r[8];
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            g_r[v] = (row_id < i_valid) ? tCrGSub(v, Int<0>{}, Int<0>{}) : 0.0f;
        }
        *reinterpret_cast<uint4*>(&tXsSubK_d(Int<0>{}, Int<0>{}, Int<0>{})) =
            *reinterpret_cast<const uint4*>(&tCrSubA(Int<0>{}, Int<0>{}, Int<0>{}));
        #pragma unroll
        for (int x = 0; x < 2; ++x) {
            *reinterpret_cast<uint4*>(&tXsSubG_d(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrGSub(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{}));
        }
        __syncthreads();

        float de_row[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // Diagonal (i, i) in FP32 with strict s < row_id
        for (int s = 0; s < SUB_CHUNK; ++s) {
            if (s < row_id && row_id < i_valid) {
                float dt = half_to_float(sDT(i_start + row_id, i_start + s));
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    de_row[v] += dt *
                                 half_to_float(sSubK(s, col_base + v)) *
                                 exp2f(g_r[v] - sSubG(s, col_base + v));
                }
            }
        }
        __syncthreads();

        // Off-diagonal (j < i) using anchor g_ref = G[i * 16] so all exp2 args <= 0
        if (i > 0) {
            const float* p_g_ref = gG_log2_h + kv_i_off + col_base;
            float4 gr0 = *reinterpret_cast<const float4*>(p_g_ref);
            float4 gr1 = *reinterpret_cast<const float4*>(p_g_ref + 4);
            float r_gref[8] = {gr0.x, gr0.y, gr0.z, gr0.w, gr1.x, gr1.y, gr1.z, gr1.w};

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

                auto s_dT_ij   = local_tile(sDT, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(i, j));
                auto tCsA_ij   = s2r_thr_a_km.partition_S(s_dT_ij);
                cute::copy(s2r_copy_a_km, tCsA_ij,  tCrA_km_vw);
                cute::copy(s2r_copy_b_mn, tCsB_mnm, tCrB_mn_vw);
                cute::gemm(MMAAtom{}, tCrA_141, tCrB_141, tCrC_141);
                __syncthreads();
            }

            cute::copy(tCrC_141, tCsC_141);
            __syncthreads();

            cute::copy(tiled_copy_f32, tXsC_sub_s, tCrGSub);
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                if (row_id < i_valid) {
                    de_row[v] += tCrGSub(v, Int<0>{}, Int<0>{}) * exp2f(g_r[v] - r_gref[v]);
                }
            }
            __syncthreads();
        }

        // In-place update gDB_h, gDK_h, gDG_h from dE_row
        auto gBi  = make_tensor(make_gmem_ptr(gB_eff_h + kv_i_off), layout_sub_gmem);
        auto gDBi = make_tensor(make_gmem_ptr(gDB_h    + kv_i_off), layout_sub_gmem);
        auto gDKi = make_tensor(make_gmem_ptr(gDK_h    + kv_i_off), layout_sub_gmem);
        auto gDGi = make_tensor(make_gmem_ptr(gDG_h    + kv_i_off), layout_sub_gmem);

        cute::copy(tiled_copy_h, tXsSubK_s, tCrSubA);
        if (is_full_i) {
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gBi),  tCrSubB);
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gDBi), tCrSubC);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDGi), tCrGSub);
        } else {
            clear(tCrSubB);
            clear(tCrSubC);
            clear(tCrGSub);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gBi),  tCrSubB);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gDBi), tCrSubC);
            cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gDGi), tCrGSub);
        }

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float k_f  = half_to_float(tCrSubA(v, Int<0>{}, Int<0>{}));
            float b_f  = half_to_float(tCrSubB(v, Int<0>{}, Int<0>{}));
            float de_f = de_row[v];
            tCrSubC(v, Int<0>{}, Int<0>{})  = float_to_out<T>(half_to_float(tCrSubC(v, Int<0>{}, Int<0>{})) + de_f * k_f);
            tCrGSub(v, Int<0>{}, Int<0>{}) += de_f * b_f * k_f;
        }
        if (is_full_i) {
            cute::copy(tiled_copy_h,   tCrSubC, thr_copy_h.partition_D(gDBi));
            cute::copy(tiled_copy_f32, tCrGSub, thr_copy_f32.partition_D(gDGi));
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gDKi), tCrSubA);
        } else {
            cute::copy_if(tiled_copy_h,   tPredHalfSub, tCrSubC, thr_copy_h.partition_D(gDBi));
            cute::copy_if(tiled_copy_f32, tPredF32Sub,  tCrGSub, thr_copy_f32.partition_D(gDGi));
            clear(tCrSubA);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gDKi), tCrSubA);
        }

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            tCrSubA(v, Int<0>{}, Int<0>{}) = float_to_out<T>(
                half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) + de_row[v] * half_to_float(tCrSubB(v, Int<0>{}, Int<0>{}))
            );
        }
        if (is_full_i) {
            cute::copy(tiled_copy_h, tCrSubA, thr_copy_h.partition_D(gDKi));
        } else {
            cute::copy_if(tiled_copy_h, tPredHalfSub, tCrSubA, thr_copy_h.partition_D(gDKi));
        }
        __syncthreads();
    }

    // 4. Phase B: Column sub-chunks j (dK_col -> gDK_h, -k * dK_col -> gDG_h)
    for (int j = 0; j <= last_sub; ++j) {
        int     j_start   = j * SUB_CHUNK;
        int     j_valid   = (j == last_sub) ? last_sub_valid : SUB_CHUNK;
        bool    is_full_j = (j_valid == SUB_CHUNK);
        int64_t kv_j_off  = kv_tok_base + static_cast<int64_t>(j_start) * D_K;

        auto gKj = make_tensor(make_gmem_ptr(gK_h      + kv_j_off), layout_sub_gmem);
        auto gBj = make_tensor(make_gmem_ptr(gB_eff_h  + kv_j_off), layout_sub_gmem);
        auto gGj = make_tensor(make_gmem_ptr(gG_log2_h + kv_j_off), layout_sub_gmem);

        if (is_full_j) {
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKj),   tCrSubC);
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gBj),   tCrSubB);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGj), tCrGSub);
        } else {
            clear(tCrSubC);
            clear(tCrSubB);
            clear(tCrGSub);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gKj),   tCrSubC);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gBj),   tCrSubB);
            cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGj), tCrGSub);
        }

        float g_s[8];
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            bool valid_row = (row_id < j_valid);
            g_s[v] = valid_row ? tCrGSub(v, Int<0>{}, Int<0>{}) : 0.0f;
            float e_val = valid_row
                ? (half_to_float(tCrSubB(v, Int<0>{}, Int<0>{})) * half_to_float(tCrSubC(v, Int<0>{}, Int<0>{})))
                : 0.0f;
            tCrGSubAlt(v, Int<0>{}, Int<0>{}) = e_val;
        }
        #pragma unroll
        for (int x = 0; x < 2; ++x) {
            *reinterpret_cast<uint4*>(&tXsSubE_d(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrGSubAlt(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{}));
            *reinterpret_cast<uint4*>(&tXsSubG_d(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrGSub(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{}));
        }
        __syncthreads();

        float dk_col[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // Diagonal (j, j) in FP32 with strict r > row_id
        for (int r = 0; r < SUB_CHUNK; ++r) {
            if (r > row_id && r < j_valid && row_id < j_valid) {
                float dt = half_to_float(sDT(j_start + r, j_start + row_id));
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    dk_col[v] += dt *
                                 sSubE_f32(r, col_base + v) *
                                 exp2f(sSubG(r, col_base + v) - g_s[v]);
                }
            }
        }
        __syncthreads();

        // Off-diagonal (i > j) with per-block anchor g_ref = G[i * 16]
        for (int i = j + 1; i <= last_sub; ++i) {
            int     i_start   = i * SUB_CHUNK;
            int     i_valid   = (i == last_sub) ? last_sub_valid : SUB_CHUNK;
            bool    is_full_i = (i_valid == SUB_CHUNK);
            int64_t kv_i_off  = kv_tok_base + static_cast<int64_t>(i_start) * D_K;

            const float* p_g_ref = gG_log2_h + kv_i_off + col_base;
            float4 gr0 = *reinterpret_cast<const float4*>(p_g_ref);
            float4 gr1 = *reinterpret_cast<const float4*>(p_g_ref + 4);
            float r_gref[8] = {gr0.x, gr0.y, gr0.z, gr0.w, gr1.x, gr1.y, gr1.z, gr1.w};

            auto gKi = make_tensor(make_gmem_ptr(gK_h      + kv_i_off), layout_sub_gmem);
            auto gBi = make_tensor(make_gmem_ptr(gB_eff_h  + kv_i_off), layout_sub_gmem);
            auto gGi = make_tensor(make_gmem_ptr(gG_log2_h + kv_i_off), layout_sub_gmem);

            if (is_full_i) {
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKi),   tCrSubA);
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gBi),   tCrSubB);
                cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGi), tCrGSub);
            } else {
                clear(tCrSubA);
                clear(tCrSubB);
                clear(tCrGSub);
                cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gKi),   tCrSubA);
                cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gBi),   tCrSubB);
                cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGi), tCrGSub);
            }

            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float val = (row_id < i_valid)
                    ? (half_to_float(tCrSubB(v, Int<0>{}, Int<0>{})) *
                       half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) *
                       exp2f(tCrGSub(v, Int<0>{}, Int<0>{}) - r_gref[v]))
                    : 0.0f;
                tCrSubA(v, Int<0>{}, Int<0>{}) = float_to_out<T>(val);
            }
            *reinterpret_cast<uint4*>(&tXsSubA_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrSubA(Int<0>{}, Int<0>{}, Int<0>{}));
            __syncthreads();

            auto s_dT_T_ji = local_tile(sDT_T, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(j, i));
            auto tCsA_ji   = s2r_thr_a_mn.partition_S(s_dT_T_ji);

            clear(tCrC_141);
            cute::copy(s2r_copy_a_mn, tCsA_ji,  tCrA_mn_vw);
            cute::copy(s2r_copy_b_mn, tCsB_mnm, tCrB_mn_vw);
            cute::gemm(MMAAtom{}, tCrA_141, tCrB_141, tCrC_141);

            cute::copy(tCrC_141, tCsC_141);
            __syncthreads();

            cute::copy(tiled_copy_f32, tXsC_sub_s, tCrGSub);
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                if (row_id < j_valid) {
                    dk_col[v] += tCrGSub(v, Int<0>{}, Int<0>{}) * exp2f(r_gref[v] - g_s[v]);
                }
            }
            __syncthreads();
        }

        // In-place update gDK_h += dK_col and gDG_h -= dK_col * k_j
        auto gDKj = make_tensor(make_gmem_ptr(gDK_h + kv_j_off), layout_sub_gmem);
        auto gDGj = make_tensor(make_gmem_ptr(gDG_h + kv_j_off), layout_sub_gmem);

        if (is_full_j) {
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gDKj),   tCrSubA);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDGj), tCrGSub);
        } else {
            clear(tCrSubA);
            clear(tCrGSub);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gDKj),   tCrSubA);
            cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gDGj), tCrGSub);
        }

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            tCrSubA(v, Int<0>{}, Int<0>{})  = float_to_out<T>(
                half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) + dk_col[v]
            );
            tCrGSub(v, Int<0>{}, Int<0>{}) -= dk_col[v] * half_to_float(tCrSubC(v, Int<0>{}, Int<0>{}));
        }

        if (is_full_j) {
            cute::copy(tiled_copy_h,   tCrSubA, thr_copy_h.partition_D(gDKj));
            cute::copy(tiled_copy_f32, tCrGSub, thr_copy_f32.partition_D(gDGj));
        } else {
            cute::copy_if(tiled_copy_h,   tPredHalfSub, tCrSubA, thr_copy_h.partition_D(gDKj));
            cute::copy_if(tiled_copy_f32, tPredF32Sub,  tCrGSub, thr_copy_f32.partition_D(gDGj));
        }
        __syncthreads();
    }
}

// ============================================================================
// Part 2: Host Dispatch Functions
// ============================================================================

// Outputs: (dk_h, dv_h, db_h, dw_h, dG_h [FP32])
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
chunk_wy_bwd_with_meta(
    const at::Tensor& dR_total,
    const at::Tensor& R_list,
    const at::Tensor& S_list,
    const at::Tensor& dS_next_list,
    const at::Tensor& A_list,
    const at::Tensor& k_h,
    const at::Tensor& v_h,
    const at::Tensor& b_eff_h,
    const at::Tensor& w_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& dk_ext,
    const at::Tensor& dG_ext,
    const VarlenChunkMeta& meta,
    int64_t chunk_size
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(dR_total.is_cuda(), "dR_total must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(dR_total.device());

    auto check_t = [&](const at::Tensor& t, const char* name, int64_t expected_dim, at::ScalarType dtype) {
        TORCH_CHECK(t.is_cuda() && t.device() == dR_total.device(), name, " must be on the same CUDA device as dR_total");
        TORCH_CHECK(t.scalar_type() == dtype, name, " dtype mismatch");
        TORCH_CHECK(t.dim() == expected_dim && t.is_contiguous(), name, " must be contiguous with expected rank");
    };

    check_t(dR_total,     "dR_total",     4, at::ScalarType::Half);
    check_t(R_list,       "R_list",       4, at::ScalarType::Half);
    check_t(S_list,       "S_list",       4, at::ScalarType::Float);
    check_t(dS_next_list, "dS_next_list", 4, at::ScalarType::Float);
    check_t(A_list,       "A_list",       4, at::ScalarType::Half);
    check_t(k_h,          "k_h",          3, at::ScalarType::Half);
    check_t(v_h,          "v_h",          3, at::ScalarType::Half);
    check_t(b_eff_h,      "b_eff_h",      3, at::ScalarType::Half);
    check_t(w_h,          "w_h",          3, at::ScalarType::Half);
    check_t(G_log2_h,     "G_log2_h",     3, at::ScalarType::Float);
    check_t(dk_ext,       "dk_ext",       3, at::ScalarType::Half);
    check_t(dG_ext,       "dG_ext",       3, at::ScalarType::Float);

    int64_t num_kv_heads = k_h.size(0);
    int64_t total_L      = k_h.size(1);
    int64_t d_k          = k_h.size(2);
    int64_t d_v          = v_h.size(2);
    int64_t num_chunks   = meta.num_chunks;

    TORCH_CHECK(d_k == 64 && d_v == 64, "d_k and d_v must be 64");
    TORCH_CHECK(v_h.size(0) == num_kv_heads && v_h.size(1) == total_L, "v_h shape mismatch");
    TORCH_CHECK(b_eff_h.size(0) == num_kv_heads && b_eff_h.size(1) == total_L && b_eff_h.size(2) == d_k,
                "b_eff_h shape mismatch");
    TORCH_CHECK(w_h.size(0) == num_kv_heads && w_h.size(1) == total_L && w_h.size(2) == d_v,
                "w_h shape mismatch");
    TORCH_CHECK(G_log2_h.size(0) == num_kv_heads && G_log2_h.size(1) == total_L && G_log2_h.size(2) == d_k,
                "G_log2_h shape mismatch");
    TORCH_CHECK(dk_ext.size(0) == num_kv_heads && dk_ext.size(1) == total_L && dk_ext.size(2) == d_k,
                "dk_ext shape mismatch");
    TORCH_CHECK(dG_ext.size(0) == num_kv_heads && dG_ext.size(1) == total_L && dG_ext.size(2) == d_k,
                "dG_ext shape mismatch");

    TORCH_CHECK(dR_total.size(0) == num_kv_heads && dR_total.size(1) == num_chunks &&
                dR_total.size(2) == chunk_size && dR_total.size(3) == d_v, "dR_total shape mismatch");
    TORCH_CHECK(R_list.size(0) == num_kv_heads && R_list.size(1) == num_chunks &&
                R_list.size(2) == chunk_size && R_list.size(3) == d_v, "R_list shape mismatch");
    TORCH_CHECK(S_list.size(0) == num_kv_heads && S_list.size(1) == num_chunks &&
                S_list.size(2) == d_v && S_list.size(3) == d_k, "S_list shape mismatch");
    TORCH_CHECK(dS_next_list.size(0) == num_kv_heads && dS_next_list.size(1) == num_chunks &&
                dS_next_list.size(2) == d_v && dS_next_list.size(3) == d_k, "dS_next_list shape mismatch");
    TORCH_CHECK(A_list.size(0) == num_kv_heads && A_list.size(1) == num_chunks &&
                A_list.size(2) == chunk_size && A_list.size(3) == chunk_size, "A_list shape mismatch");

    auto h_opts = dR_total.options();
    auto f_opts = G_log2_h.options();

    if (num_chunks <= 0 || total_L == 0) {
        at::Tensor dk_h = at::zeros({num_kv_heads, total_L, d_k}, h_opts);
        at::Tensor dv_h = at::zeros({num_kv_heads, total_L, d_v}, h_opts);
        at::Tensor db_h = at::zeros({num_kv_heads, total_L, d_k}, h_opts);
        at::Tensor dw_h = at::zeros({num_kv_heads, total_L, d_v}, h_opts);
        at::Tensor dG_h = at::zeros({num_kv_heads, total_L, d_k}, f_opts);
        return std::make_tuple(dk_h, dv_h, db_h, dw_h, dG_h);
    }

    TORCH_CHECK(meta.chunk_offsets.is_cuda() && meta.chunk_offsets.device() == dR_total.device() &&
                meta.chunk_offsets.scalar_type() == at::ScalarType::Int &&
                meta.chunk_offsets.dim() == 1 && meta.chunk_offsets.is_contiguous() &&
                meta.chunk_offsets.size(0) >= num_chunks + 1, "meta.chunk_offsets mismatch");
    TORCH_CHECK(meta.chunk_valid_lens.is_cuda() && meta.chunk_valid_lens.device() == dR_total.device() &&
                meta.chunk_valid_lens.scalar_type() == at::ScalarType::Int &&
                meta.chunk_valid_lens.dim() == 1 && meta.chunk_valid_lens.is_contiguous() &&
                meta.chunk_valid_lens.size(0) >= num_chunks, "meta.chunk_valid_lens mismatch");

    at::Tensor dk_h = torch::empty({num_kv_heads, total_L, d_k}, h_opts);
    at::Tensor dv_h = torch::empty({num_kv_heads, total_L, d_v}, h_opts);
    at::Tensor db_h = torch::empty({num_kv_heads, total_L, d_k}, h_opts);
    at::Tensor dw_h = torch::empty({num_kv_heads, total_L, d_v}, h_opts);
    at::Tensor dG_h = torch::empty({num_kv_heads, total_L, d_k}, f_opts);
    at::Tensor dT   = torch::empty({num_kv_heads, num_chunks, chunk_size, chunk_size}, h_opts);

    at::Tensor dR_a     = ensure_aligned16(dR_total);
    at::Tensor R_a      = ensure_aligned16(R_list);
    at::Tensor S_a      = ensure_aligned16(S_list);
    at::Tensor dSn_a    = ensure_aligned16(dS_next_list);
    at::Tensor A_a      = ensure_aligned16(A_list);
    at::Tensor k_a      = ensure_aligned16(k_h);
    at::Tensor v_a      = ensure_aligned16(v_h);
    at::Tensor b_a      = ensure_aligned16(b_eff_h);
    at::Tensor w_a      = ensure_aligned16(w_h);
    at::Tensor G_a      = ensure_aligned16(G_log2_h);
    at::Tensor dk_ext_a = ensure_aligned16(dk_ext);
    at::Tensor dG_ext_a = ensure_aligned16(dG_ext);

    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_CONST = 64;
    constexpr int SUB_CONST   = 16;
    constexpr int DK_CONST    = 64;
    constexpr int DV_CONST    = 64;

    using Element     = cutlass::half_t;
    using MMAAtom     = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMA221 = TiledMMA<MMAAtom, Layout<Shape<_2, _2, _1>, Stride<_1, _2, _4>>>;
    using TiledMMA141 = TiledMMA<MMAAtom, Layout<Shape<_1, _4, _1>, Stride<_1, _1, _4>>>;

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

    const Element* p_dR     = reinterpret_cast<const Element*>(dR_a.data_ptr<at::Half>());
    const Element* p_R      = reinterpret_cast<const Element*>(R_a.data_ptr<at::Half>());
    const float*   p_S      = S_a.data_ptr<float>();
    const float*   p_dSn    = dSn_a.data_ptr<float>();
    const Element* p_A      = reinterpret_cast<const Element*>(A_a.data_ptr<at::Half>());
    const Element* p_k      = reinterpret_cast<const Element*>(k_a.data_ptr<at::Half>());
    const Element* p_v      = reinterpret_cast<const Element*>(v_a.data_ptr<at::Half>());
    const Element* p_b      = reinterpret_cast<const Element*>(b_a.data_ptr<at::Half>());
    const Element* p_w      = reinterpret_cast<const Element*>(w_a.data_ptr<at::Half>());
    const float*   p_G      = G_a.data_ptr<float>();
    const Element* p_dk_ext = reinterpret_cast<const Element*>(dk_ext_a.data_ptr<at::Half>());
    const float*   p_dG_ext = dG_ext_a.data_ptr<float>();
    const int32_t* p_off    = meta.chunk_offsets.data_ptr<int32_t>();
    const int32_t* p_vlen   = meta.chunk_valid_lens.data_ptr<int32_t>();

    Element* p_dT = reinterpret_cast<Element*>(dT.data_ptr<at::Half>());
    Element* p_dk = reinterpret_cast<Element*>(dk_h.data_ptr<at::Half>());
    Element* p_dv = reinterpret_cast<Element*>(dv_h.data_ptr<at::Half>());
    Element* p_db = reinterpret_cast<Element*>(db_h.data_ptr<at::Half>());
    Element* p_dw = reinterpret_cast<Element*>(dw_h.data_ptr<at::Half>());
    float*   p_dG = dG_h.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // 1. Launch Kernel 1: dK_tail, dY, dA, dT, dZ -> (dv, dw), dE_bar -> (db, dk, dG)
    {
        dim3 grid(static_cast<unsigned int>(num_chunks), static_cast<unsigned int>(num_kv_heads));
        dim3 block(NUM_THREADS);
        kernel_chunk_wy_dA_dT_bwd<
            Element, NUM_THREADS, CHUNK_CONST, DK_CONST, DV_CONST,
            MMAAtom, TiledMMA221,
            S2RCopyAtomA_KM, S2RCopyAtomB_KM, S2RCopyAtomA_MNM, S2RCopyAtomB_MNM,
            TiledCopyHalf, TiledCopyFloat
        ><<<grid, block, 0, stream>>>(
            p_dR, p_R, p_S, p_dSn, p_A,
            p_k, p_v, p_b, p_w, p_G, p_dk_ext, p_dG_ext,
            p_off, p_vlen,
            p_dT, p_dk, p_dv, p_db, p_dw, p_dG,
            num_kv_heads, num_chunks, total_L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 2. Launch Kernel 2: Propagate dT backwards through BC=16 intra-chunk T
    {
        dim3 grid(static_cast<unsigned int>(num_chunks), static_cast<unsigned int>(num_kv_heads));
        dim3 block(NUM_THREADS);
        kernel_chunk_wy_intra_T_bwd<
            Element, NUM_THREADS, CHUNK_CONST, SUB_CONST, DK_CONST,
            MMAAtom, TiledMMA141,
            S2RCopyAtomA_KM, S2RCopyAtomA_MNM, S2RCopyAtomB_MNM,
            TiledCopyHalf, TiledCopyFloat
        ><<<grid, block, 0, stream>>>(
            p_dT, p_k, p_b, p_G, p_off, p_vlen,
            p_db, p_dk, p_dG,
            num_kv_heads, num_chunks, total_L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    dT.reset();
    return std::make_tuple(dk_h, dv_h, db_h, dw_h, dG_h);
}

// Outputs: (dk_h, dv_h, db_h, dw_h, dG_h [FP32])
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
chunk_wy_bwd(
    const at::Tensor& dR_total,
    const at::Tensor& R_list,
    const at::Tensor& S_list,
    const at::Tensor& dS_next_list,
    const at::Tensor& A_list,
    const at::Tensor& k_h,
    const at::Tensor& v_h,
    const at::Tensor& b_eff_h,
    const at::Tensor& w_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& dk_ext,
    const at::Tensor& dG_ext,
    const at::Tensor& cu_seq_lens,
    int64_t chunk_size
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 1,
                "cu_seq_lens must be a 1D CUDA tensor with size >= 1");
    VarlenChunkMeta meta = compute_varlen_chunk_metadata(cu_seq_lens, chunk_size);
    return chunk_wy_bwd_with_meta(
        dR_total, R_list, S_list, dS_next_list, A_list,
        k_h, v_h, b_eff_h, w_h, G_log2_h, dk_ext, dG_ext,
        meta, chunk_size
    );
}
