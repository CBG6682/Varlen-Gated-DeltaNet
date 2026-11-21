#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

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

// Output combine forward kernel fusing inter-chunk and intra-chunk attention.
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_K,
    int D_V,
    typename MMAAtom,
    typename TiledMMAType,
    typename S2RCopyAtomA,
    typename S2RCopyAtomB_KMajor,
    typename S2RCopyAtomB_MNMajor,
    typename TiledCopyG2R
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_output_combine_fwd(
    const T* __restrict__ gQ,                   // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const T* __restrict__ gK,                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const T* __restrict__ gV,                   // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const T* __restrict__ gB,                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const T* __restrict__ gW_gate,              // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const T* __restrict__ gGamma,               // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const T* __restrict__ gAlpha,               // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const T* __restrict__ gS,                   // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent State
    const int32_t* __restrict__ gChunkValidLen, // (num_chunks,) [Int32]
    const int32_t* __restrict__ gChunkOffsets,  // (num_chunks + 1,) [Int32]
    T* __restrict__ gOut,                       // (num_q_heads, total_L, d_v) [FP16] - Output
    int64_t num_q_heads,                        // scalar [int64]
    int64_t num_kv_heads,                       // scalar [int64]
    int64_t num_groups,                         // scalar [int64] - num_q_heads / num_kv_heads
    int64_t num_chunks,                         // scalar [int64]
    int64_t total_L                             // scalar [int64]
) {
    // Stage 1 - Thread and Block indexing (Grid: num_kv_heads, num_chunks)
    int h_kv       = blockIdx.x;
    int curr_chunk = blockIdx.y;
    int tid        = threadIdx.x;

    if (h_kv >= num_kv_heads || curr_chunk >= num_chunks) return;

    int valid_len   = gChunkValidLen[curr_chunk];
    int chunk_start = gChunkOffsets[curr_chunk];
    if (valid_len <= 0) return;

    int64_t kv_head_offset = (int64_t)h_kv * total_L;

    constexpr int ROWS_PER_STP = 16;
    constexpr int NUM_STPS     = CHUNK_SIZE / ROWS_PER_STP; // 4 steps
    int row_in_step = tid / (D_K / 8);                      // 0..15
    int col_vec_idx = tid % (D_K / 8);                      // 0..7
    int col_base    = col_vec_idx * 8;                      // 0..56

    int warp_id      = tid / 32;                            // 0..3
    int lane_id      = tid % 32;                            // 0..31
    int quad_id      = lane_id / 4;                         // 0..7
    int lane_in_quad = lane_id % 4;                         // 0..3

    int row_a = warp_id * 16 + quad_id;
    int row_b = warp_id * 16 + quad_id + 8;

    // Allocate 32 KB SMEM: 4 buffers of 8 KB aligned to 16 bytes
    struct SmemStorage {
        alignas(16) T smem_1[CHUNK_SIZE * D_K]; // 8 KB: Region 1 (Q for group g)
        alignas(16) T smem_2[CHUNK_SIZE * D_K]; // 8 KB: Region 2 (stationary K_scaled)
        alignas(16) T smem_3[CHUNK_SIZE * D_V]; // 8 KB: Region 3 (stationary V_new)
        alignas(16) T smem_4[D_V * D_K];        // 8 KB: Region 4 (stationary S; prefix scan in prologue)
    };
    static_assert(sizeof(SmemStorage) == 32768, "SmemStorage must be exactly 32 KB (32768 bytes)");
    __shared__ SmemStorage smem;

    T* s_Q = smem.smem_1;
    T* s_K = smem.smem_2;
    T* s_V = smem.smem_3;
    T* s_S = smem.smem_4;

    using SwizzleOp = Swizzle<3, 3, 3>;

    auto smem_layout_flat_k = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_flat_v = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto smem_layout_flat_s = composition(SwizzleOp{}, make_layout(make_shape(Int<D_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    auto sQ_flat = make_tensor(make_smem_ptr(s_Q), smem_layout_flat_k);
    auto sK_flat = make_tensor(make_smem_ptr(s_K), smem_layout_flat_k);
    auto sV_flat = make_tensor(make_smem_ptr(s_V), smem_layout_flat_v);
    auto sS_flat = make_tensor(make_smem_ptr(s_S), smem_layout_flat_s);

    // SMEM layouts for GEMMs
    // GEMM 1: Q_S = Q @ S^T. Q: M=64, K=64 (K-major). S^T: N=64, K=64. S(v, k) stride k=1 (K-major).
    auto smem_layout_A_Q = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_B_S = composition(SwizzleOp{}, make_layout(make_shape(Int<D_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto sA_Q = make_tensor(make_smem_ptr(s_Q), smem_layout_A_Q);
    auto sB_S = make_tensor(make_smem_ptr(s_S), smem_layout_B_S);

    // GEMM 2: R = Q @ K_scaled^T. K_scaled^T: N=64, K=64. K(j, k) stride k=1 (K-major).
    auto smem_layout_B_K = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto sB_K = make_tensor(make_smem_ptr(s_K), smem_layout_B_K);

    // GEMM 3: O_total = A @ V_new + O_inter. A: M=64, K=64. V_new: N=64, K=64.
    // In CuTe B shape (N, K) = (D_V, CHUNK_SIZE). V_new(j, v) stride v=1 (MN-major).
    auto smem_layout_B_V = composition(SwizzleOp{}, make_layout(make_shape(Int<D_V>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_V>{})));
    auto sB_V = make_tensor(make_smem_ptr(s_V), smem_layout_B_V);

    // Dummy layout for partition_C
    auto smem_layout_C = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto sC_dummy = make_tensor(make_smem_ptr(s_Q), smem_layout_C);

    TiledMMAType tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // S2R Tiled Copies
    auto s2r_copy_a_km = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_copy_b_km = make_tiled_copy_B(S2RCopyAtomB_KMajor{}, tiled_mma);
    auto s2r_copy_b_mn = make_tiled_copy_B(S2RCopyAtomB_MNMajor{}, tiled_mma);

    auto s2r_thr_copy_a_km = s2r_copy_a_km.get_slice(tid);
    auto s2r_thr_copy_b_km = s2r_copy_b_km.get_slice(tid);
    auto s2r_thr_copy_b_mn = s2r_copy_b_mn.get_slice(tid);

    auto tCsA_Q = s2r_thr_copy_a_km.partition_S(sA_Q);
    auto tCsB_S = s2r_thr_copy_b_km.partition_S(sB_S);
    auto tCsB_K = s2r_thr_copy_b_km.partition_S(sB_K);
    auto tCsB_V = s2r_thr_copy_b_mn.partition_S(sB_V);

    // Single pair (tCrA, tCrB) reused across all 3 GEMMs
    auto tCrA = thr_mma.partition_fragment_A(sA_Q);
    auto tCrB = thr_mma.partition_fragment_B(sB_S);

    auto tCrA_view    = s2r_thr_copy_a_km.retile_D(tCrA);
    auto tCrB_km_view = s2r_thr_copy_b_km.retile_D(tCrB);
    auto tCrB_mn_view = s2r_thr_copy_b_mn.retile_D(tCrB);

    // FP32 Accumulators for C
    auto tCrC_O = thr_mma.partition_fragment_C(sC_dummy);
    auto tCrC_R = thr_mma.partition_fragment_C(sC_dummy);

    TiledCopyG2R tiled_copy_g2r;
    auto thr_copy_g2r = tiled_copy_g2r.get_thread_slice(tid);

    auto dummy_tile = local_tile(sQ_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(0, 0));
    auto dummy_part = thr_copy_g2r.partition_S(dummy_tile);

    // Register prefetch buffer for Q_next
    decltype(make_fragment_like(dummy_part)) r_q_next[NUM_STPS];

    // Stage 1.1 - 2-Warp Prefix Scan computing intra-chunk log_alpha_cumsum
    // Borrow smem_4 as warp exchange buffer, then release
    float* s_temp_log_alpha = reinterpret_cast<float*>(smem.smem_4);
    float* s_temp_warp0_sum = reinterpret_cast<float*>(smem.smem_4) + CHUNK_SIZE;

    if (tid < CHUNK_SIZE) {
        float alpha_val = 0.0f;
        if (tid < valid_len) {
            int64_t token_idx = (int64_t)chunk_start + tid;
            alpha_val = stable_log_sigmoid(__half2float(to_half_val(gAlpha[kv_head_offset + token_idx])));
        }
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            float up = __shfl_up_sync(0xffffffff, alpha_val, offset);
            if ((tid & 31) >= offset) {
                alpha_val += up;
            }
        }
        if (tid == 31) {
            *s_temp_warp0_sum = alpha_val;
        }
        s_temp_log_alpha[tid] = alpha_val;
    }
    __syncthreads();

    if (tid >= 32 && tid < CHUNK_SIZE) {
        s_temp_log_alpha[tid] += *s_temp_warp0_sum;
    }
    __syncthreads();

    // Cache row and 16 column coordinates on registers
    float r_c_row_a = (row_a < valid_len) ? s_temp_log_alpha[row_a] : 0.0f;
    float r_c_row_b = (row_b < valid_len) ? s_temp_log_alpha[row_b] : 0.0f;
    float r_a_t_a   = (row_a < valid_len) ? expf(r_c_row_a) : 0.0f;
    float r_a_t_b   = (row_b < valid_len) ? expf(r_c_row_b) : 0.0f;

    float r_c_col[16];
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
        int col_0 = n * 8 + lane_in_quad * 2 + 0;
        int col_1 = col_0 + 1;
        r_c_col[2 * n + 0] = (col_0 < valid_len) ? s_temp_log_alpha[col_0] : 0.0f;
        r_c_col[2 * n + 1] = (col_1 < valid_len) ? s_temp_log_alpha[col_1] : 0.0f;
    }
    __syncthreads(); // Release smem_4 completely

    // Stage 2 - Prologue: Load K_scaled (Region 2), V_new (Region 3), S (Region 4), Q_0 (Region 1)
    int64_t k_offset_curr  = kv_head_offset * D_K + (int64_t)chunk_start * D_K;
    int64_t v_offset_curr  = kv_head_offset * D_V + (int64_t)chunk_start * D_V;
    const T* p_gamma_chunk = gGamma + kv_head_offset;

    // 1. Recompute K_scaled and write stationary copy into Region 2 (smem_2)
    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_len);
        int64_t token_idx = (int64_t)chunk_start + local_row;

        decltype(make_fragment_like(dummy_part)) rK;
        clear(rK);
        if (row_valid) {
            *reinterpret_cast<uint4*>(&rK(0)) = *reinterpret_cast<const uint4*>(gK + k_offset_curr + local_row * D_K + col_base);
        }

        float sqrt_gamma_f = 0.0f;
        if (row_valid) {
            __half g_val = to_half_val(p_gamma_chunk[token_idx]);
            __half sp    = fast_softplus_h(g_val);
            sqrt_gamma_f = sqrtf(fmaxf(__half2float(sp), 0.0f) + 1e-4f);
        }

        float thread_sq_sum = 0.0f;
        if (row_valid) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float ki = half_to_float(rK(i));
                thread_sq_sum += ki * ki;
            }
        }
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            thread_sq_sum += __shfl_xor_sync(0xffffffff, thread_sq_sum, mask);
        }
        float k_norm = fmaxf(sqrtf(fmaxf(thread_sq_sum, 0.0f)), 1e-4f);
        float inv_norm_scale = row_valid ? (sqrt_gamma_f / k_norm) : 0.0f;
        __half2 inv_norm_scale_h2 = __half2half2(__float2half(inv_norm_scale));

        decltype(make_fragment_like(dummy_part)) rK_scaled;
        half2* rK_h2    = reinterpret_cast<half2*>(&rK(0));
        half2* rK_sc_h2 = reinterpret_cast<half2*>(&rK_scaled(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            rK_sc_h2[p] = __hmul2(rK_h2[p], inv_norm_scale_h2);
        }

        auto sK_step = local_tile(sK_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tKsK    = thr_copy_g2r.partition_D(sK_step);
        *reinterpret_cast<uint4*>(&tKsK(0)) = *reinterpret_cast<const uint4*>(&rK_scaled(0));
    }

    // 2. Recompute V_new and write stationary copy into Region 3 (smem_3)
    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_len);
        int64_t token_idx = (int64_t)chunk_start + local_row;

        decltype(make_fragment_like(dummy_part)) rV_raw, rW_raw;
        clear(rV_raw); clear(rW_raw);
        if (row_valid) {
            *reinterpret_cast<uint4*>(&rV_raw(0)) = *reinterpret_cast<const uint4*>(gV + v_offset_curr + local_row * D_V + col_base);
            *reinterpret_cast<uint4*>(&rW_raw(0)) = *reinterpret_cast<const uint4*>(gW_gate + v_offset_curr + local_row * D_V + col_base);
        }

        float sqrt_g = 0.0f;
        if (row_valid) {
            __half g_val = to_half_val(p_gamma_chunk[token_idx]);
            __half sp    = fast_softplus_h(g_val);
            sqrt_g       = sqrtf(fmaxf(__half2float(sp), 0.0f) + 1e-4f);
        }
        __half2 sqrt_g_h2 = __half2half2(__float2half(sqrt_g));

        decltype(make_fragment_like(dummy_part)) rV_new;
        half2* rV_raw_h2 = reinterpret_cast<half2*>(&rV_raw(0));
        half2* rW_raw_h2 = reinterpret_cast<half2*>(&rW_raw(0));
        half2* rV_new_h2 = reinterpret_cast<half2*>(&rV_new(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            half2 v_sc   = __hmul2(rV_raw_h2[p], sqrt_g_h2);
            half2 w_s    = fast_sigmoid_h2(rW_raw_h2[p]);
            rV_new_h2[p] = __hmul2(w_s, v_sc);
        }

        auto sV_step = local_tile(sV_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto tVsV    = thr_copy_g2r.partition_D(sV_step);
        *reinterpret_cast<uint4*>(&tVsV(0)) = *reinterpret_cast<const uint4*>(&rV_new(0));
    }

    // 3. Load S from GMEM into Region 4 (smem_4)
    int64_t s_offset_curr = ((int64_t)h_kv * num_chunks + curr_chunk) * ((int64_t)D_V * D_K);
    const T* p_s_curr     = gS + s_offset_curr;

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_v = st * ROWS_PER_STP + row_in_step;
        decltype(make_fragment_like(dummy_part)) rS;
        *reinterpret_cast<uint4*>(&rS(0)) = *reinterpret_cast<const uint4*>(p_s_curr + local_v * D_K + col_base);

        auto sS_step = local_tile(sS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tSsS    = thr_copy_g2r.partition_D(sS_step);
        *reinterpret_cast<uint4*>(&tSsS(0)) = *reinterpret_cast<const uint4*>(&rS(0));
    }

    // 4. Load Q for group 0 (g = 0) from GMEM into Region 1 (smem_1)
    int64_t q_offset_0 = ((int64_t)(h_kv * num_groups) * total_L + chunk_start) * D_K;
    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_len);
        decltype(make_fragment_like(dummy_part)) rQ;
        clear(rQ);
        if (row_valid) {
            *reinterpret_cast<uint4*>(&rQ(0)) = *reinterpret_cast<const uint4*>(gQ + q_offset_0 + local_row * D_K + col_base);
        }

        auto sQ_step = local_tile(sQ_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tQsQ    = thr_copy_g2r.partition_D(sQ_step);
        *reinterpret_cast<uint4*>(&tQsQ(0)) = *reinterpret_cast<const uint4*>(&rQ(0));
    }
    __syncthreads(); // [SYNC 0] (RAW: Q in Region 1, K in Region 2, V in Region 3, S in Region 4)

    // Stage 3 - Sequential loop over Query heads in GQA group (g = 0..num_groups - 1)
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t curr_h_q      = (int64_t)h_kv * num_groups + g;
        int64_t q_head_offset = curr_h_q * total_L;
        int64_t out_base      = (q_head_offset + (int64_t)chunk_start) * D_V;

        // Step 3.1 - GEMM 1: Q_S = Q @ S^T & Compute O_inter
        cute::copy(s2r_copy_a_km, tCsA_Q, tCrA_view);
        cute::copy(s2r_copy_b_km, tCsB_S, tCrB_km_view);

        clear(tCrC_O);
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_O);

        // In-place register epilogue: O_inter = (Q @ S^T) * a_t (FP32)
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            tCrC_O(4 * n + 0) *= r_a_t_a;
            tCrC_O(4 * n + 1) *= r_a_t_a;
            tCrC_O(4 * n + 2) *= r_a_t_b;
            tCrC_O(4 * n + 3) *= r_a_t_b;
        }
        // Retain tCrA containing Q directly for GEMM 2 without SMEM reload

        // Step 3.2 - GEMM 2: R = Q @ K_scaled^T & Epilogue into tCrA
        cute::copy(s2r_copy_b_km, tCsB_K, tCrB_km_view);

        clear(tCrC_R);
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_R);

        // Prefetch next query head: g2r load Q_next from GMEM into r_q_next
        if (g + 1 < num_groups) {
            int64_t next_h_q      = (int64_t)h_kv * num_groups + (g + 1);
            int64_t q_offset_next = (next_h_q * total_L + chunk_start) * D_K;
            #pragma unroll
            for (int st = 0; st < NUM_STPS; ++st) {
                int local_row = st * ROWS_PER_STP + row_in_step;
                clear(r_q_next[st]);
                if (local_row < valid_len) {
                    *reinterpret_cast<uint4*>(&r_q_next[st](0)) = *reinterpret_cast<const uint4*>(gQ + q_offset_next + local_row * D_K + col_base);
                }
            }
        }

        // In-place register epilogue: A = R * E, cast to FP16 and overwrite into tCrA
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int col_0 = n * 8 + lane_in_quad * 2 + 0;
            int col_1 = col_0 + 1;

            float e_a0 = (row_a >= col_0 && row_a < valid_len && col_0 < valid_len) ? expf(r_c_row_a - r_c_col[2 * n + 0]) : 0.0f;
            float e_a1 = (row_a >= col_1 && row_a < valid_len && col_1 < valid_len) ? expf(r_c_row_a - r_c_col[2 * n + 1]) : 0.0f;
            float e_b0 = (row_b >= col_0 && row_b < valid_len && col_0 < valid_len) ? expf(r_c_row_b - r_c_col[2 * n + 0]) : 0.0f;
            float e_b1 = (row_b >= col_1 && row_b < valid_len && col_1 < valid_len) ? expf(r_c_row_b - r_c_col[2 * n + 1]) : 0.0f;

            tCrA(4 * n + 0) = float_to_out<T>(tCrC_R(4 * n + 0) * e_a0);
            tCrA(4 * n + 1) = float_to_out<T>(tCrC_R(4 * n + 1) * e_a1);
            tCrA(4 * n + 2) = float_to_out<T>(tCrC_R(4 * n + 2) * e_b0);
            tCrA(4 * n + 3) = float_to_out<T>(tCrC_R(4 * n + 3) * e_b1);
        }

        // Step 3.3 - GEMM 3: O_total = A @ V_new + O_inter & Staging to Region 1
        cute::copy(s2r_copy_b_mn, tCsB_V, tCrB_mn_view);

        // Accumulate A @ V_new directly into tCrC_O (which already holds O_inter)
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_O);

        // [SYNC 1] (WAR: ensure all warps have finished reading Q from Region 1)
        __syncthreads();

        // Stage tCrC_O (O_total, cast to half2) into Region 1 (sQ_flat with Swizzle<3, 3, 3>)
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int target_col = n * 8 + lane_in_quad * 2;
            half2 val_a = __floats2half2_rn(tCrC_O(4 * n + 0), tCrC_O(4 * n + 1));
            half2 val_b = __floats2half2_rn(tCrC_O(4 * n + 2), tCrC_O(4 * n + 3));

            reinterpret_cast<half2*>(&sQ_flat(row_a, target_col))[0] = val_a;
            reinterpret_cast<half2*>(&sQ_flat(row_b, target_col))[0] = val_b;
        }

        // [SYNC 2] (RAW: ensure O_total has been fully written to Region 1)
        __syncthreads();

        // Step 3.4 - Store O_total from Region 1 to GMEM (uint4) & update Region 1 with Q_next
        #pragma unroll
        for (int st = 0; st < NUM_STPS; ++st) {
            int local_row = st * ROWS_PER_STP + row_in_step;
            uint4 o_vec = *reinterpret_cast<const uint4*>(&sQ_flat(local_row, col_base));

            if (local_row < valid_len) {
                *reinterpret_cast<uint4*>(gOut + out_base + local_row * D_V + col_base) = o_vec;
            }

            // If next group exists, overwrite r_q_next into sQ_flat
            if (g + 1 < num_groups) {
                auto sQ_step = local_tile(sQ_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                auto tQsQ    = thr_copy_g2r.partition_D(sQ_step);
                *reinterpret_cast<uint4*>(&tQsQ(0)) = *reinterpret_cast<const uint4*>(&r_q_next[st](0));
            }
        }

        if (g + 1 < num_groups) {
            // [SYNC 3] (RAW: ensure Q_next is ready in Region 1 for next iteration)
            __syncthreads();
        }
    }
}

// Host wrapper for output_combine_fwd. Output: out (H_q, total_L, d_v).
at::Tensor
output_combine_fwd(
    const at::Tensor& q,                // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const at::Tensor& k,                // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,                // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& b,                // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& w,                // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& gamma,            // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& a,                // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& S,                // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent state
    const at::Tensor& cu_seq_lens,      // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t num_q_heads,                // scalar [int64] - Number of query heads
    int64_t num_kv_heads,               // scalar [int64] - Number of KV heads (GQA supported)
    int64_t d_k,                        // scalar [int64] - Head dimension Key (default: 64)
    int64_t d_v,                        // scalar [int64] - Head dimension Value (default: 64)
    int64_t chunk_size                  // scalar [int64] - Chunk size (default: 64)
) {
    // Stage 1 - Input Validation
    at::cuda::CUDAGuard device_guard(q.device());

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && b.is_cuda() && w.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(gamma.is_cuda() && a.is_cuda() && S.is_cuda() && cu_seq_lens.is_cuda(), "Gamma, a, S, cu_seq_lens must be CUDA tensors");

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && b.is_contiguous() && w.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(gamma.is_contiguous() && a.is_contiguous() && S.is_contiguous(), "Gamma, a, S must be contiguous");

    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half, "q must be float16");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Half, "k must be float16");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::Half, "v must be float16");
    TORCH_CHECK(b.scalar_type() == at::ScalarType::Half, "b must be float16");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Half, "w must be float16");
    TORCH_CHECK(gamma.scalar_type() == at::ScalarType::Half, "gamma must be float16");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half, "a must be float16");
    TORCH_CHECK(S.scalar_type() == at::ScalarType::Half, "S must be float16");

    TORCH_CHECK(d_k == 64, "d_k must be 64 for SM75");
    TORCH_CHECK(d_v == 64, "d_v must be 64 for SM75");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64 for SM75");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads for GQA");

    int64_t total_L    = q.size(1);
    int64_t num_groups = num_q_heads / num_kv_heads;

    // Stage 2 - Kernel Launch Configuration
    auto [chunk_offsets_gpu, chunk_valid_lens_gpu, num_chunks] = compute_chunk_offsets_and_valid_lens(cu_seq_lens, chunk_size);

    at::Tensor out = torch::empty({num_q_heads, total_L, d_v}, q.options());
    if (num_chunks <= 0) {
        return out;
    }

    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_SIZE_CONST = 64;
    constexpr int D_K_CONST = 64;
    constexpr int D_V_CONST = 64;

    dim3 grid(num_kv_heads, num_chunks);
    dim3 block(NUM_THREADS);

    using Element = cutlass::half_t;
    using MMAAtom = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMAType = TiledMMA<MMAAtom, Layout<Shape<_4, _1, _1>, Stride<_1, _4, _4>>>;

    using S2RCopyAtomA          = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using S2RCopyAtomB_KMajor   = Copy_Atom<SM75_U32x1_LDSM_N, Element>;
    using S2RCopyAtomB_MNMajor  = Copy_Atom<SM75_U16x2_LDSM_T, Element>;

    constexpr int ROWS_PER_STP = 16;
    using CopyAtom = Copy_Atom<UniversalCopy<uint4>, Element>;
    using TiledCopyG2R = decltype(make_tiled_copy(
        CopyAtom{},
        make_layout(make_shape(Int<ROWS_PER_STP>{}, Int<D_K_CONST / 8>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<8>{}), LayoutRight{})
    ));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    {
        kernel_output_combine_fwd<
            Element,
            NUM_THREADS,
            CHUNK_SIZE_CONST,
            D_K_CONST,
            D_V_CONST,
            MMAAtom,
            TiledMMAType,
            S2RCopyAtomA,
            S2RCopyAtomB_KMajor,
            S2RCopyAtomB_MNMajor,
            TiledCopyG2R
        ><<<grid, block, 0, stream>>>(
            reinterpret_cast<const Element*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(v.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(b.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(w.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(gamma.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(a.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(S.data_ptr<at::Half>()),
            chunk_valid_lens_gpu.data_ptr<int32_t>(),
            chunk_offsets_gpu.data_ptr<int32_t>(),
            reinterpret_cast<Element*>(out.data_ptr<at::Half>()),
            num_q_heads,
            num_kv_heads,
            num_groups,
            num_chunks,
            total_L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Stage 3 - Return Output
    return out;
}
