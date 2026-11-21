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

// GLOBAL KERNELS (BWD OUTPUT COMBINE 3-KERNEL ARCHITECTURE)

// kernel_output_combine_dc_inter_dS
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_K,
    int D_V,
    typename MMAAtom,
    typename TiledMMAType,
    typename S2RCopyAtomA_MNMajor,
    typename S2RCopyAtomB_MNMajor,
    typename S2RCopyAtomA_KMajor,
    typename S2RCopyAtomB_KMajor,
    typename TiledCopyG2R
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_output_combine_dc_inter_dS(
    const T* __restrict__ gQ,                   // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const T* __restrict__ gdO,                  // (num_q_heads, total_L, d_v) [FP16] - Upstream dO
    const T* __restrict__ gS,                   // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent State
    const T* __restrict__ gAlpha,               // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const int32_t* __restrict__ gChunkValidLen, // (num_chunks,) [Int32]
    const int32_t* __restrict__ gChunkOffsets,  // (num_chunks + 1,) [Int32]
    T* __restrict__ gdQ,                        // (num_q_heads, total_L, d_k) [FP16] - Grad Query (reads dq_intra, overwrites dq_total)
    T* __restrict__ gdS,                        // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Grad dS
    float* __restrict__ gDcInter,               // (num_kv_heads, total_L) [FP32] - Grad dc_inter
    int64_t num_q_heads,                        // scalar [int64]
    int64_t num_kv_heads,                       // scalar [int64]
    int64_t num_groups,                         // scalar [int64] - num_q_heads / num_kv_heads
    int64_t num_chunks,                         // scalar [int64]
    int64_t total_L                             // scalar [int64]
) {
    // Stage 1 - Thread & Block Indexing
    int h_kv       = blockIdx.x;
    int curr_chunk = blockIdx.y;
    int tid        = threadIdx.x;

    if (curr_chunk >= num_chunks) return;

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

    // In Layout<Shape<_2, _2, _1>> configuration:
    int warp_m = warp_id % 2;                               // 0..1 (handles rows 0..31 or 32..63)
    int warp_n = warp_id / 2;                               // 0..1 (handles column block 0 or 1)

    // Allocate 32 KB Shared Memory: 4 buffers of 8 KB aligned to 16 bytes
    struct SmemStorage {
        alignas(16) T smem_1[D_V * D_K];        // 8 KB: Region 1 (stationary S)
        alignas(16) T smem_2[CHUNK_SIZE * D_V]; // 8 KB: Region 2 (G, later dc_inter reduction and dS staging)
        alignas(16) T smem_3[CHUNK_SIZE * D_K]; // 8 KB: Region 3 (dq_intra, later stages dq_total)
        alignas(16) T smem_4[CHUNK_SIZE * D_K]; // 8 KB: Region 4 (prefix scan workspace, then holds Q)
    };
    static_assert(sizeof(SmemStorage) == 32768, "SmemStorage must be exactly 32 KB (32768 bytes)");
    __shared__ SmemStorage smem;

    T* s_S  = smem.smem_1;
    T* s_G  = smem.smem_2;
    T* s_Dq = smem.smem_3;
    T* s_Q  = smem.smem_4;

    using SwizzleOp = Swizzle<3, 3, 3>;

    auto smem_layout_flat_k = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_flat_v = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto smem_layout_flat_s = composition(SwizzleOp{}, make_layout(make_shape(Int<D_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    auto sS_flat  = make_tensor(make_smem_ptr(s_S),  smem_layout_flat_s);
    auto sG_flat  = make_tensor(make_smem_ptr(s_G),  smem_layout_flat_v);
    auto sDq_flat = make_tensor(make_smem_ptr(s_Dq), smem_layout_flat_k);
    auto sQ_flat  = make_tensor(make_smem_ptr(s_Q),  smem_layout_flat_k);

    // SMEM Layouts for GEMMs (operand B has shape (N, K) per CuTe MMA convention):
    // GEMM 1: dq_total(i, k) = G(i, v) @ S(v, k) + dq_intra
    // G: A shape (CHUNK_SIZE, D_V), K-major (stride: D_V, 1). LDSM_N
    // S: B shape (D_K, D_V), MN-major (stride: 1, D_K). LDSM_T
    auto smem_layout_A_G1 = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto smem_layout_B_G1 = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<D_V>{}), make_stride(Int<1>{}, Int<D_K>{})));
    auto sA_G1 = make_tensor(make_smem_ptr(s_G), smem_layout_A_G1);
    auto sB_G1 = make_tensor(make_smem_ptr(s_S), smem_layout_B_G1);

    // GEMM 2: dS(v, k) += G^T(v, i) @ Q(i, k)
    // G^T: A shape (D_V, CHUNK_SIZE), MN-major (stride: 1, D_V). LDSM_T
    // Q: B shape (D_K, CHUNK_SIZE), MN-major (stride: 1, D_K). LDSM_T
    auto smem_layout_A_G2 = composition(SwizzleOp{}, make_layout(make_shape(Int<D_V>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_V>{})));
    auto smem_layout_B_G2 = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_K>{})));
    auto sA_G2 = make_tensor(make_smem_ptr(s_G), smem_layout_A_G2);
    auto sB_G2 = make_tensor(make_smem_ptr(s_Q), smem_layout_B_G2);

    // GEMM 3: Q_S(i, v) = Q(i, k) @ S^T(k, v)
    // Q: A shape (CHUNK_SIZE, D_K), K-major (stride: D_K, 1). LDSM_N
    // S^T: B shape (D_V, D_K), K-major (stride: D_K, 1). LDSM_N
    auto smem_layout_A_G3 = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_B_G3 = composition(SwizzleOp{}, make_layout(make_shape(Int<D_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto sA_G3 = make_tensor(make_smem_ptr(s_Q), smem_layout_A_G3);
    auto sB_G3 = make_tensor(make_smem_ptr(s_S), smem_layout_B_G3);

    // Layout C for partition_C
    auto smem_layout_C_dS = composition(SwizzleOp{}, make_layout(make_shape(Int<D_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_C_QS = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto smem_layout_C_dq = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto sC_dS_dummy = make_tensor(make_smem_ptr(s_S),  smem_layout_C_dS);
    auto sC_QS_dummy = make_tensor(make_smem_ptr(s_G),  smem_layout_C_QS);
    auto sDq_mma     = make_tensor(make_smem_ptr(s_Dq), smem_layout_C_dq);
    auto sC_G        = make_tensor(make_smem_ptr(s_G),  smem_layout_C_QS);

    TiledMMAType tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // S2R Tiled Copies
    auto s2r_copy_a_km = make_tiled_copy_A(S2RCopyAtomA_KMajor{},  tiled_mma);
    auto s2r_copy_a_mn = make_tiled_copy_A(S2RCopyAtomA_MNMajor{}, tiled_mma);
    auto s2r_copy_b_km = make_tiled_copy_B(S2RCopyAtomB_KMajor{},  tiled_mma);
    auto s2r_copy_b_mn = make_tiled_copy_B(S2RCopyAtomB_MNMajor{}, tiled_mma);

    auto s2r_thr_copy_a_km = s2r_copy_a_km.get_slice(tid);
    auto s2r_thr_copy_a_mn = s2r_copy_a_mn.get_slice(tid);
    auto s2r_thr_copy_b_km = s2r_copy_b_km.get_slice(tid);
    auto s2r_thr_copy_b_mn = s2r_copy_b_mn.get_slice(tid);

    // Single pair of fragments (tCrA, tCrB) reused across all GEMMs
    auto tCrA = thr_mma.partition_fragment_A(sA_G1);
    auto tCrB = thr_mma.partition_fragment_B(sB_G1);

    auto tCrA_km_view = s2r_thr_copy_a_km.retile_D(tCrA);
    auto tCrA_mn_view = s2r_thr_copy_a_mn.retile_D(tCrA);
    auto tCrB_km_view = s2r_thr_copy_b_km.retile_D(tCrB);
    auto tCrB_mn_view = s2r_thr_copy_b_mn.retile_D(tCrB);

    // Accumulator dS accumulated across query heads g
    auto tCsC_dS = thr_mma.partition_C(sC_dS_dummy);
    auto tCrC_dS = thr_mma.make_fragment_C(tCsC_dS);
    clear(tCrC_dS);

    // Temporary accumulators
    auto tCsC_dq = thr_mma.partition_C(sDq_mma);
    auto tCrC_dq = thr_mma.make_fragment_C(tCsC_dq);
    auto tCrC_QS = thr_mma.make_fragment_C(tCsC_dq);

    auto tCsC_G  = thr_mma.partition_C(sC_G);

    TiledCopyG2R tiled_copy_g2r;
    auto thr_copy_g2r = tiled_copy_g2r.get_thread_slice(tid);

    auto dummy_tile = local_tile(sQ_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(0, 0));
    auto dummy_part = thr_copy_g2r.partition_S(dummy_tile);

    // Register prefetch buffers for Q_next, dO_next, and dq_intra_next
    decltype(make_fragment_like(dummy_part)) r_q_next[NUM_STPS];
    decltype(make_fragment_like(dummy_part)) r_dO_next[NUM_STPS];
    decltype(make_fragment_like(dummy_part)) r_dq_next[NUM_STPS];

    // Stage 1.1 - Prefix Scan for log_alpha_cumsum (temporarily borrows smem_4)
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

    // Cache a_t for g2r load steps: r_a_t_load[NUM_STPS]
    float r_a_t_load[NUM_STPS];
    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        float c_val = (local_row < valid_len) ? s_temp_log_alpha[local_row] : 0.0f;
        r_a_t_load[st] = (local_row < valid_len) ? expf(c_val) : 0.0f;
    }
    __syncthreads(); // Release smem_4 completely to load Q

    // Caching registers for dc_inter (2 repeats m_rep x 2 rows a/b)
    float r_dc_inter_a[2] = {0.0f, 0.0f};
    float r_dc_inter_b[2] = {0.0f, 0.0f};

    // Stage 2 - Prologue: Load S into Region 1 and Group 0 into Regions 2 & 4
    // 1. Load stationary S into Region 1 (smem_1)
    int64_t s_chunk_offset = ((int64_t)h_kv * num_chunks + curr_chunk) * ((int64_t)D_V * D_K);
    const T* p_s_chunk = gS + s_chunk_offset;

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_v = st * ROWS_PER_STP + row_in_step;
        auto sS_step = local_tile(sS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tSsS    = thr_copy_g2r.partition_D(sS_step);
        *reinterpret_cast<uint4*>(&tSsS(0)) = *reinterpret_cast<const uint4*>(&p_s_chunk[local_v * D_K + col_base]);
    }

    // 2. Load group 0: Q and dO, compute G = dO * a_t, write G to Region 2 (smem_2), Q to Region 4 (smem_4)
    int64_t q_offset_0  = ((int64_t)(h_kv * num_groups) * total_L + (int64_t)chunk_start) * D_K;
    int64_t do_offset_0 = ((int64_t)(h_kv * num_groups) * total_L + (int64_t)chunk_start) * D_V;

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_len);
        float a_t_val = r_a_t_load[st];

        decltype(make_fragment_like(dummy_part)) r_q_slice;
        decltype(make_fragment_like(dummy_part)) r_do_slice;
        clear(r_q_slice);
        clear(r_do_slice);

        if (row_valid) {
            *reinterpret_cast<uint4*>(&r_q_slice(0))  = *reinterpret_cast<const uint4*>(gQ + q_offset_0 + local_row * D_K + col_base);
            *reinterpret_cast<uint4*>(&r_do_slice(0)) = *reinterpret_cast<const uint4*>(gdO + do_offset_0 + local_row * D_V + col_base);
        }

        decltype(make_fragment_like(dummy_part)) r_G_slice;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float do_f = half_to_float(r_do_slice(i));
            r_G_slice(i) = float_to_out<T>(do_f * a_t_val);
        }

        auto sQ_step = local_tile(sQ_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto sG_step = local_tile(sG_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto tQsQ    = thr_copy_g2r.partition_D(sQ_step);
        auto tGsG    = thr_copy_g2r.partition_D(sG_step);

        *reinterpret_cast<uint4*>(&tQsQ(0)) = *reinterpret_cast<const uint4*>(&r_q_slice(0));
        *reinterpret_cast<uint4*>(&tGsG(0)) = *reinterpret_cast<const uint4*>(&r_G_slice(0));
    }

    // 3. Load dq_intra group 0 from GMEM gdQ into Region 3 (smem_3) via TiledCopy uint4
    int64_t q_out_base_0 = ((int64_t)(h_kv * num_groups) * total_L + (int64_t)chunk_start) * D_K;
    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_len);
        decltype(make_fragment_like(dummy_part)) r_dq_0;
        clear(r_dq_0);
        if (row_valid) {
            *reinterpret_cast<uint4*>(&r_dq_0(0)) = *reinterpret_cast<const uint4*>(gdQ + q_out_base_0 + local_row * D_K + col_base);
        }
        auto sDq_step = local_tile(sDq_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tDqsDq   = thr_copy_g2r.partition_D(sDq_step);
        *reinterpret_cast<uint4*>(&tDqsDq(0)) = *reinterpret_cast<const uint4*>(&r_dq_0(0));
    }
    __syncthreads(); // [SYNC 0] (RAW: S in Region 1, G in Region 2, dq_intra in Region 3, Q in Region 4)

    // Stage 3 - Sequential Loop over Query Heads in GQA Group (g = 0..num_groups - 1)
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t curr_h_q   = (int64_t)h_kv * num_groups + g;
        int64_t q_out_base = (curr_h_q * total_L + (int64_t)chunk_start) * D_K;

        // Step 3.1 - GEMM 1: dq_total = G @ S + dq_intra & Stage back to Region 3
        // Load dq_intra from Region 3 (sDq_mma) into accumulator tCrC_dq
        #pragma unroll
        for (int i = 0; i < size(tCrC_dq); ++i) {
            tCrC_dq(i) = __half2float(to_half_val(tCsC_dq(i)));
        }

        // LDSM_N reads G from Region 2, LDSM_T reads S from Region 1
        cute::copy(s2r_copy_a_km, s2r_thr_copy_a_km.partition_S(sA_G1), tCrA_km_view);
        cute::copy(s2r_copy_b_mn, s2r_thr_copy_b_mn.partition_S(sB_G1), tCrB_mn_view);
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_dq); // tCrC_dq = G @ S + dq_intra

        __syncthreads(); // [SYNC 1] (WAR: all warps have read dq_intra from Region 3)

        // Write tCrC_dq (dq_total, cast to half2) back to Region 3 (sDq_flat with Swizzle)
        #pragma unroll
        for (int m_rep = 0; m_rep < 2; ++m_rep) {
            int row_a = m_rep * 32 + warp_m * 16 + quad_id;
            int row_b = row_a + 8;
            #pragma unroll
            for (int n_rep = 0; n_rep < 4; ++n_rep) {
                int col_0 = n_rep * 16 + warp_n * 8 + lane_in_quad * 2;
                half2 val_a = __floats2half2_rn(
                    tCrC_dq(make_coord(0, 0), m_rep, n_rep),
                    tCrC_dq(make_coord(1, 0), m_rep, n_rep)
                );
                half2 val_b = __floats2half2_rn(
                    tCrC_dq(make_coord(0, 1), m_rep, n_rep),
                    tCrC_dq(make_coord(1, 1), m_rep, n_rep)
                );
                reinterpret_cast<half2*>(&sDq_flat(row_a, col_0))[0] = val_a;
                reinterpret_cast<half2*>(&sDq_flat(row_b, col_0))[0] = val_b;
            }
        }

        // Step 3.2 - GEMM 2: dS += G^T @ Q, Prefetch next iteration & GEMM 3: Q_S = Q @ S^T
        // LDSM_T reads G^T from Region 2, LDSM_T reads Q from Region 4
        cute::copy(s2r_copy_a_mn, s2r_thr_copy_a_mn.partition_S(sA_G2), tCrA_mn_view);
        cute::copy(s2r_copy_b_mn, s2r_thr_copy_b_mn.partition_S(sB_G2), tCrB_mn_view);
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_dS); // accumulate into tCrC_dS

        // Prefetch g2r Q_next, dO_next, and dq_intra_next for next group (if g + 1 < num_groups)
        if (g + 1 < num_groups) {
            int64_t next_h_q       = (int64_t)h_kv * num_groups + (g + 1);
            int64_t q_offset_next  = (next_h_q * total_L + (int64_t)chunk_start) * D_K;
            int64_t do_offset_next = (next_h_q * total_L + (int64_t)chunk_start) * D_V;

            #pragma unroll
            for (int st = 0; st < NUM_STPS; ++st) {
                int local_row = st * ROWS_PER_STP + row_in_step;
                bool row_valid = (local_row < valid_len);
                clear(r_q_next[st]);
                clear(r_dO_next[st]);
                clear(r_dq_next[st]);
                if (row_valid) {
                    *reinterpret_cast<uint4*>(&r_q_next[st](0))  = *reinterpret_cast<const uint4*>(gQ + q_offset_next + local_row * D_K + col_base);
                    *reinterpret_cast<uint4*>(&r_dO_next[st](0)) = *reinterpret_cast<const uint4*>(gdO + do_offset_next + local_row * D_V + col_base);
                    *reinterpret_cast<uint4*>(&r_dq_next[st](0)) = *reinterpret_cast<const uint4*>(gdQ + q_offset_next + local_row * D_K + col_base);
                }
            }
        }

        // LDSM_N reads Q from Region 4, LDSM_N reads S^T from Region 1
        clear(tCrC_QS);
        cute::copy(s2r_copy_a_km, s2r_thr_copy_a_km.partition_S(sA_G3), tCrA_km_view);
        cute::copy(s2r_copy_b_km, s2r_thr_copy_b_km.partition_S(sB_G3), tCrB_km_view);
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_QS); // Q_S = Q @ S^T

        // Epilogue dc_inter
        #pragma unroll
        for (int m_rep = 0; m_rep < 2; ++m_rep) {
            #pragma unroll
            for (int n_rep = 0; n_rep < 4; ++n_rep) {
                r_dc_inter_a[m_rep] += tCrC_QS(make_coord(0, 0), m_rep, n_rep) * half_to_float(tCsC_G(make_coord(0, 0), m_rep, n_rep));
                r_dc_inter_a[m_rep] += tCrC_QS(make_coord(1, 0), m_rep, n_rep) * half_to_float(tCsC_G(make_coord(1, 0), m_rep, n_rep));
                r_dc_inter_b[m_rep] += tCrC_QS(make_coord(0, 1), m_rep, n_rep) * half_to_float(tCsC_G(make_coord(0, 1), m_rep, n_rep));
                r_dc_inter_b[m_rep] += tCrC_QS(make_coord(1, 1), m_rep, n_rep) * half_to_float(tCsC_G(make_coord(1, 1), m_rep, n_rep));
            }
        }

        __syncthreads(); // [SYNC 2] (RAW: dq_total written to Region 3; WAR: all warps finished reading G from Region 2 and Q from Region 4)

        // Step 3.3 - Write dq_total from Region 3 to GMEM (uint4) & Update Regions 2, 3, 4
        // Use TiledCopy uint4 to read dq_total from Region 3 and write 128-bit stores to GMEM gdQ
        #pragma unroll
        for (int st = 0; st < NUM_STPS; ++st) {
            int local_row = st * ROWS_PER_STP + row_in_step;
            if (local_row < valid_len) {
                *reinterpret_cast<uint4*>(gdQ + q_out_base + local_row * D_K + col_base) =
                    *reinterpret_cast<const uint4*>(&sDq_flat(local_row, col_base));
            }
            if (g + 1 < num_groups) {
                // Overwrite r_dq_next into Region 3
                auto sDq_step = local_tile(sDq_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                auto tDqsDq   = thr_copy_g2r.partition_D(sDq_step);
                *reinterpret_cast<uint4*>(&tDqsDq(0)) = *reinterpret_cast<const uint4*>(&r_dq_next[st](0));
            }
        }

        if (g + 1 < num_groups) {
            #pragma unroll
            for (int st = 0; st < NUM_STPS; ++st) {
                float a_t_val = r_a_t_load[st];
                decltype(make_fragment_like(dummy_part)) r_G_next;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float do_f = half_to_float(r_dO_next[st](i));
                    r_G_next(i) = float_to_out<T>(do_f * a_t_val);
                }
                auto sG_step = local_tile(sG_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
                auto sQ_step = local_tile(sQ_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                auto tGsG    = thr_copy_g2r.partition_D(sG_step);
                auto tQsQ    = thr_copy_g2r.partition_D(sQ_step);

                *reinterpret_cast<uint4*>(&tGsG(0)) = *reinterpret_cast<const uint4*>(&r_G_next(0));
                *reinterpret_cast<uint4*>(&tQsQ(0)) = *reinterpret_cast<const uint4*>(&r_q_next[st](0));
            }

            __syncthreads(); // [SYNC 3] (RAW: G_next in Region 2, dq_intra_next in Region 3, Q_next in Region 4 ready for g + 1)
        }
    }

    // Stage 4 - Final Epilogue & GMEM Stores
    __syncthreads(); // Ensure all warps have finished loop g

    // 1. Quad reduction (warp shuffle) for r_dc_inter
    #pragma unroll
    for (int m_rep = 0; m_rep < 2; ++m_rep) {
        #pragma unroll
        for (int mask = 2; mask > 0; mask /= 2) {
            r_dc_inter_a[m_rep] += __shfl_xor_sync(0xffffffff, r_dc_inter_a[m_rep], mask);
            r_dc_inter_b[m_rep] += __shfl_xor_sync(0xffffffff, r_dc_inter_b[m_rep], mask);
        }
    }

    // 2. Inter-warp reduction via Region 2 (smem_2) between 2 warps in the same row in Shape<_2, _2, _1>
    float* s_inter_reduce = reinterpret_cast<float*>(smem.smem_2);
    if (lane_in_quad == 0 && warp_n == 1) {
        #pragma unroll
        for (int m_rep = 0; m_rep < 2; ++m_rep) {
            int row_a = m_rep * 32 + warp_m * 16 + quad_id;
            int row_b = row_a + 8;
            s_inter_reduce[row_a] = r_dc_inter_a[m_rep];
            s_inter_reduce[row_b] = r_dc_inter_b[m_rep];
        }
    }
    __syncthreads();

    if (lane_in_quad == 0 && warp_n == 0) {
        #pragma unroll
        for (int m_rep = 0; m_rep < 2; ++m_rep) {
            int row_a = m_rep * 32 + warp_m * 16 + quad_id;
            int row_b = row_a + 8;
            float total_a = r_dc_inter_a[m_rep] + s_inter_reduce[row_a];
            float total_b = r_dc_inter_b[m_rep] + s_inter_reduce[row_b];

            if (row_a < valid_len) {
                gDcInter[kv_head_offset + chunk_start + row_a] = total_a;
            } else if (chunk_start + row_a < total_L) {
                gDcInter[kv_head_offset + chunk_start + row_a] = 0.0f;
            }
            if (row_b < valid_len) {
                gDcInter[kv_head_offset + chunk_start + row_b] = total_b;
            } else if (chunk_start + row_b < total_L) {
                gDcInter[kv_head_offset + chunk_start + row_b] = 0.0f;
            }
        }
    }
    __syncthreads(); // Release SMEM to stage dS

    // 3. Stage dS into Region 2 (smem.smem_2) via CuTe Tensor View with Swizzle<3, 3, 3>
    auto sdS_flat = make_tensor(make_smem_ptr(smem.smem_2), smem_layout_flat_s);

    #pragma unroll
    for (int m_rep = 0; m_rep < 2; ++m_rep) {
        int row_a = m_rep * 32 + warp_m * 16 + quad_id;
        int row_b = row_a + 8;

        #pragma unroll
        for (int n_rep = 0; n_rep < 4; ++n_rep) {
            int col_0 = n_rep * 16 + warp_n * 8 + lane_in_quad * 2;

            half2 val_a = __floats2half2_rn(
                tCrC_dS(make_coord(0, 0), m_rep, n_rep),
                tCrC_dS(make_coord(1, 0), m_rep, n_rep)
            );
            half2 val_b = __floats2half2_rn(
                tCrC_dS(make_coord(0, 1), m_rep, n_rep),
                tCrC_dS(make_coord(1, 1), m_rep, n_rep)
            );

            reinterpret_cast<half2*>(&sdS_flat(row_a, col_0))[0] = val_a;
            reinterpret_cast<half2*>(&sdS_flat(row_b, col_0))[0] = val_b;
        }
    }
    __syncthreads(); // [SYNC RAW: dS fully staged into SMEM]

    // 4. Store dS to GMEM via 128-bit vectorized stores through swizzled SMEM view
    T* gdS_chunk = gdS + s_chunk_offset;

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_v = st * ROWS_PER_STP + row_in_step; // 0..63
        *reinterpret_cast<uint4*>(&gdS_chunk[local_v * D_K + col_base]) =
            *reinterpret_cast<const uint4*>(&sdS_flat(local_v, col_base));
    }
}

// Dynamic Shared Memory Structure for kernel_output_combine_dq_dk_dv (57,600 bytes = 56.25 KB)
template <typename T, int CHUNK_SIZE, int D_K, int D_V>
struct SmemStorageKernel1 {
    alignas(16) T smem_1[CHUNK_SIZE * D_V];        // 8 KB: Region 1 (smem_dO - holds dO for current query group g)
    alignas(16) T smem_2[CHUNK_SIZE * D_K];        // 8 KB: Region 2 (smem_Q - holds Q for current query group g; stages dk in epilogue)
    alignas(16) T smem_3[CHUNK_SIZE * D_V];        // 8 KB: Region 3 (smem_V - stationary V_new throughout kernel)
    alignas(16) T smem_4[CHUNK_SIZE * D_K];        // 8 KB: Region 4 (smem_K - stationary K_scaled throughout kernel)
    alignas(16) T smem_5[CHUNK_SIZE * CHUNK_SIZE]; // 8 KB: Region 5 (smem_work - transpose buffer dR^T -> dR & dq_intra staging)
    alignas(16) T smem_6[CHUNK_SIZE * CHUNK_SIZE]; // 8 KB: Region 6 (smem_dD - SMEM FP16 accumulator for dD^T)
    alignas(16) T smem_7[CHUNK_SIZE * D_V];        // 8 KB: Region 7 (smem_dv - SMEM FP16 accumulator for dv)
    alignas(16) float s_c[CHUNK_SIZE];             // 256 bytes: log_alpha_cumsum
};

// kernel_output_combine_dq_dk_dv
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_K,
    int D_V,
    typename MMAAtom,
    typename TiledMMAType,
    typename S2RCopyAtomA_KMajor,
    typename S2RCopyAtomA_MNMajor,
    typename S2RCopyAtomB_KMajor,
    typename S2RCopyAtomB_MNMajor,
    typename TiledCopyG2R
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_output_combine_dq_dk_dv(
    const T* __restrict__ gQ,                   // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const T* __restrict__ gdO,                  // (num_q_heads, total_L, d_v) [FP16] - Upstream dO
    const T* __restrict__ gK,                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const T* __restrict__ gV,                   // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const T* __restrict__ gW_gate,              // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const T* __restrict__ gGamma,               // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const T* __restrict__ gAlpha,               // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const int32_t* __restrict__ gChunkValidLen, // (num_chunks,) [Int32]
    const int32_t* __restrict__ gChunkOffsets,  // (num_chunks + 1,) [Int32]
    T* __restrict__ gdQ,                        // (num_q_heads, total_L, d_k) [FP16] - Output grad Query (dq_intra)
    T* __restrict__ gdK,                        // (num_kv_heads, total_L, d_k) [FP16] - Output grad Key
    T* __restrict__ gdV,                        // (num_kv_heads, total_L, d_v) [FP16] - Output grad Value
    T* __restrict__ gdD,                        // (num_kv_heads, num_chunks, chunk_size, chunk_size) [FP16] - Inter grad dD^T
    int64_t num_q_heads,                        // scalar [int64]
    int64_t num_kv_heads,                       // scalar [int64]
    int64_t num_groups,                         // scalar [int64] - num_q_heads / num_kv_heads
    int64_t num_chunks,                         // scalar [int64]
    int64_t total_L                             // scalar [int64]
) {
    // Stage 1 - Thread & Block Indexing
    int h_kv       = blockIdx.x;
    int curr_chunk = blockIdx.y;
    int tid        = threadIdx.x;

    if (curr_chunk >= num_chunks) return;

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

    // Allocate Dynamic Shared Memory of 57,600 bytes (56.25 KB)
    using SmemStorage = SmemStorageKernel1<T, CHUNK_SIZE, D_K, D_V>;
    static_assert(sizeof(SmemStorage) == 57600, "SmemStorage must be exactly 57,600 bytes (56.25 KB)");
    extern __shared__ __align__(16) uint8_t smem_raw[];
    SmemStorage& smem = *reinterpret_cast<SmemStorage*>(smem_raw);

    T* s_dO   = smem.smem_1; // Region 1: dO of current group g (8 KB)
    T* s_Q    = smem.smem_2; // Region 2: Q of current group g; stages dk in epilogue (8 KB)
    T* s_V    = smem.smem_3; // Region 3: Stationary V_new throughout kernel (8 KB)
    T* s_K    = smem.smem_4; // Region 4: Stationary K_scaled throughout kernel (8 KB)
    T* s_work = smem.smem_5; // Region 5: Work buffer for dR^T -> dR transpose & dq_intra staging (8 KB)
    T* s_dD   = smem.smem_6; // Region 6: smem_dD FP16 accumulator for dD^T (8 KB)
    T* s_dv   = smem.smem_7; // Region 7: smem_dv FP16 accumulator for dv (8 KB)

    using SwizzleOp = Swizzle<3, 3, 3>;

    auto smem_layout_flat_k = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_flat_v = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto smem_layout_flat_c = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{})));

    auto sdO_flat    = make_tensor(make_smem_ptr(s_dO),   smem_layout_flat_v); // Region 1
    auto sQ_flat     = make_tensor(make_smem_ptr(s_Q),    smem_layout_flat_k); // Region 2
    auto sV_flat     = make_tensor(make_smem_ptr(s_V),    smem_layout_flat_v); // Region 3
    auto sK_flat     = make_tensor(make_smem_ptr(s_K),    smem_layout_flat_k); // Region 4
    auto s_work_flat = make_tensor(make_smem_ptr(s_work), smem_layout_flat_c); // Region 5 flat
    auto s_dD_flat   = make_tensor(make_smem_ptr(s_dD),   smem_layout_flat_c); // Region 6 accumulator dD^T
    auto s_dv_flat   = make_tensor(make_smem_ptr(s_dv),   smem_layout_flat_v); // Region 7 accumulator dv

    // SMEM Layouts for GEMMs (operand B has shape (N, K) per CuTe MMA convention)
    // GEMM 1: dA^T(j, i) = V_new(j, v) @ dO^T(v, i)
    // V_new: A shape (CHUNK_SIZE, D_V), K-major (stride: D_V, 1). LDSM_N
    // dO^T: B shape (CHUNK_SIZE, D_V), K-major (stride: D_V, 1). LDSM_N
    auto smem_layout_A_G1_V  = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto smem_layout_B_G1_dO = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto sA_G1_V  = make_tensor(make_smem_ptr(s_V),  smem_layout_A_G1_V);
    auto sB_G1_dO = make_tensor(make_smem_ptr(s_dO), smem_layout_B_G1_dO);

    // GEMM 2: R^T(j, i) = K_scaled(j, k) @ Q^T(k, i)
    // K_scaled: A shape (CHUNK_SIZE, D_K), K-major (stride: D_K, 1). LDSM_N
    // Q^T: B shape (CHUNK_SIZE, D_K), K-major (stride: D_K, 1). LDSM_N
    auto smem_layout_A_G2_K = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_B_G2_Q = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto sA_G2_K = make_tensor(make_smem_ptr(s_K), smem_layout_A_G2_K);
    auto sB_G2_Q = make_tensor(make_smem_ptr(s_Q), smem_layout_B_G2_Q);

    // GEMM 3: dk(j, k) += dR^T(j, i) @ Q(i, k)
    // dR^T resides in register fragment tCrA
    // Q: B shape (D_K, CHUNK_SIZE), MN-major (stride: 1, D_K). LDSM_T
    auto smem_layout_B_G3_Q = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_K>{})));
    auto sB_G3_Q = make_tensor(make_smem_ptr(s_Q), smem_layout_B_G3_Q);

    // GEMM 4: dv(j, v) += A^T(j, i) @ dO(i, v)
    // A^T resides in register fragment tCrA (after GEMM 2 epilogue)
    // dO: B shape (D_V, CHUNK_SIZE), MN-major (stride: 1, D_V). LDSM_T
    auto smem_layout_B_G4_dO = composition(SwizzleOp{}, make_layout(make_shape(Int<D_V>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_V>{})));
    auto sB_G4_dO = make_tensor(make_smem_ptr(s_dO), smem_layout_B_G4_dO);

    // GEMM 5: dq_intra(i, k) = dR(i, j) @ K_scaled(j, k)
    // dR: A shape (CHUNK_SIZE, CHUNK_SIZE), MN-major (stride: 1, CHUNK_SIZE). LDSM_T from Region 5 (s_work)
    // K_scaled: B shape (D_K, CHUNK_SIZE), MN-major (stride: 1, D_K). LDSM_T from Region 4 (s_K)
    auto smem_layout_A_G5_dR = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<CHUNK_SIZE>{})));
    auto smem_layout_B_G5_K  = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_K>{})));
    auto sA_G5_dR = make_tensor(make_smem_ptr(s_work), smem_layout_A_G5_dR); // Read transposed dR from Region 5
    auto sB_G5_K  = make_tensor(make_smem_ptr(s_K),    smem_layout_B_G5_K);

    // Layout C dummy (64x64)
    auto smem_layout_C = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{})));
    auto sC_dummy = make_tensor(make_smem_ptr(s_work), smem_layout_C);

    TiledMMAType tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // S2R Tiled Copies
    auto s2r_copy_a_km = make_tiled_copy_A(S2RCopyAtomA_KMajor{},  tiled_mma);
    auto s2r_copy_a_mn = make_tiled_copy_A(S2RCopyAtomA_MNMajor{}, tiled_mma);
    auto s2r_copy_b_km = make_tiled_copy_B(S2RCopyAtomB_KMajor{},  tiled_mma);
    auto s2r_copy_b_mn = make_tiled_copy_B(S2RCopyAtomB_MNMajor{}, tiled_mma);

    auto s2r_thr_copy_a_km = s2r_copy_a_km.get_slice(tid);
    auto s2r_thr_copy_a_mn = s2r_copy_a_mn.get_slice(tid);
    auto s2r_thr_copy_b_km = s2r_copy_b_km.get_slice(tid);
    auto s2r_thr_copy_b_mn = s2r_copy_b_mn.get_slice(tid);

    // Single pair of fragments (tCrA, tCrB) reused across all GEMMs
    auto tCrA = thr_mma.partition_fragment_A(sA_G1_V);
    auto tCrB = thr_mma.partition_fragment_B(sB_G1_dO);

    auto tCrA_km_view = s2r_thr_copy_a_km.retile_D(tCrA);
    auto tCrA_mn_view = s2r_thr_copy_a_mn.retile_D(tCrA);
    auto tCrB_km_view = s2r_thr_copy_b_km.retile_D(tCrB);
    auto tCrB_mn_view = s2r_thr_copy_b_mn.retile_D(tCrB);

    // Auxiliary fragment r_dAT_h (16 regs, 32 halves FP16) stores dA^T and then dR^T
    decltype(tCrA) r_dAT_h;

    // FP32 Accumulators: Single permanent accumulator tCrC_dk
    auto tCrC_dk = thr_mma.partition_fragment_C(sC_dummy);
    clear(tCrC_dk);

    // Single temporary accumulator (32 registers FP32)
    // Reused for: GEMM 1 (dA^T), GEMM 2 (R^T), and GEMM 5 (dq_intra)
    auto tCrC_tmp = thr_mma.partition_fragment_C(sC_dummy);

    TiledCopyG2R tiled_copy_g2r;
    auto thr_copy_g2r = tiled_copy_g2r.get_thread_slice(tid);

    auto dummy_tile = local_tile(sK_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(0, 0));
    auto dummy_part = thr_copy_g2r.partition_S(dummy_tile);

    // Single staging register buffer r_staging (16 registers FP16)
    // Reused for prologue and prefetching dO_next
    decltype(make_fragment_like(dummy_part)) r_staging[NUM_STPS];

    // Stage 1.1 - 2-Warp Prefix Scan for log_alpha_cumsum written to smem.s_c[CHUNK_SIZE]
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
        smem.s_c[tid] = alpha_val;
    }
    __syncthreads();

    if (tid >= 32 && tid < CHUNK_SIZE) {
        float warp0_sum = smem.s_c[31];
        smem.s_c[tid] += warp0_sum;
    }
    __syncthreads();

    // Initialize Region 6 (smem_dD) = 0 and Region 7 (smem_dv) = 0 via uint4(0)
    uint4 zero_u4 = make_uint4(0, 0, 0, 0);
    uint4* p_dD_u4 = reinterpret_cast<uint4*>(s_dD);
    uint4* p_dv_u4 = reinterpret_cast<uint4*>(s_dv);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        p_dD_u4[i * 128 + tid] = zero_u4;
        p_dv_u4[i * 128 + tid] = zero_u4;
    }

    // Stage 2 - Prologue: Recompute K_scaled (Region 4), V_new (Region 3), dO_0 (Region 1), Q_0 (Region 2)
    int64_t k_offset_curr  = kv_head_offset * D_K + (int64_t)chunk_start * D_K;
    int64_t v_offset_curr  = kv_head_offset * D_V + (int64_t)chunk_start * D_V;
    const T* p_gamma_chunk = gGamma + kv_head_offset;

    // 1. Recompute K_scaled and store into Region 4 (smem_4)
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

    // 2. Recompute V_new and store into Region 3 (smem_3)
    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_len);
        int64_t token_idx = (int64_t)chunk_start + local_row;

        decltype(make_fragment_like(dummy_part)) rV;
        decltype(make_fragment_like(dummy_part)) rW;
        clear(rV);
        clear(rW);
        if (row_valid) {
            *reinterpret_cast<uint4*>(&rV(0)) = *reinterpret_cast<const uint4*>(gV + v_offset_curr + local_row * D_V + col_base);
            *reinterpret_cast<uint4*>(&rW(0)) = *reinterpret_cast<const uint4*>(gW_gate + v_offset_curr + local_row * D_V + col_base);
        }

        float sqrt_gamma_v = 0.0f;
        if (row_valid) {
            __half g_val = to_half_val(p_gamma_chunk[token_idx]);
            __half sp    = fast_softplus_h(g_val);
            sqrt_gamma_v = sqrtf(fmaxf(__half2float(sp), 0.0f) + 1e-4f);
        }

        decltype(make_fragment_like(dummy_part)) rV_new;
        clear(rV_new);
        if (row_valid) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v_val = half_to_float(rV(i));
                float w_val = half_to_float(fast_sigmoid_h(to_half_val(rW(i))));
                rV_new(i) = float_to_out<T>(w_val * (v_val * sqrt_gamma_v));
            }
        }

        auto sV_step = local_tile(sV_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto tVsV    = thr_copy_g2r.partition_D(sV_step);
        *reinterpret_cast<uint4*>(&tVsV(0)) = *reinterpret_cast<const uint4*>(&rV_new(0));
    }

    // 3. Load dO group 0 into Region 1 (smem_1)
    int64_t do_offset_0 = ((int64_t)(h_kv * num_groups) * total_L + (int64_t)chunk_start) * D_V;
    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_len);
        clear(r_staging[st]);
        if (row_valid) {
            *reinterpret_cast<uint4*>(&r_staging[st](0)) = *reinterpret_cast<const uint4*>(gdO + do_offset_0 + local_row * D_V + col_base);
        }
        auto sdO_step = local_tile(sdO_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto tdOsdO   = thr_copy_g2r.partition_D(sdO_step);
        *reinterpret_cast<uint4*>(&tdOsdO(0)) = *reinterpret_cast<const uint4*>(&r_staging[st](0));
    }

    // 4. Load Q group 0 into Region 2 (smem_2)
    int64_t q_offset_0 = ((int64_t)(h_kv * num_groups) * total_L + (int64_t)chunk_start) * D_K;
    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_len);
        clear(r_staging[st]);
        if (row_valid) {
            *reinterpret_cast<uint4*>(&r_staging[st](0)) = *reinterpret_cast<const uint4*>(gQ + q_offset_0 + local_row * D_K + col_base);
        }
        auto sQ_step = local_tile(sQ_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tQsQ    = thr_copy_g2r.partition_D(sQ_step);
        *reinterpret_cast<uint4*>(&tQsQ(0)) = *reinterpret_cast<const uint4*>(&r_staging[st](0));
    }
    __syncthreads(); // [SYNC 0] (RAW: dO in Region 1, Q in Region 2, V_new in Region 3, K_scaled in Region 4)

    // Stage 3 - Sequential Loop over Query Heads in GQA Group (g = 0..num_groups - 1)
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t curr_h_q = (int64_t)h_kv * num_groups + g;

        // Step 3.1 - GEMM 1: dA^T = V_new @ dO^T & GEMM 2: R^T = K_scaled @ Q^T
        // (Reuses tCrC_tmp and performs numerically stable exp)
        // GEMM 1: V_new (Region 3) @ dO^T (Region 1)
        cute::copy(s2r_copy_a_km, s2r_thr_copy_a_km.partition_S(sA_G1_V),  tCrA_km_view);
        cute::copy(s2r_copy_b_km, s2r_thr_copy_b_km.partition_S(sB_G1_dO), tCrB_km_view);
        clear(tCrC_tmp); // tCrC_tmp receives dA^T
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_tmp);

        // Pack dA^T immediately from tCrC_tmp into r_dAT_h (FP16) to release tCrC_tmp
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            r_dAT_h(i) = float_to_out<T>(tCrC_tmp(i));
        }

        // GEMM 2: K_scaled (Region 4) @ Q^T (Region 2) - Reuses tCrC_tmp to receive R^T
        cute::copy(s2r_copy_a_km, s2r_thr_copy_a_km.partition_S(sA_G2_K), tCrA_km_view);
        cute::copy(s2r_copy_b_km, s2r_thr_copy_b_km.partition_S(sB_G2_Q), tCrB_km_view);
        clear(tCrC_tmp); // tCrC_tmp receives R^T
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_tmp);

        // In-place epilogue: numerically stable exp and accumulate dD^T into Region 6
        float c_j_a = (row_a < valid_len) ? smem.s_c[row_a] : 0.0f;
        float c_j_b = (row_b < valid_len) ? smem.s_c[row_b] : 0.0f;

        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int target_col = n * 8 + lane_in_quad * 2;
            int col_0 = target_col + 0;
            int col_1 = target_col + 1;

            float c_i_0 = (col_0 < valid_len) ? smem.s_c[col_0] : 0.0f;
            float c_i_1 = (col_1 < valid_len) ? smem.s_c[col_1] : 0.0f;

            // E = exp(min(c_i - c_j, 0)) directly in FP32, preventing NaN/Inf overflow
            float e_a0 = (col_0 >= row_a && row_a < valid_len && col_0 < valid_len) ? expf(fminf(c_i_0 - c_j_a, 0.0f)) : 0.0f;
            float e_a1 = (col_1 >= row_a && row_a < valid_len && col_1 < valid_len) ? expf(fminf(c_i_1 - c_j_a, 0.0f)) : 0.0f;
            float e_b0 = (col_0 >= row_b && row_b < valid_len && col_0 < valid_len) ? expf(fminf(c_i_0 - c_j_b, 0.0f)) : 0.0f;
            float e_b1 = (col_1 >= row_b && row_b < valid_len && col_1 < valid_len) ? expf(fminf(c_i_1 - c_j_b, 0.0f)) : 0.0f;

            // dA^T from r_dAT_h (FP16 -> float)
            float dat_0 = half_to_float(r_dAT_h(4 * n + 0));
            float dat_1 = half_to_float(r_dAT_h(4 * n + 1));
            float dat_2 = half_to_float(r_dAT_h(4 * n + 2));
            float dat_3 = half_to_float(r_dAT_h(4 * n + 3));

            // R^T from tCrC_tmp (FP32)
            float rt_0 = tCrC_tmp(4 * n + 0);
            float rt_1 = tCrC_tmp(4 * n + 1);
            float rt_2 = tCrC_tmp(4 * n + 2);
            float rt_3 = tCrC_tmp(4 * n + 3);

            // Same-Thread Read-Modify-Write accumulate dD^T = dA^T * R^T into Region 6 (smem_dD)
            half2 curr_dD_a = reinterpret_cast<half2*>(&s_dD_flat(row_a, target_col))[0];
            float new_dD_a0 = __low2float(curr_dD_a) + dat_0 * rt_0;
            float new_dD_a1 = __high2float(curr_dD_a) + dat_1 * rt_1;
            reinterpret_cast<half2*>(&s_dD_flat(row_a, target_col))[0] = __floats2half2_rn(new_dD_a0, new_dD_a1);

            half2 curr_dD_b = reinterpret_cast<half2*>(&s_dD_flat(row_b, target_col))[0];
            float new_dD_b0 = __low2float(curr_dD_b) + dat_2 * rt_2;
            float new_dD_b1 = __high2float(curr_dD_b) + dat_3 * rt_3;
            reinterpret_cast<half2*>(&s_dD_flat(row_b, target_col))[0] = __floats2half2_rn(new_dD_b0, new_dD_b1);

            // A^T = R^T * E^T overwrite into tCrA
            tCrA(4 * n + 0) = float_to_out<T>(rt_0 * e_a0);
            tCrA(4 * n + 1) = float_to_out<T>(rt_1 * e_a1);
            tCrA(4 * n + 2) = float_to_out<T>(rt_2 * e_b0);
            tCrA(4 * n + 3) = float_to_out<T>(rt_3 * e_b1);

            // dR^T = dA^T * E^T overwrite into r_dAT_h
            r_dAT_h(4 * n + 0) = float_to_out<T>(dat_0 * e_a0);
            r_dAT_h(4 * n + 1) = float_to_out<T>(dat_1 * e_a1);
            r_dAT_h(4 * n + 2) = float_to_out<T>(dat_2 * e_b0);
            r_dAT_h(4 * n + 3) = float_to_out<T>(dat_3 * e_b1);
        }

        // Step 3.2 - Write dR^T to Region 5, Accumulate dv into Region 7, and Compute dk
        // 1. Write dR^T (half2) to Region 5 (smem_work with Swizzle)
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int target_col = n * 8 + lane_in_quad * 2;
            __half val_a0 = to_half_val(r_dAT_h(4 * n + 0));
            __half val_a1 = to_half_val(r_dAT_h(4 * n + 1));
            __half val_b0 = to_half_val(r_dAT_h(4 * n + 2));
            __half val_b1 = to_half_val(r_dAT_h(4 * n + 3));
            reinterpret_cast<half2*>(&s_work_flat(row_a, target_col))[0] = make_half2(val_a0, val_a1);
            reinterpret_cast<half2*>(&s_work_flat(row_b, target_col))[0] = make_half2(val_b0, val_b1);
        }

        // 2. Load accumulated dv from Region 7 (s_dv_flat) into tCrC_tmp (FP32)
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int target_col = n * 8 + lane_in_quad * 2;
            half2 curr_dv_a = reinterpret_cast<const half2*>(&s_dv_flat(row_a, target_col))[0];
            half2 curr_dv_b = reinterpret_cast<const half2*>(&s_dv_flat(row_b, target_col))[0];
            tCrC_tmp(4 * n + 0) = __low2float(curr_dv_a);
            tCrC_tmp(4 * n + 1) = __high2float(curr_dv_a);
            tCrC_tmp(4 * n + 2) = __low2float(curr_dv_b);
            tCrC_tmp(4 * n + 3) = __high2float(curr_dv_b);
        }

        // Read dO from Region 1 (MN-major / LDSM_T) into tCrB -> GEMM 4: tCrC_tmp += tCrA (A^T) @ tCrB (dO)
        cute::copy(s2r_copy_b_mn, s2r_thr_copy_b_mn.partition_S(sB_G4_dO), tCrB_mn_view);
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_tmp);

        // Write updated dv (cast to half2) back to Region 7 (s_dv_flat) (Same-thread RMW)
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int target_col = n * 8 + lane_in_quad * 2;
            half2 new_dv_a = __floats2half2_rn(tCrC_tmp(4 * n + 0), tCrC_tmp(4 * n + 1));
            half2 new_dv_b = __floats2half2_rn(tCrC_tmp(4 * n + 2), tCrC_tmp(4 * n + 3));
            reinterpret_cast<half2*>(&s_dv_flat(row_a, target_col))[0] = new_dv_a;
            reinterpret_cast<half2*>(&s_dv_flat(row_b, target_col))[0] = new_dv_b;
        }
        // Release tCrC_tmp and tCrA

        // 3. GEMM 3: dk += dR^T @ Q
        // Copy dR^T from r_dAT_h to tCrA
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            tCrA(i) = r_dAT_h(i);
        }

        // Read Q from Region 2 (MN-major / LDSM_T) into tCrB
        cute::copy(s2r_copy_b_mn, s2r_thr_copy_b_mn.partition_S(sB_G3_Q), tCrB_mn_view);
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_dk);

        // Prefetch dO_next into single buffer r_staging
        if (g + 1 < num_groups) {
            int64_t next_h_q       = (int64_t)h_kv * num_groups + (g + 1);
            int64_t do_offset_next = (next_h_q * total_L + (int64_t)chunk_start) * D_V;
            #pragma unroll
            for (int st = 0; st < NUM_STPS; ++st) {
                int local_row = st * ROWS_PER_STP + row_in_step;
                clear(r_staging[st]);
                if (local_row < valid_len) {
                    *reinterpret_cast<uint4*>(&r_staging[st](0)) =
                        *reinterpret_cast<const uint4*>(gdO + do_offset_next + local_row * D_V + col_base);
                }
            }
        }

        __syncthreads(); // [SYNC 1] (RAW: dR^T in Region 5 ready for LDSM_T; WAR: all warps finished reading dO in Region 1 and Q in Region 2)

        // Step 3.3 - Load dO_next, Q_next, Compute dq_intra = dR @ K_scaled & Stage via Region 5
        if (g + 1 < num_groups) {
        // Write dO_next from r_staging into Region 1 (sdO_flat)
            #pragma unroll
            for (int st = 0; st < NUM_STPS; ++st) {
                auto sdO_step = local_tile(sdO_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
                auto tdOsdO   = thr_copy_g2r.partition_D(sdO_step);
                *reinterpret_cast<uint4*>(&tdOsdO(0)) = *reinterpret_cast<const uint4*>(&r_staging[st](0));
            }

            // Load Q_next from GMEM into r_staging, then write to Region 2 (sQ_flat)
            int64_t next_h_q      = (int64_t)h_kv * num_groups + (g + 1);
            int64_t q_offset_next = (next_h_q * total_L + (int64_t)chunk_start) * D_K;
            #pragma unroll
            for (int st = 0; st < NUM_STPS; ++st) {
                int local_row = st * ROWS_PER_STP + row_in_step;
                clear(r_staging[st]);
                if (local_row < valid_len) {
                    *reinterpret_cast<uint4*>(&r_staging[st](0)) =
                        *reinterpret_cast<const uint4*>(gQ + q_offset_next + local_row * D_K + col_base);
                }
                auto sQ_step = local_tile(sQ_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                auto tQsQ    = thr_copy_g2r.partition_D(sQ_step);
                *reinterpret_cast<uint4*>(&tQsQ(0)) = *reinterpret_cast<const uint4*>(&r_staging[st](0));
            }
        }

        // Read transposed dR from Region 5 (LDSM_T) into tCrA
        // Read K_scaled from Region 4 (LDSM_T) into tCrB
        cute::copy(s2r_copy_a_mn, s2r_thr_copy_a_mn.partition_S(sA_G5_dR), tCrA_mn_view);
        cute::copy(s2r_copy_b_mn, s2r_thr_copy_b_mn.partition_S(sB_G5_K),  tCrB_mn_view);
        clear(tCrC_tmp); // tCrC_tmp receives dq_intra
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_tmp);

        __syncthreads(); // [SYNC 2] (WAR: Ensure all warps have read dR in Region 5)

        // Stage dq_intra (half2) into Region 5 (s_work_flat with Swizzle<3,3,3>)
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int target_col = n * 8 + lane_in_quad * 2;
            half2 val_a = __floats2half2_rn(tCrC_tmp(4 * n + 0), tCrC_tmp(4 * n + 1));
            half2 val_b = __floats2half2_rn(tCrC_tmp(4 * n + 2), tCrC_tmp(4 * n + 3));
            reinterpret_cast<half2*>(&s_work_flat(row_a, target_col))[0] = val_a;
            reinterpret_cast<half2*>(&s_work_flat(row_b, target_col))[0] = val_b;
        }

        __syncthreads(); // [SYNC 3] (RAW: dq_intra in Region 5 and dO_next, Q_next in Regions 1, 2 ready)

        // 128-bit vectorized load from Region 5 (s_work_flat) to GMEM gdQ
        int64_t q_out_base = (curr_h_q * total_L + (int64_t)chunk_start) * D_K;
        #pragma unroll
        for (int st = 0; st < NUM_STPS; ++st) {
            int local_row = st * ROWS_PER_STP + row_in_step;
            if (local_row < valid_len) {
                *reinterpret_cast<uint4*>(gdQ + q_out_base + local_row * D_K + col_base) =
                    *reinterpret_cast<const uint4*>(&s_work_flat(local_row, col_base));
            }
        }
    }

    // Stage 4 - Final Epilogue & GMEM Stores
    __syncthreads(); // Ensure all warps finished loop g

    // Stage tCrC_dk (FP16) into Region 2 (sQ_flat with Swizzle<3, 3, 3>)
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
        int target_col = n * 8 + lane_in_quad * 2;

        // dk -> Region 2
        half dk_a0 = __float2half(tCrC_dk(4 * n + 0));
        half dk_a1 = __float2half(tCrC_dk(4 * n + 1));
        half dk_b0 = __float2half(tCrC_dk(4 * n + 2));
        half dk_b1 = __float2half(tCrC_dk(4 * n + 3));
        reinterpret_cast<half2*>(&sQ_flat(row_a, target_col))[0] = make_half2(dk_a0, dk_a1);
        reinterpret_cast<half2*>(&sQ_flat(row_b, target_col))[0] = make_half2(dk_b0, dk_b1);
    }
    __syncthreads(); // RAW for Region 2

    // Store gdV (from Region 7 smem_dv), gdK (from Region 2 smem_Q), gdD (from Region 6 smem_dD) to GMEM (uint4)
    int64_t v_out_offset = (kv_head_offset + (int64_t)chunk_start) * D_V;
    int64_t k_out_offset = (kv_head_offset + (int64_t)chunk_start) * D_K;
    int64_t d_out_offset = ((int64_t)h_kv * num_chunks + curr_chunk) * ((int64_t)CHUNK_SIZE * CHUNK_SIZE);

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        if (local_row < valid_len) {
            *reinterpret_cast<uint4*>(&gdV[v_out_offset + local_row * D_V + col_base]) =
                *reinterpret_cast<const uint4*>(&s_dv_flat(local_row, col_base));
            *reinterpret_cast<uint4*>(&gdK[k_out_offset + local_row * D_K + col_base]) =
                *reinterpret_cast<const uint4*>(&sQ_flat(local_row, col_base));
        }
    // gdD stores full 64x64 chunk for Kernel 3 (read directly from Region 6 smem_dD with Swizzle)
        *reinterpret_cast<uint4*>(&gdD[d_out_offset + local_row * CHUNK_SIZE + col_base]) =
            *reinterpret_cast<const uint4*>(&s_dD_flat(local_row, col_base));
    }
}

// kernel_output_combine_dc_total
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_output_combine_dc_total(
    const T* __restrict__ gdD,                  // (num_kv_heads, num_chunks, chunk_size, chunk_size) [FP16] - dD^T from Kernel 1
    const T* __restrict__ gAlpha,               // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    float* __restrict__ gDcTotal,               // (num_kv_heads, total_L, 1) [FP32] - Kernel 2 wrote dc_inter, overwritten with da_total
    const int32_t* __restrict__ gChunkValidLen, // (num_chunks,) [Int32]
    const int32_t* __restrict__ gChunkOffsets,  // (num_chunks + 1,) [Int32]
    int64_t num_kv_heads,                       // scalar [int64]
    int64_t num_chunks,                         // scalar [int64]
    int64_t total_L                             // scalar [int64]
) {
    int h_kv       = blockIdx.x;
    int curr_chunk = blockIdx.y;
    int tid        = threadIdx.x;

    if (h_kv >= num_kv_heads || curr_chunk >= num_chunks) {
        return;
    }

    int valid_len   = gChunkValidLen[curr_chunk];
    int chunk_start = gChunkOffsets[curr_chunk];
    if (valid_len <= 0) {
        return;
    }

    int64_t kv_head_offset = (int64_t)h_kv * total_L;

    // Allocate Shared Memory < 2 KB (1540 B)
    struct SmemStorageDcTotal {
    alignas(16) float s_c[CHUNK_SIZE];           // 256 B: intra-chunk cumsum alpha
    alignas(4)  float s_warp0_sum;               // 4 B: prefix sum of warp 0
    alignas(16) float s_da_row[CHUNK_SIZE];      // 256 B: row sum gradient
    alignas(16) float s_warp_col[4 * CHUNK_SIZE];// 1024 B: column sum gradients for 4 warps
    };
    __shared__ SmemStorageDcTotal smem;

    // Stage 1 - Load alpha and Compute Local log_alpha_cumsum
    if (tid < CHUNK_SIZE) {
        float alpha_val = 0.0f;
        if (tid < valid_len) {
            int64_t alpha_idx = kv_head_offset + chunk_start + tid;
            alpha_val = stable_log_sigmoid(half_to_float(gAlpha[alpha_idx]));
        }
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float up = __shfl_up_sync(0xffffffff, alpha_val, offset);
            if ((tid & 31) >= offset) {
                alpha_val += up;
            }
        }
        if (tid == 31) {
            smem.s_warp0_sum = alpha_val;
        }
        smem.s_c[tid ^ ((tid >= 32) ? 4 : 0)] = alpha_val;
    }
    __syncthreads();

    if (tid >= 32 && tid < CHUNK_SIZE) {
        smem.s_c[tid ^ ((tid >= 32) ? 4 : 0)] += smem.s_warp0_sum;
    }
    __syncthreads();

    // Stage 2 - Compute Row Sum and Column Sum from dD^T Matrix
    constexpr int NUM_STPS     = 4;
    constexpr int ROWS_PER_STP = CHUNK_SIZE / NUM_STPS; // 16 rows

    int row_in_step = tid / 8; // 0..15 (16 rows)
    int col_vec_idx = tid % 8; // 0..7 (8 threads per row)
    int col_base    = col_vec_idx * 8; // 0, 8, 16, 24, 32, 40, 48, 56

    int64_t d_base = ((int64_t)h_kv * num_chunks + curr_chunk) * ((int64_t)CHUNK_SIZE * CHUNK_SIZE);

    // Pre-cache c_i (thread 8 corresponding columns) into private registers to avoid bank conflicts
    float r_c_i[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int col = col_base + k;
        r_c_i[k] = (col < valid_len) ? smem.s_c[col ^ ((col >= 32) ? 4 : 0)] : 0.0f;
    }

    float col_da[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        col_da[k] = 0.0f;
    }

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int j = st * ROWS_PER_STP + row_in_step; // row j in dD^T
        float c_j = (j < valid_len) ? smem.s_c[j ^ ((j >= 32) ? 4 : 0)] : 0.0f;

        // Load 8-element FP16 vector (128-bit) from gdD
        int64_t elem_offset = d_base + (int64_t)j * CHUNK_SIZE + col_base;
        uint4 r_dd_raw = *reinterpret_cast<const uint4*>(gdD + elem_offset);
        const T* dD_ptr = reinterpret_cast<const T*>(&r_dd_raw);

        float thread_row_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int i = col_base + k; // column i in dD^T
            if (i >= j && j < valid_len && i < valid_len) {
                float dd_val = half_to_float(dD_ptr[k]);
                float c_i = r_c_i[k];
                float e_val = expf(c_i - c_j);
                float m_val = dd_val * e_val;
                thread_row_sum += m_val;
                col_da[k] += m_val;
            }
        }

        // Warp reduction for thread_row_sum across 8 threads in the same row
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            thread_row_sum += __shfl_down_sync(0xffffffff, thread_row_sum, offset);
        }

        if (col_vec_idx == 0) {
            smem.s_da_row[j] = (j < valid_len) ? thread_row_sum : 0.0f;
        }
    }

    // Warp reduction for col_da across 4 rows within the warp
    int warp_id = tid / 32;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float val = col_da[k];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);

        if (row_in_step % 4 == 0) {
            int col = col_base + k;
            int swz_col = col ^ ((col >= 32) ? 4 : 0);
            smem.s_warp_col[warp_id * CHUNK_SIZE + swz_col] = (col < valid_len) ? val : 0.0f;
        }
    }
    __syncthreads();

    // Stage 3 - First 64 threads synthesize da_total = dc_intra + dc_inter & store to GMEM
    if (tid < CHUNK_SIZE) {
        int t = tid;
        if (t < valid_len) {
            float row_sum_t = smem.s_da_row[t];
            int swz_t = t ^ ((t >= 32) ? 4 : 0);
            float col_sum_t = smem.s_warp_col[0 * CHUNK_SIZE + swz_t] +
                              smem.s_warp_col[1 * CHUNK_SIZE + swz_t] +
                              smem.s_warp_col[2 * CHUNK_SIZE + swz_t] +
                              smem.s_warp_col[3 * CHUNK_SIZE + swz_t];
            float dc_intra = col_sum_t - row_sum_t;
            float dc_inter = gDcTotal[kv_head_offset + chunk_start + t];
            float da_total = dc_intra + dc_inter;
            gDcTotal[kv_head_offset + chunk_start + t] = da_total;
        }
    }
}

// HOST COORDINATOR FUNCTION (output_combine_bwd)

// Host coordinator for output_combine_bwd. Outputs: (dq, dk_ext, dv_ext, da_ext [FP32], dS).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
output_combine_bwd(
    const at::Tensor& grad_output,      // (num_q_heads, total_L, d_v) [FP16] - Upstream gradient dO
    const at::Tensor& q,                // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const at::Tensor& k,                // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,                // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    [[maybe_unused]] const at::Tensor& b,// (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& w,                // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& gamma,            // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& a,                // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& S,                // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent state S
    const at::Tensor& cu_seq_lens,      // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t num_q_heads,                // scalar [int64] - Number of query heads
    int64_t num_kv_heads,               // scalar [int64] - Number of KV heads
    int64_t d_k,                        // scalar [int64] - Head dimension Key (default: 64)
    int64_t d_v,                        // scalar [int64] - Head dimension Value (default: 64)
    int64_t chunk_size                  // scalar [int64] - Chunk size (default: 64)
) {
    // Stage 1 - Input Validation
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(S.is_cuda(), "S must be a CUDA tensor");
    TORCH_CHECK(cu_seq_lens.is_cuda(), "cu_seq_lens must be a CUDA tensor");

    TORCH_CHECK(q.device() == grad_output.device(), "q must be on same device as grad_output");
    TORCH_CHECK(k.device() == grad_output.device(), "k must be on same device as grad_output");
    TORCH_CHECK(v.device() == grad_output.device(), "v must be on same device as grad_output");
    TORCH_CHECK(w.device() == grad_output.device(), "w must be on same device as grad_output");
    TORCH_CHECK(gamma.device() == grad_output.device(), "gamma must be on same device as grad_output");
    TORCH_CHECK(a.device() == grad_output.device(), "a must be on same device as grad_output");
    TORCH_CHECK(S.device() == grad_output.device(), "S must be on same device as grad_output");
    TORCH_CHECK(cu_seq_lens.device() == grad_output.device(), "cu_seq_lens must be on same device as grad_output");

    TORCH_CHECK(grad_output.scalar_type() == at::ScalarType::Half, "grad_output must be FP16");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half, "q must be FP16");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Half, "k must be FP16");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::Half, "v must be FP16");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Half, "w must be FP16");
    TORCH_CHECK(gamma.scalar_type() == at::ScalarType::Half, "gamma must be FP16");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half, "a must be FP16");
    TORCH_CHECK(S.scalar_type() == at::ScalarType::Half, "S must be FP16");

    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(S.is_contiguous(), "S must be contiguous");
    TORCH_CHECK(cu_seq_lens.is_contiguous(), "cu_seq_lens must be contiguous");

    int64_t total_L = q.size(1);

    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64 on SM75");
    TORCH_CHECK(d_k == 64, "d_k must be 64 on SM75");
    TORCH_CHECK(d_v == 64, "d_v must be 64 on SM75");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 2, "cu_seq_lens must be 1D with >= 2 elements");

    int64_t num_groups = num_q_heads / num_kv_heads;

    // Compute chunk_offsets and chunk_valid_lens
    auto [chunk_offsets, chunk_valid_lens, num_chunks] = compute_chunk_offsets_and_valid_lens(cu_seq_lens, chunk_size);

    // Stage 2 - Memory Allocation for Outputs and Intermediate dD Buffer
    auto options_half  = grad_output.options();
    auto options_float = grad_output.options().dtype(at::kFloat);

    at::Tensor dq     = at::empty({num_q_heads, total_L, d_k}, options_half);
    at::Tensor dk_ext = at::empty({num_kv_heads, total_L, d_k}, options_half);
    at::Tensor dv_ext = at::empty({num_kv_heads, total_L, d_v}, options_half);
    at::Tensor da_ext = at::zeros({num_kv_heads, total_L, 1}, options_float); // Initialized with zeros
    at::Tensor dS     = at::empty({num_kv_heads, num_chunks, d_v, d_k}, options_half);

    TORCH_CHECK(reinterpret_cast<uintptr_t>(dq.data_ptr()) % 16 == 0, "dq tensor data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(dk_ext.data_ptr()) % 16 == 0, "dk_ext tensor data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(dv_ext.data_ptr()) % 16 == 0, "dv_ext tensor data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(da_ext.data_ptr()) % 16 == 0, "da_ext tensor data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(dS.data_ptr()) % 16 == 0, "dS tensor data pointer must be 16-byte aligned");

    if (total_L == 0 || num_chunks == 0) {
        return std::make_tuple(dq, dk_ext, dv_ext, da_ext, dS);
    }

    // Single intermediate buffer: dD_buf (FP16) - stores dD^T shape (Hkv, num_chunks, 64, 64)
    at::Tensor dD_buf = at::empty({num_kv_heads, num_chunks, chunk_size, chunk_size}, options_half);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(dD_buf.data_ptr()) % 16 == 0, "dD_buf tensor data pointer must be 16-byte aligned");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    using T = cutlass::half_t;

    const T* p_do        = reinterpret_cast<const T*>(grad_output.data_ptr<at::Half>());
    const T* p_q         = reinterpret_cast<const T*>(q.data_ptr<at::Half>());
    const T* p_k         = reinterpret_cast<const T*>(k.data_ptr<at::Half>());
    const T* p_v         = reinterpret_cast<const T*>(v.data_ptr<at::Half>());
    const T* p_w         = reinterpret_cast<const T*>(w.data_ptr<at::Half>());
    const T* p_gamma     = reinterpret_cast<const T*>(gamma.data_ptr<at::Half>());
    const T* p_alpha     = reinterpret_cast<const T*>(a.data_ptr<at::Half>());
    const T* p_s         = reinterpret_cast<const T*>(S.data_ptr<at::Half>());
    const int32_t* p_vlen    = chunk_valid_lens.data_ptr<int32_t>();
    const int32_t* p_offsets = chunk_offsets.data_ptr<int32_t>();

    T* p_dq              = reinterpret_cast<T*>(dq.data_ptr<at::Half>());
    T* p_dk              = reinterpret_cast<T*>(dk_ext.data_ptr<at::Half>());
    T* p_dv              = reinterpret_cast<T*>(dv_ext.data_ptr<at::Half>());
    float* p_da_ext      = da_ext.data_ptr<float>();
    T* p_ds              = reinterpret_cast<T*>(dS.data_ptr<at::Half>());
    T* p_dd              = reinterpret_cast<T*>(dD_buf.data_ptr<at::Half>());

    constexpr int NUM_THRDS      = 128;
    constexpr int CHUNK_SZ       = 64;
    constexpr int DK_DIM         = 64;
    constexpr int DV_DIM         = 64;
    constexpr int ROWS_PER_STP   = 16;

    using MMAAtomOp = SM75_16x8x8_F32F16F16F32_TN;
    using MMAAtom   = MMA_Atom<MMAAtomOp>;
    using TiledMMA_411 = TiledMMA<MMAAtom, Layout<Shape<_4, _1, _1>, Stride<_1, _4, _4>>>;
    using TiledMMA_221 = TiledMMA<MMAAtom, Layout<Shape<_2, _2, _1>, Stride<_1, _2, _4>>>;

    // S2R Copy Atoms
    using S2RCopyAtomA_KMajor   = Copy_Atom<SM75_U32x2_LDSM_N, T>;
    using S2RCopyAtomB_KMajor   = Copy_Atom<SM75_U32x1_LDSM_N, T>;
    using S2RCopyAtomA_MNMajor  = Copy_Atom<SM75_U16x4_LDSM_T, T>;
    using S2RCopyAtomB_MNMajor  = Copy_Atom<SM75_U16x2_LDSM_T, T>;

    using CopyAtom = Copy_Atom<UniversalCopy<uint4>, T>;
    using TiledCopyG2R = decltype(make_tiled_copy(
        CopyAtom{},
        make_layout(make_shape(Int<ROWS_PER_STP>{}, Int<DK_DIM / 8>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<8>{}), LayoutRight{})
    ));

    // Stage 3 - Launch Sequential 3 Kernels on CUDA Stream

    // 1. Launch Kernel 1: kernel_output_combine_dq_dk_dv (Computes dq_intra, dk, dv, dD^T)
    {
        dim3 grid(num_kv_heads, num_chunks);
        dim3 block(NUM_THRDS);

        using Kernel1SmemStorage = SmemStorageKernel1<T, CHUNK_SZ, DK_DIM, DV_DIM>;
        constexpr int smem_size_k1 = sizeof(Kernel1SmemStorage);
        static_assert(smem_size_k1 == 57600, "Kernel 1 Dynamic Shared Memory must be 57,600 bytes");

        auto kernel_k1 = kernel_output_combine_dq_dk_dv<
            T, NUM_THRDS, CHUNK_SZ, DK_DIM, DV_DIM,
            MMAAtom, TiledMMA_411,
            S2RCopyAtomA_KMajor, S2RCopyAtomA_MNMajor,
            S2RCopyAtomB_KMajor, S2RCopyAtomB_MNMajor,
            TiledCopyG2R
        >;

        C10_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel_k1),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size_k1
        ));

        kernel_k1<<<grid, block, smem_size_k1, stream>>>(
            p_q, p_do, p_k, p_v, p_w, p_gamma, p_alpha,
            p_vlen, p_offsets,
            p_dq, p_dk, p_dv, p_dd,
            num_q_heads, num_kv_heads, num_groups, num_chunks, total_L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 2. Launch Kernel 2: kernel_output_combine_dc_inter_dS (dq_intra -> dq_total, dS, dc_inter)
    {
        dim3 grid(num_kv_heads, num_chunks);
        dim3 block(NUM_THRDS);

        kernel_output_combine_dc_inter_dS<
            T, NUM_THRDS, CHUNK_SZ, DK_DIM, DV_DIM,
            MMAAtom, TiledMMA_221,
            S2RCopyAtomA_MNMajor, S2RCopyAtomB_MNMajor,
            S2RCopyAtomA_KMajor, S2RCopyAtomB_KMajor,
            TiledCopyG2R
        ><<<grid, block, 0, stream>>>(
            p_q, p_do, p_s, p_alpha, p_vlen, p_offsets,
            p_dq, p_ds, p_da_ext,
            num_q_heads, num_kv_heads, num_groups, num_chunks, total_L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 3. Launch Kernel 3: kernel_output_combine_dc_total (Accumulate da_total = dc_intra + dc_inter)
    {
        dim3 grid(num_kv_heads, num_chunks);
        dim3 block(NUM_THRDS);

        kernel_output_combine_dc_total<
            T, NUM_THRDS, CHUNK_SZ
        ><<<grid, block, 0, stream>>>(
            p_dd, p_alpha, p_da_ext,
            p_vlen, p_offsets,
            num_kv_heads, num_chunks, total_L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Stage 4 - Return Output Tuple
    dD_buf.reset();

    return std::make_tuple(dq, dk_ext, dv_ext, da_ext, dS);
}

