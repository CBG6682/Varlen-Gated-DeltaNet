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
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

using namespace cute;

// Chunk WY forward kernel computing transition matrix W and update matrix Y.
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_K,
    int D_V,
    typename MMAAtom,
    typename TiledMMAType,
    typename S2RCopyAtomA,
    typename S2RCopyAtomB,
    typename TiledCopyG2R,
    typename TiledCopyStore
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_chunk_wy_fwd(
    const T* __restrict__ gK,                  // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const T* __restrict__ gV,                  // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const T* __restrict__ gB,                  // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const T* __restrict__ gW_gate,             // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const T* __restrict__ gGamma,              // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const T* __restrict__ gAlpha,              // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    T* __restrict__ gW,                        // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Output W
    T* __restrict__ gY,                        // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Output Y
    const int32_t* __restrict__ chunk_offsets, // (num_chunks + 1,) [Int32] - Token offsets per chunk
    int64_t total_L,                           // scalar [int64] - Total tokens
    int64_t num_chunks                         // scalar [int64] - Total chunks
) {
    // Stage 1 - Setup & Grid Initialization
    int chunk_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (chunk_idx >= num_chunks) return;

    int32_t chunk_start_token = chunk_offsets[chunk_idx];
    int32_t chunk_end_token   = chunk_offsets[chunk_idx + 1];
    int valid_tokens          = chunk_end_token - chunk_start_token;

    int64_t head_offset = (int64_t)head_idx * total_L;

    // Shared Memory allocation: 24 KB matrices + 256 B log_alpha + 4 B warp sum
    __shared__ alignas(16) T sU[CHUNK_SIZE * D_K]; // 8 KB
    __shared__ alignas(16) T sV[CHUNK_SIZE * D_V]; // 8 KB
    __shared__ alignas(16) T sK[CHUNK_SIZE * D_K]; // 8 KB
    __shared__ float s_log_alpha[CHUNK_SIZE];      // 256 B: prefix scan log alpha
    __shared__ float s_warp0_sum;                  // 4 B: warp 0 reduction sum

    // Reuse SMEM buffers for epilogue outputs
    T* sW = sU;
    T* sY = sV;

    using SwizzleOp = Swizzle<3, 3, 3>;

    auto smem_layout_flat_k = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_flat_v = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));

    // SMEM MMA layouts:
    // Matrix A (U^T or V^T): Shape (M=64, K=64), Stride (1, 64) -> MN-major
    auto smem_layout_mma_A = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_K>{})));
    // Matrix B (K_decay): Shape (N=64, K=64), Stride (1, 64) -> MN-major
    auto smem_layout_mma_B = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_K>{})));
    // Matrix C (W or Y): Shape (M=64, N=64), Stride (64, 1) -> Row-Major
    auto smem_layout_mma_C = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    auto sU_flat = make_tensor(make_smem_ptr(sU), smem_layout_flat_k);
    auto sV_flat = make_tensor(make_smem_ptr(sV), smem_layout_flat_v);
    auto sK_flat = make_tensor(make_smem_ptr(sK), smem_layout_flat_k);

    auto sW_flat = make_tensor(make_smem_ptr(sW), smem_layout_mma_C);
    auto sY_flat = make_tensor(make_smem_ptr(sY), smem_layout_mma_C);

    auto sU_mma = make_tensor(make_smem_ptr(sU), smem_layout_mma_A);
    auto sV_mma = make_tensor(make_smem_ptr(sV), smem_layout_mma_A);
    auto sK_mma = make_tensor(make_smem_ptr(sK), smem_layout_mma_B);
    auto sW_mma = make_tensor(make_smem_ptr(sW), smem_layout_mma_C);
    auto sY_mma = make_tensor(make_smem_ptr(sY), smem_layout_mma_C);

    TiledMMAType tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_copy_a.get_slice(tid);
    auto s2r_thr_copy_b = s2r_copy_b.get_slice(tid);

    auto tCsA_U = s2r_thr_copy_a.partition_S(sU_mma);
    auto tCsA_V = s2r_thr_copy_a.partition_S(sV_mma);
    auto tCsB_K = s2r_thr_copy_b.partition_S(sK_mma);
    auto tCsC_W = thr_mma.partition_C(sW_mma);
    auto tCsC_Y = thr_mma.partition_C(sY_mma);

    auto tCrA = thr_mma.partition_fragment_A(sU_mma);
    auto tCrB = thr_mma.partition_fragment_B(sK_mma);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    TiledCopyG2R tiled_copy_g2r;
    TiledCopyStore tiled_copy_store;
    auto thr_copy_g2r   = tiled_copy_g2r.get_thread_slice(tid);
    auto thr_copy_store = tiled_copy_store.get_thread_slice(tid);

    constexpr int ROWS_PER_STP = 16;
    constexpr int NUM_STPS     = CHUNK_SIZE / ROWS_PER_STP; // 4 steps

    int row_in_step = tid / (D_K / 8);
    int col_vec_idx = tid % (D_K / 8);
    int col_base    = col_vec_idx * 8;

    // Empty chunk handling (valid_tokens <= 0)
    if (valid_tokens <= 0) {
        int64_t mat_offset_w = ((int64_t)head_idx * num_chunks + chunk_idx) * ((int64_t)D_K * D_K);
        int64_t mat_offset_y = ((int64_t)head_idx * num_chunks + chunk_idx) * ((int64_t)D_V * D_K);
        T* p_w_chunk = gW + mat_offset_w;
        T* p_y_chunk = gY + mat_offset_y;
        #pragma unroll
        for (int st = 0; st < NUM_STPS; ++st) {
            int global_row = st * ROWS_PER_STP + row_in_step;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int global_col = col_base + i;
                p_w_chunk[global_row * D_K + global_col] = float_to_out<T>(global_row == global_col ? 1.0f : 0.0f);
                p_y_chunk[global_row * D_K + global_col] = float_to_out<T>(0.0f);
            }
        }
        return;
    }

    // Stage 1.1 - Prefix Scan for log_alpha_cumsum inside chunk
    if (tid < CHUNK_SIZE) {
        float alpha_val = 0.0f;
        if (tid < valid_tokens) {
            int64_t token_idx = (int64_t)chunk_start_token + tid;
            int64_t a_offset  = head_offset + token_idx;
            alpha_val = stable_log_sigmoid(__half2float(to_half_val(gAlpha[a_offset])));
        }
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            float up = __shfl_up_sync(0xffffffff, alpha_val, offset);
            if ((tid & 31) >= offset) {
                alpha_val += up;
            }
        }
        if (tid == 31) {
            s_warp0_sum = alpha_val;
        }
        s_log_alpha[tid] = alpha_val;
    }
    __syncthreads();

    if (tid >= 32 && tid < CHUNK_SIZE) {
        s_log_alpha[tid] += s_warp0_sum;
    }
    __syncthreads();

    float chunk_last_val = expf(s_log_alpha[valid_tokens - 1]);

    // Stage 2 - Load & Recompute Activations for K, B, U, K_decay
    const T* p_k_chunk      = gK      + head_offset * D_K + (int64_t)chunk_start_token * D_K;
    const T* p_v_chunk      = gV      + head_offset * D_V + (int64_t)chunk_start_token * D_V;
    const T* p_b_chunk      = gB      + head_offset * D_K + (int64_t)chunk_start_token * D_K;
    const T* p_w_gate_chunk = gW_gate + head_offset * D_V + (int64_t)chunk_start_token * D_V;
    const T* p_gamma_chunk  = gGamma  + head_offset;

    auto gK_chunk   = make_tensor(make_gmem_ptr(p_k_chunk), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gV_chunk   = make_tensor(make_gmem_ptr(p_v_chunk), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto gB_chunk   = make_tensor(make_gmem_ptr(p_b_chunk), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gW_g_chk   = make_tensor(make_gmem_ptr(p_w_gate_chunk), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));

    auto dummy_tile = local_tile(sU_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(0, 0));
    auto dummy_part = thr_copy_g2r.partition_S(dummy_tile);

    __half2 r_sqrt_gamma_h2[NUM_STPS];

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_tokens);
        int64_t token_idx = (int64_t)chunk_start_token + local_row;

        auto gK_step = local_tile(gK_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto gB_step = local_tile(gB_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

        decltype(make_fragment_like(dummy_part)) rK, rB;
        clear(rK); clear(rB);

        decltype(make_fragment_like<bool>(dummy_part)) tPred;
        #pragma unroll
        for (int i = 0; i < size(tPred); ++i) tPred(i) = row_valid;

        // G2R: Load 128-bit vectors for K, B
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gK_step), rK);
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gB_step), rB);

        // 1. Gamma activation: softplus + sqrt
        float sqrt_gamma_f = 0.0f;
        if (row_valid) {
            __half g_val = to_half_val(p_gamma_chunk[token_idx]);
            __half sp    = fast_softplus_h(g_val);
            sqrt_gamma_f = sqrtf(fmaxf(__half2float(sp), 0.0f) + 1e-4f);
        }
        __half2 sqrt_gamma_h2 = __half2half2(__float2half(sqrt_gamma_f));
        r_sqrt_gamma_h2[st] = sqrt_gamma_h2;

        // 2. L2-norm of K via warp reduction across 8 row threads
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

        // 3. K_scaled = K * sqrt_gamma / k_norm
        half2* rK_h2 = reinterpret_cast<half2*>(&rK(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            rK_h2[p] = __hmul2(rK_h2[p], inv_norm_scale_h2);
        }

        // 4. U = sigmoid(B) * K_scaled
        half2* rB_h2 = reinterpret_cast<half2*>(&rB(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            half2 b_sig = fast_sigmoid_h2(rB_h2[p]);
            rB_h2[p]    = row_valid ? __hmul2(b_sig, rK_h2[p]) : __float2half2_rn(0.0f);
        }

        // 5. K_decay = K_scaled * exp(c_last - c_t)
        float decay_t = row_valid ? expf(s_log_alpha[valid_tokens - 1] - s_log_alpha[local_row]) : 0.0f;
        __half2 decay_h2 = __half2half2(__float2half(decay_t));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            rK_h2[p] = row_valid ? __hmul2(rK_h2[p], decay_h2) : __float2half2_rn(0.0f);
        }

        // 6. Store U and K_decay into SMEM
        auto sU_step = local_tile(sU_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto sK_step = local_tile(sK_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tUsU    = thr_copy_g2r.partition_D(sU_step);
        auto tKsK    = thr_copy_g2r.partition_D(sK_step);
        *reinterpret_cast<uint4*>(&tUsU(0)) = *reinterpret_cast<const uint4*>(&rB(0));
        *reinterpret_cast<uint4*>(&tKsK(0)) = *reinterpret_cast<const uint4*>(&rK(0));
    }
    __syncthreads();

    // Stage 3 - GEMM 1: Transition Matrix W & V_new Generation (Tensor Core)
    // Prefetch V and W_gate to overlap GMEM latency
    decltype(make_fragment_like(dummy_part)) rV[NUM_STPS];
    decltype(make_fragment_like(dummy_part)) rW_g[NUM_STPS];

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_tokens);

        decltype(make_fragment_like<bool>(dummy_part)) tPred;
        #pragma unroll
        for (int i = 0; i < size(tPred); ++i) tPred(i) = row_valid;

        clear(rV[st]);
        clear(rW_g[st]);

        auto gV_step = local_tile(gV_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto gW_step = local_tile(gW_g_chk, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));

        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gV_step), rV[st]);
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gW_step), rW_g[st]);
    }

    auto tCrC_erase = thr_mma.make_fragment_C(tCsC_W);
    clear(tCrC_erase);

    // Load U and K_decay into registers via LDSM
    cute::copy(s2r_copy_a, tCsA_U, tCrA_view);
    cute::copy(s2r_copy_b, tCsB_K, tCrB_view);
    __syncthreads();

    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_erase);

    // Epilogue W: convert FP32 -> FP16 and store to SMEM sW
    auto rC_half_erase = make_tensor_like<T>(tCrC_erase);
    #pragma unroll
    for (int i = 0; i < size(rC_half_erase); ++i) {
        rC_half_erase(i) = float_to_out<T>(tCrC_erase(i));
    }
    cute::copy(rC_half_erase, tCsC_W);
    __syncthreads();

    // W = I * chunk_last_val - erase, store to GMEM
    // Concurrently compute V_new = sigmoid(W_gate) * (V * sqrt_gamma) and store into sV
    int64_t mat_offset_w = ((int64_t)head_idx * num_chunks + chunk_idx) * ((int64_t)D_K * D_K);
    T* p_w_mat_chunk = gW + mat_offset_w;
    auto gW_chunk = make_tensor(make_gmem_ptr(p_w_mat_chunk), make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int global_row = st * ROWS_PER_STP + row_in_step;
        int local_row  = global_row;
        bool row_valid = (local_row < valid_tokens);

        // 1. Store W to GMEM
        auto sW_step = local_tile(sW_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tWsW = thr_copy_store.partition_S(sW_step);
        decltype(make_fragment_like(tWsW)) rErase_slice;
        cute::copy(tWsW, rErase_slice);

        decltype(make_fragment_like(tWsW)) rW_out;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int global_col = col_base + i;
            float erase_val = half_to_float(rErase_slice(i));
            float w_val = (global_row == global_col) ? (chunk_last_val - erase_val) : (-erase_val);
            rW_out(i) = float_to_out<T>(w_val);
        }

        auto gW_step = local_tile(gW_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tWgW = thr_copy_store.partition_D(gW_step);
        cute::copy(tiled_copy_store, rW_out, tWgW);

        // 2. Epilogue: Compute V_new = sigmoid(W_gate) * (V * sqrt_gamma) and write to sV
        half2* rV_h2 = reinterpret_cast<half2*>(&rV[st](0));
        half2* rW_h2 = reinterpret_cast<half2*>(&rW_g[st](0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            half2 v_scaled = __hmul2(rV_h2[p], r_sqrt_gamma_h2[st]);
            half2 w_sig    = fast_sigmoid_h2(rW_h2[p]);
            rV_h2[p]       = row_valid ? __hmul2(w_sig, v_scaled) : __float2half2_rn(0.0f);
        }

        auto sV_step = local_tile(sV_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto tVsV    = thr_copy_g2r.partition_D(sV_step);
        *reinterpret_cast<uint4*>(&tVsV(0)) = *reinterpret_cast<const uint4*>(&rV[st](0));
    }
    __syncthreads();

    // Stage 4 - GEMM 2: Update Matrix Y (Tensor Core)
    auto tCrC_Y = thr_mma.make_fragment_C(tCsC_Y);
    clear(tCrC_Y);

    // Load V_new and K_decay into registers via LDSM
    cute::copy(s2r_copy_a, tCsA_V, tCrA_view);
    cute::copy(s2r_copy_b, tCsB_K, tCrB_view);
    __syncthreads();

    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_Y);

    // Epilogue Y: convert FP32 -> FP16 and store to SMEM sY
    auto rC_half_Y = make_tensor_like<T>(tCrC_Y);
    #pragma unroll
    for (int i = 0; i < size(rC_half_Y); ++i) {
        rC_half_Y(i) = float_to_out<T>(tCrC_Y(i));
    }
    cute::copy(rC_half_Y, tCsC_Y);
    __syncthreads();

    // Store Y to GMEM via vectorized stores
    int64_t mat_offset_y = ((int64_t)head_idx * num_chunks + chunk_idx) * ((int64_t)D_V * D_K);
    T* p_y_mat_chunk = gY + mat_offset_y;
    auto gY_chunk = make_tensor(make_gmem_ptr(p_y_mat_chunk), make_layout(make_shape(Int<D_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        auto sY_step = local_tile(sY_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tYsY = thr_copy_store.partition_S(sY_step);
        decltype(make_fragment_like(tYsY)) rY_slice;
        cute::copy(tYsY, rY_slice);

        auto gY_step = local_tile(gY_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tYgY = thr_copy_store.partition_D(gY_step);
        cute::copy(tiled_copy_store, rY_slice, tYgY);
    }
}

// HOST FUNCTION (chunk_wy_fwd)

std::tuple<at::Tensor, at::Tensor>
chunk_wy_fwd(
    const at::Tensor& k,               // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,               // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& b,               // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& w,               // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& gamma,           // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& a,               // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& cu_seq_lens,     // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t chunk_size                 // scalar [int64] - Chunk size (default 64)
) {
    // Stage 1 - Input Validation
    at::cuda::CUDAGuard device_guard(k.device());

    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(cu_seq_lens.is_cuda(), "cu_seq_lens must be a CUDA tensor");

    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");

    TORCH_CHECK(k.scalar_type() == at::ScalarType::Half, "k must be float16");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::Half, "v must be float16");
    TORCH_CHECK(b.scalar_type() == at::ScalarType::Half, "b must be float16");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Half, "w must be float16");
    TORCH_CHECK(gamma.scalar_type() == at::ScalarType::Half, "gamma must be float16");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half, "a must be float16");

    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64 for SM75 Tensor Core alignment");
    TORCH_CHECK(k.dim() == 3, "k must be 3D (num_kv_heads, total_L, d_k)");
    TORCH_CHECK(v.dim() == 3, "v must be 3D (num_kv_heads, total_L, d_v)");
    TORCH_CHECK(b.dim() == 3, "b must be 3D (num_kv_heads, total_L, d_k)");
    TORCH_CHECK(w.dim() == 3, "w must be 3D (num_kv_heads, total_L, d_v)");

    int64_t num_kv_heads = k.size(0);
    int64_t total_L      = k.size(1);
    int64_t d_k          = k.size(2);
    int64_t d_v          = v.size(2);

    TORCH_CHECK(d_k == 64, "d_k must be 64");
    TORCH_CHECK(d_v == 64, "d_v must be 64");
    TORCH_CHECK(v.size(0) == num_kv_heads && v.size(1) == total_L, "v shape mismatch");
    TORCH_CHECK(b.size(0) == num_kv_heads && b.size(1) == total_L && b.size(2) == d_k, "b shape mismatch");
    TORCH_CHECK(w.size(0) == num_kv_heads && w.size(1) == total_L && w.size(2) == d_v, "w shape mismatch");
    TORCH_CHECK(gamma.size(0) == num_kv_heads && gamma.size(1) == total_L, "gamma shape mismatch");
    TORCH_CHECK(a.size(0) == num_kv_heads && a.size(1) == total_L, "a shape mismatch");

    // Stage 2 - Kernel Launch
    auto [chunk_offsets_gpu, chunk_valid_lens_gpu, num_chunks] = compute_chunk_offsets_and_valid_lens(cu_seq_lens, chunk_size);

    auto options = k.options();
    if (num_chunks <= 0) {
        at::Tensor W = at::empty({num_kv_heads, 0, d_k, d_k}, options);
        at::Tensor Y = at::empty({num_kv_heads, 0, d_v, d_k}, options);
        return std::make_tuple(W, Y);
    }

    at::Tensor W = torch::empty({num_kv_heads, num_chunks, d_k, d_k}, options);
    at::Tensor Y = torch::empty({num_kv_heads, num_chunks, d_v, d_k}, options);

    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_SIZE_CONST = 64;
    constexpr int D_K_CONST = 64;
    constexpr int D_V_CONST = 64;

    dim3 grid(num_chunks, num_kv_heads);
    dim3 block(NUM_THREADS);

    using Element = cutlass::half_t;
    using MMAAtom = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMAType = TiledMMA<MMAAtom, Layout<Shape<_2, _2, _1>, Stride<_1, _2, _4>>>;

    using S2RCopyAtomA = Copy_Atom<SM75_U16x4_LDSM_T, Element>;
    using S2RCopyAtomB = Copy_Atom<SM75_U16x2_LDSM_T, Element>;

    using CopyAtomG2R = Copy_Atom<UniversalCopy<uint4>, Element>;
    using TiledCopyG2R = decltype(make_tiled_copy(
        CopyAtomG2R{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));

    using CopyAtomStore = Copy_Atom<UniversalCopy<uint4>, Element>;
    using TiledCopyStore = decltype(make_tiled_copy(
        CopyAtomStore{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    {
        kernel_chunk_wy_fwd<
            Element,
            NUM_THREADS,
            CHUNK_SIZE_CONST,
            D_K_CONST,
            D_V_CONST,
            MMAAtom,
            TiledMMAType,
            S2RCopyAtomA,
            S2RCopyAtomB,
            TiledCopyG2R,
            TiledCopyStore
        ><<<grid, block, 0, stream>>>(
            reinterpret_cast<const Element*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(v.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(b.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(w.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(gamma.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(a.data_ptr<at::Half>()),
            reinterpret_cast<Element*>(W.data_ptr<at::Half>()),
            reinterpret_cast<Element*>(Y.data_ptr<at::Half>()),
            chunk_offsets_gpu.data_ptr<int32_t>(),
            total_L,
            num_chunks
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Stage 3 - Return Results
    return std::make_tuple(W, Y);
}
