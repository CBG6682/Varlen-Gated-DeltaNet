#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

using namespace cute;

// HELPER FUNCTIONS & GLOBAL KERNEL

// Helper: 8-lane reduction sum across a group of 8 threads on the same row
__device__ __forceinline__ float octet_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    return val;
}

// Shared Memory Allocation Layout (16 KB, 16-byte aligned)
struct alignas(16) SmemStorage {
    half sS[64 * 64];   // 8192 bytes: stores past_state matrix S (Swizzle<3, 3, 3>)
    half sVec[64 * 64]; // 8192 bytes: inter-warp buffer (Swizzle<3, 3, 3>)
};

// Single-step (L = 1) DeltaNet decode kernel updating past_state in-place.
template <
    typename Element,
    int NUM_THREADS,
    int D_K,
    int D_V,
    int FIXED_G,
    typename SmemLayoutS,
    typename TiledCopyS,
    typename TiledCopyVec
>
__global__ void __launch_bounds__(NUM_THREADS)
kernel_deltanet_infer(
    Element*       __restrict__ past_state_ptr, // shape: (B, H, 64, 64), dtype: half* (in-place update)
    const Element* __restrict__ q_ptr,          // shape: (B, H, G, 64),  dtype: const half*
    const Element* __restrict__ k_ptr,          // shape: (B, H, 64),     dtype: const half*
    const Element* __restrict__ v_ptr,          // shape: (B, H, 64),     dtype: const half*
    const Element* __restrict__ b_ptr,          // shape: (B, H, 64),     dtype: const half*
    const Element* __restrict__ w_ptr,          // shape: (B, H, 64),     dtype: const half*
    const Element* __restrict__ a_ptr,          // shape: (B, H, 1),      dtype: const half*
    const Element* __restrict__ gamma_ptr,      // shape: (B, H, 1),      dtype: const half*
    Element*       __restrict__ out_ptr,        // shape: (B, H, G, 64),  dtype: half*
    int G,                                      // scalar: int
    int64_t num_q_heads,                        // scalar: int64_t
    int64_t num_kv_heads                        // scalar: int64_t
) {
    // Stage 1 - Setup & CuTe Tensor Partitioning
    int batch_idx = blockIdx.y;
    int head_idx  = blockIdx.x;
    int tid       = threadIdx.x;
    int row_id    = tid / 8; // 0..15
    int col_id    = tid % 8; // 0..7

    extern __shared__ __align__(16) uint8_t smem_raw[];
    SmemStorage& smem = *reinterpret_cast<SmemStorage*>(smem_raw);

    SmemLayoutS smem_layout_s{};
    Tensor sS   = make_tensor(make_smem_ptr(smem.sS), smem_layout_s);
    Tensor sVec = make_tensor(make_smem_ptr(smem.sVec), smem_layout_s);

    // GMEM Offsets
    int64_t offset_S        = (static_cast<int64_t>(batch_idx) * num_kv_heads + head_idx) * (D_V * D_K);
    int64_t offset_kv       = (static_cast<int64_t>(batch_idx) * num_kv_heads + head_idx) * D_K;
    int64_t offset_scalar   = static_cast<int64_t>(batch_idx) * num_kv_heads + head_idx;
    int64_t offset_q_base   = (static_cast<int64_t>(batch_idx) * num_q_heads + head_idx * G) * D_K;
    int64_t offset_out_base = (static_cast<int64_t>(batch_idx) * num_q_heads + head_idx * G) * D_V;

    // 1. GMEM Tensor for past state matrix S (64 x 64)
    Tensor gS = make_tensor(make_gmem_ptr(past_state_ptr + offset_S),
                            make_layout(make_shape(Int<D_V>{}, Int<D_K>{}),
                                        make_stride(Int<D_K>{}, Int<1>{})));

    // 2. GMEM Tensors for 1D vectors mapped to 2D broadcast layout (16 x 64, stride (0, 1))
    auto bcast_layout = make_layout(make_shape(Int<16>{}, Int<D_K>{}), make_stride(Int<0>{}, Int<1>{}));
    Tensor gK = make_tensor(make_gmem_ptr(k_ptr + offset_kv), bcast_layout);
    Tensor gB = make_tensor(make_gmem_ptr(b_ptr + offset_kv), bcast_layout);
    Tensor gV = make_tensor(make_gmem_ptr(v_ptr + offset_kv), bcast_layout);
    Tensor gW = make_tensor(make_gmem_ptr(w_ptr + offset_kv), bcast_layout);

    // 3. Thread-level partitioning for matrix S (64 x 64)
    TiledCopyS tiled_copy_s;
    auto thr_copy_s = tiled_copy_s.get_slice(tid);
    auto tSgS       = thr_copy_s.partition_S(gS);
    auto tSsS_D     = thr_copy_s.partition_D(sS);
    auto tSsS_S     = thr_copy_s.partition_S(sS);
    Tensor rS       = make_fragment_like(tSgS);
    Tensor rS_new   = make_fragment_like(tSgS);

    // 4. Thread-level partitioning for broadcast 1D vectors (16 x 64 -> 1 x 8 per thread)
    TiledCopyVec tiled_copy_vec;
    auto thr_copy_vec = tiled_copy_vec.get_slice(tid);
    auto tVgK         = thr_copy_vec.partition_S(gK);
    auto tVgB         = thr_copy_vec.partition_S(gB);
    auto tVgV         = thr_copy_vec.partition_S(gV);
    auto tVgW         = thr_copy_vec.partition_S(gW);

    Tensor r_k        = make_fragment_like(tVgK);
    Tensor r_b        = make_fragment_like(tVgB);
    Tensor r_v        = make_fragment_like(tVgV);
    Tensor r_w        = make_fragment_like(tVgW);
    Tensor r_k_scaled = make_fragment_like(tVgK);
    Tensor r_u        = make_fragment_like(tVgK);
    Tensor r_v_new    = make_fragment_like(tVgV);
    Tensor r_out      = make_fragment_like(tVgV);

    // Predication tensor for 1D vector writes to GMEM (leader thread row_id == 0 only)
    Tensor tVpOut = make_tensor<bool>(make_layout(shape(r_out), LayoutRight{}));
    #pragma unroll
    for (int i = 0; i < size(tVpOut); ++i) {
        tVpOut(i) = (row_id == 0);
    }

    // 16-byte aligned register buffer for G query vectors
    static_assert(sizeof(Element) * 8 == 16, "uint4 = 8 half elements");
    constexpr int STATIC_G = (FIXED_G > 0) ? FIXED_G : 1;
    alignas(16) Element q_buf[STATIC_G * 8];
    Tensor r_q_storage = make_tensor(
        make_rmem_ptr(q_buf),
        make_layout(make_shape(Int<STATIC_G>{}, Int<8>{}), LayoutRight{}));

    // Stage 2 - Issue Global Memory Loads
    Element raw_a     = a_ptr[offset_scalar];
    Element raw_gamma = gamma_ptr[offset_scalar];

    // Load broadcast vectors k, b, v, w via 128-bit cute::copy
    cute::copy(tiled_copy_vec, tVgK, r_k);
    cute::copy(tiled_copy_vec, tVgB, r_b);
    cute::copy(tiled_copy_vec, tVgV, r_v);
    cute::copy(tiled_copy_vec, tVgW, r_w);

    // Load tile S (64 x 64) into swizzled SMEM sS
    cute::copy(tiled_copy_s, tSgS, rS);
    cute::copy(tiled_copy_s, rS, tSsS_D);
    __syncthreads();

    // Stage 3 - Transcendental Function Evaluation
    Element decay        = fast_sigmoid_h(raw_a);
    float   gamma_sp_f   = fmaxf(__half2float(fast_softplus_h(raw_gamma)), 0.0f);
    float   sqrt_gamma_f = sqrtf(gamma_sp_f);
    Element sqrt_gamma   = __float2half(sqrt_gamma_f);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        half2 b_pair = reinterpret_cast<half2*>(r_b.data())[i];
        reinterpret_cast<half2*>(r_b.data())[i] = fast_sigmoid_h2(b_pair);

        half2 w_pair = reinterpret_cast<half2*>(r_w.data())[i];
        reinterpret_cast<half2*>(r_w.data())[i] = fast_sigmoid_h2(w_pair);
    }

    // Stage 4 - L2-Normalize Key, Compute k_scaled and u
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float k_val = __half2float(r_k(i));
        sum_sq += k_val * k_val;
    }
    sum_sq = octet_reduce_sum(sum_sq);

    // Compute scale factor in FP32 to prevent FP16 overflow/underflow
    float rsqrt_val = (sum_sq > 0.0f) ? rsqrtf(fmaxf(sum_sq, 1e-12f)) : 0.0f;
    float scale_f   = sqrt_gamma_f * rsqrt_val;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float k_val   = __half2float(r_k(i));
        r_k_scaled(i) = __float2half(k_val * scale_f);
        r_u(i)        = __hmul(r_b(i), r_k_scaled(i));
    }

    // Stage 5 - Compute v_new
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        Element v_scaled = __hmul(r_v(i), sqrt_gamma);
        r_v_new(i)       = __hmul(r_w(i), v_scaled);
    }

    // Stage 6 - Compute Su and delta_v
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        float local_su = 0.0f;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            local_su += __half2float(rS(c + r * 8)) * __half2float(r_u(c));
        }
        local_su = octet_reduce_sum(local_su);

        if (col_id == 0) {
            sVec(0, r * 16 + row_id) = __float2half(local_su);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        int col_idx    = col_id * 8 + c;
        Element su_val = sVec(0, col_idx);
        Element dv_val = __hsub(r_v_new(c), su_val);
        if (row_id == 0) {
            sVec(1, col_idx) = dv_val;
        }
    }
    __syncthreads();

    // Stage 7 - Compute S_new and Write In-place to GMEM
    Element r_delta_v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        r_delta_v[r] = sVec(1, r * 16 + row_id);
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            float y_val = __half2float(r_delta_v[r]) * __half2float(r_k_scaled(c));
            float s_val = __half2float(rS(c + r * 8)) * __half2float(decay) + y_val;
            rS_new(c + r * 8) = __float2half(s_val);
        }
    }

    // Write updated state S_new in-place to GMEM
    cute::copy(tiled_copy_s, rS_new, tSgS);

    // Stage 8 & 9 - Loop across G heads: Compute out_inter + out_intra and Write Output
    if constexpr (FIXED_G > 0) {
        #pragma unroll
        for (int g = 0; g < FIXED_G; ++g) {
            Tensor gQ_g   = make_tensor(make_gmem_ptr(q_ptr + offset_q_base + g * D_K), bcast_layout);
            auto   tVgQ_g = thr_copy_vec.partition_S(gQ_g);
            Tensor r_q_g  = make_tensor(make_rmem_ptr(&r_q_storage(g, 0)), make_layout(shape(tVgK), LayoutRight{}));
            cute::copy(tiled_copy_vec, tVgQ_g, r_q_g);
        }

        // Reload S_old from swizzled SMEM sS
        cute::copy(tiled_copy_s, tSsS_S, rS);

        #pragma unroll
        for (int g = 0; g < FIXED_G; ++g) {
            // 1. Compute out_inter = decay * (S_old * q_g)
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                float local_qs = 0.0f;
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    local_qs += __half2float(rS(c + r * 8)) * __half2float(r_q_storage(g, c));
                }
                local_qs = octet_reduce_sum(local_qs);
                float out_inter_f = local_qs * __half2float(decay);
                if (col_id == 0) {
                    sVec(2, r * 16 + row_id) = __float2half(out_inter_f);
                }
            }
            __syncthreads();

            // 2. Compute out_intra = (q_g * k_scaled) * v_new
            float local_attn = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                local_attn += __half2float(r_q_storage(g, c)) * __half2float(r_k_scaled(c));
            }
            local_attn = octet_reduce_sum(local_attn);

            // 3. Combine out_g = out_intra + out_inter
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                int col_idx           = col_id * 8 + c;
                Element out_inter_val = sVec(2, col_idx);
                float out_f = local_attn * __half2float(r_v_new(c)) + __half2float(out_inter_val);
                r_out(c) = __float2half(out_f);
            }

            // 4. Store 128-bit out_g to GMEM with leader-thread predication
            Tensor gOut_g   = make_tensor(make_gmem_ptr(out_ptr + offset_out_base + g * D_V),
                                          make_layout(make_shape(Int<16>{}, Int<D_V>{}),
                                                      make_stride(Int<D_V>{}, Int<1>{})));
            auto   tVgOut_g = thr_copy_vec.partition_D(gOut_g);
            cute::copy_if(tiled_copy_vec, tVpOut, r_out, tVgOut_g);
            __syncthreads();
        }
    } else {
        cute::copy(tiled_copy_s, tSsS_S, rS);

        for (int g = 0; g < G; ++g) {
            Tensor gQ_g   = make_tensor(make_gmem_ptr(q_ptr + offset_q_base + g * D_K), bcast_layout);
            auto   tVgQ_g = thr_copy_vec.partition_S(gQ_g);
            Tensor r_q_0  = make_tensor(make_rmem_ptr(&r_q_storage(0, 0)), make_layout(shape(tVgK), LayoutRight{}));
            cute::copy(tiled_copy_vec, tVgQ_g, r_q_0);

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                float local_qs = 0.0f;
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    local_qs += __half2float(rS(c + r * 8)) * __half2float(r_q_storage(0, c));
                }
                local_qs = octet_reduce_sum(local_qs);
                float out_inter_f = local_qs * __half2float(decay);
                if (col_id == 0) {
                    sVec(2, r * 16 + row_id) = __float2half(out_inter_f);
                }
            }
            __syncthreads();

            float local_attn = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                local_attn += __half2float(r_q_storage(0, c)) * __half2float(r_k_scaled(c));
            }
            local_attn = octet_reduce_sum(local_attn);

            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                int col_idx           = col_id * 8 + c;
                Element out_inter_val = sVec(2, col_idx);
                float out_f = local_attn * __half2float(r_v_new(c)) + __half2float(out_inter_val);
                r_out(c) = __float2half(out_f);
            }

            Tensor gOut_g   = make_tensor(make_gmem_ptr(out_ptr + offset_out_base + g * D_V),
                                          make_layout(make_shape(Int<16>{}, Int<D_V>{}),
                                                      make_stride(Int<D_V>{}, Int<1>{})));
            auto   tVgOut_g = thr_copy_vec.partition_D(gOut_g);
            cute::copy_if(tiled_copy_vec, tVpOut, r_out, tVgOut_g);
            __syncthreads();
        }
    }
}

// HOST FUNCTION (deltanet_infer)

std::tuple<at::Tensor, at::Tensor>
deltanet_infer(
    const at::Tensor& past_state,       // shape: (B, H, dv, dk),        dtype: torch::kFloat16 (in-place update)
    const at::Tensor& q,                // shape: (B, num_q_heads, dk),  dtype: torch::kFloat16
    const at::Tensor& k,                // shape: (B, H, dk),            dtype: torch::kFloat16
    const at::Tensor& v,                // shape: (B, H, dv),            dtype: torch::kFloat16
    const at::Tensor& b,                // shape: (B, H, dk),            dtype: torch::kFloat16
    const at::Tensor& w,                // shape: (B, H, dv),            dtype: torch::kFloat16
    const at::Tensor& a,                // shape: (B, H, 1) or (B, H),   dtype: torch::kFloat16
    const at::Tensor& gamma,            // shape: (B, H, 1) or (B, H),   dtype: torch::kFloat16
    int64_t num_q_heads,                // scalar: int64_t
    int64_t num_kv_heads,               // scalar: int64_t
    int64_t d_k,                        // scalar: int64_t (default 64)
    int64_t d_v                         // scalar: int64_t (default 64)
) {
    // Stage 1 - Input Validation
    TORCH_CHECK(past_state.is_cuda(), "Input tensor 'past_state' must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(past_state.device());

    TORCH_CHECK(q.is_cuda(), "Input tensor 'q' must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "Input tensor 'k' must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "Input tensor 'v' must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input tensor 'b' must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "Input tensor 'w' must be a CUDA tensor");
    TORCH_CHECK(a.is_cuda(), "Input tensor 'a' must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Input tensor 'gamma' must be a CUDA tensor");

    TORCH_CHECK(q.device() == past_state.device(), "Tensor 'q' must be on same device as 'past_state'");
    TORCH_CHECK(k.device() == past_state.device(), "Tensor 'k' must be on same device as 'past_state'");
    TORCH_CHECK(v.device() == past_state.device(), "Tensor 'v' must be on same device as 'past_state'");
    TORCH_CHECK(b.device() == past_state.device(), "Tensor 'b' must be on same device as 'past_state'");
    TORCH_CHECK(w.device() == past_state.device(), "Tensor 'w' must be on same device as 'past_state'");
    TORCH_CHECK(a.device() == past_state.device(), "Tensor 'a' must be on same device as 'past_state'");
    TORCH_CHECK(gamma.device() == past_state.device(), "Tensor 'gamma' must be on same device as 'past_state'");

    TORCH_CHECK(past_state.scalar_type() == at::ScalarType::Half, "Tensor 'past_state' must be FP16 (torch.float16)");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half, "Tensor 'q' must be FP16");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Half, "Tensor 'k' must be FP16");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::Half, "Tensor 'v' must be FP16");
    TORCH_CHECK(b.scalar_type() == at::ScalarType::Half, "Tensor 'b' must be FP16");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Half, "Tensor 'w' must be FP16");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half, "Tensor 'a' must be FP16");
    TORCH_CHECK(gamma.scalar_type() == at::ScalarType::Half, "Tensor 'gamma' must be FP16");

    TORCH_CHECK(past_state.is_contiguous(), "Tensor 'past_state' must be contiguous");
    TORCH_CHECK(q.is_contiguous(), "Tensor 'q' must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Tensor 'k' must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Tensor 'v' must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "Tensor 'w' must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor 'b' must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "Tensor 'a' must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "Tensor 'gamma' must be contiguous");

    // Enforce 16-byte alignment for 128-bit vectorized uint4 access
    TORCH_CHECK(reinterpret_cast<uintptr_t>(past_state.data_ptr()) % 16 == 0, "Tensor 'past_state' must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0, "Tensor 'q' must be 16-byte aligned (call .contiguous() after split)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0, "Tensor 'k' must be 16-byte aligned (call .contiguous() after split)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "Tensor 'v' must be 16-byte aligned (call .contiguous() after split)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0, "Tensor 'b' must be 16-byte aligned (call .contiguous() after split)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "Tensor 'w' must be 16-byte aligned (call .contiguous() after split)");

    int64_t B = past_state.size(0);
    int64_t H = num_kv_heads;

    TORCH_CHECK(d_k == 64, "DeltaNet requires d_k == 64, got ", d_k);
    TORCH_CHECK(d_v == 64, "DeltaNet requires d_v == 64, got ", d_v);
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    int64_t G = num_q_heads / num_kv_heads;

    TORCH_CHECK(past_state.dim() == 4, "Tensor 'past_state' must be 4D (B, H, dv, dk)");
    TORCH_CHECK(past_state.size(1) == H && past_state.size(2) == d_v && past_state.size(3) == d_k,
                "Tensor 'past_state' shape mismatch with (B, H, 64, 64)");

    TORCH_CHECK(q.dim() == 3 && q.size(0) == B && q.size(1) == num_q_heads && q.size(2) == d_k,
                "Tensor 'q' shape mismatch with (B, num_q_heads, 64)");

    TORCH_CHECK(k.dim() == 3 && k.size(0) == B && k.size(1) == H && k.size(2) == d_k,
                "Tensor 'k' shape mismatch with (B, H, 64)");
    TORCH_CHECK(v.dim() == 3 && v.size(0) == B && v.size(1) == H && v.size(2) == d_v,
                "Tensor 'v' shape mismatch with (B, H, 64)");
    TORCH_CHECK(b.dim() == 3 && b.size(0) == B && b.size(1) == H && b.size(2) == d_k,
                "Tensor 'b' shape mismatch with (B, H, 64)");
    TORCH_CHECK(w.dim() == 3 && w.size(0) == B && w.size(1) == H && w.size(2) == d_v,
                "Tensor 'w' shape mismatch with (B, H, 64)");

    TORCH_CHECK(a.numel() == B * H, "Tensor 'a' must have B * H elements");
    TORCH_CHECK(gamma.numel() == B * H, "Tensor 'gamma' must have B * H elements");

    // Stage 2 - Kernel Execution
    at::Tensor out = torch::empty({B, num_q_heads, d_v}, q.options());

    constexpr int NUM_THREADS = 128;
    constexpr int DK_DIM      = 64;
    constexpr int DV_DIM      = 64;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(past_state.device().index());

    {
        using Element = half;

        using SmemLayoutS = decltype(composition(
            Swizzle<3, 3, 3>{},
            make_layout(make_shape(Int<DV_DIM>{}, Int<DK_DIM>{}), make_stride(Int<DK_DIM>{}, Int<1>{}))
        ));

        using CopyAtom = Copy_Atom<UniversalCopy<uint4>, Element>;

        using TiledCopyS = decltype(make_tiled_copy(
            CopyAtom{},
            make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
            make_layout(make_shape(Int<1>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}))
        ));

        using TiledCopyVec = TiledCopyS;

        size_t smem_size = sizeof(SmemStorage);
        dim3 grid(H, B);
        dim3 block(NUM_THREADS);

        const Element* p_q     = reinterpret_cast<const Element*>(q.data_ptr<at::Half>());
        const Element* p_k     = reinterpret_cast<const Element*>(k.data_ptr<at::Half>());
        const Element* p_v     = reinterpret_cast<const Element*>(v.data_ptr<at::Half>());
        const Element* p_b     = reinterpret_cast<const Element*>(b.data_ptr<at::Half>());
        const Element* p_w     = reinterpret_cast<const Element*>(w.data_ptr<at::Half>());
        const Element* p_a     = reinterpret_cast<const Element*>(a.data_ptr<at::Half>());
        const Element* p_gamma = reinterpret_cast<const Element*>(gamma.data_ptr<at::Half>());
        Element*       p_s     = reinterpret_cast<Element*>(past_state.data_ptr<at::Half>());
        Element*       p_out   = reinterpret_cast<Element*>(out.data_ptr<at::Half>());

        switch (G) {
            case 1:
                kernel_deltanet_infer<Element, NUM_THREADS, DK_DIM, DV_DIM, 1, SmemLayoutS, TiledCopyS, TiledCopyVec>
                    <<<grid, block, smem_size, stream>>>(
                        p_s, p_q, p_k, p_v, p_b, p_w, p_a, p_gamma, p_out,
                        static_cast<int>(G), num_q_heads, num_kv_heads
                    );
                break;
            case 2:
                kernel_deltanet_infer<Element, NUM_THREADS, DK_DIM, DV_DIM, 2, SmemLayoutS, TiledCopyS, TiledCopyVec>
                    <<<grid, block, smem_size, stream>>>(
                        p_s, p_q, p_k, p_v, p_b, p_w, p_a, p_gamma, p_out,
                        static_cast<int>(G), num_q_heads, num_kv_heads
                    );
                break;
            case 4:
                kernel_deltanet_infer<Element, NUM_THREADS, DK_DIM, DV_DIM, 4, SmemLayoutS, TiledCopyS, TiledCopyVec>
                    <<<grid, block, smem_size, stream>>>(
                        p_s, p_q, p_k, p_v, p_b, p_w, p_a, p_gamma, p_out,
                        static_cast<int>(G), num_q_heads, num_kv_heads
                    );
                break;
            case 8:
                kernel_deltanet_infer<Element, NUM_THREADS, DK_DIM, DV_DIM, 8, SmemLayoutS, TiledCopyS, TiledCopyVec>
                    <<<grid, block, smem_size, stream>>>(
                        p_s, p_q, p_k, p_v, p_b, p_w, p_a, p_gamma, p_out,
                        static_cast<int>(G), num_q_heads, num_kv_heads
                    );
                break;
            default:
                kernel_deltanet_infer<Element, NUM_THREADS, DK_DIM, DV_DIM, 0, SmemLayoutS, TiledCopyS, TiledCopyVec>
                    <<<grid, block, smem_size, stream>>>(
                        p_s, p_q, p_k, p_v, p_b, p_w, p_a, p_gamma, p_out,
                        static_cast<int>(G), num_q_heads, num_kv_heads
                    );
                break;
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Stage 3 - Return Results
    return std::make_tuple(out, past_state);
}
