#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>

using namespace cute;

// Half/half2 transcendental helpers (clamped to prevent FP16 overflow/underflow)

__device__ __forceinline__ __half fast_softplus_h(__half x) {
    __half limit = __float2half(10.0f);
    if (__hgt(x, limit)) {
        return x; // Prevent FP16 overflow (>65504)
    }
    __half one = __float2half(1.0f);
    return hlog(__hadd(one, hexp(x)));
}

__device__ __forceinline__ __half2 fast_softplus_h2(__half2 x) {
    __half2 limit = __float2half2_rn(10.0f);
    __half2 one = __float2half2_rn(1.0f);
    __half2 x_clamped = __hmin2(x, limit);
    __half2 exp_val = h2exp(x_clamped);
    __half2 sp = h2log(__hadd2(one, exp_val));
    __half low = __hgt(__low2half(x), __float2half(10.0f)) ? __low2half(x) : __low2half(sp);
    __half high = __hgt(__high2half(x), __float2half(10.0f)) ? __high2half(x) : __high2half(sp);
    return __halves2half2(low, high);
}

__device__ __forceinline__ __half fast_sigmoid_h(__half x) {
    __half limit_neg = __float2half(-10.0f);
    __half limit_pos = __float2half(10.0f);
    __half x_clamped = __hmax(__hmin(x, limit_pos), limit_neg);
    __half one = __float2half(1.0f);
    return __hdiv(one, __hadd(one, hexp(__hneg(x_clamped))));
}

__device__ __forceinline__ __half2 fast_sigmoid_h2(__half2 x) {
    __half2 limit_neg = __float2half2_rn(-10.0f);
    __half2 limit_pos = __float2half2_rn(10.0f);
    __half2 x_clamped = __hmax2(__hmin2(x, limit_pos), limit_neg);
    __half2 one = __float2half2_rn(1.0f);
    __half2 exp_val = h2exp(__hneg2(x_clamped));
    return __h2div(one, __hadd2(one, exp_val));
}

__device__ __forceinline__ float stable_log_sigmoid(float a) {
    if (a >= 0.0f) {
        return -log1pf(expf(-a));
    } else {
        return a - log1pf(expf(a));
    }
}

#ifndef DELTANET_SIGMOID_HALF2_DEFINED
#define DELTANET_SIGMOID_HALF2_DEFINED

// Fast half2 sigmoid using base-2 exponential and reciprocal
__device__ __forceinline__ half2 sigmoid_half2(half2 x) {
    half2 neg_log2e = __float2half2_rn(-1.442695f);
    half2 scaled_x  = __hmul2(x, neg_log2e);
    half2 exp_neg_x = h2exp2(scaled_x);
    half2 one       = __float2half2_rn(1.0f);
    half2 denom     = __hadd2(one, exp_neg_x);
    return h2rcp(denom);
}

// Warp-level sum reduction in FP32
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Transpose 4x4 uint32_t matrix across a 4-lane quad for contiguous 128-bit stores
__device__ __forceinline__ void quad_transpose_4x4_u32(
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3, int lane_in_quad
) {
    // Swap bit 0 between adjacent lanes
    uint32_t s0 = ((lane_in_quad & 1) == 0) ? v1 : v0;
    uint32_t s1 = ((lane_in_quad & 1) == 0) ? v3 : v2;
    uint32_t r0 = __shfl_xor_sync(0xffffffff, s0, 1);
    uint32_t r1 = __shfl_xor_sync(0xffffffff, s1, 1);
    if ((lane_in_quad & 1) == 0) {
        v1 = r0;
        v3 = r1;
    } else {
        v0 = r0;
        v2 = r1;
    }

    // Swap bit 1 between lane pairs separated by 2
    uint32_t s2 = ((lane_in_quad & 2) == 0) ? v2 : v0;
    uint32_t s3 = ((lane_in_quad & 2) == 0) ? v3 : v1;
    uint32_t r2 = __shfl_xor_sync(0xffffffff, s2, 2);
    uint32_t r3 = __shfl_xor_sync(0xffffffff, s3, 2);
    if ((lane_in_quad & 2) == 0) {
        v2 = r2;
        v3 = r3;
    } else {
        v0 = r2;
        v1 = r3;
    }
}
#endif

// Type conversion helpers compatible with -D__CUDA_NO_HALF_CONVERSIONS__

template <typename InT>
__device__ __forceinline__ float half_to_float(InT val) {
    return __half2float(reinterpret_cast<const __half&>(val));
}

template <typename InT>
__device__ __forceinline__ __half to_half_val(InT val) {
    return reinterpret_cast<const __half&>(val);
}

template <typename OutT>
__device__ __forceinline__ OutT float_to_out(float val) {
    __half h = __float2half(val);
    return reinterpret_cast<const OutT&>(h);
}

template <typename OutT>
__device__ __forceinline__ OutT half_to_out(__half h) {
    return reinterpret_cast<const OutT&>(h);
}

// Clone only when a contiguous tensor is not 16-byte aligned for 128-bit uint4 loads.
inline at::Tensor ensure_aligned16(const at::Tensor& t) {
    if (t.numel() == 0 || (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0) {
        return t;
    }
    return t.clone();
}

inline at::Tensor compute_cu_chunk_lens(
    const at::Tensor& cu_seq_lens,
    int64_t chunk_size = 64
) {
    int64_t batch_size = cu_seq_lens.size(0) - 1;
    at::Tensor cu_seq_lens_i32 = cu_seq_lens.to(at::kInt).contiguous();
    at::Tensor cu_seq_lens_cpu = cu_seq_lens_i32.cpu();
    const int32_t* h_cu = cu_seq_lens_cpu.data_ptr<int32_t>();

    std::vector<int32_t> h_cu_chunk_lens;
    h_cu_chunk_lens.reserve(batch_size + 1);
    int32_t cur_chunks = 0;
    h_cu_chunk_lens.push_back(0);

    for (int64_t b = 0; b < batch_size; ++b) {
        int32_t start = h_cu[b];
        int32_t end   = h_cu[b + 1];
        int32_t len   = end - start;
        int32_t num_c = (len > 0) ? (len + chunk_size - 1) / chunk_size : 0;
        cur_chunks += num_c;
        h_cu_chunk_lens.push_back(cur_chunks);
    }

    return torch::from_blob(
        h_cu_chunk_lens.data(), {batch_size + 1}, at::kInt
    ).clone().to(cu_seq_lens.device());
}

inline std::tuple<at::Tensor, at::Tensor, int64_t> compute_chunk_offsets_and_valid_lens(
    const at::Tensor& cu_seq_lens,
    int64_t chunk_size = 64
) {
    int64_t batch_size = cu_seq_lens.size(0) - 1;
    at::Tensor cu_seq_lens_i32 = cu_seq_lens.to(at::kInt).contiguous();
    at::Tensor cu_seq_lens_cpu = cu_seq_lens_i32.cpu();
    const int32_t* h_cu = cu_seq_lens_cpu.data_ptr<int32_t>();

    std::vector<int32_t> h_chunk_offsets;
    std::vector<int32_t> h_chunk_valid_lens;
    h_chunk_offsets.reserve(batch_size * 16 + 1);
    h_chunk_valid_lens.reserve(batch_size * 16);

    for (int64_t b = 0; b < batch_size; ++b) {
        int32_t start = h_cu[b];
        int32_t end   = h_cu[b + 1];
        int32_t len   = end - start;
        if (len <= 0) continue;
        int32_t num_c = (len + chunk_size - 1) / chunk_size;
        for (int32_t c = 0; c < num_c; ++c) {
            int32_t c_start = start + c * chunk_size;
            h_chunk_offsets.push_back(c_start);
            int32_t rem = end - c_start;
            h_chunk_valid_lens.push_back(rem < chunk_size ? rem : (int32_t)chunk_size);
        }
    }
    h_chunk_offsets.push_back(h_cu[batch_size]);
    int64_t num_chunks = (int64_t)h_chunk_valid_lens.size();

    at::Tensor chunk_offsets_gpu = torch::from_blob(
        h_chunk_offsets.data(), {(int64_t)h_chunk_offsets.size()}, at::kInt
    ).clone().to(cu_seq_lens.device());

    at::Tensor chunk_valid_lens_gpu = torch::from_blob(
        h_chunk_valid_lens.data(), {num_chunks}, at::kInt
    ).clone().to(cu_seq_lens.device());

    return std::make_tuple(chunk_offsets_gpu, chunk_valid_lens_gpu, num_chunks);
}
