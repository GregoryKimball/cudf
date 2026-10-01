/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "for_bitpack.cuh"

namespace {

__device__ long long load_signed(void const* values, cudf_lto_u32 i, cudf_lto_u32 element_bytes)
{
  switch (element_bytes) {
    case 1: return static_cast<signed char const*>(values)[i];
    case 2: return static_cast<short const*>(values)[i];
    case 4: return static_cast<int const*>(values)[i];
    default: return static_cast<long long const*>(values)[i];
  }
}

__device__ void store(void* values, cudf_lto_u32 i, cudf_lto_u32 element_bytes, long long value)
{
  switch (element_bytes) {
    case 1: static_cast<signed char*>(values)[i] = static_cast<signed char>(value); break;
    case 2: static_cast<short*>(values)[i] = static_cast<short>(value); break;
    case 4: static_cast<int*>(values)[i] = static_cast<int>(value); break;
    default: static_cast<long long*>(values)[i] = value; break;
  }
}

/// Block-wide minimum and maximum; every thread receives both.
__device__ void block_min_max(long long& lo, long long& hi, void* scratch)
{
  for (int offset = 16; offset > 0; offset /= 2) {
    lo = min(lo, __shfl_down_sync(0xffffffffU, lo, offset));
    hi = max(hi, __shfl_down_sync(0xffffffffU, hi, offset));
  }
  auto* const partial = static_cast<long long*>(scratch);
  auto const warp     = threadIdx.x / 32;
  if (threadIdx.x % 32 == 0) {
    partial[2 * warp]     = lo;
    partial[2 * warp + 1] = hi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (cudf_lto_u32 w = 1; w < (blockDim.x + 31) / 32; ++w) {
      lo = min(lo, partial[2 * w]);
      hi = max(hi, partial[2 * w + 1]);
    }
    partial[64] = lo;
    partial[65] = hi;
  }
  __syncthreads();
  lo = partial[64];
  hi = partial[65];
  __syncthreads();
}

}  // namespace

extern "C" __device__ cudf_lto_u32 cudf_lto_encode_chunk(void const* input,
                                                         cudf_lto_u32 input_bytes,
                                                         cudf_lto_u32 element_bytes,
                                                         void* output,
                                                         cudf_lto_u32 output_capacity,
                                                         void* scratch)
{
  if (element_bytes != 1 && element_bytes != 2 && element_bytes != 4 && element_bytes != 8) {
    return 0;
  }
  auto const rows = input_bytes / element_bytes;
  long long lo    = 0x7fffffffffffffffLL;
  long long hi    = -lo - 1;
  for (cudf_lto_u32 i = threadIdx.x; i < rows; i += blockDim.x) {
    auto const value = load_signed(input, i, element_bytes);
    lo               = min(lo, value);
    hi               = max(hi, value);
  }
  block_min_max(lo, hi, scratch);
  if (rows == 0) { lo = hi = 0; }

  auto const range = static_cast<unsigned long long>(hi) - static_cast<unsigned long long>(lo);
  cudf_lto_u32 const width = range == 0 ? 0 : 64 - __clzll(static_cast<long long>(range));
  auto const num_words     = (static_cast<unsigned long long>(rows) * width + 63) / 64;
  auto const bytes         = for_bitpack_header_bytes + num_words * 8;
  if (bytes > output_capacity) { return 0; }

  auto* const chunk = static_cast<unsigned char*>(output);
  if (threadIdx.x == 0) {
    *reinterpret_cast<for_bitpack_header*>(chunk) = for_bitpack_header{lo, width, rows};
  }
  auto* const words = reinterpret_cast<unsigned long long*>(chunk + for_bitpack_header_bytes);
  for (unsigned long long k = threadIdx.x; k < num_words; k += blockDim.x) {
    auto const first_bit = k * 64;
    auto const first     = static_cast<cudf_lto_u32>(first_bit / width);
    auto const last      = static_cast<cudf_lto_u32>(
      min(static_cast<unsigned long long>(rows) - 1, (first_bit + 63) / width));
    unsigned long long word = 0;
    for (cudf_lto_u32 i = first; i <= last; ++i) {
      auto const delta = static_cast<unsigned long long>(load_signed(input, i, element_bytes)) -
                         static_cast<unsigned long long>(lo);
      auto const position = static_cast<long long>(i) * width - static_cast<long long>(first_bit);
      if (position >= 0) {
        word |= delta << position;
      } else {
        word |= delta >> -position;
      }
    }
    words[k] = word;
  }
  return static_cast<cudf_lto_u32>(bytes);
}

extern "C" __device__ int cudf_lto_decode_chunk(void const* input,
                                                cudf_lto_u32,
                                                cudf_lto_u32 element_bytes,
                                                void* output,
                                                cudf_lto_u32 output_bytes,
                                                void*)
{
  auto const* chunk = static_cast<unsigned char const*>(input);
  auto const rows   = output_bytes / element_bytes;
  if (reinterpret_cast<for_bitpack_header const*>(chunk)->rows != rows) { return 1; }
  for (cudf_lto_u32 i = threadIdx.x; i < rows; i += blockDim.x) {
    store(output, i, element_bytes, for_bitpack_get(chunk, i));
  }
  return 0;
}
