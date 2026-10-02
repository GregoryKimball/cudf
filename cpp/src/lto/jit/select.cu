/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <lto/jit/tile.cuh>

namespace {
constexpr unsigned int warps_per_block = CUDF_LTO_BLOCK_SIZE / 32;
constexpr unsigned int segment_steps   = 32;
constexpr unsigned int segment_words   = segment_steps * warps_per_block;
constexpr unsigned int segment_rows    = segment_words * 32;
}  // namespace

// Filtered scan. A block stages a tile, then walks it in segments of `segment_rows` rows. Each
// segment evaluates the row predicate into one keep bit per row, ranks the kept rows with a block
// scan, reserves their output rows with one atomic, and emits them in row order. Output order
// across segments is unspecified. Rows beyond `capacity` are counted but not written.
//
// Word `w` of a segment holds the keep bits of rows `w * 32` to `w * 32 + 31`, so thread `t` of
// the scan owns word `t` and words are in row order.
extern "C" __global__ void __launch_bounds__(CUDF_LTO_BLOCK_SIZE)
  cudf_kernel_entry(cudf_lto_select_args const args)
{
  static_assert(segment_words == CUDF_LTO_BLOCK_SIZE, "one scanned word per thread");
  extern __shared__ uint4 dynamic_smem[];
  auto* const smem = reinterpret_cast<unsigned char*>(dynamic_smem);
  __shared__ cudf_lto_tile tile;
  __shared__ unsigned int masks[segment_words];
  __shared__ unsigned int offsets[segment_words];
  __shared__ unsigned int warp_offsets[warps_per_block];
  __shared__ cudf_lto_u64 segment_base;

  auto const lane = threadIdx.x % 32;
  auto const warp = threadIdx.x / 32;

  for (cudf_lto_u64 t = blockIdx.x; t < args.source.num_tiles; t += gridDim.x) {
    cudf_lto_kernel::load_tile(args.source, t, smem, tile);
    __syncthreads();
    for (cudf_lto_u32 segment = 0; segment < tile.num_rows; segment += segment_rows) {
      for (unsigned int step = 0; step < segment_steps; ++step) {
        auto const row     = segment + step * CUDF_LTO_BLOCK_SIZE + threadIdx.x;
        auto const current = cudf_lto_kernel::row_of(tile, row);
        auto const keep = row < tile.num_rows && cudf_lto_select_row(args.user_data, &current) != 0;
        auto const mask = __ballot_sync(0xffffffffU, keep);
        if (lane == 0) { masks[step * warps_per_block + warp] = mask; }
      }
      __syncthreads();

      auto const count = static_cast<unsigned int>(__popc(masks[threadIdx.x]));
      auto inclusive   = count;
      for (unsigned int delta = 1; delta < 32; delta *= 2) {
        auto const other = __shfl_up_sync(0xffffffffU, inclusive, delta);
        if (lane >= delta) { inclusive += other; }
      }
      if (lane == 31) { warp_offsets[warp] = inclusive; }
      __syncthreads();
      if (threadIdx.x == 0) {
        unsigned int total = 0;
        for (unsigned int w = 0; w < warps_per_block; ++w) {
          auto const warp_total = warp_offsets[w];
          warp_offsets[w]       = total;
          total += warp_total;
        }
        segment_base =
          total == 0 ? 0 : atomicAdd(reinterpret_cast<unsigned long long*>(args.kept), total);
      }
      __syncthreads();
      offsets[threadIdx.x] = warp_offsets[warp] + inclusive - count;
      __syncthreads();

      for (unsigned int step = 0; step < segment_steps; ++step) {
        auto const word = step * warps_per_block + warp;
        auto const bits = masks[word];
        if ((bits >> lane) & 1U) {
          auto const output_row = segment_base + offsets[word] + __popc(bits & ((1U << lane) - 1U));
          if (output_row < args.capacity) {
            auto const current =
              cudf_lto_kernel::row_of(tile, segment + step * CUDF_LTO_BLOCK_SIZE + threadIdx.x);
            cudf_lto_select_emit(args.user_data, &current, args.outputs, output_row);
          }
        }
      }
      __syncthreads();
    }
  }
}
