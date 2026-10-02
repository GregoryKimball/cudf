/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <lto/jit/reduce_args.h>

namespace cudf_lto_kernel {

__device__ inline void copy_state(unsigned char* destination,
                                  unsigned char const* source,
                                  cudf_lto_u32 bytes)
{
  for (cudf_lto_u32 i = 0; i < bytes; ++i) {
    destination[i] = source[i];
  }
}

/// Block-cooperative copy of one encoded chunk into shared memory with 16-byte loads.
__device__ inline void stage(cudf_lto_chunk_ref const chunk, unsigned char* destination)
{
  constexpr int loads_per_thread = 4;
  auto const* source             = reinterpret_cast<uint4 const*>(chunk.data);
  auto* target                   = reinterpret_cast<uint4*>(destination);
  auto const num_vectors         = static_cast<cudf_lto_u32>(chunk.bytes / sizeof(uint4));
  auto const stride              = blockDim.x * loads_per_thread;
  for (cudf_lto_u32 base = threadIdx.x; base < num_vectors; base += stride) {
    uint4 values[loads_per_thread];
#pragma unroll
    for (int k = 0; k < loads_per_thread; ++k) {
      auto const i = base + k * blockDim.x;
      if (i < num_vectors) { values[k] = source[i]; }
    }
#pragma unroll
    for (int k = 0; k < loads_per_thread; ++k) {
      auto const i = base + k * blockDim.x;
      if (i < num_vectors) { target[i] = values[k]; }
    }
  }
  auto const tail_begin = static_cast<cudf_lto_u64>(num_vectors) * sizeof(uint4);
  if (tail_begin + threadIdx.x < chunk.bytes) {
    destination[tail_begin + threadIdx.x] = chunk.data[tail_begin + threadIdx.x];
  }
}

/// Block-cooperative: describes tile `t` in `tile` and stages its chunks at `smem`.
__device__ inline void load_tile(cudf_lto_reduce_args const& args,
                                 cudf_lto_u64 t,
                                 unsigned char* smem,
                                 cudf_lto_tile& tile)
{
  auto const* const chunks = args.chunks + t * args.num_columns;
  if (threadIdx.x < args.num_columns) {
    auto const lazy = (args.lazy_columns >> threadIdx.x) & 1U;
    tile.columns[threadIdx.x] =
      lazy ? chunks[threadIdx.x].data : smem + args.column_offsets[threadIdx.x];
    tile.column_bytes[threadIdx.x] = static_cast<cudf_lto_u32>(chunks[threadIdx.x].bytes);
  }
  if (threadIdx.x == 0) {
    tile.first_row   = args.tiles[t].first_row;
    tile.num_rows    = args.tiles[t].num_rows;
    tile.num_columns = args.num_columns;
  }
  for (cudf_lto_u32 c = 0; c < args.num_columns; ++c) {
    if (((args.lazy_columns >> c) & 1U) == 0) { stage(chunks[c], smem + args.column_offsets[c]); }
  }
}

}  // namespace cudf_lto_kernel
