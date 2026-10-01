/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <lto/jit/reduce_args.h>

namespace {

__device__ void copy_state(unsigned char* destination,
                           unsigned char const* source,
                           cudf_lto_u32 bytes)
{
  for (cudf_lto_u32 i = 0; i < bytes; ++i) {
    destination[i] = source[i];
  }
}

/// Block-cooperative copy of one encoded chunk into shared memory with 16-byte loads.
__device__ void stage(cudf_lto_chunk_ref const chunk, unsigned char* destination)
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

}  // namespace

// Grid-stride over tiles: stage every column's encoded chunk of a tile into shared memory, fold
// each row into a per-thread state, then merge states across the block and, in the last block to
// finish, across the grid.
extern "C" __global__ void __launch_bounds__(CUDF_LTO_REDUCE_BLOCK_SIZE)
  cudf_kernel_entry(cudf_lto_reduce_args const args)
{
  extern __shared__ uint4 dynamic_smem[];
  auto* const smem = reinterpret_cast<unsigned char*>(dynamic_smem);
  __shared__ cudf_lto_tile tile;
  __shared__ bool is_last_block;

  __align__(16) unsigned char state[CUDF_LTO_MAX_STATE_BYTES];
  cudf_lto_reduce_init(state);

  for (cudf_lto_u64 t = blockIdx.x; t < args.num_tiles; t += gridDim.x) {
    auto const* const chunks = args.chunks + t * args.num_columns;
    if (threadIdx.x < args.num_columns) {
      tile.columns[threadIdx.x]      = smem + args.column_offsets[threadIdx.x];
      tile.column_bytes[threadIdx.x] = static_cast<cudf_lto_u32>(chunks[threadIdx.x].bytes);
    }
    if (threadIdx.x == 0) {
      tile.first_row   = args.tiles[t].first_row;
      tile.num_rows    = args.tiles[t].num_rows;
      tile.num_columns = args.num_columns;
    }
    for (cudf_lto_u32 c = 0; c < args.num_columns; ++c) {
      stage(chunks[c], smem + args.column_offsets[c]);
    }
    __syncthreads();
    for (cudf_lto_u32 row = threadIdx.x; row < tile.num_rows; row += blockDim.x) {
      cudf_lto_reduce_row(args.user_data, &tile, row, state);
    }
    __syncthreads();
  }

  auto* const slots = smem;
  copy_state(slots + threadIdx.x * args.state_bytes, state, args.state_bytes);
  __syncthreads();
  for (unsigned int half = CUDF_LTO_REDUCE_BLOCK_SIZE / 2; half > 0; half /= 2) {
    if (threadIdx.x < half) {
      cudf_lto_reduce_merge(slots + threadIdx.x * args.state_bytes,
                            slots + (threadIdx.x + half) * args.state_bytes);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    copy_state(args.partials + blockIdx.x * args.state_bytes, slots, args.state_bytes);
    __threadfence();
    is_last_block = atomicAdd(args.counter, 1U) == gridDim.x - 1;
  }
  __syncthreads();
  if (is_last_block && threadIdx.x == 0) {
    __threadfence();
    cudf_lto_reduce_init(state);
    for (unsigned int block = 0; block < gridDim.x; ++block) {
      cudf_lto_reduce_merge(state, args.partials + block * args.state_bytes);
    }
    copy_state(args.result, state, args.state_bytes);
    *args.counter = 0;
  }
}
