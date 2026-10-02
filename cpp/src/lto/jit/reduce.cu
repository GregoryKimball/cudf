/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <lto/jit/tile.cuh>

using cudf_lto_kernel::copy_state;

// Grid-stride over tiles: stage every column's encoded chunk of a tile into shared memory, fold
// each row into a per-thread state, then merge states across the block and, in the last block to
// finish, across the grid.
extern "C" __global__ void __launch_bounds__(CUDF_LTO_BLOCK_SIZE)
  cudf_kernel_entry(cudf_lto_reduce_args const args)
{
  extern __shared__ uint4 dynamic_smem[];
  auto* const smem = reinterpret_cast<unsigned char*>(dynamic_smem);
  __shared__ cudf_lto_tile tile;
  __shared__ bool is_last_block;

  __align__(16) unsigned char state[CUDF_LTO_MAX_STATE_BYTES];
  cudf_lto_reduce_init(state);

  for (cudf_lto_u64 t = blockIdx.x; t < args.source.num_tiles; t += gridDim.x) {
    cudf_lto_kernel::load_tile(args.source, t, smem, tile);
    __syncthreads();
    for (cudf_lto_u32 row = threadIdx.x; row < tile.num_rows; row += blockDim.x) {
      cudf_lto_reduce_row(args.user_data, &tile, row, state);
    }
    __syncthreads();
  }

  auto* const slots = smem;
  copy_state(slots + threadIdx.x * args.state_bytes, state, args.state_bytes);
  __syncthreads();
  for (unsigned int half = CUDF_LTO_BLOCK_SIZE / 2; half > 0; half /= 2) {
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
