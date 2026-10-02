/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <lto/jit/tile.cuh>

using cudf_lto_kernel::copy_state;

namespace {
constexpr unsigned int warps_per_block = CUDF_LTO_BLOCK_SIZE / 32;
}

// Direct groupby into dense slots. Each warp owns one shared-memory state per slot. After every
// warp-wide step of rows, the lanes that hit the same slot are matched and their leader merges
// their row states into the warp's slot, so no two lanes write one state concurrently and no
// atomics are needed. Warp slots merge per block, and the last block merges across the grid.
//
// Dynamic shared memory: [staged tile][warp slot states][one row state per thread].
extern "C" __global__ void __launch_bounds__(CUDF_LTO_BLOCK_SIZE)
  cudf_kernel_entry(cudf_lto_reduce_args const args)
{
  extern __shared__ uint4 dynamic_smem[];
  auto* const smem = reinterpret_cast<unsigned char*>(dynamic_smem);
  __shared__ cudf_lto_tile tile;
  __shared__ bool is_last_block;

  auto const state_bytes  = args.state_bytes;
  auto const num_groups   = args.num_groups;
  auto* const warp_states = smem + args.source.tile_bytes;
  auto* const row_states  = warp_states + warps_per_block * num_groups * state_bytes;
  auto const lane         = threadIdx.x % 32;
  auto const warp         = threadIdx.x / 32;
  auto* const row_state   = row_states + threadIdx.x * state_bytes;

  for (unsigned int i = threadIdx.x; i < warps_per_block * num_groups; i += blockDim.x) {
    cudf_lto_reduce_init(warp_states + i * state_bytes);
  }

  for (cudf_lto_u64 t = blockIdx.x; t < args.source.num_tiles; t += gridDim.x) {
    cudf_lto_kernel::load_tile(args.source, t, smem, tile);
    __syncthreads();
    for (cudf_lto_u32 base = 0; base < tile.num_rows; base += blockDim.x) {
      auto const row    = base + threadIdx.x;
      cudf_lto_u32 slot = CUDF_LTO_SKIP_ROW;
      if (row < tile.num_rows) {
        auto const current = cudf_lto_kernel::row_of(tile, row);
        slot               = cudf_lto_groupby_row(args.user_data, &current, row_state);
        if (slot >= num_groups) { slot = CUDF_LTO_SKIP_ROW; }
      }
      auto const peers = __match_any_sync(0xffffffffU, slot);
      __syncwarp();
      if (slot != CUDF_LTO_SKIP_ROW && lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
        auto* const target = warp_states + (warp * num_groups + slot) * state_bytes;
        for (auto remaining = peers; remaining != 0; remaining &= remaining - 1) {
          auto const peer = warp * 32 + static_cast<unsigned int>(__ffs(remaining) - 1);
          cudf_lto_reduce_merge(target, row_states + peer * state_bytes);
        }
      }
      __syncwarp();
    }
    __syncthreads();
  }

  for (unsigned int group = threadIdx.x; group < num_groups; group += blockDim.x) {
    auto* const target = warp_states + group * state_bytes;
    for (unsigned int w = 1; w < warps_per_block; ++w) {
      cudf_lto_reduce_merge(target, warp_states + (w * num_groups + group) * state_bytes);
    }
    copy_state(
      args.partials + (blockIdx.x * num_groups + group) * state_bytes, target, state_bytes);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) { is_last_block = atomicAdd(args.counter, 1U) == gridDim.x - 1; }
  __syncthreads();
  if (!is_last_block) { return; }
  __threadfence();
  __align__(16) unsigned char state[CUDF_LTO_MAX_STATE_BYTES];
  for (unsigned int group = threadIdx.x; group < num_groups; group += blockDim.x) {
    cudf_lto_reduce_init(state);
    for (unsigned int block = 0; block < gridDim.x; ++block) {
      cudf_lto_reduce_merge(state, args.partials + (block * num_groups + group) * state_bytes);
    }
    copy_state(args.result + group * state_bytes, state, state_bytes);
  }
  if (threadIdx.x == 0) { *args.counter = 0; }
}
