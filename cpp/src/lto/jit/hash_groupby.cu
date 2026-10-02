/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <lto/jit/tile.cuh>

namespace {

/**
 * Merges `row_state` into the slot of `key` in the global table, claiming an empty slot for a new
 * key. Merges into one slot are serialized by its status word, since `cudf_lto_reduce_merge` is
 * not atomic. The atomic status operations with `__threadfence()` form release and acquire
 * patterns, so a thread that observes a ready slot also observes its key and state.
 *
 * @return False when every slot holds another key
 */
__device__ bool merge_into_table(cudf_lto_hash_groupby_args const& args,
                                 long long const* key,
                                 unsigned char const* row_state)
{
  auto const num_words = args.num_key_words;
  auto slot            = cudf_lto_hash_words(key, num_words);
  for (cudf_lto_u64 probe = 0; probe <= args.capacity_mask; ++probe, ++slot) {
    auto const s       = slot & args.capacity_mask;
    auto* const status = args.status + s;
    auto* const words  = args.keys + s * num_words;
    auto* const state  = args.states + s * args.state_bytes;
    auto observed      = atomicCAS(status, CUDF_LTO_SLOT_EMPTY, CUDF_LTO_SLOT_CLAIMED);
    if (observed == CUDF_LTO_SLOT_EMPTY) {
      for (cudf_lto_u32 w = 0; w < num_words; ++w) {
        words[w] = key[w];
      }
      cudf_lto_reduce_init(state);
      cudf_lto_reduce_merge(state, row_state);
      __threadfence();
      atomicExch(status, CUDF_LTO_SLOT_READY);
      return true;
    }
    while (observed == CUDF_LTO_SLOT_CLAIMED) {
      observed = atomicAdd(status, 0U);
    }
    __threadfence();
    bool same = true;
    for (cudf_lto_u32 w = 0; w < num_words; ++w) {
      same = same && __ldcg(words + w) == key[w];
    }
    if (!same) { continue; }
    while (atomicCAS(status, CUDF_LTO_SLOT_READY, CUDF_LTO_SLOT_LOCKED) != CUDF_LTO_SLOT_READY) {}
    __threadfence();
    cudf_lto_reduce_merge(state, row_state);
    __threadfence();
    atomicExch(status, CUDF_LTO_SLOT_READY);
    return true;
  }
  return false;
}

}  // namespace

// Hash groupby into a global-memory table of keyed slots, for more groups than shared memory
// holds. Lanes of a warp with equal keys first merge into one row state, so each distinct key of a
// warp-wide step touches the table once.
//
// Dynamic shared memory: [staged tile][one row state per thread].
extern "C" __global__ void __launch_bounds__(CUDF_LTO_BLOCK_SIZE)
  cudf_kernel_entry(cudf_lto_hash_groupby_args const args)
{
  extern __shared__ uint4 dynamic_smem[];
  auto* const smem = reinterpret_cast<unsigned char*>(dynamic_smem);
  __shared__ cudf_lto_tile tile;

  auto const state_bytes = args.state_bytes;
  auto const num_words   = args.num_key_words;
  auto* const row_states = smem + args.source.tile_bytes;
  auto* const row_state  = row_states + threadIdx.x * state_bytes;
  auto const lane        = threadIdx.x % 32;
  auto const warp_base   = threadIdx.x - lane;

  for (cudf_lto_u64 t = blockIdx.x; t < args.source.num_tiles; t += gridDim.x) {
    cudf_lto_kernel::load_tile(args.source, t, smem, tile);
    __syncthreads();
    for (cudf_lto_u32 base = 0; base < tile.num_rows; base += blockDim.x) {
      auto const row = base + threadIdx.x;
      long long key[CUDF_LTO_MAX_KEY_WORDS];
      auto const current = cudf_lto_kernel::row_of(tile, row);
      bool const kept =
        row < tile.num_rows && cudf_lto_hash_groupby_row(args.user_data, &current, key, row_state);
      __syncwarp();
      auto const hash   = kept ? cudf_lto_hash_words(key, num_words) : 0ULL;
      auto const peers  = __match_any_sync(0xffffffffU, hash) & __ballot_sync(0xffffffffU, kept);
      auto const leader = kept ? static_cast<unsigned int>(__ffs(peers) - 1) : lane;
      bool same         = kept;
      for (cudf_lto_u32 w = 0; w < num_words; ++w) {
        same = __shfl_sync(0xffffffffU, key[w], leader) == key[w] && same;
      }
      auto const merged = __ballot_sync(0xffffffffU, same && leader != lane);
      if (kept && leader == lane) {
        for (auto remaining = peers & merged; remaining != 0; remaining &= remaining - 1) {
          auto const peer = warp_base + static_cast<unsigned int>(__ffs(remaining) - 1);
          cudf_lto_reduce_merge(row_state, row_states + peer * state_bytes);
        }
      }
      __syncwarp();
      if (kept && ((merged >> lane) & 1U) == 0 && !merge_into_table(args, key, row_state)) {
        atomicExch(args.overflow, 1U);
      }
      __syncwarp();
    }
    __syncthreads();
  }
}
