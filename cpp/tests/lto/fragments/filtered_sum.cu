/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "for_bitpack.cuh"

// SELECT SUM(c1), COUNT(*) WHERE c0 < *user_data, over FOR+bitpack tiles. c1 is read only for
// rows that pass the predicate.
struct filtered_sum_state {
  long long sum;
  long long count;
};

extern "C" __device__ void cudf_lto_reduce_init(void* state)
{
  *static_cast<filtered_sum_state*>(state) = filtered_sum_state{0, 0};
}

extern "C" __device__ void cudf_lto_reduce_row(void const* user_data,
                                               cudf_lto_tile const* tile,
                                               cudf_lto_u32 row,
                                               void* state)
{
  auto const threshold = *static_cast<long long const*>(user_data);
  if (for_bitpack_get(tile->columns[0], row) < threshold) {
    auto* const s = static_cast<filtered_sum_state*>(state);
    s->sum += for_bitpack_get(tile->columns[1], row);
    s->count += 1;
  }
}

extern "C" __device__ void cudf_lto_reduce_merge(void* state, void const* other)
{
  auto* const s       = static_cast<filtered_sum_state*>(state);
  auto const* const o = static_cast<filtered_sum_state const*>(other);
  s->sum += o->sum;
  s->count += o->count;
}
