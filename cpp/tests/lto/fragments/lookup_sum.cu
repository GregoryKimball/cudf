/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "for_bitpack.cuh"

// SELECT b.g, SUM(t.c1), COUNT(*) FROM t JOIN b ON t.c0 = b.key GROUP BY b.g, over FOR+bitpack
// tiles, where b is reached through a lookup in user data.
struct lookup_sum_data {
  cudf_lto_lookup lookup;
  int const* groups;
};

struct lookup_sum_state {
  long long sum;
  long long count;
};

extern "C" __device__ void cudf_lto_reduce_init(void* state)
{
  *static_cast<lookup_sum_state*>(state) = lookup_sum_state{0, 0};
}

extern "C" __device__ cudf_lto_u32 cudf_lto_groupby_row(void const* user_data,
                                                        cudf_lto_tile const* tile,
                                                        cudf_lto_u32 row,
                                                        void* row_state)
{
  auto const* const data = static_cast<lookup_sum_data const*>(user_data);
  auto const build_row =
    cudf_lto_lookup_find(&data->lookup, for_bitpack_get(tile->columns[0], row));
  if (build_row == CUDF_LTO_NOT_FOUND) { return CUDF_LTO_SKIP_ROW; }
  *static_cast<lookup_sum_state*>(row_state) =
    lookup_sum_state{for_bitpack_get(tile->columns[1], row), 1};
  return static_cast<cudf_lto_u32>(data->groups[build_row]);
}

extern "C" __device__ void cudf_lto_reduce_merge(void* state, void const* other)
{
  auto* const s       = static_cast<lookup_sum_state*>(state);
  auto const* const o = static_cast<lookup_sum_state const*>(other);
  s->sum += o->sum;
  s->count += o->count;
}
