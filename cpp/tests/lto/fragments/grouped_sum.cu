/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "for_bitpack.cuh"

// SELECT c0, SUM(c1), COUNT(*) WHERE c1 % 7 != 0 GROUP BY c0, where c0 holds dense group slots
// (for example dictionary codes), over FOR+bitpack tiles.
struct grouped_sum_state {
  long long sum;
  long long count;
};

extern "C" __device__ void cudf_lto_reduce_init(void* state)
{
  *static_cast<grouped_sum_state*>(state) = grouped_sum_state{0, 0};
}

extern "C" __device__ cudf_lto_u32 cudf_lto_groupby_row(void const*,
                                                        cudf_lto_tile const* tile,
                                                        cudf_lto_u32 row,
                                                        void* row_state)
{
  auto const value = for_bitpack_get(tile->columns[1], row);
  if (value % 7 == 0) { return CUDF_LTO_SKIP_ROW; }
  *static_cast<grouped_sum_state*>(row_state) = grouped_sum_state{value, 1};
  return static_cast<cudf_lto_u32>(for_bitpack_get(tile->columns[0], row));
}

extern "C" __device__ void cudf_lto_reduce_merge(void* state, void const* other)
{
  auto* const s       = static_cast<grouped_sum_state*>(state);
  auto const* const o = static_cast<grouped_sum_state const*>(other);
  s->sum += o->sum;
  s->count += o->count;
}
