/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/lto/udf_abi.h>

// SELECT SUM(c1), COUNT(*) WHERE c0 < *user_data. c1 is read only for rows that pass the
// predicate.
struct filtered_sum_state {
  long long sum;
  long long count;
};

extern "C" __device__ void cudf_lto_reduce_init(void* state)
{
  *static_cast<filtered_sum_state*>(state) = filtered_sum_state{0, 0};
}

extern "C" __device__ void cudf_lto_reduce_row(void const* user_data,
                                               cudf_lto_row const* row,
                                               void* state)
{
  auto const threshold = *static_cast<long long const*>(user_data);
  if (cudf_lto_get(row, 0) < threshold) {
    auto* const s = static_cast<filtered_sum_state*>(state);
    s->sum += cudf_lto_get(row, 1);
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
