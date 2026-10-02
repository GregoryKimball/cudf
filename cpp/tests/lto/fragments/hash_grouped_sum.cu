/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/lto/udf_abi.h>

// SELECT c0, c1 % 3, SUM(c1), COUNT(*) FROM t WHERE c1 % 11 != 0 GROUP BY c0, c1 % 3.
struct hash_grouped_sum_state {
  long long sum;
  long long count;
};

extern "C" __device__ void cudf_lto_reduce_init(void* state)
{
  *static_cast<hash_grouped_sum_state*>(state) = hash_grouped_sum_state{0, 0};
}

extern "C" __device__ int cudf_lto_hash_groupby_row(void const*,
                                                    cudf_lto_row const* row,
                                                    long long* key,
                                                    void* row_state)
{
  auto const value = cudf_lto_get(row, 1);
  if (value % 11 == 0) { return 0; }
  key[0]                                           = cudf_lto_get(row, 0);
  key[1]                                           = ((value % 3) + 3) % 3;
  *static_cast<hash_grouped_sum_state*>(row_state) = hash_grouped_sum_state{value, 1};
  return 1;
}

extern "C" __device__ void cudf_lto_reduce_merge(void* state, void const* other)
{
  auto* const s       = static_cast<hash_grouped_sum_state*>(state);
  auto const* const o = static_cast<hash_grouped_sum_state const*>(other);
  s->sum += o->sum;
  s->count += o->count;
}
