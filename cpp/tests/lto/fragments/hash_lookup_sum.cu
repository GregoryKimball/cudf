/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/lto/udf_abi.h>

// SELECT SUM(t.c2 * b.v), COUNT(*) FROM t JOIN b ON (t.c0, t.c1) = (b.k0, b.k1), where b has
// repeated keys and is reached through a hash lookup in user data.
struct hash_lookup_sum_data {
  cudf_lto_hash_lookup lookup;
  long long const* values;
};

struct hash_lookup_sum_state {
  long long sum;
  long long count;
};

extern "C" __device__ void cudf_lto_reduce_init(void* state)
{
  *static_cast<hash_lookup_sum_state*>(state) = hash_lookup_sum_state{0, 0};
}

extern "C" __device__ void cudf_lto_reduce_row(void const* user_data,
                                               cudf_lto_row const* row,
                                               void* state)
{
  auto const* const data = static_cast<hash_lookup_sum_data const*>(user_data);
  long long const key[2] = {cudf_lto_get(row, 0), cudf_lto_get(row, 1)};
  auto const group       = cudf_lto_hash_find(&data->lookup, key);
  if (group == CUDF_LTO_NOT_FOUND) { return; }
  auto* const s         = static_cast<hash_lookup_sum_state*>(state);
  auto const multiplier = cudf_lto_get(row, 2);
  for (auto i = data->lookup.offsets[group]; i < data->lookup.offsets[group + 1]; ++i) {
    s->sum += multiplier * data->values[data->lookup.rows[i]];
    s->count += 1;
  }
}

extern "C" __device__ void cudf_lto_reduce_merge(void* state, void const* other)
{
  auto* const s       = static_cast<hash_lookup_sum_state*>(state);
  auto const* const o = static_cast<hash_lookup_sum_state const*>(other);
  s->sum += o->sum;
  s->count += o->count;
}
