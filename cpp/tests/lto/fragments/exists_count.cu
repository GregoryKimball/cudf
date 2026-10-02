/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/lto/udf_abi.h>

// Counts the rows of t whose c0 is (semi) and is not (anti) among the build keys, once through a
// dense key set and once through a key-only hash lookup.
struct exists_count_data {
  cudf_lto_lookup set;
  cudf_lto_hash_lookup keys;
};

struct exists_count_state {
  long long dense_semi;
  long long dense_anti;
  long long hash_semi;
  long long hash_anti;
};

extern "C" __device__ void cudf_lto_reduce_init(void* state)
{
  *static_cast<exists_count_state*>(state) = exists_count_state{0, 0, 0, 0};
}

extern "C" __device__ void cudf_lto_reduce_row(void const* user_data,
                                               cudf_lto_row const* row,
                                               void* state)
{
  auto const* const data = static_cast<exists_count_data const*>(user_data);
  auto* const s          = static_cast<exists_count_state*>(state);
  auto const key         = cudf_lto_get(row, 0);
  if (cudf_lto_lookup_contains(&data->set, key)) {
    s->dense_semi += 1;
  } else {
    s->dense_anti += 1;
  }
  if (cudf_lto_hash_contains(&data->keys, &key)) {
    s->hash_semi += 1;
  } else {
    s->hash_anti += 1;
  }
}

extern "C" __device__ void cudf_lto_reduce_merge(void* state, void const* other)
{
  auto* const s       = static_cast<exists_count_state*>(state);
  auto const* const o = static_cast<exists_count_state const*>(other);
  s->dense_semi += o->dense_semi;
  s->dense_anti += o->dense_anti;
  s->hash_semi += o->hash_semi;
  s->hash_anti += o->hash_anti;
}
