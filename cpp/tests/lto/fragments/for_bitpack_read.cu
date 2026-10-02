/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "for_bitpack.cuh"

extern "C" __device__ long long test_for_bitpack_read(unsigned char const* chunk,
                                                      cudf_lto_u32 index)
{
  return for_bitpack_get(chunk, index);
}
