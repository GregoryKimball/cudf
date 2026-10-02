/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "for_bitpack.cuh"

// SELECT row, c1, c0 WHERE c0 < *user_data, over FOR+bitpack tiles, where row is the row's
// position in the source. Outputs are INT64, INT64, and INT32; c1 is read only for kept rows.
extern "C" __device__ int cudf_lto_select_row(void const* user_data,
                                              cudf_lto_tile const* tile,
                                              cudf_lto_u32 row)
{
  return for_bitpack_get(tile->columns[0], row) < *static_cast<long long const*>(user_data);
}

extern "C" __device__ void cudf_lto_select_emit(void const*,
                                                cudf_lto_tile const* tile,
                                                cudf_lto_u32 row,
                                                void* const* outputs,
                                                cudf_lto_u64 output_row)
{
  static_cast<long long*>(outputs[0])[output_row] = static_cast<long long>(tile->first_row + row);
  static_cast<long long*>(outputs[1])[output_row] = for_bitpack_get(tile->columns[1], row);
  static_cast<int*>(outputs[2])[output_row] =
    static_cast<int>(for_bitpack_get(tile->columns[0], row));
}
