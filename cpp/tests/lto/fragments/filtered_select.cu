/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/lto/udf_abi.h>

// SELECT row, c1, c0 WHERE c0 < *user_data, where row is the row's position in the source. Outputs
// are INT64, INT64, and INT32; c1 is read only for kept rows.
extern "C" __device__ int cudf_lto_select_row(void const* user_data, cudf_lto_row const* row)
{
  return cudf_lto_get(row, 0) < *static_cast<long long const*>(user_data);
}

extern "C" __device__ void cudf_lto_select_emit(void const*,
                                                cudf_lto_row const* row,
                                                void* const* outputs,
                                                cudf_lto_u64 output_row)
{
  static_cast<long long*>(outputs[0])[output_row] = static_cast<long long>(cudf_lto_row_index(row));
  static_cast<long long*>(outputs[1])[output_row] = cudf_lto_get(row, 1);
  static_cast<int*>(outputs[2])[output_row]       = static_cast<int>(cudf_lto_get(row, 0));
}
