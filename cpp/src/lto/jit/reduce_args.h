/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/lto/udf_abi.h>

/// Threads per block of the reduce kernel.
#define CUDF_LTO_REDUCE_BLOCK_SIZE 256

/// One encoded chunk of one column.
typedef struct cudf_lto_chunk_ref {
  unsigned char const* data;
  cudf_lto_u64 bytes;
} cudf_lto_chunk_ref;

typedef struct cudf_lto_tile_desc {
  cudf_lto_u64 first_row;
  cudf_lto_u32 num_rows;
  cudf_lto_u32 reserved;
} cudf_lto_tile_desc;

typedef struct cudf_lto_reduce_args {
  cudf_lto_tile_desc const* tiles;
  cudf_lto_chunk_ref const* chunks;  ///< `num_columns` per tile, tile-major
  cudf_lto_u64 num_tiles;
  cudf_lto_u32 num_columns;
  cudf_lto_u32 state_bytes;
  cudf_lto_u32 column_offsets[CUDF_LTO_MAX_COLUMNS];  ///< Shared-memory offset of each column
  cudf_lto_u32 tile_bytes;                            ///< Shared memory holding one staged tile
  cudf_lto_u32 num_groups;                            ///< Group slots; groupby only
  void const* user_data;
  unsigned char* partials;  ///< One state per block, or per block and group for groupby
  unsigned int* counter;    ///< Zero before launch; counts finished blocks
  unsigned char* result;    ///< Final state, or one per group for groupby
} cudf_lto_reduce_args;
