/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/lto/udf_abi.h>

/// Threads per block of every tile kernel.
#define CUDF_LTO_BLOCK_SIZE 256

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

/**
 * @brief One row-aligned tile: the same rows of every column, as a block sees them.
 *
 * `columns[c]` points to column `c`'s encoded chunk: staged in shared memory, or for a lazy
 * column, in place in the packed payload.
 */
typedef struct cudf_lto_tile {
  cudf_lto_u64 first_row;  ///< Row of the whole input that is row 0 of this tile
  cudf_lto_u32 num_rows;
  cudf_lto_u32 num_columns;
  unsigned char const* columns[CUDF_LTO_MAX_COLUMNS];
  cudf_lto_u32 column_bytes[CUDF_LTO_MAX_COLUMNS];
} cudf_lto_tile;

/// The row a row program is called for. `cudf_lto_row_source` in `udf.cu` repeats this layout.
struct cudf_lto_row {
  unsigned char const* const* columns;  ///< Encoded chunk of each column of the row's tile
  cudf_lto_u64 first_row;               ///< Input row of the tile's first row
  cudf_lto_u32 index;                   ///< Row within the tile
};

/// The tiles of a packed source and where a block stages each tile's columns.
typedef struct cudf_lto_tile_source {
  cudf_lto_tile_desc const* tiles;
  cudf_lto_chunk_ref const* chunks;  ///< `num_columns` per tile, tile-major
  cudf_lto_u64 num_tiles;
  cudf_lto_u32 num_columns;
  cudf_lto_u32 tile_bytes;                            ///< Shared memory holding one staged tile
  cudf_lto_u32 column_offsets[CUDF_LTO_MAX_COLUMNS];  ///< Shared-memory offset of each column
  cudf_lto_u32 lazy_columns;  ///< Bit `c` set when column `c` is read in place, not staged
} cudf_lto_tile_source;

typedef struct cudf_lto_reduce_args {
  cudf_lto_tile_source source;
  cudf_lto_u32 state_bytes;
  cudf_lto_u32 num_groups;  ///< Group slots; groupby only
  void const* user_data;
  unsigned char* partials;  ///< One state per block, or per block and group for groupby
  unsigned int* counter;    ///< Zero before launch; counts finished blocks
  unsigned char* result;    ///< Final state, or one per group for groupby
} cudf_lto_reduce_args;

/// Slot states of a hash groupby table.
#define CUDF_LTO_SLOT_EMPTY   0U
#define CUDF_LTO_SLOT_CLAIMED 1U  ///< Key and state being initialized
#define CUDF_LTO_SLOT_READY   2U
#define CUDF_LTO_SLOT_LOCKED  3U  ///< A thread is merging into the state

typedef struct cudf_lto_hash_groupby_args {
  cudf_lto_tile_source source;
  cudf_lto_u32 state_bytes;
  cudf_lto_u32 num_key_words;
  void const* user_data;
  cudf_lto_u64 capacity_mask;  ///< Slots minus one; the slot count is a power of two
  unsigned int* status;        ///< Per slot; `CUDF_LTO_SLOT_EMPTY` before launch
  long long* keys;             ///< `num_key_words` words per slot, slot-major
  unsigned char* states;       ///< One state per slot
  unsigned int* overflow;      ///< Zero before launch; set when a row found no free slot
} cudf_lto_hash_groupby_args;

typedef struct cudf_lto_select_args {
  cudf_lto_tile_source source;
  void const* user_data;
  void* const* outputs;   ///< Data of each output column
  cudf_lto_u64 capacity;  ///< Rows each output column holds
  cudf_lto_u64* kept;     ///< Zero before launch; counts every kept row, even beyond `capacity`
} cudf_lto_select_args;
