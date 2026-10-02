/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file udf_abi.h
 * @brief Device ABI between libcudf's precompiled LTO kernel fragments and caller UDFs.
 *
 * libcudf ships each kernel as a precompiled fragment in LTO-IR. A caller supplies LTO-IR
 * fragments whose UDFs define the `extern "C"` symbols a kernel calls; libcudf links the two with
 * nvJitLink at runtime. This header is the complete contract: it includes nothing, so fragments
 * can be compiled by nvcc, NVRTC, or any other frontend that emits LTO-IR.
 *
 * Every hook marked block-cooperative is called by all threads of a block with identical
 * arguments, and may use `__syncthreads()`.
 */

#ifndef CUDF_LTO_UDF_ABI_H
#define CUDF_LTO_UDF_ABI_H

#define CUDF_LTO_ABI_VERSION 1

/// Bytes of block-shared scratch memory passed to block-cooperative codec hooks.
#define CUDF_LTO_SCRATCH_BYTES 2048
/// Largest per-thread reduction state.
#define CUDF_LTO_MAX_STATE_BYTES 128
/// Most columns a reduction can consume.
#define CUDF_LTO_MAX_COLUMNS 32
/// Alignment of every encoded chunk in a packed payload and of every staged tile column.
#define CUDF_LTO_CHUNK_ALIGNMENT 16
/// Slot a groupby row program returns for a row it drops.
#define CUDF_LTO_SKIP_ROW 0xffffffffU
/// Row `cudf_lto_lookup_find` returns for an absent key.
#define CUDF_LTO_NOT_FOUND 0xffffffffU

typedef unsigned int cudf_lto_u32;
typedef unsigned long long cudf_lto_u64;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief One chunk for a codec kernel: encode reads `input_bytes` raw values and writes at most
 * `output_bytes`; decode reads `input_bytes` encoded bytes and writes exactly `output_bytes`.
 */
typedef struct cudf_lto_codec_job {
  void const* input;
  void* output;
  cudf_lto_u64 input_bytes;
  cudf_lto_u64 output_bytes;
  cudf_lto_u32 element_bytes;  ///< Width of one raw value: 1, 2, 4, 8, or 16
  cudf_lto_u32 reserved;
} cudf_lto_codec_job;

/**
 * @brief One row-aligned tile of a reduction: the same rows of every consumed column.
 *
 * `columns[c]` points to column `c`'s encoded chunk, aligned to `CUDF_LTO_CHUNK_ALIGNMENT`: staged
 * in shared memory, or for a column the caller marked lazy, in place in the packed payload (device
 * or mapped host memory), so only the bytes the row program reads are fetched. Its bytes are
 * exactly what the column's codec wrote for this chunk.
 */
typedef struct cudf_lto_tile {
  cudf_lto_u64 first_row;  ///< Row of the whole input that is row 0 of this tile
  cudf_lto_u32 num_rows;
  cudf_lto_u32 num_columns;
  unsigned char const* columns[CUDF_LTO_MAX_COLUMNS];
  cudf_lto_u32 column_bytes[CUDF_LTO_MAX_COLUMNS];
} cudf_lto_tile;

/**
 * @brief Dense map from unique integer build keys to build rows, made by `lto::make_lookup_table`.
 *
 * Bit `k` of `bits` is set when key `min_key + k` is present, `ranks[w]` counts the set bits
 * before word `w`, and `rows` holds the build row of each present key in key order. All pointers
 * are device memory. A row program reaches a lookup through its `user_data`.
 */
typedef struct cudf_lto_lookup {
  cudf_lto_u64 const* bits;
  cudf_lto_u32 const* ranks;
  cudf_lto_u32 const* rows;
  long long min_key;
  cudf_lto_u64 num_keys;  ///< Width of the key range; zero when the build side is empty
} cudf_lto_lookup;

#ifdef __CUDACC__

/// The build row holding `key`, or `CUDF_LTO_NOT_FOUND`.
static __device__ __forceinline__ cudf_lto_u32 cudf_lto_lookup_find(cudf_lto_lookup const* lookup,
                                                                    long long key)
{
  cudf_lto_u64 const k = (cudf_lto_u64)key - (cudf_lto_u64)lookup->min_key;
  if (k >= lookup->num_keys) { return CUDF_LTO_NOT_FOUND; }
  cudf_lto_u64 const word = lookup->bits[k >> 6];
  cudf_lto_u64 const bit  = 1ULL << (k & 63);
  if ((word & bit) == 0) { return CUDF_LTO_NOT_FOUND; }
  return lookup->rows[lookup->ranks[k >> 6] + (cudf_lto_u32)__popcll(word & (bit - 1))];
}

/* ---- Codec hooks: one codec fragment defines both. --------------------------------------- */

/**
 * @brief Block-cooperative: encodes one chunk of raw values.
 *
 * @param input Raw values in device memory
 * @param input_bytes Bytes of raw values; a multiple of `element_bytes`
 * @param element_bytes Width of one value
 * @param output Destination in device memory, aligned to `CUDF_LTO_CHUNK_ALIGNMENT`
 * @param output_capacity Bytes available at `output`
 * @param scratch `CUDF_LTO_SCRATCH_BYTES` of shared memory
 * @return Bytes written, the same in every thread; zero reports failure
 */
__device__ cudf_lto_u32 cudf_lto_encode_chunk(void const* input,
                                              cudf_lto_u32 input_bytes,
                                              cudf_lto_u32 element_bytes,
                                              void* output,
                                              cudf_lto_u32 output_capacity,
                                              void* scratch);

/**
 * @brief Block-cooperative: decodes one chunk written by `cudf_lto_encode_chunk`.
 *
 * `input` may be device memory or mapped host memory.
 *
 * @return Zero on success, the same in every thread
 */
__device__ int cudf_lto_decode_chunk(void const* input,
                                     cudf_lto_u32 input_bytes,
                                     cudf_lto_u32 element_bytes,
                                     void* output,
                                     cudf_lto_u32 output_bytes,
                                     void* scratch);

/* ---- Reduction hooks: one row-program fragment defines all three. ------------------------ */

/// Initializes a per-thread state of the requested size to the reduction's identity.
__device__ void cudf_lto_reduce_init(void* state);

/**
 * @brief Folds row `row` of `tile` into `state`.
 *
 * Called once per row by one thread. The fragment owns how columns are decoded: it typically
 * reads `tile->columns[c]` with the same codec's random-access accessor.
 */
__device__ void cudf_lto_reduce_row(void const* user_data,
                                    cudf_lto_tile const* tile,
                                    cudf_lto_u32 row,
                                    void* state);

/// Combines `other` into `state`; must be associative and commutative.
__device__ void cudf_lto_reduce_merge(void* state, void const* other);

/* ---- Groupby hook: a groupby row program defines this plus `_init` and `_merge`. ---------- */

/**
 * @brief Classifies row `row` of `tile` into a dense group slot.
 *
 * Called once per row by one thread. For a kept row, writes the complete state of that row alone
 * to `row_state` and returns its slot, below the `num_groups` passed to the groupby; the kernel
 * merges it into the slot with `cudf_lto_reduce_merge`.
 *
 * @return The row's slot, or `CUDF_LTO_SKIP_ROW` to drop the row
 */
__device__ cudf_lto_u32 cudf_lto_groupby_row(void const* user_data,
                                             cudf_lto_tile const* tile,
                                             cudf_lto_u32 row,
                                             void* row_state);

/* ---- Filtered-scan hooks: one select row program defines both. --------------------------- */

/**
 * @brief Decides whether row `row` of `tile` is kept.
 *
 * Called once per row by one thread. It should read only the columns the decision needs, so that
 * columns read in place are fetched for kept rows alone.
 *
 * @return Nonzero to keep the row
 */
__device__ int cudf_lto_select_row(void const* user_data,
                                   cudf_lto_tile const* tile,
                                   cudf_lto_u32 row);

/**
 * @brief Writes kept row `row` of `tile` as row `output_row` of every output column.
 *
 * Called once per kept row by one thread. `outputs[k]` is the data of output column `k`, of the
 * fixed-width type the caller passed to the filtered scan.
 */
__device__ void cudf_lto_select_emit(void const* user_data,
                                     cudf_lto_tile const* tile,
                                     cudf_lto_u32 row,
                                     void* const* outputs,
                                     cudf_lto_u64 output_row);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* CUDF_LTO_UDF_ABI_H */
