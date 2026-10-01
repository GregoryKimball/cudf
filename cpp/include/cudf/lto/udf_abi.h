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
 * `columns[c]` points to column `c`'s encoded chunk staged in shared memory, aligned to
 * `CUDF_LTO_CHUNK_ALIGNMENT`. Its bytes are exactly what the column's codec wrote for this chunk.
 */
typedef struct cudf_lto_tile {
  cudf_lto_u64 first_row;  ///< Row of the whole input that is row 0 of this tile
  cudf_lto_u32 num_rows;
  cudf_lto_u32 num_columns;
  unsigned char const* columns[CUDF_LTO_MAX_COLUMNS];
  cudf_lto_u32 column_bytes[CUDF_LTO_MAX_COLUMNS];
} cudf_lto_tile;

#ifdef __CUDACC__

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

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* CUDF_LTO_UDF_ABI_H */
