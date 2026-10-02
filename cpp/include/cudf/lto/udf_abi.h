/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file udf_abi.h
 * @brief Device ABI between libcudf's precompiled LTO kernel fragments and caller fragments.
 *
 * libcudf ships each kernel as a precompiled fragment in LTO-IR. A caller supplies LTO-IR
 * fragments defining the `extern "C"` symbols a kernel calls; libcudf links them with nvJitLink at
 * runtime. This header is the complete contract: it includes nothing, so fragments can be compiled
 * by nvcc, NVRTC, or any other frontend that emits LTO-IR.
 *
 * The contract has two independent halves:
 *
 * - The codec ABI is chunk-relative. A codec encodes and decodes whole chunks and reads one value
 *   at an index within a chunk. Only codec authors see it.
 * - The UDF ABI is row-relative. A row program reads column `c` of the current row with
 *   `cudf_lto_get(row, c)`. libcudf resolves the row's chunk, stages it, and calls that column's
 *   codec reader; LTO inlines the call. How rows are tiled and staged is internal to libcudf.
 *
 * Every hook marked block-cooperative is called by all threads of a block with identical
 * arguments, and may use `__syncthreads()`.
 */

#ifndef CUDF_LTO_UDF_ABI_H
#define CUDF_LTO_UDF_ABI_H

#define CUDF_LTO_ABI_VERSION 3

/// Bytes of block-shared scratch memory passed to block-cooperative codec hooks.
#define CUDF_LTO_SCRATCH_BYTES 2048
/// Largest per-thread reduction state.
#define CUDF_LTO_MAX_STATE_BYTES 128
/// Most columns a reduction can consume.
#define CUDF_LTO_MAX_COLUMNS 32
/// Alignment of every encoded chunk in a packed payload and of every staged chunk.
#define CUDF_LTO_CHUNK_ALIGNMENT 16
/// Slot a groupby row program returns for a row it drops.
#define CUDF_LTO_SKIP_ROW 0xffffffffU
/// Row `cudf_lto_lookup_find` returns for an absent key.
#define CUDF_LTO_NOT_FOUND 0xffffffffU
/// Most 64-bit words in a hash groupby or hash lookup key.
#define CUDF_LTO_MAX_KEY_WORDS 4

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

/// The current row of a row program; opaque, read only through the functions below.
typedef struct cudf_lto_row cudf_lto_row;

/**
 * @brief Dense map from unique integer build keys to build rows, made by `lto::make_lookup_table`,
 * or dense set of integer keys, made by `lto::make_key_set`.
 *
 * Bit `k` of `bits` is set when key `min_key + k` is present, `ranks[w]` counts the set bits
 * before word `w`, and `rows` holds the build row of each present key in key order. A key set has
 * no `ranks` or `rows`. All pointers are device memory. A row program reaches a lookup through its
 * `user_data`.
 */
typedef struct cudf_lto_lookup {
  cudf_lto_u64 const* bits;
  cudf_lto_u32 const* ranks;
  cudf_lto_u32 const* rows;
  long long min_key;
  cudf_lto_u64 num_keys;  ///< Width of the key range; zero when the build side is empty
} cudf_lto_lookup;

/**
 * @brief Hash map from composite integer build keys to their build rows, made by
 * `lto::make_hash_lookup_table`.
 *
 * Build rows are grouped by key. Group `g` has key words `keys[w * num_groups + g]` and build rows
 * `rows[offsets[g]]` to `rows[offsets[g + 1] - 1]`; a key-only lookup has no `offsets` or `rows`.
 * `slots` is an open-addressing table of `capacity_mask + 1` group indices, `CUDF_LTO_NOT_FOUND`
 * for an empty slot, probed linearly from `cudf_lto_hash_words(key) & capacity_mask`. All pointers
 * are device memory.
 */
typedef struct cudf_lto_hash_lookup {
  cudf_lto_u32 const* slots;
  long long const* keys;
  cudf_lto_u32 const* offsets;
  cudf_lto_u32 const* rows;
  cudf_lto_u64 capacity_mask;
  cudf_lto_u32 num_groups;  ///< Distinct keys; zero when the build side is empty
  cudf_lto_u32 num_key_words;
} cudf_lto_hash_lookup;

#ifdef __CUDACC__

/// Hash of a key of `num_words` 64-bit words; libcudf builds and probes hash tables with it.
static __device__ __forceinline__ cudf_lto_u64 cudf_lto_hash_words(long long const* key,
                                                                   cudf_lto_u32 num_words)
{
  cudf_lto_u64 h = 0x9e3779b97f4a7c15ULL;
  for (cudf_lto_u32 w = 0; w < num_words; ++w) {
    h ^= (cudf_lto_u64)key[w];
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 31;
  }
  h *= 0x94d049bb133111ebULL;
  return h ^ (h >> 32);
}

/// The group holding `key`, or `CUDF_LTO_NOT_FOUND`.
static __device__ __forceinline__ cudf_lto_u32
cudf_lto_hash_find(cudf_lto_hash_lookup const* lookup, long long const* key)
{
  if (lookup->num_groups == 0) { return CUDF_LTO_NOT_FOUND; }
  cudf_lto_u32 const n = lookup->num_key_words;
  for (cudf_lto_u64 s = cudf_lto_hash_words(key, n);; ++s) {
    cudf_lto_u32 const group = lookup->slots[s & lookup->capacity_mask];
    if (group == CUDF_LTO_NOT_FOUND) { return CUDF_LTO_NOT_FOUND; }
    cudf_lto_u32 w = 0;
    while (w < n && lookup->keys[(cudf_lto_u64)w * lookup->num_groups + group] == key[w]) {
      ++w;
    }
    if (w == n) { return group; }
  }
}

/// Whether some build row has `key`.
static __device__ __forceinline__ int cudf_lto_hash_contains(cudf_lto_hash_lookup const* lookup,
                                                             long long const* key)
{
  return cudf_lto_hash_find(lookup, key) != CUDF_LTO_NOT_FOUND;
}

/// Whether a dense lookup or key set holds `key`.
static __device__ __forceinline__ int cudf_lto_lookup_contains(cudf_lto_lookup const* lookup,
                                                               long long key)
{
  cudf_lto_u64 const k = (cudf_lto_u64)key - (cudf_lto_u64)lookup->min_key;
  return k < lookup->num_keys && ((lookup->bits[k >> 6] >> (k & 63)) & 1ULL) != 0;
}

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

/* ---- Codec ABI: chunk-relative. ------------------------------------------------------------ */

/*
 * A codec fragment defines `cudf_lto_encode_chunk` and `cudf_lto_decode_chunk`. A separate reader
 * fragment defines the codec's random-access reader under a codec-specific name, so that readers of
 * several codecs can be linked into one kernel:
 *
 *   long long <reader>(unsigned char const* chunk, cudf_lto_u32 index);
 *
 * It returns value `index` of an encoded chunk, sign-extended to 64 bits. `chunk` is aligned to
 * `CUDF_LTO_CHUNK_ALIGNMENT` and holds exactly the bytes `cudf_lto_encode_chunk` wrote; it may be
 * in shared, global, or mapped host memory.
 */

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

/* ---- UDF ABI: row-relative. -------------------------------------------------------------- */

/**
 * @brief Value of column `column` in the current row, sign-extended to 64 bits.
 *
 * Defined by libcudf for each kernel. Reading only the columns a row needs keeps columns read in
 * place from being fetched for other rows.
 */
__device__ long long cudf_lto_get(cudf_lto_row const* row, cudf_lto_u32 column);

/// Index of the current row within the whole input; defined by libcudf.
__device__ cudf_lto_u64 cudf_lto_row_index(cudf_lto_row const* row);

/* ---- Reduction hooks: one row-program fragment defines all three. ------------------------ */

/// Initializes a per-thread state of the requested size to the reduction's identity.
__device__ void cudf_lto_reduce_init(void* state);

/**
 * @brief Folds row `row` into `state`.
 *
 * Called once per row by one thread.
 */
__device__ void cudf_lto_reduce_row(void const* user_data, cudf_lto_row const* row, void* state);

/// Combines `other` into `state`; must be associative and commutative.
__device__ void cudf_lto_reduce_merge(void* state, void const* other);

/* ---- Groupby hook: a groupby row program defines this plus `_init` and `_merge`. ---------- */

/**
 * @brief Classifies row `row` into a dense group slot.
 *
 * Called once per row by one thread. For a kept row, writes the complete state of that row alone
 * to `row_state` and returns its slot, below the `num_groups` passed to the groupby; the kernel
 * merges it into the slot with `cudf_lto_reduce_merge`.
 *
 * @return The row's slot, or `CUDF_LTO_SKIP_ROW` to drop the row
 */
__device__ cudf_lto_u32 cudf_lto_groupby_row(void const* user_data,
                                             cudf_lto_row const* row,
                                             void* row_state);

/* ---- Hash groupby hook: a hash groupby row program defines this plus `_init` and `_merge`. - */

/**
 * @brief Classifies row `row` into a group by key.
 *
 * Called once per row by one thread. For a kept row, writes the key's `num_key_words` words to
 * `key` and the complete state of that row alone to `row_state`; the kernel merges it into the
 * key's group with `cudf_lto_reduce_merge`.
 *
 * @return Nonzero to keep the row
 */
__device__ int cudf_lto_hash_groupby_row(void const* user_data,
                                         cudf_lto_row const* row,
                                         long long* key,
                                         void* row_state);

/* ---- Filtered-scan hooks: one select row program defines both. --------------------------- */

/**
 * @brief Decides whether row `row` is kept.
 *
 * Called once per row by one thread. It should read only the columns the decision needs, so that
 * columns read in place are fetched for kept rows alone.
 *
 * @return Nonzero to keep the row
 */
__device__ int cudf_lto_select_row(void const* user_data, cudf_lto_row const* row);

/**
 * @brief Writes kept row `row` as row `output_row` of every output column.
 *
 * Called once per kept row by one thread. `outputs[k]` is the data of output column `k`, of the
 * fixed-width type the caller passed to the filtered scan.
 */
__device__ void cudf_lto_select_emit(void const* user_data,
                                     cudf_lto_row const* row,
                                     void* const* outputs,
                                     cudf_lto_u64 output_row);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* CUDF_LTO_UDF_ABI_H */
