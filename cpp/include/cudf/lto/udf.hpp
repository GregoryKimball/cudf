/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/lto/udf_abi.h>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

/**
 * @file udf.hpp
 * @brief LTO extensibility points: precompiled kernel fragments linked with caller fragments.
 *
 * Callers pass LTO-IR fragments whose UDFs implement the hooks declared in `cudf/lto/udf_abi.h`.
 * libcudf owns grids, tiling, shared-memory staging, output allocation, and merging; UDFs own
 * encodings, accessors, predicates, and aggregation state.
 */

namespace CUDF_EXPORT cudf {
namespace experimental::lto {

/**
 * @brief A caller-compiled fragment containing UDFs, as passed to `transform_lto`: LTO-IR, or a
 * fatbin containing LTO-IR.
 */
struct udf {
  std::span<uint8_t const> binary;                ///< Bytes; must outlive every call using them
  lto_binary_type type{lto_binary_type::LTO_IR};  ///< Binary format of `binary`
};

/**
 * @brief A caller-defined chunk codec used by `pack_compression::lto` regions.
 *
 * The fragment defines the `cudf_lto_encode_chunk` and `cudf_lto_decode_chunk` UDFs.
 */
struct codec {
  std::vector<uint8_t> binary;                    ///< Owned fragment bytes
  lto_binary_type type{lto_binary_type::LTO_IR};  ///< Binary format of `binary`
  /// Upper bound on encoded bytes for a chunk of the given raw size
  std::function<std::size_t(std::size_t)> max_encoded_bytes;
};

/**
 * @brief Registers `codec` under `id` for every later pack and materialize in this process.
 *
 * The id is stored in packed metadata, so the same id must name the same codec wherever the
 * payload is read.
 *
 * @throws std::invalid_argument if `id` is zero, already registered, or `codec` is incomplete
 */
void register_codec(uint32_t id, codec codec);

/**
 * @return Whether a codec is registered under `id`
 * @param id Codec id
 */
[[nodiscard]] bool is_codec_registered(uint32_t id);

/**
 * @brief Row-aligned tiles over packed columns, prepared once for repeated reductions.
 *
 * Built from partitions of separately packed columns. Every consumed column region must use
 * `pack_compression::lto` and, within a partition, hold the same number of rows per chunk; each
 * chunk index is then one tile. The packed buffers are borrowed and must outlive the source.
 */
class packed_source {
 public:
  packed_source(packed_source&&) noexcept;             ///< Move constructor
  packed_source& operator=(packed_source&&) noexcept;  ///< Move assignment @return `*this`
  ~packed_source();

  [[nodiscard]] std::size_t num_columns() const;  ///< @return Columns per tile
  [[nodiscard]] std::size_t num_tiles() const;    ///< @return Tiles over all partitions
  [[nodiscard]] uint64_t num_rows() const;        ///< @return Rows over all partitions
  /// @return Shared memory one tile occupies when staged
  [[nodiscard]] std::size_t tile_bytes() const;

  struct impl;  ///< Implementation

 private:
  std::unique_ptr<impl> _impl;
  explicit packed_source(std::unique_ptr<impl>&& implementation);
  friend packed_source make_packed_source(std::span<std::vector<packed_data_view> const>,
                                          cuda::stream_ref,
                                          rmm::device_async_resource_ref);
  friend rmm::device_buffer reduce(packed_source const&,
                                   udf,
                                   std::size_t,
                                   void const*,
                                   std::span<bool const>,
                                   cuda::stream_ref,
                                   rmm::device_async_resource_ref);
  friend rmm::device_buffer groupby(packed_source const&,
                                    udf,
                                    std::size_t,
                                    std::size_t,
                                    void const*,
                                    std::span<bool const>,
                                    cuda::stream_ref,
                                    rmm::device_async_resource_ref);
};

/**
 * @brief Prepares tiles over `partitions`, where `partitions[p][c]` is column `c` of partition `p`.
 *
 * Each view holds one packed column without nulls. Synchronizes `stream` to read chunk sizes.
 *
 * @param partitions Packed columns per partition; every partition has the same column count
 * @param stream Stream for reading chunk sizes and uploading the tile directory
 * @param mr Device memory for the tile directory
 * @return The prepared source
 */
packed_source make_packed_source(
  std::span<std::vector<packed_data_view> const> partitions,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reduces every row of `source` with a caller row program, without unpacking.
 *
 * The kernel stages each tile's encoded chunks into shared memory and calls
 * `cudf_lto_reduce_row` once per row, then merges per-thread, per-block, and per-grid states with
 * `cudf_lto_reduce_merge`.
 *
 * @param source Prepared tiles
 * @param row_program Fragment defining the `cudf_lto_reduce_init`, `_row`, and `_merge` UDFs
 * @param state_bytes Size of the reduction state, at most `CUDF_LTO_MAX_STATE_BYTES`
 * @param user_data Device-accessible pointer passed to every `cudf_lto_reduce_row` call
 * @param lazy_columns Per column, whether the row program reads its chunks in place instead of from
 * shared memory; empty stages every column
 * @param stream Stream for the reduction
 * @param mr Device memory for the returned state
 * @return The final `state_bytes` state in device memory
 */
rmm::device_buffer reduce(
  packed_source const& source,
  udf row_program,
  std::size_t state_bytes,
  void const* user_data              = nullptr,
  std::span<bool const> lazy_columns = {},
  cuda::stream_ref stream            = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr  = cudf::get_current_device_resource_ref());

/**
 * @brief Groups every row of `source` into `num_groups` dense slots with a caller row program,
 * without unpacking.
 *
 * The kernel stages tiles like `reduce`, calls `cudf_lto_groupby_row` once per row, and merges
 * each kept row's state into its slot with `cudf_lto_reduce_merge`; slots start from
 * `cudf_lto_reduce_init`. Each warp keeps its own shared-memory copy of every slot, so
 * `num_groups` is limited by shared memory.
 *
 * @param source Prepared tiles
 * @param row_program Fragment defining the `cudf_lto_groupby_row`, `cudf_lto_reduce_init`, and
 * `cudf_lto_reduce_merge` UDFs
 * @param num_groups Number of dense group slots
 * @param state_bytes Size of one slot's state, at most `CUDF_LTO_MAX_STATE_BYTES`
 * @param user_data Device-accessible pointer passed to every `cudf_lto_groupby_row` call
 * @param lazy_columns As for `reduce`
 * @param stream Stream for the groupby
 * @param mr Device memory for the returned states
 * @return `num_groups` states of `state_bytes` each, in slot order, in device memory
 */
rmm::device_buffer groupby(
  packed_source const& source,
  udf row_program,
  std::size_t num_groups,
  std::size_t state_bytes,
  void const* user_data              = nullptr,
  std::span<bool const> lazy_columns = {},
  cuda::stream_ref stream            = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr  = cudf::get_current_device_resource_ref());

/**
 * @brief Dense map from unique integer build keys to build rows, for row programs.
 *
 * Owns the device memory behind a `cudf_lto_lookup`; a caller copies `view()` into its row
 * program's `user_data` and calls `cudf_lto_lookup_find`.
 */
class lookup_table {
 public:
  /**
   * @brief Takes ownership of the buffers `view` points into; made by `make_lookup_table`.
   *
   * @param storage Device buffers backing `view`
   * @param view The device view
   * @param num_rows Build rows the lookup indexes
   */
  lookup_table(std::vector<rmm::device_buffer> storage, cudf_lto_lookup view, size_type num_rows);

  [[nodiscard]] cudf_lto_lookup const& view() const { return _view; }  ///< @return Device view
  [[nodiscard]] size_type num_rows() const { return _num_rows; }  ///< @return Indexed build rows

 private:
  std::vector<rmm::device_buffer> _storage;
  cudf_lto_lookup _view;
  size_type _num_rows;
};

/**
 * @brief Builds a dense lookup over `keys`: a bitmap of the key range with per-word ranks, and
 * the row of each key in key order.
 *
 * Null keys are skipped and never found. Memory is about `(max - min) / 6` bytes plus four bytes
 * per row, and a lookup is two dependent loads.
 *
 * @throws std::invalid_argument if `keys` is not INT8, INT16, INT32, or INT64, if the valid keys
 * are not unique, or if their range exceeds 2^37
 *
 * @param keys Build keys
 * @param stream Stream for building
 * @param mr Device memory for the lookup
 * @return The lookup
 */
lookup_table make_lookup_table(
  column_view const& keys,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace experimental::lto
}  // namespace CUDF_EXPORT cudf
