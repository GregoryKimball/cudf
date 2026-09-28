/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>

namespace CUDF_EXPORT cudf {

/**
 * @brief Immutable view of a block-packed DECIMAL128 column.
 *
 * Coefficients are grouped in blocks of 256 rows. Each block uses one byte width
 * in the range [0, 16], and its descriptor is `(payload_byte_offset << 5) | width`.
 * Coefficients are stored little-endian in signed two's-complement form.
 */
class packed_decimal128_column_view : private column_view {
 public:
  static constexpr size_type block_size{256};
  static constexpr size_type payload_child_index{0};
  static constexpr size_type descriptor_child_index{1};

  explicit packed_decimal128_column_view(column_view column);

  using column_view::has_nulls;
  using column_view::is_empty;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::offset;
  using column_view::size;
  using column_view::type;

  [[nodiscard]] column_view parent() const;
  [[nodiscard]] column_view descriptors() const;
  [[nodiscard]] uint8_t const* payload_begin() const noexcept;
};

/**
 * @brief Constructs and validates a canonical packed DECIMAL128 column.
 *
 * The descriptor child must be a non-nullable UINT64 column with exactly
 * `ceil(size / 256)` entries. Descriptors must identify contiguous payload blocks.
 */
std::unique_ptr<column> make_packed_decimal128_column(
  size_type size,
  numeric::scale_type scale,
  rmm::device_buffer&& payload,
  std::unique_ptr<column> descriptors,
  rmm::device_buffer&& null_mask,
  size_type null_count,
  cuda::stream_ref stream = cudf::get_default_stream());

/** @brief Encodes a DECIMAL128 column using fixed 256-row blocks. */
std::unique_ptr<column> encode_packed_decimal128(
  column_view input,
  cuda::stream_ref stream                 = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @brief Decodes a packed DECIMAL128 column to DECIMAL128. */
std::unique_ptr<column> decode_packed_decimal128(
  column_view input,
  cuda::stream_ref stream                 = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace CUDF_EXPORT cudf
