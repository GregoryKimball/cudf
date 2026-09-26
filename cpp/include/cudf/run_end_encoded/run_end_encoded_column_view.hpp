/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/column/column_view.hpp>

/**
 * @file
 * @brief Class definition for `cudf::run_end_encoded_column_view`.
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief A host-side wrapper for a run-end encoded column.
 *
 * Child 0 contains non-nullable `INT32` exclusive run ends and child 1 contains the corresponding
 * non-nullable fixed-width values. Both children have one row per physical run, while `size()`
 * reports the decoded logical row count.
 *
 * Run ends must be positive and strictly increasing. For an unsliced owning column the final run
 * end must equal the logical size; for a sliced view they must cover `[offset(), offset() +
 * size())`. These are device-content preconditions and are not checked. Violating them is undefined
 * behavior.
 */
class run_end_encoded_column_view : private column_view {
 public:
  /**
   * @brief Constructs a run-end encoded view and validates its host-visible structure.
   *
   * @throws cudf::logic_error if the parent or children do not have the required structure
   */
  explicit run_end_encoded_column_view(column_view const& run_end_encoded_column);

  run_end_encoded_column_view(run_end_encoded_column_view const&)            = default;
  run_end_encoded_column_view(run_end_encoded_column_view&&)                 = default;
  run_end_encoded_column_view& operator=(run_end_encoded_column_view const&) = default;
  run_end_encoded_column_view& operator=(run_end_encoded_column_view&&)      = default;
  ~run_end_encoded_column_view() override                                    = default;

  /// Child index of the run ends column
  static constexpr size_type run_ends_column_index = cudf::run_end_encoded_run_ends_column_index;
  /// Child index of the run values column
  static constexpr size_type values_column_index = cudf::run_end_encoded_values_column_index;

  using column_view::has_nulls;
  using column_view::is_empty;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::offset;
  using column_view::size;

  [[nodiscard]] column_view parent() const noexcept;
  [[nodiscard]] column_view run_ends() const noexcept;
  [[nodiscard]] column_view values() const noexcept;
  [[nodiscard]] size_type num_runs() const noexcept;
  [[nodiscard]] data_type values_type() const noexcept;
};

//! Run-end encoded column APIs.
namespace run_end_encoded {
}

}  // namespace CUDF_EXPORT cudf
