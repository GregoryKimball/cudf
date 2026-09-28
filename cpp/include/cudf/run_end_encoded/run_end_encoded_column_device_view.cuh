/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/column/column_device_view.cuh>

namespace cudf {

/**
 * @brief Device-side logical access to a run-end encoded column.
 *
 * The supplied `column_device_view` must describe a structurally valid RUN_END_ENCODED column whose
 * run ends satisfy the documented factory preconditions.
 */
class run_end_encoded_column_device_view {
 public:
  CUDF_HOST_DEVICE explicit run_end_encoded_column_device_view(column_device_view const& parent)
    : _parent{parent}
  {
  }

  /**
   * @brief Returns the physical run containing logical `row`.
   *
   * Lookup uses the first exclusive run end greater than `parent.offset() + row`.
   */
  [[nodiscard]] __device__ size_type find_run(size_type row) const noexcept
  {
    auto const& ends  = _parent.child(run_end_encoded_run_ends_column_index);
    auto const target = _parent.offset() + row;
    size_type first   = 0;
    size_type count   = ends.size();
    while (count > 0) {
      auto const step = count / 2;
      auto const mid  = first + step;
      if (ends.element<int32_t>(mid) <= target) {
        first = mid + 1;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    return first;
  }

  [[nodiscard]] __device__ bool is_null(size_type row) const noexcept
  {
    return _parent.is_null(row);
  }

  template <typename T>
  [[nodiscard]] __device__ T value(size_type row) const noexcept
  {
    return _parent.child(run_end_encoded_values_column_index).element<T>(find_run(row));
  }

 private:
  column_device_view _parent;
};

}  // namespace cudf
