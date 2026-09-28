/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

/**
 * @file
 * @brief Factory functions for constructing run-end encoded columns.
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief Constructs a non-null run-end encoded column by copying its physical children.
 *
 * `run_ends` must be a non-nullable `INT32` column and `values` must be a non-nullable,
 * supported fixed-width column. The children must have equal sizes. A zero logical size requires
 * two typed empty children; a positive logical size requires at least one run.
 *
 * Run ends are exclusive logical positions. They must be positive and strictly increasing, and the
 * final run end must equal `logical_size`. These device-content preconditions are not checked;
 * violating them is undefined behavior.
 */
std::unique_ptr<column> make_run_end_encoded_column(
  size_type logical_size,
  column_view const& run_ends,
  column_view const& values,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Constructs a run-end encoded column by taking ownership of its physical children.
 *
 * Structural requirements and unchecked run-end content preconditions are the same as the copying
 * overload. The parent owns logical validity; both children must be non-nullable.
 *
 * @param logical_size Decoded logical row count
 * @param run_ends Non-nullable `INT32` exclusive run ends
 * @param values Non-nullable fixed-width, decimal, or chrono run values
 * @param null_count Number of null logical rows
 * @param null_mask Parent logical-row null mask
 * @return New run-end encoded column
 */
std::unique_ptr<column> make_run_end_encoded_column(size_type logical_size,
                                                    std::unique_ptr<column> run_ends,
                                                    std::unique_ptr<column> values,
                                                    size_type null_count,
                                                    rmm::device_buffer&& null_mask);

/**
 * @brief Creates an empty run-end encoded column retaining typed empty children.
 *
 * @param values_type Fixed-width, decimal, or chrono values-child type
 */
std::unique_ptr<column> make_empty_run_end_encoded_column(data_type values_type);

}  // namespace CUDF_EXPORT cudf
