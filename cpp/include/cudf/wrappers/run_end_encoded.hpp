/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

/**
 * @file
 * @brief Concrete dispatch type for run-end encoded columns.
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief A strongly typed wrapper used when dispatching a RUN_END_ENCODED column.
 *
 * The wrapper does not represent the logical value type of the column. That type is described by
 * the values child of the run-end encoded column.
 */
struct run_end_encoded32 {
  using value_type = int32_t;  ///< The run-end storage type
};

}  // namespace CUDF_EXPORT cudf
