/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/run_end_encoded/run_end_encoded_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace CUDF_EXPORT cudf {
namespace run_end_encoded {

/**
 * @brief Encodes adjacent equal rows as run ends and physical values.
 *
 * Validity participates in run equality: a valid row and a null row never share a run, while
 * adjacent null rows do. The physical value stored for a null run is unspecified. The input must
 * have a fixed-width, decimal, or chrono type and must not itself be run-end encoded.
 *
 * @param input Column to encode
 * @param stream CUDA stream used for device work
 * @param mr Memory resources used for temporary and output allocations
 * @return Canonical owning run-end encoded column
 */
std::unique_ptr<column> encode(column_view const& input,
                               cuda::stream_ref stream   = cudf::get_default_stream(),
                               cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/**
 * @brief Decodes a run-end encoded column to its logical fixed-width values.
 *
 * Sliced inputs resolve rows using `input.offset() + row`. The returned column has the values-child
 * type and a copy of the sliced parent logical null mask.
 *
 * @param input Run-end encoded input
 * @param stream CUDA stream used for device work
 * @param mr Memory resources used for temporary and output allocations
 * @return Owning decoded column
 */
std::unique_ptr<column> decode(run_end_encoded_column_view const& input,
                               cuda::stream_ref stream   = cudf::get_default_stream(),
                               cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/**
 * @brief Decodes a RUN_END_ENCODED column to its logical fixed-width values.
 *
 * @throws cudf::logic_error if `input` is not a structurally valid RUN_END_ENCODED column
 */
std::unique_ptr<column> decode(column_view const& input,
                               cuda::stream_ref stream   = cudf::get_default_stream(),
                               cudf::memory_resources mr = cudf::get_current_device_resource_ref());

}  // namespace run_end_encoded
}  // namespace CUDF_EXPORT cudf
