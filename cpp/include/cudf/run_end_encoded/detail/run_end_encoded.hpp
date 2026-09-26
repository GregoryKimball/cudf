/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/run_end_encoded/run_end_encoded_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf::run_end_encoded::detail {

std::unique_ptr<column> encode(column_view const& input,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr);

std::unique_ptr<column> decode(run_end_encoded_column_view const& input,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr);

std::unique_ptr<column> decode(column_view const& input,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr);

}  // namespace cudf::run_end_encoded::detail
