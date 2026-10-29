/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lto/codec.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <span>
#include <variant>

namespace CUDF_EXPORT cudf {
namespace lto {

/// An input of `filter_gather`: a plain column or a column read through a codec.
using filter_input = std::variant<column_view, encoded_column_view>;

/**
 * @brief Decodes, filters, and gathers columns in one fused kernel.
 *
 * Evaluates `predicate` for every row, decoding only the inputs the predicate references, and
 * returns the inputs listed in `gather_map` for the rows where it is true, decoded into plain
 * columns. Gathered columns are decoded only for the selected rows. Codec hooks are linked into
 * the kernel with LTO.
 *
 * @throws std::invalid_argument if the inputs differ in size, if an input is not fixed-width, if a
 * plain input has nulls, if an encoded input has no codec or no load hook, or if `predicate` does
 * not evaluate to BOOL8
 * @throws std::out_of_range if `gather_map` names a missing input
 *
 * @param inputs Columns referenced by `predicate` (by index) and `gather_map`
 * @param predicate Boolean expression over `inputs`
 * @param gather_map Indices of the inputs to return
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table
 * @return The gathered columns for the rows that satisfy `predicate`
 */
std::unique_ptr<table> filter_gather(
  std::span<filter_input const> inputs,
  ast::expression const& predicate,
  std::span<size_type const> gather_map,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Decodes columns into plain columns in one fused kernel.
 *
 * Equivalent to `filter_gather` with a predicate that is always true.
 *
 * @throws std::invalid_argument under the same conditions as `filter_gather`
 * @throws std::out_of_range if `gather_map` names a missing input
 *
 * @param inputs Columns to read from
 * @param gather_map Indices of the inputs to return
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table
 * @return The decoded columns
 */
std::unique_ptr<table> decode(
  std::span<filter_input const> inputs,
  std::span<size_type const> gather_map,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace lto
}  // namespace CUDF_EXPORT cudf
