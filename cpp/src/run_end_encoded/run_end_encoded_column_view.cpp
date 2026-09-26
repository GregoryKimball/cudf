/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/run_end_encoded/run_end_encoded_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf {

run_end_encoded_column_view::run_end_encoded_column_view(column_view const& column)
  : column_view{column}
{
  CUDF_EXPECTS(type().id() == type_id::RUN_END_ENCODED,
               "run_end_encoded_column_view only supports RUN_END_ENCODED columns");
  CUDF_EXPECTS(num_children() == 2, "A run-end encoded column must have exactly two children");

  auto const ends = run_ends();
  auto const vals = values();
  CUDF_EXPECTS(ends.type().id() == type_id::INT32, "Run ends must have type INT32");
  CUDF_EXPECTS(ends.size() == vals.size(), "Run ends and values must have the same number of rows");
  CUDF_EXPECTS(not ends.nullable(), "Run ends must be non-nullable");
  CUDF_EXPECTS(not vals.nullable(), "Run values must be non-nullable");
  CUDF_EXPECTS(is_fixed_width(vals.type()),
               "Run values must have a fixed-width, decimal, or chrono type");
  CUDF_EXPECTS(size() == 0 or ends.size() > 0,
               "A non-empty run-end encoded column must contain at least one run");
}

column_view run_end_encoded_column_view::parent() const noexcept
{
  return static_cast<column_view>(*this);
}

column_view run_end_encoded_column_view::run_ends() const noexcept
{
  return child(run_ends_column_index);
}

column_view run_end_encoded_column_view::values() const noexcept
{
  return child(values_column_index);
}

size_type run_end_encoded_column_view::num_runs() const noexcept { return run_ends().size(); }

data_type run_end_encoded_column_view::values_type() const noexcept { return values().type(); }

}  // namespace cudf
