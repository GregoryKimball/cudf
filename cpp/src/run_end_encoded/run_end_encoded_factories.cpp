/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/run_end_encoded/run_end_encoded_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf {
namespace {

void validate_structure(size_type logical_size,
                        column_view const& run_ends,
                        column_view const& values)
{
  CUDF_EXPECTS(logical_size >= 0, "Logical size cannot be negative", std::invalid_argument);
  CUDF_EXPECTS(
    run_ends.type().id() == type_id::INT32, "Run ends must have type INT32", std::invalid_argument);
  CUDF_EXPECTS(run_ends.size() == values.size(),
               "Run ends and values must have the same number of rows",
               std::invalid_argument);
  CUDF_EXPECTS(not run_ends.nullable(), "Run ends must be non-nullable", std::invalid_argument);
  CUDF_EXPECTS(not values.nullable(), "Run values must be non-nullable", std::invalid_argument);
  CUDF_EXPECTS(is_fixed_width(values.type()),
               "Run values must have a fixed-width, decimal, or chrono type",
               std::invalid_argument);
  CUDF_EXPECTS((logical_size == 0) == (run_ends.size() == 0),
               "An empty owning column must have empty children and a non-empty owning column must "
               "have at least one run",
               std::invalid_argument);
}

}  // namespace

std::unique_ptr<column> make_run_end_encoded_column(size_type logical_size,
                                                    column_view const& run_ends,
                                                    column_view const& values,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  validate_structure(logical_size, run_ends, values);
  return make_run_end_encoded_column(logical_size,
                                     std::make_unique<column>(run_ends, stream, mr),
                                     std::make_unique<column>(values, stream, mr),
                                     0,
                                     rmm::device_buffer{});
}

std::unique_ptr<column> make_run_end_encoded_column(size_type logical_size,
                                                    std::unique_ptr<column> run_ends,
                                                    std::unique_ptr<column> values,
                                                    size_type null_count,
                                                    rmm::device_buffer&& null_mask)
{
  CUDF_EXPECTS(run_ends != nullptr, "Run ends child must not be null", std::invalid_argument);
  CUDF_EXPECTS(values != nullptr, "Run values child must not be null", std::invalid_argument);
  validate_structure(logical_size, run_ends->view(), values->view());
  CUDF_EXPECTS(null_count >= 0 and null_count <= logical_size,
               "Null count must be between zero and the logical size",
               std::invalid_argument);
  CUDF_EXPECTS(null_count == 0 or null_mask.size() > 0,
               "A column with nulls must have a parent null mask",
               std::invalid_argument);

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(run_ends));
  children.emplace_back(std::move(values));
  return std::make_unique<column>(data_type{type_id::RUN_END_ENCODED},
                                  logical_size,
                                  rmm::device_buffer{},
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

std::unique_ptr<column> make_empty_run_end_encoded_column(data_type values_type)
{
  CUDF_EXPECTS(is_fixed_width(values_type),
               "Run values must have a fixed-width, decimal, or chrono type",
               std::invalid_argument);
  return make_run_end_encoded_column(0,
                                     make_empty_column(data_type{type_id::INT32}),
                                     make_empty_column(values_type),
                                     0,
                                     rmm::device_buffer{});
}

}  // namespace cudf
