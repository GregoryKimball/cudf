/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/io/experimental/feather.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <string_view>

/**
 * @brief File formats compared by the format comparison benchmarks.
 */
enum class file_format { PARQUET, CUDFTABLE, FEATHER };

inline file_format retrieve_file_format_enum(std::string_view name)
{
  if (name == "PARQUET") { return file_format::PARQUET; }
  if (name == "CUDFTABLE") { return file_format::CUDFTABLE; }
  if (name == "FEATHER") { return file_format::FEATHER; }
  CUDF_FAIL("Unsupported file format: " + std::string{name});
}

/**
 * @brief Skips format/compression combinations that do not exist.
 *
 * @return true if the state was skipped
 */
inline bool skip_unsupported(nvbench::state& state,
                             file_format format,
                             cudf::io::compression_type compression)
{
  if (format == file_format::CUDFTABLE and compression != cudf::io::compression_type::NONE) {
    state.skip("CUDFTABLE does not support compression");
    return true;
  }
  return false;
}

inline void write_table(file_format format,
                        cudf::io::compression_type compression,
                        cudf::io::sink_info const& sink,
                        cudf::table_view const& view)
{
  switch (format) {
    case file_format::PARQUET:
      cudf::io::write_parquet(
        cudf::io::parquet_writer_options::builder(sink, view).compression(compression).build());
      break;
    case file_format::CUDFTABLE:
      cudf::io::experimental::write_cudftable(
        cudf::io::experimental::cudftable_writer_options::builder(sink, view).build());
      break;
    case file_format::FEATHER:
      cudf::io::experimental::write_feather(
        cudf::io::experimental::feather_writer_options::builder(sink, view)
          .compression(compression)
          .build());
      break;
  }
}

/**
 * @brief Reads a table and returns its {num_rows, num_columns}.
 */
inline std::pair<cudf::size_type, cudf::size_type> read_table(file_format format,
                                                              cudf::io::source_info const& source)
{
  switch (format) {
    case file_format::PARQUET: {
      auto const result =
        cudf::io::read_parquet(cudf::io::parquet_reader_options::builder(source).build());
      return {result.tbl->num_rows(), result.tbl->num_columns()};
    }
    case file_format::CUDFTABLE: {
      auto const result = cudf::io::experimental::read_cudftable(
        cudf::io::experimental::cudftable_reader_options::builder(source).build());
      return {result.table.num_rows(), result.table.num_columns()};
    }
    case file_format::FEATHER: {
      auto const result = cudf::io::experimental::read_feather(
        cudf::io::experimental::feather_reader_options::builder(source).build());
      return {result.tbl->num_rows(), result.tbl->num_columns()};
    }
  }
  CUDF_FAIL("Unsupported file format");
}

/**
 * @brief Builds the random input profile from the cardinality and run length axes.
 */
inline data_profile profile_from_axes(nvbench::state& state)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  return data_profile_builder().cardinality(cardinality).avg_run_length(run_length);
}

/// Types every compared format can store; DURATION is excluded because Arrow has no day duration
using format_comparison_types = nvbench::enum_type_list<data_type::INTEGRAL,
                                                        data_type::FLOAT,
                                                        data_type::BOOL8,
                                                        data_type::DECIMAL,
                                                        data_type::TIMESTAMP,
                                                        data_type::STRING>;
