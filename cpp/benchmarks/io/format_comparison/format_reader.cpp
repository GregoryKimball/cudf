/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "format_common.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr cudf::size_type num_cols = 64;

void format_read_common(cudf::size_type num_rows_to_read,
                        cudf::size_type num_cols_to_read,
                        file_format format,
                        cuio_source_sink_pair& source_sink,
                        nvbench::state& state)
{
  auto const data_size = static_cast<size_t>(state.get_int64("data_size"));
  auto source_info     = source_sink.make_source_info();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch&, auto& timer) {
               drop_page_cache_if_enabled(source_info.filepaths());

               timer.start();
               auto const [num_rows, num_columns] = read_table(format, source_info);
               timer.stop();

               CUDF_EXPECTS(num_columns == num_cols_to_read, "Unexpected number of columns");
               CUDF_EXPECTS(num_rows == num_rows_to_read, "Unexpected number of rows");
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

template <data_type DataType>
void BM_format_read_data_types(nvbench::state& state,
                               nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const format      = retrieve_file_format_enum(state.get_string("format"));
  auto const compression = retrieve_compression_type_enum(state.get_string("compression_type"));
  if (skip_unsupported(state, format, compression)) { return; }

  auto const d_type      = get_type_or_group(static_cast<int32_t>(DataType));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile_from_axes(state));
    auto const view = tbl->view();
    write_table(format, compression, source_sink.make_sink_info(), view);
    return view.num_rows();
  }();

  format_read_common(num_rows_written, num_cols, format, source_sink, state);
}

void BM_format_read_data_sizes(nvbench::state& state)
{
  auto const format      = retrieve_file_format_enum(state.get_string("format"));
  auto const compression = retrieve_compression_type_enum(state.get_string("compression_type"));
  if (skip_unsupported(state, format, compression)) { return; }

  auto const d_type      = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile_from_axes(state));
    auto const view = tbl->view();
    write_table(format, compression, source_sink.make_sink_info(), view);
    return view.num_rows();
  }();

  format_read_common(num_rows_written, num_cols, format, source_sink, state);
}

NVBENCH_BENCH_TYPES(BM_format_read_data_types, NVBENCH_TYPE_AXES(format_comparison_types))
  .set_name("format_read_data_types")
  .set_type_axes_names({"data_type"})
  .add_string_axis("format", {"PARQUET", "CUDFTABLE", "FEATHER"})
  .add_string_axis("compression_type", {"NONE", "LZ4", "ZSTD"})
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH(BM_format_read_data_sizes)
  .set_name("format_read_data_sizes")
  .add_string_axis("format", {"PARQUET", "CUDFTABLE", "FEATHER"})
  .add_string_axis("compression_type", {"NONE", "LZ4", "ZSTD"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0})
  .add_int64_axis("run_length", {1})
  .add_int64_power_of_two_axis("data_size", nvbench::range(24, 31, 1));  // 16MB to 2GB
