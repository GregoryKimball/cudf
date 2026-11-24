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

void format_write_common(cudf::table_view const& view,
                         file_format format,
                         cudf::io::compression_type compression,
                         io_type sink_type,
                         nvbench::state& state)
{
  auto const data_size          = static_cast<size_t>(state.get_int64("data_size"));
  std::size_t encoded_file_size = 0;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch&, auto& timer) {
               cuio_source_sink_pair source_sink(sink_type);
               auto const& sink_info = source_sink.make_sink_info();
               drop_page_cache_if_enabled(sink_info.filepaths());

               timer.start();
               write_table(format, compression, sink_info, view);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

template <data_type DataType>
void BM_format_write_data_types(nvbench::state& state,
                                nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const format      = retrieve_file_format_enum(state.get_string("format"));
  auto const compression = retrieve_compression_type_enum(state.get_string("compression_type"));
  if (skip_unsupported(state, format, compression)) { return; }

  auto const d_type    = get_type_or_group(static_cast<int32_t>(DataType));
  auto const sink_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size = static_cast<size_t>(state.get_int64("data_size"));

  auto const tbl = create_random_table(
    cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile_from_axes(state));
  format_write_common(tbl->view(), format, compression, sink_type, state);
}

void BM_format_write_data_sizes(nvbench::state& state)
{
  auto const format      = retrieve_file_format_enum(state.get_string("format"));
  auto const compression = retrieve_compression_type_enum(state.get_string("compression_type"));
  if (skip_unsupported(state, format, compression)) { return; }

  auto const d_type    = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
  auto const sink_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size = static_cast<size_t>(state.get_int64("data_size"));

  auto const tbl = create_random_table(
    cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile_from_axes(state));
  format_write_common(tbl->view(), format, compression, sink_type, state);
}

NVBENCH_BENCH_TYPES(BM_format_write_data_types, NVBENCH_TYPE_AXES(format_comparison_types))
  .set_name("format_write_data_types")
  .set_type_axes_names({"data_type"})
  .add_string_axis("format", {"PARQUET", "CUDFTABLE", "FEATHER"})
  .add_string_axis("compression_type", {"NONE", "LZ4", "ZSTD"})
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "VOID"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH(BM_format_write_data_sizes)
  .set_name("format_write_data_sizes")
  .add_string_axis("format", {"PARQUET", "CUDFTABLE", "FEATHER"})
  .add_string_axis("compression_type", {"NONE", "LZ4", "ZSTD"})
  .add_string_axis("io_type", {"VOID"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0})
  .add_int64_axis("run_length", {1})
  .add_int64_power_of_two_axis("data_size", nvbench::range(24, 31, 1));  // 16MB to 2GB
