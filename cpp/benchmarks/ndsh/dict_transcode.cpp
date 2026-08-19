/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/ndsh_data_generator/ndsh_data_generator.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <memory>
#include <vector>

namespace {

constexpr cudf::size_type row_group_size_rows = 1'000;
std::vector<cudf::size_type> const part_string_columns{2, 3, 4, 6};

void verify_transcode(cudf::io::source_info const& source)
{
  auto const plain = read_parquet(source);
  auto const dict  = read_parquet(source, {}, nullptr, true);

  auto const plain_view = plain->table();
  auto const dict_view  = dict->table();
  CUDF_EXPECTS(plain_view.num_columns() == dict_view.num_columns(),
               "Dictionary transcode changed the number of columns");
  CUDF_EXPECTS(plain_view.num_rows() == dict_view.num_rows(),
               "Dictionary transcode changed the number of rows");

  for (cudf::size_type column_index = 0; column_index < plain_view.num_columns(); ++column_index) {
    auto const plain_column = plain_view.column(column_index);
    auto const dict_column  = dict_view.column(column_index);
    CUDF_EXPECTS(plain_column.type().id() == cudf::type_id::STRING,
                 "Dictionary transcode baseline did not produce STRING");
    CUDF_EXPECTS(dict_column.type().id() == cudf::type_id::DICTIONARY32,
                 "Dictionary transcode did not produce DICTIONARY32");

    auto const decoded = cudf::dictionary::decode(cudf::dictionary_column_view{dict_column});
    auto const matches = cudf::binary_operation(plain_column,
                                                decoded->view(),
                                                cudf::binary_operator::EQUAL,
                                                cudf::data_type{cudf::type_id::BOOL8});
    auto const all_aggregation = cudf::make_all_aggregation<cudf::reduce_aggregation>();
    auto const all_match =
      cudf::reduce(matches->view(), *all_aggregation, cudf::data_type{cudf::type_id::BOOL8});
    CUDF_EXPECTS(static_cast<cudf::numeric_scalar<bool> const&>(*all_match).value(),
                 "Dictionary transcode changed column values");
  }
}

void ndsh_dict_transcode(nvbench::state& state)
{
  auto const scale_factor        = state.get_float64("scale_factor");
  auto const output_dict_columns = state.get_int64("output_dict_columns") != 0;

  auto [orders, lineitem, part] = cudf::datagen::generate_orders_lineitem_part(scale_factor);
  orders.reset();
  lineitem.reset();

  auto const part_strings = part->view().select(part_string_columns);
  auto const num_rows     = part_strings.num_rows();
  cuio_source_sink_pair source_sink{io_type::HOST_BUFFER};
  auto const write_options =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), part_strings)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .row_group_size_rows(row_group_size_rows)
      .build();
  cudf::io::write_parquet(write_options);
  part.reset();

  auto const source = source_sink.make_source_info();
  verify_transcode(source);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto const result = read_parquet(source, {}, nullptr, output_dict_columns);
    auto const expected_type =
      output_dict_columns ? cudf::type_id::DICTIONARY32 : cudf::type_id::STRING;
    auto const result_view = result->table();
    CUDF_EXPECTS(std::all_of(result_view.begin(),
                             result_view.end(),
                             [expected_type](auto const& column) {
                               return column.type().id() == expected_type;
                             }),
                 "Parquet reader returned an unexpected column type");
  });

  state.add_element_count(num_rows);
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

NVBENCH_BENCH(ndsh_dict_transcode)
  .set_name("ndsh_dict_transcode")
  .add_int64_axis("output_dict_columns", {0, 1})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
