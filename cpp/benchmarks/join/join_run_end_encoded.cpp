/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/join/join.hpp>
#include <cudf/run_end_encoded/run_end_encoded.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace {

void inner_join_run_end_encoded(nvbench::state& state)
{
  auto const left_size  = static_cast<cudf::size_type>(state.get_int64("left_size"));
  auto const right_size = static_cast<cudf::size_type>(state.get_int64("right_size"));
  auto const num_keys   = static_cast<std::size_t>(state.get_int64("num_keys"));

  // Multiplicity one bounds the result to at most one match per probe row. A moderate fixed
  // selectivity keeps output allocation visible without allowing it to dominate the compatibility
  // measurement.
  constexpr auto selectivity = 0.3;
  auto const matching_rows =
    static_cast<cudf::size_type>(static_cast<double>(left_size) * selectivity);

  std::vector<std::unique_ptr<cudf::column>> plain_build_columns;
  std::vector<std::unique_ptr<cudf::column>> plain_probe_columns;
  plain_build_columns.reserve(num_keys);
  plain_probe_columns.reserve(num_keys);
  for (std::size_t key = 0; key < num_keys; ++key) {
    std::vector<int32_t> build_keys(right_size);
    std::vector<int32_t> probe_keys(left_size);
    for (cudf::size_type row = 0; row < right_size; ++row) {
      // Every build key is unique: build-side multiplicity is exactly one.
      build_keys[row] = static_cast<int32_t>(row + key * (right_size + left_size));
    }
    for (cudf::size_type row = 0; row < left_size; ++row) {
      probe_keys[row] = row < matching_rows
                          ? build_keys[row % right_size]
                          : static_cast<int32_t>((key + 1) * (right_size + left_size) + row);
    }
    cudf::test::fixed_width_column_wrapper<int32_t> build(build_keys.begin(), build_keys.end());
    cudf::test::fixed_width_column_wrapper<int32_t> probe(probe_keys.begin(), probe_keys.end());
    plain_build_columns.push_back(build.release());
    plain_probe_columns.push_back(probe.release());
  }

  auto const build_table = std::make_unique<cudf::table>(std::move(plain_build_columns));
  auto const probe_table = std::make_unique<cudf::table>(std::move(plain_probe_columns));

  auto encode = [](cudf::table_view const& table) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(table.num_columns());
    for (auto const& column : table) {
      columns.push_back(cudf::run_end_encoded::encode(column));
    }
    return columns;
  };
  auto build_columns = encode(build_table->view());
  auto probe_columns = encode(probe_table->view());

  auto make_view = [](auto const& columns) {
    std::vector<cudf::column_view> views;
    views.reserve(columns.size());
    for (auto const& column : columns) {
      views.push_back(column->view());
    }
    return cudf::table_view{views};
  };
  auto const build_view = make_view(build_columns);
  auto const probe_view = make_view(probe_columns);

  auto allocation_size = [](auto const& columns) {
    return std::accumulate(
      columns.begin(), columns.end(), int64_t{0}, [](auto total, auto const& c) {
        return total + static_cast<int64_t>(c->alloc_size());
      });
  };
  auto const input_bytes = allocation_size(build_columns) + allocation_size(probe_columns);
  state.add_element_count(input_bytes, "join_input_allocated_bytes");
  state.add_element_count(left_size, "probe_rows");
  state.add_element_count(right_size, "build_rows");
  state.add_element_count(matching_rows, "expected_output_rows");
  state.add_global_memory_reads<nvbench::int8_t>(input_bytes);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::inner_join(probe_view, build_view, cudf::null_equality::UNEQUAL, stream);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  set_throughputs(state);
}

}  // namespace

NVBENCH_BENCH(inner_join_run_end_encoded)
  .set_name("inner_join_run_end_encoded")
  .add_int64_axis("num_keys", {1})
  .add_int64_axis("left_size", {1 << 18, 1 << 20})
  .add_int64_axis("right_size", {1 << 16});
