/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/hashing.hpp>
#include <cudf/reduction.hpp>
#include <cudf/run_end_encoded/run_end_encoded.hpp>
#include <cudf/run_end_encoded/run_end_encoded_column_view.hpp>
#include <cudf/run_end_encoded/run_end_encoded_factories.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

struct run_layout {
  cudf::size_type run_length;
  bool constant;
};

run_layout parse_layout(std::string const& profile, cudf::size_type rows)
{
  if (profile == "constant") { return {rows, true}; }
  if (profile == "high_cardinality") { return {1, false}; }
  return {static_cast<cudf::size_type>(std::stoll(profile)), false};
}

template <typename T>
struct prepared_input {
  cudf::size_type rows;
  cudf::size_type runs;
  std::unique_ptr<cudf::column> source;
  std::unique_ptr<cudf::column> run_ends;
  std::unique_ptr<cudf::column> run_values;
  std::unique_ptr<cudf::column> dictionary;
  std::unique_ptr<cudf::column> ree;
};

template <typename T>
prepared_input<T> prepare(cudf::size_type rows, std::string const& profile)
{
  auto const layout = parse_layout(profile, rows);
  auto const runs   = static_cast<cudf::size_type>(
    (static_cast<int64_t>(rows) + layout.run_length - 1) / layout.run_length);

  std::vector<T> logical_values(static_cast<std::size_t>(rows));
  std::vector<int32_t> ends(static_cast<std::size_t>(runs));
  std::vector<T> values(static_cast<std::size_t>(runs));
  for (cudf::size_type run = 0; run < runs; ++run) {
    auto const value = static_cast<T>(layout.constant ? 7 : (run % 1024) + 1);
    auto const begin = static_cast<int64_t>(run) * layout.run_length;
    auto const end   = std::min<int64_t>(begin + layout.run_length, rows);
    ends[run]        = static_cast<int32_t>(end);
    values[run]      = value;
    std::fill(logical_values.begin() + begin, logical_values.begin() + end, value);
  }

  cudf::test::fixed_width_column_wrapper<T> source_wrapper(logical_values.begin(),
                                                           logical_values.end());
  cudf::test::fixed_width_column_wrapper<int32_t> ends_wrapper(ends.begin(), ends.end());
  cudf::test::fixed_width_column_wrapper<T> values_wrapper(values.begin(), values.end());

  auto source     = source_wrapper.release();
  auto run_ends   = ends_wrapper.release();
  auto run_values = values_wrapper.release();
  auto dictionary = cudf::dictionary::encode(source->view());
  auto ree        = cudf::make_run_end_encoded_column(rows, run_ends->view(), run_values->view());
  return {rows,
          runs,
          std::move(source),
          std::move(run_ends),
          std::move(run_values),
          std::move(dictionary),
          std::move(ree)};
}

template <typename T>
cudf::column_view selected_view(prepared_input<T> const& input, std::string const& encoding)
{
  if (encoding == "plain") { return input.source->view(); }
  if (encoding == "dictionary") { return input.dictionary->view(); }
  return input.ree->view();
}

template <typename T>
int64_t nominal_payload(prepared_input<T> const& input, std::string const& encoding)
{
  if (encoding == "plain") { return static_cast<int64_t>(input.rows) * sizeof(T); }
  if (encoding == "dictionary") {
    auto const keys = cudf::dictionary_column_view{input.dictionary->view()}.keys().size();
    return static_cast<int64_t>(input.rows) * sizeof(int32_t) +
           static_cast<int64_t>(keys) * sizeof(T);
  }
  return static_cast<int64_t>(input.runs) * (sizeof(int32_t) + sizeof(T));
}

template <typename T>
void report_input(nvbench::state& state,
                  prepared_input<T> const& input,
                  std::string const& encoding)
{
  auto const nominal   = nominal_payload(input, encoding);
  auto const allocated = encoding == "plain"        ? input.source->alloc_size()
                         : encoding == "dictionary" ? input.dictionary->alloc_size()
                                                    : input.ree->alloc_size();
  state.add_element_count(input.rows, "logical_rows");
  state.add_element_count(input.runs, "physical_runs");
  state.add_element_count(nominal, "nominal_payload_bytes");
  state.add_element_count(allocated, "input_allocated_bytes");
  state.add_global_memory_reads<nvbench::int8_t>(nominal);
}

template <typename T>
void run_construction(nvbench::state& state)
{
  auto const rows     = static_cast<cudf::size_type>(state.get_int64("rows"));
  auto const profile  = state.get_string("run_profile");
  auto const encoding = state.get_string("encoding");
  auto input          = prepare<T>(rows, profile);
  report_input(state, input, encoding);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (encoding == "plain") {
      auto result = std::make_unique<cudf::column>(input.source->view(), stream);
    } else if (encoding == "dictionary") {
      auto result = cudf::dictionary::encode(
        input.source->view(), cudf::data_type{cudf::type_id::INT32}, stream);
    } else {
      auto result = cudf::make_run_end_encoded_column(
        rows, input.run_ends->view(), input.run_values->view(), stream);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  set_throughputs(state);
}

template <typename T>
void run_sum(nvbench::state& state)
{
  auto const rows     = static_cast<cudf::size_type>(state.get_int64("rows"));
  auto const profile  = state.get_string("run_profile");
  auto const encoding = state.get_string("encoding");
  auto input          = prepare<T>(rows, profile);
  auto const view     = selected_view(input, encoding);
  report_input(state, input, encoding);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_writes<nvbench::int64_t>(1);
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (encoding == "run_end") {
      auto result = cudf::run_end_encoded::sum(view, cudf::data_type{cudf::type_id::INT64}, stream);
    } else {
      auto aggregation = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto result = cudf::reduce(view, *aggregation, cudf::data_type{cudf::type_id::INT64}, stream);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  set_throughputs(state);
}

template <typename T>
void run_hash(nvbench::state& state)
{
  auto const rows     = static_cast<cudf::size_type>(state.get_int64("rows"));
  auto const profile  = state.get_string("run_profile");
  auto const encoding = state.get_string("encoding");
  auto input          = prepare<T>(rows, profile);
  auto const view     = selected_view(input, encoding);
  report_input(state, input, encoding);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_writes<nvbench::uint32_t>(rows);
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::hashing::murmurhash3_x86_32(cudf::table_view{{view}}, 0, stream);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  set_throughputs(state);
}

template <typename Function>
void dispatch_type(nvbench::state& state, Function function)
{
  if (state.get_string("type") == "INT32") {
    function.template operator()<int32_t>(state);
  } else {
    function.template operator()<int64_t>(state);
  }
}

struct construction_runner {
  template <typename T>
  void operator()(nvbench::state& state) const
  {
    run_construction<T>(state);
  }
};

struct sum_runner {
  template <typename T>
  void operator()(nvbench::state& state) const
  {
    run_sum<T>(state);
  }
};

struct hash_runner {
  template <typename T>
  void operator()(nvbench::state& state) const
  {
    run_hash<T>(state);
  }
};

void benchmark_construction(nvbench::state& state) { dispatch_type(state, construction_runner{}); }
void benchmark_sum(nvbench::state& state) { dispatch_type(state, sum_runner{}); }
void benchmark_hash(nvbench::state& state) { dispatch_type(state, hash_runner{}); }

}  // namespace

NVBENCH_BENCH(benchmark_construction)
  .set_name("run_end_encoded_construction")
  .add_int64_axis("rows", {1 << 18, 1 << 24})
  .add_string_axis("run_profile", {"constant", "4", "32", "1024", "high_cardinality"})
  .add_string_axis("encoding", {"plain", "dictionary", "run_end"})
  .add_string_axis("type", {"INT32", "INT64"});

NVBENCH_BENCH(benchmark_sum)
  .set_name("run_end_encoded_sum")
  .add_int64_axis("rows", {1 << 18, 1 << 24})
  .add_string_axis("run_profile", {"constant", "4", "32", "1024", "high_cardinality"})
  .add_string_axis("encoding", {"plain", "dictionary", "run_end"})
  .add_string_axis("type", {"INT32", "INT64"});

NVBENCH_BENCH(benchmark_hash)
  .set_name("run_end_encoded_hash")
  .add_int64_axis("rows", {1 << 18, 1 << 24})
  .add_string_axis("run_profile", {"constant", "4", "32", "1024", "high_cardinality"})
  .add_string_axis("encoding", {"plain", "dictionary", "run_end"})
  .add_string_axis("type", {"INT32", "INT64"});
