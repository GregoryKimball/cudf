/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lto/udf.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cudf_test_fragments.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace lto = cudf::experimental::lto;

constexpr uint32_t for_bitpack_id = 1;
constexpr std::size_t tile_rows   = 4096;

std::span<uint8_t const> test_fragment(std::size_t index)
{
  auto const range = cudf_test_fragments::file_ranges[index];
  return cudf_test_fragments::files.subspan(range[0], range[1]);
}

void register_for_bitpack()
{
  if (lto::is_codec_registered(for_bitpack_id)) { return; }
  auto const binary = test_fragment(cudf_test_fragments::lto_for_bitpack);
  lto::register_codec(
    for_bitpack_id,
    lto::codec{std::vector<uint8_t>(binary.begin(), binary.end()),
               cudf::lto_binary_type::FATBIN,
               [](std::size_t chunk_bytes) { return 16 + (chunk_bytes + 7) / 8 * 8 + 8; }});
}

struct packed {
  std::vector<uint8_t> metadata;
  std::shared_ptr<uint8_t> payload;  ///< Pinned, mapped host memory
  std::size_t payload_bytes = 0;
  cudf::experimental::pack_compression compression{};

  [[nodiscard]] cudf::experimental::packed_data_view view() const
  {
    return {metadata, {payload.get(), payload_bytes}, compression};
  }
};

/// Packs one column with the FOR+bitpack LTO codec into pinned host memory.
packed pack_lto(cudf::column_view column)
{
  auto const stream = cudf::get_default_stream();
  cudf::experimental::pack_options options;
  options.output_mode = cudf::experimental::compressed_output_mode::compact;
  auto builder =
    cudf::experimental::make_pack_plan_builder(cudf::table_view{{column}}, options, stream);
  for (auto& region : builder.regions()) {
    if (region.info.kind != cudf::experimental::pack_region_kind::data) { continue; }
    region.options.codec        = cudf::experimental::pack_compression::lto;
    region.options.lto_codec_id = for_bitpack_id;
    region.options.compression_chunk_bytes =
      tile_rows * cudf::size_of(cudf::data_type{region.info.type});
  }
  auto const plan = std::move(builder).build();
  rmm::device_buffer scratch(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan, {static_cast<uint8_t*>(scratch.data()), plan.sizes().payload_bytes});
  void* host = nullptr;
  EXPECT_EQ(
    cudaHostAlloc(&host, std::max<std::size_t>(result.payload_bytes, 1), cudaHostAllocMapped),
    cudaSuccess);
  EXPECT_EQ(cudaMemcpy(host, scratch.data(), result.payload_bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
  return packed{
    std::move(result.metadata),
    std::shared_ptr<uint8_t>{static_cast<uint8_t*>(host), [](uint8_t* p) { cudaFreeHost(p); }},
    result.payload_bytes,
    result.compression};
}

template <typename T>
std::vector<T> random_values(std::size_t rows, T lo, T hi, unsigned seed)
{
  std::mt19937_64 engine{seed};
  std::uniform_int_distribution<T> distribution{lo, hi};
  std::vector<T> values(rows);
  for (auto& value : values) {
    value = distribution(engine);
  }
  return values;
}

}  // namespace

struct LTOUDFTest : public cudf::test::BaseFixture {
  void SetUp() override { register_for_bitpack(); }
};

TEST_F(LTOUDFTest, PackMaterializeRoundTrip)
{
  auto const rows  = 3 * tile_rows + 17;
  auto const small = random_values<int32_t>(rows, -1000, 1000, 1);
  auto const wide  = random_values<int64_t>(rows, -(int64_t{1} << 40), int64_t{1} << 40, 2);
  cudf::test::fixed_width_column_wrapper<int32_t> small_column(small.begin(), small.end());
  cudf::test::fixed_width_column_wrapper<int64_t> wide_column(wide.begin(), wide.end());

  for (cudf::column_view column :
       {cudf::column_view{small_column}, cudf::column_view{wide_column}}) {
    auto const packed_column = pack_lto(column);
    EXPECT_LT(packed_column.payload_bytes, rows * cudf::size_of(column.type()));
    auto const result = cudf::experimental::materialize(packed_column.view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0).view(), column);
  }
}

TEST_F(LTOUDFTest, ConstantColumnRoundTrip)
{
  std::vector<int64_t> const values(tile_rows + 5, 42);
  cudf::test::fixed_width_column_wrapper<int64_t> column(values.begin(), values.end());
  auto const packed_column = pack_lto(column);
  auto const result        = cudf::experimental::materialize(packed_column.view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0).view(), column);
}

TEST_F(LTOUDFTest, FilteredSumOverPartitions)
{
  constexpr int64_t threshold = 500;
  std::vector<packed> keep_alive;
  std::vector<std::vector<cudf::experimental::packed_data_view>> partitions;
  int64_t expected_sum   = 0;
  int64_t expected_count = 0;
  for (unsigned p = 0; p < 3; ++p) {
    auto const rows   = (p + 1) * tile_rows + 123 * p;
    auto const keys   = random_values<int32_t>(rows, 0, 999, 10 + p);
    auto const values = random_values<int64_t>(rows, -5'000'000, 5'000'000, 20 + p);
    for (std::size_t i = 0; i < rows; ++i) {
      if (keys[i] < threshold) {
        expected_sum += values[i];
        expected_count += 1;
      }
    }
    cudf::test::fixed_width_column_wrapper<int32_t> key_column(keys.begin(), keys.end());
    cudf::test::fixed_width_column_wrapper<int64_t> value_column(values.begin(), values.end());
    keep_alive.push_back(pack_lto(key_column));
    keep_alive.push_back(pack_lto(value_column));
    partitions.push_back({keep_alive[keep_alive.size() - 2].view(), keep_alive.back().view()});
  }

  auto const stream = cudf::get_default_stream();
  auto const source = lto::make_packed_source(partitions, stream);
  EXPECT_EQ(source.num_columns(), 2);

  rmm::device_buffer threshold_buffer(&threshold, sizeof(threshold), stream);
  auto const program =
    lto::udf{test_fragment(cudf_test_fragments::lto_filtered_sum), cudf::lto_binary_type::FATBIN};
  for (int run = 0; run < 2; ++run) {
    auto const state =
      lto::reduce(source, program, 2 * sizeof(int64_t), threshold_buffer.data(), {}, stream);
    int64_t result[2];
    ASSERT_EQ(cudaMemcpy(result, state.data(), sizeof(result), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(result[0], expected_sum);
    EXPECT_EQ(result[1], expected_count);
  }
}

TEST_F(LTOUDFTest, GroupedSumOverDictionaryCodes)
{
  std::vector<std::string> const keys{"apple", "fig", "kiwi", "lime", "pear"};
  std::vector<packed> keep_alive;
  std::vector<std::vector<cudf::experimental::packed_data_view>> partitions;
  std::vector<int64_t> expected_sum(keys.size(), 0);
  std::vector<int64_t> expected_count(keys.size(), 0);
  for (unsigned p = 0; p < 3; ++p) {
    auto const rows = (p + 2) * tile_rows + 77 * p;
    auto const codes =
      random_values<int32_t>(rows, 0, static_cast<int32_t>(keys.size()) - 1, 30 + p);
    auto const values = random_values<int64_t>(rows, -1'000'000, 1'000'000, 40 + p);
    std::vector<std::string> strings;
    for (std::size_t i = 0; i < rows; ++i) {
      strings.push_back(keys[codes[i]]);
    }
    cudf::test::dictionary_column_wrapper<std::string> key_column(strings.begin(), strings.end());
    auto const indices =
      cudf::test::to_host<int32_t>(cudf::dictionary_column_view{key_column}.get_indices_annotated())
        .first;
    for (std::size_t i = 0; i < rows; ++i) {
      if (values[i] % 7 != 0) {
        expected_sum[indices[i]] += values[i];
        expected_count[indices[i]] += 1;
      }
    }
    cudf::test::fixed_width_column_wrapper<int64_t> value_column(values.begin(), values.end());
    keep_alive.push_back(pack_lto(key_column));
    keep_alive.push_back(pack_lto(value_column));
    auto const round_trip =
      cudf::experimental::materialize(keep_alive[keep_alive.size() - 2].view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(round_trip->get_column(0).view(), key_column);
    partitions.push_back({keep_alive[keep_alive.size() - 2].view(), keep_alive.back().view()});
  }

  auto const stream = cudf::get_default_stream();
  auto const source = lto::make_packed_source(partitions, stream);
  auto const program =
    lto::udf{test_fragment(cudf_test_fragments::lto_grouped_sum), cudf::lto_binary_type::FATBIN};
  for (int run = 0; run < 2; ++run) {
    auto const states =
      lto::groupby(source, program, keys.size(), 2 * sizeof(int64_t), nullptr, {}, stream);
    std::vector<int64_t> result(2 * keys.size());
    ASSERT_EQ(cudaMemcpy(result.data(), states.data(), states.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (std::size_t g = 0; g < keys.size(); ++g) {
      EXPECT_EQ(result[2 * g], expected_sum[g]);
      EXPECT_EQ(result[2 * g + 1], expected_count[g]);
    }
  }
}

TEST_F(LTOUDFTest, LookupJoinGroupbyWithLazyColumn)
{
  // Build side: even keys in [-5000, 20000) in shuffled order, plus a null key.
  std::vector<int64_t> build_keys;
  for (int64_t k = -5000; k < 20000; k += 2) {
    build_keys.push_back(k);
  }
  std::shuffle(build_keys.begin(), build_keys.end(), std::mt19937{7});
  std::vector<bool> build_valid(build_keys.size(), true);
  build_keys.push_back(1);
  build_valid.push_back(false);
  auto const group_of = [](int64_t key) { return static_cast<int32_t>(((key % 5) + 5) % 5); };
  std::vector<int32_t> build_groups;
  for (auto const key : build_keys) {
    build_groups.push_back(group_of(key));
  }
  cudf::test::fixed_width_column_wrapper<int64_t> key_column(
    build_keys.begin(), build_keys.end(), build_valid.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> group_column(build_groups.begin(),
                                                               build_groups.end());

  auto const stream = cudf::get_default_stream();
  auto const lookup = lto::make_lookup_table(key_column, stream);
  EXPECT_EQ(lookup.num_rows(), static_cast<cudf::size_type>(build_keys.size() - 1));
  struct {
    cudf_lto_lookup lookup;
    int32_t const* groups;
  } const host_data{lookup.view(), cudf::column_view{group_column}.data<int32_t>()};
  rmm::device_buffer user_data(&host_data, sizeof(host_data), stream);

  std::vector<packed> keep_alive;
  std::vector<std::vector<cudf::experimental::packed_data_view>> partitions;
  constexpr int num_groups = 5;
  std::vector<int64_t> expected_sum(num_groups, 0);
  std::vector<int64_t> expected_count(num_groups, 0);
  for (unsigned p = 0; p < 3; ++p) {
    auto const rows   = (p + 2) * tile_rows + 51 * p;
    auto const probe  = random_values<int64_t>(rows, -6000, 21000, 50 + p);
    auto const values = random_values<int64_t>(rows, -1'000'000, 1'000'000, 60 + p);
    for (std::size_t i = 0; i < rows; ++i) {
      if (probe[i] % 2 == 0 && probe[i] >= -5000 && probe[i] < 20000) {
        expected_sum[group_of(probe[i])] += values[i];
        expected_count[group_of(probe[i])] += 1;
      }
    }
    cudf::test::fixed_width_column_wrapper<int64_t> probe_column(probe.begin(), probe.end());
    cudf::test::fixed_width_column_wrapper<int64_t> value_column(values.begin(), values.end());
    keep_alive.push_back(pack_lto(probe_column));
    keep_alive.push_back(pack_lto(value_column));
    partitions.push_back({keep_alive[keep_alive.size() - 2].view(), keep_alive.back().view()});
  }

  auto const source = lto::make_packed_source(partitions, stream);
  auto const program =
    lto::udf{test_fragment(cudf_test_fragments::lto_lookup_sum), cudf::lto_binary_type::FATBIN};
  std::vector<std::vector<bool>> const lazy_cases{{}, {false, true}, {true, true}};
  for (auto const& lazy_case : lazy_cases) {
    std::unique_ptr<bool[]> lazy(new bool[lazy_case.size()]);
    std::copy(lazy_case.begin(), lazy_case.end(), lazy.get());
    auto const states = lto::groupby(source,
                                     program,
                                     num_groups,
                                     2 * sizeof(int64_t),
                                     user_data.data(),
                                     {lazy.get(), lazy_case.size()},
                                     stream);
    std::vector<int64_t> result(2 * num_groups);
    ASSERT_EQ(cudaMemcpy(result.data(), states.data(), states.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int g = 0; g < num_groups; ++g) {
      EXPECT_EQ(result[2 * g], expected_sum[g]);
      EXPECT_EQ(result[2 * g + 1], expected_count[g]);
    }
  }
}

TEST_F(LTOUDFTest, LookupTableRejectsDuplicatesAndAcceptsEmpty)
{
  cudf::test::fixed_width_column_wrapper<int32_t> duplicated{4, 9, 4};
  EXPECT_THROW(lto::make_lookup_table(duplicated), std::invalid_argument);
  cudf::test::fixed_width_column_wrapper<double> floating{1.0, 2.0};
  EXPECT_THROW(lto::make_lookup_table(floating), std::invalid_argument);
  cudf::test::fixed_width_column_wrapper<int32_t> all_null({1, 2}, {false, false});
  auto const empty = lto::make_lookup_table(all_null);
  EXPECT_EQ(empty.num_rows(), 0);
  EXPECT_EQ(empty.view().num_keys, 0u);
}

TEST_F(LTOUDFTest, FilteredScanEmitsKeptRows)
{
  std::vector<packed> keep_alive;
  std::vector<std::vector<cudf::experimental::packed_data_view>> partitions;
  std::vector<int32_t> all_keys;
  std::vector<int64_t> all_values;
  for (unsigned p = 0; p < 2; ++p) {
    auto const rows   = (p + 2) * tile_rows + 77 * p;
    auto const keys   = random_values<int32_t>(rows, 0, 999, 30 + p);
    auto const values = random_values<int64_t>(rows, -(int64_t{1} << 40), int64_t{1} << 40, 40 + p);
    all_keys.insert(all_keys.end(), keys.begin(), keys.end());
    all_values.insert(all_values.end(), values.begin(), values.end());
    cudf::test::fixed_width_column_wrapper<int32_t> key_column(keys.begin(), keys.end());
    cudf::test::fixed_width_column_wrapper<int64_t> value_column(values.begin(), values.end());
    keep_alive.push_back(pack_lto(key_column));
    keep_alive.push_back(pack_lto(value_column));
    partitions.push_back({keep_alive[keep_alive.size() - 2].view(), keep_alive.back().view()});
  }

  auto const stream  = cudf::get_default_stream();
  auto const source  = lto::make_packed_source(partitions, stream);
  auto const program = lto::udf{test_fragment(cudf_test_fragments::lto_filtered_select),
                                cudf::lto_binary_type::FATBIN};
  std::vector<cudf::data_type> const types{cudf::data_type{cudf::type_id::INT64},
                                           cudf::data_type{cudf::type_id::INT64},
                                           cudf::data_type{cudf::type_id::INT32}};

  auto const check =
    [&](int64_t threshold, cudf::size_type expected_rows, std::span<bool const> lazy) {
      std::vector<int64_t> expected_rows_kept;
      for (std::size_t i = 0; i < all_keys.size(); ++i) {
        if (all_keys[i] < threshold) { expected_rows_kept.push_back(static_cast<int64_t>(i)); }
      }
      rmm::device_buffer threshold_buffer(&threshold, sizeof(threshold), stream);
      auto const result =
        lto::select(source, program, types, threshold_buffer.data(), lazy, expected_rows, stream);
      ASSERT_EQ(result->num_columns(), 3);
      ASSERT_EQ(static_cast<std::size_t>(result->num_rows()), expected_rows_kept.size());
      auto const n = expected_rows_kept.size();
      std::vector<int64_t> rows(n);
      std::vector<int64_t> values(n);
      std::vector<int32_t> keys(n);
      if (n > 0) {
        ASSERT_EQ(cudaMemcpy(rows.data(),
                             result->get_column(0).view().data<int64_t>(),
                             n * sizeof(int64_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(values.data(),
                             result->get_column(1).view().data<int64_t>(),
                             n * sizeof(int64_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(keys.data(),
                             result->get_column(2).view().data<int32_t>(),
                             n * sizeof(int32_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
      }
      std::vector<std::size_t> order(n);
      std::iota(order.begin(), order.end(), std::size_t{0});
      std::sort(order.begin(), order.end(), [&](auto a, auto b) { return rows[a] < rows[b]; });
      for (std::size_t i = 0; i < n; ++i) {
        auto const k   = order[i];
        auto const row = expected_rows_kept[i];
        ASSERT_EQ(rows[k], row);
        EXPECT_EQ(values[k], all_values[row]);
        EXPECT_EQ(keys[k], all_keys[row]);
      }
    };

  std::array<bool, 2> const lazy_values{false, true};
  check(100, 0, {});
  check(100, 10, {});
  check(100, 1 << 20, {});
  check(100, 0, lazy_values);
  check(1000, 0, {});
  check(0, 0, {});

  std::vector<cudf::data_type> const strings{cudf::data_type{cudf::type_id::STRING}};
  EXPECT_THROW(lto::select(source, program, strings), std::invalid_argument);
}
