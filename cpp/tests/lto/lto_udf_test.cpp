/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/lto/udf.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cudf_test_fragments.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
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
    region.options.codec                   = cudf::experimental::pack_compression::lto;
    region.options.lto_codec_id            = for_bitpack_id;
    region.options.compression_chunk_bytes = tile_rows * cudf::size_of(column.type());
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
      lto::reduce(source, program, 2 * sizeof(int64_t), threshold_buffer.data(), stream);
    int64_t result[2];
    ASSERT_EQ(cudaMemcpy(result, state.data(), sizeof(result), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(result[0], expected_sum);
    EXPECT_EQ(result[1], expected_count);
  }
}
