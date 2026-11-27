/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/experimental/feather.hpp>
#include <cudf/io/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/buffer>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace io = cudf::io;
namespace feather = cudf::io::experimental;

// Global environment for temporary files
cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

namespace {

std::vector<io::compression_type> const codecs{
  io::compression_type::NONE, io::compression_type::LZ4, io::compression_type::ZSTD};

std::vector<std::string> default_names(cudf::size_type num_columns)
{
  std::vector<std::string> names;
  for (cudf::size_type i = 0; i < num_columns; ++i) {
    names.push_back("c" + std::to_string(i));
  }
  return names;
}

void write_file(std::string const& filepath,
                cudf::table_view const& table,
                io::compression_type compression,
                cudf::size_type max_rows_per_batch = std::numeric_limits<cudf::size_type>::max())
{
  feather::write_feather(feather::feather_writer_options::builder(io::sink_info{filepath}, table)
                           .column_names(default_names(table.num_columns()))
                           .compression(compression)
                           .max_rows_per_batch(max_rows_per_batch)
                           .build());
}

io::table_with_metadata read_file(std::string const& filepath,
                                  std::vector<std::string> columns = {})
{
  auto builder = feather::feather_reader_options::builder(io::source_info{filepath});
  if (not columns.empty()) { builder.columns(std::move(columns)); }
  return feather::read_feather(builder.build());
}

}  // namespace

struct FeatherTest : public cudf::test::BaseFixture {
  void run_test(cudf::table_view const& expected,
                cudf::size_type max_rows_per_batch = std::numeric_limits<cudf::size_type>::max())
  {
    for (auto const compression : codecs) {
      SCOPED_TRACE(static_cast<int>(compression));
      auto const filepath = temp_env->get_temp_filepath("test.feather");
      write_file(filepath, expected, compression, max_rows_per_batch);
      auto const result = read_file(filepath);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, result.tbl->view());

      auto const names = default_names(expected.num_columns());
      ASSERT_EQ(result.metadata.schema_info.size(), names.size());
      for (std::size_t i = 0; i < names.size(); ++i) {
        EXPECT_EQ(result.metadata.schema_info[i].name, names[i]);
      }
    }
  }
};

TEST_F(FeatherTest, FixedWidth)
{
  using namespace cuda::std::chrono;

  cudf::test::fixed_width_column_wrapper<int8_t> int8_col({-128, -1, 0, 1, 127},
                                                          {true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<int64_t> int64_col({1, 2, 3, 4, 5});
  std::vector<uint32_t> const uint32_data{0, 1, 2, 3, 4294967295U};
  std::vector<bool> const uint32_valid{true, true, false, true, true};
  cudf::test::fixed_width_column_wrapper<uint32_t> uint32_col(
    uint32_data.begin(), uint32_data.end(), uint32_valid.begin());
  cudf::test::fixed_width_column_wrapper<float> float_col({1.5f, 2.5f, 3.5f, 4.5f, 5.5f},
                                                          {false, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<double> double_col({1.0, 2.0, 3.0, 4.0, 5.0});
  cudf::test::fixed_width_column_wrapper<bool> bool_col({true, false, true, false, true},
                                                        {true, true, true, false, true});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D> days_col(
    {cudf::timestamp_D{cudf::duration_D{0}},
     cudf::timestamp_D{cudf::duration_D{1}},
     cudf::timestamp_D{cudf::duration_D{2}},
     cudf::timestamp_D{cudf::duration_D{3}},
     cudf::timestamp_D{cudf::duration_D{4}}},
    {true, true, true, true, false});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us> us_col(
    {cudf::timestamp_us{0us},
     cudf::timestamp_us{1us},
     cudf::timestamp_us{2us},
     cudf::timestamp_us{3us},
     cudf::timestamp_us{4us}});
  cudf::test::fixed_width_column_wrapper<cudf::duration_ms> duration_col(
    {cudf::duration_ms{0ms},
     cudf::duration_ms{10ms},
     cudf::duration_ms{20ms},
     cudf::duration_ms{30ms},
     cudf::duration_ms{40ms}},
    {true, false, true, true, true});

  auto const expected = cudf::table_view{{int8_col,
                                          int64_col,
                                          uint32_col,
                                          float_col,
                                          double_col,
                                          bool_col,
                                          days_col,
                                          us_col,
                                          duration_col}};
  run_test(expected);
}

TEST_F(FeatherTest, Decimals)
{
  using namespace numeric;

  cudf::test::fixed_point_column_wrapper<int32_t> decimal32_col{
    {12345, -67890, 0, 99999}, {true, true, false, true}, scale_type{-2}};
  cudf::test::fixed_point_column_wrapper<int64_t> decimal64_col{
    {123456789012345LL, -987654321098765LL, 0LL, 111LL}, {true, true, false, true}, scale_type{-5}};
  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128_col{
    {__int128_t{1} << 100, -(__int128_t{1} << 90), 0, 42}, {true, true, true, false}, scale_type{-10}};

  auto const expected = cudf::table_view{{decimal32_col, decimal64_col, decimal128_col}};
  run_test(expected);
}

TEST_F(FeatherTest, Strings)
{
  cudf::test::strings_column_wrapper strings_col({"Lorem", "", "dolor", "sit", "世界", "🌍"},
                                                 {true, true, false, true, true, true});
  cudf::test::strings_column_wrapper all_null_col({"", "", "", "", "", ""},
                                                  cudf::test::iterators::all_nulls());

  auto const expected = cudf::table_view{{strings_col, all_null_col}};
  run_test(expected);
}

TEST_F(FeatherTest, Lists)
{
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::lists_column_wrapper<int32_t> list_col{
    {{1, 2, 3}, valids}, {4, 5}, {}, {{6, 7, 8, 9}, valids}};
  cudf::test::lists_column_wrapper<int32_t> null_rows_col{{{1, 2}, {3}, {}, {4, 5, 6}},
                                                          cudf::test::iterators::null_at(1)};
  cudf::test::lists_column_wrapper<cudf::string_view> list_of_lists_col{
    {{"a", "b"}, {"c"}}, {{"d"}}, {}, {{}, {"e", "f", "g"}}};

  auto const expected = cudf::table_view{{list_col, null_rows_col, list_of_lists_col}};
  run_test(expected);
}

TEST_F(FeatherTest, Structs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ints{{1, 2, 3, 4}, {true, false, true, true}};
  cudf::test::strings_column_wrapper strings{{"a", "b", "", "d"}, {true, true, false, true}};
  cudf::test::structs_column_wrapper inner{{ints, strings}, {true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<double> doubles{1.0, 2.0, 3.0, 4.0};
  cudf::test::structs_column_wrapper outer{{inner, doubles}, {false, true, true, true}};

  cudf::test::lists_column_wrapper<int32_t> list_child{{1}, {2, 3}, {}, {4, 5, 6}};
  cudf::test::fixed_width_column_wrapper<int64_t> longs{10, 20, 30, 40};
  cudf::test::structs_column_wrapper struct_of_list{{list_child, longs}};

  auto const expected = cudf::table_view{{outer, struct_of_list}};
  run_test(expected);
}

TEST_F(FeatherTest, ListsOfStructs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ints{1, 2, 3, 4, 5, 6};
  cudf::test::strings_column_wrapper strings{"a", "b", "c", "d", "e", "f"};
  cudf::test::structs_column_wrapper structs{{ints, strings}, {true, false, true, true, true, true}};
  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 3, 3, 6}.release();
  auto lists   = cudf::make_lists_column(4,
                                       std::move(offsets),
                                       structs.release(),
                                       0,
                                       cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto const expected = cudf::table_view{{lists->view()}};
  run_test(expected);
}

TEST_F(FeatherTest, StructNullsCoverChildren)
{
  // Arrow lets struct children hold valid values under null parents; cudf columns must not
  auto const validity = std::vector<bool>{true, false, true, false};
  auto make_children  = [] {
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4}.release());
    children.push_back(cudf::test::strings_column_wrapper{"a", "bb", "ccc", "dddd"}.release());
    return children;
  };
  auto [mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());
  auto const input = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                                    4,
                                                    rmm::device_buffer{},
                                                    std::move(mask),
                                                    null_count,
                                                    make_children());

  cudf::test::structs_column_wrapper expected_col(make_children(), validity);
  auto const expected = cudf::table_view{{expected_col}};

  for (auto const compression : codecs) {
    SCOPED_TRACE(static_cast<int>(compression));
    auto const filepath = temp_env->get_temp_filepath("struct_nulls.feather");
    write_file(filepath, cudf::table_view{{input->view()}}, compression);
    auto const result = read_file(filepath);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  }
}

TEST_F(FeatherTest, MultipleBatches)
{
  constexpr cudf::size_type num_rows = 10'000;
  auto ints    = cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>(0));
  auto doubles = cudf::sequence(num_rows, cudf::numeric_scalar<double>(0.5));
  auto nulls   = cudf::test::iterators::nulls_at({0, 1'234, 5'000, 9'999});
  cudf::test::fixed_width_column_wrapper<int64_t> nullable(
    thrust::make_counting_iterator<int64_t>(0),
    thrust::make_counting_iterator<int64_t>(num_rows),
    nulls);
  std::vector<std::string> strings(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings[i] = std::string(i % 7, 'x');
  }
  cudf::test::strings_column_wrapper strings_col(strings.begin(), strings.end(), nulls);
  cudf::test::fixed_width_column_wrapper<int32_t> struct_child(
    thrust::make_counting_iterator<int32_t>(0), thrust::make_counting_iterator<int32_t>(num_rows));
  cudf::test::structs_column_wrapper struct_col({struct_child}, nulls);

  auto const expected =
    cudf::table_view{{ints->view(), doubles->view(), nullable, strings_col, struct_col}};
  run_test(expected, 3'000);
}

TEST_F(FeatherTest, SlicedTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ints({1, 2, 3, 4, 5, 6, 7, 8},
                                                       cudf::test::iterators::null_at(3));
  cudf::test::strings_column_wrapper strings({"a", "bb", "ccc", "", "eeee", "f", "gg", "hhh"},
                                             cudf::test::iterators::null_at(5));
  cudf::test::lists_column_wrapper<int32_t> lists{{1}, {2, 3}, {}, {4}, {5, 6}, {7}, {}, {8, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> struct_child{1, 2, 3, 4, 5, 6, 7, 8};
  cudf::test::structs_column_wrapper structs({struct_child}, cudf::test::iterators::null_at(2));
  auto const table = cudf::table_view{{ints, strings, lists, structs}};

  // Sliced from row 0 and from the middle, with and without batching
  for (auto const& [begin, end] : std::vector<std::pair<int, int>>{{0, 5}, {2, 7}}) {
    auto const expected = cudf::slice(table, {begin, end})[0];
    run_test(expected);
    run_test(expected, 2);
  }
}

TEST_F(FeatherTest, EmptyColumns)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ints{};
  cudf::test::strings_column_wrapper strings{};

  auto const expected = cudf::table_view{{ints, strings}};
  run_test(expected);
}

TEST_F(FeatherTest, SelectColumns)
{
  cudf::test::fixed_width_column_wrapper<int32_t> c0{1, 2, 3};
  cudf::test::strings_column_wrapper c1{"a", "b", "c"};
  cudf::test::fixed_width_column_wrapper<double> c2{1.0, 2.0, 3.0};
  auto const filepath = temp_env->get_temp_filepath("select.feather");
  write_file(filepath, cudf::table_view{{c0, c1, c2}}, io::compression_type::ZSTD);

  auto const result = read_file(filepath, {"c2", "c0"});
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({c2, c0}), result.tbl->view());
  ASSERT_EQ(result.metadata.schema_info.size(), 2);
  EXPECT_EQ(result.metadata.schema_info[0].name, "c2");
  EXPECT_EQ(result.metadata.schema_info[1].name, "c0");

  EXPECT_THROW(read_file(filepath, {"missing"}), cudf::logic_error);
}

TEST_F(FeatherTest, NestedColumnNames)
{
  cudf::test::lists_column_wrapper<int32_t> lists{{1}, {2, 3}};
  cudf::test::fixed_width_column_wrapper<int32_t> struct_child{1, 2};
  cudf::test::structs_column_wrapper structs({struct_child});

  io::column_name_info list_info{"l"};
  list_info.children.emplace_back("offsets");
  list_info.children.emplace_back("element");
  io::column_name_info struct_info{"s"};
  struct_info.children.emplace_back("field");

  auto const filepath = temp_env->get_temp_filepath("names.feather");
  feather::write_feather(
    feather::feather_writer_options::builder(io::sink_info{filepath},
                                             cudf::table_view{{lists, structs}})
      .column_metadata({list_info, struct_info})
      .build());

  auto const result = read_file(filepath);
  auto const& schema = result.metadata.schema_info;
  ASSERT_EQ(schema.size(), 2);
  EXPECT_EQ(schema[0].name, "l");
  ASSERT_EQ(schema[0].children.size(), 2);
  EXPECT_EQ(schema[0].children[1].name, "element");
  EXPECT_EQ(schema[1].name, "s");
  ASSERT_EQ(schema[1].children.size(), 1);
  EXPECT_EQ(schema[1].children[0].name, "field");
}

TEST_F(FeatherTest, HostAndDeviceBuffers)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ints({1, 2, 3, 4, 5},
                                                       cudf::test::iterators::null_at(2));
  cudf::test::strings_column_wrapper strings({"a", "bb", "", "dddd", "e"});
  auto const expected = cudf::table_view{{ints, strings}};

  for (auto const compression : codecs) {
    SCOPED_TRACE(static_cast<int>(compression));
    std::vector<char> buffer;
    feather::write_feather(
      feather::feather_writer_options::builder(io::sink_info{&buffer}, expected)
        .compression(compression)
        .build());

    auto const host_span = cudf::host_span<std::byte const>(
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());
    auto const from_host = feather::read_feather(
      feather::feather_reader_options::builder(io::source_info{host_span}).build());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, from_host.tbl->view());

    auto const stream = cudf::get_default_stream();
    cuda::device_buffer<std::byte> device_buffer(
      stream, cudf::get_current_device_resource_ref(), buffer.size(), cuda::no_init);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      device_buffer.data(), buffer.data(), buffer.size(), cudaMemcpyDefault, stream.get()));
    stream.sync();
    auto const device_span =
      cudf::device_span<std::byte const>(device_buffer.data(), device_buffer.size());
    auto const from_device = feather::read_feather(
      feather::feather_reader_options::builder(io::source_info{device_span}).build());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, from_device.tbl->view());
  }
}

TEST_F(FeatherTest, UnsupportedTypes)
{
  auto const filepath = temp_env->get_temp_filepath("unsupported.feather");

  cudf::test::fixed_width_column_wrapper<cudf::duration_D> days{cudf::duration_D{1}};
  EXPECT_THROW(write_file(filepath, cudf::table_view{{days}}, io::compression_type::NONE),
               cudf::logic_error);

  cudf::test::fixed_width_column_wrapper<int64_t> keys{10, 20};
  cudf::test::fixed_width_column_wrapper<int32_t> indices{0, 1, 0};
  auto const dictionary = cudf::make_dictionary_column(keys, indices);
  EXPECT_THROW(
    write_file(filepath, cudf::table_view{{dictionary->view()}}, io::compression_type::NONE),
    cudf::logic_error);
}

TEST_F(FeatherTest, UnsupportedCompression)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ints{1, 2, 3};
  auto builder = feather::feather_writer_options::builder(
    io::sink_info{temp_env->get_temp_filepath("snappy.feather")}, cudf::table_view{{ints}});
  EXPECT_THROW(builder.compression(io::compression_type::SNAPPY), cudf::logic_error);
}

TEST_F(FeatherTest, InvalidFiles)
{
  auto const filepath = temp_env->get_temp_filepath("invalid.feather");
  {
    std::ofstream file(filepath, std::ios::binary);
    file.write("SMALL", 5);
  }
  EXPECT_THROW(read_file(filepath), cudf::logic_error);

  cudf::test::fixed_width_column_wrapper<int32_t> ints{1, 2, 3};
  write_file(filepath, cudf::table_view{{ints}}, io::compression_type::NONE);
  {
    std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(-6, std::ios::end);
    file.write("XXXXXX", 6);
  }
  EXPECT_THROW(read_file(filepath), cudf::logic_error);
}

TEST_F(FeatherTest, MultipleSourcesError)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ints{1, 2, 3};
  auto const filepath = temp_env->get_temp_filepath("source.feather");
  write_file(filepath, cudf::table_view{{ints}}, io::compression_type::NONE);

  EXPECT_THROW(feather::read_feather(feather::feather_reader_options::builder(
                                       io::source_info{std::vector<std::string>{filepath, filepath}})
                                       .build()),
               cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
