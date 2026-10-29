/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/lto/codecs/for_bitpack.hpp>
#include <cudf/lto/filter_gather.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using cudf::lto::encoded_column_view;
using cudf::lto::filter_input;

struct LtoFilterGatherTest : public cudf::test::BaseFixture {};

/// An encoded column together with the buffer that owns its data.
struct encoded_column {
  rmm::device_buffer data;
  encoded_column_view view;
};

encoded_column for_bitpack(cudf::column_view const& col, cudf::size_type block_size = 1024)
{
  auto data = cudf::lto::codecs::for_bitpack_encode(col, block_size);
  auto view = encoded_column_view{.type        = col.type(),
                                  .size        = col.size(),
                                  .column_data = data.data(),
                                  .codec       = &cudf::lto::codecs::for_bitpack_codec()};
  return {std::move(data), view};
}

/// Applies `predicate` to `table` with the plain cuDF operators and keeps the `gather_map` columns.
std::unique_ptr<cudf::table> reference(cudf::table_view const& table,
                                       cudf::ast::expression const& predicate,
                                       std::vector<cudf::size_type> const& gather_map)
{
  auto mask = cudf::compute_column(table, predicate);
  return cudf::apply_retention_mask(table.select(gather_map), mask->view());
}

template <typename T>
std::vector<T> random_values(std::size_t n, T lo, T hi, unsigned seed)
{
  std::mt19937_64 gen{seed};
  std::uniform_int_distribution<T> dist{lo, hi};
  std::vector<T> values(n);
  for (auto& v : values) {
    v = dist(gen);
  }
  return values;
}

}  // namespace

TEST_F(LtoFilterGatherTest, ForBitpackInt64)
{
  auto const n = 100'003;
  auto a       = random_values<int64_t>(n, -1'000'000, 1'000'000, 1);
  auto b       = random_values<int64_t>(n, 0, 1'000, 2);
  cudf::test::fixed_width_column_wrapper<int64_t> col_a(a.begin(), a.end());
  cudf::test::fixed_width_column_wrapper<int64_t> col_b(b.begin(), b.end());
  auto enc_a = for_bitpack(col_a);
  auto enc_b = for_bitpack(col_b);

  auto threshold = cudf::numeric_scalar<int64_t>(250'000);
  auto literal   = cudf::ast::literal(threshold);
  auto column    = cudf::ast::column_reference(0);
  auto predicate = cudf::ast::operation(cudf::ast::ast_operator::GREATER, column, literal);

  std::vector<filter_input> inputs{enc_a.view, enc_b.view};
  std::vector<cudf::size_type> gather_map{1, 0};
  auto result = cudf::lto::filter_gather(inputs, predicate, gather_map);

  auto expected = reference(cudf::table_view{{col_a, col_b}}, predicate, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(LtoFilterGatherTest, ForBitpackWidths)
{
  auto const n = 5'000;
  for (int width : {0, 1, 7, 31, 32, 33, 63, 64}) {
    std::vector<int64_t> values(n);
    std::mt19937_64 gen{static_cast<unsigned>(width)};
    auto const reference_value = width == 64 ? std::numeric_limits<int64_t>::min() : int64_t{-17};
    for (int i = 0; i < n; ++i) {
      auto const mask  = width == 64 ? ~0ULL : (1ULL << width) - 1;
      auto const delta = width == 0 ? 0ULL : (gen() & mask);
      values[i]        = static_cast<int64_t>(static_cast<uint64_t>(reference_value) + delta);
    }
    if (width > 0) { values[0] = static_cast<int64_t>(static_cast<uint64_t>(reference_value)); }
    cudf::test::fixed_width_column_wrapper<int64_t> col(values.begin(), values.end());
    auto enc = for_bitpack(col, 777);

    auto column    = cudf::ast::column_reference(0);
    auto predicate = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, column, column);

    std::vector<filter_input> inputs{enc.view};
    std::vector<cudf::size_type> gather_map{0};
    auto result = cudf::lto::filter_gather(inputs, predicate, gather_map);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col, result->get_column(0).view());
  }
}

// TPC-H Q6 shape: date range, DECIMAL64 between, and an INT32 comparison over a plain column.
TEST_F(LtoFilterGatherTest, MixedTypesAndPlainInputs)
{
  auto const n = 70'001;
  auto days    = random_values<int32_t>(n, 8'000, 10'600, 3);
  auto disc    = random_values<int64_t>(n, 0, 10, 4);
  auto qty     = random_values<int32_t>(n, 1, 50, 5);
  auto price   = random_values<int64_t>(n, 90'000, 10'500'000, 6);

  using cudf::timestamp_D;
  std::vector<timestamp_D> dates;
  for (auto d : days) {
    dates.emplace_back(timestamp_D{cudf::duration_D{d}});
  }
  cudf::test::fixed_width_column_wrapper<timestamp_D, timestamp_D::rep> col_date(dates.begin(),
                                                                                 dates.end());
  cudf::test::fixed_point_column_wrapper<int64_t> col_disc(
    disc.begin(), disc.end(), numeric::scale_type{-2});
  cudf::test::fixed_width_column_wrapper<int32_t> col_qty(qty.begin(), qty.end());
  cudf::test::fixed_point_column_wrapper<int64_t> col_price(
    price.begin(), price.end(), numeric::scale_type{-2});

  auto enc_date  = for_bitpack(col_date);
  auto enc_disc  = for_bitpack(col_disc);
  auto enc_price = for_bitpack(col_price);

  auto lo_date  = cudf::timestamp_scalar<timestamp_D>(timestamp_D{cudf::duration_D{8'766}}, true);
  auto hi_date  = cudf::timestamp_scalar<timestamp_D>(timestamp_D{cudf::duration_D{9'131}}, true);
  auto lo_disc  = cudf::fixed_point_scalar<numeric::decimal64>(5, numeric::scale_type{-2});
  auto hi_disc  = cudf::fixed_point_scalar<numeric::decimal64>(7, numeric::scale_type{-2});
  auto max_qty  = cudf::numeric_scalar<int32_t>(24);
  auto lit_lo_d = cudf::ast::literal(lo_date);
  auto lit_hi_d = cudf::ast::literal(hi_date);
  auto lit_lo_x = cudf::ast::literal(lo_disc);
  auto lit_hi_x = cudf::ast::literal(hi_disc);
  auto lit_qty  = cudf::ast::literal(max_qty);

  cudf::ast::tree tree;
  auto const& date  = tree.push(cudf::ast::column_reference(0));
  auto const& d     = tree.push(cudf::ast::column_reference(1));
  auto const& q     = tree.push(cudf::ast::column_reference(2));
  using op          = cudf::ast::ast_operator;
  auto const& c0    = tree.push(cudf::ast::operation(op::GREATER_EQUAL, date, lit_lo_d));
  auto const& c1    = tree.push(cudf::ast::operation(op::LESS, date, lit_hi_d));
  auto const& c2    = tree.push(cudf::ast::operation(op::GREATER_EQUAL, d, lit_lo_x));
  auto const& c3    = tree.push(cudf::ast::operation(op::LESS_EQUAL, d, lit_hi_x));
  auto const& c4    = tree.push(cudf::ast::operation(op::LESS, q, lit_qty));
  auto const& c01   = tree.push(cudf::ast::operation(op::LOGICAL_AND, c0, c1));
  auto const& c23   = tree.push(cudf::ast::operation(op::LOGICAL_AND, c2, c3));
  auto const& c0123 = tree.push(cudf::ast::operation(op::LOGICAL_AND, c01, c23));
  auto const& pred  = tree.push(cudf::ast::operation(op::LOGICAL_AND, c0123, c4));

  std::vector<filter_input> inputs{
    enc_date.view, enc_disc.view, cudf::column_view{col_qty}, enc_price.view};
  std::vector<cudf::size_type> gather_map{3, 1};
  auto result = cudf::lto::filter_gather(inputs, pred, gather_map);

  auto expected =
    reference(cudf::table_view{{col_date, col_disc, col_qty, col_price}}, pred, gather_map);
  EXPECT_GT(expected->num_rows(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

// A codec supplied by the application: value = base + step * row.
TEST_F(LtoFilterGatherTest, UserCodec)
{
  constexpr char const* source = R"***(
#include <cudf/lto/codec.cuh>

struct affine {
  long long base;
  long long step;
};

extern "C" __device__ void test_affine_load(void* values,
                                            cudf::lto::codec_scope const* scope,
                                            long long row,
                                            long long count)
{
  auto const* a = static_cast<affine const*>(scope->column_data);
  for (long long i = 0; i < count; ++i) {
    static_cast<long long*>(values)[i] = a->base + a->step * (row + i);
  }
}
)***";
  auto codec                   = cudf::lto::column_codec{
                      .fragments = {cudf::lto::compile_lto_ir("test_affine_codec.cu", source)},
                      .load      = "test_affine_load"};

  auto const n = 10'000;
  std::vector<int64_t> params{-500, 3};
  rmm::device_buffer affine_data(
    params.data(), params.size() * sizeof(int64_t), cudf::get_default_stream());
  auto affine = encoded_column_view{.type        = cudf::data_type{cudf::type_id::INT64},
                                    .size        = n,
                                    .column_data = affine_data.data(),
                                    .codec       = &codec};

  std::vector<int64_t> expected_values(n);
  for (int i = 0; i < n; ++i) {
    expected_values[i] = params[0] + params[1] * i;
  }
  auto other = random_values<int64_t>(n, 0, 100, 7);
  cudf::test::fixed_width_column_wrapper<int64_t> col_affine(expected_values.begin(),
                                                             expected_values.end());
  cudf::test::fixed_width_column_wrapper<int64_t> col_other(other.begin(), other.end());
  auto enc_other = for_bitpack(col_other);

  auto threshold = cudf::numeric_scalar<int64_t>(4'000);
  auto literal   = cudf::ast::literal(threshold);
  auto column    = cudf::ast::column_reference(0);
  auto predicate = cudf::ast::operation(cudf::ast::ast_operator::LESS, column, literal);

  std::vector<filter_input> inputs{affine, enc_other.view};
  std::vector<cudf::size_type> gather_map{0, 1};
  auto result = cudf::lto::filter_gather(inputs, predicate, gather_map);

  auto expected = reference(cudf::table_view{{col_affine, col_other}}, predicate, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(LtoFilterGatherTest, SelectNoneAllAndEmpty)
{
  auto values = random_values<int32_t>(3'000, 0, 99, 8);
  cudf::test::fixed_width_column_wrapper<int32_t> col(values.begin(), values.end());
  auto enc = for_bitpack(col, 100);
  std::vector<filter_input> inputs{enc.view};
  std::vector<cudf::size_type> gather_map{0};
  auto column = cudf::ast::column_reference(0);

  auto none_scalar = cudf::numeric_scalar<int32_t>(100);
  auto none_lit    = cudf::ast::literal(none_scalar);
  auto none        = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, column, none_lit);
  EXPECT_EQ(cudf::lto::filter_gather(inputs, none, gather_map)->num_rows(), 0);

  auto all_scalar = cudf::numeric_scalar<int32_t>(0);
  auto all_lit    = cudf::ast::literal(all_scalar);
  auto all        = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, column, all_lit);
  auto result     = cudf::lto::filter_gather(inputs, all, gather_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col, result->get_column(0).view());

  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  auto empty = for_bitpack(empty_col);
  std::vector<filter_input> empty_inputs{empty.view};
  auto empty_result = cudf::lto::filter_gather(empty_inputs, all, gather_map);
  EXPECT_EQ(empty_result->num_rows(), 0);
  EXPECT_EQ(empty_result->get_column(0).type().id(), cudf::type_id::INT32);
}

TEST_F(LtoFilterGatherTest, Slices)
{
  auto a = random_values<int64_t>(20'000, -50, 50, 9);
  auto b = random_values<int64_t>(20'000, -50, 50, 10);
  cudf::test::fixed_width_column_wrapper<int64_t> col_a(a.begin(), a.end());
  cudf::test::fixed_width_column_wrapper<int64_t> col_b(b.begin(), b.end());
  auto slice_a = cudf::slice(col_a, {1'234, 17'777})[0];
  auto slice_b = cudf::slice(col_b, {1'234, 17'777})[0];
  auto enc_a   = for_bitpack(slice_a, 300);

  auto zero      = cudf::numeric_scalar<int64_t>(0);
  auto literal   = cudf::ast::literal(zero);
  auto column    = cudf::ast::column_reference(1);
  auto predicate = cudf::ast::operation(cudf::ast::ast_operator::GREATER, column, literal);

  std::vector<filter_input> inputs{enc_a.view, slice_b};
  std::vector<cudf::size_type> gather_map{0, 1};
  auto result = cudf::lto::filter_gather(inputs, predicate, gather_map);

  auto expected = reference(cudf::table_view{{slice_a, slice_b}}, predicate, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(LtoFilterGatherTest, Decode)
{
  auto a = random_values<int32_t>(9'999, -7, 1'000'000, 11);
  auto b = random_values<int64_t>(9'999, 5, 9, 12);
  cudf::test::fixed_width_column_wrapper<int32_t> col_a(a.begin(), a.end());
  cudf::test::fixed_width_column_wrapper<int64_t> col_b(b.begin(), b.end());
  auto enc_a = for_bitpack(col_a, 512);

  std::vector<filter_input> inputs{enc_a.view, cudf::column_view{col_b}};
  std::vector<cudf::size_type> gather_map{1, 0};
  auto result = cudf::lto::decode(inputs, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({col_b, col_a}), result->view());
}

TEST_F(LtoFilterGatherTest, InvalidInputs)
{
  cudf::test::fixed_width_column_wrapper<int64_t> nullable({1, 2, 3}, {true, false, true});
  cudf::test::fixed_width_column_wrapper<int64_t> plain({1, 2, 3});
  auto column    = cudf::ast::column_reference(0);
  auto predicate = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, column, column);
  std::vector<cudf::size_type> gather_map{0};

  std::vector<filter_input> with_nulls{cudf::column_view{nullable}};
  EXPECT_THROW(cudf::lto::filter_gather(with_nulls, predicate, gather_map), std::invalid_argument);

  auto no_codec = encoded_column_view{.type = cudf::data_type{cudf::type_id::INT64}, .size = 3};
  std::vector<filter_input> missing_codec{no_codec};
  EXPECT_THROW(cudf::lto::filter_gather(missing_codec, predicate, gather_map),
               std::invalid_argument);

  std::vector<filter_input> plain_inputs{cudf::column_view{plain}};
  std::vector<cudf::size_type> bad_map{1};
  EXPECT_THROW(cudf::lto::filter_gather(plain_inputs, predicate, bad_map), std::out_of_range);

  EXPECT_THROW(cudf::lto::codecs::for_bitpack_encode(nullable), std::invalid_argument);
}

CUDF_TEST_PROGRAM_MAIN()
