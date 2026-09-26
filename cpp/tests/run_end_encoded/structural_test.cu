/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/reshape.hpp>
#include <cudf/run_end_encoded/run_end_encoded.hpp>
#include <cudf/run_end_encoded/run_end_encoded_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_checks.hpp>

struct RunEndEncodedStructuralTest : public cudf::test::BaseFixture {};

TEST_F(RunEndEncodedStructuralTest, ZeroCopySliceAndCanonicalDeepCopy)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{10, 10, 10, 20, 20,
                                                        20, 30, 30, 30, 30};
  auto encoded = cudf::run_end_encoded::encode(input);
  auto sliced  = cudf::slice(encoded->view(), {2, 8}).front();
  cudf::run_end_encoded_column_view source{encoded->view()};
  cudf::run_end_encoded_column_view slice{sliced};

  EXPECT_EQ(slice.offset(), 2);
  EXPECT_EQ(slice.run_ends().head(), source.run_ends().head());
  EXPECT_EQ(slice.values().head(), source.values().head());

  auto copied = std::make_unique<cudf::column>(sliced);
  cudf::run_end_encoded_column_view result{copied->view()};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_ends{1, 4, 6};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_values{10, 20, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.run_ends(), expected_ends);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.values(), expected_values);
  EXPECT_EQ(result.offset(), 0);
  auto decoded = cudf::run_end_encoded::decode(result);
  cudf::test::fixed_width_column_wrapper<int32_t> expected{10, 20, 20, 20, 30, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
}

TEST_F(RunEndEncodedStructuralTest, EmptyLikeRetainsTypedChildren)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{4, 4, 9};
  auto encoded = cudf::run_end_encoded::encode(input);
  auto empty   = cudf::empty_like(encoded->view());
  cudf::run_end_encoded_column_view result{empty->view()};
  EXPECT_EQ(result.size(), 0);
  EXPECT_EQ(result.num_runs(), 0);
  EXPECT_EQ(result.values_type(), cudf::data_type{cudf::type_id::INT64});
}

TEST_F(RunEndEncodedStructuralTest, DeepCopyRebasesNullableSlice)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input(
    {1, 1, 8, 8, 2, 2, 2}, {true, true, false, false, true, true, true});
  auto encoded = cudf::run_end_encoded::encode(input);
  auto sliced  = cudf::slice(encoded->view(), {1, 6}).front();
  auto copied  = std::make_unique<cudf::column>(sliced);
  cudf::run_end_encoded_column_view result{copied->view()};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_ends{1, 3, 5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.run_ends(), expected_ends);

  auto decoded = cudf::run_end_encoded::decode(result);
  cudf::test::fixed_width_column_wrapper<int32_t> expected(
    {1, 8, 8, 2, 2}, {true, false, false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*decoded, expected);

  auto unsliced_copy = std::make_unique<cudf::column>(encoded->view());
  auto original      = cudf::run_end_encoded::decode(encoded->view());
  auto copied_plain  = cudf::run_end_encoded::decode(unsliced_copy->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*copied_plain, *original);
}

TEST_F(RunEndEncodedStructuralTest, GetElementUsesLogicalRowsAndValidity)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input(
    {5, 5, 7, 7, 9}, {true, true, false, false, true});
  auto encoded = cudf::run_end_encoded::encode(input);

  auto value = cudf::get_element(encoded->view(), 1);
  EXPECT_TRUE(value->is_valid());
  EXPECT_EQ(static_cast<cudf::numeric_scalar<int32_t> const*>(value.get())->value(), 5);
  EXPECT_FALSE(cudf::get_element(encoded->view(), 2)->is_valid());
}

TEST_F(RunEndEncodedStructuralTest, DirectGatherCanonicalizesAndPreservesConstant)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 1, 2, 2, 3, 3};
  auto encoded = cudf::run_end_encoded::encode(input);
  cudf::test::fixed_width_column_wrapper<int32_t> map{5, 4, 1, 0, 2, 3, 0};
  auto gathered = cudf::gather(cudf::table_view{{encoded->view()}}, map);
  cudf::run_end_encoded_column_view result{gathered->view().column(0)};
  cudf::test::fixed_width_column_wrapper<int32_t> expected{3, 3, 1, 1, 2, 2, 1};
  auto decoded = cudf::run_end_encoded::decode(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_ends{2, 4, 6, 7};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.run_ends(), expected_ends);

  cudf::test::fixed_width_column_wrapper<int32_t> constant{8, 8, 8};
  auto encoded_constant = cudf::run_end_encoded::encode(constant);
  cudf::test::fixed_width_column_wrapper<int32_t> constant_map{2, 0, 1, 1, 0};
  auto gathered_constant =
    cudf::gather(cudf::table_view{{encoded_constant->view()}}, constant_map);
  EXPECT_EQ(cudf::run_end_encoded_column_view{gathered_constant->view().column(0)}.num_runs(), 1);
}

TEST_F(RunEndEncodedStructuralTest, GatherNullableAndSliced)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input(
    {1, 1, 7, 7, 2, 2}, {true, true, false, false, true, true});
  auto encoded = cudf::run_end_encoded::encode(input);
  auto sliced  = cudf::slice(encoded->view(), {1, 6}).front();
  cudf::test::fixed_width_column_wrapper<int32_t> map{0, 1, 2, 4, 3};
  auto gathered = cudf::gather(cudf::table_view{{sliced}}, map);
  auto decoded  = cudf::run_end_encoded::decode(gathered->view().column(0));
  cudf::test::fixed_width_column_wrapper<int32_t> expected(
    {1, 7, 7, 2, 2}, {true, false, false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*decoded, expected);
  EXPECT_EQ(cudf::run_end_encoded_column_view{gathered->view().column(0)}.num_runs(), 3);
}

TEST_F(RunEndEncodedStructuralTest, ConcatenateCoalescesEqualBoundaries)
{
  cudf::test::fixed_width_column_wrapper<int32_t> lhs{1, 1, 2, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> equal_rhs{2, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> unequal_rhs{4, 4};
  auto lhs_ree       = cudf::run_end_encoded::encode(lhs);
  auto equal_ree     = cudf::run_end_encoded::encode(equal_rhs);
  auto unequal_ree   = cudf::run_end_encoded::encode(unequal_rhs);
  auto equal_result  = cudf::concatenate(
    std::vector<cudf::column_view>{lhs_ree->view(), equal_ree->view()});
  auto unequal_result = cudf::concatenate(
    std::vector<cudf::column_view>{lhs_ree->view(), unequal_ree->view()});

  EXPECT_EQ(cudf::run_end_encoded_column_view{equal_result->view()}.num_runs(), 3);
  EXPECT_EQ(cudf::run_end_encoded_column_view{unequal_result->view()}.num_runs(), 3);
  cudf::test::fixed_width_column_wrapper<int32_t> expected{1, 1, 2, 2, 2, 2, 3};
  auto decoded = cudf::run_end_encoded::decode(equal_result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
}

TEST_F(RunEndEncodedStructuralTest, TypeCompatibilityUsesValuesChild)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ints{1, 1};
  cudf::test::fixed_width_column_wrapper<int64_t> longs{1, 1};
  auto lhs  = cudf::run_end_encoded::encode(ints);
  auto rhs  = cudf::run_end_encoded::encode(ints);
  auto other = cudf::run_end_encoded::encode(longs);
  EXPECT_TRUE(cudf::have_same_types(lhs->view(), rhs->view()));
  EXPECT_FALSE(cudf::have_same_types(lhs->view(), other->view()));
}

TEST_F(RunEndEncodedStructuralTest, OneRowTileRemainsOneRun)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{42};
  auto encoded = cudf::run_end_encoded::encode(input);
  auto tiled   = cudf::tile(cudf::table_view{{encoded->view()}}, 32);
  cudf::run_end_encoded_column_view result{tiled->view().column(0)};
  EXPECT_EQ(result.size(), 32);
  EXPECT_EQ(result.num_runs(), 1);
}
