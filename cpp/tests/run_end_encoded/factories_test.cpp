/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/null_mask.hpp>
#include <cudf/run_end_encoded/run_end_encoded_column_view.hpp>
#include <cudf/run_end_encoded/run_end_encoded_factories.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

struct RunEndEncodedFactoriesTest : public cudf::test::BaseFixture {};

TEST_F(RunEndEncodedFactoriesTest, TypeDispatchAndTraits)
{
  static_assert(cudf::type_to_id<cudf::run_end_encoded32>() == cudf::type_id::RUN_END_ENCODED);
  static_assert(cudf::is_run_end_encoded<cudf::run_end_encoded32>());
  static_assert(cudf::is_compound<cudf::run_end_encoded32>());
  static_assert(not cudf::is_nested<cudf::run_end_encoded32>());

  EXPECT_TRUE(cudf::is_run_end_encoded(cudf::data_type{cudf::type_id::RUN_END_ENCODED}));
  EXPECT_TRUE(cudf::is_compound(cudf::data_type{cudf::type_id::RUN_END_ENCODED}));
  EXPECT_FALSE(cudf::is_nested(cudf::data_type{cudf::type_id::RUN_END_ENCODED}));
  EXPECT_TRUE(cudf::is_equality_comparable(cudf::data_type{cudf::type_id::RUN_END_ENCODED}));
  EXPECT_TRUE(cudf::is_relationally_comparable(cudf::data_type{cudf::type_id::RUN_END_ENCODED}));
  EXPECT_FALSE(cudf::is_fixed_width(cudf::data_type{cudf::type_id::RUN_END_ENCODED}));
  EXPECT_FALSE(cudf::is_numeric(cudf::data_type{cudf::type_id::RUN_END_ENCODED}));
  EXPECT_FALSE(cudf::is_dictionary(cudf::data_type{cudf::type_id::RUN_END_ENCODED}));
}

TEST_F(RunEndEncodedFactoriesTest, CreateFromColumnViews)
{
  cudf::test::fixed_width_column_wrapper<int32_t> run_ends{2, 5, 10};
  cudf::test::fixed_width_column_wrapper<int64_t> values{7, 11, 13};

  auto result = cudf::make_run_end_encoded_column(10, run_ends, values);
  cudf::run_end_encoded_column_view view{result->view()};

  EXPECT_EQ(view.size(), 10);
  EXPECT_EQ(view.num_runs(), 3);
  EXPECT_EQ(view.values_type(), cudf::data_type{cudf::type_id::INT64});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.run_ends(), run_ends);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.values(), values);
}

TEST_F(RunEndEncodedFactoriesTest, ParentOwnsLogicalNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> run_ends{3, 8};
  cudf::test::fixed_width_column_wrapper<float> values{1.5f, 2.5f};
  auto mask = cudf::create_null_mask(8, cudf::mask_state::ALL_NULL);

  auto result =
    cudf::make_run_end_encoded_column(8, run_ends.release(), values.release(), 8, std::move(mask));
  cudf::run_end_encoded_column_view view{result->view()};

  EXPECT_EQ(view.null_count(), 8);
  EXPECT_FALSE(view.run_ends().nullable());
  EXPECT_FALSE(view.values().nullable());
}

TEST_F(RunEndEncodedFactoriesTest, EmptyRetainsTypedChildren)
{
  auto const values_type = cudf::data_type{cudf::type_id::DECIMAL64, -3};
  auto result            = cudf::make_empty_run_end_encoded_column(values_type);
  cudf::run_end_encoded_column_view view{result->view()};

  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.num_runs(), 0);
  EXPECT_EQ(view.run_ends().type(), cudf::data_type{cudf::type_id::INT32});
  EXPECT_EQ(view.values().type(), values_type);
  EXPECT_EQ(result->num_children(), 2);
}

TEST_F(RunEndEncodedFactoriesTest, SupportsChronoValues)
{
  cudf::test::fixed_width_column_wrapper<int32_t> run_ends{4};
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> values{
    cudf::timestamp_s{cudf::duration_s{17}}};

  auto result = cudf::make_run_end_encoded_column(4, run_ends, values);
  cudf::run_end_encoded_column_view view{result->view()};
  EXPECT_EQ(view.values_type(), cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS});
}

TEST_F(RunEndEncodedFactoriesTest, DoesNotInspectRunEndContents)
{
  cudf::test::fixed_width_column_wrapper<int32_t> invalid_run_ends{8, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> values{1, 2};

  EXPECT_NO_THROW(cudf::make_run_end_encoded_column(100, invalid_run_ends, values));
}

TEST_F(RunEndEncodedFactoriesTest, RejectsInvalidChildStructure)
{
  cudf::test::fixed_width_column_wrapper<int64_t> wrong_type_ends{2, 5};
  cudf::test::fixed_width_column_wrapper<int32_t> ends{2, 5};
  cudf::test::fixed_width_column_wrapper<int32_t> one_value{1};
  cudf::test::fixed_width_column_wrapper<int32_t> values{1, 2};
  cudf::test::strings_column_wrapper strings{"a", "b"};

  EXPECT_THROW(cudf::make_run_end_encoded_column(5, wrong_type_ends, values),
               std::invalid_argument);
  EXPECT_THROW(cudf::make_run_end_encoded_column(5, ends, one_value), std::invalid_argument);
  EXPECT_THROW(cudf::make_run_end_encoded_column(5, ends, strings), std::invalid_argument);
  EXPECT_THROW(cudf::make_run_end_encoded_column(0, ends, values), std::invalid_argument);
}

TEST_F(RunEndEncodedFactoriesTest, RejectsNullableChildren)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ends_wrapper{2, 5};
  cudf::test::fixed_width_column_wrapper<int32_t> values_wrapper{1, 2};
  auto ends   = ends_wrapper.release();
  auto values = values_wrapper.release();
  ends->set_null_mask(cudf::create_null_mask(2, cudf::mask_state::ALL_VALID), 0);

  EXPECT_THROW(cudf::make_run_end_encoded_column(
                 5, std::move(ends), std::move(values), 0, rmm::device_buffer{}),
               std::invalid_argument);
}

TEST_F(RunEndEncodedFactoriesTest, RejectsMalformedParentView)
{
  cudf::column_view malformed{
    cudf::data_type{cudf::type_id::RUN_END_ENCODED}, 0, nullptr, nullptr, 0};
  EXPECT_THROW(cudf::run_end_encoded_column_view{malformed}, cudf::logic_error);
}
