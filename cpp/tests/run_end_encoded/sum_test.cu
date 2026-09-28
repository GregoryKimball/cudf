/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/run_end_encoded/run_end_encoded.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/wrappers/timestamps.hpp>

struct RunEndEncodedSumTest : public cudf::test::BaseFixture {};

template <typename T>
T scalar_value(std::unique_ptr<cudf::scalar> const& result)
{
  EXPECT_TRUE(result->is_valid());
  return static_cast<cudf::numeric_scalar<T> const&>(*result).value();
}

TEST_F(RunEndEncodedSumTest, ConstantInt32WidensToInt64)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{7, 7, 7, 7, 7};
  auto encoded = cudf::run_end_encoded::encode(input);

  auto result = cudf::run_end_encoded::sum(encoded->view(), cudf::data_type{cudf::type_id::INT64});

  EXPECT_EQ(scalar_value<int64_t>(result), 35);
}

TEST_F(RunEndEncodedSumTest, ManyRunsInt64)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{1, 1, 4, 2, 2, 2, -3, 8, 8};
  auto encoded = cudf::run_end_encoded::encode(input);

  auto result = cudf::run_end_encoded::sum(encoded->view(), cudf::data_type{cudf::type_id::INT64});

  EXPECT_EQ(scalar_value<int64_t>(result), 25);
}

TEST_F(RunEndEncodedSumTest, SlicedBoundaryRuns)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{100, 3, 3, 3, 10, 10, 4, 4, 100};
  auto encoded = cudf::run_end_encoded::encode(input);
  auto sliced  = cudf::slice(encoded->view(), {2, 8}).front();

  auto result = cudf::run_end_encoded::sum(sliced, cudf::data_type{cudf::type_id::INT64});

  EXPECT_EQ(scalar_value<int64_t>(result), 34);
}

TEST_F(RunEndEncodedSumTest, RejectsNullableWithoutDecoding)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input({1, 1, 2, 2}, {true, true, false, false});
  auto encoded = cudf::run_end_encoded::encode(input);

  EXPECT_THROW(cudf::run_end_encoded::sum(encoded->view(), cudf::data_type{cudf::type_id::INT64}),
               cudf::logic_error);
}

TEST_F(RunEndEncodedSumTest, RejectsUnsupportedAndNarrowingTypes)
{
  using timestamp = cudf::timestamp_ms;
  cudf::test::fixed_width_column_wrapper<timestamp> timestamps{timestamp{cudf::duration_ms{1}},
                                                               timestamp{cudf::duration_ms{1}},
                                                               timestamp{cudf::duration_ms{2}}};
  auto encoded_timestamps = cudf::run_end_encoded::encode(timestamps);
  EXPECT_THROW(
    cudf::run_end_encoded::sum(encoded_timestamps->view(), cudf::data_type{cudf::type_id::INT64}),
    cudf::data_type_error);

  cudf::test::fixed_width_column_wrapper<int64_t> integers{1, 1, 2};
  auto encoded_integers = cudf::run_end_encoded::encode(integers);
  EXPECT_THROW(
    cudf::run_end_encoded::sum(encoded_integers->view(), cudf::data_type{cudf::type_id::INT32}),
    cudf::data_type_error);
}
