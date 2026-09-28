/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/run_end_encoded/run_end_encoded.hpp>
#include <cudf/run_end_encoded/run_end_encoded_column_device_view.cuh>
#include <cudf/run_end_encoded/run_end_encoded_column_view.hpp>
#include <cudf/run_end_encoded/run_end_encoded_factories.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

#include <numeric>
#include <vector>

struct RunEndEncodedEncodeDecodeTest : public cudf::test::BaseFixture {};

struct lookup_run {
  cudf::run_end_encoded_column_device_view input;

  __device__ cudf::size_type operator()(cudf::size_type row) const { return input.find_run(row); }
};

TEST_F(RunEndEncodedEncodeDecodeTest, EmptyRetainsType)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{};
  auto encoded = cudf::run_end_encoded::encode(input);
  cudf::run_end_encoded_column_view ree{encoded->view()};

  EXPECT_EQ(ree.size(), 0);
  EXPECT_EQ(ree.num_runs(), 0);
  EXPECT_EQ(ree.values_type(), static_cast<cudf::column_view>(input).type());

  auto decoded = cudf::run_end_encoded::decode(ree);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, input);
  EXPECT_THROW(cudf::make_empty_column(cudf::type_id::RUN_END_ENCODED), cudf::data_type_error);
}

TEST_F(RunEndEncodedEncodeDecodeTest, ConstantAndManyRunInputs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> constant{9, 9, 9, 9, 9};
  auto constant_encoded = cudf::run_end_encoded::encode(constant);
  cudf::run_end_encoded_column_view constant_ree{constant_encoded->view()};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_end{5};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_value{9};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(constant_ree.run_ends(), expected_end);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(constant_ree.values(), expected_value);

  std::vector<int32_t> values(128);
  std::iota(values.begin(), values.end(), 0);
  cudf::test::fixed_width_column_wrapper<int32_t> many(values.begin(), values.end());
  auto many_encoded = cudf::run_end_encoded::encode(many);
  cudf::run_end_encoded_column_view many_ree{many_encoded->view()};
  EXPECT_EQ(many_ree.num_runs(), static_cast<cudf::size_type>(values.size()));
  auto decoded = cudf::run_end_encoded::decode(many_ree);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, many);
}

TEST_F(RunEndEncodedEncodeDecodeTest, RejectsCompoundInputBeforeDispatch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 1, 2};
  auto encoded = cudf::run_end_encoded::encode(input);
  EXPECT_THROW(cudf::run_end_encoded::encode(encoded->view()), std::invalid_argument);
}

TEST_F(RunEndEncodedEncodeDecodeTest, ValidityParticipatesInRunEquality)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input(
    {1, 9, 1, 1, 2, 7, 8}, {true, false, true, true, true, false, false});

  auto encoded = cudf::run_end_encoded::encode(input);
  cudf::run_end_encoded_column_view ree{encoded->view()};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_ends{1, 2, 4, 5, 7};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(ree.run_ends(), expected_ends);
  EXPECT_FALSE(ree.values().nullable());
  EXPECT_EQ(ree.parent().null_count(), 3);

  auto decoded = cudf::run_end_encoded::decode(ree);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*decoded, input);
}

TEST_F(RunEndEncodedEncodeDecodeTest, DecimalRoundTrip)
{
  cudf::test::fixed_point_column_wrapper<int64_t> input({125, 125, -80, -80, -80, 42},
                                                        numeric::scale_type{-2});
  auto encoded = cudf::run_end_encoded::encode(input);
  cudf::run_end_encoded_column_view ree{encoded->view()};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_ends{2, 5, 6};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(ree.run_ends(), expected_ends);
  EXPECT_EQ(ree.values_type(), static_cast<cudf::column_view>(input).type());
  auto decoded = cudf::run_end_encoded::decode(ree);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, input);
}

TEST_F(RunEndEncodedEncodeDecodeTest, ChronoRoundTrip)
{
  using timestamp = cudf::timestamp_ms;
  using duration  = cudf::duration_ms;
  cudf::test::fixed_width_column_wrapper<timestamp> input{timestamp{duration{1}},
                                                          timestamp{duration{1}},
                                                          timestamp{duration{4}},
                                                          timestamp{duration{9}},
                                                          timestamp{duration{9}}};
  auto encoded = cudf::run_end_encoded::encode(input);
  cudf::run_end_encoded_column_view ree{encoded->view()};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_ends{2, 3, 5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(ree.run_ends(), expected_ends);
  auto decoded = cudf::run_end_encoded::decode(encoded->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, input);
}

TEST_F(RunEndEncodedEncodeDecodeTest, SlicedLookupAndDecode)
{
  cudf::test::fixed_width_column_wrapper<int32_t> run_ends{3, 6, 10};
  cudf::test::fixed_width_column_wrapper<int32_t> values{10, 20, 30};
  auto encoded = cudf::make_run_end_encoded_column(10, run_ends, values);
  auto sliced  = cudf::slice(encoded->view(), {2, 8}).front();
  cudf::run_end_encoded_column_view ree{sliced};

  auto decoded = cudf::run_end_encoded::decode(ree);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_values{10, 20, 20, 20, 30, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected_values);

  auto lookup = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, ree.size(), cudf::mask_state::UNALLOCATED);
  auto d_parent   = cudf::column_device_view::create(ree.parent());
  auto device_ree = cudf::run_end_encoded_column_device_view{*d_parent};
  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{ree.size()},
                    lookup->mutable_view().begin<cudf::size_type>(),
                    lookup_run{device_ree});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_runs{0, 1, 1, 1, 2, 2};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*lookup, expected_runs);
}
