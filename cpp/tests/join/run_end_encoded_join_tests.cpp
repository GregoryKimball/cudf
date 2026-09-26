/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/run_end_encoded/run_end_encoded.hpp>
#include <cudf/run_end_encoded/run_end_encoded_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace {

using join_result = std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                              std::unique_ptr<rmm::device_uvector<cudf::size_type>>>;

std::unique_ptr<cudf::column> make_ree(std::initializer_list<int32_t> run_ends,
                                       std::initializer_list<int32_t> values)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ends(run_ends.begin(), run_ends.end());
  cudf::test::fixed_width_column_wrapper<int32_t> vals(values.begin(), values.end());
  return cudf::make_run_end_encoded_column(*(run_ends.end() - 1), ends, vals);
}

std::vector<std::pair<cudf::size_type, cudf::size_type>> sorted_pairs(join_result const& result)
{
  auto const stream = cudf::get_default_stream();
  auto lhs          = cudf::detail::make_std_vector<cudf::size_type>(*result.first, stream);
  auto rhs          = cudf::detail::make_std_vector<cudf::size_type>(*result.second, stream);
  std::vector<std::pair<cudf::size_type, cudf::size_type>> pairs;
  pairs.reserve(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    pairs.emplace_back(lhs[i], rhs[i]);
  }
  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

void expect_same_join(join_result const& expected, join_result const& actual)
{
  EXPECT_EQ(sorted_pairs(expected), sorted_pairs(actual));
}

struct RunEndEncodedHashJoinTest : cudf::test::BaseFixture {};

TEST_F(RunEndEncodedHashJoinTest, InnerAndLeftDifferentRunLayouts)
{
  cudf::test::fixed_width_column_wrapper<int32_t> left_plain{1, 1, 1, 2, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> right_plain{1, 1, 2, 2, 3, 3};
  auto left_ree  = make_ree({3, 5, 6}, {1, 2, 3});
  auto right_ree = make_ree({1, 2, 4, 5, 6}, {1, 1, 2, 3, 3});

  auto plain_left  = cudf::table_view{{left_plain}};
  auto plain_right = cudf::table_view{{right_plain}};
  auto ree_left    = cudf::table_view{{left_ree->view()}};
  auto ree_right   = cudf::table_view{{right_ree->view()}};

  expect_same_join(cudf::inner_join(plain_left, plain_right, cudf::null_equality::EQUAL),
                   cudf::inner_join(ree_left, ree_right, cudf::null_equality::EQUAL));
  expect_same_join(cudf::left_join(plain_left, plain_right, cudf::null_equality::EQUAL),
                   cudf::left_join(ree_left, ree_right, cudf::null_equality::EQUAL));
}

TEST_F(RunEndEncodedHashJoinTest, MultipleKeys)
{
  cudf::test::fixed_width_column_wrapper<int32_t> left_first{1, 1, 1, 2, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> right_first{1, 1, 2, 2, 3, 3};
  cudf::test::fixed_width_column_wrapper<int64_t> left_second{5, 6, 5, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<int64_t> right_second{6, 5, 8, 7, 9, 10};
  auto left_ree  = make_ree({3, 5, 6}, {1, 2, 3});
  auto right_ree = make_ree({1, 2, 4, 5, 6}, {1, 1, 2, 3, 3});

  auto plain_left  = cudf::table_view{{left_first, left_second}};
  auto plain_right = cudf::table_view{{right_first, right_second}};
  auto ree_left    = cudf::table_view{{left_ree->view(), left_second}};
  auto ree_right   = cudf::table_view{{right_ree->view(), right_second}};

  expect_same_join(cudf::inner_join(plain_left, plain_right, cudf::null_equality::EQUAL),
                   cudf::inner_join(ree_left, ree_right, cudf::null_equality::EQUAL));
  expect_same_join(cudf::left_join(plain_left, plain_right, cudf::null_equality::EQUAL),
                   cudf::left_join(ree_left, ree_right, cudf::null_equality::EQUAL));
}

TEST_F(RunEndEncodedHashJoinTest, NullableSlicesEqualAndUnequal)
{
  cudf::test::fixed_width_column_wrapper<int32_t> left_plain(
    {9, 1, 1, 7, 2, 2, 8}, {true, true, false, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<int32_t> right_plain(
    {0, 1, 7, 7, 2, 6, 0}, {true, true, false, true, true, true, true});
  auto left_encoded  = cudf::run_end_encoded::encode(left_plain);
  auto right_encoded = cudf::run_end_encoded::encode(right_plain);
  auto left_ree      = cudf::slice(left_encoded->view(), {1, 6}).front();
  auto right_ree     = cudf::slice(right_encoded->view(), {1, 6}).front();
  auto left_slice    = cudf::slice(left_plain, {1, 6}).front();
  auto right_slice   = cudf::slice(right_plain, {1, 6}).front();

  for (auto const nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    expect_same_join(
      cudf::inner_join(cudf::table_view{{left_slice}}, cudf::table_view{{right_slice}}, nulls),
      cudf::inner_join(cudf::table_view{{left_ree}}, cudf::table_view{{right_ree}}, nulls));
    expect_same_join(
      cudf::left_join(cudf::table_view{{left_slice}}, cudf::table_view{{right_slice}}, nulls),
      cudf::left_join(cudf::table_view{{left_ree}}, cudf::table_view{{right_ree}}, nulls));
  }
}

TEST_F(RunEndEncodedHashJoinTest, ValuesChildTypeMismatchRejected)
{
  cudf::test::fixed_width_column_wrapper<int32_t> left_plain{1, 1};
  cudf::test::fixed_width_column_wrapper<int64_t> right_plain{1, 1};
  auto left_ree  = cudf::run_end_encoded::encode(left_plain);
  auto right_ree = cudf::run_end_encoded::encode(right_plain);

  EXPECT_THROW(cudf::inner_join(cudf::table_view{{left_ree->view()}},
                                cudf::table_view{{right_ree->view()}},
                                cudf::null_equality::EQUAL),
               cudf::data_type_error);
}

}  // namespace
