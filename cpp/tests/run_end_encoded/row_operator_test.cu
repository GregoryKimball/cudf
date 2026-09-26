/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/hashing.hpp>
#include <cudf/run_end_encoded/run_end_encoded.hpp>
#include <cudf/run_end_encoded/run_end_encoded_factories.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

#include <initializer_list>

namespace {

std::unique_ptr<cudf::column> make_ree(std::initializer_list<int32_t> run_ends,
                                       std::initializer_list<int32_t> values)
{
  cudf::test::fixed_width_column_wrapper<int32_t> ends(run_ends.begin(), run_ends.end());
  cudf::test::fixed_width_column_wrapper<int32_t> vals(values.begin(), values.end());
  return cudf::make_run_end_encoded_column(*(run_ends.end() - 1), ends, vals);
}

std::unique_ptr<cudf::column> equality_pairs(cudf::table_view const& lhs,
                                             cudf::table_view const& rhs,
                                             cudf::column_view const& lhs_indices,
                                             cudf::column_view const& rhs_indices,
                                             cudf::null_equality nulls)
{
  auto const stream = cudf::get_default_stream();
  auto comparator   = cudf::detail::row::equality::two_table_comparator{
    lhs, rhs, stream, cudf::get_current_device_resource_ref()};
  auto const equal = comparator.equal_to<false>(cudf::nullate::DYNAMIC{true}, nulls);
  auto output      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, lhs_indices.size(), cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    lhs_indices.begin<int32_t>(),
                    lhs_indices.end<int32_t>(),
                    rhs_indices.begin<int32_t>(),
                    output->mutable_view().begin<bool>(),
                    [equal] __device__(auto l, auto r) {
                      return equal(cudf::detail::row::lhs_index_type{l},
                                   cudf::detail::row::rhs_index_type{r});
                    });
  return output;
}

std::unique_ptr<cudf::column> self_equality_pairs(cudf::table_view const& input,
                                                  cudf::column_view const& lhs_indices,
                                                  cudf::column_view const& rhs_indices)
{
  auto const stream = cudf::get_default_stream();
  auto comparator   = cudf::detail::row::equality::self_comparator{
    input, stream, cudf::get_current_device_resource_ref()};
  auto const equal = comparator.equal_to<false>(cudf::nullate::DYNAMIC{true});
  auto output      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, lhs_indices.size(), cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    lhs_indices.begin<int32_t>(),
                    lhs_indices.end<int32_t>(),
                    rhs_indices.begin<int32_t>(),
                    output->mutable_view().begin<bool>(),
                    equal);
  return output;
}

std::unique_ptr<cudf::column> less_pairs(cudf::table_view const& lhs,
                                         cudf::table_view const& rhs,
                                         cudf::column_view const& lhs_indices,
                                         cudf::column_view const& rhs_indices)
{
  auto const stream = cudf::get_default_stream();
  auto comparator =
    cudf::detail::row::lexicographic::two_table_comparator{lhs, rhs, {}, {}, stream};
  auto const less = comparator.less<false>(cudf::nullate::DYNAMIC{true});
  auto output     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, lhs_indices.size(), cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    lhs_indices.begin<int32_t>(),
                    lhs_indices.end<int32_t>(),
                    rhs_indices.begin<int32_t>(),
                    output->mutable_view().begin<bool>(),
                    [less] __device__(auto l, auto r) {
                      return less(cudf::detail::row::lhs_index_type{l},
                                  cudf::detail::row::rhs_index_type{r});
                    });
  return output;
}

struct RunEndEncodedRowOperatorTest : cudf::test::BaseFixture {};

TEST_F(RunEndEncodedRowOperatorTest, EqualityAndOrderingUseLogicalValues)
{
  // Both columns decode to [1, 1, 1, 2, 2, 3], but use different physical run layouts.
  auto lhs = make_ree({3, 5, 6}, {1, 2, 3});
  auto rhs = make_ree({1, 3, 4, 5, 6}, {1, 1, 2, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> lhs_indices{0, 1, 3, 5, 5};
  cudf::test::fixed_width_column_wrapper<int32_t> rhs_indices{2, 3, 4, 5, 0};

  auto equal = equality_pairs(cudf::table_view{{lhs->view()}},
                              cudf::table_view{{rhs->view()}},
                              lhs_indices,
                              rhs_indices,
                              cudf::null_equality::EQUAL);
  cudf::test::fixed_width_column_wrapper<bool> expected_equal{true, false, true, true, false};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*equal, expected_equal);

  auto self_equal = self_equality_pairs(cudf::table_view{{lhs->view()}}, lhs_indices, rhs_indices);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*self_equal, expected_equal);

  auto less = less_pairs(
    cudf::table_view{{lhs->view()}}, cudf::table_view{{rhs->view()}}, lhs_indices, rhs_indices);
  cudf::test::fixed_width_column_wrapper<bool> expected_less{false, true, false, false, false};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*less, expected_less);
}

TEST_F(RunEndEncodedRowOperatorTest, ParentNullsAndSlices)
{
  cudf::test::fixed_width_column_wrapper<int32_t> plain(
    {9, 1, 1, 7, 7, 2, 2, 8}, {true, true, false, false, true, true, true, true});
  auto encoded     = cudf::run_end_encoded::encode(plain);
  auto ree_slice   = cudf::slice(encoded->view(), {1, 7}).front();
  auto plain_slice = cudf::slice(plain, {1, 7}).front();
  cudf::test::fixed_width_column_wrapper<int32_t> indices{0, 1, 2, 3, 4, 5};

  auto equal_nulls = equality_pairs(cudf::table_view{{ree_slice}},
                                    cudf::table_view{{ree_slice}},
                                    indices,
                                    indices,
                                    cudf::null_equality::EQUAL);
  cudf::test::fixed_width_column_wrapper<bool> all_equal{true, true, true, true, true, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*equal_nulls, all_equal);

  auto unequal_nulls = equality_pairs(cudf::table_view{{ree_slice}},
                                      cudf::table_view{{ree_slice}},
                                      indices,
                                      indices,
                                      cudf::null_equality::UNEQUAL);
  cudf::test::fixed_width_column_wrapper<bool> expected_unequal{
    true, false, false, true, true, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*unequal_nulls, expected_unequal);

  auto ree_hash   = cudf::hashing::murmurhash3_x86_32(cudf::table_view{{ree_slice}});
  auto plain_hash = cudf::hashing::murmurhash3_x86_32(cudf::table_view{{plain_slice}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ree_hash, *plain_hash);
}

TEST_F(RunEndEncodedRowOperatorTest, HashMatchesPlainAcrossRunLayouts)
{
  auto lhs = make_ree({3, 5, 6}, {1, 2, 3});
  auto rhs = make_ree({1, 3, 4, 5, 6}, {1, 1, 2, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> plain{1, 1, 1, 2, 2, 3};

  auto lhs_hash   = cudf::hashing::murmurhash3_x86_32(cudf::table_view{{lhs->view()}});
  auto rhs_hash   = cudf::hashing::murmurhash3_x86_32(cudf::table_view{{rhs->view()}});
  auto plain_hash = cudf::hashing::murmurhash3_x86_32(cudf::table_view{{plain}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*lhs_hash, *plain_hash);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*rhs_hash, *plain_hash);
}

}  // namespace
