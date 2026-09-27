/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/packed_decimal/packed_decimal128.cuh>
#include <cudf/packed_decimal/packed_decimal128.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <vector>

class PackedDecimal128RowTest : public cudf::test::BaseFixture {};

void read_packed_coefficients(cudf::table_device_view table,
                              cudf::mutable_column_view output,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  thrust::transform(
    rmm::exec_policy_nosync(stream, mr),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{output.size()},
    output.begin<__int128_t>(),
    [table] __device__(cudf::size_type i) {
      auto const column   = table.column(0);
      auto const accessor = cudf::packed_decimal128_device_view{
        column.child(0).head<uint8_t>(),
        column.child(1).head<uint64_t>(),
        column.offset()};
      return accessor[i];
    });
}

template <typename Element>
void hash_elements(cudf::table_device_view table,
                   cudf::mutable_column_view output,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
{
  using hasher_type = cudf::detail::row::hash::element_hasher<
    cudf::hashing::detail::XXHash_64, cudf::nullate::DYNAMIC>;
  auto const hasher = hasher_type{cudf::nullate::DYNAMIC{false}};
  thrust::transform(
    rmm::exec_policy_nosync(stream, mr),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{output.size()},
    output.begin<uint64_t>(),
    [table, hasher] __device__(cudf::size_type i) {
      return hasher.template operator()<Element>(table.column(0), i);
    });
}

TEST_F(PackedDecimal128RowTest, DeviceTableViewReadsCoefficients)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> values{9, -4, 9, 0, static_cast<__int128_t>(1) << 90, -17};
  auto const dense = cudf::test::fixed_point_column_wrapper<__int128_t>(
    values.begin(), values.end(), numeric::scale_type{-6});
  auto packed  = cudf::encode_packed_decimal128(dense, stream, mr);
  auto d_table = cudf::table_device_view::create(cudf::table_view{{packed->view()}}, stream, mr);
  auto output  = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::DECIMAL128, -6},
    static_cast<cudf::size_type>(values.size()),
    cudf::mask_state::UNALLOCATED,
    stream,
    mr);

  read_packed_coefficients(*d_table, output->mutable_view(), stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dense, output->view());
}

TEST_F(PackedDecimal128RowTest, ElementHasherMatchesDenseStorage)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> values{9, -4, 9, 0, static_cast<__int128_t>(1) << 90, -17};
  auto const dense = cudf::test::fixed_point_column_wrapper<__int128_t>(
    values.begin(), values.end(), numeric::scale_type{-6});
  auto packed        = cudf::encode_packed_decimal128(dense, stream, mr);
  auto d_dense       = cudf::table_device_view::create(cudf::table_view{{dense}}, stream, mr);
  auto d_packed      = cudf::table_device_view::create(cudf::table_view{{packed->view()}}, stream, mr);
  auto dense_hashes  = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT64},
    static_cast<cudf::size_type>(values.size()),
    cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  auto packed_hashes = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT64},
    static_cast<cudf::size_type>(values.size()),
    cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  hash_elements<__int128_t>(*d_dense, dense_hashes->mutable_view(), stream, mr);
  hash_elements<cudf::packed_decimal128>(*d_packed, packed_hashes->mutable_view(), stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dense_hashes->view(), packed_hashes->view());
}

void hash_rows(cudf::detail::row::hash::row_hasher const& owning_hasher,
               cudf::mutable_column_view output,
               bool nullable,
               cuda::stream_ref stream,
               rmm::device_async_resource_ref mr)
{
  auto const hasher =
    owning_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(nullable);
  thrust::transform(
    rmm::exec_policy_nosync(stream, mr),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{output.size()},
    output.begin<uint64_t>(),
    hasher);
}

TEST_F(PackedDecimal128RowTest, RowHasherMatchesDenseStorage)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> values{9, -4, 9, 0, static_cast<__int128_t>(1) << 90, -17};
  std::vector<bool> validity{true, true, true, false, true, true};
  auto const dense = cudf::test::fixed_point_column_wrapper<__int128_t>(
    values.begin(), values.end(), validity.begin(), numeric::scale_type{-6});
  auto packed = cudf::encode_packed_decimal128(dense, stream, mr);
  auto dense_hashes = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT64},
    static_cast<cudf::size_type>(values.size()),
    cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  auto packed_hashes = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT64},
    static_cast<cudf::size_type>(values.size()),
    cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  auto const dense_hasher =
    cudf::detail::row::hash::row_hasher(cudf::table_view{{dense}}, stream, mr);
  auto const packed_hasher =
    cudf::detail::row::hash::row_hasher(cudf::table_view{{packed->view()}}, stream, mr);
  hash_rows(dense_hasher, dense_hashes->mutable_view(), true, stream, mr);
  hash_rows(packed_hasher, packed_hashes->mutable_view(), true, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dense_hashes->view(), packed_hashes->view());
}

TEST_F(PackedDecimal128RowTest, EqualityPreprocessingRetainsPackedChildren)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> values{9, -4, 9, 0, static_cast<__int128_t>(1) << 90, -17};
  auto const dense = cudf::test::fixed_point_column_wrapper<__int128_t>(
    values.begin(), values.end(), numeric::scale_type{-6});
  auto packed       = cudf::encode_packed_decimal128(dense, stream, mr);
  auto preprocessed = cudf::detail::row::equality::preprocessed_table::create(
    cudf::table_view{{packed->view()}}, stream, mr);
  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::DECIMAL128, -6},
    static_cast<cudf::size_type>(values.size()),
    cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  read_packed_coefficients(
    static_cast<cudf::table_device_view>(*preprocessed), output->mutable_view(), stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dense, output->view());
}

template <typename Equality>
void compare_corresponding_rows(Equality equality,
                                cudf::mutable_column_view output,
                                cudf::size_type size,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  thrust::transform(
    rmm::exec_policy_nosync(stream, mr),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{size},
    output.begin<bool>(),
    [equality] __device__(cudf::size_type i) {
      return equality(cudf::detail::row::lhs_index_type{i},
                      cudf::detail::row::rhs_index_type{i});
    });
}

TEST_F(PackedDecimal128RowTest, MixedEqualityAndOrdering)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> values{7, -2, 0, 91, static_cast<__int128_t>(1) << 80};
  auto const dense = cudf::test::fixed_point_column_wrapper<__int128_t>(
    values.begin(), values.end(), numeric::scale_type{-4});
  auto packed = cudf::encode_packed_decimal128(dense, stream, mr);

  auto equality = cudf::detail::row::equality::two_table_comparator(
    cudf::table_view{{packed->view()}},
    cudf::table_view{{dense}},
    stream,
    mr);
  auto equal = equality.equal_to<false>(cudf::nullate::DYNAMIC{false});
  auto const size = static_cast<cudf::size_type>(values.size());
  auto result     = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                              size,
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
  compare_corresponding_rows(equal, result->mutable_view(), size, stream, mr);
  auto const expected_equal = cudf::test::fixed_width_column_wrapper<bool>(
    {true, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_equal, result->view());

  auto const sorted_dense = cudf::test::fixed_point_column_wrapper<__int128_t>(
    {-9, -2, 0, 7, static_cast<__int128_t>(1) << 80}, numeric::scale_type{-4});
  auto sorted_packed = cudf::encode_packed_decimal128(sorted_dense);
  auto const needles = cudf::test::fixed_point_column_wrapper<__int128_t>(
    {-3, -2, 8, static_cast<__int128_t>(1) << 81}, numeric::scale_type{-4});
  auto packed_needles = cudf::encode_packed_decimal128(needles);
  auto dense_bounds   = cudf::lower_bound(cudf::table_view{{sorted_dense}},
                                        cudf::table_view{{packed_needles->view()}},
                                        {cudf::order::ASCENDING},
                                        {cudf::null_order::BEFORE});
  auto packed_bounds  = cudf::lower_bound(cudf::table_view{{sorted_packed->view()}},
                                         cudf::table_view{{needles}},
                                         {cudf::order::ASCENDING},
                                         {cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dense_bounds->view(), packed_bounds->view());
}
