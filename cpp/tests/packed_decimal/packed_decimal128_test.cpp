/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/packed_decimal/packed_decimal128.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/sorting.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cuda/std/limits>

#include <algorithm>
#include <cstdint>
#include <vector>

class PackedDecimal128Test : public cudf::test::BaseFixture {};

TEST_F(PackedDecimal128Test, RoundTripMixedWidthsNullsAndPartialBlock)
{
  constexpr cudf::size_type size = 530;
  std::vector<__int128_t> values(size);
  std::vector<bool> validity(size, true);
  for (cudf::size_type i = 0; i < 256; ++i) {
    values[i] = (i % 2 == 0) ? 42 : -42;
  }
  for (cudf::size_type i = 256; i < 512; ++i) {
    values[i] = (static_cast<__int128_t>(1) << 100) + i;
  }
  for (cudf::size_type i = 512; i < size; ++i) {
    values[i] = (i % 2 == 0) ? 300 : -300;
  }
  values[17]   = cuda::std::numeric_limits<__int128_t>::max();
  validity[17] = false;  // Null payload values do not determine block width.
  validity[300] = false;

  auto const input = cudf::test::fixed_point_column_wrapper<__int128_t>(
    values.begin(), values.end(), validity.begin(), numeric::scale_type{-7});
  auto packed = cudf::encode_packed_decimal128(input);
  cudf::packed_decimal128_column_view packed_view{packed->view()};

  EXPECT_EQ(packed->type(), (cudf::data_type{cudf::type_id::PACKED_DECIMAL128, -7}));
  EXPECT_EQ(packed_view.descriptors().size(), 3);
  EXPECT_EQ(packed->null_count(), 2);
  auto const host_descriptors =
    cudf::test::to_host<uint64_t>(packed_view.descriptors()).first;
  ASSERT_EQ(host_descriptors.size(), 3);
  EXPECT_EQ(host_descriptors[0] & 0x1f, 1);
  EXPECT_EQ(host_descriptors[1] & 0x1f, 13);
  EXPECT_EQ(host_descriptors[2] & 0x1f, 2);

  auto decoded = cudf::decode_packed_decimal128(packed->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, decoded->view());
}

TEST_F(PackedDecimal128Test, ZeroCopySliceAndCanonicalDeepCopy)
{
  std::vector<__int128_t> values(600);
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(values.size()); ++i) {
    values[i] = i % 3 == 0 ? -i : i;
  }
  auto const input = cudf::test::fixed_point_column_wrapper<__int128_t>(
    values.begin(), values.end(), numeric::scale_type{-2});
  auto packed       = cudf::encode_packed_decimal128(input);
  auto packed_slice = cudf::slice(packed->view(), {123, 517}).front();

  cudf::packed_decimal128_column_view sliced_view{packed_slice};
  EXPECT_EQ(sliced_view.offset(), 123);
  EXPECT_EQ(sliced_view.payload_begin(),
            cudf::packed_decimal128_column_view{packed->view()}.payload_begin());

  auto deep_copy = cudf::column{packed_slice};
  cudf::packed_decimal128_column_view copied_view{deep_copy.view()};
  EXPECT_EQ(copied_view.offset(), 0);
  EXPECT_EQ(copied_view.descriptors().size(), 2);

  auto expected_view = cudf::slice(input, {123, 517}).front();
  auto decoded       = cudf::decode_packed_decimal128(deep_copy.view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_view, decoded->view());
}

TEST_F(PackedDecimal128Test, AllocationOverheadAndSavings)
{
  constexpr cudf::size_type size = 512;
  auto const max = cuda::std::numeric_limits<__int128_t>::max();
  std::vector<__int128_t> wide(size, max);
  auto const wide_input = cudf::test::fixed_point_column_wrapper<__int128_t>(
    wide.begin(), wide.end(), numeric::scale_type{0});
  auto wide_packed = cudf::encode_packed_decimal128(wide_input);
  EXPECT_EQ(wide_packed->alloc_size(), size * sizeof(__int128_t) + 2 * sizeof(uint64_t));

  std::vector<__int128_t> narrow(size, 7);
  auto const narrow_input = cudf::test::fixed_point_column_wrapper<__int128_t>(
    narrow.begin(), narrow.end(), numeric::scale_type{0});
  auto narrow_packed = cudf::encode_packed_decimal128(narrow_input);
  EXPECT_LT(narrow_packed->alloc_size(), size * sizeof(__int128_t));
}

TEST_F(PackedDecimal128Test, EmptyRetainsTypedDescriptor)
{
  auto const input =
    cudf::test::fixed_point_column_wrapper<__int128_t>{{}, numeric::scale_type{-4}};
  auto packed = cudf::encode_packed_decimal128(input);
  cudf::packed_decimal128_column_view view{packed->view()};

  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.descriptors().type().id(), cudf::type_id::UINT64);
  EXPECT_EQ(view.descriptors().size(), 0);
  EXPECT_FALSE(view.descriptors().nullable());

  auto decoded = cudf::decode_packed_decimal128(packed->view());
  EXPECT_EQ(decoded->type(), (cudf::data_type{cudf::type_id::DECIMAL128, -4}));
  EXPECT_EQ(decoded->size(), 0);
}

TEST_F(PackedDecimal128Test, GatherSlicePermutationsDuplicatesNullsAndOutOfBounds)
{
  std::vector<__int128_t> values(700);
  std::vector<bool> validity(values.size(), true);
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(values.size()); ++i) {
    values[i] = i % 5 == 0 ? -static_cast<__int128_t>(i * 17) : i * 31;
  }
  validity[355] = false;

  auto const dense = cudf::test::fixed_point_column_wrapper<__int128_t>(
    values.begin(), values.end(), validity.begin(), numeric::scale_type{-3});
  auto packed       = cudf::encode_packed_decimal128(dense);
  auto dense_slice  = cudf::slice(dense, {250, 620}).front();
  auto packed_slice = cudf::slice(packed->view(), {250, 620}).front();
  auto const map    = cudf::test::fixed_width_column_wrapper<int32_t>{{300, 2, 105, 2, -1, 999}};

  auto expected = cudf::gather(cudf::table_view{{dense_slice}},
                               map,
                               cudf::out_of_bounds_policy::NULLIFY);
  auto gathered = cudf::gather(cudf::table_view{{packed_slice}},
                               map,
                               cudf::out_of_bounds_policy::NULLIFY);
  ASSERT_EQ(gathered->view().column(0).type().id(), cudf::type_id::PACKED_DECIMAL128);
  EXPECT_EQ(gathered->view().column(0).type().scale(), -3);
  auto decoded = cudf::decode_packed_decimal128(gathered->view().column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view().column(0), decoded->view());
}

TEST_F(PackedDecimal128Test, RowHashAndOrderingMatchDecimal128)
{
  std::vector<__int128_t> values{9, -4, 9, 0, static_cast<__int128_t>(1) << 90, -17};
  std::vector<bool> validity{true, true, true, false, true, true};
  auto const dense = cudf::test::fixed_point_column_wrapper<__int128_t>(
    values.begin(), values.end(), validity.begin(), numeric::scale_type{-6});
  auto packed = cudf::encode_packed_decimal128(dense);

  auto dense_hash  = cudf::hashing::xxhash_64(cudf::table_view{{dense}});
  auto packed_hash = cudf::hashing::xxhash_64(cudf::table_view{{packed->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dense_hash->view(), packed_hash->view());

  auto const integers = cudf::test::fixed_width_column_wrapper<int32_t>{3, 1, 4, 1, 5, 9};
  auto dense_mixed_hash =
    cudf::hashing::xxhash_64(cudf::table_view{{dense, integers}});
  auto packed_mixed_hash =
    cudf::hashing::xxhash_64(cudf::table_view{{packed->view(), integers}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dense_mixed_hash->view(), packed_mixed_hash->view());

  auto dense_order  = cudf::sorted_order(cudf::table_view{{dense}});
  auto packed_order = cudf::sorted_order(cudf::table_view{{packed->view()}});
  EXPECT_EQ(cudf::test::to_host<int32_t>(dense_order->view()).first,
            cudf::test::to_host<int32_t>(packed_order->view()).first);
}

