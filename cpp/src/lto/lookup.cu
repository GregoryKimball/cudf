/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lto/udf.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda_runtime_api.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <array>
#include <bit>

namespace cudf::experimental::lto {
namespace {

constexpr uint64_t max_lookup_range = uint64_t{1} << 37;

using key_range = cuda::std::pair<int64_t, int64_t>;

/// Builds a dense lookup over unique keys, or with `with_rows` false a dense set of any keys.
template <typename Key>
lookup_table build_lookup(column_view const& keys,
                          bool with_rows,
                          cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr)
{
  auto const temp_mr  = cudf::get_current_device_resource_ref();
  auto const policy   = rmm::exec_policy_nosync(stream, temp_mr);
  auto const d_keys   = column_device_view::create(keys, stream);
  auto const num_rows = keys.size();
  auto const valid    = num_rows - keys.null_count();
  if (valid == 0) { return lookup_table{{}, cudf_lto_lookup{nullptr, nullptr, nullptr, 0, 0}, 0}; }

  auto const range = thrust::transform_reduce(
    policy,
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{num_rows},
    [d = *d_keys] __device__(size_type i) -> key_range {
      if (d.is_null(i)) {
        return {cuda::std::numeric_limits<int64_t>::max(),
                cuda::std::numeric_limits<int64_t>::min()};
      }
      auto const key = static_cast<int64_t>(d.element<Key>(i));
      return {key, key};
    },
    key_range{cuda::std::numeric_limits<int64_t>::max(), cuda::std::numeric_limits<int64_t>::min()},
    [] __device__(key_range a, key_range b) -> key_range {
      return {a.first < b.first ? a.first : b.first, a.second > b.second ? a.second : b.second};
    });
  auto const min_key  = range.first;
  auto const num_keys = static_cast<uint64_t>(range.second) - static_cast<uint64_t>(min_key) + 1;
  CUDF_EXPECTS(num_keys <= max_lookup_range,
               "LTO lookup keys span more than 2^37 values",
               std::invalid_argument);
  auto const num_words = static_cast<std::size_t>((num_keys + 63) / 64);

  rmm::device_buffer bits(num_words * sizeof(uint64_t), stream, mr);
  auto* const d_bits = static_cast<unsigned long long*>(bits.data());
  CUDF_CUDA_TRY(cudaMemsetAsync(d_bits, 0, bits.size(), stream.get()));
  thrust::for_each_n(policy,
                     cuda::counting_iterator<size_type>{0},
                     num_rows,
                     [d = *d_keys, d_bits, min_key] __device__(size_type i) {
                       if (d.is_null(i)) { return; }
                       auto const k =
                         static_cast<uint64_t>(d.element<Key>(i)) - static_cast<uint64_t>(min_key);
                       atomicOr(d_bits + (k >> 6), 1ULL << (k & 63));
                     });
  if (!with_rows) {
    cudf_lto_lookup const view{d_bits, nullptr, nullptr, static_cast<long long>(min_key), num_keys};
    std::vector<rmm::device_buffer> storage;
    storage.push_back(std::move(bits));
    return lookup_table{std::move(storage), view, valid};
  }

  rmm::device_buffer ranks(num_words * sizeof(uint32_t), stream, mr);
  rmm::device_buffer rows(static_cast<std::size_t>(valid) * sizeof(uint32_t), stream, mr);
  auto* const d_ranks  = static_cast<uint32_t*>(ranks.data());
  auto* const d_rows   = static_cast<uint32_t*>(rows.data());
  auto const popcounts = cuda::transform_iterator(
    d_bits, [] __device__(unsigned long long word) -> uint32_t { return __popcll(word); });
  thrust::exclusive_scan(policy, popcounts, popcounts + num_words, d_ranks, uint32_t{0});

  std::array<uint64_t, 1> last_word{};
  std::array<uint32_t, 1> last_rank{};
  CUDF_CUDA_TRY(cudaMemcpyAsync(last_word.data(),
                                d_bits + num_words - 1,
                                sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(last_rank.data(),
                                d_ranks + num_words - 1,
                                sizeof(uint32_t),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  stream.sync();
  auto const distinct =
    static_cast<uint64_t>(last_rank[0]) + static_cast<uint64_t>(__builtin_popcountll(last_word[0]));
  CUDF_EXPECTS(distinct == static_cast<uint64_t>(valid),
               "LTO lookup keys must be unique",
               std::invalid_argument);

  thrust::for_each_n(
    policy,
    cuda::counting_iterator<size_type>{0},
    num_rows,
    [d = *d_keys, d_bits, d_ranks, d_rows, min_key] __device__(size_type i) {
      if (d.is_null(i)) { return; }
      auto const k   = static_cast<uint64_t>(d.element<Key>(i)) - static_cast<uint64_t>(min_key);
      auto const bit = 1ULL << (k & 63);
      d_rows[d_ranks[k >> 6] + __popcll(d_bits[k >> 6] & (bit - 1))] = static_cast<uint32_t>(i);
    });

  cudf_lto_lookup const view{d_bits, d_ranks, d_rows, static_cast<long long>(min_key), num_keys};
  std::vector<rmm::device_buffer> storage;
  storage.push_back(std::move(bits));
  storage.push_back(std::move(ranks));
  storage.push_back(std::move(rows));
  return lookup_table{std::move(storage), view, valid};
}

lookup_table dispatch_lookup(column_view const& keys,
                             bool with_rows,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  switch (keys.type().id()) {
    case type_id::INT8: return build_lookup<int8_t>(keys, with_rows, stream, mr);
    case type_id::INT16: return build_lookup<int16_t>(keys, with_rows, stream, mr);
    case type_id::INT32: return build_lookup<int32_t>(keys, with_rows, stream, mr);
    case type_id::INT64: return build_lookup<int64_t>(keys, with_rows, stream, mr);
    default:
      CUDF_FAIL("LTO lookup keys must be INT8, INT16, INT32, or INT64", std::invalid_argument);
  }
}

template <typename Key>
void widen_as(column_view const& keys, long long* words, cuda::stream_ref stream)
{
  auto const d_keys = column_device_view::create(keys, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{keys.size()},
    words,
    [d = *d_keys] __device__(size_type i) { return static_cast<long long>(d.element<Key>(i)); });
}

/// Writes the values of integer column `keys` as 64-bit words.
void widen(column_view const& keys, long long* words, cuda::stream_ref stream)
{
  switch (keys.type().id()) {
    case type_id::INT8: return widen_as<int8_t>(keys, words, stream);
    case type_id::INT16: return widen_as<int16_t>(keys, words, stream);
    case type_id::INT32: return widen_as<int32_t>(keys, words, stream);
    case type_id::INT64: return widen_as<int64_t>(keys, words, stream);
    default:
      CUDF_FAIL("LTO lookup keys must be INT8, INT16, INT32, or INT64", std::invalid_argument);
  }
}

}  // namespace

lookup_table::lookup_table(std::vector<rmm::device_buffer> storage,
                           cudf_lto_lookup view,
                           size_type num_rows)
  : _storage{std::move(storage)}, _view{view}, _num_rows{num_rows}
{
}

lookup_table make_lookup_table(column_view const& keys,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return dispatch_lookup(keys, true, stream, mr);
}

lookup_table make_key_set(column_view const& keys,
                          cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return dispatch_lookup(keys, false, stream, mr);
}

hash_lookup_table::hash_lookup_table(std::vector<rmm::device_buffer> storage,
                                     cudf_lto_hash_lookup view,
                                     size_type num_rows)
  : _storage{std::move(storage)}, _view{view}, _num_rows{num_rows}
{
}

hash_lookup_table make_hash_lookup_table(table_view const& keys,
                                         bool with_rows,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const num_words = static_cast<uint32_t>(keys.num_columns());
  CUDF_EXPECTS(num_words > 0 && num_words <= CUDF_LTO_MAX_KEY_WORDS,
               "An LTO hash lookup needs between 1 and CUDF_LTO_MAX_KEY_WORDS key columns",
               std::invalid_argument);
  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const policy  = rmm::exec_policy_nosync(stream, temp_mr);
  auto const n       = static_cast<std::size_t>(keys.num_rows());

  // Rows with a null key word are never found.
  rmm::device_uvector<bool> valid(n, stream, temp_mr);
  thrust::fill(policy, valid.begin(), valid.end(), true);
  rmm::device_uvector<long long> words(n * num_words, stream, temp_mr);
  for (uint32_t w = 0; w < num_words; ++w) {
    auto const column = keys.column(static_cast<size_type>(w));
    widen(column, words.data() + w * n, stream);
    if (column.has_nulls()) {
      auto const d_column = column_device_view::create(column, stream);
      thrust::for_each_n(policy,
                         cuda::counting_iterator<size_type>{0},
                         keys.num_rows(),
                         [d = *d_column, valid = valid.data()] __device__(size_type i) {
                           if (d.is_null(i)) { valid[i] = false; }
                         });
    }
  }

  // Insert every row: a slot holds the first row seen with its key.
  auto const capacity = std::bit_ceil(std::max<std::size_t>(2 * n, 2));
  auto const mask     = static_cast<uint64_t>(capacity - 1);
  rmm::device_uvector<uint32_t> representative(capacity, stream, temp_mr);
  rmm::device_uvector<uint32_t> row_slots(n, stream, temp_mr);
  thrust::fill(policy, representative.begin(), representative.end(), CUDF_LTO_NOT_FOUND);
  thrust::for_each_n(policy,
                     cuda::counting_iterator<std::size_t>{0},
                     n,
                     [words     = words.data(),
                      valid     = valid.data(),
                      slots     = representative.data(),
                      row_slots = row_slots.data(),
                      n,
                      num_words,
                      mask] __device__(std::size_t i) {
                       if (!valid[i]) {
                         row_slots[i] = CUDF_LTO_NOT_FOUND;
                         return;
                       }
                       long long key[CUDF_LTO_MAX_KEY_WORDS];
                       for (uint32_t w = 0; w < num_words; ++w) {
                         key[w] = words[w * n + i];
                       }
                       for (auto s = cudf_lto_hash_words(key, num_words);; ++s) {
                         auto const slot = static_cast<uint32_t>(s & mask);
                         auto const rep =
                           atomicCAS(slots + slot, CUDF_LTO_NOT_FOUND, static_cast<uint32_t>(i));
                         bool same = true;
                         for (uint32_t w = 0; rep != CUDF_LTO_NOT_FOUND && w < num_words; ++w) {
                           same = same && words[w * n + rep] == key[w];
                         }
                         if (same) {
                           row_slots[i] = slot;
                           return;
                         }
                       }
                     });

  // Number the occupied slots as groups, in slot order.
  rmm::device_uvector<uint32_t> slot_groups(capacity, stream, temp_mr);
  auto const occupied = cuda::transform_iterator(
    representative.begin(),
    [] __device__(uint32_t rep) -> uint32_t { return rep != CUDF_LTO_NOT_FOUND ? 1 : 0; });
  thrust::exclusive_scan(policy, occupied, occupied + capacity, slot_groups.begin(), uint32_t{0});
  auto const num_groups =
    slot_groups.back_element(stream) + (representative.back_element(stream) != CUDF_LTO_NOT_FOUND);

  rmm::device_buffer slots(capacity * sizeof(uint32_t), stream, mr);
  rmm::device_buffer group_keys(
    std::size_t{num_groups} * num_words * sizeof(long long), stream, mr);
  auto* const d_slots = static_cast<uint32_t*>(slots.data());
  auto* const d_keys  = static_cast<long long*>(group_keys.data());
  thrust::for_each_n(policy,
                     cuda::counting_iterator<std::size_t>{0},
                     capacity,
                     [representative = representative.data(),
                      slot_groups    = slot_groups.data(),
                      words          = words.data(),
                      d_slots,
                      d_keys,
                      n,
                      num_words,
                      num_groups] __device__(std::size_t s) {
                       auto const rep = representative[s];
                       if (rep == CUDF_LTO_NOT_FOUND) {
                         d_slots[s] = CUDF_LTO_NOT_FOUND;
                         return;
                       }
                       auto const group = slot_groups[s];
                       d_slots[s]       = group;
                       for (uint32_t w = 0; w < num_words; ++w) {
                         d_keys[std::size_t{w} * num_groups + group] = words[w * n + rep];
                       }
                     });

  std::vector<rmm::device_buffer> storage;
  cudf_lto_hash_lookup view{d_slots, d_keys, nullptr, nullptr, mask, num_groups, num_words};
  if (with_rows) {
    // Lay out each group's rows contiguously: count per slot, scan into offsets, then scatter.
    rmm::device_uvector<uint32_t> cursor(capacity, stream, temp_mr);
    thrust::fill(policy, cursor.begin(), cursor.end(), 0U);
    thrust::for_each_n(
      policy,
      cuda::counting_iterator<std::size_t>{0},
      n,
      [row_slots = row_slots.data(), cursor = cursor.data()] __device__(std::size_t i) {
        if (row_slots[i] != CUDF_LTO_NOT_FOUND) { atomicAdd(cursor + row_slots[i], 1U); }
      });
    rmm::device_uvector<uint32_t> slot_offsets(capacity, stream, temp_mr);
    thrust::exclusive_scan(policy, cursor.begin(), cursor.end(), slot_offsets.begin(), 0U);
    auto const num_valid = slot_offsets.back_element(stream) + cursor.back_element(stream);

    rmm::device_buffer offsets((std::size_t{num_groups} + 1) * sizeof(uint32_t), stream, mr);
    rmm::device_buffer rows(std::size_t{num_valid} * sizeof(uint32_t), stream, mr);
    auto* const d_offsets = static_cast<uint32_t*>(offsets.data());
    auto* const d_rows    = static_cast<uint32_t*>(rows.data());
    thrust::for_each_n(policy,
                       cuda::counting_iterator<std::size_t>{0},
                       capacity,
                       [representative = representative.data(),
                        slot_groups    = slot_groups.data(),
                        slot_offsets   = slot_offsets.data(),
                        d_offsets,
                        num_groups,
                        num_valid] __device__(std::size_t s) {
                         if (representative[s] != CUDF_LTO_NOT_FOUND) {
                           d_offsets[slot_groups[s]] = slot_offsets[s];
                         }
                         if (s == 0) { d_offsets[num_groups] = num_valid; }
                       });
    thrust::fill(policy, cursor.begin(), cursor.end(), 0U);
    thrust::for_each_n(policy,
                       cuda::counting_iterator<std::size_t>{0},
                       n,
                       [row_slots    = row_slots.data(),
                        slot_offsets = slot_offsets.data(),
                        cursor       = cursor.data(),
                        d_rows] __device__(std::size_t i) {
                         auto const slot = row_slots[i];
                         if (slot == CUDF_LTO_NOT_FOUND) { return; }
                         d_rows[slot_offsets[slot] + atomicAdd(cursor + slot, 1U)] =
                           static_cast<uint32_t>(i);
                       });
    view.offsets = d_offsets;
    view.rows    = d_rows;
    stream.sync();
    storage.push_back(std::move(offsets));
    storage.push_back(std::move(rows));
    storage.push_back(std::move(slots));
    storage.push_back(std::move(group_keys));
    return hash_lookup_table{std::move(storage), view, static_cast<size_type>(num_valid)};
  }
  auto const num_valid =
    static_cast<size_type>(thrust::count(policy, valid.begin(), valid.end(), true));
  stream.sync();
  storage.push_back(std::move(slots));
  storage.push_back(std::move(group_keys));
  return hash_lookup_table{std::move(storage), view, num_valid};
}

}  // namespace cudf::experimental::lto
