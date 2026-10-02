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
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>

#include <array>

namespace cudf::experimental::lto {
namespace {

constexpr uint64_t max_lookup_range = uint64_t{1} << 37;

using key_range = cuda::std::pair<int64_t, int64_t>;

template <typename Key>
lookup_table build_lookup(column_view const& keys,
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
  rmm::device_buffer ranks(num_words * sizeof(uint32_t), stream, mr);
  rmm::device_buffer rows(static_cast<std::size_t>(valid) * sizeof(uint32_t), stream, mr);
  auto* const d_bits  = static_cast<unsigned long long*>(bits.data());
  auto* const d_ranks = static_cast<uint32_t*>(ranks.data());
  auto* const d_rows  = static_cast<uint32_t*>(rows.data());
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
  switch (keys.type().id()) {
    case type_id::INT8: return build_lookup<int8_t>(keys, stream, mr);
    case type_id::INT16: return build_lookup<int16_t>(keys, stream, mr);
    case type_id::INT32: return build_lookup<int32_t>(keys, stream, mr);
    case type_id::INT64: return build_lookup<int64_t>(keys, stream, mr);
    default:
      CUDF_FAIL("LTO lookup keys must be INT8, INT16, INT32, or INT64", std::invalid_argument);
  }
}

}  // namespace cudf::experimental::lto
