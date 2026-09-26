/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/run_end_encoded/detail/run_end_encoded.hpp>
#include <cudf/run_end_encoded/run_end_encoded.hpp>
#include <cudf/run_end_encoded/run_end_encoded_column_device_view.cuh>
#include <cudf/run_end_encoded/run_end_encoded_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/transform.h>

namespace cudf::run_end_encoded {
namespace detail {
namespace {

size_type sliced_null_count(column_view const& input, cuda::stream_ref stream)
{
  return input.nullable()
           ? cudf::detail::count_unset_bits(
               input.null_mask(), input.offset(), input.offset() + input.size(), stream)
           : 0;
}

template <typename T>
struct is_run_start {
  column_device_view input;

  __device__ bool operator()(size_type row) const
  {
    if (row == 0) { return true; }
    auto const valid      = input.is_valid(row);
    auto const prev_valid = input.is_valid(row - 1);
    if (valid != prev_valid) { return true; }
    return valid && input.element<T>(row) != input.element<T>(row - 1);
  }
};

template <typename T>
struct is_run_end {
  column_device_view input;
  size_type size;

  __device__ bool operator()(size_type row) const
  {
    if (row + 1 == size) { return true; }
    auto const valid      = input.is_valid(row);
    auto const next_valid = input.is_valid(row + 1);
    if (valid != next_valid) { return true; }
    return valid && input.element<T>(row) != input.element<T>(row + 1);
  }
};

struct increment {
  __device__ size_type operator()(size_type value) const { return value + 1; }
};

struct encode_dispatch {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
    requires(cudf::is_fixed_width<T>())
  {
    auto const output_mr = mr.get_output_mr();
    auto const temp_mr   = mr.get_temporary_mr();
    auto d_input         = column_device_view::create(input, stream, temp_mr);
    auto const begin     = cuda::counting_iterator<size_type>{0};
    auto const end       = begin + input.size();
    auto const starts    = is_run_start<T>{*d_input};
    auto const ends      = is_run_end<T>{*d_input, input.size()};

    auto const num_runs = static_cast<size_type>(
      thrust::count_if(rmm::exec_policy(stream, temp_mr), begin, end, starts));

    auto run_starts = make_fixed_width_column(
      data_type{type_id::INT32}, num_runs, mask_state::UNALLOCATED, stream, temp_mr);
    auto run_ends = make_fixed_width_column(
      data_type{type_id::INT32}, num_runs, mask_state::UNALLOCATED, stream, output_mr);

    thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                    begin,
                    end,
                    run_starts->mutable_view().begin<size_type>(),
                    starts);
    thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                    begin,
                    end,
                    run_ends->mutable_view().begin<size_type>(),
                    ends);
    thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                      run_ends->view().begin<size_type>(),
                      run_ends->view().end<size_type>(),
                      run_ends->mutable_view().begin<size_type>(),
                      increment{});

    auto gathered = cudf::detail::gather(table_view{{input}},
                                         run_starts->view(),
                                         out_of_bounds_policy::DONT_CHECK,
                                         negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr)
                      ->release();
    auto values = std::move(gathered.front());
    // Null runs carry no semantic physical value. Children are always non-nullable.
    values->set_null_mask(rmm::device_buffer{}, 0);

    auto const null_count = sliced_null_count(input, stream);
    return make_run_end_encoded_column(input.size(),
                                       std::move(run_ends),
                                       std::move(values),
                                       null_count,
                                       cudf::detail::copy_bitmask(input, stream, output_mr));
  }

  template <typename T>
  std::unique_ptr<column> operator()(column_view const&,
                                     cuda::stream_ref,
                                     cudf::memory_resources) const
    requires(not cudf::is_fixed_width<T>())
  {
    CUDF_FAIL("Run-end encoding only supports fixed-width, decimal, and chrono values",
              cudf::data_type_error);
  }
};

struct logical_to_run {
  run_end_encoded_column_device_view input;

  // Device lookup deliberately begins with binary search; run layouts are not materialized.
  __device__ size_type operator()(size_type row) const { return input.find_run(row); }
};

}  // namespace

std::unique_ptr<column> encode(column_view const& input,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  CUDF_EXPECTS(input.type().id() != type_id::RUN_END_ENCODED,
               "Cannot run-end encode a run-end encoded column",
               std::invalid_argument);
  CUDF_EXPECTS(is_fixed_width(input.type()),
               "Run-end encoding only supports fixed-width, decimal, and chrono values",
               cudf::data_type_error);
  if (input.is_empty()) { return make_empty_run_end_encoded_column(input.type()); }
  return cudf::type_dispatcher(input.type(), encode_dispatch{}, input, stream, mr);
}

std::unique_ptr<column> decode(run_end_encoded_column_view const& input,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  if (input.is_empty()) { return make_empty_column(input.values_type()); }

  auto const output_mr = mr.get_output_mr();
  auto const temp_mr   = mr.get_temporary_mr();
  auto gather_map      = make_fixed_width_column(
    data_type{type_id::INT32}, input.size(), mask_state::UNALLOCATED, stream, temp_mr);
  auto d_input = column_device_view::create(input.parent(), stream, temp_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    gather_map->mutable_view().begin<size_type>(),
                    logical_to_run{run_end_encoded_column_device_view{*d_input}});

  auto gathered = cudf::detail::gather(table_view{{input.values()}},
                                       gather_map->view(),
                                       out_of_bounds_policy::DONT_CHECK,
                                       negative_index_policy::NOT_ALLOWED,
                                       stream,
                                       mr)
                    ->release();
  auto result = std::move(gathered.front());
  result->set_null_mask(cudf::detail::copy_bitmask(input.parent(), stream, output_mr),
                        sliced_null_count(input.parent(), stream));
  return result;
}

std::unique_ptr<column> decode(column_view const& input,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  return decode(run_end_encoded_column_view{input}, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> encode(column_view const& input,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::encode(input, stream, mr);
}

std::unique_ptr<column> decode(run_end_encoded_column_view const& input,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::decode(input, stream, mr);
}

std::unique_ptr<column> decode(column_view const& input,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::decode(input, stream, mr);
}

}  // namespace cudf::run_end_encoded
