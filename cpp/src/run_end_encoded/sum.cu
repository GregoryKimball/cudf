/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/run_end_encoded/run_end_encoded.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf::run_end_encoded {
namespace {

template <typename Source, typename Output>
struct weighted_run_value {
  Source const* values;
  int32_t const* run_ends;
  int64_t slice_begin;
  int64_t slice_end;

  __device__ Output operator()(size_type run) const
  {
    auto const run_begin = run == 0 ? int64_t{0} : static_cast<int64_t>(run_ends[run - 1]);
    auto const run_end   = static_cast<int64_t>(run_ends[run]);
    auto const begin     = run_begin > slice_begin ? run_begin : slice_begin;
    auto const end       = run_end < slice_end ? run_end : slice_end;
    auto const length    = end > begin ? end - begin : int64_t{0};
    return static_cast<Output>(values[run]) * static_cast<Output>(length);
  }
};

struct sum_dispatch {
  template <typename Source, typename Output>
  std::unique_ptr<scalar> operator()(run_end_encoded_column_view const& input,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
    requires(cudf::is_numeric<Source>() && cudf::is_numeric<Output>() &&
             !std::is_same_v<Source, bool> && !std::is_same_v<Output, bool>)
  {
    auto result = cudf::make_fixed_width_scalar<Output>(Output{}, stream, mr.get_output_mr());
    auto output = static_cast<cudf::numeric_scalar<Output>*>(result.get());

    auto const runs  = input.num_runs();
    auto const first = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_type>(0),
      weighted_run_value<Source, Output>{input.values().begin<Source>(),
                                         input.run_ends().begin<int32_t>(),
                                         input.offset(),
                                         static_cast<int64_t>(input.offset()) + input.size()});

    std::size_t temporary_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temporary_bytes, first, output->data(), runs, stream.get());
    rmm::device_buffer temporary{temporary_bytes, stream, mr.get_temporary_mr()};
    cub::DeviceReduce::Sum(
      temporary.data(), temporary_bytes, first, output->data(), runs, stream.get());
    return result;
  }

  template <typename Source, typename Output>
  std::unique_ptr<scalar> operator()(run_end_encoded_column_view const&,
                                     cuda::stream_ref,
                                     cudf::memory_resources) const
  {
    CUDF_FAIL("Run-end encoded sum only supports non-BOOL numeric input and output types",
              cudf::data_type_error);
  }
};

void validate_sum_types(data_type input, data_type output)
{
  CUDF_EXPECTS(is_numeric(input) && input.id() != type_id::BOOL8,
               "Run-end encoded sum only supports non-BOOL numeric values",
               cudf::data_type_error);
  CUDF_EXPECTS(is_numeric(output) && output.id() != type_id::BOOL8,
               "Run-end encoded sum requires a non-BOOL numeric output type",
               cudf::data_type_error);
  CUDF_EXPECTS(is_integral(input) == is_integral(output),
               "Run-end encoded sum output must preserve integral or floating-point representation",
               cudf::data_type_error);
  CUDF_EXPECTS(!is_integral(input) || is_signed(input) == is_signed(output),
               "Run-end encoded sum output must preserve integral signedness",
               cudf::data_type_error);
  CUDF_EXPECTS(size_of(output) >= size_of(input),
               "Run-end encoded sum output type must not narrow the input type",
               cudf::data_type_error);
}

}  // namespace

std::unique_ptr<scalar> sum(run_end_encoded_column_view const& input,
                            data_type output_type,
                            cuda::stream_ref stream,
                            cudf::memory_resources mr)
{
  CUDF_EXPECTS(!input.is_empty(), "Run-end encoded sum does not support empty input");
  CUDF_EXPECTS(!input.parent().nullable(), "Run-end encoded sum does not support nullable input");
  validate_sum_types(input.values_type(), output_type);
  return cudf::double_type_dispatcher(
    input.values_type(), output_type, sum_dispatch{}, input, stream, mr);
}

std::unique_ptr<scalar> sum(column_view const& input,
                            data_type output_type,
                            cuda::stream_ref stream,
                            cudf::memory_resources mr)
{
  return sum(run_end_encoded_column_view{input}, output_type, stream, mr);
}

}  // namespace cudf::run_end_encoded
