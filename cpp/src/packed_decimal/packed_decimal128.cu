/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/packed_decimal/packed_decimal128.hpp>
#include <cudf/packed_decimal/packed_decimal128.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime_api.h>
#include <cuda/std/functional>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace cudf {
namespace {

constexpr size_type div_rounding_up(size_type value, size_type divisor)
{
  return (value + divisor - 1) / divisor;
}

__global__ void compute_block_widths(__int128_t const* values,
                                     bitmask_type const* validity,
                                     size_type validity_offset,
                                     size_type size,
                                     uint64_t* widths)
{
  auto const block = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const begin = block * packed_decimal128_column_view::block_size;
  if (begin >= size) { return; }

  uint8_t width = 0;
  auto const candidate_end =
    begin + static_cast<size_type>(packed_decimal128_column_view::block_size);
  auto const end = size < candidate_end ? size : candidate_end;
  for (auto row = begin; row < end; ++row) {
    if (validity == nullptr || bit_is_set(validity, validity_offset + row)) {
      auto const row_width = packed_decimal128_width(values[row]);
      width                = width < row_width ? row_width : width;
    }
  }
  widths[block] = width;
}

__global__ void encode_blocks(__int128_t const* values,
                              bitmask_type const* validity,
                              size_type validity_offset,
                              size_type size,
                              uint64_t const* descriptors,
                              uint8_t* payload)
{
  auto const row = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= size) { return; }
  auto const descriptor =
    descriptors[row / packed_decimal128_column_view::block_size];
  auto const width  = static_cast<uint8_t>(descriptor & 0x1f);
  auto const offset = descriptor >> 5;
  auto const slot   = row % packed_decimal128_column_view::block_size;
  auto value = (validity == nullptr || bit_is_set(validity, validity_offset + row))
                 ? static_cast<unsigned __int128>(values[row])
                 : 0;
  for (uint8_t byte = 0; byte < width; ++byte) {
    payload[offset + static_cast<uint64_t>(slot) * width + byte] =
      static_cast<uint8_t>(value >> (byte * 8));
  }
}

__global__ void decode_blocks(uint8_t const* payload,
                              uint64_t const* descriptors,
                              size_type input_offset,
                              size_type size,
                              __int128_t* output)
{
  auto const row = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= size) { return; }
  output[row] = packed_decimal128_device_view{payload, descriptors, input_offset}[row];
}

}  // namespace

packed_decimal128_column_view::packed_decimal128_column_view(column_view input) : column_view(input)
{
  CUDF_EXPECTS(type().id() == type_id::PACKED_DECIMAL128,
               "packed_decimal128_column_view only supports PACKED_DECIMAL128");
  CUDF_EXPECTS(num_children() == 2, "PACKED_DECIMAL128 requires payload and descriptor children");
  auto const payload = child(payload_child_index);
  CUDF_EXPECTS(payload.type().id() == type_id::UINT8,
               "PACKED_DECIMAL128 payload child must be UINT8");
  CUDF_EXPECTS(!payload.nullable(), "PACKED_DECIMAL128 payload child must be non-nullable");
  auto const descriptor = child(descriptor_child_index);
  CUDF_EXPECTS(descriptor.type().id() == type_id::UINT64,
               "PACKED_DECIMAL128 descriptors must be UINT64");
  CUDF_EXPECTS(!descriptor.nullable(), "PACKED_DECIMAL128 descriptors must be non-nullable");
  auto const required =
    size() == 0 ? 0 : div_rounding_up(offset() + size(), block_size);
  CUDF_EXPECTS(descriptor.size() >= required, "PACKED_DECIMAL128 descriptor child is too small");
}

column_view packed_decimal128_column_view::parent() const
{
  return static_cast<column_view>(*this);
}

column_view packed_decimal128_column_view::descriptors() const
{
  return child(descriptor_child_index);
}

uint8_t const* packed_decimal128_column_view::payload_begin() const noexcept
{
  return child(payload_child_index).head<uint8_t>();
}

std::unique_ptr<column> make_packed_decimal128_column(size_type size,
                                                      numeric::scale_type scale,
                                                      rmm::device_buffer&& payload,
                                                      std::unique_ptr<column> descriptors,
                                                      rmm::device_buffer&& null_mask,
                                                      size_type null_count,
                                                      cuda::stream_ref stream)
{
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative");
  CUDF_EXPECTS(descriptors != nullptr, "Descriptor child must not be null");
  CUDF_EXPECTS(descriptors->type().id() == type_id::UINT64,
               "PACKED_DECIMAL128 descriptors must be UINT64");
  CUDF_EXPECTS(!descriptors->nullable(), "PACKED_DECIMAL128 descriptors must be non-nullable");
  auto const block_count = div_rounding_up(size, packed_decimal128_column_view::block_size);
  CUDF_EXPECTS(descriptors->size() == block_count,
               "PACKED_DECIMAL128 descriptor count must equal ceil(size / 256)");
  CUDF_EXPECTS(null_count >= 0 && null_count <= size, "Invalid null count");
  CUDF_EXPECTS(null_count == 0 || null_mask.size() >= bitmask_allocation_size_bytes(size),
               "Null mask is too small");

  auto const payload_child_size = static_cast<size_type>(
    std::min<std::size_t>(payload.size(), std::numeric_limits<size_type>::max()));
  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::make_unique<column>(data_type{type_id::UINT8},
                                                  payload_child_size,
                                                  std::move(payload),
                                                  rmm::device_buffer{},
                                                  0));
  children.emplace_back(std::move(descriptors));
  return std::make_unique<column>(data_type{type_id::PACKED_DECIMAL128,
                                            static_cast<int32_t>(scale)},
                                  size,
                                  rmm::device_buffer{},
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

std::unique_ptr<column> encode_packed_decimal128(column_view input,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == type_id::DECIMAL128,
               "encode_packed_decimal128 requires DECIMAL128 input");
  auto const block_count =
    div_rounding_up(input.size(), packed_decimal128_column_view::block_size);
  auto descriptors = make_numeric_column(
    data_type{type_id::UINT64}, block_count, mask_state::UNALLOCATED, stream, mr);
  auto widths = make_numeric_column(
    data_type{type_id::UINT64}, block_count, mask_state::UNALLOCATED, stream, mr);

  if (block_count != 0) {
    constexpr int threads = 128;
    compute_block_widths<<<(block_count + threads - 1) / threads, threads, 0, stream.get()>>>(
      input.data<__int128_t>(),
      input.null_mask(),
      input.offset(),
      input.size(),
      widths->mutable_view().data<uint64_t>());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }

  auto const widths_data     = widths->view().data<uint64_t>();
  auto const descriptor_data = descriptors->mutable_view().data<uint64_t>();
  auto const counting_begin  = thrust::make_counting_iterator<size_type>(0);
  if (block_count != 0) {
    thrust::transform_exclusive_scan(
      rmm::exec_policy_nosync(stream, mr),
      counting_begin,
      counting_begin + block_count,
      descriptor_data,
      [widths_data, size = input.size()] __device__(size_type block) {
        auto const begin = block * packed_decimal128_column_view::block_size;
        auto const rows =
          min(packed_decimal128_column_view::block_size, static_cast<size_type>(size - begin));
        return widths_data[block] * static_cast<uint64_t>(rows);
      },
      uint64_t{0},
      cuda::std::plus<uint64_t>{});
  }

  uint64_t payload_size = 0;
  if (block_count != 0) {
    uint64_t last_offset{};
    uint64_t last_width{};
    CUDF_CUDA_TRY(cudaMemcpyAsync(&last_offset,
                                  descriptor_data + block_count - 1,
                                  sizeof(last_offset),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(&last_width,
                                  widths_data + block_count - 1,
                                  sizeof(last_width),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    auto const last_block_rows =
      input.size() - (block_count - 1) * packed_decimal128_column_view::block_size;
    payload_size = last_offset + last_width * static_cast<uint64_t>(last_block_rows);

    thrust::transform(
      rmm::exec_policy_nosync(stream, mr),
      counting_begin,
      counting_begin + block_count,
      descriptor_data,
      [widths_data, descriptor_data] __device__(size_type block) {
        return (descriptor_data[block] << 5) | widths_data[block];
      });
  }

  rmm::device_buffer payload{static_cast<std::size_t>(payload_size), stream, mr};
  if (input.size() != 0) {
    constexpr int threads = 256;
    encode_blocks<<<(input.size() + threads - 1) / threads, threads, 0, stream.get()>>>(
      input.data<__int128_t>(),
      input.null_mask(),
      input.offset(),
      input.size(),
      descriptors->view().data<uint64_t>(),
      static_cast<uint8_t*>(payload.data()));
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }

  auto mask = cudf::detail::copy_bitmask(input, stream, mr);
  return make_packed_decimal128_column(input.size(),
                                       numeric::scale_type{input.type().scale()},
                                       std::move(payload),
                                       std::move(descriptors),
                                       std::move(mask),
                                       input.null_count(),
                                       stream);
}

std::unique_ptr<column> decode_packed_decimal128(column_view input,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  packed_decimal128_column_view const packed{input};
  auto mask = cudf::detail::copy_bitmask(input, stream, mr);
  auto output = make_fixed_width_column(data_type{type_id::DECIMAL128, input.type().scale()},
                                        input.size(),
                                        std::move(mask),
                                        input.null_count(),
                                        stream,
                                        mr);
  if (input.size() != 0) {
    constexpr int threads = 256;
    decode_blocks<<<(input.size() + threads - 1) / threads, threads, 0, stream.get()>>>(
      packed.payload_begin(),
      packed.descriptors().data<uint64_t>(),
      input.offset(),
      input.size(),
      output->mutable_view().data<__int128_t>());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }
  return output;
}

}  // namespace cudf
