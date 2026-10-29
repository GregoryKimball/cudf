/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/lto/codecs/for_bitpack.cuh>
#include <cudf/lto/codecs/for_bitpack.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/limits>
#include <thrust/scan.h>

namespace cudf {
namespace lto {
namespace codecs {
namespace {

namespace fb = for_bitpack;

constexpr int encode_threads = 256;

constexpr char const* load_source = R"***(
#include <cudf/lto/codecs/for_bitpack.cuh>

extern "C" __device__ void cudf_lto_for_bitpack_load(void* values,
                                                     cudf::lto::codec_scope const* scope,
                                                     long long row,
                                                     long long count)
{
  auto const* data = static_cast<unsigned char const*>(scope->column_data);
  auto const value_bytes =
    reinterpret_cast<cudf::lto::codecs::for_bitpack::header const*>(data)->value_bytes;
  for (long long i = 0; i < count; ++i) {
    auto const value = cudf::lto::codecs::for_bitpack::decode(data, row + i);
    if (value_bytes == 4) {
      static_cast<int*>(values)[i] = static_cast<int>(value);
    } else {
      static_cast<long long*>(values)[i] = value;
    }
  }
}
)***";

template <typename T>
__device__ int64_t load_value(void const* data, int64_t row)
{
  return static_cast<int64_t>(static_cast<T const*>(data)[row]);
}

template <typename T>
CUDF_KERNEL void block_stats_kernel(void const* data,
                                    int64_t num_rows,
                                    int block_size,
                                    int64_t* references,
                                    uint8_t* widths,
                                    int64_t* block_words)
{
  using reduce = cub::BlockReduce<int64_t, encode_threads>;
  __shared__ typename reduce::TempStorage storage;

  auto const block = static_cast<int64_t>(blockIdx.x);
  auto const begin = block * block_size;
  auto const end   = cuda::std::min(begin + block_size, num_rows);

  int64_t lo = cuda::std::numeric_limits<int64_t>::max();
  int64_t hi = cuda::std::numeric_limits<int64_t>::min();
  for (auto row = begin + threadIdx.x; row < end; row += encode_threads) {
    auto const value = load_value<T>(data, row);
    lo               = cuda::std::min(lo, value);
    hi               = cuda::std::max(hi, value);
  }
  lo = reduce(storage).Reduce(lo, cuda::minimum<>{});
  __syncthreads();
  hi = reduce(storage).Reduce(hi, cuda::maximum<>{});

  if (threadIdx.x == 0) {
    auto const range   = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
    auto const width   = range == 0 ? 0 : 64 - __clzll(static_cast<long long>(range));
    references[block]  = lo;
    widths[block]      = static_cast<uint8_t>(width);
    block_words[block] = static_cast<int64_t>(
      (static_cast<uint64_t>(end - begin) * static_cast<uint64_t>(width) + 63) / 64);
  }
}

template <typename T>
CUDF_KERNEL void pack_kernel(void const* data,
                             int64_t num_rows,
                             int block_size,
                             int64_t const* references,
                             uint8_t const* widths,
                             int64_t const* word_offsets,
                             unsigned long long* words)
{
  auto const row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_rows) { return; }
  auto const block = row / block_size;
  auto const width = static_cast<unsigned>(widths[block]);
  if (width == 0) { return; }

  auto const delta =
    static_cast<uint64_t>(load_value<T>(data, row)) - static_cast<uint64_t>(references[block]);
  auto const bit   = static_cast<uint64_t>(row - block * block_size) * width;
  auto const word  = word_offsets[block] + static_cast<int64_t>(bit >> 6);
  auto const shift = static_cast<unsigned>(bit & 63);

  atomicOr(words + word, static_cast<unsigned long long>(delta << shift));
  if (shift + width > 64) {
    atomicOr(words + word + 1, static_cast<unsigned long long>(delta >> (64 - shift)));
  }
}

int64_t align8(int64_t bytes) { return (bytes + 7) / 8 * 8; }

}  // namespace

column_codec const& for_bitpack_codec()
{
  static column_codec const codec{
    .fragments = {compile_lto_ir("cudf/lto/codecs/for_bitpack_load.cu", load_source)},
    .load      = "cudf_lto_for_bitpack_load"};
  return codec;
}

rmm::device_buffer for_bitpack_encode(column_view const& input,
                                      size_type block_size,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(!input.has_nulls(), "FOR-bitpack encoding requires no nulls", std::invalid_argument);
  CUDF_EXPECTS(block_size > 0, "FOR-bitpack block size must be positive", std::invalid_argument);
  auto const value_bytes = static_cast<int>(size_of(input.type()));
  CUDF_EXPECTS(is_fixed_width(input.type()) && input.type().id() != type_id::BOOL8 &&
                 !is_floating_point(input.type()) && (value_bytes == 4 || value_bytes == 8),
               "FOR-bitpack encoding requires 32- or 64-bit integer storage",
               std::invalid_argument);

  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const num_rows   = static_cast<int64_t>(input.size());
  auto const num_blocks = cudf::util::div_rounding_up_safe<int64_t>(num_rows, block_size);
  auto const* data      = input.head<uint8_t>() + static_cast<size_t>(input.offset()) * value_bytes;

  rmm::device_uvector<int64_t> references(num_blocks, stream, temp_mr);
  rmm::device_uvector<uint8_t> widths(num_blocks, stream, temp_mr);
  rmm::device_uvector<int64_t> word_offsets(num_blocks + 1, stream, temp_mr);

  if (num_blocks > 0) {
    auto const grid = static_cast<unsigned>(num_blocks);
    if (value_bytes == 4) {
      block_stats_kernel<int32_t><<<grid, encode_threads, 0, stream.get()>>>(
        data, num_rows, block_size, references.data(), widths.data(), word_offsets.data());
    } else {
      block_stats_kernel<int64_t><<<grid, encode_threads, 0, stream.get()>>>(
        data, num_rows, block_size, references.data(), widths.data(), word_offsets.data());
    }
  }
  CUDF_CUDA_TRY(
    cudaMemsetAsync(word_offsets.data() + num_blocks, 0, sizeof(int64_t), stream.get()));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         word_offsets.begin(),
                         word_offsets.end(),
                         word_offsets.begin());
  auto const num_words = word_offsets.element(num_blocks, stream) + 1;

  fb::header h{};
  h.num_rows            = num_rows;
  h.num_blocks          = num_blocks;
  h.block_size          = block_size;
  h.value_bytes         = value_bytes;
  h.references_offset   = align8(sizeof(fb::header));
  h.word_offsets_offset = h.references_offset + num_blocks * 8;
  h.widths_offset       = h.word_offsets_offset + num_blocks * 8;
  h.words_offset        = align8(h.widths_offset + num_blocks);
  auto const total      = h.words_offset + num_words * 8;

  rmm::device_buffer encoded(total, stream, mr);
  auto* out = static_cast<uint8_t*>(encoded.data());
  CUDF_CUDA_TRY(cudaMemsetAsync(out, 0, total, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(out, &h, sizeof(h), cudaMemcpyDefault, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    out + h.references_offset, references.data(), num_blocks * 8, cudaMemcpyDefault, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(out + h.word_offsets_offset,
                                word_offsets.data(),
                                num_blocks * 8,
                                cudaMemcpyDefault,
                                stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    out + h.widths_offset, widths.data(), num_blocks, cudaMemcpyDefault, stream.get()));

  if (num_rows > 0) {
    auto const grid =
      static_cast<unsigned>(cudf::util::div_rounding_up_safe<int64_t>(num_rows, encode_threads));
    auto* words = reinterpret_cast<unsigned long long*>(out + h.words_offset);
    if (value_bytes == 4) {
      pack_kernel<int32_t><<<grid, encode_threads, 0, stream.get()>>>(
        data, num_rows, block_size, references.data(), widths.data(), word_offsets.data(), words);
    } else {
      pack_kernel<int64_t><<<grid, encode_threads, 0, stream.get()>>>(
        data, num_rows, block_size, references.data(), widths.data(), word_offsets.data(), words);
    }
  }
  CUDF_CHECK_CUDA(stream.get());
  // The header is copied from host stack memory.
  stream.sync();
  return encoded;
}

}  // namespace codecs
}  // namespace lto
}  // namespace cudf
