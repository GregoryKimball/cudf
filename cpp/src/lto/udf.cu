/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "copying/packed_chunks.hpp"
#include "lto/codec.hpp"
#include "lto/jit/reduce_args.h"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lto/udf.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda.h>
#include <cuda/iterator>
#include <cuda_runtime_api.h>
#include <thrust/for_each.h>

#include <cudf_fragments.hpp>
#include <jit/cache.hpp>

#include <algorithm>
#include <array>
#include <mutex>
#include <string>
#include <unordered_map>

namespace cudf::experimental::lto {
namespace {

using cudf::io::detail::codec_exec_result;
using cudf::io::detail::codec_status;

constexpr unsigned int codec_block_size = 256;
constexpr std::size_t max_codec_grid    = std::size_t{1} << 20;

struct codec_registry {
  std::mutex mutex;
  std::unordered_map<uint32_t, codec> codecs;
};

codec_registry& registry()
{
  static codec_registry instance;
  return instance;
}

codec const& registered_codec(uint32_t id)
{
  auto& reg = registry();
  std::lock_guard const lock{reg.mutex};
  auto const it = reg.codecs.find(id);
  CUDF_EXPECTS(it != reg.codecs.end(),
               "LTO codec " + std::to_string(id) + " is not registered",
               std::invalid_argument);
  return it->second;
}

rtcx::binary_type to_rtcx(lto_binary_type type)
{
  return type == lto_binary_type::FATBIN ? rtcx::binary_type::FATBIN : rtcx::binary_type::LTO_IR;
}

std::span<uint8_t const> kernel_fragment(std::size_t file_index)
{
  auto const range = cudf_fragments::file_ranges[file_index];
  return cudf_fragments::files.subspan(range[0], range[1]);
}

/// Links a precompiled kernel fragment with one caller fragment; the JIT cache keys on both.
kernel link(char const* name, std::size_t fragment_index, udf const& user)
{
  rtcx::memory_fragment const fragments[] = {
    {.data = kernel_fragment(fragment_index), .type = rtcx::binary_type::FATBIN, .name = name},
    {.data = user.binary, .type = to_rtcx(user.type), .name = nullptr}};
  return get_lto_linked_kernel(name, {}, fragments);
}

template <typename Function>
Function driver_entry_point(char const* symbol)
{
  void* function = nullptr;
  cudaDriverEntryPointQueryResult query{};
  CUDF_CUDA_TRY(
    cudaGetDriverEntryPointByVersion(symbol, &function, 12000, cudaEnableDefault, &query));
  CUDF_EXPECTS(query == cudaDriverEntryPointSuccess && function != nullptr,
               std::string{"Missing CUDA driver entry point "} + symbol);
  return reinterpret_cast<Function>(function);
}

/// Opts the kernel into `bytes` of dynamic shared memory and returns its resident blocks per SM.
int prepare_dynamic_shared_memory(kernel const& linked, std::size_t bytes, unsigned int block_size)
{
  using get_function_fn           = CUresult (*)(CUfunction*, CUkernel);
  using set_attribute_fn          = CUresult (*)(CUfunction, CUfunction_attribute, int);
  using occupancy_fn              = CUresult (*)(int*, CUfunction, int, std::size_t);
  static auto const get_function  = driver_entry_point<get_function_fn>("cuKernelGetFunction");
  static auto const set_attribute = driver_entry_point<set_attribute_fn>("cuFuncSetAttribute");
  static auto const occupancy =
    driver_entry_point<occupancy_fn>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
  CUfunction function = nullptr;
  CUDF_EXPECTS(get_function(&function, linked.get().get()) == CUDA_SUCCESS,
               "Failed to resolve the LTO kernel in the current context");
  CUDF_EXPECTS(set_attribute(function,
                             CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             static_cast<int>(bytes)) == CUDA_SUCCESS,
               "Failed to set the LTO kernel's dynamic shared memory size");
  int blocks = 0;
  CUDF_EXPECTS(occupancy(&blocks, function, static_cast<int>(block_size), bytes) == CUDA_SUCCESS,
               "Failed to query the LTO kernel's occupancy");
  return blocks;
}

int device_attribute(cudaDeviceAttr attribute)
{
  int device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  int value = 0;
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&value, attribute, device));
  return value;
}

rmm::device_uvector<cudf_lto_codec_job> make_jobs(
  uint32_t element_bytes,
  device_span<device_span<uint8_t const> const> inputs,
  device_span<device_span<uint8_t> const> outputs,
  cuda::stream_ref stream)
{
  auto const temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<cudf_lto_codec_job> jobs(inputs.size(), stream, temp_mr);
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<std::size_t>{0},
    inputs.size(),
    [inputs  = inputs.data(),
     outputs = outputs.data(),
     jobs    = jobs.data(),
     element_bytes] __device__(std::size_t i) {
      jobs[i] = cudf_lto_codec_job{
        inputs[i].data(), outputs[i].data(), inputs[i].size(), outputs[i].size(), element_bytes, 0};
    });
  return jobs;
}

void launch_codec(kernel const& linked,
                  rmm::device_uvector<cudf_lto_codec_job> const& jobs,
                  void* per_job_output,
                  cuda::stream_ref stream)
{
  auto const* job_data  = jobs.data();
  cudf_lto_u64 num_jobs = jobs.size();
  void* params[]        = {&job_data, &num_jobs, &per_job_output};
  auto const grid       = static_cast<unsigned int>(std::min(jobs.size(), max_codec_grid));
  linked.launch({grid}, {codec_block_size}, 0, stream, params);
}

}  // namespace

void register_codec(uint32_t id, codec codec)
{
  CUDF_EXPECTS(id != 0, "LTO codec id 0 is reserved", std::invalid_argument);
  CUDF_EXPECTS(!codec.binary.empty() && codec.max_encoded_bytes,
               "An LTO codec needs a fragment and an encoded-size bound",
               std::invalid_argument);
  auto& reg = registry();
  std::lock_guard const lock{reg.mutex};
  CUDF_EXPECTS(reg.codecs.emplace(id, std::move(codec)).second,
               "LTO codec " + std::to_string(id) + " is already registered",
               std::invalid_argument);
}

bool is_codec_registered(uint32_t id)
{
  auto& reg = registry();
  std::lock_guard const lock{reg.mutex};
  return reg.codecs.contains(id);
}

namespace detail {

std::size_t max_encoded_bytes(uint32_t codec_id, std::size_t chunk_bytes)
{
  return registered_codec(codec_id).max_encoded_bytes(chunk_bytes);
}

void encode_chunks(uint32_t codec_id,
                   uint32_t element_bytes,
                   device_span<device_span<uint8_t const> const> inputs,
                   device_span<device_span<uint8_t> const> outputs,
                   device_span<codec_exec_result> results,
                   cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  if (inputs.empty()) { return; }
  auto const& codec = registered_codec(codec_id);
  auto const linked =
    link("cudf/lto/encode", cudf_fragments::lto_encode_kernel, {codec.binary, codec.type});
  auto const jobs    = make_jobs(element_bytes, inputs, outputs, stream);
  auto const temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<cudf_lto_u64> written(inputs.size(), stream, temp_mr);
  launch_codec(linked, jobs, written.data(), stream);
  thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                     cuda::counting_iterator<std::size_t>{0},
                     inputs.size(),
                     [written = written.data(),
                      outputs = outputs.data(),
                      results = results.data()] __device__(std::size_t i) {
                       auto const bytes = written[i];
                       results[i]       = bytes > 0 && bytes <= outputs[i].size()
                                            ? codec_exec_result{bytes, codec_status::SUCCESS}
                                            : codec_exec_result{0, codec_status::FAILURE};
                     });
}

void decode_chunks(uint32_t codec_id,
                   uint32_t element_bytes,
                   device_span<device_span<uint8_t const> const> inputs,
                   device_span<device_span<uint8_t> const> outputs,
                   device_span<codec_exec_result> results,
                   cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  if (inputs.empty()) { return; }
  auto const& codec = registered_codec(codec_id);
  auto const linked =
    link("cudf/lto/decode", cudf_fragments::lto_decode_kernel, {codec.binary, codec.type});
  auto const jobs    = make_jobs(element_bytes, inputs, outputs, stream);
  auto const temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<int> statuses(inputs.size(), stream, temp_mr);
  launch_codec(linked, jobs, statuses.data(), stream);
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<std::size_t>{0},
    inputs.size(),
    [statuses = statuses.data(), outputs = outputs.data(), results = results.data()] __device__(
      std::size_t i) {
      results[i] = statuses[i] == 0 ? codec_exec_result{outputs[i].size(), codec_status::SUCCESS}
                                    : codec_exec_result{0, codec_status::FAILURE};
    });
}

}  // namespace detail

struct packed_source::impl {
  std::size_t num_columns = 0;
  uint64_t num_rows       = 0;
  std::size_t tile_bytes  = 0;
  std::array<std::size_t, CUDF_LTO_MAX_COLUMNS> max_chunk_bytes{};
  rmm::device_uvector<cudf_lto_tile_desc> tiles;
  rmm::device_uvector<cudf_lto_chunk_ref> chunks;
};

packed_source::packed_source(std::unique_ptr<impl>&& implementation)
  : _impl(std::move(implementation))
{
}
packed_source::packed_source(packed_source&&) noexcept            = default;
packed_source& packed_source::operator=(packed_source&&) noexcept = default;
packed_source::~packed_source()                                   = default;

std::size_t packed_source::num_columns() const { return _impl->num_columns; }
std::size_t packed_source::num_tiles() const { return _impl->tiles.size(); }
uint64_t packed_source::num_rows() const { return _impl->num_rows; }
std::size_t packed_source::tile_bytes() const { return _impl->tile_bytes; }

packed_source make_packed_source(std::span<std::vector<packed_data_view> const> partitions,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    !partitions.empty(), "An LTO source needs at least one partition", std::invalid_argument);
  auto const num_columns = partitions.front().size();
  CUDF_EXPECTS(num_columns > 0 && num_columns <= CUDF_LTO_MAX_COLUMNS,
               "An LTO source needs between 1 and CUDF_LTO_MAX_COLUMNS columns",
               std::invalid_argument);

  std::vector<cudf_lto_tile_desc> tiles;
  std::vector<cudf_lto_chunk_ref> chunks;
  std::vector<std::size_t> max_bytes(num_columns, 0);
  uint64_t row_base = 0;
  for (auto const& partition : partitions) {
    CUDF_EXPECTS(partition.size() == num_columns,
                 "Every partition of an LTO source needs the same columns",
                 std::invalid_argument);
    std::vector<cudf::experimental::detail::packed_column_chunks> directories;
    directories.reserve(num_columns);
    for (auto const& view : partition) {
      auto directory = cudf::experimental::detail::read_packed_column_chunks(view, stream);
      CUDF_EXPECTS(directory.compression == pack_compression::lto,
                   "Every column of an LTO source must use an LTO codec",
                   std::invalid_argument);
      directories.push_back(std::move(directory));
    }
    auto const& first        = directories.front();
    auto const rows_per_tile = first.chunk_bytes / size_of(first.type);
    for (auto const& directory : directories) {
      CUDF_EXPECTS(directory.num_rows == first.num_rows &&
                     directory.offsets.size() == first.offsets.size() &&
                     directory.chunk_bytes / size_of(directory.type) == rows_per_tile,
                   "Columns of an LTO partition need the same rows per chunk",
                   std::invalid_argument);
    }
    auto const num_rows = static_cast<uint64_t>(first.num_rows);
    for (std::size_t k = 0; k < first.offsets.size(); ++k) {
      auto const begin = k * rows_per_tile;
      if (begin >= num_rows) { break; }
      tiles.push_back(
        {row_base + begin,
         static_cast<cudf_lto_u32>(std::min<uint64_t>(rows_per_tile, num_rows - begin)),
         0});
      for (std::size_t c = 0; c < num_columns; ++c) {
        auto const* data = partition[c].payload.data() + directories[c].offsets[k];
        CUDF_EXPECTS(reinterpret_cast<std::uintptr_t>(data) % CUDF_LTO_CHUNK_ALIGNMENT == 0,
                     "LTO chunks must be aligned to CUDF_LTO_CHUNK_ALIGNMENT");
        chunks.push_back({data, directories[c].sizes[k]});
        max_bytes[c] = std::max<std::size_t>(max_bytes[c], directories[c].sizes[k]);
      }
    }
    row_base += num_rows;
  }

  auto result = std::make_unique<packed_source::impl>(
    packed_source::impl{num_columns,
                        row_base,
                        0,
                        {},
                        cudf::detail::make_device_uvector_async(tiles, stream, mr),
                        cudf::detail::make_device_uvector_async(chunks, stream, mr)});
  for (std::size_t c = 0; c < num_columns; ++c) {
    result->max_chunk_bytes[c] = max_bytes[c];
    result->tile_bytes +=
      cudf::util::round_up_safe(max_bytes[c], std::size_t{CUDF_LTO_CHUNK_ALIGNMENT});
  }
  stream.sync();
  return packed_source{std::move(result)};
}

namespace {

/// Shared-memory placement of a tile's staged columns; lazy columns take no space.
struct tile_layout {
  std::array<cudf_lto_u32, CUDF_LTO_MAX_COLUMNS> offsets{};
  std::size_t bytes      = 0;
  cudf_lto_u32 lazy_mask = 0;
};

tile_layout make_layout(packed_source::impl const& impl, std::span<bool const> lazy_columns)
{
  CUDF_EXPECTS(lazy_columns.empty() || lazy_columns.size() == impl.num_columns,
               "Lazy column flags must cover every column of the LTO source",
               std::invalid_argument);
  tile_layout layout;
  for (std::size_t c = 0; c < impl.num_columns; ++c) {
    if (!lazy_columns.empty() && lazy_columns[c]) {
      layout.lazy_mask |= cudf_lto_u32{1} << c;
      continue;
    }
    layout.offsets[c] = static_cast<cudf_lto_u32>(layout.bytes);
    layout.bytes +=
      cudf::util::round_up_safe(impl.max_chunk_bytes[c], std::size_t{CUDF_LTO_CHUNK_ALIGNMENT});
  }
  return layout;
}

/// Launches a tile kernel over `impl` that produces `num_groups` states.
rmm::device_buffer launch_tiles(kernel const& linked,
                                packed_source::impl const& impl,
                                tile_layout const& layout,
                                std::size_t smem_bytes,
                                std::size_t num_groups,
                                std::size_t state_bytes,
                                void const* user_data,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(smem_bytes <= static_cast<std::size_t>(
                               device_attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)),
               "An LTO tile and its states do not fit in shared memory; pack with smaller chunks",
               std::invalid_argument);
  auto const blocks_per_sm =
    prepare_dynamic_shared_memory(linked, smem_bytes, CUDF_LTO_REDUCE_BLOCK_SIZE);
  CUDF_EXPECTS(blocks_per_sm > 0, "The LTO kernel cannot be resident");
  auto const grid = static_cast<unsigned int>(std::max<std::size_t>(
    1,
    std::min<std::size_t>(
      impl.tiles.size(),
      static_cast<std::size_t>(blocks_per_sm) * device_attribute(cudaDevAttrMultiProcessorCount))));

  auto const temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_buffer partials(grid * num_groups * state_bytes, stream, temp_mr);
  rmm::device_buffer counter(sizeof(unsigned int), stream, temp_mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(counter.data(), 0, sizeof(unsigned int), stream.get()));
  rmm::device_buffer result(num_groups * state_bytes, stream, mr);

  cudf_lto_reduce_args args{};
  args.tiles       = impl.tiles.data();
  args.chunks      = impl.chunks.data();
  args.num_tiles   = impl.tiles.size();
  args.num_columns = static_cast<cudf_lto_u32>(impl.num_columns);
  args.state_bytes = static_cast<cudf_lto_u32>(state_bytes);
  std::copy(layout.offsets.begin(), layout.offsets.end(), args.column_offsets);
  args.tile_bytes   = static_cast<cudf_lto_u32>(layout.bytes);
  args.num_groups   = static_cast<cudf_lto_u32>(num_groups);
  args.lazy_columns = layout.lazy_mask;
  args.user_data    = user_data;
  args.partials     = static_cast<unsigned char*>(partials.data());
  args.counter      = static_cast<unsigned int*>(counter.data());
  args.result       = static_cast<unsigned char*>(result.data());
  void* params[]    = {&args};
  linked.launch(
    {grid}, {CUDF_LTO_REDUCE_BLOCK_SIZE}, static_cast<uint32_t>(smem_bytes), stream, params);
  return result;
}

void check_state_bytes(std::size_t state_bytes)
{
  CUDF_EXPECTS(state_bytes > 0 && state_bytes <= CUDF_LTO_MAX_STATE_BYTES && state_bytes % 8 == 0,
               "LTO state must be a multiple of 8 bytes, at most CUDF_LTO_MAX_STATE_BYTES",
               std::invalid_argument);
}

}  // namespace

rmm::device_buffer reduce(packed_source const& source,
                          udf row_program,
                          std::size_t state_bytes,
                          void const* user_data,
                          std::span<bool const> lazy_columns,
                          cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(source._impl != nullptr, "Cannot reduce a moved-from LTO source");
  check_state_bytes(state_bytes);
  auto const& impl  = *source._impl;
  auto const linked = link("cudf/lto/reduce", cudf_fragments::lto_reduce_kernel, row_program);
  auto const layout = make_layout(impl, lazy_columns);
  auto const smem_bytes =
    std::max(layout.bytes, std::size_t{CUDF_LTO_REDUCE_BLOCK_SIZE} * state_bytes);
  return launch_tiles(linked, impl, layout, smem_bytes, 1, state_bytes, user_data, stream, mr);
}

rmm::device_buffer groupby(packed_source const& source,
                           udf row_program,
                           std::size_t num_groups,
                           std::size_t state_bytes,
                           void const* user_data,
                           std::span<bool const> lazy_columns,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(source._impl != nullptr, "Cannot group a moved-from LTO source");
  CUDF_EXPECTS(num_groups > 0, "An LTO groupby needs at least one group", std::invalid_argument);
  check_state_bytes(state_bytes);
  auto const& impl  = *source._impl;
  auto const linked = link("cudf/lto/groupby", cudf_fragments::lto_groupby_kernel, row_program);
  auto const warps  = std::size_t{CUDF_LTO_REDUCE_BLOCK_SIZE / 32};
  auto const layout = make_layout(impl, lazy_columns);
  auto const smem_bytes =
    layout.bytes + (warps * num_groups + CUDF_LTO_REDUCE_BLOCK_SIZE) * state_bytes;
  return launch_tiles(
    linked, impl, layout, smem_bytes, num_groups, state_bytes, user_data, stream, mr);
}

}  // namespace cudf::experimental::lto
