/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "feather_common.hpp"
#include "io/comp/nvcomp_adapter.hpp"

#include <cudf/column/column.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/experimental/feather.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace cudf::io::feather::detail {
namespace {

namespace nvcomp = cudf::io::detail::nvcomp;
using cudf::io::detail::codec_exec_result;
using cudf::io::detail::codec_status;

arrow_layout layout_of(column_view const& col)
{
  switch (col.type().id()) {
    case type_id::BOOL8: return arrow_layout::BOOLEAN;
    case type_id::STRING:
      return col.num_children() > 0 &&
                 strings_column_view(col).offsets().type().id() == type_id::INT64
               ? arrow_layout::LARGE_UTF8
               : arrow_layout::UTF8;
    default: return arrow_layout::FIXED_WIDTH;
  }
}

struct source_buffer {
  uint8_t const* data;
  std::size_t size;
};

/**
 * @brief Device data of one record batch, in IPC buffer order.
 */
struct batch_buffers {
  std::vector<flatbuf::FieldNode> nodes;
  std::vector<source_buffer> buffers;
  std::vector<std::unique_ptr<column>> owned_columns;
  std::vector<cuda::device_buffer<bitmask_type>> owned_masks;
};

batch_buffers collect_buffers(table_view const& batch,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  batch_buffers out;
  for (auto const& input : batch) {
    // Sliced columns are copied so that buffers start at row 0 and offsets start at 0
    auto col = input;
    if (col.offset() != 0) {
      col = out.owned_columns.emplace_back(std::make_unique<column>(input, stream, mr))->view();
    }
    auto const num_rows   = static_cast<std::size_t>(col.size());
    auto const null_count = col.null_count();
    out.nodes.emplace_back(col.size(), null_count);

    auto const bitmap_size = (num_rows + 7) / 8;
    out.buffers.push_back(
      {reinterpret_cast<uint8_t const*>(col.null_mask()), null_count > 0 ? bitmap_size : 0});
    if (num_rows == 0) {
      for (int32_t i = 1; i < num_buffers(layout_of(col)); ++i) {
        out.buffers.push_back({nullptr, 0});
      }
      continue;
    }

    switch (layout_of(col)) {
      case arrow_layout::BOOLEAN: {
        auto mask = cudf::detail::bools_to_mask(col, stream, mr).first;
        out.buffers.push_back({reinterpret_cast<uint8_t const*>(mask->data()), bitmap_size});
        out.owned_masks.push_back(std::move(*mask));
        break;
      }
      case arrow_layout::FIXED_WIDTH:
        out.buffers.push_back({col.head<uint8_t>(), num_rows * size_of(col.type())});
        break;
      default: {
        auto const strings = strings_column_view(col);
        auto const offsets = strings.offsets();
        out.buffers.push_back({offsets.head<uint8_t>(), (num_rows + 1) * size_of(offsets.type())});
        out.buffers.push_back({reinterpret_cast<uint8_t const*>(strings.chars_begin(stream)),
                               static_cast<std::size_t>(strings.chars_size(stream))});
        break;
      }
    }
  }
  return out;
}

/**
 * @brief Compresses chunks of `buffers` with nvCOMP into `scratch`.
 *
 * @return The compressed size of each chunk; chunk `c` is at `scratch + c * slot_size`
 */
std::vector<std::size_t> compress_chunks(std::vector<source_buffer> const& buffers,
                                         std::vector<compression_chunk> const& chunks,
                                         nvcomp::compression_type type,
                                         rmm::device_buffer& scratch,
                                         std::size_t& slot_size,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
{
  auto const codec_name = type == nvcomp::compression_type::LZ4 ? "LZ4" : "ZSTD";
  if (auto const reason = nvcomp::is_compression_disabled(type)) {
    CUDF_FAIL(std::string{"nvCOMP "} + codec_name + " compression is unavailable: " + *reason);
  }
  auto const temp_mr   = mr.get_temporary_mr();
  auto const alignment = nvcomp::compress_required_alignment(type);
  slot_size =
    round_up(nvcomp::compress_max_output_chunk_size(type, compression_chunk_size), alignment);
  scratch = rmm::device_buffer(slot_size * chunks.size(), stream, temp_mr);
  if (chunks.empty()) { return {}; }

  std::vector<device_span<uint8_t const>> inputs;
  std::vector<device_span<uint8_t>> outputs;
  for (std::size_t c = 0; c < chunks.size(); ++c) {
    auto const src = buffers[chunks[c].buffer].data + chunks[c].offset;
    CUDF_EXPECTS(reinterpret_cast<std::uintptr_t>(src) % alignment == 0,
                 "Feather writer input buffer does not meet nvCOMP alignment requirements");
    inputs.emplace_back(src, chunks[c].size);
    outputs.emplace_back(static_cast<uint8_t*>(scratch.data()) + c * slot_size, slot_size);
  }
  auto const d_inputs  = cudf::detail::make_device_uvector_async(inputs, stream, temp_mr);
  auto const d_outputs = cudf::detail::make_device_uvector_async(outputs, stream, temp_mr);
  // nvCOMP leaves results that are already SKIPPED untouched, so they must be initialized
  std::vector<codec_exec_result> const initial_results(chunks.size(), {0, codec_status::FAILURE});
  auto d_results = cudf::detail::make_device_uvector_async(initial_results, stream, temp_mr);
  nvcomp::batched_compress(type, d_inputs, d_outputs, d_results, stream, mr);

  auto const results = cudf::detail::make_host_vector(d_results, stream);
  std::vector<std::size_t> sizes;
  for (auto const& result : results) {
    CUDF_EXPECTS(result.status == codec_status::SUCCESS,
                 std::string{"nvCOMP failed to compress a Feather "} + codec_name + " chunk");
    sizes.push_back(result.bytes_written);
  }
  return sizes;
}

void batched_copy(std::vector<void const*> const& srcs,
                  std::vector<void*> const& dsts,
                  std::vector<std::size_t> const& sizes,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr)
{
  if (srcs.empty()) { return; }
  auto const temp_mr = mr.get_temporary_mr();
  auto const d_srcs  = cudf::detail::make_device_uvector_async(srcs, stream, temp_mr);
  auto const d_dsts  = cudf::detail::make_device_uvector_async(dsts, stream, temp_mr);
  auto const d_sizes = cudf::detail::make_device_uvector_async(sizes, stream, temp_mr);
  cudf::detail::batched_memcpy_async(
    d_srcs.begin(), d_dsts.begin(), d_sizes.begin(), srcs.size(), stream, mr);
}

/**
 * @brief Encoded record batch body.
 */
struct encoded_body {
  body_plan plan;
  rmm::device_buffer data;
};

encoded_body encode_body(std::vector<source_buffer> const& buffers,
                         compression_type compression,
                         cuda::stream_ref stream,
                         cudf::memory_resources mr)
{
  std::vector<std::size_t> sizes;
  std::transform(buffers.begin(), buffers.end(), std::back_inserter(sizes), [](auto const& buf) {
    return buf.size;
  });

  rmm::device_buffer scratch;
  std::size_t slot_size = 0;
  body_plan plan;
  if (compression == compression_type::NONE) {
    plan = plan_uncompressed_body(sizes);
  } else {
    auto const is_lz4 = compression == compression_type::LZ4;
    auto const chunks = split_into_chunks(sizes);
    auto const compressed_sizes =
      compress_chunks(buffers,
                      chunks,
                      is_lz4 ? nvcomp::compression_type::LZ4 : nvcomp::compression_type::ZSTD,
                      scratch,
                      slot_size,
                      stream,
                      mr);
    plan = plan_compressed_body(
      sizes,
      chunks,
      compressed_sizes,
      is_lz4 ? flatbuf::CompressionType_LZ4_FRAME : flatbuf::CompressionType_ZSTD);
  }

  rmm::device_buffer body(plan.size, stream, mr.get_temporary_mr());
  CUDF_CUDA_TRY(cudaMemsetAsync(body.data(), 0, plan.size, stream.get()));
  std::vector<void const*> srcs;
  std::vector<void*> dsts;
  std::vector<std::size_t> copy_sizes;
  for (auto const& copy : plan.copies) {
    srcs.push_back(copy.from_chunk
                     ? static_cast<uint8_t const*>(scratch.data()) + copy.index * slot_size
                     : buffers[copy.index].data + copy.src_offset);
    dsts.push_back(static_cast<uint8_t*>(body.data()) + copy.dst_offset);
    copy_sizes.push_back(copy.size);
  }
  batched_copy(srcs, dsts, copy_sizes, stream, mr);

  // Frame and block headers depend on the compressed sizes; write them into the body on the device
  std::vector<uint8_t> patch_bytes;
  for (auto const& patch : plan.patches) {
    auto const value = reinterpret_cast<uint8_t const*>(&patch.value);
    patch_bytes.insert(patch_bytes.end(), value, value + patch.width);
  }
  auto const d_patch_bytes =
    cudf::detail::make_device_uvector_async(patch_bytes, stream, mr.get_temporary_mr());
  srcs.clear();
  dsts.clear();
  copy_sizes.clear();
  std::size_t patch_offset = 0;
  for (auto const& patch : plan.patches) {
    srcs.push_back(d_patch_bytes.data() + patch_offset);
    dsts.push_back(static_cast<uint8_t*>(body.data()) + patch.offset);
    copy_sizes.push_back(patch.width);
    patch_offset += patch.width;
  }
  batched_copy(srcs, dsts, copy_sizes, stream, mr);
  plan.patches.clear();
  stream.sync();
  return {std::move(plan), std::move(body)};
}

void write_body(data_sink* sink, encoded_body const& body, cuda::stream_ref stream)
{
  auto const size = body.data.size();
  if (size == 0) { return; }
  if (sink->is_device_write_preferred(size)) {
    sink->device_write(body.data.data(), size, stream);
    return;
  }
  auto const host = cudf::detail::make_host_vector(
    device_span<uint8_t const>{static_cast<uint8_t const*>(body.data.data()), size}, stream);
  sink->host_write(host.data(), size);
}

}  // namespace

void write_feather(data_sink* sink,
                   table_view const& table,
                   std::vector<std::string> const& column_names,
                   compression_type compression,
                   size_type max_rows_per_batch,
                   cuda::stream_ref stream)
{
  CUDF_EXPECTS(
    column_names.empty() || static_cast<size_type>(column_names.size()) == table.num_columns(),
    "Number of Feather column names must match the number of columns");
  std::vector<field_info> fields;
  for (size_type c = 0; c < table.num_columns(); ++c) {
    auto const& col = table.column(c);
    fields.push_back(
      {column_names.empty() ? std::to_string(c) : column_names[c], col.type(), layout_of(col)});
  }
  auto const codec = compression == compression_type::NONE
                       ? std::optional<flatbuf::CompressionType>{}
                       : std::optional<flatbuf::CompressionType>{
                           compression == compression_type::LZ4 ? flatbuf::CompressionType_LZ4_FRAME
                                                                : flatbuf::CompressionType_ZSTD};

  auto const mr           = cudf::memory_resources{cudf::get_current_device_resource_ref()};
  std::size_t file_offset = 0;
  auto const write_host   = [&](void const* data, std::size_t size) {
    sink->host_write(data, size);
    file_offset += size;
  };

  static_assert(file_magic_size <= ipc_alignment);
  std::array<uint8_t, ipc_alignment> leading_magic{};
  std::memcpy(leading_magic.data(), file_magic, file_magic_size);
  write_host(leading_magic.data(), leading_magic.size());

  auto const schema_message = make_schema_message(fields);
  write_host(schema_message.data(), schema_message.size());

  std::vector<size_type> splits;
  for (int64_t row = max_rows_per_batch; row < table.num_rows(); row += max_rows_per_batch) {
    splits.push_back(static_cast<size_type>(row));
  }
  auto const batches =
    splits.empty() ? std::vector<table_view>{table} : cudf::detail::split(table, splits, stream);

  std::vector<flatbuf::Block> blocks;
  for (auto const& batch : batches) {
    auto const buffers = collect_buffers(batch, stream, mr);
    auto const body    = encode_body(buffers.buffers, compression, stream, mr);
    auto const message = make_record_batch_message(batch.num_rows(),
                                                   buffers.nodes,
                                                   body.plan.buffers,
                                                   codec,
                                                   static_cast<int64_t>(body.plan.size));
    blocks.emplace_back(static_cast<int64_t>(file_offset),
                        static_cast<int32_t>(message.size()),
                        static_cast<int64_t>(body.plan.size));
    write_host(message.data(), message.size());
    write_body(sink, body, stream);
    file_offset += body.plan.size;
  }

  std::array<uint32_t, 2> const end_of_stream{continuation_marker, 0};
  write_host(end_of_stream.data(), sizeof(end_of_stream));

  auto const footer = make_footer(fields, blocks);
  write_host(footer.data(), footer.size());
  auto const footer_size = static_cast<int32_t>(footer.size());
  write_host(&footer_size, sizeof(footer_size));
  write_host(file_magic, file_magic_size);

  sink->flush();
}

}  // namespace cudf::io::feather::detail
