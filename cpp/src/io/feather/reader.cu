/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "feather_common.hpp"
#include "io/comp/nvcomp_adapter.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/experimental/feather.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudf::io::feather::detail {
namespace {

namespace nvcomp = cudf::io::detail::nvcomp;
using cudf::io::detail::codec_exec_result;
using cudf::io::detail::codec_status;

struct copy_task {
  uint8_t const* src;
  uint8_t* dst;
  std::size_t size;
};

struct decompress_task {
  uint8_t const* src;
  std::size_t src_size;
  uint8_t* dst;
  std::size_t dst_size;
};

/**
 * @brief Device work needed to materialize all buffers read from the file.
 */
struct decode_plan {
  std::vector<copy_task> copies;
  std::vector<decompress_task> lz4;
  std::vector<decompress_task> zstd;
};

/**
 * @brief A buffer of a record batch body, visible on the host (if compressed) and on the device.
 */
struct body_buffer {
  uint8_t const* host;  ///< nullptr when the body was read directly to the device
  uint8_t const* device;
  std::size_t size;
};

/**
 * @brief Returns the number of bytes a body buffer decodes to.
 */
std::size_t decoded_size(body_buffer const& buf, std::optional<flatbuf::CompressionType> codec)
{
  if (not codec.has_value() or buf.size == 0) { return buf.size; }
  CUDF_EXPECTS(buf.size >= compressed_length_prefix_size,
               "Compressed Feather buffer is missing its length prefix");
  auto const length = read_le<int64_t>(buf.host);
  if (length == uncompressed_buffer_marker) { return buf.size - compressed_length_prefix_size; }
  CUDF_EXPECTS(length >= 0, "Invalid uncompressed length in Feather buffer");
  return static_cast<std::size_t>(length);
}

/**
 * @brief Adds the work needed to decode `buf` into `dst` to the plan.
 */
void plan_buffer(body_buffer const& buf,
                 std::optional<flatbuf::CompressionType> codec,
                 uint8_t* dst,
                 std::size_t dst_size,
                 decode_plan& plan)
{
  if (dst_size == 0) { return; }
  if (not codec.has_value()) {
    plan.copies.push_back({buf.device, dst, dst_size});
    return;
  }
  auto const host   = buf.host + compressed_length_prefix_size;
  auto const device = buf.device + compressed_length_prefix_size;
  auto const size   = buf.size - compressed_length_prefix_size;
  if (read_le<int64_t>(buf.host) == uncompressed_buffer_marker) {
    plan.copies.push_back({device, dst, dst_size});
    return;
  }
  auto const is_lz4 = *codec == flatbuf::CompressionType_LZ4_FRAME;
  auto const chunks =
    is_lz4 ? parse_lz4_frame({host, size}, dst_size) : parse_zstd_frames({host, size}, dst_size);
  auto& decompress = is_lz4 ? plan.lz4 : plan.zstd;
  for (auto const& chunk : chunks) {
    if (chunk.compressed) {
      decompress.push_back(
        {device + chunk.src_offset, chunk.src_size, dst + chunk.dst_offset, chunk.dst_size});
    } else {
      plan.copies.push_back({device + chunk.src_offset, dst + chunk.dst_offset, chunk.src_size});
    }
  }
}

void batched_copy(std::vector<copy_task> const& tasks,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr)
{
  if (tasks.empty()) { return; }
  std::vector<void const*> srcs;
  std::vector<void*> dsts;
  std::vector<std::size_t> sizes;
  for (auto const& task : tasks) {
    srcs.push_back(task.src);
    dsts.push_back(task.dst);
    sizes.push_back(task.size);
  }
  auto const temp_mr = mr.get_temporary_mr();
  auto const d_srcs  = cudf::detail::make_device_uvector_async(srcs, stream, temp_mr);
  auto const d_dsts  = cudf::detail::make_device_uvector_async(dsts, stream, temp_mr);
  auto const d_sizes = cudf::detail::make_device_uvector_async(sizes, stream, temp_mr);
  cudf::detail::batched_memcpy_async(
    d_srcs.begin(), d_dsts.begin(), d_sizes.begin(), tasks.size(), stream, mr);
}

void batched_decompress(nvcomp::compression_type type,
                        std::vector<decompress_task> tasks,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  if (tasks.empty()) { return; }
  auto const codec_name = type == nvcomp::compression_type::LZ4 ? "LZ4" : "ZSTD";
  if (auto const reason = nvcomp::is_decompression_disabled(type)) {
    CUDF_FAIL(std::string{"nvCOMP "} + codec_name + " decompression is unavailable: " + *reason);
  }
  auto const temp_mr    = mr.get_temporary_mr();
  auto const alignment  = nvcomp::decompress_required_alignment(type);
  auto const is_aligned = [&](void const* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
  };

  // nvCOMP requires aligned inputs; LZ4 blocks follow 4-byte headers so most need restaging
  std::size_t staging_size = 0;
  for (auto const& task : tasks) {
    CUDF_EXPECTS(is_aligned(task.dst),
                 std::string{"Feather "} + codec_name +
                   " chunk output offset does not meet nvCOMP alignment requirements");
    if (not is_aligned(task.src)) {
      staging_size = cudf::detail::round_up_safe(staging_size, alignment) + task.src_size;
    }
  }
  rmm::device_buffer staging(staging_size, stream, temp_mr);
  std::vector<copy_task> restage;
  std::size_t staging_offset = 0;
  for (auto& task : tasks) {
    if (is_aligned(task.src)) { continue; }
    staging_offset = cudf::detail::round_up_safe(staging_offset, alignment);
    auto const dst = static_cast<uint8_t*>(staging.data()) + staging_offset;
    restage.push_back({task.src, dst, task.src_size});
    task.src = dst;
    staging_offset += task.src_size;
  }
  batched_copy(restage, stream, mr);

  std::vector<device_span<uint8_t const>> inputs;
  std::vector<device_span<uint8_t>> outputs;
  std::size_t max_chunk_size = 0;
  std::size_t total_size     = 0;
  for (auto const& task : tasks) {
    inputs.emplace_back(task.src, task.src_size);
    outputs.emplace_back(task.dst, task.dst_size);
    max_chunk_size = std::max(max_chunk_size, task.dst_size);
    total_size += task.dst_size;
  }
  auto const d_inputs  = cudf::detail::make_device_uvector_async(inputs, stream, temp_mr);
  auto const d_outputs = cudf::detail::make_device_uvector_async(outputs, stream, temp_mr);
  // nvCOMP leaves results that are already SKIPPED untouched, so they must be initialized
  std::vector<codec_exec_result> const initial_results(tasks.size(), {0, codec_status::FAILURE});
  auto d_results = cudf::detail::make_device_uvector_async(initial_results, stream, temp_mr);

  nvcomp::batched_decompress(
    type, d_inputs, d_outputs, d_results, max_chunk_size, total_size, stream, mr);

  auto const results = cudf::detail::make_host_vector(d_results, stream);
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    CUDF_EXPECTS(results[i].status == codec_status::SUCCESS,
                 std::string{"nvCOMP failed to decompress a Feather "} + codec_name + " chunk");
    CUDF_EXPECTS(results[i].bytes_written == tasks[i].dst_size,
                 std::string{"Feather "} + codec_name +
                   " chunk decompressed to an unexpected size; this layout is not supported");
  }
}

/**
 * @brief Decoded device buffers of one column in one record batch.
 */
struct batch_column {
  size_type num_rows;
  size_type null_count;
  std::optional<cuda::device_buffer<bitmask_type>> null_mask;
  std::optional<cuda::device_buffer<bitmask_type>> bits;  ///< BOOLEAN values
  std::optional<rmm::device_buffer> values;               ///< values, or string offsets
  std::optional<rmm::device_buffer> chars;
};

cuda::device_buffer<bitmask_type> make_mask_buffer(size_type num_rows,
                                                   std::size_t decoded_bytes,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(decoded_bytes >= static_cast<std::size_t>(cudf::detail::div_rounding_up_safe(
                                  static_cast<int64_t>(num_rows), int64_t{8})),
               "Feather bitmap buffer is too small");
  auto const words =
    std::max(bitmask_allocation_size_bytes(num_rows) / sizeof(bitmask_type),
             cudf::detail::div_rounding_up_safe(decoded_bytes, sizeof(bitmask_type)));
  return cuda::device_buffer<bitmask_type>(stream, mr, words, cuda::no_init);
}

std::unique_ptr<column> make_column(column_spec const& spec,
                                    batch_column&& buffers,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  auto const num_rows = buffers.num_rows;
  if (num_rows == 0) { return make_empty_column(spec.type); }
  auto null_mask = buffers.null_mask.has_value() ? std::move(*buffers.null_mask)
                                                 : cuda::device_buffer<bitmask_type>(
                                                     stream, mr.get_output_mr(), 0, cuda::no_init);

  switch (spec.layout) {
    case arrow_layout::FIXED_WIDTH:
      return std::make_unique<column>(
        spec.type, num_rows, std::move(*buffers.values), std::move(null_mask), buffers.null_count);
    case arrow_layout::BOOLEAN: {
      auto col = cudf::detail::mask_to_bools(buffers.bits->data(), 0, num_rows, stream, mr);
      if (buffers.null_count > 0) { col->set_null_mask(std::move(null_mask), buffers.null_count); }
      return col;
    }
    default: {
      auto const offset_type =
        data_type{spec.layout == arrow_layout::UTF8 ? type_id::INT32 : type_id::INT64};
      auto offsets = std::make_unique<column>(
        offset_type,
        num_rows + 1,
        std::move(*buffers.values),
        cuda::device_buffer<bitmask_type>(stream, mr.get_output_mr(), 0, cuda::no_init),
        0);
      return make_strings_column(num_rows,
                                 std::move(offsets),
                                 std::move(*buffers.chars),
                                 buffers.null_count,
                                 std::move(null_mask));
    }
  }
}

}  // namespace

table_with_metadata read_feather(datasource* source,
                                 std::optional<std::vector<std::string>> const& columns,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
{
  auto const file_size     = source->size();
  auto const trailer_size  = sizeof(int32_t) + file_magic_size;
  auto const leading_magic = cudf::detail::round_up_safe(file_magic_size, ipc_alignment);
  CUDF_EXPECTS(file_size >= leading_magic + trailer_size, "File too small to be a Feather file");

  auto const trailer = source->host_read(file_size - trailer_size, trailer_size);
  CUDF_EXPECTS(std::memcmp(trailer->data() + sizeof(int32_t), file_magic, file_magic_size) == 0,
               "Invalid Feather file magic; only Feather V2 (Arrow IPC file) format is supported");
  auto const footer_size = read_le<int32_t>(trailer->data());
  CUDF_EXPECTS(footer_size > 0 &&
                 static_cast<std::size_t>(footer_size) <= file_size - leading_magic - trailer_size,
               "Invalid Feather footer size");
  auto const footer_buf = source->host_read(file_size - trailer_size - footer_size, footer_size);
  flatbuffers::Verifier footer_verifier(footer_buf->data(), footer_buf->size());
  CUDF_EXPECTS(flatbuf::VerifyFooterBuffer(footer_verifier), "Corrupted Feather file footer");
  auto const footer = flatbuf::GetFooter(footer_buf->data());
  CUDF_EXPECTS(footer->dictionaries() == nullptr || footer->dictionaries()->size() == 0,
               "Feather files with dictionary batches are not supported");

  auto const specs       = select_columns(footer->schema(), columns);
  auto const num_batches = footer->recordBatches() ? footer->recordBatches()->size() : 0;

  // Batch columns are intermediate when several batches are concatenated
  auto const batch_mr =
    num_batches > 1 ? cudf::memory_resources{mr.get_temporary_mr(), mr.get_temporary_mr()} : mr;

  std::vector<std::unique_ptr<datasource::buffer>> host_bodies;
  std::vector<cudf::detail::host_vector<uint8_t>> pinned_bodies;
  std::vector<rmm::device_buffer> device_bodies;
  std::vector<std::vector<batch_column>> batches;
  decode_plan plan;
  int64_t total_rows = 0;

  for (uint32_t b = 0; b < num_batches; ++b) {
    auto const block = footer->recordBatches()->Get(b);
    CUDF_EXPECTS(block->offset() >= 0 && block->metaDataLength() > 0 && block->bodyLength() >= 0 &&
                   static_cast<std::size_t>(block->offset() + block->metaDataLength() +
                                            block->bodyLength()) <= file_size,
                 "Feather record batch block is out of bounds");

    auto const meta         = source->host_read(block->offset(), block->metaDataLength());
    std::size_t prefix_size = sizeof(int32_t);
    if (read_le<uint32_t>(meta->data()) == continuation_marker) { prefix_size += sizeof(int32_t); }
    CUDF_EXPECTS(meta->size() >= prefix_size, "Truncated Feather record batch message");
    auto const message_size = read_le<int32_t>(meta->data() + prefix_size - sizeof(int32_t));
    CUDF_EXPECTS(message_size > 0 && prefix_size + message_size <= meta->size(),
                 "Invalid Feather record batch message size");
    flatbuffers::Verifier message_verifier(meta->data() + prefix_size, message_size);
    CUDF_EXPECTS(flatbuf::VerifyMessageBuffer(message_verifier),
                 "Corrupted Feather record batch message");
    auto const message = flatbuf::GetMessage(meta->data() + prefix_size);
    auto const batch   = message->header_as_RecordBatch();
    CUDF_EXPECTS(batch != nullptr && batch->nodes() != nullptr && batch->buffers() != nullptr,
                 "Feather file block does not reference a record batch");

    auto const num_rows = batch->length();
    total_rows += num_rows;
    CUDF_EXPECTS(total_rows <= std::numeric_limits<size_type>::max(),
                 "Feather file has more rows than cudf supports",
                 std::overflow_error);

    auto const codec = batch->compression() != nullptr
                         ? std::optional<flatbuf::CompressionType>{batch->compression()->codec()}
                         : std::optional<flatbuf::CompressionType>{};
    CUDF_EXPECTS(batch->compression() == nullptr ||
                   batch->compression()->method() == flatbuf::BodyCompressionMethod_BUFFER,
                 "Unsupported Feather body compression method");

    // Compressed bodies are parsed on the host to locate nvCOMP chunks
    auto const body_offset   = static_cast<std::size_t>(block->offset() + block->metaDataLength());
    auto const body_size     = static_cast<std::size_t>(block->bodyLength());
    uint8_t const* host_body = nullptr;
    auto& device_body        = device_bodies.emplace_back(body_size, stream, mr.get_temporary_mr());
    if (source->is_device_read_preferred(body_size)) {
      source->device_read(
        body_offset, body_size, static_cast<uint8_t*>(device_body.data()), stream);
      if (codec.has_value()) {
        // Pageable device-to-host copies are several times slower than pinned ones
        host_body = pinned_bodies
                      .emplace_back(cudf::detail::make_pinned_vector_async(
                        device_span<uint8_t const>{static_cast<uint8_t const*>(device_body.data()),
                                                   body_size},
                        stream))
                      .data();
        stream.sync();
      }
    } else {
      auto& body = host_bodies.emplace_back(source->host_read(body_offset, body_size));
      host_body  = body->data();
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(device_body.data(), host_body, body_size, stream));
    }
    if (not codec.has_value()) { host_body = nullptr; }

    auto const body_ref = [&](int32_t index) {
      CUDF_EXPECTS(index < static_cast<int32_t>(batch->buffers()->size()),
                   "Feather record batch has too few buffers");
      auto const buf = batch->buffers()->Get(index);
      CUDF_EXPECTS(buf->offset() >= 0 && buf->length() >= 0 &&
                     static_cast<std::size_t>(buf->offset() + buf->length()) <= body_size,
                   "Feather buffer is out of bounds");
      return body_buffer{host_body ? host_body + buf->offset() : nullptr,
                         static_cast<uint8_t const*>(device_body.data()) + buf->offset(),
                         static_cast<std::size_t>(buf->length())};
    };

    auto& batch_columns = batches.emplace_back();
    for (auto const& spec : specs) {
      CUDF_EXPECTS(spec.node_index < static_cast<int32_t>(batch->nodes()->size()),
                   "Feather record batch has too few field nodes");
      auto const node = batch->nodes()->Get(spec.node_index);
      CUDF_EXPECTS(node->length() == num_rows, "Feather field length does not match its batch");
      CUDF_EXPECTS(node->null_count() >= 0 && node->null_count() <= num_rows,
                   "Invalid Feather field null count");
      auto& col      = batch_columns.emplace_back();
      col.num_rows   = static_cast<size_type>(num_rows);
      col.null_count = static_cast<size_type>(node->null_count());
      if (num_rows == 0) { continue; }

      auto const validity = body_ref(spec.buffer_index);
      if (col.null_count > 0) {
        CUDF_EXPECTS(validity.size > 0, "Feather column with nulls has no validity buffer");
        auto const size = decoded_size(validity, codec);
        col.null_mask   = make_mask_buffer(col.num_rows, size, stream, batch_mr.get_output_mr());
        plan_buffer(validity, codec, reinterpret_cast<uint8_t*>(col.null_mask->data()), size, plan);
      }

      auto const values      = body_ref(spec.buffer_index + 1);
      auto const values_size = decoded_size(values, codec);
      switch (spec.layout) {
        case arrow_layout::BOOLEAN: {
          col.bits = make_mask_buffer(col.num_rows, values_size, stream, batch_mr.get_output_mr());
          plan_buffer(
            values, codec, reinterpret_cast<uint8_t*>(col.bits->data()), values_size, plan);
          break;
        }
        case arrow_layout::FIXED_WIDTH: {
          CUDF_EXPECTS(values_size >= static_cast<std::size_t>(num_rows) * size_of(spec.type),
                       "Feather values buffer is too small");
          col.values.emplace(values_size, stream, batch_mr.get_output_mr());
          plan_buffer(values, codec, static_cast<uint8_t*>(col.values->data()), values_size, plan);
          break;
        }
        default: {
          auto const offset_width = spec.layout == arrow_layout::UTF8 ? 4 : 8;
          CUDF_EXPECTS(values_size >= static_cast<std::size_t>(num_rows + 1) * offset_width,
                       "Feather offsets buffer is too small");
          col.values.emplace(values_size, stream, batch_mr.get_output_mr());
          plan_buffer(values, codec, static_cast<uint8_t*>(col.values->data()), values_size, plan);

          auto const chars      = body_ref(spec.buffer_index + 2);
          auto const chars_size = decoded_size(chars, codec);
          col.chars.emplace(chars_size, stream, batch_mr.get_output_mr());
          plan_buffer(chars, codec, static_cast<uint8_t*>(col.chars->data()), chars_size, plan);
          break;
        }
      }
    }
  }

  batched_copy(plan.copies, stream, mr);
  batched_decompress(nvcomp::compression_type::LZ4, std::move(plan.lz4), stream, mr);
  batched_decompress(nvcomp::compression_type::ZSTD, std::move(plan.zstd), stream, mr);

  std::vector<std::unique_ptr<column>> output;
  table_metadata metadata;
  for (std::size_t c = 0; c < specs.size(); ++c) {
    metadata.schema_info.emplace_back(specs[c].name);
    if (num_batches == 0) {
      output.push_back(make_empty_column(specs[c].type));
      continue;
    }
    std::vector<std::unique_ptr<column>> pieces;
    for (auto& batch : batches) {
      pieces.push_back(make_column(specs[c], std::move(batch[c]), stream, batch_mr));
    }
    if (pieces.size() == 1) {
      output.push_back(std::move(pieces.front()));
      continue;
    }
    std::vector<column_view> views;
    std::transform(pieces.begin(), pieces.end(), std::back_inserter(views), [](auto const& piece) {
      return piece->view();
    });
    output.push_back(cudf::detail::concatenate(views, stream, mr));
  }
  metadata.num_rows_per_source = {static_cast<std::size_t>(total_rows)};

  // Host bodies and staging buffers must outlive the copies enqueued above
  stream.sync();
  return {std::make_unique<table>(std::move(output)), std::move(metadata)};
}

}  // namespace cudf::io::feather::detail
