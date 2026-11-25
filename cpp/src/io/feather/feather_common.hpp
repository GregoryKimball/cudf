/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/parquet/ipc/File_generated.h"
#include "io/parquet/ipc/Message_generated.h"
#include "io/parquet/ipc/Schema_generated.h"

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cudf::io::feather::detail {

namespace flatbuf = cudf::io::parquet::flatbuf;

/// File magic, written at the start (padded to 8 bytes) and end of the file
inline constexpr char file_magic[]           = "ARROW1";
inline constexpr std::size_t file_magic_size = 6;
/// Marks the start of an encapsulated IPC message
inline constexpr uint32_t continuation_marker = 0xFFFF'FFFFu;
/// Alignment of messages and of every buffer inside a message body
inline constexpr std::size_t ipc_alignment = 8;
/// Alignment the writer uses for buffers inside a message body
inline constexpr std::size_t body_buffer_alignment = 64;
/// Prefix of each compressed buffer: the int64 uncompressed length
inline constexpr std::size_t compressed_length_prefix_size = sizeof(int64_t);
/// Uncompressed length value marking a buffer stored without compression
inline constexpr int64_t uncompressed_buffer_marker = -1;

inline constexpr uint32_t lz4_frame_magic             = 0x184D'2204u;
inline constexpr uint32_t lz4_block_uncompressed_flag = 0x8000'0000u;
inline constexpr uint8_t lz4_flg_version_mask         = 0xC0;
inline constexpr uint8_t lz4_flg_version              = 0x40;
inline constexpr uint8_t lz4_flg_block_indep          = 0x20;
inline constexpr uint8_t lz4_flg_block_checksum       = 0x10;
inline constexpr uint8_t lz4_flg_content_size         = 0x08;
inline constexpr uint8_t lz4_flg_content_checksum     = 0x04;
inline constexpr uint8_t lz4_flg_dict_id              = 0x01;

inline constexpr std::size_t lz4_frame_header_size = 7;
inline constexpr std::size_t lz4_block_header_size = 4;
inline constexpr std::size_t lz4_end_mark_size     = 4;

inline constexpr uint32_t zstd_frame_magic          = 0xFD2F'B528u;
inline constexpr uint32_t zstd_skippable_magic_mask = 0xFFFF'FFF0u;
inline constexpr uint32_t zstd_skippable_magic      = 0x184D'2A50u;

/**
 * @brief Returns the LZ4 frame maximum block size for a block size ID (4 through 7).
 */
constexpr std::size_t lz4_block_max_size(uint8_t block_size_id)
{
  return std::size_t{1} << (8 + 2 * block_size_id);
}

constexpr std::size_t round_up(std::size_t value, std::size_t alignment)
{
  return (value + alignment - 1) / alignment * alignment;
}

/**
 * @brief XXH32 hash with seed 0, used for the LZ4 frame header checksum.
 */
constexpr uint32_t xxh32(std::span<uint8_t const> input)
{
  constexpr uint32_t p1 = 2654435761u;
  constexpr uint32_t p2 = 2246822519u;
  constexpr uint32_t p3 = 3266489917u;
  constexpr uint32_t p4 = 668265263u;
  constexpr uint32_t p5 = 374761393u;

  auto const rotl  = [](uint32_t x, int r) { return (x << r) | (x >> (32 - r)); };
  auto const read4 = [&](std::size_t i) {
    return static_cast<uint32_t>(input[i]) | (static_cast<uint32_t>(input[i + 1]) << 8) |
           (static_cast<uint32_t>(input[i + 2]) << 16) |
           (static_cast<uint32_t>(input[i + 3]) << 24);
  };
  auto const round = [&](uint32_t acc, uint32_t lane) { return rotl(acc + lane * p2, 13) * p1; };

  std::size_t i = 0;
  uint32_t h    = 0;
  if (input.size() >= 16) {
    uint32_t v1 = p1 + p2;
    uint32_t v2 = p2;
    uint32_t v3 = 0;
    uint32_t v4 = 0u - p1;
    for (; i + 16 <= input.size(); i += 16) {
      v1 = round(v1, read4(i));
      v2 = round(v2, read4(i + 4));
      v3 = round(v3, read4(i + 8));
      v4 = round(v4, read4(i + 12));
    }
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
  } else {
    h = p5;
  }
  h += static_cast<uint32_t>(input.size());
  for (; i + 4 <= input.size(); i += 4) {
    h = rotl(h + read4(i) * p3, 17) * p4;
  }
  for (; i < input.size(); ++i) {
    h = rotl(h + input[i] * p5, 11) * p1;
  }
  h ^= h >> 15;
  h *= p2;
  h ^= h >> 13;
  h *= p3;
  h ^= h >> 16;
  return h;
}

/**
 * @brief Reads a little-endian scalar from an unaligned host address.
 */
template <typename T>
T read_le(uint8_t const* src)
{
  T value;
  std::memcpy(&value, src, sizeof(T));
  return value;
}

/**
 * @brief Writes a little-endian scalar to an unaligned host address.
 */
template <typename T>
void write_le(uint8_t* dst, T value)
{
  std::memcpy(dst, &value, sizeof(T));
}

/**
 * @brief Physical layout of a supported Arrow column.
 */
enum class arrow_layout : uint8_t {
  FIXED_WIDTH,  ///< validity + values
  BOOLEAN,      ///< validity + bit-packed values
  UTF8,         ///< validity + int32 offsets + chars
  LARGE_UTF8,   ///< validity + int64 offsets + chars
  LIST,         ///< validity + int32 offsets; one child
  STRUCT,       ///< validity; any number of children
};

/**
 * @brief Returns the number of IPC buffers for a supported Arrow layout, excluding children.
 */
constexpr int32_t num_buffers(arrow_layout layout)
{
  switch (layout) {
    case arrow_layout::UTF8:
    case arrow_layout::LARGE_UTF8: return 3;
    case arrow_layout::STRUCT: return 1;
    default: return 2;
  }
}

/**
 * @brief A supported column in the file schema.
 */
struct column_spec {
  std::string name;
  data_type type;
  arrow_layout layout;
  int32_t node_index;    ///< Index of the column's node in each record batch
  int32_t buffer_index;  ///< Index of the column's first buffer in each record batch
  std::vector<column_spec> children;
};

/**
 * @brief Resolves the columns to read from the file schema.
 *
 * @throws cudf::logic_error if a requested column is missing or has an unsupported type
 *
 * @param schema File schema
 * @param columns Names of the columns to read, or all columns if not set
 * @return Specs of the columns to read, in output order
 */
std::vector<column_spec> select_columns(flatbuf::Schema const* schema,
                                        std::optional<std::vector<std::string>> const& columns);

/**
 * @brief A piece of a compressed buffer that decodes independently.
 */
struct frame_chunk {
  std::size_t src_offset;  ///< Offset of the payload within the compressed data
  std::size_t src_size;    ///< Size of the payload
  std::size_t dst_offset;  ///< Offset of the decoded bytes within the buffer
  std::size_t dst_size;    ///< Number of decoded bytes
  bool compressed;         ///< false if the payload is stored raw
};

/**
 * @brief Splits an LZ4 frame into blocks that nvCOMP can decompress independently.
 *
 * @throws cudf::logic_error if the frame is malformed or uses linked blocks
 *
 * @param frame The frame bytes (after the buffer's length prefix)
 * @param decoded_size Uncompressed length of the buffer
 * @return Blocks in frame order
 */
std::vector<frame_chunk> parse_lz4_frame(std::span<uint8_t const> frame, std::size_t decoded_size);

/**
 * @brief Splits a sequence of ZSTD frames into independently decodable frames.
 *
 * @throws cudf::logic_error if a frame is malformed, or if there are several frames and one does
 * not record its content size
 *
 * @param frames The frame bytes (after the buffer's length prefix)
 * @param decoded_size Uncompressed length of the buffer
 * @return Frames in order
 */
std::vector<frame_chunk> parse_zstd_frames(std::span<uint8_t const> frames,
                                           std::size_t decoded_size);

/// Size of the chunks the writer splits each buffer into for compression; nvCOMP needs many
/// small chunks to fill the GPU
inline constexpr std::size_t compression_chunk_size = std::size_t{64} << 10;

/**
 * @brief A column to be written.
 */
struct field_info {
  std::string name;
  data_type type;
  arrow_layout layout;
  std::vector<field_info> children;
};

/**
 * @brief Returns the encapsulated IPC schema message for the fields.
 *
 * @throws cudf::logic_error if a field type cannot be stored in Feather
 */
std::vector<uint8_t> make_schema_message(std::vector<field_info> const& fields);

/**
 * @brief Returns the encapsulated IPC record batch message.
 */
std::vector<uint8_t> make_record_batch_message(int64_t num_rows,
                                               std::vector<flatbuf::FieldNode> const& nodes,
                                               std::vector<flatbuf::Buffer> const& buffers,
                                               std::optional<flatbuf::CompressionType> codec,
                                               int64_t body_size);

/**
 * @brief Returns the file footer flatbuffer (without its length and trailing magic).
 */
std::vector<uint8_t> make_footer(std::vector<field_info> const& fields,
                                 std::vector<flatbuf::Block> const& record_batches);

/**
 * @brief A range of a source buffer compressed as one unit.
 */
struct compression_chunk {
  std::size_t buffer;  ///< Index of the source buffer
  std::size_t offset;  ///< Offset within the source buffer
  std::size_t size;
};

/**
 * @brief Splits buffers into `compression_chunk_size` chunks.
 */
std::vector<compression_chunk> split_into_chunks(std::vector<std::size_t> const& buffer_sizes);

/**
 * @brief A copy into the body, from a source buffer or a compressed chunk.
 */
struct body_copy {
  bool from_chunk;  ///< true: `index` is a chunk; false: `index` is a source buffer
  std::size_t index;
  std::size_t src_offset;
  std::size_t dst_offset;
  std::size_t size;
};

/**
 * @brief Little-endian scalar written into the body on the host.
 */
struct header_patch {
  std::size_t offset;
  uint64_t value;
  uint8_t width;
};

/**
 * @brief Layout of an encoded record batch body.
 */
struct body_plan {
  std::vector<flatbuf::Buffer> buffers;
  std::vector<body_copy> copies;
  std::vector<header_patch> patches;
  std::size_t size;
};

/**
 * @brief Lays out uncompressed buffers in a body.
 */
body_plan plan_uncompressed_body(std::vector<std::size_t> const& buffer_sizes);

/**
 * @brief Lays out compressed buffers in a body.
 *
 * LZ4 buffers become a single frame of independent blocks, ZSTD buffers a sequence of frames.
 * Buffers that do not shrink, and LZ4 blocks that do not shrink, are stored raw.
 *
 * @param buffer_sizes Size of each source buffer
 * @param chunks Chunks from `split_into_chunks(buffer_sizes)`
 * @param compressed_sizes Compressed size of each chunk
 * @param codec Compression codec
 */
body_plan plan_compressed_body(std::vector<std::size_t> const& buffer_sizes,
                               std::vector<compression_chunk> const& chunks,
                               std::vector<std::size_t> const& compressed_sizes,
                               flatbuf::CompressionType codec);

}  // namespace cudf::io::feather::detail
