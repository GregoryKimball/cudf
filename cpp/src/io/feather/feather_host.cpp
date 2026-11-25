/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "feather_common.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <string>

namespace cudf::io::feather::detail {
namespace {

data_type to_timestamp_type(flatbuf::TimeUnit unit)
{
  switch (unit) {
    case flatbuf::TimeUnit_SECOND: return data_type{type_id::TIMESTAMP_SECONDS};
    case flatbuf::TimeUnit_MILLISECOND: return data_type{type_id::TIMESTAMP_MILLISECONDS};
    case flatbuf::TimeUnit_MICROSECOND: return data_type{type_id::TIMESTAMP_MICROSECONDS};
    case flatbuf::TimeUnit_NANOSECOND: return data_type{type_id::TIMESTAMP_NANOSECONDS};
    default: CUDF_FAIL("Unsupported Arrow timestamp unit");
  }
}

data_type to_duration_type(flatbuf::TimeUnit unit)
{
  switch (unit) {
    case flatbuf::TimeUnit_SECOND: return data_type{type_id::DURATION_SECONDS};
    case flatbuf::TimeUnit_MILLISECOND: return data_type{type_id::DURATION_MILLISECONDS};
    case flatbuf::TimeUnit_MICROSECOND: return data_type{type_id::DURATION_MICROSECONDS};
    case flatbuf::TimeUnit_NANOSECOND: return data_type{type_id::DURATION_NANOSECONDS};
    default: CUDF_FAIL("Unsupported Arrow duration unit");
  }
}

data_type to_integer_type(flatbuf::Int const* type)
{
  auto const is_signed = type->is_signed();
  switch (type->bitWidth()) {
    case 8: return data_type{is_signed ? type_id::INT8 : type_id::UINT8};
    case 16: return data_type{is_signed ? type_id::INT16 : type_id::UINT16};
    case 32: return data_type{is_signed ? type_id::INT32 : type_id::UINT32};
    case 64: return data_type{is_signed ? type_id::INT64 : type_id::UINT64};
    default: CUDF_FAIL("Unsupported Arrow integer bit width");
  }
}

/**
 * @brief Maps an Arrow field to its cudf type and IPC layout, ignoring its children.
 *
 * @throws cudf::logic_error if the field type is not supported
 */
std::pair<data_type, arrow_layout> to_cudf_type(flatbuf::Field const* field)
{
  auto const name = field->name() ? field->name()->str() : std::string{};
  CUDF_EXPECTS(field->dictionary() == nullptr,
               "Dictionary-encoded Feather column '" + name + "' is not supported");
  auto const num_children = field->children() ? field->children()->size() : 0;
  switch (field->type_type()) {
    case flatbuf::Type_List:
      CUDF_EXPECTS(num_children == 1, "Feather list column '" + name + "' must have one child");
      return {data_type{type_id::LIST}, arrow_layout::LIST};
    case flatbuf::Type_Struct_: return {data_type{type_id::STRUCT}, arrow_layout::STRUCT};
    case flatbuf::Type_LargeList:
      CUDF_FAIL("Feather column '" + name +
                "' is a large list, which cudf lists with 32-bit offsets cannot hold");
    default:
      CUDF_EXPECTS(num_children == 0,
                   "Feather column '" + name + "' has unsupported nested Arrow type " +
                     flatbuf::EnumNameType(field->type_type()));
  }

  switch (field->type_type()) {
    case flatbuf::Type_Bool: return {data_type{type_id::BOOL8}, arrow_layout::BOOLEAN};
    case flatbuf::Type_Int:
      return {to_integer_type(field->type_as_Int()), arrow_layout::FIXED_WIDTH};
    case flatbuf::Type_FloatingPoint: {
      switch (field->type_as_FloatingPoint()->precision()) {
        case flatbuf::Precision_SINGLE:
          return {data_type{type_id::FLOAT32}, arrow_layout::FIXED_WIDTH};
        case flatbuf::Precision_DOUBLE:
          return {data_type{type_id::FLOAT64}, arrow_layout::FIXED_WIDTH};
        default: CUDF_FAIL("Half-precision Feather column '" + name + "' is not supported");
      }
    }
    case flatbuf::Type_Decimal: {
      auto const decimal = field->type_as_Decimal();
      auto const scale   = -decimal->scale();
      switch (decimal->bitWidth()) {
        case 32: return {data_type{type_id::DECIMAL32, scale}, arrow_layout::FIXED_WIDTH};
        case 64: return {data_type{type_id::DECIMAL64, scale}, arrow_layout::FIXED_WIDTH};
        case 128: return {data_type{type_id::DECIMAL128, scale}, arrow_layout::FIXED_WIDTH};
        default: CUDF_FAIL("Unsupported decimal bit width in Feather column '" + name + "'");
      }
    }
    case flatbuf::Type_Date: {
      return {field->type_as_Date()->unit() == flatbuf::DateUnit_DAY
                ? data_type{type_id::TIMESTAMP_DAYS}
                : data_type{type_id::TIMESTAMP_MILLISECONDS},
              arrow_layout::FIXED_WIDTH};
    }
    case flatbuf::Type_Timestamp:
      return {to_timestamp_type(field->type_as_Timestamp()->unit()), arrow_layout::FIXED_WIDTH};
    case flatbuf::Type_Duration:
      return {to_duration_type(field->type_as_Duration()->unit()), arrow_layout::FIXED_WIDTH};
    case flatbuf::Type_Utf8: return {data_type{type_id::STRING}, arrow_layout::UTF8};
    case flatbuf::Type_LargeUtf8: return {data_type{type_id::STRING}, arrow_layout::LARGE_UTF8};
    default:
      CUDF_FAIL("Feather column '" + name + "' has unsupported Arrow type " +
                flatbuf::EnumNameType(field->type_type()));
  }
}

/**
 * @brief Counts the IPC field nodes and buffers used by a field and its children.
 *
 * Used to step over columns that are not read, which may have any type.
 */
void count_nodes_and_buffers(flatbuf::Field const* field, int32_t& nodes, int32_t& buffers)
{
  ++nodes;
  if (field->dictionary() != nullptr) {
    buffers += 2;
    return;
  }
  switch (field->type_type()) {
    case flatbuf::Type_Null:
    case flatbuf::Type_RunEndEncoded: break;
    case flatbuf::Type_Struct_:
    case flatbuf::Type_FixedSizeList: buffers += 1; break;
    case flatbuf::Type_Union:
      buffers += field->type_as_Union()->mode() == flatbuf::UnionMode_Dense ? 2 : 1;
      break;
    case flatbuf::Type_Utf8:
    case flatbuf::Type_Binary:
    case flatbuf::Type_LargeUtf8:
    case flatbuf::Type_LargeBinary:
    case flatbuf::Type_ListView:
    case flatbuf::Type_LargeListView: buffers += 3; break;
    case flatbuf::Type_Utf8View:
    case flatbuf::Type_BinaryView:
      CUDF_FAIL("Feather files with view-typed columns are not supported");
    default: buffers += 2; break;
  }
  if (field->children() != nullptr) {
    for (auto const child : *field->children()) {
      count_nodes_and_buffers(child, nodes, buffers);
    }
  }
}

/**
 * @brief Builds the spec of a field and its children, whose nodes and buffers start at the given
 * positions; advances the positions past them.
 */
column_spec make_spec(flatbuf::Field const* field, int32_t& nodes, int32_t& buffers)
{
  auto const [type, layout] = to_cudf_type(field);
  column_spec spec{
    field->name() ? field->name()->str() : std::string{}, type, layout, nodes, buffers, {}};
  ++nodes;
  buffers += num_buffers(layout);
  if (layout == arrow_layout::LIST || layout == arrow_layout::STRUCT) {
    for (auto const child : *field->children()) {
      spec.children.push_back(make_spec(child, nodes, buffers));
    }
  }
  return spec;
}

}  // namespace

std::vector<column_spec> select_columns(flatbuf::Schema const* schema,
                                        std::optional<std::vector<std::string>> const& columns)
{
  CUDF_EXPECTS(schema != nullptr && schema->fields() != nullptr, "Feather file has no schema");
  CUDF_EXPECTS(schema->endianness() == flatbuf::Endianness_Little,
               "Big-endian Feather files are not supported");

  struct field_position {
    flatbuf::Field const* field;
    int32_t node_index;
    int32_t buffer_index;
  };
  std::vector<field_position> positions;
  int32_t nodes   = 0;
  int32_t buffers = 0;
  for (auto const field : *schema->fields()) {
    positions.push_back({field, nodes, buffers});
    count_nodes_and_buffers(field, nodes, buffers);
  }

  auto const spec_at = [](field_position const& pos) {
    auto node   = pos.node_index;
    auto buffer = pos.buffer_index;
    return make_spec(pos.field, node, buffer);
  };

  std::vector<column_spec> specs;
  if (not columns.has_value()) {
    std::transform(positions.begin(), positions.end(), std::back_inserter(specs), spec_at);
    return specs;
  }
  for (auto const& name : *columns) {
    auto const it = std::find_if(positions.begin(), positions.end(), [&](auto const& pos) {
      return pos.field->name() != nullptr && pos.field->name()->str() == name;
    });
    CUDF_EXPECTS(it != positions.end(), "Column '" + name + "' not found in Feather file");
    specs.push_back(spec_at(*it));
  }
  return specs;
}

std::vector<frame_chunk> parse_lz4_frame(std::span<uint8_t const> frame, std::size_t decoded_size)
{
  std::size_t pos  = 0;
  auto const check = [&](std::size_t count) {
    CUDF_EXPECTS(pos + count <= frame.size(), "Truncated LZ4 frame in Feather buffer");
  };

  check(7);
  CUDF_EXPECTS(read_le<uint32_t>(frame.data()) == lz4_frame_magic, "Invalid LZ4 frame magic");
  auto const flg = frame[4];
  auto const bd  = frame[5];
  CUDF_EXPECTS((flg & lz4_flg_version_mask) == lz4_flg_version, "Unsupported LZ4 frame version");
  CUDF_EXPECTS((flg & lz4_flg_dict_id) == 0, "LZ4 frames with a dictionary ID are not supported");
  auto const block_size_id = static_cast<uint8_t>((bd >> 4) & 0x7);
  CUDF_EXPECTS(block_size_id >= 4, "Invalid LZ4 frame block size");
  auto const block_max         = lz4_block_max_size(block_size_id);
  auto const independent       = (flg & lz4_flg_block_indep) != 0;
  auto const checksum_size     = (flg & lz4_flg_block_checksum) != 0 ? 4 : 0;
  auto const content_size_size = (flg & lz4_flg_content_size) != 0 ? 8 : 0;
  pos                          = 6 + content_size_size + 1;

  std::vector<frame_chunk> chunks;
  std::size_t out = 0;
  while (true) {
    check(4);
    auto const word = read_le<uint32_t>(frame.data() + pos);
    pos += 4;
    if (word == 0) { break; }
    auto const block_size = static_cast<std::size_t>(word & ~lz4_block_uncompressed_flag);
    check(block_size + checksum_size);
    CUDF_EXPECTS(independent or chunks.empty(),
                 "LZ4 frames with linked blocks are not supported by nvCOMP");
    auto const remaining = decoded_size - out;
    if ((word & lz4_block_uncompressed_flag) != 0) {
      CUDF_EXPECTS(block_size <= remaining, "LZ4 frame decodes to more data than expected");
      chunks.push_back({pos, block_size, out, block_size, false});
      out += block_size;
    } else {
      auto const expected = std::min(block_max, remaining);
      CUDF_EXPECTS(expected > 0, "LZ4 frame decodes to more data than expected");
      chunks.push_back({pos, block_size, out, expected, true});
      out += expected;
    }
    pos += block_size + checksum_size;
  }
  CUDF_EXPECTS(out == decoded_size, "LZ4 frame size does not match the Feather buffer length");
  return chunks;
}

std::vector<frame_chunk> parse_zstd_frames(std::span<uint8_t const> frames,
                                           std::size_t decoded_size)
{
  struct zstd_frame {
    std::size_t offset;
    std::size_t size;
    std::optional<std::size_t> content_size;
  };
  std::vector<zstd_frame> found;

  std::size_t pos  = 0;
  auto const check = [&](std::size_t count) {
    CUDF_EXPECTS(pos + count <= frames.size(), "Truncated ZSTD frame in Feather buffer");
  };
  while (pos < frames.size()) {
    check(4);
    auto const magic = read_le<uint32_t>(frames.data() + pos);
    if ((magic & zstd_skippable_magic_mask) == zstd_skippable_magic) {
      check(8);
      pos += 8 + read_le<uint32_t>(frames.data() + pos + 4);
      continue;
    }
    CUDF_EXPECTS(magic == zstd_frame_magic, "Invalid ZSTD frame magic");
    auto const start = pos;
    pos += 4;
    check(1);
    auto const fhd                        = frames[pos++];
    auto const fcs_flag                   = fhd >> 6;
    auto const single_segment             = ((fhd >> 5) & 1) != 0;
    auto const has_checksum               = ((fhd >> 2) & 1) != 0;
    constexpr std::size_t dict_id_sizes[] = {0, 1, 2, 4};
    pos += (single_segment ? 0 : 1) + dict_id_sizes[fhd & 3];
    auto const fcs_size = fcs_flag == 0 ? (single_segment ? 1 : 0) : (std::size_t{1} << fcs_flag);
    check(fcs_size);
    std::optional<std::size_t> content_size;
    switch (fcs_size) {
      case 1: content_size = frames[pos]; break;
      case 2: content_size = read_le<uint16_t>(frames.data() + pos) + 256; break;
      case 4: content_size = read_le<uint32_t>(frames.data() + pos); break;
      case 8: content_size = read_le<uint64_t>(frames.data() + pos); break;
      default: break;
    }
    pos += fcs_size;

    auto last_block = false;
    while (not last_block) {
      check(3);
      auto const header = static_cast<uint32_t>(frames[pos]) |
                          (static_cast<uint32_t>(frames[pos + 1]) << 8) |
                          (static_cast<uint32_t>(frames[pos + 2]) << 16);
      pos += 3;
      last_block            = (header & 1) != 0;
      auto const block_type = (header >> 1) & 3;
      auto const block_size = static_cast<std::size_t>(header >> 3);
      CUDF_EXPECTS(block_type != 3, "Corrupted ZSTD block in Feather buffer");
      auto const payload_size = block_type == 1 ? std::size_t{1} : block_size;
      check(payload_size);
      pos += payload_size;
    }
    if (has_checksum) {
      check(4);
      pos += 4;
    }
    found.push_back({start, pos - start, content_size});
  }
  CUDF_EXPECTS(not found.empty(), "Feather buffer contains no ZSTD frame");

  if (found.size() == 1) { return {{found[0].offset, found[0].size, 0, decoded_size, true}}; }
  std::vector<frame_chunk> chunks;
  std::size_t out = 0;
  for (auto const& frame : found) {
    CUDF_EXPECTS(frame.content_size.has_value(),
                 "Multi-frame ZSTD Feather buffers must record each frame's content size");
    CUDF_EXPECTS(out + *frame.content_size <= decoded_size,
                 "ZSTD frames decode to more data than expected");
    chunks.push_back({frame.offset, frame.size, out, *frame.content_size, true});
    out += *frame.content_size;
  }
  CUDF_EXPECTS(out == decoded_size, "ZSTD frame sizes do not match the Feather buffer length");
  return chunks;
}

namespace {

using FlatBufferBuilder = flatbuffers::FlatBufferBuilder;

inline constexpr uint8_t lz4_block_size_id = 4;
static_assert(lz4_block_max_size(lz4_block_size_id) == compression_chunk_size);

inline constexpr uint8_t lz4_frame_flg = lz4_flg_version | lz4_flg_block_indep;
inline constexpr uint8_t lz4_frame_bd  = lz4_block_size_id << 4;
inline constexpr uint8_t lz4_frame_hc  = [] {
  std::array<uint8_t, 2> const descriptor{lz4_frame_flg, lz4_frame_bd};
  return static_cast<uint8_t>((xxh32(descriptor) >> 8) & 0xFF);
}();

static_assert(xxh32(std::span<uint8_t const>{}) == 0x02CC'5D05u);
static_assert([] {
  std::array<uint8_t, 2> const descriptor{0x64, 0x40};
  return ((xxh32(descriptor) >> 8) & 0xFF) == 0xA7;
}());

flatbuf::TimeUnit to_time_unit(type_id id)
{
  switch (id) {
    case type_id::TIMESTAMP_SECONDS:
    case type_id::DURATION_SECONDS: return flatbuf::TimeUnit_SECOND;
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::DURATION_MILLISECONDS: return flatbuf::TimeUnit_MILLISECOND;
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::DURATION_MICROSECONDS: return flatbuf::TimeUnit_MICROSECOND;
    default: return flatbuf::TimeUnit_NANOSECOND;
  }
}

flatbuffers::Offset<flatbuf::Field> make_field(FlatBufferBuilder& fbb, field_info const& field)
{
  auto const id = field.type.id();
  flatbuf::Type type_type{};
  flatbuffers::Offset<void> type{};
  auto const set_int = [&](int32_t bit_width, bool is_signed) {
    type_type = flatbuf::Type_Int;
    type      = flatbuf::CreateInt(fbb, bit_width, is_signed).Union();
  };
  auto const set_decimal = [&](int32_t precision, int32_t bit_width) {
    type_type = flatbuf::Type_Decimal;
    type      = flatbuf::CreateDecimal(fbb, precision, -field.type.scale(), bit_width).Union();
  };
  switch (id) {
    case type_id::BOOL8:
      type_type = flatbuf::Type_Bool;
      type      = flatbuf::CreateBool(fbb).Union();
      break;
    case type_id::INT8: set_int(8, true); break;
    case type_id::INT16: set_int(16, true); break;
    case type_id::INT32: set_int(32, true); break;
    case type_id::INT64: set_int(64, true); break;
    case type_id::UINT8: set_int(8, false); break;
    case type_id::UINT16: set_int(16, false); break;
    case type_id::UINT32: set_int(32, false); break;
    case type_id::UINT64: set_int(64, false); break;
    case type_id::FLOAT32:
    case type_id::FLOAT64:
      type_type = flatbuf::Type_FloatingPoint;
      type      = flatbuf::CreateFloatingPoint(
               fbb, id == type_id::FLOAT32 ? flatbuf::Precision_SINGLE : flatbuf::Precision_DOUBLE)
               .Union();
      break;
    case type_id::DECIMAL32: set_decimal(9, 32); break;
    case type_id::DECIMAL64: set_decimal(18, 64); break;
    case type_id::DECIMAL128: set_decimal(38, 128); break;
    case type_id::TIMESTAMP_DAYS:
      type_type = flatbuf::Type_Date;
      type      = flatbuf::CreateDate(fbb, flatbuf::DateUnit_DAY).Union();
      break;
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
      type_type = flatbuf::Type_Timestamp;
      type      = flatbuf::CreateTimestamp(fbb, to_time_unit(id)).Union();
      break;
    case type_id::DURATION_SECONDS:
    case type_id::DURATION_MILLISECONDS:
    case type_id::DURATION_MICROSECONDS:
    case type_id::DURATION_NANOSECONDS:
      type_type = flatbuf::Type_Duration;
      type      = flatbuf::CreateDuration(fbb, to_time_unit(id)).Union();
      break;
    case type_id::STRING:
      if (field.layout == arrow_layout::LARGE_UTF8) {
        type_type = flatbuf::Type_LargeUtf8;
        type      = flatbuf::CreateLargeUtf8(fbb).Union();
      } else {
        type_type = flatbuf::Type_Utf8;
        type      = flatbuf::CreateUtf8(fbb).Union();
      }
      break;
    case type_id::LIST:
      type_type = flatbuf::Type_List;
      type      = flatbuf::CreateList(fbb).Union();
      break;
    case type_id::STRUCT:
      type_type = flatbuf::Type_Struct_;
      type      = flatbuf::CreateStruct_(fbb).Union();
      break;
    default: CUDF_FAIL("Feather writer does not support the type of column '" + field.name + "'");
  }
  // Arrow readers require a children vector even for flat fields
  std::vector<flatbuffers::Offset<flatbuf::Field>> child_offsets;
  for (auto const& child : field.children) {
    child_offsets.push_back(make_field(fbb, child));
  }
  auto const children = fbb.CreateVector(child_offsets);
  return flatbuf::CreateField(
    fbb, fbb.CreateString(field.name), true, type_type, type, 0, children);
}

flatbuffers::Offset<flatbuf::Schema> make_schema(FlatBufferBuilder& fbb,
                                                 std::vector<field_info> const& fields)
{
  std::vector<flatbuffers::Offset<flatbuf::Field>> offsets;
  for (auto const& field : fields) {
    offsets.push_back(make_field(fbb, field));
  }
  return flatbuf::CreateSchema(fbb, flatbuf::Endianness_Little, fbb.CreateVector(offsets));
}

/**
 * @brief Wraps a finished flatbuffer as an encapsulated IPC message with 8-byte aligned length.
 */
std::vector<uint8_t> encapsulate(FlatBufferBuilder const& fbb)
{
  auto const prefix_size = 2 * sizeof(int32_t);
  auto const total       = round_up(prefix_size + fbb.GetSize(), ipc_alignment);
  std::vector<uint8_t> message(total, 0);
  write_le<uint32_t>(message.data(), continuation_marker);
  write_le<int32_t>(message.data() + sizeof(int32_t), static_cast<int32_t>(total - prefix_size));
  std::memcpy(message.data() + prefix_size, fbb.GetBufferPointer(), fbb.GetSize());
  return message;
}

}  // namespace

std::vector<uint8_t> make_schema_message(std::vector<field_info> const& fields)
{
  FlatBufferBuilder fbb;
  auto const schema = make_schema(fbb, fields);
  fbb.Finish(flatbuf::CreateMessage(
    fbb, flatbuf::MetadataVersion_V5, flatbuf::MessageHeader_Schema, schema.Union(), 0));
  return encapsulate(fbb);
}

std::vector<uint8_t> make_record_batch_message(int64_t num_rows,
                                               std::vector<flatbuf::FieldNode> const& nodes,
                                               std::vector<flatbuf::Buffer> const& buffers,
                                               std::optional<flatbuf::CompressionType> codec,
                                               int64_t body_size)
{
  FlatBufferBuilder fbb;
  auto const node_offsets   = fbb.CreateVectorOfStructs(nodes);
  auto const buffer_offsets = fbb.CreateVectorOfStructs(buffers);
  auto const compression    = codec.has_value() ? flatbuf::CreateBodyCompression(fbb, *codec)
                                                : flatbuffers::Offset<flatbuf::BodyCompression>{};
  auto const batch =
    flatbuf::CreateRecordBatch(fbb, num_rows, node_offsets, buffer_offsets, compression);
  fbb.Finish(flatbuf::CreateMessage(fbb,
                                    flatbuf::MetadataVersion_V5,
                                    flatbuf::MessageHeader_RecordBatch,
                                    batch.Union(),
                                    body_size));
  return encapsulate(fbb);
}

std::vector<uint8_t> make_footer(std::vector<field_info> const& fields,
                                 std::vector<flatbuf::Block> const& record_batches)
{
  FlatBufferBuilder fbb;
  auto const schema       = make_schema(fbb, fields);
  auto const dictionaries = fbb.CreateVectorOfStructs(std::vector<flatbuf::Block>{});
  auto const batches      = fbb.CreateVectorOfStructs(record_batches);
  fbb.Finish(
    flatbuf::CreateFooter(fbb, flatbuf::MetadataVersion_V5, schema, dictionaries, batches));
  return {fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize()};
}

std::vector<compression_chunk> split_into_chunks(std::vector<std::size_t> const& buffer_sizes)
{
  std::vector<compression_chunk> chunks;
  for (std::size_t i = 0; i < buffer_sizes.size(); ++i) {
    for (std::size_t offset = 0; offset < buffer_sizes[i]; offset += compression_chunk_size) {
      chunks.push_back({i, offset, std::min(compression_chunk_size, buffer_sizes[i] - offset)});
    }
  }
  return chunks;
}

body_plan plan_uncompressed_body(std::vector<std::size_t> const& buffer_sizes)
{
  body_plan plan{};
  std::size_t pos = 0;
  for (std::size_t i = 0; i < buffer_sizes.size(); ++i) {
    pos = round_up(pos, body_buffer_alignment);
    plan.buffers.emplace_back(static_cast<int64_t>(pos), static_cast<int64_t>(buffer_sizes[i]));
    if (buffer_sizes[i] > 0) { plan.copies.push_back({false, i, 0, pos, buffer_sizes[i]}); }
    pos += buffer_sizes[i];
  }
  plan.size = round_up(pos, ipc_alignment);
  return plan;
}

body_plan plan_compressed_body(std::vector<std::size_t> const& buffer_sizes,
                               std::vector<compression_chunk> const& chunks,
                               std::vector<std::size_t> const& compressed_sizes,
                               flatbuf::CompressionType codec)
{
  auto const is_lz4 = codec == flatbuf::CompressionType_LZ4_FRAME;
  // LZ4 blocks that do not shrink are stored raw; ZSTD frames cannot be
  auto const stored_raw = [&](std::size_t c) {
    return is_lz4 && compressed_sizes[c] >= chunks[c].size;
  };
  auto const stored_size = [&](std::size_t c) {
    return stored_raw(c) ? chunks[c].size : compressed_sizes[c];
  };

  body_plan plan{};
  std::size_t pos = 0;
  std::size_t c   = 0;
  for (std::size_t i = 0; i < buffer_sizes.size(); ++i) {
    pos              = round_up(pos, body_buffer_alignment);
    auto const start = pos;
    auto const size  = buffer_sizes[i];
    if (size == 0) {
      plan.buffers.emplace_back(static_cast<int64_t>(start), 0);
      continue;
    }
    auto const first = c;
    auto const last  = first + (size + compression_chunk_size - 1) / compression_chunk_size;
    CUDF_EXPECTS(last <= chunks.size(), "Chunk list does not cover the buffers");

    std::size_t payload_size = is_lz4 ? lz4_frame_header_size + lz4_end_mark_size : 0;
    for (auto k = first; k < last; ++k) {
      payload_size += stored_size(k) + (is_lz4 ? lz4_block_header_size : 0);
    }

    pos += compressed_length_prefix_size;
    if (payload_size >= size) {
      plan.patches.push_back({start, static_cast<uint64_t>(uncompressed_buffer_marker), 8});
      plan.copies.push_back({false, i, 0, pos, size});
      pos += size;
    } else {
      plan.patches.push_back({start, static_cast<uint64_t>(size), 8});
      if (is_lz4) {
        plan.patches.push_back({pos, lz4_frame_magic, 4});
        plan.patches.push_back({pos + 4, lz4_frame_flg, 1});
        plan.patches.push_back({pos + 5, lz4_frame_bd, 1});
        plan.patches.push_back({pos + 6, lz4_frame_hc, 1});
        pos += lz4_frame_header_size;
      }
      for (auto k = first; k < last; ++k) {
        if (is_lz4) {
          auto const word = static_cast<uint32_t>(stored_size(k)) |
                            (stored_raw(k) ? lz4_block_uncompressed_flag : 0u);
          plan.patches.push_back({pos, word, 4});
          pos += lz4_block_header_size;
        }
        if (stored_raw(k)) {
          plan.copies.push_back({false, i, chunks[k].offset, pos, chunks[k].size});
        } else {
          plan.copies.push_back({true, k, 0, pos, compressed_sizes[k]});
        }
        pos += stored_size(k);
      }
      if (is_lz4) {
        plan.patches.push_back({pos, 0, 4});
        pos += lz4_end_mark_size;
      }
    }
    plan.buffers.emplace_back(static_cast<int64_t>(start), static_cast<int64_t>(pos - start));
    c = last;
  }
  plan.size = round_up(pos, ipc_alignment);
  return plan;
}

}  // namespace cudf::io::feather::detail
