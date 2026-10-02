/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/contiguous_split.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudf::experimental::detail {

/**
 * @brief Chunk directory of the data buffer of a packed table's only column.
 */
struct packed_column_chunks {
  data_type type;
  size_type num_rows;
  pack_compression compression;   ///< Codec of every chunk; never `automatic`
  uint32_t lto_codec_id;          ///< Codec id when `compression` is `lto`
  std::size_t chunk_bytes;        ///< Raw bytes per chunk except possibly the last
  std::size_t data_bytes;         ///< Raw bytes over all chunks
  std::vector<uint64_t> offsets;  ///< Payload offset of each chunk
  std::vector<uint64_t> sizes;    ///< Encoded bytes of each chunk
};

/**
 * @brief Reads the chunk directory of the single, non-nullable, fixed-width column in `input`.
 *
 * Synchronizes `stream` to read the chunk sizes stored at the front of the payload.
 *
 * @throws std::invalid_argument if `input` is not a compressed single-column payload whose data
 * buffer is chunked with one codec and stored without raw chunks
 */
packed_column_chunks read_packed_column_chunks(packed_data_view input, cuda::stream_ref stream);

}  // namespace cudf::experimental::detail
