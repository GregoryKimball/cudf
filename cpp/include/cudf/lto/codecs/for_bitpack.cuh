/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/lto/codec.cuh>

/**
 * @file for_bitpack.cuh
 * @brief Device decoder of the reference frame-of-reference bit-packing codec.
 *
 * Rows are split into fixed-size blocks. Each block stores its minimum (the reference) and the
 * differences from it packed with the fewest bits that hold the block's largest difference.
 * Decoding a row is O(1).
 *
 * Encoded layout, starting at `codec_scope::column_data` (all offsets in bytes from the start):
 * - `header`
 * - `references`: `int64` per block
 * - `word_offsets`: `int64` per block, the block's first word in `words`
 * - `widths`: `uint8` per block, 0 to 64 bits
 * - `words`: little-endian `uint64` bit stream, one padding word at the end
 */

namespace cudf {
namespace lto {
namespace codecs {
namespace for_bitpack {

/// Header at the start of the encoded data.
struct header {
  long long num_rows;             ///< Number of rows
  long long num_blocks;           ///< Number of blocks
  int block_size;                 ///< Rows per block
  int value_bytes;                ///< Bytes per decoded value: 4 or 8
  long long references_offset;    ///< Offset of `references`
  long long word_offsets_offset;  ///< Offset of `word_offsets`
  long long widths_offset;        ///< Offset of `widths`
  long long words_offset;         ///< Offset of `words`
};

/// Decodes the value of `row` as a 64-bit integer.
__device__ inline long long decode(unsigned char const* data, long long row)
{
  auto const* h    = reinterpret_cast<header const*>(data);
  auto const block = row / h->block_size;
  auto const reference =
    reinterpret_cast<unsigned long long const*>(data + h->references_offset)[block];
  auto const width = static_cast<unsigned>(data[h->widths_offset + block]);
  if (width == 0) { return static_cast<long long>(reference); }

  auto const* words = reinterpret_cast<unsigned long long const*>(data + h->words_offset) +
                      reinterpret_cast<long long const*>(data + h->word_offsets_offset)[block];
  auto const bit   = static_cast<unsigned long long>(row - block * h->block_size) * width;
  auto const word  = bit >> 6;
  auto const shift = static_cast<unsigned>(bit & 63);

  auto bits = words[word] >> shift;
  if (shift + width > 64) { bits |= words[word + 1] << (64 - shift); }
  auto const mask = width == 64 ? ~0ULL : (1ULL << width) - 1;
  return static_cast<long long>(reference + (bits & mask));
}

}  // namespace for_bitpack
}  // namespace codecs
}  // namespace lto
}  // namespace cudf
