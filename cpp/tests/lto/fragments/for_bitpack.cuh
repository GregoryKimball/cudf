/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/lto/udf_abi.h>

// Frame-of-reference plus bit-packing: a 16-byte header, then each value's offset from the
// chunk minimum in `width` bits, packed little-endian into 64-bit words.
struct for_bitpack_header {
  long long base;
  cudf_lto_u32 width;
  cudf_lto_u32 rows;
};

constexpr cudf_lto_u32 for_bitpack_header_bytes = sizeof(for_bitpack_header);

/// Random access to row `row` of an encoded chunk in shared, global, or mapped host memory.
__device__ __forceinline__ long long for_bitpack_get(unsigned char const* chunk, cudf_lto_u32 row)
{
  auto const* header = reinterpret_cast<for_bitpack_header const*>(chunk);
  auto const width   = header->width;
  if (width == 0) { return header->base; }
  auto const* words = reinterpret_cast<unsigned long long const*>(chunk + for_bitpack_header_bytes);
  auto const bit    = static_cast<unsigned long long>(row) * width;
  auto const index  = static_cast<cudf_lto_u32>(bit >> 6);
  auto const shift  = static_cast<cudf_lto_u32>(bit & 63);
  auto value        = words[index] >> shift;
  if (shift + width > 64) { value |= words[index + 1] << (64 - shift); }
  if (width < 64) { value &= (1ULL << width) - 1; }
  return header->base + static_cast<long long>(value);
}
