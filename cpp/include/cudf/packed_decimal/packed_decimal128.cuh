/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <cstdint>

namespace cudf {

[[nodiscard]] __device__ inline uint8_t packed_decimal128_width(__int128_t value) noexcept
{
  if (value == 0) { return 0; }
  for (uint8_t width = 1; width < 16; ++width) {
    auto const limit = static_cast<unsigned __int128>(1) << (width * 8 - 1);
    if (value >= -static_cast<__int128_t>(limit) &&
        value <= static_cast<__int128_t>(limit - 1)) {
      return width;
    }
  }
  return 16;
}

/**
 * @brief Device accessor for logical coefficients in a packed DECIMAL128 column or slice.
 */
class packed_decimal128_device_view {
 public:
  CUDF_HOST_DEVICE packed_decimal128_device_view(uint8_t const* payload,
                                                 uint64_t const* descriptors,
                                                 size_type offset) noexcept
    : _payload{payload}, _descriptors{descriptors}, _offset{offset}
  {
  }

  [[nodiscard]] __device__ __int128_t coefficient(size_type row) const noexcept
  {
    auto const absolute_row = _offset + row;
    auto const descriptor   = _descriptors[absolute_row / block_size];
    auto const width        = static_cast<uint8_t>(descriptor & 0x1f);
    auto const byte_offset  = descriptor >> 5;
    auto const slot         = absolute_row % block_size;
    auto const address = _payload + byte_offset + static_cast<uint64_t>(slot) * width;

    switch (width) {
      case 0: return 0;
      case 1: return *reinterpret_cast<int8_t const*>(address);
      case 2: return *reinterpret_cast<int16_t const*>(address);
      case 4: return *reinterpret_cast<int32_t const*>(address);
      case 8: return *reinterpret_cast<int64_t const*>(address);
      case 16: return *reinterpret_cast<__int128_t const*>(address);
    }
    unsigned __int128 value = 0;
    for (uint8_t byte = 0; byte < width; ++byte) {
      value |= static_cast<unsigned __int128>(address[byte])
               << (byte * 8);
    }
    if (width != 0 && width != 16 &&
        (value & (static_cast<unsigned __int128>(0x80) << ((width - 1) * 8)))) {
      value |= ~static_cast<unsigned __int128>(0) << (width * 8);
    }
    return static_cast<__int128_t>(value);
  }

  [[nodiscard]] __device__ __int128_t operator[](size_type row) const noexcept
  {
    return coefficient(row);
  }

 private:
  static constexpr size_type block_size{256};

  uint8_t const* _payload;
  uint64_t const* _descriptors;
  size_type _offset;
};

}  // namespace cudf
