/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/lto/codec.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

/**
 * @file for_bitpack.hpp
 * @brief Encoder and codec descriptor of the reference frame-of-reference bit-packing codec.
 *
 * See `cudf/lto/codecs/for_bitpack.cuh` for the encoded layout.
 */

namespace CUDF_EXPORT cudf {
namespace lto {
namespace codecs {

/**
 * @brief Returns the descriptor of the frame-of-reference bit-packing codec.
 *
 * The decoder is compiled with `cudf::lto::compile_lto_ir` from `cudf/lto/codecs/for_bitpack.cuh`,
 * the same way an application compiles its own codecs.
 *
 * @return The codec descriptor
 */
column_codec const& for_bitpack_codec();

/**
 * @brief Encodes a column with frame-of-reference bit-packing.
 *
 * @throws std::invalid_argument if `input` has nulls, or if its device storage is not a 32- or
 * 64-bit integer (integers, timestamps, durations, DECIMAL32, and DECIMAL64 are supported)
 * @throws std::invalid_argument if `block_size` is not positive
 *
 * @param input Column to encode
 * @param block_size Rows per block
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the encoded data
 * @return The encoded data, laid out as described in `cudf/lto/codecs/for_bitpack.cuh`
 */
rmm::device_buffer for_bitpack_encode(
  column_view const& input,
  size_type block_size              = 1024,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace codecs
}  // namespace lto
}  // namespace CUDF_EXPORT cudf
