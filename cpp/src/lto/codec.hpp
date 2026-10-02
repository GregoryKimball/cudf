/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/detail/codec.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>

namespace cudf::experimental::lto::detail {

/// Encoded chunks of every registered codec are aligned to this many bytes.
inline constexpr std::size_t chunk_alignment = 16;

/// @return The registered codec's bound on encoded bytes for `chunk_bytes` raw bytes
[[nodiscard]] std::size_t max_encoded_bytes(uint32_t codec_id, std::size_t chunk_bytes);

/**
 * @brief Encodes each input chunk into its output with the registered codec `codec_id`.
 *
 * `results[i]` receives the bytes written, or `FAILURE` if the codec failed or overflowed.
 */
void encode_chunks(uint32_t codec_id,
                   uint32_t element_bytes,
                   device_span<device_span<uint8_t const> const> inputs,
                   device_span<device_span<uint8_t> const> outputs,
                   device_span<cudf::io::detail::codec_exec_result> results,
                   cuda::stream_ref stream);

/**
 * @brief Decodes each input chunk into its output, which it must fill exactly.
 */
void decode_chunks(uint32_t codec_id,
                   uint32_t element_bytes,
                   device_span<device_span<uint8_t const> const> inputs,
                   device_span<device_span<uint8_t> const> outputs,
                   device_span<cudf::io::detail::codec_exec_result> results,
                   cuda::stream_ref stream);

}  // namespace cudf::experimental::lto::detail
