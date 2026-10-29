/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

/**
 * @file codec.cuh
 * @brief Device ABI for column codecs linked into cuDF LTO kernels.
 *
 * A codec is a set of `extern "C"` device functions, supplied as LTO-IR, that cuDF kernels call to
 * read values of an encoded column. The kernel never interprets the encoded bytes itself.
 *
 * Load hook, identified by symbol name in `cudf::lto::column_codec::load`:
 * @code{.cpp}
 * extern "C" __device__ void my_load(void* values,
 *                                    cudf::lto::codec_scope const* scope,
 *                                    long long row,
 *                                    long long count);
 * @endcode
 *
 * The hook writes `count` consecutive values, starting at logical row `row`, to `values` in the
 * column's device storage representation: the integer `rep` for fixed-point, timestamp, and
 * duration types and the type itself otherwise. It may be called from any thread at any point of
 * the kernel and must not synchronize with other threads.
 */

namespace cudf {
namespace lto {

/**
 * @brief Arguments passed to every codec hook.
 */
struct codec_scope {
  void const* column_data;  ///< Codec-defined encoded data of the column, stable for the call
  void* user_data;          ///< Call-scope data supplied with the column; may be null
};

}  // namespace lto
}  // namespace cudf
