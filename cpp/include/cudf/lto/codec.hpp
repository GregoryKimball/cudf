/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/udf.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/**
 * @file codec.hpp
 * @brief Host-side descriptors for columns read through user-supplied LTO codecs.
 *
 * See `cudf/lto/codec.cuh` for the device ABI of the codec hooks.
 */

namespace CUDF_EXPORT cudf {
namespace lto {

/**
 * @brief A device code fragment linked into cuDF kernels.
 */
struct lto_fragment {
  std::vector<std::uint8_t> data;                  ///< Fragment binary
  lto_binary_type type = lto_binary_type::LTO_IR;  ///< Format of `data`
};

/**
 * @brief Describes how cuDF kernels read the values of an encoded column.
 *
 * The fragments must together define the hook symbols named here. Codecs that cuDF ships, such
 * as `cudf::lto::codecs::for_bitpack_codec()`, are described the same way and take the same path
 * through the kernels as codecs supplied by applications.
 */
struct column_codec {
  std::vector<std::shared_ptr<lto_fragment const>> fragments;  ///< Fragments defining the hooks
  std::string load;  ///< Symbol of the load hook, see `cudf/lto/codec.cuh`
};

/**
 * @brief A non-owning view of a column whose values are read through a codec.
 *
 * Encoded columns have no null values.
 */
struct encoded_column_view {
  data_type type;                       ///< Logical type of the decoded values
  size_type size            = 0;        ///< Number of rows
  void const* column_data   = nullptr;  ///< Device pointer to the codec-defined encoded data
  void* user_data           = nullptr;  ///< Call-scope data passed to the hooks; may be null
  column_codec const* codec = nullptr;  ///< Codec that decodes `column_data`
};

/**
 * @brief Compiles CUDA C++ device code to an LTO-IR fragment.
 *
 * The source is compiled for the architecture cuDF links LTO kernels for and may include cuDF
 * headers such as `cudf/lto/codec.cuh`. Results are cached by source text.
 *
 * @param name Debug name of the compilation unit
 * @param source CUDA C++ source
 * @return The LTO-IR fragment
 */
std::shared_ptr<lto_fragment const> compile_lto_ir(std::string const& name,
                                                   std::string const& source);

}  // namespace lto
}  // namespace CUDF_EXPORT cudf
