# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../set_cuda_architecture.cmake)
rapids_cuda_init_architectures(fixed_width_transforms_examples)

project(
  fixed_width_transforms_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA ASM
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

find_package(CUDAToolkit REQUIRED)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/thirdparty/get_zstd.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/thirdparty/get_xxhash.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/thirdparty/get_nvtx.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/thirdparty/get_rtcx.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/Modules/AddFragment.cmake)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

rtcx_add_embed(oracle_number_fragments)
add_fragment(
  oracle_number_fragments
  FRAGMENT
  oracle_number
  SOURCE
  oracle_number/transform_lto.cu
  INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_LIST_DIR}/oracle_number
  LINK_LIBRARIES
  cudf::cudf
)
rtcx_embed(
  oracle_number_fragments COMPRESSION none OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/embed"
)

add_executable(oracle_number_inspired_lto oracle_number/oracle_number.cpp)
target_compile_features(oracle_number_inspired_lto PRIVATE cxx_std_20)
target_link_libraries(oracle_number_inspired_lto PRIVATE cudf::cudf)
target_sources(
  oracle_number_inspired_lto PRIVATE ${oracle_number_fragments_SOURCE_DIR}/oracle_number_fragments.s
)
target_include_directories(oracle_number_inspired_lto PRIVATE ${oracle_number_fragments_SOURCE_DIR})
add_dependencies(oracle_number_inspired_lto oracle_number_fragments)

install(TARGETS oracle_number_inspired_lto
        DESTINATION bin/examples/libcudf/fixed_width_transformers
)
