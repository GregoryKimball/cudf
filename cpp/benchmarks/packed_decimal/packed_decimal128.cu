/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/packed_decimal/packed_decimal128.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <nvbench/nvbench.cuh>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>

#include <cstdint>
#include <memory>
#include <string>

namespace {

enum class representation { DECIMAL32, DECIMAL64, DECIMAL128, PACKED_DECIMAL128 };
enum class operation { HASH, GATHER, SORTED_ORDER, SORTED_ORDER_MULTI };

representation parse_representation(std::string const& name)
{
  if (name == "decimal32") { return representation::DECIMAL32; }
  if (name == "decimal64") { return representation::DECIMAL64; }
  if (name == "decimal128") { return representation::DECIMAL128; }
  if (name == "packed_decimal128") { return representation::PACKED_DECIMAL128; }
  CUDF_FAIL("Unknown packed-decimal benchmark representation");
}

operation parse_operation(std::string const& name)
{
  if (name == "hash") { return operation::HASH; }
  if (name == "gather") { return operation::GATHER; }
  if (name == "sorted_order") { return operation::SORTED_ORDER; }
  if (name == "sorted_order_multi") { return operation::SORTED_ORDER_MULTI; }
  CUDF_FAIL("Unknown packed-decimal benchmark operation");
}

int representation_bits(representation rep)
{
  switch (rep) {
    case representation::DECIMAL32: return 32;
    case representation::DECIMAL64: return 64;
    case representation::DECIMAL128:
    case representation::PACKED_DECIMAL128: return 128;
  }
  CUDF_UNREACHABLE("Invalid packed-decimal benchmark representation");
}

__device__ __int128_t make_coefficient(int value_bits, cudf::size_type row)
{
  // Values are chosen from the outer half of the requested signed range, so every non-empty
  // 256-row block requires exactly ceil(value_bits / 8) bytes.
  auto const leading_bit   = static_cast<unsigned __int128>(1) << (value_bits - 2);
  auto const random        = static_cast<uint64_t>(row) * 0x9e3779b97f4a7c15ULL;
  auto const low_mask_bits = value_bits > 65 ? 63 : value_bits - 2;
  auto const low_mask =
    low_mask_bits == 63 ? uint64_t{0x7fff'ffff'ffff'ffff} : (uint64_t{1} << low_mask_bits) - 1;
  auto const magnitude = leading_bit | static_cast<unsigned __int128>(random & low_mask);
  auto const value     = static_cast<__int128_t>(magnitude);
  return (row & 1) == 0 ? value : -value;
}

template <typename Rep>
__global__ void generate_coefficients(Rep* output, cudf::size_type size, int value_bits)
{
  auto const row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < size) { output[row] = static_cast<Rep>(make_coefficient(value_bits, row)); }
}

template <typename Rep>
void generate(cudf::mutable_column_view output, int value_bits, cuda::stream_ref stream)
{
  constexpr int threads = 256;
  auto const blocks     = (output.size() + threads - 1) / threads;
  generate_coefficients<<<blocks, threads, 0, stream.get()>>>(
    output.data<Rep>(), output.size(), value_bits);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

std::unique_ptr<cudf::column> make_decimal_input(representation rep,
                                                 cudf::size_type num_rows,
                                                 int value_bits,
                                                 cuda::stream_ref stream)
{
  auto const mr     = cudf::get_current_device_resource_ref();
  auto make_decimal = [&](cudf::type_id id) {
    return cudf::make_fixed_width_column(
      cudf::data_type{id, 0}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  };

  switch (rep) {
    case representation::DECIMAL32: {
      auto result = make_decimal(cudf::type_id::DECIMAL32);
      generate<int32_t>(result->mutable_view(), value_bits, stream);
      return result;
    }
    case representation::DECIMAL64: {
      auto result = make_decimal(cudf::type_id::DECIMAL64);
      generate<int64_t>(result->mutable_view(), value_bits, stream);
      return result;
    }
    case representation::DECIMAL128: {
      auto result = make_decimal(cudf::type_id::DECIMAL128);
      generate<__int128_t>(result->mutable_view(), value_bits, stream);
      return result;
    }
    case representation::PACKED_DECIMAL128: {
      auto dense = make_decimal(cudf::type_id::DECIMAL128);
      generate<__int128_t>(dense->mutable_view(), value_bits, stream);
      return cudf::encode_packed_decimal128(dense->view(), stream, mr);
    }
  }
  CUDF_UNREACHABLE("Invalid packed-decimal benchmark representation");
}

void add_footprint_summaries(nvbench::state& state,
                             std::size_t input_bytes,
                             std::size_t decimal128_bytes,
                             cudf::size_type num_rows)
{
  auto& bytes = state.add_summary("input_bytes");
  bytes.set_string("name", "Input Bytes");
  bytes.set_string("description", "Allocator-visible bytes owned by the input column");
  bytes.set_string("hint", "bytes");
  bytes.set_int64("value", static_cast<nvbench::int64_t>(input_bytes));

  auto& bytes_per_row = state.add_summary("input_bytes_per_row");
  bytes_per_row.set_string("name", "Bytes/Row");
  bytes_per_row.set_string("description", "Allocator-visible input bytes divided by row count");
  bytes_per_row.set_float64("value",
                            num_rows == 0 ? 0.0 : static_cast<double>(input_bytes) / num_rows);

  auto& compression = state.add_summary("vs_decimal128");
  compression.set_string("name", "vs Decimal128");
  compression.set_string("description", "Dense DECIMAL128 bytes divided by input bytes");
  compression.set_float64(
    "value", input_bytes == 0 ? 0.0 : static_cast<double>(decimal128_bytes) / input_bytes);
}

void bench_packed_decimal_algorithms(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const value_bits = static_cast<int>(state.get_int64("value_bits"));
  auto const rep        = parse_representation(state.get_string("representation"));
  auto const op         = parse_operation(state.get_string("operation"));

  if (value_bits > representation_bits(rep)) {
    state.skip("value_bits exceeds the selected representation");
    return;
  }
  auto const stream = cudf::get_default_stream();
  auto input        = make_decimal_input(rep, num_rows, value_bits, stream);
  std::unique_ptr<cudf::column> secondary_key;
  if (op == operation::SORTED_ORDER_MULTI) {
    secondary_key = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     secondary_key->mutable_view().begin<int32_t>(),
                     secondary_key->mutable_view().end<int32_t>());
  }
  std::unique_ptr<cudf::column> gather_map;
  if (op == operation::GATHER) {
    gather_map = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     gather_map->mutable_view().begin<cudf::size_type>(),
                     gather_map->mutable_view().end<cudf::size_type>(),
                     num_rows - 1,
                     -1);
  }
  auto const input_table = cudf::table_view{{input->view()}};
  auto const multi_column_input =
    op == operation::SORTED_ORDER_MULTI
      ? cudf::table_view{{input->view(), secondary_key->view()}}
      : cudf::table_view{};
  auto const input_bytes =
    input->alloc_size() +
    (op == operation::SORTED_ORDER_MULTI ? secondary_key->alloc_size() : std::size_t{0});
  auto const decimal128_bytes =
    static_cast<std::size_t>(num_rows) *
    (sizeof(__int128_t) +
     (op == operation::SORTED_ORDER_MULTI ? sizeof(int32_t) : std::size_t{0}));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_element_count(num_rows);
  state.add_global_memory_reads<nvbench::uint8_t>(input_bytes);
  switch (op) {
    case operation::HASH: state.add_global_memory_writes<uint64_t>(num_rows); break;
    case operation::GATHER:
      state.add_global_memory_reads<cudf::size_type>(num_rows);
      state.add_global_memory_writes<nvbench::uint8_t>(input->alloc_size());
      break;
    case operation::SORTED_ORDER:
    case operation::SORTED_ORDER_MULTI:
      state.add_global_memory_writes<cudf::size_type>(num_rows);
      break;
  }
  add_footprint_summaries(state, input_bytes, decimal128_bytes, num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    switch (rep) {
      case representation::DECIMAL32:
      case representation::DECIMAL64:
      case representation::DECIMAL128:
      case representation::PACKED_DECIMAL128:
        switch (op) {
          case operation::HASH:
            cudf::hashing::xxhash_64(
              input_table, cudf::DEFAULT_HASH_SEED, stream, cudf::get_current_device_resource_ref());
            break;
          case operation::GATHER:
            cudf::gather(input_table,
                         gather_map->view(),
                         cudf::out_of_bounds_policy::DONT_CHECK,
                         stream,
                         cudf::get_current_device_resource_ref());
            break;
          case operation::SORTED_ORDER:
            cudf::sorted_order(
              input_table, {}, {}, stream, cudf::get_current_device_resource_ref());
            break;
          case operation::SORTED_ORDER_MULTI:
            cudf::sorted_order(
              multi_column_input, {}, {}, stream, cudf::get_current_device_resource_ref());
            break;
        }
        break;
      }
  });
}

}  // namespace

NVBENCH_BENCH(bench_packed_decimal_algorithms)
  .set_name("packed_decimal128_algorithms")
  .add_string_axis(
    "operation", {"hash", "gather", "sorted_order", "sorted_order_multi"})
  .add_string_axis("representation", {"decimal32", "decimal64", "decimal128", "packed_decimal128"})
  .add_int64_axis("value_bits", {8, 16, 32, 64, 96, 128})
  .add_int64_axis(
    "num_rows", {16384, 65536, 262144, 1048576, 2097152, 4194304, 8388608, 16777216});
