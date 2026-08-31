/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <oracle_number_fragments.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

__int128_t parse_coefficient(std::string_view text)
{
  auto negative = !text.empty() && text.front() == '-';
  if (negative) { text.remove_prefix(1); }
  __int128_t value = 0;
  for (auto const character : text) {
    if (character < '0' || character > '9') {
      throw std::invalid_argument("invalid decimal coefficient");
    }
    value = value * 10 + (character - '0');
  }
  return negative ? -value : value;
}

std::string to_string(__int128_t value)
{
  if (value == 0) { return "0"; }
  auto const negative = value < 0;
  auto magnitude =
    negative ? static_cast<__uint128_t>(-(value + 1)) + 1 : static_cast<__uint128_t>(value);
  std::string result;
  while (magnitude != 0) {
    result.push_back(static_cast<char>('0' + magnitude % 10));
    magnitude /= 10;
  }
  if (negative) { result.push_back('-'); }
  std::reverse(result.begin(), result.end());
  return result;
}

template <typename T>
std::unique_ptr<cudf::column> make_column(cudf::data_type type,
                                          std::span<T const> values,
                                          rmm::cuda_stream_view stream)
{
  auto result = cudf::make_fixed_width_column(type, values.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(result->mutable_view().template data<T>(),
                                values.data(),
                                values.size_bytes(),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  return result;
}

std::unique_ptr<cudf::column> make_number_column(std::span<__int128_t const> coefficients,
                                                 std::span<int16_t const> exponents,
                                                 rmm::cuda_stream_view stream)
{
  if (coefficients.size() != exponents.size()) {
    throw std::invalid_argument("coefficient and exponent sizes differ");
  }
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(make_column(
    cudf::data_type{cudf::type_id::DECIMAL128, numeric::scale_type{0}}, coefficients, stream));
  children.push_back(make_column(cudf::data_type{cudf::type_id::INT16}, exponents, stream));
  return cudf::make_structs_column(
    coefficients.size(), std::move(children), 0, rmm::device_buffer{}, stream);
}

template <typename T>
std::vector<T> copy_to_host(cudf::column_view input, rmm::cuda_stream_view stream)
{
  std::vector<T> result(input.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(result.data(),
                                input.data<T>(),
                                result.size() * sizeof(T),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  return result;
}

std::unique_ptr<cudf::column> run_lto(cudf::column_view lhs,
                                      cudf::column_view rhs,
                                      int8_t operation,
                                      rmm::cuda_stream_view stream)
{
  auto const lhs_number = cudf::structs_column_view{lhs};
  auto const rhs_number = cudf::structs_column_view{rhs};
  std::vector<int8_t> const operation_value{operation};
  auto operation_column =
    make_column<int8_t>(cudf::data_type{cudf::type_id::INT8}, operation_value, stream);

  cudf::transform_input inputs[]   = {lhs_number.get_sliced_child(0),
                                      lhs_number.get_sliced_child(1),
                                      rhs_number.get_sliced_child(0),
                                      rhs_number.get_sliced_child(1),
                                      cudf::scalar_column_view(operation_column->view())};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::DECIMAL128, numeric::scale_type{0}},
     cudf::output_nullability::ALL_VALID},
    {cudf::data_type{cudf::type_id::INT16}, cudf::output_nullability::ALL_VALID}};

  auto const range = oracle_number_fragments::file_ranges[oracle_number_fragments::oracle_number];
  auto const udf   = oracle_number_fragments::files.subspan(range[0], range[1]);
  auto result      = cudf::transform_lto(udf,
                                    cudf::lto_binary_type::FATBIN,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    inputs,
                                    outputs,
                                         {},
                                    std::nullopt,
                                    stream);

  auto children = result->release();
  return cudf::make_structs_column(
    lhs.size(), std::move(children), 0, rmm::device_buffer{}, stream);
}

}  // namespace

int main(int argc, char const** argv)
try {
  if (argc != 2 ||
      (std::string_view{argv[1]} != "add" && std::string_view{argv[1]} != "multiply")) {
    std::cerr << "usage: oracle_number_inspired_lto <add|multiply>\n";
    return EXIT_FAILURE;
  }

  auto const stream = cudf::get_default_stream();
  std::vector<__int128_t> const lhs_coefficients{
    parse_coefficient("12345"),
    parse_coefficient("99999999999999999999999999999999999999"),
    parse_coefficient("1"),
    parse_coefficient("-5"),
    parse_coefficient("12345678901234567890123456789012345678"),
    parse_coefficient("1200"),
    parse_coefficient("1")};
  std::vector<int16_t> const lhs_exponents{-2, 0, 125, 0, 0, -2, -130};
  std::vector<__int128_t> const rhs_coefficients{
    55, parse_coefficient("99999999999999999999999999999999999999"), 1, 5, 15, 300, 1};
  std::vector<int16_t> const rhs_exponents{-1, 0, -130, 0, 0, -2, -130};

  auto lhs = make_number_column(lhs_coefficients, lhs_exponents, stream);
  auto rhs = make_number_column(rhs_coefficients, rhs_exponents, stream);
  auto result =
    run_lto(lhs->view(), rhs->view(), std::string_view{argv[1]} == "add" ? 0 : 1, stream);

  auto const result_view = cudf::structs_column_view{result->view()};
  auto coefficients      = copy_to_host<__int128_t>(result_view.get_sliced_child(0), stream);
  auto exponents         = copy_to_host<int16_t>(result_view.get_sliced_child(1), stream);

  auto const is_add = std::string_view{argv[1]} == "add";
  std::vector<__int128_t> const expected_coefficients =
    is_add ? std::vector<__int128_t>{parse_coefficient("12895"),
                                     parse_coefficient("2"),
                                     parse_coefficient("1"),
                                     parse_coefficient("0"),
                                     parse_coefficient("12345678901234567890123456789012345693"),
                                     parse_coefficient("15"),
                                     parse_coefficient("2")}
           : std::vector<__int128_t>{parse_coefficient("678975"),
                                     parse_coefficient("99999999999999999999999999999999999998"),
                                     parse_coefficient("1"),
                                     parse_coefficient("-25"),
                                     parse_coefficient("18518518351851851835185185183518518517"),
                                     parse_coefficient("36"),
                                     parse_coefficient("0")};
  std::vector<int16_t> const expected_exponents =
    is_add ? std::vector<int16_t>{-2, 38, 125, 0, 0, 0, -130}
           : std::vector<int16_t>{-3, 38, -5, 0, 1, 0, 0};
  if (coefficients != expected_coefficients || exponents != expected_exponents) {
    throw std::runtime_error("LTO transform result did not match expected values");
  }

  std::cout << "coefficient,exponent\n";
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    std::cout << to_string(coefficients[i]) << ',' << exponents[i] << '\n';
  }
  return EXIT_SUCCESS;
} catch (std::exception const& error) {
  std::cerr << error.what() << '\n';
  return EXIT_FAILURE;
}
