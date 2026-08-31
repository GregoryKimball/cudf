/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

// A 38-digit decimal arithmetic model inspired by Oracle NUMBER's documented behavior.
// This is not a byte-compatible or complete Oracle NUMBER implementation.
namespace oracle_number_inspired {

using cuda::std::int16_t;
using cuda::std::int32_t;
using cuda::std::uint64_t;

constexpr uint64_t limb_base = 10'000'000'000'000'000'000ULL;
constexpr int32_t limb_count = 5;
constexpr int32_t precision  = 38;

struct wide_uint {
  uint64_t limbs[limb_count]{};
};

struct result {
  __int128_t coefficient{};
  int16_t exponent{};
  int32_t error{};
};

__device__ inline bool is_zero(wide_uint const& value)
{
  for (auto limb : value.limbs) {
    if (limb != 0) { return false; }
  }
  return true;
}

__device__ inline wide_uint from_u128(__uint128_t value)
{
  wide_uint result;
  result.limbs[0] = static_cast<uint64_t>(value % limb_base);
  value /= limb_base;
  result.limbs[1] = static_cast<uint64_t>(value);
  return result;
}

__device__ inline __uint128_t magnitude(__int128_t value)
{
  return value < 0 ? static_cast<__uint128_t>(-(value + 1)) + 1 : static_cast<__uint128_t>(value);
}

__device__ inline int compare(wide_uint const& lhs, wide_uint const& rhs)
{
  for (int32_t i = limb_count - 1; i >= 0; --i) {
    if (lhs.limbs[i] < rhs.limbs[i]) { return -1; }
    if (lhs.limbs[i] > rhs.limbs[i]) { return 1; }
  }
  return 0;
}

__device__ inline bool add_in_place(wide_uint& lhs, wide_uint const& rhs)
{
  __uint128_t carry = 0;
  for (int32_t i = 0; i < limb_count; ++i) {
    auto sum     = static_cast<__uint128_t>(lhs.limbs[i]) + rhs.limbs[i] + carry;
    lhs.limbs[i] = static_cast<uint64_t>(sum % limb_base);
    carry        = sum / limb_base;
  }
  return carry == 0;
}

__device__ inline void subtract_in_place(wide_uint& lhs, wide_uint const& rhs)
{
  uint64_t borrow = 0;
  for (int32_t i = 0; i < limb_count; ++i) {
    auto const subtrahend = static_cast<__uint128_t>(rhs.limbs[i]) + borrow;
    if (static_cast<__uint128_t>(lhs.limbs[i]) < subtrahend) {
      lhs.limbs[i] =
        static_cast<uint64_t>(static_cast<__uint128_t>(lhs.limbs[i]) + limb_base - subtrahend);
      borrow = 1;
    } else {
      lhs.limbs[i] = static_cast<uint64_t>(static_cast<__uint128_t>(lhs.limbs[i]) - subtrahend);
      borrow       = 0;
    }
  }
}

__device__ inline bool multiply_small(wide_uint& value, uint64_t multiplier)
{
  __uint128_t carry = 0;
  for (int32_t i = 0; i < limb_count; ++i) {
    auto product   = static_cast<__uint128_t>(value.limbs[i]) * multiplier + carry;
    value.limbs[i] = static_cast<uint64_t>(product % limb_base);
    carry          = product / limb_base;
  }
  return carry == 0;
}

__device__ inline bool multiply_pow10(wide_uint& value, int32_t exponent)
{
  for (int32_t i = 0; i < exponent; ++i) {
    if (!multiply_small(value, 10)) { return false; }
  }
  return true;
}

__device__ inline wide_uint multiply(wide_uint const& lhs, wide_uint const& rhs, bool& valid)
{
  wide_uint product;
  valid = true;
  // A valid input coefficient has at most 38 digits and therefore occupies at most two limbs.
  for (int32_t i = 0; i < 2; ++i) {
    __uint128_t carry = 0;
    for (int32_t j = 0; j < 2; ++j) {
      auto const k = i + j;
      auto value = static_cast<__uint128_t>(lhs.limbs[i]) * rhs.limbs[j] + product.limbs[k] + carry;
      product.limbs[k] = static_cast<uint64_t>(value % limb_base);
      carry            = value / limb_base;
    }
    auto k = i + 2;
    while (carry != 0 && k < limb_count) {
      auto value       = static_cast<__uint128_t>(product.limbs[k]) + carry;
      product.limbs[k] = static_cast<uint64_t>(value % limb_base);
      carry            = value / limb_base;
      ++k;
    }
    valid = valid && carry == 0;
  }
  return product;
}

__device__ inline uint64_t divide_small(wide_uint& value, uint64_t divisor)
{
  __uint128_t remainder = 0;
  for (int32_t i = limb_count - 1; i >= 0; --i) {
    auto current   = remainder * limb_base + value.limbs[i];
    value.limbs[i] = static_cast<uint64_t>(current / divisor);
    remainder      = current % divisor;
  }
  return static_cast<uint64_t>(remainder);
}

__device__ inline int32_t decimal_digits(wide_uint value)
{
  if (is_zero(value)) { return 1; }
  int32_t digits = 0;
  while (!is_zero(value)) {
    divide_small(value, 10);
    ++digits;
  }
  return digits;
}

__device__ inline bool valid_input(__int128_t coefficient, int16_t exponent)
{
  if (coefficient == 0) { return true; }
  auto value  = from_u128(magnitude(coefficient));
  auto digits = decimal_digits(value);
  if (digits > precision) { return false; }
  auto adjusted_exponent = static_cast<int32_t>(exponent) + digits - 1;
  return adjusted_exponent >= -130 && adjusted_exponent <= 125;
}

__device__ inline __uint128_t to_u128(wide_uint const& value, bool& valid)
{
  valid = value.limbs[2] == 0 && value.limbs[3] == 0 && value.limbs[4] == 0;
  return static_cast<__uint128_t>(value.limbs[1]) * limb_base + value.limbs[0];
}

__device__ inline result normalize(bool negative, wide_uint value, int32_t exponent)
{
  if (is_zero(value)) { return {}; }

  auto digits = decimal_digits(value);
  if (digits > precision) {
    auto const discarded_digits = digits - precision;
    uint64_t rounding_digit     = 0;
    for (int32_t i = 0; i < discarded_digits; ++i) {
      rounding_digit = divide_small(value, 10);
    }
    exponent += discarded_digits;

    // The Oracle NUMBER-inspired arithmetic contract rounds decimal ties away from zero.
    if (rounding_digit >= 5) {
      auto one = from_u128(1);
      if (!add_in_place(value, one)) { return {.error = 1}; }
    }

    if (decimal_digits(value) > precision) {
      divide_small(value, 10);
      ++exponent;
    }
  }

  while (true) {
    auto candidate = value;
    if (divide_small(candidate, 10) != 0) { break; }
    value = candidate;
    ++exponent;
  }

  digits                       = decimal_digits(value);
  auto const adjusted_exponent = exponent + digits - 1;
  if (adjusted_exponent < -130) { return {}; }
  if (adjusted_exponent > 125) { return {.error = 1}; }

  bool conversion_valid = false;
  auto coefficient      = to_u128(value, conversion_valid);
  if (!conversion_valid) { return {.error = 1}; }

  auto const signed_coefficient =
    negative ? -static_cast<__int128_t>(coefficient) : static_cast<__int128_t>(coefficient);
  return {signed_coefficient, static_cast<int16_t>(exponent), 0};
}

__device__ inline result add(__int128_t lhs_coefficient,
                             int16_t lhs_exponent,
                             __int128_t rhs_coefficient,
                             int16_t rhs_exponent)
{
  if (!valid_input(lhs_coefficient, lhs_exponent) || !valid_input(rhs_coefficient, rhs_exponent)) {
    return {.error = 1};
  }
  if (lhs_coefficient == 0) {
    return normalize(rhs_coefficient < 0, from_u128(magnitude(rhs_coefficient)), rhs_exponent);
  }
  if (rhs_coefficient == 0) {
    return normalize(lhs_coefficient < 0, from_u128(magnitude(lhs_coefficient)), lhs_exponent);
  }

  auto lhs = from_u128(magnitude(lhs_coefficient));
  auto rhs = from_u128(magnitude(rhs_coefficient));

  auto const lhs_adjusted = static_cast<int32_t>(lhs_exponent) + decimal_digits(lhs) - 1;
  auto const rhs_adjusted = static_cast<int32_t>(rhs_exponent) + decimal_digits(rhs) - 1;

  if (lhs_adjusted - rhs_adjusted > precision) {
    return normalize(lhs_coefficient < 0, lhs, lhs_exponent);
  }
  if (rhs_adjusted - lhs_adjusted > precision) {
    return normalize(rhs_coefficient < 0, rhs, rhs_exponent);
  }

  auto const common_exponent = lhs_exponent < rhs_exponent ? static_cast<int32_t>(lhs_exponent)
                                                           : static_cast<int32_t>(rhs_exponent);
  if (!multiply_pow10(lhs, static_cast<int32_t>(lhs_exponent) - common_exponent) ||
      !multiply_pow10(rhs, static_cast<int32_t>(rhs_exponent) - common_exponent)) {
    return {.error = 1};
  }

  auto negative = false;
  if ((lhs_coefficient < 0) == (rhs_coefficient < 0)) {
    if (!add_in_place(lhs, rhs)) { return {.error = 1}; }
    negative = lhs_coefficient < 0;
  } else {
    auto const comparison = compare(lhs, rhs);
    if (comparison == 0) { return {}; }
    if (comparison > 0) {
      subtract_in_place(lhs, rhs);
      negative = lhs_coefficient < 0;
    } else {
      subtract_in_place(rhs, lhs);
      lhs      = rhs;
      negative = rhs_coefficient < 0;
    }
  }
  return normalize(negative, lhs, common_exponent);
}

__device__ inline result multiply(__int128_t lhs_coefficient,
                                  int16_t lhs_exponent,
                                  __int128_t rhs_coefficient,
                                  int16_t rhs_exponent)
{
  if (!valid_input(lhs_coefficient, lhs_exponent) || !valid_input(rhs_coefficient, rhs_exponent)) {
    return {.error = 1};
  }
  if (lhs_coefficient == 0 || rhs_coefficient == 0) { return {}; }

  bool product_valid = false;
  auto product       = multiply(
    from_u128(magnitude(lhs_coefficient)), from_u128(magnitude(rhs_coefficient)), product_valid);
  if (!product_valid) { return {.error = 1}; }
  return normalize((lhs_coefficient < 0) != (rhs_coefficient < 0),
                   product,
                   static_cast<int32_t>(lhs_exponent) + rhs_exponent);
}

}  // namespace oracle_number_inspired
