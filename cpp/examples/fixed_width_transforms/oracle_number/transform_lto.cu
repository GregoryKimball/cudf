/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "oracle_number.cuh"

#include <cudf/fixed_point/fixed_point.hpp>

#include <cuda/std/cstdint>

extern "C" __device__ int transform(numeric::decimal128* out_coefficient,
                                    cuda::std::int16_t* out_exponent,
                                    numeric::decimal128 lhs_coefficient,
                                    cuda::std::int16_t lhs_exponent,
                                    numeric::decimal128 rhs_coefficient,
                                    cuda::std::int16_t rhs_exponent,
                                    cuda::std::int8_t operation)
{
  if (operation < 0 || operation > 1) { return 1; }
  auto const value =
    operation == 0
      ? oracle_number_inspired::add(
          lhs_coefficient.value(), lhs_exponent, rhs_coefficient.value(), rhs_exponent)
      : oracle_number_inspired::multiply(
          lhs_coefficient.value(), lhs_exponent, rhs_coefficient.value(), rhs_exponent);

  *out_coefficient = numeric::decimal128{value.coefficient, numeric::scale_type{0}};
  *out_exponent    = value.exponent;
  return value.error;
}
