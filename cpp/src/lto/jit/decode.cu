/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/lto/udf_abi.h>

// One block per chunk; the codec fragment's decoder is block-cooperative.
extern "C" __global__ void cudf_kernel_entry(cudf_lto_codec_job const* __restrict__ jobs,
                                             cudf_lto_u64 num_jobs,
                                             int* __restrict__ statuses)
{
  __shared__ __align__(16) unsigned char scratch[CUDF_LTO_SCRATCH_BYTES];
  for (cudf_lto_u64 j = blockIdx.x; j < num_jobs; j += gridDim.x) {
    cudf_lto_codec_job const job = jobs[j];
    auto const status            = cudf_lto_decode_chunk(job.input,
                                              static_cast<cudf_lto_u32>(job.input_bytes),
                                              job.element_bytes,
                                              job.output,
                                              static_cast<cudf_lto_u32>(job.output_bytes),
                                              scratch);
    if (threadIdx.x == 0) { statuses[j] = status; }
    __syncthreads();
  }
}
