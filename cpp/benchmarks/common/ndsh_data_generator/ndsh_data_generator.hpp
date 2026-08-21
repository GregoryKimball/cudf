/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <functional>

namespace CUDF_EXPORT cudf {
namespace datagen {

/**
 * @brief Deterministic state shared by staged `orders` and `lineitem` generation
 */
struct orders_lineitem_state {
  double scale_factor;
};

/**
 * @brief Generate the shared state required to materialize `orders` and `lineitem`
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned state
 */
orders_lineitem_state generate_orders_lineitem_state(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Materialize `orders` without retaining generated `lineitem` columns
 *
 * @param state State returned by `generate_orders_lineitem_state`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table
 */
std::unique_ptr<cudf::table> materialize_orders(
  orders_lineitem_state& state,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Materialize `lineitem` by deterministically regenerating its core columns
 *
 * @param state State returned by `generate_orders_lineitem_state`
 * @param include_lineitem_comment Whether to generate the `l_comment` column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table
 */
std::unique_ptr<cudf::table> materialize_lineitem(
  orders_lineitem_state&& state,
  bool include_lineitem_comment     = false,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Materialize and consume bounded-size `lineitem` partitions
 *
 * @param state State returned by `generate_orders_lineitem_state`
 * @param consumer Function invoked once for each generated partition
 * @param include_lineitem_comment Whether to generate the `l_comment` column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate generated tables
 */
void materialize_lineitem_partitions(
  orders_lineitem_state&& state,
  std::function<void(std::unique_ptr<cudf::table>)> const& consumer,
  bool include_lineitem_comment     = false,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate the `part` table independently
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table
 */
std::unique_ptr<cudf::table> generate_part(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate the `orders`, `lineitem`, and `part` tables
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param include_lineitem_comment Whether to generate the `l_comment` column
 */
std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
generate_orders_lineitem_part(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref(),
  bool include_lineitem_comment     = false);

/**
 * @brief Generate the `partsupp` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_partsupp(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate the `supplier` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_supplier(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate the `customer` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_customer(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate the `nation` table
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_nation(
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate the `region` table
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_region(
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace datagen
}  // namespace CUDF_EXPORT cudf
