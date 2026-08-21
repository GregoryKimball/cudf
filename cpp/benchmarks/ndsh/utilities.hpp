/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>

#include <rmm/device_uvector.hpp>

#include <string>
#include <unordered_map>
#include <vector>

enum class join_algorithm { HASH, DIRECT };

enum class direct_join_build_side { LEFT, RIGHT };

/**
 * @brief Host-backed Parquet source populated from an NDS-H device sink
 */
class ndsh_parquet_source {
 public:
  ndsh_parquet_source() = default;
  ~ndsh_parquet_source();

  ndsh_parquet_source(ndsh_parquet_source&& other) noexcept;
  ndsh_parquet_source& operator=(ndsh_parquet_source&& other) noexcept;

  ndsh_parquet_source(ndsh_parquet_source const&)            = delete;
  ndsh_parquet_source& operator=(ndsh_parquet_source const&) = delete;

  [[nodiscard]] cudf::io::source_info make_source_info() const;

  void append_from_device(void const* device_data, std::size_t size, rmm::cuda_stream_view stream);

 private:
  struct pinned_buffer {
    void* data;
    std::size_t size;
  };

  std::vector<pinned_buffer> buffers_;
};

using ndsh_data_sources = std::unordered_map<std::string, ndsh_parquet_source>;

/**
 * @brief Convert a benchmark axis value to a join algorithm
 */
[[nodiscard]] join_algorithm parse_join_algorithm(std::string const& value);

/**
 * @brief A class to represent a table with column names attached
 */
class table_with_names {
 public:
  table_with_names(std::unique_ptr<cudf::table> tbl, std::vector<std::string> col_names)
    : tbl(std::move(tbl)), col_names(col_names) {};
  /**
   * @brief Return the table view
   */
  [[nodiscard]] cudf::table_view table() const;
  /**
   * @brief Return the column view for a given column name
   *
   * @param col_name The name of the column
   */
  [[nodiscard]] cudf::column_view column(std::string const& col_name) const;
  /**
   * @param Return the column names of the table
   */
  [[nodiscard]] std::vector<std::string> const& column_names() const;
  /**
   * @brief Translate a column name to a column index
   *
   * @param col_name The name of the column
   */
  [[nodiscard]] cudf::size_type column_id(std::string const& col_name) const;
  /**
   * @brief Append a column to the table
   *
   * @param col The column to append
   * @param col_name The name of the appended column
   */
  table_with_names& append(std::unique_ptr<cudf::column>& col, std::string const& col_name);
  /**
   * @brief Select a subset of columns from the table
   *
   * @param col_names The names of the columns to select
   */
  [[nodiscard]] cudf::table_view select(std::vector<std::string> const& col_names) const;
  /**
   * @brief Write the table to a parquet file
   *
   * @param filepath The path to the parquet file
   */
  void to_parquet(std::string const& filepath) const;

 private:
  std::unique_ptr<cudf::table> tbl;
  std::vector<std::string> col_names;
};

/**
 * @brief Inner join two tables and gather the result
 *
 * @param left_input The left input table
 * @param right_input The right input table
 * @param left_on The columns to join on in the left table
 * @param right_on The columns to join on in the right table
 * @param compare_nulls The null equality policy
 * @param algorithm The join implementation to use
 * @param direct_build_side The input with distinct keys when using direct join
 * @param direct_capacity The known exclusive upper bound for direct-join keys
 */
[[nodiscard]] std::unique_ptr<cudf::table> join_and_gather(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls,
  join_algorithm algorithm                 = join_algorithm::HASH,
  direct_join_build_side direct_build_side = direct_join_build_side::RIGHT,
  std::size_t direct_capacity              = 0);

/**
 * @brief Apply an inner join operation to two tables
 *
 * @param left_input The left input table
 * @param right_input The right input table
 * @param left_on The columns to join on in the left table
 * @param right_on The columns to join on in the right table
 * @param compare_nulls The null equality policy
 * @param algorithm The join implementation to use
 * @param direct_build_side The input with distinct keys when using direct join
 * @param direct_capacity The known exclusive upper bound for direct-join keys
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_inner_join(
  std::unique_ptr<table_with_names> const& left_input,
  std::unique_ptr<table_with_names> const& right_input,
  std::vector<std::string> const& left_on,
  std::vector<std::string> const& right_on,
  cudf::null_equality compare_nulls        = cudf::null_equality::EQUAL,
  join_algorithm algorithm                 = join_algorithm::HASH,
  direct_join_build_side direct_build_side = direct_join_build_side::RIGHT,
  std::size_t direct_capacity              = 0);

/**
 * @brief Apply a filter predicate to a table
 *
 * @param table The input table
 * @param predicate The filter predicate
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_filter(
  std::unique_ptr<table_with_names> const& table, cudf::ast::operation const& predicate);

/**
 * @brief Apply a boolean mask to a table
 *
 * @param table The input table
 * @param mask The boolean mask
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_mask(
  std::unique_ptr<table_with_names> const& table, std::unique_ptr<cudf::column> const& mask);

/**
 * Struct representing group by key columns, value columns, and the type of aggregations to perform
 * on the value columns
 */
struct groupby_context_t {
  std::vector<std::string> keys;
  std::unordered_map<std::string, std::vector<std::pair<cudf::aggregation::Kind, std::string>>>
    values;
};

/**
 * @brief Apply a groupby operation to a table
 *
 * @param table The input table
 * @param ctx The groupby context
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_groupby(
  std::unique_ptr<table_with_names> const& table, groupby_context_t const& ctx);

/**
 * @brief Apply an order by operation to a table
 *
 * @param table The input table
 * @param sort_keys The sort keys
 * @param sort_key_orders The sort key orders
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_orderby(
  std::unique_ptr<table_with_names> const& table,
  std::vector<std::string> const& sort_keys,
  std::vector<cudf::order> const& sort_key_orders);

/**
 * @brief Apply a reduction operation to a column
 *
 * @param column The input column
 * @param agg_kind The aggregation kind
 * @param col_name The name of the output column
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_reduction(
  cudf::column_view const& column,
  cudf::aggregation::Kind const& agg_kind,
  std::string const& col_name);

/**
 * @brief Read a parquet file into a table
 *
 * @param source_info The source of the parquet file
 * @param columns The columns to read
 * @param predicate The filter predicate to pushdown
 */
[[nodiscard]] std::unique_ptr<table_with_names> read_parquet(
  cudf::io::source_info const& source_info,
  std::vector<std::string> const& columns                = {},
  std::unique_ptr<cudf::ast::operation> const& predicate = nullptr);

/**
 * @brief Generate the `std::tm` structure from year, month, and day
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
std::tm make_tm(int year, int month, int day);

/**
 * @brief Calculate the number of days since the UNIX epoch
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
int32_t days_since_epoch(int year, int month, int day);

/**
 * @brief Write a `cudf::table` to a parquet cuio sink
 *
 * @param table The `cudf::table` to write
 * @param col_names The column names of the table
 * @param source The source sink pair to write the table to
 */
void write_to_parquet_device_buffer(std::unique_ptr<cudf::table> const& table,
                                    std::vector<std::string> const& col_names,
                                    ndsh_parquet_source& source);

/**
 * @brief Generate NDS-H tables and write to parquet device buffers
 *
 * @param scale_factor The scale factor of NDS-H tables to generate
 * @param table_names The names of the tables to generate
 * @param sources The parquet data sources to populate
 * @param include_lineitem_comment Whether to generate the `l_comment` column
 * @param use_managed_memory Whether to use a managed pool for data generation
 */
void generate_parquet_data_sources(double scale_factor,
                                   std::vector<std::string> const& table_names,
                                   ndsh_data_sources& sources,
                                   bool include_lineitem_comment = false,
                                   bool use_managed_memory       = true);
