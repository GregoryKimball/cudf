/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::experimental {

/**
 * @file
 * @brief Feather (Arrow IPC file format, version 2) reader and writer APIs
 *
 * Supported column types are integers, floating point, booleans, decimals (32, 64 and 128 bit),
 * timestamps, durations and strings, with or without nulls. Nested and dictionary-encoded columns
 * are not supported.
 *
 * Compressed bodies are decoded and encoded on the GPU with nvCOMP. `LZ4_FRAME` buffers must use
 * independent blocks (or a single block); linked-block frames, which Arrow C++ produces by default
 * for buffers larger than 64KB, are rejected.
 */

/**
 * @addtogroup io_writers
 * @{
 */

class feather_writer_options_builder;

/**
 * @brief Settings for `write_feather()`.
 */
class feather_writer_options {
  sink_info _sink;
  table_view _table;
  std::vector<std::string> _column_names;
  std::vector<column_name_info> _column_metadata;
  compression_type _compression = compression_type::LZ4;
  size_type _max_rows_per_batch = std::numeric_limits<size_type>::max();

  friend feather_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit feather_writer_options(sink_info sink, table_view table)
    : _sink(std::move(sink)), _table(std::move(table))
  {
  }

 public:
  /**
   * @brief Create builder to create `feather_writer_options`.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build feather_writer_options
   */
  static feather_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink used for writer output.
   *
   * @return sink used for writer output
   */
  [[nodiscard]] sink_info const& get_sink() const noexcept { return _sink; }

  /**
   * @brief Returns table that would be written to output.
   *
   * @return Table that would be written to output
   */
  [[nodiscard]] table_view const& get_table() const noexcept { return _table; }

  /**
   * @brief Returns the column names; empty means columns are named by their index.
   *
   * @return Column names
   */
  [[nodiscard]] std::vector<std::string> const& get_column_names() const noexcept
  {
    return _column_names;
  }

  /**
   * @brief Returns the names of the columns and their children; takes precedence over the column
   * names when not empty.
   *
   * @return Column name hierarchy, one entry per column
   */
  [[nodiscard]] std::vector<column_name_info> const& get_column_metadata() const noexcept
  {
    return _column_metadata;
  }

  /**
   * @brief Returns the buffer compression codec.
   *
   * @return Compression codec, one of `NONE`, `LZ4` or `ZSTD`
   */
  [[nodiscard]] compression_type get_compression() const noexcept { return _compression; }

  /**
   * @brief Returns the maximum number of rows in each record batch.
   *
   * @return Maximum number of rows per record batch
   */
  [[nodiscard]] size_type get_max_rows_per_batch() const noexcept { return _max_rows_per_batch; }

  /**
   * @brief Sets sink info.
   *
   * @param sink The sink info.
   */
  void set_sink(sink_info sink) { _sink = std::move(sink); }

  /**
   * @brief Sets column names.
   *
   * @param names One name per column in the table
   */
  void set_column_names(std::vector<std::string> names) { _column_names = std::move(names); }

  /**
   * @brief Sets the names of the columns and their children.
   *
   * Struct children are named by the entries of `children`; a list's entry holds the offsets
   * name followed by the child's names, as in the `table_metadata` returned by readers. Missing
   * child names default to the child index for struct fields and to "item" for list elements.
   *
   * @param metadata One entry per column in the table
   */
  void set_column_metadata(std::vector<column_name_info> metadata)
  {
    _column_metadata = std::move(metadata);
  }

  /**
   * @brief Sets the buffer compression codec.
   *
   * @throws cudf::logic_error if the codec is not `NONE`, `LZ4` or `ZSTD`
   *
   * @param compression Compression codec
   */
  void set_compression(compression_type compression);

  /**
   * @brief Sets the maximum number of rows in each record batch.
   *
   * @throws cudf::logic_error if `rows` is not positive
   *
   * @param rows Maximum number of rows per record batch
   */
  void set_max_rows_per_batch(size_type rows);
};

/**
 * @brief Class to build `feather_writer_options`.
 */
class feather_writer_options_builder {
 public:
  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit feather_writer_options_builder(sink_info const& sink, table_view const& table)
    : _options(sink, table)
  {
  }

  /**
   * @brief Sets column names.
   *
   * @param names One name per column in the table
   * @return this for chaining
   */
  feather_writer_options_builder& column_names(std::vector<std::string> names)
  {
    _options.set_column_names(std::move(names));
    return *this;
  }

  /**
   * @brief Sets the names of the columns and their children.
   *
   * @param metadata One entry per column in the table; see
   * `feather_writer_options::set_column_metadata`
   * @return this for chaining
   */
  feather_writer_options_builder& column_metadata(std::vector<column_name_info> metadata)
  {
    _options.set_column_metadata(std::move(metadata));
    return *this;
  }

  /**
   * @brief Sets the buffer compression codec.
   *
   * @param compression Compression codec, one of `NONE`, `LZ4` or `ZSTD`
   * @return this for chaining
   */
  feather_writer_options_builder& compression(compression_type compression)
  {
    _options.set_compression(compression);
    return *this;
  }

  /**
   * @brief Sets the maximum number of rows in each record batch.
   *
   * @param rows Maximum number of rows per record batch
   * @return this for chaining
   */
  feather_writer_options_builder& max_rows_per_batch(size_type rows)
  {
    _options.set_max_rows_per_batch(rows);
    return *this;
  }

  /**
   * @brief Build `feather_writer_options`.
   *
   * @return The constructed `feather_writer_options` object
   */
  [[nodiscard]] feather_writer_options build() const { return _options; }

 private:
  feather_writer_options _options;
};

/** @} */  // end of io_writers group

/**
 * @addtogroup io_readers
 * @{
 */

class feather_reader_options_builder;

/**
 * @brief Settings for `read_feather()`.
 */
class feather_reader_options {
  source_info _source;
  std::optional<std::vector<std::string>> _columns;

  friend feather_reader_options_builder;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read feather file
   */
  explicit feather_reader_options(source_info src) : _source{std::move(src)} {}

 public:
  /**
   * @brief Creates a `feather_reader_options_builder` to build `feather_reader_options`.
   *
   * @param src Source information to read feather file
   * @return Builder to build reader options
   */
  static feather_reader_options_builder builder(source_info src = source_info{});

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const noexcept { return _source; }

  /**
   * @brief Returns the names of the columns to read, if set.
   *
   * @return Column names to read; `std::nullopt` reads all columns
   */
  [[nodiscard]] auto const& get_columns() const noexcept { return _columns; }

  /**
   * @brief Sets source info.
   *
   * @param src The source info.
   */
  void set_source(source_info src) { _source = std::move(src); }

  /**
   * @brief Sets the names of the columns to read, in output order.
   *
   * @param col_names Column names
   */
  void set_columns(std::vector<std::string> col_names) { _columns = std::move(col_names); }
};

/**
 * @brief Class to build `feather_reader_options`.
 */
class feather_reader_options_builder {
 public:
  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read feather file
   */
  explicit feather_reader_options_builder(source_info src) : _options(std::move(src)) {}

  /**
   * @brief Sets the names of the columns to read, in output order.
   *
   * @param col_names Column names
   * @return this for chaining
   */
  feather_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    _options.set_columns(std::move(col_names));
    return *this;
  }

  /**
   * @brief Build `feather_reader_options`.
   *
   * @return The constructed `feather_reader_options` object
   */
  [[nodiscard]] feather_reader_options build() const { return _options; }

 private:
  feather_reader_options _options;
};

/** @} */  // end of io_readers group

/**
 * @addtogroup io_writers
 * @{
 */

/**
 * @brief Write a table in the Feather (Arrow IPC file) format.
 *
 * The table is split into record batches of at most `max_rows_per_batch` rows. When compression
 * is enabled, each buffer is compressed independently with nvCOMP; buffers that do not shrink are
 * stored uncompressed, as permitted by the Arrow IPC format.
 *
 * @throws cudf::logic_error if the table contains an unsupported column type
 * @throws cudf::logic_error if nvCOMP is unavailable for the requested compression
 *
 * @param options Options specifying the sink, table, and encoding settings
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void write_feather(feather_writer_options const& options,
                   cuda::stream_ref stream = cudf::get_default_stream());

/** @} */  // end of group

/**
 * @addtogroup io_readers
 * @{
 */

/**
 * @brief Read a Feather (Arrow IPC file) into a table.
 *
 * Record batches are decoded independently and concatenated into a single table.
 *
 * @throws cudf::logic_error if the file contains an unsupported column type or encoding
 * @throws cudf::logic_error if nvCOMP cannot decompress a buffer
 *
 * @param options Options specifying the source and columns to read
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resources used for the returned table and temporary allocations
 * @return The table and its column names
 */
table_with_metadata read_feather(
  feather_reader_options const& options,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace io::experimental
}  // namespace CUDF_EXPORT cudf
