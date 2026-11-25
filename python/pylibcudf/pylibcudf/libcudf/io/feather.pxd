# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cimport pylibcudf.libcudf.io.types as cudf_io_types
cimport pylibcudf.libcudf.table.table_view as cudf_table_view
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/io/experimental/feather.hpp" \
        namespace "cudf::io::experimental" nogil:

    cdef cppclass feather_writer_options:
        feather_writer_options() except +libcudf_exception_handler
        cudf_io_types.sink_info get_sink() except +libcudf_exception_handler
        cudf_table_view.table_view get_table() except +libcudf_exception_handler
        vector[string] get_column_names() except +libcudf_exception_handler
        cudf_io_types.compression_type get_compression() \
            except +libcudf_exception_handler
        size_type get_max_rows_per_batch() except +libcudf_exception_handler

        void set_column_names(vector[string] names) except +libcudf_exception_handler
        void set_column_metadata(
            vector[cudf_io_types.column_name_info] metadata
        ) except +libcudf_exception_handler
        void set_compression(
            cudf_io_types.compression_type compression
        ) except +libcudf_exception_handler
        void set_max_rows_per_batch(size_type rows) except +libcudf_exception_handler

        @staticmethod
        feather_writer_options_builder builder(
            cudf_io_types.sink_info &sink,
            cudf_table_view.table_view &table
        ) except +libcudf_exception_handler

    cdef cppclass feather_writer_options_builder:
        feather_writer_options_builder& column_names(
            vector[string] names
        ) except +libcudf_exception_handler
        feather_writer_options_builder& column_metadata(
            vector[cudf_io_types.column_name_info] metadata
        ) except +libcudf_exception_handler
        feather_writer_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except +libcudf_exception_handler
        feather_writer_options_builder& max_rows_per_batch(
            size_type rows
        ) except +libcudf_exception_handler
        feather_writer_options build() except +libcudf_exception_handler

    cdef void write_feather(
        feather_writer_options options,
        cudaStream_t stream,
    ) except +libcudf_exception_handler

    cdef cppclass feather_reader_options:
        feather_reader_options() except +libcudf_exception_handler
        cudf_io_types.source_info get_source() except +libcudf_exception_handler
        optional[vector[string]] get_columns() except +libcudf_exception_handler

        void set_source(cudf_io_types.source_info src) except +libcudf_exception_handler
        void set_columns(vector[string] col_names) except +libcudf_exception_handler

        @staticmethod
        feather_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +libcudf_exception_handler

    cdef cppclass feather_reader_options_builder:
        feather_reader_options_builder& columns(
            vector[string] col_names
        ) except +libcudf_exception_handler
        feather_reader_options build() except +libcudf_exception_handler

    cdef cudf_io_types.table_with_metadata read_feather(
        feather_reader_options options,
        cudaStream_t stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler
