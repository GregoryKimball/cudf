# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.io.types cimport SinkInfo, SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.feather cimport (
    feather_reader_options,
    feather_reader_options_builder,
    feather_writer_options,
    feather_writer_options_builder,
)
from pylibcudf.libcudf.io.types cimport compression_type
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table


cdef class FeatherWriterOptions:
    cdef feather_writer_options c_obj
    cdef Table table
    cdef SinkInfo sink
    cpdef void set_column_names(self, list names)
    cpdef void set_compression(self, compression_type compression)
    cpdef void set_max_rows_per_batch(self, size_type rows)

cdef class FeatherWriterOptionsBuilder:
    cdef feather_writer_options_builder c_obj
    cdef Table table
    cdef SinkInfo sink
    cpdef FeatherWriterOptionsBuilder column_names(self, list names)
    cpdef FeatherWriterOptionsBuilder compression(self, compression_type compression)
    cpdef FeatherWriterOptionsBuilder max_rows_per_batch(self, size_type rows)
    cpdef FeatherWriterOptions build(self)

cpdef void write_feather(FeatherWriterOptions options, object stream = *)

cdef class FeatherReaderOptions:
    cdef feather_reader_options c_obj
    cdef SourceInfo source
    cpdef void set_columns(self, list col_names)
    cpdef void set_source(self, SourceInfo src)

cdef class FeatherReaderOptionsBuilder:
    cdef feather_reader_options_builder c_obj
    cdef SourceInfo source
    cpdef FeatherReaderOptionsBuilder columns(self, list col_names)
    cpdef FeatherReaderOptions build(self)

cpdef TableWithMetadata read_feather(
    FeatherReaderOptions options, object stream = *, DeviceMemoryResource mr=*
)
