# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.io.types cimport SinkInfo, SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.feather cimport (
    feather_reader_options,
    feather_writer_options,
    read_feather as cpp_read_feather,
    write_feather as cpp_write_feather,
)
from pylibcudf.libcudf.io.types cimport column_name_info, compression_type
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table

from pylibcudf.utils cimport _get_stream, _get_memory_resource
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike


__all__ = [
    "FeatherReaderOptions",
    "FeatherReaderOptionsBuilder",
    "FeatherWriterOptions",
    "FeatherWriterOptionsBuilder",
    "read_feather",
    "write_feather",
]


cdef vector[string] _encode_names(list names):
    cdef vector[string] vec
    vec.reserve(len(names))
    for name in names:
        vec.push_back(str(name).encode())
    return vec


cdef vector[column_name_info] _make_column_info(list column_names):
    cdef vector[column_name_info] infos
    cdef column_name_info info
    infos.reserve(len(column_names))
    for name, child_names in column_names:
        info.name = str(name).encode()
        info.children = _make_column_info(list(child_names))
        infos.push_back(info)
    return infos


cdef class FeatherWriterOptions:
    """
    The settings to use for ``write_feather``.

    For details, see :cpp:class:`cudf::io::experimental::feather_writer_options`
    """
    @staticmethod
    def builder(SinkInfo sink, Table table) -> FeatherWriterOptionsBuilder:
        """
        Create builder to create FeatherWriterOptions.

        For details, see
        :cpp:func:`cudf::io::experimental::feather_writer_options::builder`

        Parameters
        ----------
        sink : SinkInfo
            The sink used for writer output
        table : Table
            Table to be written to output

        Returns
        -------
        FeatherWriterOptionsBuilder
        """
        cdef FeatherWriterOptionsBuilder feather_builder = (
            FeatherWriterOptionsBuilder.__new__(FeatherWriterOptionsBuilder)
        )
        feather_builder.c_obj = feather_writer_options.builder(sink.c_obj, table.view())
        feather_builder.table = table
        feather_builder.sink = sink
        return feather_builder

    cpdef void set_column_names(self, list names):
        """
        Sets the column names written to the file schema.

        Parameters
        ----------
        names : list[str]
            One name per column

        Returns
        -------
        None
        """
        self.c_obj.set_column_names(_encode_names(names))

    cpdef void set_column_metadata(self, list column_names):
        """
        Sets the names of the columns and their children, taking precedence
        over the column names.

        Parameters
        ----------
        column_names : list[tuple[str, list]]
            One ``(name, children)`` entry per column, where ``children`` has
            the same form. A list's children are the offsets name followed by
            the element, as in ``TableWithMetadata``; for example
            ``[("id", []), ("point", [("x", []), ("y", [])])]``.

        Returns
        -------
        None
        """
        self.c_obj.set_column_metadata(_make_column_info(column_names))

    cpdef void set_compression(self, compression_type compression):
        """
        Sets the buffer compression codec: NONE, LZ4 (LZ4 frame) or ZSTD.

        Parameters
        ----------
        compression : CompressionType
            The compression type to use

        Returns
        -------
        None
        """
        self.c_obj.set_compression(compression)

    cpdef void set_max_rows_per_batch(self, size_type rows):
        """
        Sets the maximum number of rows in each record batch.

        Parameters
        ----------
        rows : int
            Maximum rows per record batch

        Returns
        -------
        None
        """
        self.c_obj.set_max_rows_per_batch(rows)


cdef class FeatherWriterOptionsBuilder:
    """Builder to build options for ``write_feather``."""

    cpdef FeatherWriterOptionsBuilder column_names(self, list names):
        """
        Sets the column names written to the file schema.

        Parameters
        ----------
        names : list[str]
            One name per column

        Returns
        -------
        FeatherWriterOptionsBuilder
        """
        self.c_obj.column_names(_encode_names(names))
        return self

    cpdef FeatherWriterOptionsBuilder column_metadata(self, list column_names):
        """
        Sets the names of the columns and their children, taking precedence
        over the column names.

        Parameters
        ----------
        column_names : list[tuple[str, list]]
            One ``(name, children)`` entry per column; see
            ``FeatherWriterOptions.set_column_metadata``

        Returns
        -------
        FeatherWriterOptionsBuilder
        """
        self.c_obj.column_metadata(_make_column_info(column_names))
        return self

    cpdef FeatherWriterOptionsBuilder compression(self, compression_type compression):
        """
        Sets the buffer compression codec: NONE, LZ4 (LZ4 frame) or ZSTD.

        Parameters
        ----------
        compression : CompressionType
            The compression type to use

        Returns
        -------
        FeatherWriterOptionsBuilder
        """
        self.c_obj.compression(compression)
        return self

    cpdef FeatherWriterOptionsBuilder max_rows_per_batch(self, size_type rows):
        """
        Sets the maximum number of rows in each record batch.

        Parameters
        ----------
        rows : int
            Maximum rows per record batch

        Returns
        -------
        FeatherWriterOptionsBuilder
        """
        self.c_obj.max_rows_per_batch(rows)
        return self

    cpdef FeatherWriterOptions build(self):
        """Create a FeatherWriterOptions object"""
        cdef FeatherWriterOptions feather_options = FeatherWriterOptions.__new__(
            FeatherWriterOptions
        )
        feather_options.c_obj = move(self.c_obj.build())
        feather_options.table = self.table
        feather_options.sink = self.sink
        return feather_options


cpdef void write_feather(
    FeatherWriterOptions options,
    object stream: CudaStreamLike | None = None,
):
    """
    Write to the Feather V2 (Arrow IPC file) format.

    The table to write, output paths, and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`cudf::io::experimental::write_feather`.

    Parameters
    ----------
    options : FeatherWriterOptions
        Settings for controlling writing behavior
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches

    Returns
    -------
    None
    """
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().get()
    with nogil:
        cpp_write_feather(move(options.c_obj), _cs)


cdef class FeatherReaderOptions:
    """
    The settings to use for ``read_feather``.

    For details, see :cpp:class:`cudf::io::experimental::feather_reader_options`
    """
    @staticmethod
    def builder(SourceInfo source) -> FeatherReaderOptionsBuilder:
        """
        Create a FeatherReaderOptionsBuilder object.

        For details, see
        :cpp:func:`cudf::io::experimental::feather_reader_options::builder`

        Parameters
        ----------
        source : SourceInfo
            The source to read the Feather file from.

        Returns
        -------
        FeatherReaderOptionsBuilder
        """
        cdef FeatherReaderOptionsBuilder feather_builder = (
            FeatherReaderOptionsBuilder.__new__(FeatherReaderOptionsBuilder)
        )
        feather_builder.c_obj = feather_reader_options.builder(source.c_obj)
        feather_builder.source = source
        return feather_builder

    cpdef void set_columns(self, list col_names):
        """
        Sets the names of the columns to read.

        Parameters
        ----------
        col_names : list[str]
            List of column names

        Returns
        -------
        None
        """
        self.c_obj.set_columns(_encode_names(col_names))

    cpdef void set_source(self, SourceInfo src):
        """
        Set a new source info location.

        Parameters
        ----------
        src : SourceInfo
            New source information, replacing existing information.

        Returns
        -------
        None
        """
        self.c_obj.set_source(src.c_obj)
        self.source = src


cdef class FeatherReaderOptionsBuilder:
    """Builder to build options for ``read_feather``."""

    cpdef FeatherReaderOptionsBuilder columns(self, list col_names):
        """
        Sets the names of the columns to read.

        Parameters
        ----------
        col_names : list[str]
            List of column names

        Returns
        -------
        FeatherReaderOptionsBuilder
        """
        self.c_obj.columns(_encode_names(col_names))
        return self

    cpdef FeatherReaderOptions build(self):
        """Create a FeatherReaderOptions object"""
        cdef FeatherReaderOptions feather_options = FeatherReaderOptions.__new__(
            FeatherReaderOptions
        )
        feather_options.c_obj = move(self.c_obj.build())
        feather_options.source = self.source
        return feather_options


cpdef TableWithMetadata read_feather(
    FeatherReaderOptions options,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """
    Read from the Feather V2 (Arrow IPC file) format.

    The source to read from and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`cudf::io::experimental::read_feather`.

    Parameters
    ----------
    options : FeatherReaderOptions
        Settings for controlling reading behavior
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches
    mr : DeviceMemoryResource, optional
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata (column names) that were read in.
    """
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().get()
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = move(cpp_read_feather(options.c_obj, _cs, mr.get_mr()))

    return TableWithMetadata.from_libcudf(c_result, s, mr)
