# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

from pyarrow import feather

import pylibcudf as plc

from cudf.core.column import access_columns
from cudf.core.dataframe import DataFrame
from cudf.core.dtypes import (
    CategoricalDtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.utils import ioutils

_COMPRESSION = {
    None: plc.io.types.CompressionType.LZ4,
    "lz4": plc.io.types.CompressionType.LZ4,
    "zstd": plc.io.types.CompressionType.ZSTD,
    "uncompressed": plc.io.types.CompressionType.NONE,
}


def _cpu_read_feather(path, columns, reason, **kwargs):
    warnings.warn(
        f"Using CPU via PyArrow to read feather dataset ({reason}), this "
        "may be GPU accelerated in the future"
    )
    # ``pyarrow.feather`` is deprecated as of PyArrow 24 but remains the
    # supported compatibility wrapper for the Feather-specific API.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pyarrow\.feather\.read_table is deprecated",
            category=FutureWarning,
        )
        pa_table = feather.read_table(path, columns=columns, **kwargs)
    return DataFrame.from_arrow(pa_table)


@ioutils.doc_read_feather()
def read_feather(
    path, columns=None, storage_options=None, filesystem=None, **kwargs
):
    """{docstring}"""
    if kwargs:
        return _cpu_read_feather(
            path, columns, f"unsupported arguments {sorted(kwargs)}", **kwargs
        )

    source = ioutils.get_reader_filepath_or_buffer(
        path_or_data=path,
        storage_options=storage_options,
        filesystem=filesystem,
    )
    source = ioutils._select_single_source(source, "read_feather")

    options = plc.io.experimental.feather.FeatherReaderOptions.builder(
        plc.io.types.SourceInfo([source])
    ).build()
    if columns is not None:
        options.set_columns(list(columns))

    try:
        result = plc.io.experimental.feather.read_feather(options)
    except (RuntimeError, TypeError, ValueError, NotImplementedError) as err:
        # The GPU reader rejects dictionary-encoded columns, large lists, and
        # the linked-block LZ4 frames PyArrow writes for buffers over 64 KiB.
        return _cpu_read_feather(path, columns, str(err))
    return DataFrame.from_pylibcudf(result)


def _cpu_to_feather(df, path, reason, *args, **kwargs):
    warnings.warn(
        f"Using CPU via PyArrow to write Feather dataset ({reason}), this "
        "may be GPU accelerated in the future"
    )
    # Feather doesn't support using an index
    pa_table = df.to_arrow(preserve_index=False)
    # See the corresponding note in ``_cpu_read_feather`` above.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pyarrow\.feather\.write_feather is deprecated",
            category=FutureWarning,
        )
        feather.write_feather(pa_table, path, *args, **kwargs)


def _gpu_unsupported_reason(df, compression, compression_level, version):
    if version != 2:
        return f"Feather version {version}"
    if compression_level is not None:
        return "compression_level"
    if compression not in _COMPRESSION:
        return f"compression {compression!r}"
    for name, dtype in df._dtypes:
        if isinstance(dtype, (CategoricalDtype, IntervalDtype)):
            return f"column {name!r} of type {dtype}"
    return None


def _column_name_spec(name, dtype):
    if isinstance(dtype, StructDtype):
        return (
            str(name),
            [_column_name_spec(k, v) for k, v in dtype.fields.items()],
        )
    if isinstance(dtype, ListDtype):
        return (
            str(name),
            [("offsets", []), _column_name_spec("item", dtype.element_type)],
        )
    return (str(name), [])


@ioutils.doc_to_feather()
def to_feather(
    df,
    path,
    compression=None,
    compression_level=None,
    chunksize=None,
    version=2,
    storage_options=None,
    filesystem=None,
    **kwargs,
):
    """{docstring}"""
    reason = (
        f"unsupported arguments {sorted(kwargs)}"
        if kwargs
        else _gpu_unsupported_reason(
            df, compression, compression_level, version
        )
    )
    if reason is not None:
        return _cpu_to_feather(
            df,
            path,
            reason,
            compression=compression,
            compression_level=compression_level,
            chunksize=chunksize,
            version=version,
            **kwargs,
        )

    path_or_buf = ioutils.get_writer_filepath_or_buffer(
        path_or_data=path,
        mode="wb",
        storage_options=storage_options,
        filesystem=filesystem,
    )
    if ioutils.is_fsspec_open_file(path_or_buf):
        with path_or_buf as file_obj:
            _plc_write_feather(
                df, ioutils.get_IOBase_writer(file_obj), compression, chunksize
            )
    else:
        _plc_write_feather(df, path_or_buf, compression, chunksize)


def _plc_write_feather(df, path_or_buf, compression, chunksize):
    with access_columns(*df._columns, mode="read", scope="internal"):
        builder = plc.io.experimental.feather.FeatherWriterOptions.builder(
            plc.io.SinkInfo([path_or_buf]),
            plc.Table([col.plc_column for col in df._columns]),
        )
        builder.column_metadata(
            [_column_name_spec(name, dtype) for name, dtype in df._dtypes]
        )
        builder.compression(_COMPRESSION[compression])
        if chunksize is not None:
            builder.max_rows_per_batch(chunksize)
        plc.io.experimental.feather.write_feather(builder.build())
