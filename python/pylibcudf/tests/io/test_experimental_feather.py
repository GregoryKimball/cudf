# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pyarrow.feather as pa_feather
import pytest
from utils import assert_table_and_meta_eq

import pylibcudf as plc

feather = plc.io.experimental.feather


def _make_table(num_rows):
    rng = np.random.default_rng(seed=0)
    valid = rng.random(num_rows) > 0.1

    def masked(values):
        return pa.array(values, mask=~valid)

    return pa.table(
        {
            "int8": masked(rng.integers(-100, 100, num_rows, dtype=np.int8)),
            "int64": masked(rng.integers(-(2**40), 2**40, num_rows)),
            "uint32": masked(
                rng.integers(0, 2**32 - 1, num_rows, dtype=np.uint32)
            ),
            "float32": masked(rng.random(num_rows, dtype=np.float32)),
            "float64": masked(rng.random(num_rows)),
            "bool": masked(rng.random(num_rows) > 0.5),
            "timestamp": masked(
                rng.integers(0, 2**50, num_rows).astype("datetime64[us]")
            ),
            "string": masked([f"s{i}" * int(i % 7) for i in range(num_rows)]),
        }
    )


def _read(path, columns=None):
    options = feather.FeatherReaderOptions.builder(
        plc.io.types.SourceInfo([str(path)])
    ).build()
    if columns is not None:
        options.set_columns(columns)
    return feather.read_feather(options)


@pytest.mark.parametrize("compression", ["uncompressed", "zstd", "lz4"])
@pytest.mark.parametrize("chunksize", [None, 300])
def test_read_feather_from_pyarrow(tmp_path, compression, chunksize):
    # Small enough that every PyArrow LZ4 buffer is a single frame block
    pa_table = _make_table(1000)
    path = tmp_path / "in.feather"
    pa_feather.write_feather(
        pa_table, path, compression=compression, chunksize=chunksize
    )

    assert_table_and_meta_eq(pa_table, _read(path))


def test_read_feather_columns(tmp_path):
    pa_table = _make_table(1000)
    path = tmp_path / "in.feather"
    pa_feather.write_feather(pa_table, path, compression="zstd")

    columns = ["string", "int8"]
    assert_table_and_meta_eq(pa_table.select(columns), _read(path, columns))


def test_read_feather_lz4_linked_blocks_raises(tmp_path):
    path = tmp_path / "in.feather"
    pa_feather.write_feather(
        pa.table({"a": np.arange(100_000)}), path, compression="lz4"
    )
    with pytest.raises(RuntimeError, match="linked blocks"):
        _read(path)


@pytest.mark.parametrize(
    "compression",
    [
        plc.io.types.CompressionType.NONE,
        plc.io.types.CompressionType.LZ4,
        plc.io.types.CompressionType.ZSTD,
    ],
)
@pytest.mark.parametrize("max_rows_per_batch", [None, 30_000])
def test_write_feather_roundtrip(tmp_path, compression, max_rows_per_batch):
    pa_table = _make_table(100_000)
    path = tmp_path / "out.feather"
    builder = (
        feather.FeatherWriterOptions.builder(
            plc.io.SinkInfo([str(path)]), plc.Table.from_arrow(pa_table)
        )
        .column_names(pa_table.column_names)
        .compression(compression)
    )
    if max_rows_per_batch is not None:
        builder.max_rows_per_batch(max_rows_per_batch)
    feather.write_feather(builder.build())

    assert pa_feather.read_table(path).equals(pa_table)
    assert_table_and_meta_eq(pa_table, _read(path))


def test_write_feather_unsupported_compression(tmp_path):
    pa_table = _make_table(10)
    builder = feather.FeatherWriterOptions.builder(
        plc.io.SinkInfo([str(tmp_path / "out.feather")]),
        plc.Table.from_arrow(pa_table),
    )
    with pytest.raises(RuntimeError, match="NONE, LZ4 and ZSTD"):
        builder.compression(plc.io.types.CompressionType.SNAPPY)
