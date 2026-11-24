# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from contextlib import contextmanager
from string import ascii_letters

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import NUMERIC_TYPES


@pytest.fixture(scope="module", params=[0, 10])
def pdf(request):
    # Feather reader and writer tests only read this source dataframe.
    rng = np.random.default_rng(seed=0)
    types = [*NUMERIC_TYPES, "bool"]
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.DataFrame(
        {
            f"col_{typ}": rng.integers(0, nrows, nrows).astype(typ)
            for typ in types
        }
    )
    # Create non-numeric categorical data otherwise may get typecasted
    data = [ascii_letters[rng.integers(0, 52)] for i in range(nrows)]
    test_pdf["col_category"] = pd.Series(data, dtype="category")
    return test_pdf


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:pyarrow.feather:FutureWarning")
@pytest.mark.parametrize(
    "columns",
    [["col_int8"], ["col_category"], ["col_int32", "col_float32"], None],
)
def test_feather_reader(pdf, columns, tmp_path):
    feather_file = tmp_path / "test.feather"
    pdf.to_feather(feather_file)
    expect = pa.feather.read_table(feather_file, columns=columns).to_pandas()
    got = (
        cudf.read_feather(feather_file, columns=columns)
        .to_arrow(preserve_index=False)
        .to_pandas()
    )

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:pyarrow.feather:FutureWarning")
def test_feather_writer(tmp_path, pdf):
    gdf = cudf.DataFrame(pdf)
    pdf_fname = tmp_path / "pdf.feather"
    gdf_fname = tmp_path / "gdf.feather"

    pdf.to_feather(pdf_fname)
    gdf.to_feather(gdf_fname)

    expect = pa.feather.read_table(pdf_fname)
    got = pa.feather.read_table(gdf_fname)
    # For empty dataframes, pandas writes category dict with null values type
    # while cudf writes it with string values type. Cast to match.
    if len(expect) == 0 and "col_category" in expect.column_names:
        idx = expect.schema.get_field_index("col_category")
        expect = expect.set_column(
            idx,
            got.schema.field(idx),
            expect.column(idx).cast(got.schema.field(idx).type),
        )
    assert pa.Table.equals(expect, got)


def _nullable_pdf(nrows):
    rng = np.random.default_rng(seed=0)
    nulls = rng.random(nrows) < 0.1
    return pd.DataFrame(
        {
            "int8": pd.array(
                rng.integers(-100, 100, nrows), dtype="Int8"
            ).where(~nulls),
            "int64": pd.array(
                rng.integers(-(2**40), 2**40, nrows), dtype="Int64"
            ),
            "uint32": pd.array(
                rng.integers(0, 2**32 - 1, nrows), dtype="UInt32"
            ).where(~nulls),
            "float64": pd.array(rng.random(nrows), dtype="Float64").where(
                ~nulls
            ),
            "bool": pd.array(rng.random(nrows) > 0.5, dtype="boolean").where(
                ~nulls
            ),
            "string": pd.array(
                [f"s{i % 1000}" * (i % 5) for i in range(nrows)],
                dtype="string",
            ).where(~nulls),
            "timestamp": pd.Series(
                rng.integers(0, 2**40, nrows).astype("datetime64[ms]")
            ).where(~nulls),
        }
    )


@contextmanager
def _no_cpu_fallback():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="Using CPU", category=UserWarning
        )
        yield


# Small enough that PyArrow writes every LZ4 buffer as a single frame block,
# which the GPU reader requires
@pytest.mark.parametrize("compression", ["lz4", "zstd", "uncompressed"])
@pytest.mark.parametrize("chunksize", [None, 300])
def test_feather_pandas_writer_cudf_reader(tmp_path, compression, chunksize):
    pdf = _nullable_pdf(1000)
    fname = tmp_path / "pandas.feather"
    pdf.to_feather(fname, compression=compression, chunksize=chunksize)

    with _no_cpu_fallback():
        got = cudf.read_feather(fname)

    assert_eq(cudf.from_pandas(pdf), got)


def test_feather_pandas_writer_cudf_reader_columns(tmp_path):
    pdf = _nullable_pdf(1000)
    fname = tmp_path / "pandas.feather"
    pdf.to_feather(fname, compression="zstd")
    columns = ["string", "int8"]

    with _no_cpu_fallback():
        got = cudf.read_feather(fname, columns=columns)

    assert_eq(cudf.from_pandas(pdf[columns]), got)


def test_feather_pandas_writer_cudf_reader_linked_lz4_fallback(tmp_path):
    pdf = _nullable_pdf(100_000)
    fname = tmp_path / "pandas.feather"
    pdf.to_feather(fname, compression="lz4")

    with pytest.warns(UserWarning, match="linked blocks"):
        got = cudf.read_feather(fname)

    assert_eq(cudf.from_pandas(pdf), got)


@pytest.mark.parametrize("compression", [None, "lz4", "zstd", "uncompressed"])
@pytest.mark.parametrize("chunksize", [None, 30_000])
def test_feather_cudf_writer_pandas_reader(tmp_path, compression, chunksize):
    gdf = cudf.from_pandas(_nullable_pdf(100_000))
    fname = tmp_path / "cudf.feather"

    with _no_cpu_fallback():
        gdf.to_feather(fname, compression=compression, chunksize=chunksize)

    got = pd.read_feather(fname, dtype_backend="numpy_nullable")
    assert_eq(gdf.to_pandas(nullable=True), got)


@pytest.mark.parametrize("compression", [None, "zstd", "uncompressed"])
def test_feather_cudf_writer_cudf_reader(tmp_path, compression):
    gdf = cudf.from_pandas(_nullable_pdf(100_000))
    fname = tmp_path / "cudf.feather"

    with _no_cpu_fallback():
        gdf.to_feather(fname, compression=compression)
        got = cudf.read_feather(fname)

    assert_eq(gdf, got)
