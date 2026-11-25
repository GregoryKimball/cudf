# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.io.types import (
    ColumnNameSpec,
    CompressionType,
    SinkInfo,
    SourceInfo,
    TableWithMetadata,
)
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

__all__ = [
    "FeatherReaderOptions",
    "FeatherReaderOptionsBuilder",
    "FeatherWriterOptions",
    "FeatherWriterOptionsBuilder",
    "read_feather",
    "write_feather",
]

class FeatherWriterOptions:
    def set_column_names(self, names: list[str]) -> None: ...
    def set_column_metadata(
        self, column_names: list[ColumnNameSpec]
    ) -> None: ...
    def set_compression(self, compression: CompressionType) -> None: ...
    def set_max_rows_per_batch(self, rows: int) -> None: ...
    @staticmethod
    def builder(
        sink: SinkInfo, table: Table
    ) -> FeatherWriterOptionsBuilder: ...

class FeatherWriterOptionsBuilder:
    def column_names(
        self, names: list[str]
    ) -> FeatherWriterOptionsBuilder: ...
    def column_metadata(
        self, column_names: list[ColumnNameSpec]
    ) -> FeatherWriterOptionsBuilder: ...
    def compression(
        self, compression: CompressionType
    ) -> FeatherWriterOptionsBuilder: ...
    def max_rows_per_batch(self, rows: int) -> FeatherWriterOptionsBuilder: ...
    def build(self) -> FeatherWriterOptions: ...

def write_feather(
    options: FeatherWriterOptions, stream: CudaStreamLike | None = None
) -> None: ...

class FeatherReaderOptions:
    def set_columns(self, col_names: list[str]) -> None: ...
    def set_source(self, src: SourceInfo) -> None: ...
    @staticmethod
    def builder(source: SourceInfo) -> FeatherReaderOptionsBuilder: ...

class FeatherReaderOptionsBuilder:
    def columns(self, col_names: list[str]) -> FeatherReaderOptionsBuilder: ...
    def build(self) -> FeatherReaderOptions: ...

def read_feather(
    options: FeatherReaderOptions,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> TableWithMetadata: ...
