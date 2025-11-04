from __future__ import annotations

import collections.abc as c
import logging
import typing as t

if t.TYPE_CHECKING:
    from os import PathLike

    from slope_area.console import RichTableRowData
    from slope_area.enums import TrialStatus
    from slope_area.features import Outlet, Raster
    from slope_area.geomorphometry import StreamsComputationOutput

type StrPath = str | PathLike[str]
type PlotKind = t.Literal['line', 'scatter']
type AnyLogger = logging.Logger | logging.LoggerAdapter[logging.Logger]
type AnyDEM = DEMProvider | Raster | StrPath
type Resolution = tuple[int, int] | tuple[float, float]
type EPSG = int
type AnyCRS = ConvertibleToWKT | EPSG

type TrialName = str
type RichTableLogs = c.MutableMapping[TrialName, RichTableRowData]
type SlopeProviders = c.Mapping[str, SlopeProvider | StreamSlopeProvider]


class TrialLoggingContext(t.TypedDict):
    trialStatus: t.NotRequired[TrialStatus]
    trialException: t.NotRequired[Exception]


@t.runtime_checkable
class SlopeProvider(t.Protocol):
    """Provides a percentage slope raster."""

    def get_slope(self, dem: StrPath, out_file: StrPath) -> StrPath: ...


@t.runtime_checkable
class StreamSlopeProvider(t.Protocol):
    """Provides a percentage slope raster."""

    def get_stream_slope(
        self, streams: StreamsComputationOutput, out_file: StrPath
    ) -> StrPath: ...


@t.runtime_checkable
class ConvertibleToWKT(t.Protocol):
    def to_wkt(self) -> str: ...


@t.runtime_checkable
class DEMProvider(t.Protocol):
    def get_dem(self, outlet: Outlet, out_file: StrPath) -> StrPath: ...
