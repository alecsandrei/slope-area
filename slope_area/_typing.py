from __future__ import annotations

import collections.abc as c
import logging
import typing as t

if t.TYPE_CHECKING:
    from os import PathLike

    from whitebox_workflows.whitebox_workflows import WbEnvironment

    from slope_area.builder import RichTableRowData
    from slope_area.enums import TrialStatus
    from slope_area.features import Outlet, Raster

type PlotKind = t.Literal['line', 'scatter']
type AnyLogger = logging.Logger | logging.LoggerAdapter
type AnyDEM = DEMProvider | Raster | PathLike

type Resolution[T: (int, float)] = tuple[T, T]

type TrialName = str
type RichTableLogs = c.MutableMapping[TrialName, RichTableRowData]


class TrialLoggingContext(t.TypedDict):
    trialStatus: t.NotRequired[TrialStatus]
    trialException: t.NotRequired[Exception]


@t.runtime_checkable
class DEMProvider(t.Protocol):
    def get_dem(
        self,
        *,
        outlet: Outlet,
        wbw_env: WbEnvironment | None = None,
        logger: AnyLogger | None = None,
    ) -> Raster: ...
