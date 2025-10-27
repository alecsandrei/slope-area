from __future__ import annotations

import collections.abc as c
import logging
import typing as t

if t.TYPE_CHECKING:
    from slope_area.builder import RichTableRowData
    from slope_area.enums import TrialStatus

type PlotKind = t.Literal['line', 'scatter']
type AnyLogger = logging.Logger | logging.LoggerAdapter

type Resolution[T: (int, float)] = tuple[T, T]

type TrialName = str
type RichTableLogs = c.MutableMapping[TrialName, RichTableRowData]


class TrialLoggingContext(t.TypedDict):
    trialStatus: t.NotRequired[TrialStatus]
    trialException: t.NotRequired[Exception]
