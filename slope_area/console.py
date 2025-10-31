from __future__ import annotations

from dataclasses import astuple, dataclass, fields
import logging
import queue
import typing as t

from rich import box
from rich.table import Table

from slope_area.enums import TrialStatus
from slope_area.logger import RichDictHandler, create_logger

if t.TYPE_CHECKING:
    from slope_area._typing import RichTableLogs

m_logger = create_logger(__name__)


@dataclass
class RichTableRowData:
    trial: str
    message: str
    status: str
    exception: str

    @classmethod
    def from_trial_name(cls, trial_name: str) -> t.Self:
        return cls(
            trial=trial_name,
            message='[dim]Waiting...[/dim]',
            status=TrialStatus.NOT_STARTED.display(),
            exception='',
        )


def make_table(logs: RichTableLogs) -> Table:
    table = Table(box=box.SQUARE_DOUBLE_HEAD, show_header=True, expand=True)

    for dataclass_field in fields(RichTableRowData):
        table.add_column(dataclass_field.name.capitalize())
    for row_data in logs.values():
        table.add_row(*astuple(row_data))
    return table


def create_rich_logger(
    logger_name: str, logs: RichTableLogs, q: queue.Queue[RichTableLogs | None]
) -> logging.Logger:
    r_logger = m_logger.getChild(logger_name)
    rich_handler = RichDictHandler(logs, q)
    r_logger.addHandler(rich_handler)
    return r_logger


def rich_table_logs_thread(
    logs: RichTableLogs, q: queue.Queue[RichTableLogs]
) -> None:
    while True:
        trial_log: RichTableLogs = q.get()
        if trial_log is None:
            break
        logs.update(trial_log)
