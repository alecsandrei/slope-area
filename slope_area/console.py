from __future__ import annotations

from dataclasses import dataclass, field, fields
import logging
import queue
import threading
import time
import typing as t

from rich import box
from rich.table import Table
from rich.text import Text

from slope_area.enums import TrialStatus
from slope_area.logger import RichDictHandler, create_logger

if t.TYPE_CHECKING:
    from rich.console import RenderableType

    from slope_area._typing import RichTableLogs

m_logger = create_logger(__name__)

_COLORS_TRIAL_STATUS = {
    TrialStatus.NOT_STARTED: 'gray',
    TrialStatus.RUNNING: 'cyan',
    TrialStatus.FINISHED: 'green',
    TrialStatus.ERRORED: 'red',
}


@dataclass
class RichTableRowData:
    trial: str
    message: str
    status: TrialStatus
    exception: Exception | None = None
    elapsed_time: float | None = None

    # internal fields
    _start_time: float = field(init=False, repr=False)
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False
    )
    _timer_thread: threading.Thread | None = field(
        init=False, default=None, repr=False
    )

    def __getstate__(self) -> dict[str, t.Any]:
        state = self.__dict__.copy()
        for key in ['_lock', '_timer_thread', '_start_time']:
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        self.__dict__.update(state)

    def render(self) -> tuple[RenderableType, ...]:
        rendered = (
            Text(self.trial, style='bold'),
            self.message,
            self.get_status(),
            self.get_exception(),
            self.get_elapsed_time(),
        )
        renderable_fields = [field for field in fields(self) if field.init]
        if not len(rendered) == len(renderable_fields):
            raise ValueError(
                'Forgot to update render function with new class attribute'
            )
        return rendered

    def get_elapsed_time(self) -> Text:
        if self.elapsed_time is None:
            return Text('—', style='dim')

        if self.elapsed_time < 1:
            formatted = f'{self.elapsed_time * 1000:.0f} ms'
            style = 'green'
        elif self.elapsed_time < 60:
            formatted = f'{self.elapsed_time:.1f} s'
            style = 'cyan'
        elif self.elapsed_time < 3600:
            m, s = divmod(self.elapsed_time, 60)
            formatted = f'{int(m)} m {s:.0f} s'
            style = 'yellow'
        else:
            h, rem = divmod(self.elapsed_time, 3600)
            m, _ = divmod(rem, 60)
            formatted = f'{int(h)} h {int(m)} m'
            style = 'magenta'

        return Text(formatted, style=style)

    def get_exception(self) -> Text:
        if self.exception is None:
            return Text()
        return Text(self.exception.__class__.__name__, style='red')

    def get_status(self) -> Text:
        return Text(
            format_table_value(self.status),
            style=_COLORS_TRIAL_STATUS[self.status],
        )

    def __setattr__(self, name: str, value: t.Any, /) -> None:
        if name == 'status' and value is TrialStatus.RUNNING:
            if self._timer_thread is None or not self._timer_thread.is_alive():
                self.start_timer()
        super().__setattr__(name, value)

    def start_timer(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.perf_counter()
        self._timer_thread = threading.Thread(
            target=self.update_elapsed_time, daemon=True
        )
        self._timer_thread.start()

    def update_elapsed_time(self) -> None:
        assert self._start_time is not None
        while self.status is TrialStatus.RUNNING:
            with self._lock:
                self.elapsed_time = time.perf_counter() - self._start_time
                time.sleep(0.1)

    @classmethod
    def from_trial_name(cls, trial_name: str) -> t.Self:
        return cls(
            trial=trial_name,
            message='[dim]Waiting...[/dim]',
            status=TrialStatus.NOT_STARTED,
            exception=None,
        )


def format_table_value(column: str) -> str:
    return column.replace('_', ' ').capitalize()


def make_table(logs: RichTableLogs) -> Table:
    table = Table(box=box.SQUARE_DOUBLE_HEAD, show_header=True, expand=True)

    for dataclass_field in fields(RichTableRowData):
        if dataclass_field.init:
            table.add_column(format_table_value(dataclass_field.name))
    for row_data in logs.values():
        table.add_row(*row_data.render())
    return table


def create_rich_logger(
    logger_name: str,
    row_data: RichTableRowData,
    q: queue.Queue[RichTableLogs | None],
) -> logging.Logger:
    r_logger = m_logger.getChild(logger_name)
    rich_handler = RichDictHandler(row_data, q)
    r_logger.addHandler(rich_handler)
    return r_logger


def rich_table_logs_thread(
    logs: RichTableLogs, q: queue.Queue[RichTableLogs | None]
) -> None:
    while True:
        new_logs = q.get()
        if new_logs is None:
            break
        logs.update(new_logs)
