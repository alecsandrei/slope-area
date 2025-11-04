from __future__ import annotations

import collections.abc as c
from contextlib import contextmanager
import datetime as dt
import json
import logging
import logging.config
import logging.handlers
import queue
import sys
import threading
import time
import typing as t

from slope_area.enums import TrialStatus
from slope_area.paths import LOGGING_CONFIG, PROJ_ROOT

if t.TYPE_CHECKING:
    from slope_area._typing import AnyLogger, RichTableLogs, TrialLoggingContext
    from slope_area.console import RichTableRowData

LOG_RECORD_BUILTIN_ATTRS = {
    'args',
    'asctime',
    'created',
    'exc_info',
    'exc_text',
    'filename',
    'funcName',
    'levelname',
    'levelno',
    'lineno',
    'module',
    'msecs',
    'message',
    'msg',
    'name',
    'pathname',
    'process',
    'processName',
    'relativeCreated',
    'stack_info',
    'thread',
    'threadName',
    'taskName',
}


# ANSI escape sequences
_RESET = '\x1b[0m'
_COLORS = {
    'DEBUG': '\x1b[36m',  # cyan
    'INFO': '\x1b[32m',  # green
    'WARNING': '\x1b[33m',  # yellow
    'ERROR': '\x1b[31m',  # red
    'CRITICAL': '\x1b[41m',  # red background
}
_COLORS_RICH = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


class ColoredFormatter(logging.Formatter):
    """
    Logging formatter that injects ANSI colors based on the record level.
    Works with any handler that emits to a terminal (StreamHandler to sys.stdout/stderr).
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: t.Literal['%', '{', '$'] = '%',
        use_colors: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_colors = use_colors and self._stream_supports_color()

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if self.use_colors and levelname in _COLORS:
            color = _COLORS[levelname]
            # color the levelname only (safer than coloring entire message)
            record.levelname = f'{color}{levelname}{_RESET}'
            # optionally color message or name: e.g. record.name = f"{color}{record.name}{_RESET}"
        formatted = super().format(record)
        # restore original levelname (important if other handlers use this record)
        record.levelname = levelname
        return formatted

    @staticmethod
    def _stream_supports_color() -> bool:
        """
        Return True if the running system's stdout/stderr likely supports ANSI colors.
        Simple heuristic: isatty + not Windows lacking ANSI support.
        """
        try:
            return sys.stderr.isatty() or sys.stdout.isatty()
        except Exception:
            return False


class JSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @t.override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(
        self, record: logging.LogRecord
    ) -> dict[str, str | None]:
        always_fields = {
            'message': record.getMessage(),
            'timestamp': dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
        if record.exc_info is not None:
            always_fields['exc_info'] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields['stack_info'] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val, None)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        return message


class NonErrorFilter(logging.Filter):
    @t.override
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= logging.INFO


class ErrorFilter(logging.Filter):
    @t.override
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


class RichDictHandler(logging.Handler):
    def __init__(
        self,
        row_data: RichTableRowData,
        queue: queue.Queue[RichTableLogs | None],
        update_timer: float = 0.1,
    ) -> None:
        super().__init__()
        self.queue = queue
        self.row_data = row_data
        self.update_timer = update_timer
        self.setFormatter(logging.Formatter(fmt='[%(levelname)s] %(message)s'))
        self.setLevel(logging.DEBUG)

        self.stop_event = threading.Event()
        self.update_table_thread = threading.Thread(
            target=self.send_logs, daemon=True
        )
        self.update_table_thread.start()

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in _COLORS_RICH:
            color = _COLORS_RICH[levelname]
            record.levelname = f'[{color}]{levelname}[/{color}]'
        formatted = super().format(record)
        record.levelname = levelname
        return formatted

    def send_logs(self) -> None:
        while True:
            if self.stop_event.is_set():
                break
            self.queue.put_nowait({self.row_data.trial: self.row_data})
            time.sleep(self.update_timer)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        trial_context: TrialLoggingContext | dict[str, t.Any] = getattr(
            record, 'trialContext', {}
        )

        status: TrialStatus | None = trial_context.get('trialStatus', None)
        exception: Exception | None = trial_context.get('trialException', None)

        if msg:
            self.row_data.message = msg
        if status is not None:
            self.row_data.status = status
        if exception is not None:
            self.row_data.exception = exception
        self.queue.put_nowait({self.row_data.trial: self.row_data})

        if status is TrialStatus.FINISHED:
            self.stop_event.set()
            self.update_table_thread.join()


class TrialLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    def __init__(
        self,
        logger: logging.Logger,
        trial_name: str,
        trial_context: TrialLoggingContext | None = None,
    ) -> None:
        super().__init__(logger)
        self.logger = logger
        self.trial_name = trial_name
        self.trial_context = trial_context or {}

    def process(
        self, msg: t.Any, kwargs: c.MutableMapping[str, t.Any]
    ) -> tuple[t.Any, c.MutableMapping[str, t.Any]]:
        kwargs.setdefault('extra', {}).update(
            {'trialName': self.trial_name, 'trialContext': self.trial_context}
        )
        return (msg, kwargs)

    def _log_with_context(
        self,
        level: int,
        msg: str,
        *,
        status: TrialStatus | None = None,
        exc: Exception | None = None,
    ) -> None:
        trial_context: TrialLoggingContext = {}
        if status is not None:
            trial_context['trialStatus'] = status
        if exc is not None:
            trial_context['trialException'] = exc
        self.trial_context.update(trial_context)

        self.log(level, msg, stacklevel=3)

    def mark_running(self) -> None:
        self._log_with_context(
            logging.INFO,
            'Running...',
            status=TrialStatus.RUNNING,
        )

    def mark_finished(self) -> None:
        self._log_with_context(
            logging.INFO,
            'Finished with success!',
            status=TrialStatus.FINISHED,
        )

    def mark_error(self, exc: Exception) -> None:
        self._log_with_context(
            logging.ERROR,
            f'Error: {exc}',
            status=TrialStatus.ERRORED,
            exc=exc,
        )


def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger('slopeArea')
    return logger.getChild(name)


@contextmanager
def turn_off_handlers(
    logger: str | AnyLogger, handlers: c.Sequence[str] | None = None
) -> c.Generator[None]:
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    elif isinstance(logger, logging.LoggerAdapter):
        logger = logger.logger
    assert isinstance(logger, logging.Logger)
    saved_handlers = logger.handlers[:]
    if handlers is None:
        logger.handlers.clear()
    else:
        logger.handlers = [h for h in logger.handlers if h.name not in handlers]
    try:
        yield
    finally:
        logger.handlers = saved_handlers


def setup_logging() -> None:
    with LOGGING_CONFIG.open() as file:
        config = json.load(file)
    config['handlers']['slopeAreaFile']['filename'] = (
        PROJ_ROOT / config['handlers']['slopeAreaFile']['filename']
    ).as_posix()
    logging.config.dictConfig(config)


def turn_off_logging() -> None:
    logging.getLogger('slopeArea').disabled = True
