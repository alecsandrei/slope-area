from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import json
import logging
import logging.config
import logging.handlers
import queue
import sys
import typing as t

from slope_area._typing import RichTableLogs, TrialLoggingContext
from slope_area.enums import TrialStatus
from slope_area.paths import LOGGING_CONFIG, PROJ_ROOT

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
_COLORS_TRIAL_STATUS = {
    TrialStatus.NOT_STARTED: 'gray',
    TrialStatus.RUNNING: 'cyan',
    TrialStatus.FINISHED: 'green',
    TrialStatus.ERRORED: 'red',
}


class ColoredFormatter(logging.Formatter):
    """
    Logging formatter that injects ANSI colors based on the record level.
    Works with any handler that emits to a terminal (StreamHandler to sys.stdout/stderr).
    """

    def __init__(
        self, fmt=None, datefmt=None, style='%', use_colors: bool = True
    ):
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

    def _prepare_log_dict(self, record: logging.LogRecord):
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
    def __init__(self, logs: RichTableLogs, queue: queue.Queue):
        super().__init__()
        self.queue = queue
        self.logs = logs
        self.setFormatter(logging.Formatter(fmt='[%(levelname)s] %(message)s'))
        self.setLevel(logging.DEBUG)

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in _COLORS_RICH:
            color = _COLORS_RICH[levelname]
            record.levelname = f'[{color}]{levelname}[/{color}]'
        formatted = super().format(record)
        record.levelname = levelname
        return formatted

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        trial_name: str = getattr(record, 'trialName')
        trial_context: TrialLoggingContext | dict = getattr(
            record, 'trialContext', {}
        )

        curr_data = self.logs[trial_name]

        status: TrialStatus | None = trial_context.get('trialStatus', None)
        exception: Exception | None = trial_context.get('trialException', None)

        if msg:
            curr_data.message = msg
        if status is not None:
            status_color = _COLORS_TRIAL_STATUS[status]
            curr_data.status = (
                f'[{status_color}]{status.display()}[/{status_color}]'
            )
        if exception is not None:
            curr_data.exception = exception.__class__.__name__
        self.queue.put_nowait({trial_name: curr_data})


def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger('slopeArea')
    return logger.getChild(name)


@contextmanager
def silent_logs(*logger_names):
    saved_handlers = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        saved_handlers[name] = logger.handlers[:]
        # Maybe leave slopeAreaFile?
        logger.handlers = [
            h
            for h in logger.handlers
            if not isinstance(h, logging.StreamHandler)
        ]
    try:
        yield
    finally:
        for name, handlers in saved_handlers.items():
            logging.getLogger(name).handlers = handlers


def setup_logging():
    with LOGGING_CONFIG.open() as file:
        config = json.load(file)
    config['handlers']['slopeAreaFile']['filename'] = (
        PROJ_ROOT / config['handlers']['slopeAreaFile']['filename']
    ).as_posix()
    logging.config.dictConfig(config)
