from __future__ import annotations

import datetime as dt
import json
import logging
import logging.config
import sys
import typing as t

from slope_area.config import LOGGING_CONFIG, PROJ_ROOT

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
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    @t.override
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= logging.INFO


class ErrorFilter(logging.Filter):
    @t.override
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


def setup_logging():
    with LOGGING_CONFIG.open() as f_in:
        config = json.load(f_in)
    config['handlers']['slopeAreaFile']['filename'] = (
        PROJ_ROOT / config['handlers']['slopeAreaFile']['filename']
    ).as_posix()
    logging.config.dictConfig(config)


def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger('slopeArea')
    return logger.getChild(name)
