from __future__ import annotations

import os
import typing as t

import PySAGA_cmd
import whitebox_workflows as wbw

from slope_area._typing import AnyLogger
from slope_area.logger import (
    ColoredFormatter,
    ErrorFilter,
    JSONFormatter,
    NonErrorFilter,
    create_logger,
    setup_logging,
)


__all__ = [
    'ErrorFilter',
    'JSONFormatter',
    'NonErrorFilter',
    'ColoredFormatter',
]

if t.TYPE_CHECKING:
    from whitebox_workflows.whitebox_workflows import WbEnvironment


def get_wbw_env(logger: AnyLogger) -> WbEnvironment:
    env: WbEnvironment = wbw.WbEnvironment()
    env.verbose = False
    logger.info('Initialized Whitebox Environment')
    return env


def get_saga_env(logger: AnyLogger) -> PySAGA_cmd.SAGA:
    env = PySAGA_cmd.SAGA(os.environ.get('saga_cmd', 'saga_cmd'))
    logger.info('Initialized SAGAGIS Environment')
    return env


setup_logging()
logger = create_logger(__name__)

WBW_ENV: WbEnvironment = get_wbw_env(logger)
SAGA_ENV = get_saga_env(logger)

# Data settings
if SAGA_ENV.version is None or SAGA_ENV.version.major <= 8:
    SAGA_RASTER_SUFFIX = '.sdat'
else:
    SAGA_RASTER_SUFFIX = '.tif'
