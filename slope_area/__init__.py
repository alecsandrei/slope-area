from __future__ import annotations

import typing as t

import PySAGA_cmd
import whitebox_workflows as wbw

from slope_area.config import SAGA_CMD
from slope_area.logger import (
    ColoredFormatter,
    ErrorFilter,
    JSONFormatter,
    NonErrorFilter,
    create_logger,
    setup_logging,
)

__all__ = ['ErrorFilter', 'JSONFormatter', 'NonErrorFilter', 'ColoredFormatter']

if t.TYPE_CHECKING:
    from whitebox_workflows.whitebox_workflows import WbEnvironment


def get_wbw_env() -> WbEnvironment:
    env: WbEnvironment = wbw.WbEnvironment()
    env.verbose = False
    return env


setup_logging()
logger = create_logger(__name__)

WBW_ENV: WbEnvironment = get_wbw_env()
logger.info('Initialized Whitebox Environment')

SAGA_ENV = PySAGA_cmd.SAGA(SAGA_CMD)
logger.info('Initialized SAGAGIS Environment')

# Data settings
if SAGA_ENV.version is None or SAGA_ENV.version.major <= 8:
    SAGA_RASTER_SUFFIX = '.sdat'
else:
    SAGA_RASTER_SUFFIX = '.tif'
