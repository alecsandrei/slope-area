from __future__ import annotations

import typing as t

import PySAGA_cmd
import whitebox_workflows as wbw

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

setup_logging()
logger = create_logger(__name__)

logger.info('Initialized Whitebox Environment')
WBW_ENV: WbEnvironment = wbw.WbEnvironment()
logger.info('Initialized SAGAGIS Environment')
SAGA_ENV = PySAGA_cmd.SAGA('saga_cmd')
