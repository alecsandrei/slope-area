from __future__ import annotations

from functools import cache
import os
import sys
import typing as t

import PySAGA_cmd
import whitebox_workflows as wbw

from slope_area.logger import create_logger

if t.TYPE_CHECKING:
    from whitebox_workflows.whitebox_workflows import WbEnvironment

    from slope_area._typing import AnyLogger

m_logger = create_logger(__name__)


@cache
def get_wbw_env() -> WbEnvironment:
    env: WbEnvironment = wbw.WbEnvironment()
    env.verbose = False
    m_logger.info('Initialized Whitebox Environment')
    return env


@cache
def get_saga_env() -> PySAGA_cmd.SAGA:
    env = PySAGA_cmd.SAGA(os.environ.get('saga_cmd', 'saga_cmd'))
    m_logger.info('Initialized SAGAGIS Environment')
    return env


def get_saga_raster_suffix(saga: PySAGA_cmd.SAGA) -> str:
    if saga.version is None or saga.version.major <= 8:
        return '.sdat'
    else:
        return '.tif'


def is_notebook(logger: AnyLogger) -> bool:
    try:
        ipython = sys.modules['IPython']
        config = ipython.get_ipython().config
        config.get('IPKernelApp')
        m_logger.debug('Code is running in notebook')
        return True
    except Exception:
        return False


IS_NOTEBOOK = is_notebook(m_logger)
