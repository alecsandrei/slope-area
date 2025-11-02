from __future__ import annotations

import collections.abc as c
import contextlib
from contextlib import contextmanager
import copy
from functools import partial, wraps
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time
import typing as t
import warnings

import numpy as np
import rasterio as rio
from rasterio.warp import Resampling
from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area.config import IS_NOTEBOOK, get_wbw_env
from slope_area.logger import create_logger, turn_off_handlers

if t.TYPE_CHECKING:
    from whitebox_workflows.whitebox_workflows import WbEnvironment

    from slope_area._typing import AnyLogger, Resolution, StrPath

m_logger = create_logger(__name__)


def resample[T: StrPath](
    path: StrPath,
    dest: T,
    res: Resolution,
    kwargs_reproject: dict[str, t.Any] | None = None,
) -> T:
    if kwargs_reproject is None:
        kwargs_reproject = {}
    with rio.open(path) as ds:
        arr = ds.read(1)
        meta = ds.meta.copy()
        meta.pop('driver')
        newaff, width, height = rio.warp.calculate_default_transform(
            ds.crs,
            ds.crs,
            ds.width,
            ds.height,
            *ds.bounds,
            resolution=res,
        )
        newarr = np.ma.asanyarray(  # type: ignore[no-untyped-call]
            np.empty(
                shape=(1, height, width),
                dtype=arr.dtype,
            )
        )
        kwargs_reproject.setdefault('resampling', Resampling.bilinear)
        kwargs_reproject['source'] = arr
        kwargs_reproject['destination'] = newarr
        kwargs_reproject['src_transform'] = ds.transform
        kwargs_reproject['dst_transform'] = newaff
        kwargs_reproject['width'] = width
        kwargs_reproject['height'] = height

        rio.warp.reproject(**kwargs_reproject)
        meta.update(
            {
                'transform': newaff,
                'width': width,
                'height': height,
                'nodata': kwargs_reproject.get('dst_nodata', None),
                'crs': kwargs_reproject.get('dst_crs', ds.crs),
            }
        )
        with rio.open(dest, mode='w', **meta) as dest_raster:
            dest_raster.write(newarr)
        return dest


def mask_raster(
    raster: WhiteboxRaster | StrPath,
    mask: WhiteboxRaster | StrPath,
    *,
    logger: AnyLogger = m_logger,
) -> WhiteboxRaster:
    raster = read_whitebox_raster(raster, logger=logger)
    mask = read_whitebox_raster(mask, logger=logger)
    return mask.con(
        'value == 1', true_raster_or_float=raster, false_raster_or_float=mask
    )


def read_whitebox_raster(
    raster: WhiteboxRaster | StrPath, *, logger: AnyLogger = m_logger
) -> WhiteboxRaster:
    if not isinstance(raster, WhiteboxRaster):
        raster_path = os.fspath(raster)
        logger.info('Reading raster %s' % Path(raster_path).name)
        raster = get_wbw_env().read_raster(raster_path)
    return raster


def read_whitebox_vector(
    vector: WhiteboxVector | StrPath, *, logger: AnyLogger = m_logger
) -> WhiteboxVector:
    if not isinstance(vector, WhiteboxVector):
        vector_path = os.fspath(vector)
        logger.info('Reading vector %s' % Path(vector_path).name)
        vector = get_wbw_env().read_vector(vector_path)
    return vector


def write_whitebox[T: WhiteboxRaster | WhiteboxVector](
    output: T,
    out_file: StrPath,
    *,
    logger: AnyLogger | None = None,
    overwrite: bool = False,
    wbw_env: WbEnvironment | None = None,
) -> T:
    if wbw_env is None:
        wbw_env = get_wbw_env()
    os.makedirs(Path(out_file).parent, exist_ok=True)
    out_file_str = os.fspath(out_file)
    if logger is None:
        logger = create_logger(__name__)
    if isinstance(output, WhiteboxRaster):
        write_func = partial(wbw_env.write_raster, output, out_file_str)
    elif isinstance(output, WhiteboxVector):
        write_func = partial(wbw_env.write_vector, output, out_file_str)

    if output.file_mode == 'r':
        logger.debug(
            'Not saving %s as it is open in read-only' % output.file_name
        )
    elif Path(out_file).exists() and not overwrite:
        logger.debug('Not saving %s as overwrite is disabled' % out_file_str)
    else:
        logger.info(
            'Saving %s object to %s' % (output.__class__.__name__, out_file_str)
        )
        write_func()
        output.file_name = out_file_str
    return output


P = t.ParamSpec('P')
R = t.TypeVar('R')


def timeit(
    logger: AnyLogger | None = None, level: int = logging.INFO
) -> c.Callable[[c.Callable[P, R]], c.Callable[P, R]]:
    def decorator(func: c.Callable[P, R]) -> c.Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            message = '%s executed in %.6f seconds' % (func.__name__, duration)
            if logger:
                logger.log(level, message)
            else:
                print(message)
            return result

        return wrapper

    return decorator


@contextmanager
def redirect_warnings(
    logger: AnyLogger,
    warning_category: t.Type[Warning],
    module: str,
) -> c.Generator[None]:
    """Redirects all warnings to logger, disabling all handlers except slopeAreaFile.

    This was written due to the many warnings displayed by GDAL with the
    ESRI Shapefile driver. slope-area uses shapefiles because this is the
    only driver currently supported by Whitebox Tools.
    """
    curr_showwarning = copy.copy(warnings.showwarning)
    spec = importlib.util.find_spec(module)

    def showwarning(message, category, filename, lineno, file=None, line=None):  # type: ignore[no-untyped-def]
        origin = None
        if spec is not None:
            origin = spec.origin
        if category is warning_category and filename == origin:
            with turn_off_handlers('slopeArea', ('stdout', 'stderr')):
                logger.warning(
                    str(message),
                    extra={
                        'warningCategory': category.__name__,
                        'warningFileName': filename,
                        'warningLineNo': lineno,
                    },
                )
        else:
            curr_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning
    try:
        yield
    finally:
        warnings.showwarning = curr_showwarning


@contextmanager
def suppress_stdout_stderr_notebook() -> c.Generator[None]:
    # This does not work, it's here as a placeholder
    # I do not think it's possible to make it work for Jupyter Notebooks
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f) and contextlib.redirect_stderr(f):
            yield


@contextmanager
def suppress_stdout_stderr_terminal() -> c.Generator[None]:
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    old_stdout = os.dup(stdout_fd)
    old_stderr = os.dup(stderr_fd)

    # Redirect to /dev/null
    with open(os.devnull, 'w') as fnull:
        os.dup2(fnull.fileno(), stdout_fd)
        os.dup2(fnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            # Restore original stdout/stderr
            os.dup2(old_stdout, stdout_fd)
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stdout)
            os.close(old_stderr)


@contextmanager
def suppress_stdout_stderr() -> c.Generator[None]:
    if IS_NOTEBOOK:
        with suppress_stdout_stderr_notebook():
            yield
    else:
        with suppress_stdout_stderr_terminal():
            yield
