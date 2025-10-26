from __future__ import annotations

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

from slope_area._typing import AnyLogger, Resolution
from slope_area.config import IS_NOTEBOOK, get_wbw_env
from slope_area.logger import create_logger

if t.TYPE_CHECKING:
    from os import PathLike

    from whitebox_workflows.whitebox_workflows import WbEnvironment

m_logger = create_logger(__name__)


def resample[T: PathLike](
    path: PathLike,
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
        newarr = np.ma.asanyarray(
            np.empty(
                shape=(1, t.cast(int, height), t.cast(int, width)),
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


def write_whitebox[T: WhiteboxRaster | WhiteboxVector](
    output: T,
    out_file: Path,
    *,
    logger: AnyLogger | None = None,
    overwrite: bool = False,
    wbw_env: WbEnvironment | None = None,
) -> T:
    if wbw_env is None:
        wbw_env = get_wbw_env(logger)
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
    elif out_file.exists() and not overwrite:
        logger.debug('Not saving %s as overwrite is disabled' % out_file_str)
    else:
        logger.info(
            'Saving %s object to %s' % (output.__class__.__name__, out_file_str)
        )
        write_func()
        output.file_name = out_file_str
    return output


def extract_class_name_from_args(args) -> str | None:
    cls_name = None
    if args:
        instance = args[0]
        if hasattr(instance, '__class__'):
            cls_name = instance.__class__.__qualname__
            if cls_name == 'type':
                cls_name = instance.__qualname__
    return cls_name


def timeit(logger: AnyLogger | None = None, level: int = logging.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            class_name = extract_class_name_from_args(args)
            class_dot_func = (
                f'{class_name}.{func.__name__}' if class_name else func.__name__
            )
            message = '%s executed in %.6f seconds' % (class_dot_func, duration)
            if logger:
                logger.log(level, message)
            else:
                print(message)
            return result

        return wrapper

    return decorator


@contextmanager
def silence_logger_stdout_stderr(parent_logger_name: str):
    parent_logger = logging.getLogger('slopeArea')
    original_handlers = parent_logger.handlers[:]
    file_handler = logging.getHandlerByName('slopeAreaFile')
    assert file_handler is not None
    handlers = [file_handler]
    parent_logger.handlers = handlers
    try:
        yield
    finally:
        parent_logger.handlers = original_handlers


@contextmanager
def redirect_warnings(
    logger: logging.Logger | logging.LoggerAdapter,
    warning_category,
    module,
):
    """Redirects all warnings to logger, disabling all handlers except slopeAreaFile.

    This was written due to the many warnings displayed by GDAL with the
    ESRI Shapefile driver. slope-area uses shapefiles because this is the
    only driver currently supported by Whitebox Tools.
    """
    extra: dict[str, t.Any] = {}
    if isinstance(logger, logging.LoggerAdapter):
        if logger.extra:
            extra.update(logger.extra)
        logger = logger.logger
    assert isinstance(logger, logging.Logger)
    curr_showwarning = copy.copy(warnings.showwarning)
    spec = importlib.util.find_spec(module)

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if category is warning_category and filename == spec.origin:
            with silence_logger_stdout_stderr('slopeArea'):
                logger.warning(
                    str(message),
                    extra={
                        'warningCategory': category.__name__,
                        'warningFileName': filename,
                        'warningLineNo': lineno,
                        **extra,
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
def suppress_stdout_stderr_notebook():
    # This func not yet tested
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f) and contextlib.redirect_stderr(f):
            yield


@contextmanager
def suppress_stdout_stderr_terminal():
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
def suppress_stdout_stderr():
    if IS_NOTEBOOK:
        with suppress_stdout_stderr_notebook():
            yield
    else:
        with suppress_stdout_stderr_terminal():
            yield
