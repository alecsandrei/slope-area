from __future__ import annotations

from functools import partial, wraps
import logging
from os import fspath
from pathlib import Path
import time
import typing as t

import numpy as np
import rasterio as rio
from rasterio.warp import Resampling
from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area import WBW_ENV
from slope_area.logger import create_logger

if t.TYPE_CHECKING:
    from os import PathLike


def resample[T: PathLike](
    path: PathLike,
    dest: T,
    res: tuple[float, float],
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
    logger: logging.Logger | None = None,
    overwrite: bool = False,
) -> T:
    out_file_str = fspath(out_file)
    if logger is None:
        logger = create_logger(__name__)
    if isinstance(output, WhiteboxRaster):
        write_func = partial(WBW_ENV.write_raster, output, out_file_str)
    elif isinstance(output, WhiteboxVector):
        write_func = partial(WBW_ENV.write_vector, output, out_file_str)

    if output.file_mode == 'r':
        logger.debug(
            'Not saving %s as it is open in read-only.' % output.file_name
        )
    elif out_file.exists() and not overwrite:
        logger.debug('Not saving %s as overwrite is disabled.' % out_file_str)
    else:
        logger.info(
            'Saving %s object to %s.'
            % (output.__class__.__name__, out_file_str)
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


def timeit(logger: logging.Logger | None = None, level: int = logging.INFO):
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
