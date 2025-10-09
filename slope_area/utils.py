from __future__ import annotations

from functools import wraps
import logging
import time


def extract_class_name_from_args(args) -> str | None:
    cls_name = None
    if args:
        instance = args[0]
        if hasattr(instance, '__class__'):
            cls_name = instance.__class__.__name__
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
