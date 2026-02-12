"""Decorator and context-manager helpers for quick profiling.

Provides convenience decorators that wrap functions with GPU profiling
so users can add profiling with a single line of code.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def profile_training(
    func: Optional[Callable[..., Any]] = None,
    *,
    interval_ms: int = 100,
    report_format: str = "terminal",
    output_path: Optional[str] = None,
) -> Any:
    """Decorator that profiles a training function.

    Usage::

        @profile_training
        def train():
            ...

        @profile_training(report_format="html", output_path="report.html")
        def train():
            ...
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from src.sdk.profiler_sdk import ChamberProfiler
            profiler = ChamberProfiler(interval_ms=interval_ms)
            with profiler.profile():
                result = fn(*args, **kwargs)
            profiler.report(format=report_format, output_path=output_path)
            return result
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def profile_gpu(
    func: Optional[Callable[..., Any]] = None,
    *,
    interval_ms: int = 100,
) -> Any:
    """Decorator that profiles GPU metrics only (no kernel tracing).

    Usage::

        @profile_gpu
        def train_step():
            ...
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from src.sdk.profiler_sdk import ChamberProfiler
            profiler = ChamberProfiler(
                gpu=True,
                kernels=False,
                memory=False,
                communication=False,
                data_loading=False,
                interval_ms=interval_ms,
            )
            with profiler.profile():
                result = fn(*args, **kwargs)
            profiler.report(format="terminal")
            return result
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# Alias
gpu_profile = profile_gpu
