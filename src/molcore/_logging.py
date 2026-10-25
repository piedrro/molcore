"""Host-configured diagnostics that summarise payloads without printing arrays."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Generator
from concurrent.futures import CancelledError
from dataclasses import fields, is_dataclass
from functools import wraps
from time import perf_counter
from typing import Any, cast


def _summary(value: Any) -> str:
    """Describe shape, cardinality or scalar context without reading lazy pixels.

    Args:
        value: Operation input or result to describe.

    Returns:
        Bounded structural diagnostic text.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value[:160])
    if isinstance(value, dict):
        return f"dict(keys={list(value)[:15]!r})"
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}(count={len(value)})"
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__
            + "("
            + ", ".join(
                f"{field.name}={_summary(getattr(value, field.name))}" for field in fields(value)[:15]
            )
            + ")"
        )
    if hasattr(value, "shape"):
        return f"{type(value).__name__}(shape={value.shape}, dtype={getattr(value, 'dtype', None)})"
    return type(value).__name__


def logged[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Record operation boundaries, duration, cancellation and exception tracebacks.

    Args:
        function: Synchronous operation or generator to instrument.

    Returns:
        Signature-preserving callable using the original module's logger.
    """
    logger = logging.getLogger(function.__module__)
    name = function.__qualname__
    level = (
        logging.INFO
        if function.__name__
        in {
            "localise",
            "detect",
            "fit",
            "track",
            "segment",
            "segment_many",
            "undrift",
            "undrift_many",
            "estimate_alignment",
            "align_channels",
            "filter_results",
            "cluster_localizations",
            "colocalise",
            "render_localizations",
            "measure_localizations",
            "stationary_traces",
            "track_traces",
            "save_experiment",
            "open_experiment",
            "read_image",
            "migrate_moltrack",
        }
        else logging.DEBUG
    )

    def begin(args: tuple[Any, ...], kwargs: dict[str, Any]) -> float:
        """Log bounded argument structure and start the elapsed-time clock.

        Args:
            args: Positional arguments.
            kwargs: Named arguments.

        Returns:
            Monotonic operation start time.
        """
        logger.log(level, "%s started", name)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s started args=%s options=%s",
                name,
                [_summary(v) for v in args],
                {k: _summary(v) for k, v in kwargs.items()},
            )
        return perf_counter()

    def failure(start: float, error: BaseException) -> None:
        """Record cancellation separately from failures with traceback context.

        Args:
            start: Monotonic operation start time.
            error: Exception currently being handled.
        """
        if isinstance(error, (CancelledError, GeneratorExit)):
            logger.debug("%s cancelled after %.3fs", name, perf_counter() - start)
        else:
            logger.debug("%s failed after %.3fs", name, perf_counter() - start, exc_info=True)

    if inspect.isgeneratorfunction(function):

        @wraps(function)
        def generator(*args: P.args, **kwargs: P.kwargs) -> Generator[Any, Any, Any]:
            """Trace a generator for its full iteration lifetime.

            Args:
                *args: Arguments forwarded to the wrapped generator.
                **kwargs: Named arguments forwarded to the wrapped generator.

            Yields:
                Values produced by the wrapped operation.
            """
            start = begin(args, kwargs)
            try:
                yield from function(*args, **kwargs)
                logger.log(level, "%s completed in %.3fs", name, perf_counter() - start)
            except BaseException as error:
                failure(start, error)
                raise

        return cast(Callable[P, R], generator)

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Trace one synchronous operation without altering its return value.

        Args:
            *args: Arguments forwarded to the wrapped function.
            **kwargs: Named arguments forwarded to the wrapped function.

        Returns:
            The wrapped operation's unchanged result.
        """
        start = begin(args, kwargs)
        try:
            result = function(*args, **kwargs)
            logger.log(level, "%s completed in %.3fs", name, perf_counter() - start)
            logger.debug("%s result=%s", name, _summary(result))
            return result
        except BaseException as error:
            failure(start, error)
            raise

    return wrapped
