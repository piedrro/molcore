"""Shared serialization of scientific model mutations and transaction adoption."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps


def model_locked[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Hold the owner dataset's reentrant lock for a public model operation.

    Args:
        function: Function used by this operation.

    Returns:
        Computed Callable[P, R] value.
    """

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Wrapped.

        Args:
            *args: Additional positional values supplied by the caller.
            **kwargs: Additional named options for this operation.

        Returns:
            Computed R value.
        """
        owner = getattr(args[0], "dataset", args[0])
        with owner._lock:
            return function(*args, **kwargs)

    return wrapped
