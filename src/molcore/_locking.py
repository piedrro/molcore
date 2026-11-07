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


def model_mutated[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Serialize a mutator and publish one change after the outer operation succeeds.

    Args:
        function: Validating public model mutator.

    Returns:
        Wrapped operation retaining its original signature.
    """

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Run a mutation and deliver its committed change.

        Args:
            *args: Original positional arguments.
            **kwargs: Original named arguments.

        Returns:
            Original mutator result.
        """
        from ._changes import partitions, versions

        receiver = args[0]
        owner = getattr(receiver, "dataset", receiver)
        change = None
        with owner._lock:
            # Temporary validation tables must not publish changes to their owner.
            if receiver is not owner and not any(
                receiver is getattr(owner, name)
                for name in (
                    "localizations",
                    "tracks",
                    "observations",
                    "photon_sequences",
                    "bounding_boxes",
                    "transforms",
                )
            ):
                return function(*args, **kwargs)
            outer = owner._mutation_depth == 0
            if outer:
                before = versions(owner)
                owner._mutation_partitions = set()
            owner._mutation_partitions.update(partitions(owner, function, args, kwargs))
            owner._mutation_depth += 1
            try:
                result = function(*args, **kwargs)
                if outer:
                    after = versions(owner)
                    changed = frozenset(k for k in before if before[k] != after[k])
                    if changed:
                        change = owner._record_change(
                            changed, frozenset(owner._mutation_partitions)
                        )
            finally:
                owner._mutation_depth -= 1
        if change is not None:
            owner._publish_change(change)
        return result

    return wrapped
