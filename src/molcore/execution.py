"""Reusable caller execution policy without owning or closing borrowed executors."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from .types import CancelEvent, ProgressCallback

if TYPE_CHECKING:
    from .cpu import WorkerProgress


class Executor(Protocol):
    """Ordered worker map compatible with CpuExecutor; ownership stays with callers.

    Implementations must supply a WorkerProgress-compatible reporter, forward
    progress on the calling thread, and honour cooperative cancellation. Workers
    accept (payload, reporter); process executors require picklable workers.
    """

    def map(
        self,
        function: Callable[[Any, WorkerProgress], Any],
        jobs: Iterable[Any],
        *,
        total: int | None = None,
        progress: ProgressCallback | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> list[Any]:
        """Execute numerical work in input order.

        Args:
            function: Worker receiving payload and reporter.
            jobs: Detached work items.
            total: Optional known work count.
            progress: Caller progress callback.
            cancel_event: Cooperative cancellation event.

        Returns:
            Detached outputs in input order.
        """
        ...


class ExecutionKwargs(TypedDict):
    """Arguments accepted by processor functions supporting CPU execution."""

    cpu: Executor | None
    num_cpus: int | None
    progress: ProgressCallback | None
    cancel_event: CancelEvent | None


@dataclass(frozen=True)
class ExecutionContext:
    """Share executor, progress and cancellation settings across processing calls.

    Use ``processor(..., **context.kwargs())``. The context never starts or closes
    its executor. Configure worker count on a borrowed executor, or supply
    num_cpus to let each processor own its default executor.
    """

    cpu: Executor | None = None
    num_cpus: int | None = None
    progress: ProgressCallback | None = None
    cancel_event: CancelEvent | None = None

    def __post_init__(self) -> None:
        if self.cpu is not None and not callable(getattr(self.cpu, "map", None)):
            raise TypeError("cpu must provide an Executor-compatible map method")
        if self.cpu is not None and self.num_cpus is not None:
            raise ValueError("Set num_cpus on the supplied executor, or omit cpu")
        if self.num_cpus is not None and (
            isinstance(self.num_cpus, bool) or not isinstance(self.num_cpus, int) or self.num_cpus < 1
        ):
            raise ValueError("num_cpus must be a positive integer")

    def check_cancelled(self) -> None:
        """Raise before any work when the caller has requested cancellation."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CancelledError("Processing cancelled")

    def kwargs(self) -> ExecutionKwargs:
        """Return fresh processor arguments after checking cancellation.

        Returns:
            Named execution arguments for a processor call.
        """
        self.check_cancelled()
        return {
            "cpu": self.cpu,
            "num_cpus": self.num_cpus,
            "progress": self.progress,
            "cancel_event": self.cancel_event,
        }

    def resolve(
        self,
        *,
        cpu: Executor | None = None,
        num_cpus: int | None = None,
        progress: ProgressCallback | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> ExecutionContext:
        """Fill unspecified fields, rejecting ambiguous duplicate configuration.

        None means unspecified. To replace or clear existing policy, construct a
        new context (or use dataclasses.replace) explicitly.

        Args:
            cpu: Optional borrowed executor.
            num_cpus: Worker count for an owned executor.
            progress: Progress callback.
            cancel_event: Cooperative cancellation event.

        Returns:
            Validated context containing merged settings.
        """
        supplied = {"cpu": cpu, "num_cpus": num_cpus, "progress": progress, "cancel_event": cancel_event}
        for key, value in supplied.items():
            if value is not None and getattr(self, key) is not None:
                raise ValueError(f"{key} was supplied both in the execution context and explicitly")
        return replace(
            self,
            cpu=self.cpu if cpu is None else cpu,
            num_cpus=self.num_cpus if num_cpus is None else num_cpus,
            progress=self.progress if progress is None else progress,
            cancel_event=self.cancel_event if cancel_event is None else cancel_event,
        )
