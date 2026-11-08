"""Reusable caller execution policy without owning or closing borrowed executors."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sized
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from queue import Queue
from threading import Event
from typing import Any, Protocol, TypedDict

from .cpu import CpuExecutor
from .types import CancelEvent, ProgressCallback, ProgressUpdate


class ProgressReporter(Protocol):
    def check_cancelled(self) -> None:
        """Raise CancelledError when cancellation has been requested."""
        ...

    def report(self, completed: float, total: float, description: str = "") -> None:
        """Report worker progress for dispatch on the calling thread.

        Args:
            completed: Completed.
            total: Known number of jobs; required for unsized iterables.
            description: Description.
        """
        ...


class Executor(Protocol):
    """Ordered worker map compatible with CpuExecutor; ownership stays with callers.

    Implementations must supply a WorkerProgress-compatible reporter, forward
    progress on the calling thread, and honour cooperative cancellation. Workers
    accept (payload, reporter); process executors require picklable workers.
    """

    def map[InputT, OutputT](
        self,
        function: Callable[[InputT, ProgressReporter], OutputT],
        jobs: Iterable[InputT],
        *,
        total: int | None = None,
        progress: ProgressCallback | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> list[OutputT]:
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

    def imap[InputT, OutputT](
        self,
        function: Callable[[InputT, ProgressReporter], OutputT],
        jobs: Iterable[InputT],
        *,
        total: int | None = None,
        progress: ProgressCallback | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> Iterator[OutputT]:
        """Yield outputs in input order with bounded pending work and buffering.

        Args:
            function: Numerical worker accepting a payload and progress reporter.
            jobs: Finite iterable of detached work items.
            total: Known number of jobs; required for unsized iterables.
            progress: Progress callback invoked on the calling thread.
            cancel_event: Cooperative cancellation event.

        Yields:
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
    chunk_size: int = 16
    progress: ProgressCallback | None = None
    cancel_event: CancelEvent | None = None

    def describe(self) -> dict[str, object]:
        """Capture execution policy without retaining executors or callbacks.

        Returns:
            Requested executor, worker count and frame chunk size. A null worker
            count means the processor selects its documented default.
        """
        return {
            "executor": type(self.cpu).__name__ if self.cpu is not None else "processor-default",
            "num_cpus": getattr(self.cpu, "num_cpus", self.num_cpus),
            "chunk_size": self.chunk_size,
        }

    def __post_init__(self) -> None:
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if self.cpu is not None and (not callable(getattr(self.cpu, "map", None))):
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


class _SerialReporter:
    def __init__(self, context: ExecutionContext, completed: int, total: int) -> None:
        self.context, self.completed, self.total = (context, completed, total)

    def check_cancelled(self) -> None:
        """Check cancelled."""
        self.context.check_cancelled()

    def report(self, completed: float, total: float, description: str = "") -> None:
        """Report.

        Args:
            completed: Completed.
            total: Total.
            description: Description.
        """
        self.check_cancelled()
        fraction = min(1.0, max(0.0, completed / total)) if total else 0.0
        if self.context.progress:
            self.context.progress(ProgressUpdate(self.completed + fraction, self.total, description))


class SerialExecutor:
    """Run ordered jobs in the calling thread without serialization."""

    def imap[InputT, OutputT](
        self,
        function: Callable[[InputT, ProgressReporter], OutputT],
        jobs: Iterable[InputT],
        *,
        total: int | None = None,
        progress: ProgressCallback | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> Iterator[OutputT]:
        """Imap.

        Args:
            function: Function.
            jobs: Jobs.
            total: Total.
            progress: Progress.
            cancel_event: Cancel event.

        Yields:
            Detached outputs in input order.
        """
        if total is None:
            if not isinstance(jobs, Sized):
                raise ValueError("Supply total for an iterable without a length")
            total = len(jobs)
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("Total must be a nonnegative integer")
        context = ExecutionContext(progress=progress, cancel_event=cancel_event)
        context.check_cancelled()
        count = 0
        for count, job in enumerate(jobs, 1):
            if count > total:
                raise ValueError("Job iterable contains more jobs than total")
            reporter = _SerialReporter(context, count - 1, total)
            reporter.check_cancelled()
            value = function(job, reporter)
            reporter.report(1, 1, "Job complete")
            context.check_cancelled()
            yield value
        if count != total:
            raise ValueError("Job iterable contains fewer jobs than total")
        if progress:
            progress(ProgressUpdate(total, total, "Complete"))
        context.check_cancelled()

    def map[InputT, OutputT](
        self,
        function: Callable[[InputT, ProgressReporter], OutputT],
        jobs: Iterable[InputT],
        *,
        total: int | None = None,
        progress: ProgressCallback | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> list[OutputT]:
        """Collect ordered worker results.

        Args:
            function: Worker receiving a payload and reporter.
            jobs: Input payloads.
            total: Optional expected job count.
            progress: Caller-thread progress callback.
            cancel_event: Cooperative cancellation event.

        Returns:
            Results in input order.
        """
        return list(self.imap(function, jobs, total=total, progress=progress, cancel_event=cancel_event))

    def close(self) -> None:
        """Close."""
        pass

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _ThreadReporter:
    def __init__(
        self, events: Queue[tuple[int, int, float, str]], stop: Event, run_id: int, index: int
    ) -> None:
        self.events, self.stop, self.run_id, self.index = (events, stop, run_id, index)

    def check_cancelled(self) -> None:
        """Check cancelled."""
        if self.stop.is_set():
            raise CancelledError("Thread processing cancelled")

    def report(self, completed: float, total: float, description: str = "") -> None:
        """Report.

        Args:
            completed: Completed.
            total: Total.
            description: Description.
        """
        self.check_cancelled()
        fraction = min(1.0, max(0.0, completed / total)) if total else 1.0
        self.events.put((self.run_id, self.index, fraction, description))


class ThreadExecutor(CpuExecutor):
    """Bounded ordered worker threads; progress is dispatched by the calling thread."""

    _worker_limit: int | None = None

    def _ensure_pool(self) -> None:
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            from queue import Queue
            from threading import Event

            self._events, self._stop = (Queue(), Event())
            self._pool = ThreadPoolExecutor(max_workers=self.num_cpus)

    def _submit(self, function: Any, payload: Any, run_id: int, index: int) -> Any:
        reporter = _ThreadReporter(self._events, self._stop, run_id, index)

        def execute() -> Any:
            """Execute.

            Returns:
                The computed value.
            """
            reporter.check_cancelled()
            value = function(payload, reporter)
            reporter.check_cancelled()
            return value

        return self._pool.submit(execute)

    def close(self) -> None:
        """Close."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = self._events = self._stop = None


def _record_execution[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Attach requested execution tuning to detached scientific results."""
    from functools import wraps
    from typing import cast

    def _attach(value: Any, policy: dict[str, object]) -> Any:
        from .types.results import ResultIdentity

        if isinstance(value, ResultIdentity):
            return replace(value, execution=policy)
        if isinstance(value, dict):
            return {key: _attach(item, policy) for key, item in value.items()}
        return value

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Run a processor and attach execution provenance.

        Args:
            args: Positional scientific inputs.
            kwargs: Named scientific and execution inputs.

        Returns:
            Detached result with execution provenance.
        """
        result = function(*args, **kwargs)
        context = cast(ExecutionContext, kwargs.get("context") or ExecutionContext())
        return cast(R, _attach(result, context.describe()))

    return wrapped
