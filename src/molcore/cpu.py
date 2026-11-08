"""Reusable spawn-based CPU work with bounded submission and caller progress."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution import ProgressReporter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution import ProgressReporter
    from .types import CancelEvent
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Callable, Iterable, Sized
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ProcessPoolExecutor, wait
from typing import TYPE_CHECKING, Any

from ._logging import logged
from .types.processing import ProgressUpdate

if TYPE_CHECKING:
    from types import TracebackType
_EVENTS: Any = None
_STOP: Any = None


def _initialize(events: Any, stop: Any) -> None:
    """Initialize.

    Args:
        events: Multiprocessing queue carrying structured progress events.
        stop: Shared cancellation event observed by spawned workers.
    """
    global _EVENTS, _STOP
    _EVENTS, _STOP = (events, stop)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"


class WorkerProgress:
    """Cooperative worker progress reporter supplied to every CPU function.

    Args:
        run_id: Executor run identifier.
        job_id: Job position in its input stream.
    """

    def __init__(self, run_id: int, job_id: int) -> None:
        """Initialise the object and validate its configuration.

        Args:
            run_id: Identifier distinguishing this processing run from previous results.
            job_id: Zero-based position of this job in the submitted input stream.
        """
        self.run_id, self.job_id = (run_id, job_id)
        self._last = 0.0

    @logged
    def check_cancelled(self) -> None:
        """Raise CancelledError when the calling process requests cancellation."""
        if _STOP.is_set():
            raise CancelledError("CPU processing cancelled")

    @logged
    def report(self, completed: float, total: float, description: str = "") -> None:
        """Report fractional progress and check cooperative cancellation.

        Args:
            completed: Completed units in this job's current stage.
            total: Total units, positive unless the job is empty.
            description: Current stage description.
        """
        self.check_cancelled()
        fraction = 1.0 if total == 0 else min(1.0, max(0.0, completed / total))
        now = time.monotonic()
        if now - self._last >= 0.02 or fraction in (0.0, 1.0):
            _EVENTS.put((self.run_id, self.job_id, fraction, description))
            self._last = now


def _execute(function: Callable[..., Any], payload: tuple, run_id: int, job_id: int) -> Any:
    """Execute.

    Args:
        function: Function used by this operation.
        payload: Serializable worker inputs; large arrays remain bounded to a job or image chunk.
        run_id: Identifier distinguishing this processing run from previous results.
        job_id: Zero-based position of this job in the submitted input stream.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    from threadpoolctl import threadpool_limits

    reporter = WorkerProgress(run_id, job_id)
    reporter.check_cancelled()
    with threadpool_limits(limits=1):
        result = function(payload, reporter)
    reporter.check_cancelled()
    return result


class CpuExecutor:
    """Reusable CPU executor with parent-only callbacks and ordered results.

    Args:
        num_cpus: Positive worker count; None uses available CPUs minus one.
        max_pending: Maximum submitted jobs, defaulting to twice the worker count.

    Notes:
        Worker functions must be importable and accept (payload, reporter).
        Use a context manager and a main guard in standalone Python programs.
        A single configured CPU still runs in a spawned child process.
    """

    _worker_limit: int | None = 61 if os.name == "nt" else None
    _pool: Any
    _events: Any
    _stop: Any

    def __init__(self, num_cpus: int | None = None, *, max_pending: int | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            num_cpus: Worker process count; None uses the executor default.
            max_pending: Max pending used by this operation.
        """
        available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        default = max(1, available - 1)
        if self._worker_limit is not None:
            default = min(default, self._worker_limit)
        self.num_cpus = default if num_cpus is None else num_cpus
        for value in (self.num_cpus, max_pending if max_pending is not None else 1):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("CPU count and pending limit must be positive integers")
        if self._worker_limit is not None and self.num_cpus > self._worker_limit:
            raise ValueError("Windows ProcessPoolExecutor supports at most 61 CPUs")
        self.max_pending = max_pending or 2 * self.num_cpus
        self._pool = self._events = self._stop = None
        self._run_id, self._busy = (0, False)

    def _ensure_pool(self) -> None:
        """Ensure pool."""
        if self._pool is None:
            context = mp.get_context("spawn")
            self._events, self._stop = (context.Queue(), context.Event())
            self._pool = ProcessPoolExecutor(
                max_workers=self.num_cpus,
                mp_context=context,
                initializer=_initialize,
                initargs=(self._events, self._stop),
            )

    def _submit(self, function: Any, payload: Any, run_id: int, index: int) -> Any:
        return self._pool.submit(_execute, function, payload, run_id, index)

    @logged
    def map[InputT, OutputT](
        self,
        function: Callable[[InputT, ProgressReporter], OutputT],
        jobs: Iterable[InputT],
        *,
        total: int | None = None,
        progress: Callable[[ProgressUpdate], None] | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> list[OutputT]:
        """Collect imap outputs in input order.

        Args:
            function: Function.
            jobs: Jobs.
            total: Total.
            progress: Progress.
            cancel_event: Cancel event.


        Returns:
            The computed value.
        """
        return list(self.imap(function, jobs, total=total, progress=progress, cancel_event=cancel_event))

    def imap[InputT, OutputT](
        self,
        function: Callable[[InputT, ProgressReporter], OutputT],
        jobs: Iterable[InputT],
        *,
        total: int | None = None,
        progress: Callable[[ProgressUpdate], None] | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> Iterator[OutputT]:
        """Yield ordered results with bounded pending jobs and completed buffers.

        Args:
            function: Function.
            jobs: Jobs.
            total: Total.
            progress: Progress.
            cancel_event: Cancel event.

        Yields:
            Detached outputs in input order.
        """
        if self._busy:
            raise RuntimeError("An executor cannot run overlapping map calls")
        if total is None:
            if not isinstance(jobs, Sized):
                raise ValueError("Supply total for an iterable without a length")
            total = len(jobs)
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("Total must be a nonnegative integer")

        def check() -> None:
            """Check."""
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("CPU processing cancelled")

        check()
        if total == 0:
            sentinel = object()
            if next(iter(jobs), sentinel) is not sentinel:
                raise ValueError("Job iterable contains more jobs than total")
            if progress:
                progress(ProgressUpdate(0, 0, "Complete"))
            check()
            return
        self._ensure_pool()
        self._busy = True
        self._run_id += 1
        run_id = self._run_id
        pending: dict[Future[OutputT], int] = {}
        ready: dict[int, OutputT] = {}
        fractions: dict[int, float] = {}
        iterator = iter(jobs)
        submitted = delivered = completed_count = 0

        def emit(description: str) -> None:
            """Emit.

            Args:
                description: Description.
            """
            check()
            if progress:
                progress(ProgressUpdate(completed_count + sum(fractions.values()), total, description))
            check()

        try:
            emit("Starting")
            while delivered < total:
                check()
                while len(pending) + len(ready) < self.max_pending and submitted < total:
                    try:
                        payload = next(iterator)
                    except StopIteration as exc:
                        raise ValueError("Job iterable contains fewer jobs than total") from exc
                    check()
                    future = self._submit(function, payload, run_id, submitted)
                    pending[future] = submitted
                    fractions[submitted] = 0.0
                    submitted += 1
                completed, _ = wait(pending, timeout=0.03, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        event_run, index, fraction, description = self._events.get_nowait()
                    except queue.Empty:
                        break
                    if event_run == run_id and index in fractions:
                        fractions[index] = max(fractions[index], fraction)
                        emit(description)
                for future in completed:
                    index = pending.pop(future)
                    ready[index] = future.result()
                    fractions.pop(index)
                    completed_count += 1
                    emit("Job complete")
                while delivered in ready:
                    check()
                    value = ready.pop(delivered)
                    delivered += 1
                    yield value
            check()
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("Job iterable contains more jobs than total")
            emit("Complete")
        except BaseException:
            self._stop.set()
            for future in pending:
                future.cancel()
            self.close()
            raise
        finally:
            self._busy = False

    @logged
    def close(self) -> None:
        """Wait for workers and release all process/queue resources."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
            while True:
                try:
                    self._events.get_nowait()
                except queue.Empty:
                    break
            self._events.close()
            self._events.join_thread()
            self._events = self._stop = None

    def __enter__(self) -> CpuExecutor:
        """Enter the managed resource lifetime.

        Returns:
            Computed CpuExecutor value.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release managed resources on exit.

        Args:
            exc_type: Exc type used by this operation.
            exc_value: Exc value used by this operation.
            traceback: Traceback used by this operation.
        """
        self.close()
