"""Host-owned Qt session and cancellable background coordination."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import CancelledError
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject, QThread, Signal, Slot

from .._logging import logged
from ..cpu import CpuExecutor
from ..staging import clone_experiment, state_token
from ..types import CoordinateSpace, Experiment, StaleResultError

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtWidgets import QWidget

# Keep threads alive even if a host destroys its dock widget during cancellation.
_RUNNING = set()
logger = logging.getLogger(__name__)


class _Job(QThread):
    update = Signal(object)

    def __init__(self, compute: Callable[..., Any], model: Experiment) -> None:
        """Initialise the object and validate its configuration.

        Args:
            compute: Background callable receiving a detached model, progress callback and cancellation event.
            model: Model or fitting method selected by the caller.
        """
        super().__init__()
        self.compute, self.model = compute, model
        self.cancel_event = threading.Event()
        self.result, self.failure = None, None

    @logged
    def run(self) -> None:
        """Run."""
        try:
            if self.cancel_event.is_set():
                raise CancelledError()
            self.result = self.compute(self.model, self.update.emit, self.cancel_event)
        except Exception as exc:
            logger.debug("Background job failed", exc_info=True)
            self.failure = exc


class UiSession(QObject):
    """Share one experiment, selection and background job across embedded tools.

    Args:
        experiment: Borrowed host model; None creates an owned empty experiment.
        parent: Optional QObject owner. The host owns the application/event loop.

    Attributes:
        experiment: Current scientific model.
        acquisition_id: Selected acquisition, or None for an empty model.
        channel: Selected logical channel, or None.
        space: Selected coordinate system.
        camera_frame: Current original camera frame.
        num_cpus: Worker count; None selects CpuExecutor's default.
    """

    experiment_changed = Signal(object)
    selection_changed = Signal()
    frame_changed = Signal()
    options_changed = Signal()
    results_changed = Signal()
    busy_changed = Signal(bool)
    progress = Signal(object)
    error = Signal(str)
    preview_changed = Signal(object)
    finished = Signal()
    shutdown_finished = Signal()

    def __init__(self, experiment: Experiment | None = None, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            experiment: Scientific model supplying acquisitions and result tables.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(parent)
        self.experiment = experiment if experiment is not None else Experiment()
        self._owned = experiment is None
        self.acquisition_id, self.channel = None, None
        self.sequence_id = None
        self.space, self.camera_frame = CoordinateSpace.RAW, 0
        self._num_cpus = None
        self._job = None
        self._closing, self.dirty = False, False
        self._select_first()

    @property
    @logged
    def busy(self) -> bool:
        """Whether a job is computing, applying or cleaning up.

        Returns:
            True while a computation or its completion is active.
        """
        return self._job is not None

    @property
    @logged
    def num_cpus(self) -> int | None:
        """Configured CPU process count, or None for automatic selection.

        Returns:
            Worker count, or None for automatic selection.
        """
        return self._num_cpus

    @num_cpus.setter
    @logged
    def num_cpus(self, value: Any) -> int | None:
        """Num cpus.

        Args:
            value: Value supplied by the caller or connected control.

        Returns:
            Detached result produced by this background computation.
        """
        CpuExecutor(num_cpus=value)  # Validate without starting any processes.
        self._num_cpus = value
        self.options_changed.emit()

    def _gui(self) -> None:
        """Gui."""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Session changes must run on its GUI thread")

    def _select_first(self) -> None:
        """Select first."""
        self.acquisition_id = next(iter(self.experiment.acquisitions), None)
        self.channel = None
        if self.acquisition_id is not None:
            self.channel = next(iter(self.experiment.acquisitions[self.acquisition_id].channels))
            view = self.experiment.acquisitions[self.acquisition_id].channels[self.channel]
            self.camera_frame = int(view.source_frames[0]) if len(view) else 0

    @logged
    def select(
        self,
        acquisition_id: str,
        channel: str,
        *,
        space: CoordinateSpace | str | None = None,
        sequence_id: str | None = None,
    ) -> None:
        """Select an existing logical channel and optional explicit sequence.

        Args:
            acquisition_id: Acquisition identifier.
            channel: Logical channel identifier.
            space: Optional raw, registered or corrected view.
            sequence_id: Optional explicit sequence containing the acquisition.
        """
        self._gui()
        view = self.experiment.acquisitions[acquisition_id].channels[channel]
        if sequence_id is not None:
            sequence = self.experiment.sequences[sequence_id]
            if acquisition_id not in sequence.offsets:
                raise ValueError("Acquisition does not belong to the sequence")
        self.acquisition_id, self.channel, self.sequence_id = acquisition_id, channel, sequence_id
        if space is not None:
            self.space = CoordinateSpace(space)
        self.camera_frame = int(view.source_frames[0]) if len(view) else 0
        self.selection_changed.emit()

    @logged
    def set_experiment(self, experiment: Experiment, *, owns_experiment: bool = False) -> None:
        """Replace an idle session's model, closing only session-owned models.

        Args:
            experiment: New Experiment to expose to every attached widget.
            owns_experiment: Close this model on subsequent replacement/shutdown.
        """
        self._gui()
        if self.busy:
            raise RuntimeError("Cancel and finish the active job before replacing the experiment")
        if self._owned and self.experiment is not experiment:
            self.experiment.close()
        self.experiment, self._owned = experiment, owns_experiment
        self.sequence_id, self.dirty = None, False
        self._select_first()
        self.experiment_changed.emit(experiment)
        self.selection_changed.emit()

    @logged
    def set_frame(self, camera_frame: int) -> None:
        """Set a measured original camera frame for current-frame previews.

        Args:
            camera_frame: Frame belonging to the selected logical channel.
        """
        self._gui()
        self.experiment.acquisitions[self.acquisition_id].channels[self.channel].channel_index(camera_frame)
        self.camera_frame = int(camera_frame)
        self.frame_changed.emit()

    @logged
    def changed(self) -> None:
        """Notify tools after a host or successful transaction changes the model."""
        self._gui()
        self.dirty = True
        self.results_changed.emit()

    @logged
    def submit(
        self,
        compute: Callable[..., Any],
        *,
        commit: Callable[..., Any] | None = None,
        completed: Callable[[Any], None] | None = None,
        check_stale: bool = True,
    ) -> None:
        """Compute against a detached snapshot and deliver completion on the GUI thread.

        Args:
            compute: Callable receiving (detached_model, progress, cancel_event).
            commit: Optional callable (live_model, result) applying scientific data.
            completed: Optional GUI callback receiving the successful result.
            check_stale: Reject completion after any protected model change.

        Raises:
            RuntimeError: Another job is active or the session is shutting down.
        """
        self._gui()
        if self.busy or self._closing:
            raise RuntimeError("A session accepts one job at a time and cannot run after shutdown")
        original, expected = self.experiment, state_token(self.experiment)
        job = _Job(compute, clone_experiment(original))
        self._completion = (original, expected, commit, completed, check_stale)
        self._job = job
        _RUNNING.add(job)
        job.update.connect(self.progress)
        job.finished.connect(self._complete)
        job.finished.connect(lambda: _RUNNING.discard(job))
        job.start()
        self.busy_changed.emit(True)

    @Slot()
    @logged
    def _complete(self) -> None:
        """Complete."""
        job = self._job
        original, expected, commit, completed, check_stale = self._completion
        try:
            if job.failure is not None:
                raise job.failure
            if job.cancel_event.is_set() or self._closing:
                raise CancelledError()
            if check_stale and (self.experiment is not original or state_token(original) != expected):
                raise StaleResultError("Model changed during computation; results were discarded")
            if commit is not None:
                commit(original, job.result)
                self.changed()
            # Job is now fully stopped; callbacks may replace the model.
            self._job = None
            if completed is not None:
                completed(job.result)
        except CancelledError:
            logger.debug("Session job cancelled; active results preserved")
        except Exception as exc:
            logger.debug("Session completion rejected or failed", exc_info=True)
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self._job = None
            job.deleteLater()
            self.busy_changed.emit(False)
            self.finished.emit()
            if self._closing:
                self._close_owned()

    @logged
    def cancel(self) -> None:
        """Request cancellation; active native kernels may finish before stopping."""
        self._gui()
        if self._job is not None:
            self._job.cancel_event.set()

    @logged
    def shutdown(self) -> None:
        """Cancel running work and release owned models after asynchronous cleanup."""
        self._gui()
        if self._closing:
            return
        self._closing = True
        if self.busy:
            self.cancel()
        else:
            self._close_owned()

    def _close_owned(self) -> None:
        """Close owned."""
        if self._owned:
            self.experiment.close()
            self._owned = False
        self.shutdown_finished.emit()
