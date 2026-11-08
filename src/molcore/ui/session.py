"""Host-owned Qt session and cancellable background coordination."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import CancelledError
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject, QThread, Signal, Slot

from .._logging import logged
from ..cpu import CpuExecutor
from ..staging import clone_dataset, clone_experiment, experiment_state_token, state_token
from ..types import CoordinateSpace, Dataset, Experiment, StaleResultError

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtWidgets import QWidget
_RUNNING = set()
logger = logging.getLogger(__name__)


class _Job(QThread):
    update = Signal(object)

    def __init__(self, compute: Callable[..., Any], model: Dataset) -> None:
        """Initialise the object and validate its configuration.

        Args:
            compute: Background callable receiving a detached model, progress callback and cancellation event.
            model: Model or fitting method selected by the caller.
        """
        super().__init__()
        self.compute, self.model = (compute, model)
        self.cancel_event = threading.Event()
        self.result, self.failure = (None, None)
        self.published = False
        self.publication_token = None

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
    """Share one dataset, selection and background job across embedded tools.

    Args:
        dataset: Borrowed host model; None creates an owned empty dataset.
        parent: Optional QObject owner. The host owns the application/event loop.

    Attributes:
        dataset: Current scientific model.
        acquisition_id: Selected acquisition, or None for an empty model.
        channel: Selected logical channel, or None.
        space: Display coordinate system; corrected by default.
        camera_frame: Current original camera frame.
        num_cpus: Worker count; None selects CpuExecutor's default.
    """

    dataset_changed = Signal(object)
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
    _model_change = Signal(object, object)

    def __init__(self, dataset: Dataset | Experiment | None = None, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            dataset: Scientific model supplying acquisitions and result tables.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(parent)
        self._empty_dataset = Dataset(name="No dataset selected")
        self.experiment = dataset if isinstance(dataset, Experiment) else Experiment()
        if not isinstance(dataset, Experiment):
            self.experiment.add_dataset(dataset if dataset is not None else Dataset())
        self.dataset_id = next(iter(self.experiment.datasets), None)
        self.dataset = (
            self.experiment.datasets[self.dataset_id] if self.dataset_id is not None else self._empty_dataset
        )
        self._owned = dataset is None
        self.acquisition_id, self.channel = (None, None)
        self.sequence_id = None
        self.space, self.camera_frame = (CoordinateSpace.CORRECTED, 0)
        self._num_cpus = None
        self._job = None
        self._closing, self.dirty = (False, False)
        self.publication_current = True
        self._unsubscribe_model = None
        self._model_change.connect(self._on_model_change)
        self._subscribe_model()
        self._select_first()

    def _subscribe_model(self) -> None:
        if self._unsubscribe_model is not None:
            self._unsubscribe_model()
        model = self.dataset
        self._unsubscribe_model = model.subscribe(lambda change: self._model_change.emit(model, change))

    @Slot(object, object)
    def _on_model_change(self, model: Dataset, change: Any) -> None:
        if model is self.dataset and (not self._closing):
            self.changed()

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
        CpuExecutor(num_cpus=value)
        self._num_cpus = value
        self.options_changed.emit()

    def _gui(self) -> None:
        """Gui."""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Session changes must run on its GUI thread")

    def _select_first(self) -> None:
        """Select first."""
        self.acquisition_id = next(iter(self.dataset.acquisitions), None)
        self.camera_frame = 0
        self.channel = None
        if self.acquisition_id is not None:
            self.channel = next(iter(self.dataset.acquisitions[self.acquisition_id].channels))
            view = self.dataset.acquisitions[self.acquisition_id].channels[self.channel]
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
        view = self.dataset.acquisitions[acquisition_id].channels[channel]
        if sequence_id is not None:
            sequence = self.dataset.sequences[sequence_id]
            if acquisition_id not in sequence.offsets:
                raise ValueError("Acquisition does not belong to the sequence")
        self.acquisition_id, self.channel, self.sequence_id = (acquisition_id, channel, sequence_id)
        if space is not None:
            self.space = CoordinateSpace(space)
        self.camera_frame = int(view.source_frames[0]) if len(view) else 0
        self.selection_changed.emit()

    @logged
    def select_dataset(self, dataset_id: str) -> None:
        """Select a probing round and reset acquisition/channel/frame selection.

        Args:
            dataset_id: Probing-round identifier.
        """
        self._gui()
        if self.busy:
            raise RuntimeError("Finish the active job before switching datasets")
        self.dataset = self.experiment.datasets[dataset_id]
        self._subscribe_model()
        self.dataset_id = dataset_id
        self.sequence_id = None
        self._select_first()
        self.dataset_changed.emit(self.dataset)
        self.selection_changed.emit()

    def set_experiment(self, experiment: Experiment, *, owns_experiment: bool = False) -> None:
        """Replace the full experiment, closing only session-owned resources.

        Args:
            experiment: Experiment containing the probing rounds.
            owns_experiment: Close the experiment sources on replacement or session shutdown.
        """
        self._gui()
        if self.busy:
            raise RuntimeError("Cancel and finish the active job before replacing the experiment")
        if self._owned and self.experiment is not experiment:
            retained = {id(a.source) for d in experiment.datasets.values() for a in d.acquisitions.values()}
            for dataset in self.experiment.datasets.values():
                for acquisition in dataset.acquisitions.values():
                    if id(acquisition.source) not in retained:
                        acquisition.close()
        self.experiment, self._owned = (experiment, owns_experiment)
        self.dataset_id = next(iter(experiment.datasets), None)
        self.dataset = (
            experiment.datasets[self.dataset_id] if self.dataset_id is not None else self._empty_dataset
        )
        self.sequence_id, self.dirty = (None, False)
        self._subscribe_model()
        self._select_first()
        self.experiment_changed.emit(experiment)
        self.dataset_changed.emit(self.dataset)
        self.selection_changed.emit()

    def set_dataset(self, dataset: Dataset, *, owns_dataset: bool = False) -> None:
        """Replace an idle session's model, closing only session-owned models.

        Args:
            dataset: New Dataset to expose to every attached widget.
            owns_dataset: Close this model on subsequent replacement/shutdown.
        """
        experiment = Experiment()
        experiment.add_dataset(dataset)
        self.set_experiment(experiment, owns_experiment=owns_dataset)

    @logged
    def set_frame(self, camera_frame: int) -> None:
        """Set a measured original camera frame for current-frame previews.

        Args:
            camera_frame: Frame belonging to the selected logical channel.
        """
        self._gui()
        self.dataset.acquisitions[self.acquisition_id].channels[self.channel].channel_index(camera_frame)
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
        experiment_scope: bool = False,
        publication: bool = False,
    ) -> None:
        """Compute against a detached snapshot and deliver completion on the GUI thread.

        Args:
            compute: Callable receiving (detached_model, progress, cancel_event).
            commit: Optional callable (live_model, result) applying scientific data.
            completed: Optional GUI callback receiving the successful result.
            check_stale: Reject completion after any protected model change.
            experiment_scope: Compute against all rounds rather than the selected dataset.
            publication: Supply a fourth compute argument guarding atomic file publication against live edits.

        Raises:
            RuntimeError: Another job is active or the session is shutting down.
        """
        self._gui()
        if self.busy or self._closing:
            raise RuntimeError("A session accepts one job at a time and cannot run after shutdown")
        original = self.experiment if experiment_scope else self.dataset
        token = experiment_state_token if experiment_scope else state_token
        clone = clone_experiment if experiment_scope else clone_dataset
        expected = token(original)
        job = _Job(compute, clone(original) if publication or experiment_scope else original.snapshot())
        if publication:

            @contextmanager
            def _publish_guard() -> Any:
                with original._lock, ExitStack() as stack:
                    if experiment_scope:
                        for dataset in sorted(original.datasets.values(), key=lambda d: d.id):
                            stack.enter_context(dataset._lock)
                    current = self.experiment if experiment_scope else self.dataset
                    if current is not original or token(original) != expected:
                        raise StaleResultError("Live inputs changed; files were not published")
                    if job.cancel_event.is_set():
                        raise CancelledError()
                    yield
                    pairs = (
                        [(original.datasets[k], d) for k, d in job.model.datasets.items()]
                        if experiment_scope
                        else [(original, job.model)]
                    )
                    for live, saved in pairs:
                        live.crops._saved = saved.crops._saved.copy()
                        live.crops._photon_saved = saved.crops._photon_saved.copy()
                        for key, acquisition in saved.acquisitions.items():
                            live._replace_source(key, acquisition.source)
                    job.published = True
                    job.publication_token = token(original)

            job.compute = lambda model, progress, cancel: compute(model, progress, cancel, _publish_guard)
        self._completion = (original, expected, commit, completed, check_stale, experiment_scope)
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
        original, expected, commit, completed, check_stale, experiment_scope = self._completion
        try:
            if job.failure is not None:
                raise job.failure
            if (job.cancel_event.is_set() or self._closing) and (not job.published):
                raise CancelledError()
            current = self.experiment if experiment_scope else self.dataset
            token = experiment_state_token if experiment_scope else state_token
            self.publication_current = (
                current is original and token(original) == job.publication_token if job.published else True
            )
            if check_stale and (not job.published):
                if current is not original:
                    raise StaleResultError("Model changed during computation; results were discarded")
                result_token = getattr(job.result, "expected", None)
                from ..revisions import DependencyToken

                if isinstance(result_token, DependencyToken) and isinstance(original, Dataset):
                    result_token.validate(original)
                elif token(original) != expected:
                    raise StaleResultError("Model changed during computation; results were discarded")
            if commit is not None:
                revision = original.revision if isinstance(original, Dataset) else None
                commit(original, job.result)
                if revision is None or original.revision == revision:
                    self.changed()
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
        if self._unsubscribe_model is not None:
            self._unsubscribe_model()
            self._unsubscribe_model = None
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
