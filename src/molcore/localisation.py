"""Model-based Picasso detection and CPU Gaussian localisation."""

from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import replace
from threading import Event, Lock
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from ._processing import backend_version, check_snapshot, executor
from .backends import BackendRegistry, LocalisationBackend, resolve_backend
from .cpu import CpuExecutor
from .execution import ExecutionContext, _record_execution
from .snapshots import DatasetSnapshot, processor_model
from .types.dataset import Dataset
from .types.images import ArraySource
from .types.processing import (
    CameraSettings,
    DetectionResult,
    LocalisationResult,
    LocalisationSettings,
    ProgressUpdate,
)
from .types.selections import PartitionSelection

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .execution import ProgressReporter
    from .types import CancelEvent, ImageSource, ProgressCallback
    from .types.acquisition import ChannelView
_FIT_SPOT_BYTES = 32 * 1024**2


class _WorkingChannel:
    """A working pixel source with the original channel's frame clock."""

    def __init__(self, channel: ChannelView, source: ImageSource) -> None:
        if source.shape != channel.shape:
            raise ValueError("Working source shape must match the logical channel")
        self.source = source
        self.source_frames = channel.source_frames
        self.shape = source.shape
        self.dtype = source.dtype

    def __getitem__(self, index: Any) -> np.ndarray:
        return self.source[index]

    def __len__(self) -> int:
        return self.shape[0]


class _SelectedFrames:
    """Expose a selection of camera frames without loading the entire stack."""

    def __init__(self, view: ChannelView | _WorkingChannel, indices: np.ndarray) -> None:
        self.view = view
        self.indices = indices
        self.dtype = view.dtype

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.view[int(self.indices[index])]


def _selected_images(view: ChannelView | _WorkingChannel, frames: np.ndarray) -> np.ndarray | _SelectedFrames:
    """Use a NumPy view for regularly spaced in-memory frames."""
    if isinstance(view, _WorkingChannel):
        return _SelectedFrames(view, np.searchsorted(view.source_frames, frames))
    source = view.acquisition.source
    if type(source) is ArraySource:
        step = int(frames[1] - frames[0]) if len(frames) > 1 else 1
        if np.all(np.diff(frames) == step):
            y0, y1, x0, x1 = view.roi
            return source.array[int(frames[0]) : int(frames[-1]) + 1 : step, y0:y1, x0:x1]
    return _SelectedFrames(view, np.searchsorted(view.source_frames, frames))


def _empty(fitted: bool = False) -> dict[str, np.ndarray]:
    """Empty.

    Args:
        fitted: Fitted used by this operation.

    Returns:
        Detached named result columns.
    """
    names = ["x", "y", "net_gradient"]
    if fitted:
        names += ["photons", "bg", "sx", "sy", "lpx", "lpy", "ellipticity", "fit_state", "iterations"]
    return {"frame": np.array([], dtype=np.uint64), **{k: np.array([], dtype=float) for k in names}}


@logged
def _job(payload: tuple, reporter: ProgressReporter) -> tuple:
    """Job.

    Args:
        payload: Serializable worker inputs; large arrays remain bounded to a job or image chunk.
        reporter: Worker progress and cooperative cancellation reporter.

    Returns:
        Computed tuple value.
    """
    images, frames, settings, camera, candidates, fitting = payload
    import pandas as pd
    from picasso import localize

    if fitting:
        from picasso.fitting import gaussfit, seeds
    images.flags.writeable = False
    chunks, failed_fits = ([], 0)
    for index, (image, camera_frame) in enumerate(zip(images, frames, strict=True)):
        reporter.report(
            index, len(frames), "Detecting localisations" if candidates is None else "Extracting spots"
        )
        if candidates is None:
            y, x, gradient = localize.identify_in_frame(image, settings.min_net_gradient, settings.box_size)
            found = pd.DataFrame(
                {"frame": np.zeros(len(x), dtype=np.uint32), "x": x, "y": y, "net_gradient": gradient}
            )
        else:
            selected = candidates["frame"] == camera_frame
            found = pd.DataFrame({k: v[selected] for k, v in candidates.items()})
            found["frame"] = np.zeros(len(found), dtype=np.uint32)
        if len(found) and fitting:
            spots = localize.get_spots(
                image[None],
                found,
                settings.box_size,
                {"Baseline": camera.baseline, "Sensitivity": camera.sensitivity, "Gain": camera.gain},
            )
            initial = seeds.initial_parameters_gauss(spots, settings.box_size)

            def fitted(count: int, index: int = index, spots: Any = spots) -> None:
                """Fitted.

                Args:
                    count: Number of items to allocate or process.
                    index: Zero-based item or logical-channel frame index.
                    spots: Spots used by this operation.
                """
                reporter.report(index + 0.2 + 0.7 * count / len(spots), len(frames), "Fitting localisations")

            theta, chi_square, states, iterations = gaussfit.fit_spots(
                gaussfit.ELLIPTIC,
                spots,
                initial,
                mle=settings.fit_method == "mle",
                tolerance=settings.tolerance,
                max_iterations=settings.max_iterations,
                progress_callback=fitted,
            )
            found = localize.locs_from_fits_gauss(
                found,
                theta,
                settings.box_size,
                em=camera.em,
                mle=settings.fit_method == "mle",
                iterations=iterations,
                chi_square=chi_square,
            )
            found["fit_state"] = states
            found["iterations"] = iterations
            valid = (states == 0) & np.isfinite(found[["x", "y", "photons", "sx", "sy"]]).all(
                axis=1
            ).to_numpy()
            valid &= ((found["photons"] >= 0) & (found["sx"] > 0) & (found["sy"] > 0)).to_numpy()
            failed_fits += int((~valid).sum())
            found = found.loc[valid]
        if len(found):
            found["frame"] = np.full(len(found), camera_frame, dtype=np.uint64)
            chunks.append({k: found[k].to_numpy(copy=True) for k in found.columns})
        reporter.report(index + 1, len(frames), "Localisation chunk complete")
    return (_combine(chunks, fitting), failed_fits)


def _combine(chunks: list[dict[str, np.ndarray]], fitting: bool) -> dict[str, np.ndarray]:
    """Combine.

    Args:
        chunks: Named columns in source order.
        fitting: Fitting used by this operation.

    Returns:
        Detached named result columns.
    """
    chunks = [chunk for chunk in chunks if len(chunk["frame"])]
    if not chunks:
        return _empty(fitting)
    return {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}


def _detect_frames(
    view: ChannelView, frames: np.ndarray, settings: LocalisationSettings, execution: ExecutionContext
) -> dict[str, np.ndarray]:
    """Run Picasso's frame detector on threads and assemble columns in frame order."""
    total = len(frames)
    workers = CpuExecutor(num_cpus=execution.num_cpus).num_cpus

    def report(completed: int) -> None:
        """Report.

        Args:
            completed: Completed.
        """
        execution.check_cancelled()
        if execution.progress is not None:
            description = "Complete" if completed == total else "Detecting localisations"
            execution.progress(ProgressUpdate(completed, total, description))
            execution.check_cancelled()

    report(0)
    if not total:
        return _empty()
    from picasso.localize import identify_in_frame

    images = _selected_images(view, frames)
    read_lock = Lock()
    claim_lock = Lock()
    stop = Event()
    starts = range(0, total, execution.chunk_size)
    jobs = iter(enumerate(starts))
    chunks = [_empty()] * len(starts)
    completed = [0] * min(workers, len(starts))

    def detect(worker: int) -> None:
        """Detect.

        Args:
            worker: Worker.
        """
        while not stop.is_set():
            with claim_lock:
                job = next(jobs, None)
            if job is None:
                return
            index, start = job
            found = []
            end = min(start + execution.chunk_size, total)
            for frame in range(start, end):
                if stop.is_set():
                    return
                if isinstance(images, np.ndarray):
                    image = images[frame]
                else:
                    with read_lock:
                        image = np.array(images[frame], copy=True)
                y, x, gradient = identify_in_frame(image, settings.min_net_gradient, settings.box_size)
                if len(x):
                    found.append((np.full(len(x), frames[frame], dtype=np.uint64), x, y, gradient))
            if found:
                chunks[index] = {
                    name: np.concatenate(values)
                    for name, values in zip(
                        ("frame", "x", "y", "net_gradient"), zip(*found, strict=True), strict=True
                    )
                }
            completed[worker] += end - start

    with ThreadPoolExecutor(max_workers=len(completed)) as pool:
        pending = {pool.submit(detect, worker) for worker in range(len(completed))}
        try:
            while pending:
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_EXCEPTION)
                for future in done:
                    future.result()
                execution.check_cancelled()
                if pending:
                    report(sum(completed))
        finally:
            stop.set()
    result = _combine(chunks, False)
    report(total)
    return result


def _fit_batches(
    view: ChannelView,
    candidates: dict[str, np.ndarray],
    settings: LocalisationSettings,
    camera: CameraSettings,
    execution: ExecutionContext,
) -> tuple[dict[str, np.ndarray], int]:
    """Fit bounded batches of spot crops in the calling process."""
    import pandas as pd
    from picasso import localize
    from picasso.fitting import gaussfit, seeds

    workers = CpuExecutor(num_cpus=execution.num_cpus).num_cpus
    frames = candidates["frame"]
    total = len(frames)
    last_update = float("-inf")

    def report(completed: float, description: str, *, force: bool = False) -> None:
        """Report.

        Args:
            completed: Completed.
            description: Description.
            force: Force.
        """
        nonlocal last_update
        execution.check_cancelled()
        if execution.progress is not None:
            now = monotonic()
            if force or now - last_update >= 0.05:
                execution.progress(ProgressUpdate(completed, total, description))
                last_update = now
                execution.check_cancelled()

    if not total:
        report(0, "Complete", force=True)
        return (_empty(True), 0)
    original_ids = candidates.get("n_id")
    order = (
        np.argsort(frames, kind="stable")
        if original_ids is None
        else np.lexsort((original_ids.astype(np.uint32), frames))
    )
    bytes_per_spot = settings.box_size**2 * max(view.dtype.itemsize, 4)
    batch_size = max(1, _FIT_SPOT_BYTES // bytes_per_spot)
    camera_info = {"Baseline": camera.baseline, "Sensitivity": camera.sensitivity, "Gain": camera.gain}
    mle = settings.fit_method == "mle"
    chunks, failed_fits = ([], 0)
    for start in range(0, total, batch_size):
        report(start, "Extracting spots", force=True)
        rows = order[start : start + batch_size]
        batch_frames = frames[rows]
        unique_frames, local_frames = np.unique(batch_frames, return_inverse=True)
        images = _selected_images(view, unique_frames)
        found = pd.DataFrame(
            {name: candidates[name][rows] for name in ("x", "y", "net_gradient")}, copy=False
        )
        found["frame"] = local_frames.astype(np.uint32)
        found["n_id"] = np.arange(len(rows), dtype=np.uint32)
        extract_progress = None
        if execution.progress is not None or execution.cancel_event is not None:

            def extract_progress(count: int, offset: int = start) -> None:
                """Extract progress.

                Args:
                    count: Count.
                    offset: Offset.
                """
                report(offset + 0.2 * count, "Extracting spots")

        spots = localize.get_spots(
            images, found, settings.box_size, camera_info, progress_callback=extract_progress
        )
        initial = seeds.initial_parameters_gauss(spots, settings.box_size)
        report(start + 0.2 * len(rows), "Fitting localisations", force=True)
        job = gaussfit.fit_spots_async(
            gaussfit.ELLIPTIC,
            spots,
            initial,
            mle=mle,
            tolerance=settings.tolerance,
            max_iterations=settings.max_iterations,
            n_threads=workers,
        )
        try:
            pending = set(job.futures)
            while pending:
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_EXCEPTION)
                for future in done:
                    future.result()
                report(start + 0.2 * len(rows) + 0.7 * job.current[0], "Fitting localisations")
        except BaseException:
            job.stop()
            wait(job.futures)
            raise
        theta, chi_square, states, iterations = job.results()
        found = localize.locs_from_fits_gauss(
            found,
            theta,
            settings.box_size,
            em=camera.em,
            mle=mle,
            iterations=iterations,
            chi_square=chi_square,
        )
        found["frame"] = batch_frames.astype(np.uint64)
        if original_ids is None:
            del found["n_id"]
        else:
            found["n_id"] = original_ids[rows].astype(np.uint32)
        found["fit_state"] = states
        valid = (states == 0) & np.isfinite(found[["x", "y", "photons", "sx", "sy"]]).all(axis=1).to_numpy()
        valid &= ((found["photons"] >= 0) & (found["sx"] > 0) & (found["sy"] > 0)).to_numpy()
        failed_fits += int((~valid).sum())
        chunks.append({name: values.to_numpy()[valid] for name, values in found.items()})
    result = _combine(chunks, True)
    report(total, "Complete", force=True)
    return (result, failed_fits)


@logged
def _run(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    *,
    settings: LocalisationSettings,
    camera: CameraSettings | None,
    detections: Any,
    fitting: bool,
    frame_range: tuple[int, int] | None,
    num_cpus: int | None,
    cpu: CpuExecutor | None,
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
    context: ExecutionContext | None = None,
    chunk_size: int = 16,
    backend: LocalisationBackend | str | None = None,
    registry: BackendRegistry | None = None,
    source: ImageSource | None = None,
) -> DetectionResult | LocalisationResult:
    """Run.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        settings: Validated settings controlling this operation.
        camera: Camera used by this operation.
        detections: Detections used by this operation.
        fitting: Fitting used by this operation.
        frame_range: Optional half-open camera-frame interval.
        apply: Whether to commit successful results to the model.
        num_cpus: CPU workers for detection or fitting; None selects the default.
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Yields:
        Values produced by the iterator, retaining their original order.
    """
    execution = (context or ExecutionContext(chunk_size=chunk_size)).resolve(
        cpu=cpu, num_cpus=num_cpus, progress=progress, cancel_event=cancel_event
    )
    options = execution.kwargs()
    cpu, num_cpus, progress, cancel_event = (
        options[k] for k in ("cpu", "num_cpus", "progress", "cancel_event")
    )
    selected_backend = resolve_backend("localise", backend, registry=registry)
    installed = backend_version() if selected_backend is None else selected_backend.version
    acquisition = dataset.acquisitions[acquisition_id]
    channel_view = acquisition.channels[channel]
    view = channel_view if source is None else _WorkingChannel(channel_view, source)
    binding = (id(acquisition.source), channel, channel_view.roi, view.source_frames.tobytes())
    token = hash(binding if source is None else (*binding, id(source)))
    if detections is not None and detections.source_token != token:
        raise ValueError("Detection image source/channel changed; detect again")
    if detections is not None and detections.expected is not None:
        from .staging import validate_expected

        validate_expected(dataset, detections.expected)
    expected = dataset.dependencies(
        acquisition_id, channel, components={"localizations", "tracks"} if fitting else ()
    )
    indices = np.arange(len(view))
    if frame_range is not None:
        first, last = frame_range
        if first < 0 or last < first:
            raise ValueError("Expected a nonnegative half-open camera-frame range")
        indices = indices[(view.source_frames >= first) & (view.source_frames < last)]
    candidates = None if detections is None else detections.columns
    if candidates is not None:
        if not {"frame", "x", "y", "net_gradient"} <= candidates.keys():
            raise ValueError("Detection result is missing required columns")
        if not np.isin(candidates["frame"], view.source_frames[indices]).all():
            raise ValueError("Detection frames do not belong to the selected channel/range")
        half = settings.box_size // 2
        if (
            not np.isfinite(candidates["x"]).all()
            or not np.isfinite(candidates["y"]).all()
            or np.any(
                (candidates["x"] < half)
                | (candidates["x"] >= view.shape[2] - half)
                | (candidates["y"] < half)
                | (candidates["y"] >= view.shape[1] - half)
            )
        ):
            raise ValueError("Detection positions must have a complete fit box inside the channel")
    if selected_backend is None and cpu is None:

        def detection_progress(update: ProgressUpdate) -> None:
            """Detection progress.

            Args:
                update: Update.
            """
            if progress is not None:
                progress(ProgressUpdate(update.percent / 2, 100, update.description))

        def fit_progress(update: ProgressUpdate) -> None:
            """Fit progress.

            Args:
                update: Update.
            """
            if progress is not None:
                progress(ProgressUpdate(50 + update.percent / 2, 100, update.description))

        combined = fitting and candidates is None
        if candidates is None:
            detection = (
                replace(execution, progress=detection_progress) if combined and progress else execution
            )
            candidates = _detect_frames(view, view.source_frames[indices], settings, detection)
        result, failed_fits = (candidates, 0)
        if fitting:
            assert camera is not None
            fit_execution = replace(execution, progress=fit_progress) if combined and progress else execution
            result, failed_fits = _fit_batches(view, candidates, settings, camera, fit_execution)
    else:

        def jobs() -> Iterator[Any]:
            """Jobs.

            Returns:
                The computed value.
            """
            for start in range(0, len(indices), execution.chunk_size):
                selected = indices[start : start + execution.chunk_size]
                frames = view.source_frames[selected].copy()
                subset = None
                if candidates is not None:
                    mask = np.isin(candidates["frame"], frames)
                    subset = {name: values[mask] for name, values in candidates.items()}
                yield (view[selected], frames, settings, camera, subset, fitting)

        total = (len(indices) + execution.chunk_size - 1) // execution.chunk_size
        with executor(cpu, num_cpus) as runner:
            chunks = runner.map(
                _job if selected_backend is None else selected_backend.compute,
                jobs(),
                total=total,
                progress=progress,
                cancel_event=cancel_event,
            )
        result = _combine([columns for columns, _ in chunks], fitting)
        failed_fits = sum((count for _, count in chunks))
    check_snapshot(dataset, acquisition_id, expected)
    if not fitting:
        return DetectionResult(acquisition_id, str(channel), result, settings, token, frame_range, expected)
    fitted = LocalisationResult(
        acquisition_id,
        str(channel),
        result,
        settings,
        failed_fits=failed_fits,
        frame_range=frame_range,
        camera=camera,
        backend_version=installed,
        expected=expected,
        source_space="working" if source is not None else "raw",
    )
    return fitted


@logged
@_record_execution
def detect(
    dataset: DatasetSnapshot,
    selection: PartitionSelection,
    *,
    settings: LocalisationSettings | None = None,
    frame_range: tuple[int, int] | None = None,
    context: ExecutionContext | None = None,
    backend: LocalisationBackend | str | None = None,
    registry: BackendRegistry | None = None,
    source: ImageSource | None = None,
) -> DetectionResult:
    """Detect candidates in MolCore channel frames using Picasso net gradient.

    Args:
        selection: Explicit acquisition/channel or sequence/channel selection.
        context: Execution policy, frame chunk size, progress and cancellation.
        dataset: MolCore Dataset supplying acquisition/channel pixels.
        settings: Detection parameters, defaulting to LocalisationSettings().
        frame_range: Optional half-open original camera-frame preview range.
    Returns:
        Detached candidates; active localization data is unchanged.
    """
    if not isinstance(selection, PartitionSelection):
        raise TypeError("Use PartitionSelection")
    acquisition_id, channel = (selection.acquisition_id, selection.channel)
    dataset = processor_model(dataset)
    context = context or ExecutionContext()
    context.check_cancelled()
    cpu, num_cpus, progress, cancel_event = (
        context.cpu,
        context.num_cpus,
        context.progress,
        context.cancel_event,
    )
    return _run(
        dataset,
        acquisition_id,
        channel,
        settings=settings or LocalisationSettings(),
        camera=None,
        detections=None,
        fitting=False,
        frame_range=frame_range,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
        backend=backend,
        registry=registry,
        source=source,
        chunk_size=context.chunk_size,
    )


@logged
@_record_execution
def fit(
    dataset: DatasetSnapshot,
    detections: DetectionResult,
    *,
    camera: CameraSettings,
    settings: LocalisationSettings | None = None,
    context: ExecutionContext | None = None,
    backend: LocalisationBackend | str | None = None,
    registry: BackendRegistry | None = None,
    source: ImageSource | None = None,
) -> LocalisationResult:
    """Fit typed detections and optionally replace their active model partition.

    Args:
        context: Execution policy, frame chunk size, progress and cancellation.
        dataset: MolCore Dataset supplying the detection image source.
        detections: Result returned by detect().
        camera: Explicit scalar camera calibration.
        settings: Fitting parameters; defaults to the detection settings.
    Returns:
        Detached fitted columns and IDs allocated when apply is true.
    """
    dataset = processor_model(dataset)
    context = context or ExecutionContext()
    context.check_cancelled()
    cpu, num_cpus, progress, cancel_event = (
        context.cpu,
        context.num_cpus,
        context.progress,
        context.cancel_event,
    )
    return _run(
        dataset,
        detections.acquisition_id,
        detections.channel,
        settings=settings or detections.settings,
        camera=camera,
        detections=detections,
        fitting=True,
        frame_range=detections.frame_range,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
        backend=backend,
        registry=registry,
        source=source,
        chunk_size=context.chunk_size,
    )


@logged
@_record_execution
def localise(
    dataset: DatasetSnapshot,
    selection: PartitionSelection,
    *,
    camera: CameraSettings,
    settings: LocalisationSettings | None = None,
    frame_range: tuple[int, int] | None = None,
    context: ExecutionContext | None = None,
    backend: LocalisationBackend | str | None = None,
    registry: BackendRegistry | None = None,
    source: ImageSource | None = None,
) -> LocalisationResult:
    """Detect and fit a MolCore channel with bounded CPU batches.

    Args:
        selection: Explicit acquisition/channel or sequence/channel selection.
        context: Execution policy, frame chunk size, progress and cancellation.
        dataset: Dataset whose active results may be updated.
        camera: Explicit camera offset, sensitivity and gain.
        settings: Detection/fitting/chunk parameters.
        frame_range: Original camera-frame preview range; returns a detached preview.
    Returns:
        Typed fitted results with actual source camera frames.

    Raises:
        StaleResultError: Model inputs changed before results could be committed.
        CancelledError: Processing was cancelled; active results remain intact.
    """
    if not isinstance(selection, PartitionSelection):
        raise TypeError("Use PartitionSelection")
    acquisition_id, channel = (selection.acquisition_id, selection.channel)
    dataset = processor_model(dataset)
    context = context or ExecutionContext()
    context.check_cancelled()
    cpu, num_cpus, progress, cancel_event = (
        context.cpu,
        context.num_cpus,
        context.progress,
        context.cancel_event,
    )
    return _run(
        dataset,
        acquisition_id,
        channel,
        settings=settings or LocalisationSettings(),
        camera=camera,
        detections=None,
        fitting=True,
        frame_range=frame_range,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
        backend=backend,
        registry=registry,
        source=source,
        chunk_size=context.chunk_size,
    )
