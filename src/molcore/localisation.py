"""Model-based Picasso detection and CPU Gaussian localisation."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from ._processing import backend_version, check_snapshot, executor, record, snapshot
from .types.experiment import Experiment
from .types.processing import CameraSettings, DetectionResult, LocalisationResult, LocalisationSettings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .cpu import CpuExecutor, WorkerProgress
    from .types import CancelEvent, ProgressCallback


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
def _job(payload: tuple, reporter: WorkerProgress) -> tuple:
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
    chunks, failed_fits = [], 0
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
    return _combine(chunks, fitting), failed_fits


def _combine(chunks: tuple[int, ...] | str, fitting: bool) -> dict[str, np.ndarray]:
    """Combine.

    Args:
        chunks: Dask chunk dimensions; None uses one camera frame per chunk.
        fitting: Fitting used by this operation.

    Returns:
        Detached named result columns.
    """
    chunks = [chunk for chunk in chunks if len(chunk["frame"])]
    if not chunks:
        return _empty(fitting)
    return {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}


@logged
def _run(
    experiment: Experiment,
    acquisition_id: str,
    channel: str,
    *,
    settings: LocalisationSettings,
    camera: CameraSettings | None,
    detections: Any,
    fitting: bool,
    frame_range: tuple[int, int] | None,
    apply: bool,
    num_cpus: int | None,
    cpu: CpuExecutor | None,
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
) -> Iterator[Any]:
    """Run.

    Args:
        experiment: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        settings: Validated settings controlling this operation.
        camera: Camera used by this operation.
        detections: Detections used by this operation.
        fitting: Fitting used by this operation.
        frame_range: Optional half-open camera-frame interval.
        apply: Whether to commit successful results to the model.
        num_cpus: Worker process count; None uses the executor default.
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Yields:
        Values produced by the iterator, retaining their original order.
    """
    installed = backend_version()
    if frame_range is not None and apply:
        raise ValueError("Subset processing is a preview; pass apply=False")
    acquisition = experiment.acquisitions[acquisition_id]
    view = acquisition.channels[channel]
    token = hash((id(acquisition.source), id(view)))
    if detections is not None and detections.source_token != token:
        raise ValueError("Detection image source/channel changed; detect again")
    expected = snapshot(experiment, acquisition_id)
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

    def jobs() -> Iterator[Any]:
        """Jobs.

        Yields:
            Values produced by the iterator, retaining their original order.
        """
        for start in range(0, len(indices), settings.chunk_size):
            selected = indices[start : start + settings.chunk_size]
            frames = view.source_frames[selected].copy()
            subset = (
                None
                if candidates is None
                else {k: v[np.isin(candidates["frame"], frames)].copy() for k, v in candidates.items()}
            )
            yield view[selected], frames, settings, camera, subset, fitting

    total = (len(indices) + settings.chunk_size - 1) // settings.chunk_size
    with executor(cpu, num_cpus) as runner:
        chunks = runner.map(_job, jobs(), total=total, progress=progress, cancel_event=cancel_event)
    check_snapshot(experiment, acquisition_id, expected)
    result = _combine([columns for columns, _ in chunks], fitting)
    failed_fits = sum(count for _, count in chunks)
    if not fitting:
        return DetectionResult(acquisition_id, str(channel), result, settings, token, frame_range)
    ids = None
    if apply and failed_fits and len(result["frame"]) == 0:
        raise ValueError("No localisation fits converged; active results were preserved")
    if apply:
        ids = experiment.localizations.replace(acquisition_id, channel, **result)
        record(
            experiment,
            "localisation",
            acquisition_id,
            str(channel),
            {
                "settings": asdict(settings),
                "camera": asdict(camera),
                "backend": installed,
                "camera_frames": "all logical-channel frames",
                "failed_fits": failed_fits,
            },
        )
    return LocalisationResult(acquisition_id, str(channel), result, settings, ids, failed_fits)


@logged
def detect(
    experiment: Experiment,
    acquisition_id: str,
    channel: str,
    *,
    settings: LocalisationSettings | None = None,
    frame_range: tuple[int, int] | None = None,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> DetectionResult:
    """Detect candidates in MolCore channel frames using Picasso net gradient.

    Args:
        experiment: MolCore Experiment supplying acquisition/channel pixels.
        acquisition_id: Acquisition identifier.
        channel: Logical channel name.
        settings: Detection parameters, defaulting to LocalisationSettings().
        frame_range: Optional half-open original camera-frame preview range.
        num_cpus: Positive process count; None selects the CPU default.
        cpu: Optional reusable CpuExecutor; configure CPU count on that executor.
        progress: Parent-process callback receiving ProgressUpdate values.
        cancel_event: Optional caller event supporting is_set().

    Returns:
        Detached candidates; active localization data is unchanged.
    """
    return _run(
        experiment,
        acquisition_id,
        channel,
        settings=settings or LocalisationSettings(),
        camera=None,
        detections=None,
        fitting=False,
        frame_range=frame_range,
        apply=False,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )


@logged
def fit(
    experiment: Experiment,
    detections: DetectionResult,
    *,
    camera: CameraSettings,
    settings: LocalisationSettings | None = None,
    apply: bool = True,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> LocalisationResult:
    """Fit typed detections and optionally replace their active model partition.

    Args:
        experiment: MolCore Experiment supplying the detection image source.
        detections: Result returned by detect().
        camera: Explicit scalar camera calibration.
        settings: Fitting parameters; defaults to the detection settings.
        apply: Replace the full acquisition/channel partition after success.
        num_cpus: Positive process count or None for the CPU default.
        cpu: Optional reusable CpuExecutor.
        progress: Parent-process ProgressUpdate callback.
        cancel_event: Optional caller cancellation event.

    Returns:
        Detached fitted columns and IDs allocated when apply is true.
    """
    return _run(
        experiment,
        detections.acquisition_id,
        detections.channel,
        settings=settings or detections.settings,
        camera=camera,
        detections=detections,
        fitting=True,
        frame_range=detections.frame_range,
        apply=apply,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )


@logged
def localise(
    experiment: Experiment,
    acquisition_id: str,
    channel: str,
    *,
    camera: CameraSettings,
    settings: LocalisationSettings | None = None,
    frame_range: tuple[int, int] | None = None,
    apply: bool = True,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> LocalisationResult:
    """Detect and fit a MolCore channel through bounded CPU image chunks.

    Args:
        experiment: Experiment whose active results may be updated.
        acquisition_id: Acquisition identifier.
        channel: Logical channel name.
        camera: Explicit camera offset, sensitivity and gain.
        settings: Detection/fitting/chunk parameters.
        frame_range: Original camera-frame preview range; requires apply=False.
        apply: Commit a full successful result in the calling process.
        num_cpus: Positive process count or None for the CPU default.
        cpu: Optional reusable CpuExecutor.
        progress: Parent-process ProgressUpdate callback.
        cancel_event: Optional caller cancellation event.

    Returns:
        Typed fitted results with actual source camera frames.

    Raises:
        StaleResultError: Model inputs changed before results could be committed.
        CancelledError: Processing was cancelled; active results remain intact.
    """
    return _run(
        experiment,
        acquisition_id,
        channel,
        settings=settings or LocalisationSettings(),
        camera=camera,
        detections=None,
        fitting=True,
        frame_range=frame_range,
        apply=apply,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )
