"""Picasso RCC/AIM estimation integrated with MolCore's non-destructive drift."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from ._processing import backend_version, check_snapshot, executor, snapshot
from .types.dataset import Dataset
from .types.processing import DriftResult, DriftSettings
from .types.tables import identifiers

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike

    from .cpu import CpuExecutor, WorkerProgress
    from .types import CancelEvent, ProgressCallback


class _AimProgress:
    def __init__(self, reporter: WorkerProgress, segments: Any) -> None:
        """Initialise the object and validate its configuration.

        Args:
            reporter: Worker progress and cooperative cancellation reporter.
            segments: Segments used by this operation.
        """
        self.reporter, self.segments, self.phase = reporter, max(1, segments), 0
        self.description = "AIM drift estimation (1/2)"

    @logged
    def set_value(self, value: Any, *args: Any, **kwargs: Any) -> None:
        """Set value.

        Args:
            value: Value supplied by the caller or connected control.
            *args: Additional positional values supplied by the caller.
            **kwargs: Additional named options for this operation.
        """
        self.reporter.report(
            (self.phase + min(1.0, value / self.segments)) / 2, 1, self.description
        )

    @logged
    def zero_progress(
        self, description: str | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Zero progress.

        Args:
            description: Human-readable description of the current processing stage.
            *args: Additional positional values supplied by the caller.
            **kwargs: Additional named options for this operation.
        """
        self.phase += 1
        self.description = description or "AIM drift estimation (2/2)"
        self.reporter.check_cancelled()

    @logged
    def setMaximum(self, value: Any) -> None:
        """Setmaximum.

        Args:
            value: Value supplied by the caller or connected control.
        """
        self.segments = max(1, value)

    @logged
    def setMinimum(self, value: Any) -> None:
        """Setminimum.

        Args:
            value: Value supplied by the caller or connected control.
        """
        pass

    @logged
    def setLabelText(self, text: str) -> None:
        """Setlabeltext.

        Args:
            text: Text supplied by the control or caller.
        """
        self.description = text

    @logged
    def get_iterator(self, start: int = 0, end: int = 100) -> range:
        """Get iterator.

        Args:
            start: Start used by this operation.
            end: End used by this operation.

        Returns:
            Result retaining the input format or backend-specific representation.
        """
        return range(start, end)

    @logged
    def close(self) -> None:
        """Close."""
        self.reporter.check_cancelled()


@logged
def _job(payload: tuple, reporter: WorkerProgress) -> Any:
    """Job.

    Args:
        payload: Serializable worker inputs; large arrays remain bounded to a job or image chunk.
        reporter: Worker progress and cooperative cancellation reporter.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    columns, info, settings = payload
    import pandas as pd

    locs = pd.DataFrame(columns)
    n_frames = info[0]["Frames"]
    segments = int(
        np.round(n_frames / settings.segment_frames)
        if settings.method == "rcc"
        else np.ceil(n_frames / settings.segment_frames)
    )
    reporter.report(0, 1, f"Estimating {settings.method.upper()} drift")
    if settings.method == "rcc":
        from picasso.postprocess import undrift

        # Picasso's RCC renderer requires precision columns even though it
        # clamps its Gaussian rendering kernel to one reference pixel.
        # These are rendering widths, never added to the dataset's table.
        locs["lpx"] = np.ones(len(locs))
        locs["lpy"] = np.ones(len(locs))

        def segmentation(value: Any) -> None:
            """Segmentation.

            Args:
                value: Value supplied by the caller or connected control.
            """
            reporter.report(0.4 * value / segments, 1, "Rendering RCC segments")

        def correlation(value: Any) -> None:
            """Correlation.

            Args:
                value: Value supplied by the caller or connected control.
            """
            pairs = max(1, segments * (segments - 1) / 2)
            reporter.report(0.4 + 0.6 * value / pairs, 1, "Correlating RCC segments")

        drift, _ = undrift(
            locs,
            info,
            segmentation=settings.segment_frames,
            display=False,
            segmentation_callback=segmentation,
            rcc_callback=correlation,
        )
    else:
        from picasso.aim import aim

        _, _, drift = aim(
            locs,
            info,
            segmentation=settings.segment_frames,
            intersect_d=settings.intersect_distance,
            roi_r=settings.search_radius,
            progress=_AimProgress(reporter, segments),
        )
    values = drift[["x", "y"]].to_numpy(copy=True)
    if values.shape != (n_frames, 2) or not np.isfinite(values).all():
        raise ValueError("Picasso returned invalid or incomplete drift estimates")
    reporter.report(1, 1, "Drift estimation complete")
    return values


def _prepare(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    settings: DriftSettings,
    ids: ArrayLike | None = None,
) -> tuple:
    """Prepare.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        settings: Validated settings controlling this operation.

    Returns:
        Computed tuple value.
    """
    expected = snapshot(dataset, acquisition_id)
    acquisition = dataset.acquisitions[acquisition_id]
    view = acquisition.channels[channel]
    rows = dataset.localizations.query(
        acquisition_id=acquisition_id, channel=channel, space="registered"
    ).to_columns()
    if len(rows["frame"]) == 0:
        raise ValueError(
            "Drift estimation requires localizations in the selected channel"
        )
    if ids is not None:
        selected = identifiers(ids)
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Selected localization IDs must be distinct")
        if not np.isin(selected, rows["id"]).all():
            raise ValueError("Selected localization IDs must belong to the partition")
        keep = np.isin(rows["id"], selected)
        rows = {name: values[keep] for name, values in rows.items()}
        if not len(rows["id"]):
            raise ValueError("Drift estimation requires selected localizations")
    logical_frames = np.searchsorted(view.source_frames, rows["frame"])
    start, last = int(logical_frames.min()), int(logical_frames.max())
    n_frames = last - start + 1
    # Both algorithms use cubic temporal interpolation internally.
    segments = int(
        np.round(n_frames / settings.segment_frames)
        if settings.method == "rcc"
        else np.ceil(n_frames / settings.segment_frames)
    )
    if segments < 4:
        raise ValueError(
            "Drift estimation requires at least four populated temporal segments"
        )
    groups = (logical_frames - start) // settings.segment_frames
    if len(np.unique(groups)) < 4:
        raise ValueError(
            "Drift estimation requires localizations in at least four segments"
        )
    if settings.method == "rcc":
        bounds = np.linspace(0, n_frames - 1, segments + 1, dtype=np.uint32)
        times = logical_frames - start
        if any(
            not np.any((times >= lo) & (times < hi))
            for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
        ):
            raise ValueError("Each RCC rendering segment must contain localizations")
    x, y = rows["x"].copy(), rows["y"].copy()
    x -= min(0.0, float(x.min()))
    y -= min(0.0, float(y.min()))
    columns = {"frame": (logical_frames - start).astype(np.uint32), "x": x, "y": y}
    info = [
        {
            "Frames": n_frames,
            "Width": max(view.shape[2], int(np.ceil(x.max() + 2))),
            "Height": max(view.shape[1], int(np.ceil(y.max() + 2))),
            "Pixelsize": acquisition.pixel_size or 1.0,
        }
    ]
    return (
        expected,
        (columns, info, settings),
        view.source_frames[start : last + 1].copy(),
    )


def _many(
    dataset: Dataset,
    selections: Mapping[str, str],
    *,
    settings: DriftSettings,
    apply: bool,
    num_cpus: int | None,
    cpu: CpuExecutor | None,
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
    ids: Mapping[str, ArrayLike] | None = None,
) -> Any:
    """Many.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        selections: Mapping from acquisition identifiers to logical channel names.
        settings: Validated settings controlling this operation.
        apply: Whether to commit successful results to the model.
        num_cpus: Worker process count; None uses the executor default.
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    installed = backend_version()
    prepared = {
        acquisition_id: _prepare(
            dataset,
            acquisition_id,
            channel,
            settings,
            None if ids is None else ids.get(acquisition_id),
        )
        for acquisition_id, channel in selections.items()
    }
    with executor(cpu, num_cpus) as runner:
        values = runner.map(
            _job,
            [value[1] for value in prepared.values()],
            progress=progress,
            cancel_event=cancel_event,
        )
    for acquisition_id, (expected, _, _) in prepared.items():
        check_snapshot(dataset, acquisition_id, expected)
    results = {}
    for (acquisition_id, (_, _, sampled_frames)), displacement in zip(
        prepared.items(), values, strict=True
    ):
        frames = np.arange(
            int(sampled_frames[0]), int(sampled_frames[-1]) + 1, dtype=np.int64
        )
        interpolated = np.column_stack(
            [
                np.interp(frames, sampled_frames, displacement[:, axis])
                for axis in (0, 1)
            ]
        )
        results[acquisition_id] = DriftResult(
            acquisition_id,
            str(selections[acquisition_id]),
            frames,
            interpolated,
            settings,
            expected=prepared[acquisition_id][0],
            backend_version=installed,
        )
    if apply:
        from .processing import apply_drift

        return apply_drift(dataset, results)
    return results


@logged
def _run(dataset: Dataset, acquisition_id: str, channel: str, **kwargs: Any) -> Any:
    """Run.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        **kwargs: Additional named options for this operation.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    selected = kwargs.pop("ids", None)
    return _many(
        dataset,
        {acquisition_id: channel},
        ids=None if selected is None else {acquisition_id: selected},
        **kwargs,
    )[acquisition_id]


@logged
def estimate_drift(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    *,
    ids: ArrayLike | None = None,
    settings: DriftSettings | None = None,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> DriftResult:
    """Estimate drift from a MolCore localization partition without applying it.

    Args:
        dataset: Dataset supplying registered localization positions.
        acquisition_id: Acquisition identifier.
        channel: Logical reference channel.
        ids: Stable localization IDs within the partition; None selects all.
            Duplicate, unknown, out-of-partition and empty selections are rejected.
        settings: RCC/AIM settings; defaults to RCC with 1000-frame segments.
        num_cpus: Positive CPU process count or None for the CPU default.
        cpu: Optional reusable CpuExecutor.
        progress: Parent-process ProgressUpdate callback, including estimator stages.
        cancel_event: Optional caller event supporting is_set().

    Returns:
        Reference-pixel displacements at actual camera frames within measured coverage.
    """
    return _run(
        dataset,
        acquisition_id,
        channel,
        settings=settings or DriftSettings(),
        ids=ids,
        apply=False,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )


@logged
def undrift(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    *,
    ids: ArrayLike | None = None,
    settings: DriftSettings | None = None,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> DriftResult:
    """Estimate and select drift without modifying original images or coordinates.

    Args:
        dataset: Dataset whose TransformManager receives the estimated drift.
        acquisition_id: Acquisition identifier.
        channel: Logical reference channel supplying localization measurements.
        ids: Stable IDs used for estimation only; None selects all localizations.
            Drift application still covers the acquisition, including other origins.
        settings: RCC/AIM settings in logical-channel frames and reference pixels.
        num_cpus: Positive CPU process count; default leaves one CPU available.
        cpu: Optional reusable CpuExecutor.
        progress: Parent-process ProgressUpdate callback.
        cancel_event: Optional caller cancellation event.

    Returns:
        DriftResult marked applied after a successful parent-process update.

    Raises:
        StaleResultError: Localization, registration or image inputs changed.
        ValueError: Too few populated segments or invalid estimated displacement.
    """
    return _run(
        dataset,
        acquisition_id,
        channel,
        settings=settings or DriftSettings(),
        ids=ids,
        apply=True,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )


@logged
def undrift_many(
    dataset: Dataset,
    selections: dict[str, str],
    *,
    settings: DriftSettings | None = None,
    apply: bool = True,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> dict[str, DriftResult]:
    """Estimate independent acquisition drifts concurrently through shared CPU tools.

    Args:
        dataset: Dataset supplying all localization partitions.
        selections: Mapping of acquisition IDs to their reference channel names.
        settings: Shared RCC/AIM estimator parameters.
        apply: Apply results only after every job and stale-input check succeeds.
        num_cpus: Number of independent acquisition jobs that can execute together.
        cpu: Optional reusable CpuExecutor.
        progress: Parent-process aggregated ProgressUpdate callback.
        cancel_event: Optional caller cancellation event.

    Returns:
        Drift results keyed by acquisition ID. A single estimator's temporal
        dependency remains sequential within its worker.
    """
    return _many(
        dataset,
        selections,
        settings=settings or DriftSettings(),
        apply=apply,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )
