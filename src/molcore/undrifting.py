"""Picasso RCC/AIM estimation integrated with MolCore's non-destructive drift."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ._processing import backend_version, check_snapshot, executor, record, snapshot
from .types.experiment import Experiment
from .types.processing import DriftResult, DriftSettings


class _AimProgress:
    def __init__(self, reporter, segments):
        self.reporter, self.segments, self.phase = reporter, max(1, segments), 0
        self.description = "AIM drift estimation (1/2)"

    def set_value(self, value, *args, **kwargs):
        self.reporter.report((self.phase + min(1.0, value / self.segments)) / 2, 1, self.description)

    def zero_progress(self, description=None, *args, **kwargs):
        self.phase += 1
        self.description = description or "AIM drift estimation (2/2)"
        self.reporter.check_cancelled()

    def setMaximum(self, value):
        self.segments = max(1, value)

    def setMinimum(self, value):
        pass

    def setLabelText(self, text):
        self.description = text

    def get_iterator(self, start=0, end=100):
        return range(start, end)

    def close(self):
        self.reporter.check_cancelled()


def _job(payload, reporter):
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
        # These are rendering widths, never added to the experiment's table.
        locs["lpx"] = np.ones(len(locs))
        locs["lpy"] = np.ones(len(locs))

        def segmentation(value):
            reporter.report(0.4 * value / segments, 1, "Rendering RCC segments")

        def correlation(value):
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


def _prepare(experiment, acquisition_id, channel, settings):
    expected = snapshot(experiment, acquisition_id)
    acquisition = experiment.acquisitions[acquisition_id]
    view = acquisition.channels[channel]
    rows = experiment.localizations.query(acquisition_id=acquisition_id, channel=channel, space="registered")
    if len(rows["frame"]) == 0:
        raise ValueError("Drift estimation requires localizations in the selected channel")
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
        raise ValueError("Drift estimation requires at least four populated temporal segments")
    groups = (logical_frames - start) // settings.segment_frames
    if len(np.unique(groups)) < 4:
        raise ValueError("Drift estimation requires localizations in at least four segments")
    if settings.method == "rcc":
        bounds = np.linspace(0, n_frames - 1, segments + 1, dtype=np.uint32)
        times = logical_frames - start
        if any(
            not np.any((times >= lo) & (times < hi)) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
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
    return expected, (columns, info, settings), view.source_frames[start : last + 1].copy()


def _many(experiment, selections, *, settings, apply, num_cpus, cpu, progress, cancel_event):
    installed = backend_version()
    prepared = {
        acquisition_id: _prepare(experiment, acquisition_id, channel, settings)
        for acquisition_id, channel in selections.items()
    }
    with executor(cpu, num_cpus) as runner:
        values = runner.map(
            _job, [value[1] for value in prepared.values()], progress=progress, cancel_event=cancel_event
        )
    for acquisition_id, (expected, _, _) in prepared.items():
        check_snapshot(experiment, acquisition_id, expected)
    results = {}
    for (acquisition_id, (_, _, sampled_frames)), displacement in zip(prepared.items(), values, strict=True):
        frames = np.arange(int(sampled_frames[0]), int(sampled_frames[-1]) + 1, dtype=np.int64)
        interpolated = np.column_stack(
            [np.interp(frames, sampled_frames, displacement[:, axis]) for axis in (0, 1)]
        )
        results[acquisition_id] = DriftResult(
            acquisition_id, str(selections[acquisition_id]), frames, interpolated, settings, apply
        )
    if apply:
        for acquisition_id, result in results.items():
            experiment.transforms.set_drift(acquisition_id, result.displacement, frames=result.frames)
            record(
                experiment,
                "undrifting",
                acquisition_id,
                result.channel,
                {
                    "settings": asdict(settings),
                    "backend": installed,
                    "frame_coverage": [int(result.frames[0]), int(result.frames[-1])],
                    "coordinates": "registered reference pixels",
                },
            )
    return results


def _run(experiment, acquisition_id, channel, **kwargs):
    return _many(experiment, {acquisition_id: channel}, **kwargs)[acquisition_id]


def estimate_drift(
    experiment: Experiment,
    acquisition_id: str,
    channel: str,
    *,
    settings: DriftSettings | None = None,
    num_cpus: int | None = None,
    cpu=None,
    progress=None,
    cancel_event=None,
) -> DriftResult:
    """Estimate drift from a MolCore localization partition without applying it.

    Args:
        experiment: Experiment supplying registered localization positions.
        acquisition_id: Acquisition identifier.
        channel: Logical reference channel.
        settings: RCC/AIM settings; defaults to RCC with 1000-frame segments.
        num_cpus: Positive CPU process count or None for the CPU default.
        cpu: Optional reusable CpuExecutor.
        progress: Parent-process ProgressUpdate callback, including estimator stages.
        cancel_event: Optional caller event supporting is_set().

    Returns:
        Reference-pixel displacements at actual camera frames within measured coverage.
    """
    return _run(
        experiment,
        acquisition_id,
        channel,
        settings=settings or DriftSettings(),
        apply=False,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )


def undrift(
    experiment: Experiment,
    acquisition_id: str,
    channel: str,
    *,
    settings: DriftSettings | None = None,
    num_cpus: int | None = None,
    cpu=None,
    progress=None,
    cancel_event=None,
) -> DriftResult:
    """Estimate and select drift without modifying original images or coordinates.

    Args:
        experiment: Experiment whose TransformManager receives the estimated drift.
        acquisition_id: Acquisition identifier.
        channel: Logical reference channel supplying localization measurements.
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
        experiment,
        acquisition_id,
        channel,
        settings=settings or DriftSettings(),
        apply=True,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )


def undrift_many(
    experiment: Experiment,
    selections: dict[str, str],
    *,
    settings: DriftSettings | None = None,
    apply: bool = True,
    num_cpus: int | None = None,
    cpu=None,
    progress=None,
    cancel_event=None,
) -> dict[str, DriftResult]:
    """Estimate independent acquisition drifts concurrently through shared CPU tools.

    Args:
        experiment: Experiment supplying all localization partitions.
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
        experiment,
        selections,
        settings=settings or DriftSettings(),
        apply=apply,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )
