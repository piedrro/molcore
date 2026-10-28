"""Original-pixel aperture photometry and multichannel intensity traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike

from ._logging import logged
from ._processing import executor
from .cpu import CpuExecutor
from .staging import state_token, transact
from .transforms import project_points
from .types import (
    ApertureShape,
    BackgroundMethod,
    CancelEvent,
    Dataset,
    PhotometryResult,
    PhotometrySettings,
    PhotonSequenceResult,
    ProgressCallback,
    TraceMode,
)
from .types.tables import identifiers

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .cpu import WorkerProgress

_DEFAULT_SETTINGS = PhotometrySettings()

_METRICS = ("sum", "mean", "median", "min", "max", "std")


def _region(
    shape: tuple[int, int], x: float, y: float, radius: float, aperture: ApertureShape
) -> tuple[tuple[slice, slice], np.ndarray]:
    """Region.

    Args:
        shape: Array dimensions in time, y, x order.
        x: X used by this operation.
        y: Y used by this operation.
        radius: Radius used by this operation.
        aperture: Aperture used by this operation.

    Returns:
        Computed tuple[tuple[slice, slice], NumPy array] value.
    """
    cx, cy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    half = int(np.ceil(radius))
    x0, x1 = np.clip([cx - half, cx + half + 1], 0, shape[1])
    y0, y1 = np.clip([cy - half, cy + half + 1], 0, shape[0])
    yy, xx = np.ogrid[int(y0) : int(y1), int(x0) : int(x1)]
    distance = (
        np.maximum(abs(xx - cx), abs(yy - cy))
        if aperture == ApertureShape.SQUARE
        else np.hypot(xx - cx, yy - cy)
    )
    return (slice(int(y0), int(y1)), slice(int(x0), int(x1))), distance <= radius


def _summary(values: np.ndarray) -> dict[str, float]:
    """Summary.

    Args:
        values: Input values to validate, convert or display.

    Returns:
        Computed dict[str, float] value.
    """
    if not values.size:
        return dict.fromkeys(_METRICS, float("nan"))
    return dict(
        zip(
            _METRICS,
            (
                float(np.sum(values, dtype=np.float64)),
                float(np.mean(values)),
                float(np.median(values)),
                float(np.min(values)),
                float(np.max(values)),
                float(np.std(values)),
            ),
            strict=True,
        )
    )


@logged
def _job(payload: tuple, reporter: WorkerProgress) -> list[dict]:
    """Job.

    Args:
        payload: Serializable worker inputs; large arrays remain bounded to a job or image chunk.
        reporter: Worker progress and cooperative cancellation reporter.

    Returns:
        Computed list[dict] value.
    """
    image, requests, neighbours, settings = payload
    image = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(image)
    radius = settings.spot_size // 2 if settings.shape == ApertureShape.SQUARE else settings.spot_size / 2
    exclusion = np.zeros(image.shape, dtype=bool)
    if settings.exclude_neighbours:
        for x, y in neighbours:
            region, mask = _region(image.shape, x, y, radius, settings.shape)
            exclusion[region] |= mask
    global_values = image[finite & ~exclusion]
    global_summary = _summary(global_values)
    output = []
    for index, request in enumerate(requests):
        reporter.check_cancelled()
        x, y = request["x"], request["y"]
        region, signal_mask = _region(image.shape, x, y, radius, settings.shape)
        signal = image[region][signal_mask & finite[region]]
        outer = radius + settings.buffer + settings.background_width
        region, ring = _region(image.shape, x, y, outer, settings.shape)
        cx, cy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        yy, xx = np.ogrid[region[0].start : region[0].stop, region[1].start : region[1].stop]
        distance = (
            np.maximum(abs(xx - cx), abs(yy - cy))
            if settings.shape == ApertureShape.SQUARE
            else np.hypot(xx - cx, yy - cy)
        )
        ring &= distance > radius + settings.buffer
        background = image[region][ring & finite[region] & ~exclusion[region]]
        summaries = {
            "signal": _summary(signal),
            "local_background": _summary(background),
            "global_background": global_summary,
        }
        row = dict(request)
        for prefix, values in summaries.items():
            row.update({f"{prefix}_{name}": value for name, value in values.items()})
        row.update(
            signal_count=signal.size,
            local_background_count=background.size,
            global_background_count=global_values.size,
        )
        chosen = summaries[
            "local_background" if settings.background == BackgroundMethod.LOCAL else "global_background"
        ]
        row["intensity"] = summaries["signal"]["sum"] - chosen["mean"] * signal.size
        output.append(row)
        reporter.report(index + 1, len(requests), "Aperture photometry")
    return output


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    """Columns.

    Args:
        rows: Named measurement arrays retaining stable IDs and source-frame provenance.

    Returns:
        Detached named result columns.
    """
    dtypes = {
        "molecule_id": str,
        "acquisition_id": str,
        "channel": str,
        "frame": np.int64,
        "channel_frame": np.int64,
        "cycle_index": np.int64,
        "timestamp": float,
        "localization_id": np.uint64,
        "track_id": np.uint64,
        "x": float,
        "y": float,
        "intensity": float,
        "signal_count": np.int64,
        "local_background_count": np.int64,
        "global_background_count": np.int64,
    }
    for prefix in ("signal", "local_background", "global_background"):
        dtypes.update({f"{prefix}_{name}": float for name in _METRICS})
    return {name: np.asarray([row[name] for row in rows], dtype=dtype) for name, dtype in dtypes.items()}


def _extract(
    dataset: Dataset,
    selected: ArrayLike,
    *,
    mode: TraceMode,
    channels: list[str] | None,
    settings: PhotometrySettings,
    frame_range: tuple[int, int] | None,
    apply: bool,
    cpu: CpuExecutor | None,
    num_cpus: int | None,
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
) -> PhotometryResult | PhotonSequenceResult:
    """Extract.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        selected: Selected used by this operation.
        mode: Requested processing or acquisition mode.
        channels: Channels used by this operation.
        settings: Validated settings controlling this operation.
        frame_range: Optional half-open camera-frame interval.
        apply: Whether to commit successful results to the model.
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        num_cpus: Worker process count; None uses the executor default.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed PhotometryResult value.
    """
    from concurrent.futures import CancelledError

    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Photometry cancelled")
    if frame_range is not None and (
        len(frame_range) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in frame_range)
        or frame_range[0] < 0
        or frame_range[1] < frame_range[0]
    ):
        raise ValueError("Expected a nonnegative half-open camera-frame range")
    selected = identifiers(selected)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Select distinct IDs")
    if channels is not None and (not channels or len(set(channels)) != len(channels)):
        raise ValueError("Select distinct target channels")
    if not isinstance(settings, PhotometrySettings):
        raise TypeError("settings must be PhotometrySettings")
    expected = state_token(dataset)
    if mode == TraceMode.TRACK:
        rows = dataset.tracks.query(track_id=selected, space="raw")
    else:
        rows = dataset.localizations.query(ids=selected, space="raw")
        if len(rows["id"]) != len(selected):
            raise ValueError("Unknown localization ID")
    requests = defaultdict(list)
    transforms = dataset.transforms
    for index in range(len(rows["id"])):
        a, source, anchor_frame = (
            str(rows["acquisition_id"][index]),
            str(rows["channel"][index]),
            int(rows["frame"][index]),
        )
        point = np.array([rows["x"][index], rows["y"][index]])
        reference = project_points(transforms._registration(a, source), point)
        if mode == TraceMode.STATIONARY and settings.follow_drift:
            reference[:2] -= transforms._displacement(a, np.array([anchor_frame]))[0]
        for channel in channels if channels is not None else [source]:
            view = dataset.acquisitions[a].channels[channel]
            frames = view.source_frames if mode == TraceMode.STATIONARY else np.array([anchor_frame])
            frames = frames[np.isin(frames, view.source_frames)]
            if frame_range is not None:
                frames = frames[(frames >= frame_range[0]) & (frames < frame_range[1])]
            inverse = np.linalg.inv(transforms._registration(a, channel))
            for frame in frames:
                mapped = reference.copy()
                if mode == TraceMode.STATIONARY and settings.follow_drift:
                    mapped[:2] += transforms._displacement(a, np.array([frame]))[0]
                target = project_points(inverse, mapped)
                if not np.isfinite(target).all():
                    raise ValueError("Photometry position lies on a projective pole")
                info = view.frame_info(view.channel_index(int(frame)))
                tid = int(rows["track_id"][index]) if mode == TraceMode.TRACK else 0
                lid = int(rows["id"][index])
                requests[a, channel, int(frame)].append(
                    {
                        "molecule_id": f"track:{tid}" if tid else f"localization:{lid}",
                        "acquisition_id": a,
                        "channel": channel,
                        "frame": int(frame),
                        "channel_frame": view.channel_index(int(frame)),
                        "cycle_index": info.cycle_index if info.cycle_index is not None else -1,
                        "timestamp": info.timestamp if info.timestamp is not None else np.nan,
                        "localization_id": lid,
                        "track_id": tid,
                        "x": float(target[0]),
                        "y": float(target[1]),
                    }
                )

    def jobs() -> Iterator[Any]:
        """Jobs.

        Yields:
            Values produced by the iterator, retaining their original order.
        """
        for (a, channel, frame), group in requests.items():
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("Photometry cancelled")
            view = dataset.acquisitions[a].channels[channel]
            neighbours = [(row["x"], row["y"]) for row in group]
            if settings.exclude_neighbours:
                other = dataset.localizations.query(
                    acquisition_id=a, frame_range=(frame, frame + 1), space="raw"
                )
                if len(other["id"]):
                    x, y = transforms.into_channel(other, channel)
                    neighbours.extend(zip(x, y, strict=True))
            yield view[view.channel_index(frame)], group, neighbours, settings

    with executor(cpu, num_cpus) as runner:
        chunks = runner.map(_job, jobs(), total=len(requests), progress=progress, cancel_event=cancel_event)
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Photometry cancelled")
    columns = _columns([row for chunk in chunks for row in chunk])
    run_id = str(uuid4())
    if mode == TraceMode.LOCALIZATIONS:
        result = PhotometryResult(columns, settings, mode, run_id, expected)
    else:
        from .photon_sequences import from_columns

        sequences = from_columns(
            columns,
            run_id=run_id,
            metadata={"settings": asdict(settings), "mode": str(mode)},
            depends_on_localizations=settings.exclude_neighbours,
        )
        result = PhotonSequenceResult(sequences, settings, mode, run_id, expected)
    if state_token(dataset) != expected:
        from .types import StaleResultError

        raise StaleResultError("Photometry inputs changed during computation")
    return apply_photometry(dataset, result) if apply else result


@logged
def apply_photometry(
    dataset: Dataset, result: PhotometryResult | PhotonSequenceResult, *, expected: tuple | None = None
) -> PhotometryResult | PhotonSequenceResult:
    """Apply localisation measurements or primary photon sequences atomically.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        result: Detached computation result to inspect or apply.
        expected: Source revision token used to reject stale results.

    Returns:
        Localisation result with observation IDs, or trace result with photon-sequence IDs.
    """

    if not isinstance(result, PhotonSequenceResult) and result.mode in (
        TraceMode.STATIONARY,
        TraceMode.TRACK,
    ):
        from .photon_sequences import from_columns

        result = PhotonSequenceResult(
            from_columns(
                result.columns,
                run_id=result.run_id,
                metadata={"settings": asdict(result.settings), "mode": str(result.mode)},
                depends_on_localizations=result.settings.exclude_neighbours,
            ),
            result.settings,
            result.mode,
            result.run_id,
            result.expected,
        )
    if isinstance(result, PhotonSequenceResult):
        from .photon_sequences import apply_photon_sequences

        return apply_photon_sequences(dataset, result, expected=expected)

    def commit(model: Dataset) -> PhotometryResult:
        """Commit.

        Args:
            model: Model or fitting method selected by the caller.

        Returns:
            Computed PhotometryResult value.
        """
        runs = model.processing.settings.setdefault("photometry", {})
        if result.run_id in runs:
            raise ValueError("Photometry run already applied")
        ids = model.observations.add(**result.columns, run_id=result.run_id)
        runs[result.run_id] = {
            "settings": asdict(result.settings),
            "mode": result.mode,
            "acquisitions": sorted(set(result.columns["acquisition_id"].tolist())),
            "channels": sorted(set(result.columns["channel"].tolist())),
            "units": "original image intensity units",
            "version": 1,
        }
        return replace(result, observation_ids=ids)

    return transact(dataset, commit, expected=result.expected if expected is None else expected)


@logged
def measure_localizations(
    dataset: Dataset,
    localization_ids: ArrayLike,
    *,
    channels: list[str] | None = None,
    settings: PhotometrySettings = _DEFAULT_SETTINGS,
    frame_range: tuple[int, int] | None = None,
    apply: bool = True,
    cpu: CpuExecutor | None = None,
    num_cpus: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> PhotometryResult:
    """Measure apertures at effective positions in measured camera frames.

    Channels default to each localization's source channel. Targets that do not
    measure the same camera frame are omitted. Raw source pixels are required.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        localization_ids: Localization ids used by this operation.
        channels: Channels used by this operation.
        settings: Validated settings controlling this operation.
        frame_range: Optional half-open camera-frame interval.
        apply: Whether to commit successful results to the model.
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        num_cpus: Worker process count; None uses the executor default.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed PhotometryResult value.
    """
    return cast(
        PhotometryResult,
        _extract(
            dataset,
            localization_ids,
            mode=TraceMode.LOCALIZATIONS,
            channels=channels,
            settings=settings,
            frame_range=frame_range,
            apply=apply,
            cpu=cpu,
            num_cpus=num_cpus,
            progress=progress,
            cancel_event=cancel_event,
        ),
    )


@logged
def stationary_traces(
    dataset: Dataset,
    localization_ids: ArrayLike,
    *,
    channels: list[str] | None = None,
    settings: PhotometrySettings = _DEFAULT_SETTINGS,
    frame_range: tuple[int, int] | None = None,
    apply: bool = True,
    cpu: CpuExecutor | None = None,
    num_cpus: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> PhotonSequenceResult:
    """Sample each channel's actual frames at drift-following stationary positions.

    Each seed defines one molecule fixed in corrected reference coordinates.
    follow_drift=False fixes the registered reference position instead. Missing
    drift coverage raises; no camera time or excitation cycle is invented.
    Successful applying runs write photon sequences directly, without creating
    per-frame localisations, spatial tracks or observation records.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        localization_ids: Localization ids used by this operation.
        channels: Channels used by this operation.
        settings: Validated settings controlling this operation.
        frame_range: Optional half-open camera-frame interval.
        apply: Whether to commit successful results to the model.
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        num_cpus: Worker process count; None uses the executor default.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Primary photon sequences with sample metrics and any applied sequence IDs.
    """
    return cast(
        PhotonSequenceResult,
        _extract(
            dataset,
            localization_ids,
            mode=TraceMode.STATIONARY,
            channels=channels,
            settings=settings,
            frame_range=frame_range,
            apply=apply,
            cpu=cpu,
            num_cpus=num_cpus,
            progress=progress,
            cancel_event=cancel_event,
        ),
    )


@logged
def track_traces(
    dataset: Dataset,
    track_ids: ArrayLike,
    *,
    channels: list[str] | None = None,
    settings: PhotometrySettings = _DEFAULT_SETTINGS,
    frame_range: tuple[int, int] | None = None,
    apply: bool = True,
    cpu: CpuExecutor | None = None,
    num_cpus: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> PhotonSequenceResult:
    """Measure moving tracks at measured members only, preserving all gaps.

    Target channels must measure each member's original camera frame. Cross-
    acquisition tracks retain acquisition-local frame and timestamp provenance.
    Applying stores independent photon-sequence records while leaving the track's
    existing localisation memberships unchanged.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        track_ids: Track ids used by this operation.
        channels: Channels used by this operation.
        settings: Validated settings controlling this operation.
        frame_range: Optional half-open camera-frame interval.
        apply: Whether to commit successful results to the model.
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        num_cpus: Worker process count; None uses the executor default.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Primary photon sequences linked to unchanged spatial tracks through provenance.
    """
    return cast(
        PhotonSequenceResult,
        _extract(
            dataset,
            track_ids,
            mode=TraceMode.TRACK,
            channels=channels,
            settings=settings,
            frame_range=frame_range,
            apply=apply,
            cpu=cpu,
            num_cpus=num_cpus,
            progress=progress,
            cancel_event=cancel_event,
        ),
    )
