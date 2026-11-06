"""Original-pixel aperture photometry and multichannel intensity traces."""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import cache
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike

from ._logging import logged
from ._processing import executor
from .cpu import CpuExecutor
from .revisions import FreshnessToken
from .staging import dependency_token, transact
from .transforms import project_points
from .types import (
    ApertureShape,
    ArraySource,
    BackgroundMethod,
    CancelEvent,
    Dataset,
    FrameInfo,
    PhotometryResult,
    PhotometrySettings,
    PhotonSequenceResult,
    ProgressCallback,
    ProgressUpdate,
    TraceMode,
)
from .types.scientific import FixedApertureSettings
from .types.tables import identifiers

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from .cpu import WorkerProgress
    from .types import ImageSource
    from .types.acquisition import ChannelView

_DEFAULT_SETTINGS = PhotometrySettings()

_METRICS = ("sum", "mean", "median", "min", "max", "std")
_PSF_METRICS = (
    "psf_amplitude", "psf_background", "psf_sx", "psf_sy",
    "psf_lpx", "psf_lpy", "psf_likelihood",
)


def _region(
    shape: tuple[int, int],
    x: float,
    y: float,
    radius: float,
    aperture: ApertureShape,
    *,
    inner: float | None = None,
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
    x0, x1 = min(max(cx - half, 0), shape[1]), min(max(cx + half + 1, 0), shape[1])
    y0, y1 = min(max(cy - half, 0), shape[0]), min(max(cy + half + 1, 0), shape[0])
    yy = (np.arange(y0, y1) - cy)[:, None]
    xx = (np.arange(x0, x1) - cx)[None, :]
    distance = (
        np.maximum(abs(xx), abs(yy))
        if aperture == ApertureShape.SQUARE
        else np.hypot(xx, yy)
    )
    mask = distance <= radius
    if inner is not None:
        mask &= distance > inner
    return (slice(y0, y1), slice(x0, x1)), mask


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


class _FrameProgress:
    def __init__(self, stop: Event, cancel_event: CancelEvent | None) -> None:
        self.stop = stop
        self.cancel_event = cancel_event

    def check_cancelled(self) -> None:
        if self.stop.is_set() or (self.cancel_event is not None and self.cancel_event.is_set()):
            raise CancelledError("Photometry cancelled")

    def report(self, *_: object) -> None:
        # The calling thread reports completed frames.
        self.check_cancelled()


@logged
def _job(payload: tuple, reporter: WorkerProgress | _FrameProgress) -> list[dict]:
    """Job.

    Args:
        payload: Serializable worker inputs; large arrays remain bounded to a job or image chunk.
        reporter: Worker progress and cooperative cancellation reporter.

    Returns:
        Computed list[dict] value.
    """
    reporter.check_cancelled()
    image, requests, neighbours, settings = payload
    image = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(image)
    radius = (
        settings.spot_size // 2
        if settings.shape == ApertureShape.SQUARE
        else settings.spot_size / 2
    )
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
        inner = radius + settings.buffer
        region, ring = _region(
            image.shape, x, y, inner + settings.background_width, settings.shape, inner=inner
        )
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
            (
                "local_background"
                if settings.background == BackgroundMethod.LOCAL
                else "global_background"
            )
        ]
        row["intensity"] = summaries["signal"]["sum"] - chosen["mean"] * signal.size
        output.append(row)
        reporter.report(index + 1, len(requests), "Aperture photometry")
    return output


def _measure_frames(
    jobs: Iterator[tuple],
    total: int,
    num_cpus: int | None,
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
    *,
    worker: Callable[[tuple, _FrameProgress], list[dict]] = _job,
) -> list[list[dict]]:
    """Read frames on the caller and bound the number queued for measurement."""
    # Small aperture reductions benefit from avoiding thread contention.
    workers = CpuExecutor(num_cpus=1 if num_cpus is None else num_cpus).num_cpus
    stop = Event()
    reporter = _FrameProgress(stop, cancel_event)
    last_update = float("-inf")

    def report(completed: int, description: str, *, force: bool = False) -> None:
        nonlocal last_update
        reporter.check_cancelled()
        now = monotonic()
        if progress is not None and (force or now - last_update >= 0.05):
            progress(ProgressUpdate(completed, total, description))
            last_update = now
            reporter.check_cancelled()

    results = []
    if total:
        report(0, "Starting", force=True)
        if min(workers, total) == 1:
            for payload in jobs:
                results.append(worker(payload, reporter))
                report(len(results), "Aperture photometry")
        else:
            with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
                pending = deque()
                try:
                    for payload in jobs:
                        pending.append(pool.submit(worker, payload, reporter))
                        if len(pending) == workers:
                            results.append(pending.popleft().result())
                            report(len(results), "Aperture photometry")
                    while pending:
                        results.append(pending.popleft().result())
                        report(len(results), "Aperture photometry")
                except BaseException:
                    stop.set()
                    for future in pending:
                        future.cancel()
                    raise
    report(total, "Complete", force=True)
    return results


def _columns(
    rows: list[dict], fixed: FixedApertureSettings | None = None
) -> dict[str, np.ndarray]:
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
    }
    if fixed is None:
        prefixes = ("signal", "local_background", "global_background")
        dtypes.update({f"{prefix}_count": np.int64 for prefix in prefixes})
    else:
        prefixes = ("signal", "local_window", "masked_local_ring")
        if fixed.global_background:
            prefixes += ("global_background", "masked_global_background")
        if fixed.psf_box_size is not None:
            dtypes.update({name: float for name in _PSF_METRICS})
    for prefix in prefixes:
        dtypes.update({f"{prefix}_{name}": float for name in _METRICS})
    return {
        name: np.asarray([row[name] for row in rows], dtype=dtype)
        for name, dtype in dtypes.items()
    }


def fixed_aperture_masks(settings: FixedApertureSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the centred fixed aperture masks.

    Args:
        settings: Aperture geometry in image pixels.

    Returns:
        Signal, exclusion buffer and background ring masks.
    """
    width = settings.spot_size + 2 * (settings.buffer + settings.background_width)
    axis = np.arange(width) - (width - 1) / 2
    radius = settings.spot_size // 2
    inner = radius + settings.buffer
    if settings.shape == ApertureShape.SQUARE:
        distance = np.maximum(abs(axis[:, None]), abs(axis[None, :]))
        buffer = distance <= inner
    else:
        distance = np.hypot(axis[:, None], axis[None, :])
        buffer = (distance > radius) & (distance <= inner)
    return distance <= radius, buffer, (distance > inner) & (distance <= inner + settings.background_width)


def _fixed_mask_region(
    shape: tuple[int, int], width: int, x: float, y: float
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    shift = 0.5 if width % 2 == 0 else 0
    origin_x = round(x + shift) - width // 2
    origin_y = round(y + shift) - width // 2
    x0, x1 = max(0, origin_x), min(shape[1], origin_x + width)
    y0, y1 = max(0, origin_y), min(shape[0], origin_y + width)
    if x1 <= x0 or y1 <= y0:
        return np.s_[:0, :0], np.s_[:0, :0]
    return np.s_[y0:y1, x0:x1], np.s_[y0-origin_y:y1-origin_y, x0-origin_x:x1-origin_x]


def _fixed_job(payload: tuple, reporter: WorkerProgress | _FrameProgress) -> list[dict]:
    image, rows, regions, exclusion, masks, settings = payload
    finite = np.isfinite(image)
    signal_mask, _, ring_mask = masks
    global_summaries = {}
    if settings.global_background:
        global_summaries = {
            "global_background": _summary(image[finite]),
            "masked_global_background": _summary(image[finite & ~exclusion]),
        }
    for row, (image_slice, mask_slice) in zip(rows, regions, strict=True):
        reporter.check_cancelled()
        values = image[image_slice]
        valid = finite[image_slice]
        signal = values[valid & signal_mask[mask_slice]]
        ring = values[valid & ring_mask[mask_slice] & ~exclusion[image_slice]]
        summaries = {
            "signal": _summary(signal),
            "local_window": _summary(values[valid]),
            "masked_local_ring": _summary(ring),
            **global_summaries,
        }
        for prefix, metrics in summaries.items():
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        row["intensity"] = summaries["signal"]["sum"]
        if settings.psf_box_size is not None:
            row.update(_fixed_psf(image, row["x"], row["y"], settings.psf_box_size))
    return rows


def _fixed_psf(image: np.ndarray, x: float, y: float, width: int) -> dict[str, float]:
    """Fit a complete uncalibrated patch with Picasso's integrated Gaussian."""
    from picasso.gaussmle import gaussmle

    region, _ = _fixed_mask_region(image.shape, width, x, y)
    spot = image[region]
    if spot.shape != (width, width) or not np.isfinite(spot).all():
        return dict.fromkeys(_PSF_METRICS, float("nan"))
    theta, bounds, likelihood, _ = gaussmle(
        spot[None].astype(np.float32), eps=0.0001, max_it=500, method="sigma"
    )
    amplitude = float(theta[0, 2])
    sigma = float(theta[0, 4])
    values = (
        amplitude, float(theta[0, 3]) if amplitude > 0 else 0.0,
        sigma, sigma, float(np.sqrt(bounds[0, 1])),
        float(np.sqrt(bounds[0, 0])), float(likelihood[0]),
    )
    return dict(zip(_PSF_METRICS, values, strict=True))


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
        num_cpus: Measurement threads without an executor, defaulting to one.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed PhotometryResult value.
    """
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
    with dataset._lock:
        initial = dataset.dependencies()
        if mode == TraceMode.TRACK:
            rows = dataset.tracks.query(track_id=selected, space="raw")
        else:
            rows = dataset.localizations.query(ids=selected, space="raw")
            if len(rows["id"]) != len(selected):
                raise ValueError("Unknown localization ID")
        selected_acquisitions = set(rows["acquisition_id"].tolist())
        expected = dependency_token(
            dataset,
            (
                (a, c)
                for a in selected_acquisitions
                for c in dataset.acquisitions[a].channels
            ),
            components={"localizations", "tracks", "transforms", "segmentations"},
        )
        initial.validate(dataset)
    requests = defaultdict(list)
    transforms = dataset.transforms

    @cache
    def channel_setup(acquisition: str, channel: str) -> tuple[ChannelView, np.ndarray]:
        view = dataset.acquisitions[acquisition].channels[channel]
        frames = view.source_frames
        if frame_range is not None:
            start, end = np.searchsorted(frames, frame_range)
            frames = frames[start:end]
        return view, frames

    @cache
    def inverse_registration(acquisition: str, channel: str) -> np.ndarray:
        return np.linalg.inv(transforms._registration(acquisition, channel))

    @cache
    def frame_info(acquisition: str, channel: str, frame: int) -> FrameInfo:
        view, _ = channel_setup(acquisition, channel)
        return view.frame_info(view.channel_index(frame))

    @cache
    def displacement(acquisition: str, frame: int) -> np.ndarray:
        return transforms._displacement(acquisition, np.array([frame]))[0]

    for index in range(len(rows["id"])):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Photometry cancelled")
        a, source, anchor_frame = (
            str(rows["acquisition_id"][index]),
            str(rows["channel"][index]),
            int(rows["frame"][index]),
        )
        tid = int(rows["track_id"][index]) if mode == TraceMode.TRACK else 0
        lid = int(rows["id"][index])
        molecule_id = f"track:{tid}" if tid else f"localization:{lid}"
        point = np.array([rows["x"][index], rows["y"][index]])
        reference = project_points(transforms._registration(a, source), point)
        follows_drift = (
            mode == TraceMode.STATIONARY and settings.follow_drift and a in transforms._drift
        )
        if follows_drift:
            reference -= displacement(a, anchor_frame)
        for channel in channels if channels is not None else [source]:
            _, frames = channel_setup(a, channel)
            if mode != TraceMode.STATIONARY:
                anchor = np.searchsorted(frames, anchor_frame)
                frames = frames[anchor : anchor + 1]
                if not len(frames) or frames[0] != anchor_frame:
                    continue
            if not len(frames):
                continue
            inverse = inverse_registration(a, channel)
            if not follows_drift:
                target = project_points(inverse, reference)
                if not np.isfinite(target).all():
                    raise ValueError("Photometry position lies on a projective pole")
            for frame in frames:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError("Photometry cancelled")
                frame = int(frame)
                if follows_drift:
                    target = project_points(inverse, reference + displacement(a, frame))
                    if not np.isfinite(target).all():
                        raise ValueError("Photometry position lies on a projective pole")
                info = frame_info(a, channel, frame)
                requests[a, channel, frame].append(
                    {
                        "molecule_id": molecule_id,
                        "acquisition_id": a,
                        "channel": channel,
                        "frame": frame,
                        "channel_frame": info.channel_frame,
                        "cycle_index": (
                            info.cycle_index if info.cycle_index is not None else -1
                        ),
                        "timestamp": (
                            info.timestamp if info.timestamp is not None else np.nan
                        ),
                        "localization_id": lid,
                        "track_id": tid,
                        "x": float(target[0]),
                        "y": float(target[1]),
                    }
                )

    def jobs() -> Iterator[tuple]:
        """Jobs.

        Yields:
            Values produced by the iterator, retaining their original order.
        """
        for (a, channel, frame), group in requests.items():
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("Photometry cancelled")
            view, _ = channel_setup(a, channel)
            neighbours = [(row["x"], row["y"]) for row in group]
            if settings.exclude_neighbours:
                other = dataset.localizations.query(
                    acquisition_id=a, frame_range=(frame, frame + 1), space="raw"
                )
                if len(other["id"]):
                    x, y = transforms.into_channel(other, channel)
                    neighbours.extend(zip(x, y, strict=True))
            image = view[group[0]["channel_frame"]]
            if type(view.acquisition.source) is not ArraySource:
                # Some lazy readers reuse their frame buffer on the next read.
                image = np.array(image, copy=True)
            yield image, group, neighbours, settings

    if cpu is None:
        chunks = _measure_frames(jobs(), len(requests), num_cpus, progress, cancel_event)
    else:
        with executor(cpu, num_cpus) as runner:
            chunks = runner.map(
                _job,
                jobs(),
                total=len(requests),
                progress=progress,
                cancel_event=cancel_event,
            )
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
    expected.validate(dataset)
    return apply_photometry(dataset, result) if apply else result


@logged
def apply_photometry(
    dataset: Dataset,
    result: PhotometryResult | PhotonSequenceResult,
    *,
    expected: FreshnessToken | None = None,
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
                metadata={
                    "settings": asdict(result.settings),
                    "mode": str(result.mode),
                },
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

    return transact(
        dataset, commit, expected=result.expected if expected is None else expected
    )


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
        num_cpus: Measurement threads without an executor, defaulting to one.
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
        num_cpus: Measurement threads without an executor, defaulting to one.
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
def stationary_site_traces(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    sites: Mapping[str, tuple[float, float]],
    *,
    source: ImageSource | None = None,
    settings: PhotometrySettings | FixedApertureSettings = _DEFAULT_SETTINGS,
    apply: bool = True,
    num_cpus: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> PhotonSequenceResult:
    """Measure fixed sites directly in a logical channel's pixel space.

    Each site key identifies a molecule. A supplied working source must have
    the same shape and frame ordering as the raw channel. Sites then refer to
    working pixels and are omitted from raw crop positions. No localisations
    or tracks are created. Centres stay fixed in the supplied pixel space.

    Args:
        dataset: Model providing the original acquisition and camera frame clock.
        acquisition_id: Original acquisition identifier.
        channel: Logical channel whose frames are sampled.
        sites: Stable molecule IDs mapped to (x, y) positions.
        source: Optional replacement pixels in the same logical frame order.
        settings: Aperture and background measurements to compute.
        apply: Commit the completed sequences to the model.
        num_cpus: Number of measurement threads.
        progress: Callback receiving progress updates.
        cancel_event: Cooperative cancellation event.

    Returns:
        Detached photon sequences, with their stored IDs when applied.
    """
    expected = dataset.dependencies(acquisition_id, channel)
    view = dataset.acquisitions[acquisition_id].channels[channel]
    pixels = view if source is None else source
    if pixels.shape != view.shape:
        raise ValueError("Working source shape must match the logical channel")
    if not isinstance(settings, (PhotometrySettings, FixedApertureSettings)):
        raise TypeError("settings must be PhotometrySettings or FixedApertureSettings")
    positions = dict(sites)
    for key, position in positions.items():
        if (
            not isinstance(key, str) or not key
            or np.shape(position) != (2,) or not np.isfinite(position).all()
        ):
            raise ValueError("Each site needs a nonempty identifier and a finite (x, y) centre")

    fixed = settings if isinstance(settings, FixedApertureSettings) else None
    if fixed is not None:
        masks = fixed_aperture_masks(fixed)
        regions = tuple(
            _fixed_mask_region(view.shape[1:], masks[0].shape[0], x, y)
            for x, y in positions.values()
        )
        exclusion = np.zeros(view.shape[1:], dtype=bool)
        for image_slice, mask_slice in regions:
            exclusion[image_slice] |= masks[1][mask_slice]

    def jobs() -> Iterator[tuple]:
        """Yield detached working-frame jobs in original camera order.

        Yields:
            Frame pixels, site requests and precomputed measurement geometry.
        """
        for index in range(len(view)):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("Photometry cancelled")
            info = view.frame_info(index)
            rows = [
                {
                    "molecule_id": molecule_id,
                    "acquisition_id": acquisition_id,
                    "channel": channel,
                    "frame": info.camera_frame,
                    "channel_frame": index,
                    "cycle_index": info.cycle_index if info.cycle_index is not None else -1,
                    "timestamp": info.timestamp if info.timestamp is not None else np.nan,
                    "localization_id": 0,
                    "track_id": 0,
                    "x": x,
                    "y": y,
                }
                for molecule_id, (x, y) in positions.items()
            ]
            image = np.array(pixels[index], copy=True)
            if fixed is None:
                yield image, rows, positions.values(), settings
            else:
                yield image, rows, regions, exclusion, masks, fixed

    chunks = _measure_frames(
        jobs() if positions else iter(()),
        len(view) if positions else 0,
        num_cpus,
        progress,
        cancel_event,
        worker=_job if fixed is None else _fixed_job,
    )
    columns = _columns([row for chunk in chunks for row in chunk], fixed)
    if source is not None:
        columns["working_x"] = columns.pop("x")
        columns["working_y"] = columns.pop("y")

    from .photon_sequences import from_columns

    run_id = str(uuid4())
    sequences = from_columns(
        columns,
        run_id=run_id,
        metadata={"settings": asdict(settings), "source_space": "working" if source is not None else "raw"},
    )
    result = PhotonSequenceResult(sequences, settings, TraceMode.STATIONARY, run_id, expected)
    expected.validate(dataset)
    return apply_photometry(dataset, result) if apply else result


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
        num_cpus: Measurement threads without an executor, defaulting to one.
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
