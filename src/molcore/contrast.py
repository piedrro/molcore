"""Display-only autocontrast for arrays and lazy microscopy channels."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from ._logging import logged
from .staging import state_token
from .types import CancelEvent, Experiment, ProgressCallback, ProgressUpdate, StaleResultError
from .types.contrast import ContrastMethod, ContrastResult, ContrastScope, ContrastSettings

_DEFAULT_SETTINGS = ContrastSettings()


@dataclass
class _Samples:
    values: list[np.ndarray] = field(default_factory=list)
    minimum: float = float("inf")
    maximum: float = float("-inf")
    valid_count: int = 0
    sample_count: int = 0


def _indices(length: int, cap: int | None) -> np.ndarray:
    """Indices.

    Args:
        length: Number of valid items in the indexed source.
        cap: Cap used by this operation.

    Returns:
        Computed NumPy array value.
    """
    count = length if cap is None else min(length, cap)
    return np.linspace(0, length - 1, count, dtype=np.int64) if count else np.array([], dtype=np.int64)


def _mask(mask: ArrayLike | None, shape: tuple[int, ...]) -> np.ndarray | None:
    """Mask.

    Args:
        mask: Mask used by this operation.
        shape: Array dimensions in time, y, x order.

    Returns:
        Computed NumPy array | None value.
    """
    if mask is None:
        return None
    result = np.asarray(mask)
    if result.dtype.kind != "b":
        raise TypeError("Masks must be boolean; True includes a pixel")
    if result.shape not in (shape, shape[1:]):
        raise ValueError("Mask must match the spatial image or complete logical channel stack")
    return result


def _report(
    completed: int, total: int, progress: ProgressCallback | None, cancel_event: CancelEvent | None
) -> None:
    """Report.

    Args:
        completed: Completed work units, or the GUI completion callback for a submitted job.
        total: Total work units represented by this progress update.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Autocontrast cancelled")
    if progress is not None:
        progress(ProgressUpdate(completed, total, "Calculating display contrast"))
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Autocontrast cancelled")


def _collect(
    read: Callable[[int], np.ndarray],
    indices: np.ndarray,
    mask: np.ndarray | None,
    settings: ContrastSettings,
    report: Callable[[], None],
) -> _Samples:
    """Collect.

    Args:
        read: Read used by this operation.
        indices: Indices used by this operation.
        mask: Mask used by this operation.
        settings: Validated settings controlling this operation.
        report: Report used by this operation.

    Returns:
        Computed _Samples value.
    """
    samples = _Samples()
    for index in indices:
        image = np.asarray(read(int(index)))
        if image.ndim != 2 or image.dtype.kind not in "biuf":
            raise ValueError("Images must have real numeric pixels and two spatial dimensions")
        selected = np.isfinite(image)
        if mask is not None:
            selected &= mask if mask.ndim == 2 else mask[index]
        values = image[selected].astype(np.float64)
        if values.size:
            samples.valid_count += int(values.size)
            samples.minimum = min(samples.minimum, float(np.min(values)))
            samples.maximum = max(samples.maximum, float(np.max(values)))
            if settings.method == ContrastMethod.PERCENTILE:
                values = values[_indices(len(values), settings.max_pixels_per_frame)]
                samples.values.append(values)
            samples.sample_count += int(values.size)
        report()
    return samples


def _limits(samples: Sequence[_Samples], settings: ContrastSettings) -> tuple[float, float]:
    """Limits.

    Args:
        samples: Samples used by this operation.
        settings: Validated settings controlling this operation.

    Returns:
        Computed tuple[float, float] value.
    """
    if not any(sample.valid_count for sample in samples):
        raise ValueError("No finite unmasked pixels in the selected frames")
    if settings.method == ContrastMethod.MIN_MAX:
        lower = min(sample.minimum for sample in samples)
        upper = max(sample.maximum for sample in samples)
    else:
        values = np.concatenate([values for sample in samples for values in sample.values])
        values.sort()
        limits = []
        for percentile in settings.percentiles:
            rank = (len(values) - 1) * (percentile / 100.0)
            lo, hi = int(np.floor(rank)), int(np.ceil(rank))
            a, b = float(values[lo]), float(values[hi])
            fraction = rank - lo
            if lo == hi:
                value = a
            elif np.signbit(a) != np.signbit(b):
                # Opposite-sign endpoints can overflow b-a. A convex sum cannot,
                # and unlike global scaling it preserves tiny finite populations.
                value = (1.0 - fraction) * a + fraction * b
            elif fraction < 0.5:
                value = a + (b - a) * fraction
            else:
                value = b - (b - a) * (1.0 - fraction)
            limits.append(value)
        lower, upper = limits
    lower, upper = float(lower), float(upper)
    if lower == upper:
        # Expand ordinary constants by at least 0.5. At float64 boundaries use
        # a finite adjacent value instead of returning an infinite display limit.
        value = lower
        delta = max(0.5, abs(value) * 0.01)
        with np.errstate(over="ignore", invalid="ignore"):
            lo, hi = value - delta, value + delta
            lower = lo if np.isfinite(lo) and lo < value else float(np.nextafter(value, -np.inf))
            upper = hi if np.isfinite(hi) and hi > value else float(np.nextafter(value, np.inf))
        if not np.isfinite(lower):
            lower = value
        if not np.isfinite(upper):
            upper = value
    return lower, upper


@logged
def autocontrast(
    image: ArrayLike,
    *,
    settings: ContrastSettings = _DEFAULT_SETTINGS,
    mask: ArrayLike | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> ContrastResult:
    """Calculate display limits for a 2D image or NumPy (time,y,x) movie.

    Returns settings, sampled array-frame indices, valid-pixel and sample counts.
    Masks are boolean with True including pixels; 2D masks apply to all frames.
    Nonfinite pixels are excluded. Empty selections raise ValueError. Constant
    limits expand to a finite increasing interval. Source pixels are unchanged.
    Use autocontrast_channels for lazy, frame-addressable model image sources.

    Args:
        image: Two-dimensional source image.
        settings: Validated settings controlling this operation.
        mask: Mask used by this operation.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed ContrastResult value.
    """
    if not isinstance(settings, ContrastSettings):
        raise TypeError("settings must be ContrastSettings")
    _report(0, 1, None, cancel_event)
    data = np.asarray(image)
    if data.ndim not in (2, 3) or data.dtype.kind not in "biuf":
        raise ValueError("Expected a real 2D image or (time,y,x) movie")
    if data.ndim == 2:
        data = data[np.newaxis]
    included = _mask(mask, data.shape)
    indices = _indices(data.shape[0], settings.max_frames)
    completed = 0
    _report(0, len(indices), progress, cancel_event)

    def report() -> None:
        """Report."""
        nonlocal completed
        completed += 1
        _report(completed, len(indices), progress, cancel_event)

    samples = _collect(data.__getitem__, indices, included, settings, report)
    lower, upper = _limits([samples], settings)
    _report(completed, len(indices), None, cancel_event)
    return ContrastResult(
        lower, upper, settings, tuple(int(i) for i in indices), samples.valid_count, samples.sample_count
    )


@logged
def autocontrast_channels(
    experiment: Experiment,
    acquisition_id: str,
    *,
    channels: Sequence[str] | None = None,
    settings: ContrastSettings = _DEFAULT_SETTINGS,
    scope: ContrastScope = ContrastScope.PER_CHANNEL,
    frame_range: tuple[int, int] | None = None,
    masks: Mapping[str, ArrayLike] | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> dict[str, ContrastResult]:
    """Calculate per-channel or pooled display limits from lazy original pixels.

    Frame range is half-open in original camera frames. Each channel is sampled
    independently across its eligible logical frames; returned frames retain camera
    identities, including ALEX timing. Masks use channel geometry and, if 3D, the
    complete logical-channel frame axis. Shared scope pools pixel samples (not
    channel percentiles). Each returned result retains its own channel's counts.
    MIN_MAX reads every pixel of selected frames; percentile samples are capped.
    No observations or display settings are applied to the model. Empty channels,
    unavailable images, stale model state and cancellation raise without mutation.

    Args:
        experiment: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channels: Channels used by this operation.
        settings: Validated settings controlling this operation.
        scope: Scope used by this operation.
        frame_range: Optional half-open camera-frame interval.
        masks: Masks used by this operation.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed dict[str, ContrastResult] value.
    """
    if not isinstance(settings, ContrastSettings) or not isinstance(scope, ContrastScope):
        raise TypeError("Use ContrastSettings and ContrastScope enum")
    _report(0, 1, None, cancel_event)
    acquisition = experiment.acquisitions[acquisition_id]
    names = list(acquisition.channels) if channels is None else list(channels)
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one distinct channel")
    if any(name not in acquisition.channels for name in names):
        raise ValueError("Unknown channel")
    if masks is not None and set(masks) - set(names):
        raise ValueError("Mask names must belong to selected channels")
    if frame_range is not None:
        if (
            not isinstance(frame_range, tuple)
            or len(frame_range) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in frame_range)
            or frame_range[0] < 0
            or frame_range[1] < frame_range[0]
        ):
            raise ValueError("Expected a nonnegative half-open camera-frame range")
    expected = state_token(experiment)
    indices = {}
    included = {}
    for name in names:
        view = acquisition.channels[name]
        eligible = np.arange(len(view), dtype=np.int64)
        if frame_range is not None:
            eligible = eligible[
                (view.source_frames >= frame_range[0]) & (view.source_frames < frame_range[1])
            ]
        if not len(eligible):
            raise ValueError(f"No selected frames in channel {name}")
        indices[name] = eligible[_indices(len(eligible), settings.max_frames)]
        included[name] = _mask(None if masks is None else masks.get(name), view.shape)
    total = sum(len(values) for values in indices.values())
    completed = 0
    _report(0, total, progress, cancel_event)

    def report() -> None:
        """Report."""
        nonlocal completed
        completed += 1
        _report(completed, total, progress, cancel_event)

    samples = {}
    for name in names:
        samples[name] = _collect(
            acquisition.channels[name].__getitem__, indices[name], included[name], settings, report
        )
        if not samples[name].valid_count:
            raise ValueError(f"No finite unmasked pixels in channel {name}")
    shared = _limits(list(samples.values()), settings) if scope == ContrastScope.SHARED else None
    result = {}
    for name in names:
        lower, upper = shared if shared is not None else _limits([samples[name]], settings)
        result[name] = ContrastResult(
            lower,
            upper,
            settings,
            tuple(int(i) for i in acquisition.channels[name].source_frames[indices[name]]),
            samples[name].valid_count,
            samples[name].sample_count,
            scope,
        )
    _report(completed, total, None, cancel_event)
    if state_token(experiment) != expected:
        raise StaleResultError("Autocontrast inputs changed during computation")
    return result
