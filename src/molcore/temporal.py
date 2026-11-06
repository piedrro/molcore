"""Temporal processing on frame-addressable microscopy sources."""

from collections import deque

import numpy as np

from .files import report
from .types import ArraySource, CancelEvent, ProgressCallback
from .types._shared import ImageSource


def rolling_filter(
    source: ImageSource,
    radius: int,
    *,
    method: str = "mean",
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> ArraySource:
    """Apply a centred rolling mean or median with truncated endpoints.

    Each source frame is read once. The detached output retains the input
    dtype and geometry, while the working window holds at most 2 * radius + 1
    frames. The input is left untouched if computation fails or is cancelled.

    Args:
        source: Frame-addressable movie in time, y, x order.
        radius: Number of frames on each side of the centre.
        method: Mean or median reduction across the time window.
        progress: Callback receiving progress updates.
        cancel_event: Cooperative cancellation event.

    Returns:
        Detached filtered pixels in the original geometry and dtype.
    """
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("Temporal filter radius must be a nonnegative integer")
    if method not in ("mean", "median"):
        raise ValueError("Temporal filter method must be mean or median")
    reduce = np.mean if method == "mean" else np.median
    count = source.shape[0]
    output = np.empty(source.shape, dtype=source.dtype)
    window = deque()
    next_frame = 0
    for index in range(count):
        report(progress, cancel_event, index, count, "Filtering time series")
        if index > radius:
            window.popleft()
        while next_frame < min(count, index + radius + 1):
            report(None, cancel_event)
            window.append(np.array(source[next_frame], copy=True))
            next_frame += 1
        output[index] = reduce(np.stack(window), axis=0)
    report(progress, cancel_event, count, count, "Filtering complete")
    return ArraySource(output)
