"""Trajectory statistics using actual elapsed times and explicit spatial units."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._logging import logged
from .types import CoordinateSpace, Dataset, DiffusionFit, SpatialUnit, TrajectoryStatistics


@logged
def trajectory_statistics(
    dataset: Dataset,
    track_id: int,
    *,
    space: CoordinateSpace = CoordinateSpace.CORRECTED,
    unit: SpatialUnit = SpatialUnit.PIXELS,
    frame_interval: float | None = None,
    lag_bin_edges: ArrayLike | None = None,
    fit_lag_range: tuple[float, float] | None = None,
) -> TrajectoryStatistics:
    """Compute steps, MSD and optional unweighted 2D diffusion fit for one track.

    Recorded timestamps take precedence over an explicit camera-frame interval.
    Every acquisition must have finite, strictly increasing selected timestamps.
    Duration/path length sum only within acquisitions. MSD uses within-acquisition
    position pairs grouped by actual elapsed lag (12 significant digits), or by
    explicit half-open lag bins with the last bin including its upper edge. Bin
    lag is the mean observed lag; empty bins return NaN with count zero.

    Fits require an explicit inclusive lag range and at least two distinct lags.
    Negative slopes are returned without clipping. Physical calibration is required
    for nanometres. Coordinates and masks are read without changing the model.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        track_id: Track id used by this operation.
        space: Coordinate space: raw channel pixels, registered reference or drift-corrected reference.
        unit: Unit used by this operation.
        frame_interval: Frame interval used by this operation.
        lag_bin_edges: Lag bin edges used by this operation.
        fit_lag_range: Fit lag range used by this operation.

    Returns:
        Computed TrajectoryStatistics value.
    """
    if not isinstance(space, CoordinateSpace) or not isinstance(unit, SpatialUnit):
        raise TypeError("Use CoordinateSpace and SpatialUnit enums")
    if frame_interval is not None and (not np.isfinite(frame_interval) or frame_interval <= 0):
        raise ValueError("frame_interval must be finite and positive")
    edges = None if lag_bin_edges is None else np.asarray(lag_bin_edges, dtype=float)
    if edges is not None and (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or edges[0] < 0
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("Lag bin edges must be finite, nonnegative and strictly increasing")
    if fit_lag_range is not None and (
        len(fit_lag_range) != 2
        or not np.isfinite(fit_lag_range).all()
        or fit_lag_range[0] < 0
        or fit_lag_range[1] <= fit_lag_range[0]
    ):
        raise ValueError("Fit lag range must be finite and increasing")
    rows = dataset.tracks.query(track_id=track_id, space=space)
    steps = []
    aggregates = {}
    duration = path_length = 0.0
    for acquisition_id in dict.fromkeys(rows["acquisition_id"]):
        subset = {k: v[rows["acquisition_id"] == acquisition_id] for k, v in rows.items()}
        order = np.argsort(subset["frame"], kind="stable")
        frames = subset["frame"][order]
        acquisition = dataset.acquisitions[acquisition_id]
        if acquisition.timestamps is not None:
            times = np.asarray(acquisition.timestamps, dtype=float)[frames]
        elif frame_interval is not None:
            times = frames.astype(float) * frame_interval
        else:
            raise ValueError("Recorded timestamps or explicit camera-frame interval are required")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(
                "Track timestamps must be finite and strictly increasing within each acquisition"
            )
        scale = 1.0
        if unit == SpatialUnit.NANOMETRES:
            scale = acquisition.pixel_size
            if scale is None or not np.isfinite(scale) or scale <= 0:
                raise ValueError("Positive pixel calibration required for physical units")
        positions = np.column_stack([subset["x"][order], subset["y"][order]]) * scale
        if len(frames):
            duration += float(times[-1] - times[0])
        for index in range(1, len(frames)):
            delta = positions[index] - positions[index - 1]
            distance = float(np.linalg.norm(delta))
            elapsed = float(times[index] - times[index - 1])
            path_length += distance
            steps.append(
                (
                    str(acquisition_id),
                    int(frames[index - 1]),
                    int(frames[index]),
                    float(times[index - 1]),
                    float(times[index]),
                    float(delta[0]),
                    float(delta[1]),
                    distance,
                    elapsed,
                    distance / elapsed,
                    int(frames[index] - frames[index - 1] - 1),
                )
            )
        for i in range(len(frames) - 1):
            lags = times[i + 1 :] - times[i]
            squared = np.sum((positions[i + 1 :] - positions[i]) ** 2, axis=1)
            for lag, value in zip(lags, squared, strict=True):
                key = (
                    float(f"{lag:.12g}")
                    if edges is None
                    else int(np.searchsorted(edges, lag, side="right") - 1)
                )
                if edges is not None:
                    if lag == edges[-1]:
                        key = len(edges) - 2
                    if lag < edges[0] or lag > edges[-1]:
                        continue
                count, total_lag, total_value = aggregates.get(key, (0, 0.0, 0.0))
                aggregates[key] = count + 1, total_lag + float(lag), total_value + float(value)
    keys = sorted(aggregates) if edges is None else range(len(edges) - 1)
    msd_rows = []
    for key in keys:
        count, lag, value = aggregates.get(key, (0, 0.0, 0.0))
        msd_rows.append((lag / count if count else np.nan, value / count if count else np.nan, count))
    msd = {
        name: np.asarray([row[i] for row in msd_rows], dtype=dtype)
        for i, (name, dtype) in enumerate((("lag", float), ("msd", float), ("count", np.int64)))
    }
    fit = None
    if fit_lag_range is not None:
        chosen = (msd["lag"] >= fit_lag_range[0]) & (msd["lag"] <= fit_lag_range[1]) & (msd["count"] > 0)
        x, y = msd["lag"][chosen], msd["msd"][chosen]
        if len(x) < 2 or len(np.unique(x)) < 2:
            raise ValueError("Diffusion fitting requires at least two distinct populated lags in range")
        slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
        rss = float(np.sum((y - (slope * x + intercept)) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        fit = DiffusionFit(
            float(slope / 4),
            float(intercept),
            float(slope),
            1.0 - rss / tss if tss else (1.0 if np.allclose(y, slope * x + intercept) else 0.0),
            rss,
            len(x),
        )
    step_fields = (
        ("acquisition_id", str),
        ("frame_start", np.int64),
        ("frame_end", np.int64),
        ("time_start", float),
        ("time_end", float),
        ("dx", float),
        ("dy", float),
        ("distance", float),
        ("elapsed_time", float),
        ("speed", float),
        ("gap_frames", np.int64),
    )
    columns = {
        name: np.asarray([row[i] for row in steps], dtype=dtype)
        for i, (name, dtype) in enumerate(step_fields)
    }
    return TrajectoryStatistics(int(track_id), space, unit, columns, msd, duration, path_length, fit)
