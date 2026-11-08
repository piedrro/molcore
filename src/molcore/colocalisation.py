"""Deterministic mutual-nearest fiducial matching in a common coordinate system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from ._analysis import finish, partition, start
from ._logging import logged
from .execution import ExecutionContext, _record_execution
from .snapshots import DatasetSnapshot, processor_model
from .types.analysis import ColocalisationResult, ColocalisationSettings
from .types.selections import PartitionSelection

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .types import CoordinateSpace


def _nearest(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest.

    Args:
        source: Image source providing lazy frame access.
        target: Target used by this operation.

    Returns:
        Computed tuple[NumPy array, NumPy array] value.
    """
    tree = cKDTree(target)
    distances, indices = tree.query(source)
    for index, (point, distance) in enumerate(zip(source, distances, strict=True)):
        candidates = tree.query_ball_point(point, np.nextafter(distance, np.inf))
        exact = [(float(np.linalg.norm(point - target[j])), j) for j in candidates]
        distances[index], indices[index] = min(exact)
    return (distances, indices)


_DEFAULT_SETTINGS = ColocalisationSettings()


@logged
@_record_execution
def colocalise(
    dataset: DatasetSnapshot,
    selection: PartitionSelection,
    target_channel: str,
    *,
    settings: ColocalisationSettings = _DEFAULT_SETTINGS,
    space: CoordinateSpace | str = "registered",
    source_ids: ArrayLike | None = None,
    target_ids: ArrayLike | None = None,
    context: ExecutionContext | None = None,
) -> ColocalisationResult:
    """Match across frames, using inclusive Euclidean distance and mutual neighbours.

    Raw matching requires identical spatial ROIs. Registered/corrected channels
    use the acquisition's common reference, including any selected calibrations.

    Args:
        selection: Explicit acquisition/channel or sequence/channel selection.
        context: Execution policy, frame chunk size, progress and cancellation.
        dataset: Scientific model supplying acquisitions and result tables.
        target_channel: Logical target channel in the same acquisition.
        settings: Validated settings controlling this operation.
        space: Coordinate space: raw channel pixels, registered reference or drift-corrected reference.
        source_ids: Optional stable IDs restricting source localizations.
        target_ids: Optional stable IDs restricting target localizations.
    Returns:
        Computed ColocalisationResult value.
    """
    if not isinstance(selection, PartitionSelection):
        raise TypeError("Use PartitionSelection")
    acquisition_id, channel = (selection.acquisition_id, selection.channel)
    dataset = processor_model(dataset)
    context = context or ExecutionContext()
    context.check_cancelled()
    progress, cancel_event = (context.progress, context.cancel_event)
    expected, space = start(dataset, space, progress, cancel_event, acquisition_id, (channel, target_channel))
    if channel == target_channel:
        raise ValueError("Select two different channels")
    source_view, source = partition(dataset, acquisition_id, channel, space, source_ids)
    target_view, target = partition(dataset, acquisition_id, target_channel, space, target_ids)
    if space == "raw" and source_view.roi != target_view.roi:
        raise ValueError("Raw channel ROIs differ; use a common registered reference")
    xy = np.column_stack((source["x"], source["y"]))
    uv = np.column_stack((target["x"], target["y"]))
    left, right = (np.flatnonzero(np.isfinite(xy).all(axis=1)), np.flatnonzero(np.isfinite(uv).all(axis=1)))
    a, b, distances = (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))
    if len(left) and len(right):
        distances, nearest = _nearest(xy[left], uv[right])
        _, reverse = _nearest(uv[right], xy[left])
        mask = (reverse[nearest] == np.arange(len(left))) & (
            distances <= np.nextafter(settings.radius, np.inf)
        )
        a, b, distances = (left[mask], right[nearest[mask]], distances[mask])
    midpoints = (xy[a] + uv[b]) / 2
    columns = {
        "source_id": source["id"][a],
        "target_id": target["id"][b],
        "distance": distances,
        "x": midpoints[:, 0],
        "y": midpoints[:, 1],
        "source_frame": source["frame"][a],
        "target_frame": target["frame"][b],
    }
    finish(dataset, expected, progress, cancel_event)
    return ColocalisationResult(
        acquisition_id,
        channel,
        space,
        settings,
        columns,
        expected,
        target_channel,
        len(xy) - len(a),
        len(uv) - len(b),
    )
