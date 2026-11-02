"""Spatial DBSCAN assignments without changing localization coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._analysis import finish, partition, start
from ._logging import logged
from .types.analysis import ClusteringResult, ClusteringSettings

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .types import CancelEvent, CoordinateSpace, Dataset, ProgressCallback


_DEFAULT_SETTINGS = ClusteringSettings()


@logged
def cluster_localizations(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    *,
    settings: ClusteringSettings = _DEFAULT_SETTINGS,
    space: CoordinateSpace | str = "raw",
    ids: ArrayLike | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> ClusteringResult:
    """Cluster a stable-ID-ordered partition; non-finite positions remain noise.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        settings: Validated settings controlling this operation.
        space: Coordinate space: raw channel pixels, registered reference or drift-corrected reference.
        ids: Stable entity IDs; None selects the available partition.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed ClusteringResult value.
    """
    from sklearn.cluster import DBSCAN

    expected, space = start(dataset, space, progress, cancel_event)
    _, rows = partition(dataset, acquisition_id, channel, space, ids)
    xy = np.column_stack((rows["x"], rows["y"]))
    valid = np.isfinite(xy).all(axis=1)
    labels = np.full(len(xy), -1, dtype=np.int64)
    if valid.any():
        labels[valid] = DBSCAN(
            eps=settings.radius, min_samples=settings.min_samples, n_jobs=1
        ).fit_predict(xy[valid])
    clusters = np.unique(labels[labels >= 0])
    centroids = {
        "cluster": clusters,
        "count": np.array([np.sum(labels == c) for c in clusters], dtype=np.int64),
        "x": np.array([xy[labels == c, 0].mean() for c in clusters]),
        "y": np.array([xy[labels == c, 1].mean() for c in clusters]),
    }
    rows["cluster"] = labels
    finish(dataset, expected, progress, cancel_event)
    return ClusteringResult(
        acquisition_id, channel, space, settings, rows, expected, centroids
    )
