"""Non-destructive localization and whole-track numeric selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._analysis import finish, partition, start
from ._logging import logged
from .types.analysis import FilterResult, FilterSettings

if TYPE_CHECKING:
    from .types import CancelEvent, CoordinateSpace, Experiment, ProgressCallback


_DEFAULT_SETTINGS = FilterSettings()


@logged
def filter_results(
    experiment: Experiment,
    acquisition_id: str,
    channel: str,
    *,
    settings: FilterSettings = _DEFAULT_SETTINGS,
    space: CoordinateSpace | str = "raw",
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> FilterResult:
    """Select finite inclusive values without removing or modifying any source rows.

    Args:
        experiment: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        settings: Validated settings controlling this operation.
        space: Coordinate space: raw channel pixels, registered reference or drift-corrected reference.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed FilterResult value.
    """
    expected, space = start(experiment, space, progress, cancel_event)
    partition(experiment, acquisition_id, channel, space)
    if settings.target == "tracks":
        rows = experiment.tracks.query(acquisition_id=acquisition_id, channel=channel, space=space)
        rows = experiment.tracks.query(track_id=np.unique(rows["track_id"]), space=space)
    else:
        _, rows = partition(experiment, acquisition_id, channel, space)
    if settings.metric not in rows or rows[settings.metric].dtype.kind not in "iuf":
        raise ValueError("Select an available numeric metric")
    values = np.asarray(rows[settings.metric], dtype=float)
    if settings.target == "tracks":
        ids = np.unique(rows["track_id"])
        reduced = []
        for track in ids:
            members = values[(rows["track_id"] == track) & np.isfinite(values)]
            reduced.append(getattr(np, settings.aggregation)(members) if len(members) else np.nan)
        values = np.asarray(reduced)
        ids = ids[np.isfinite(values) & (values >= settings.minimum) & (values <= settings.maximum)]
        mask = np.isin(rows["track_id"], ids)
    else:
        mask = np.isfinite(values) & (values >= settings.minimum) & (values <= settings.maximum)
        ids = rows["id"][mask]
    finish(experiment, expected, progress, cancel_event)
    return FilterResult(
        acquisition_id,
        channel,
        space,
        settings,
        {k: v[mask] for k, v in rows.items()},
        expected,
        ids.copy(),
        values[np.isfinite(values)].copy(),
    )
