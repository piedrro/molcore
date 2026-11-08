"""Non-destructive localization and whole-track numeric selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._analysis import finish, partition, start
from ._logging import logged
from .execution import ExecutionContext, _record_execution
from .snapshots import DatasetSnapshot, processor_model
from .types.analysis import FilterResult, FilterSettings
from .types.selections import PartitionSelection

if TYPE_CHECKING:
    from .types import CoordinateSpace
_DEFAULT_SETTINGS = FilterSettings()


@logged
@_record_execution
def filter_results(
    dataset: DatasetSnapshot,
    selection: PartitionSelection,
    *,
    settings: FilterSettings = _DEFAULT_SETTINGS,
    space: CoordinateSpace | str = "raw",
    context: ExecutionContext | None = None,
) -> FilterResult:
    """Select finite inclusive values without removing or modifying any source rows.

    Args:
        selection: Explicit acquisition/channel or sequence/channel selection.
        context: Execution policy, frame chunk size, progress and cancellation.
        dataset: Scientific model supplying acquisitions and result tables.
        settings: Validated settings controlling this operation.
        space: Coordinate space: raw channel pixels, registered reference or drift-corrected reference.
    Returns:
        Computed FilterResult value.
    """
    if not isinstance(selection, PartitionSelection):
        raise TypeError("Use PartitionSelection")
    acquisition_id, channel = (selection.acquisition_id, selection.channel)
    dataset = processor_model(dataset)
    context = context or ExecutionContext()
    context.check_cancelled()
    progress, cancel_event = (context.progress, context.cancel_event)
    expected, space = start(dataset, space, progress, cancel_event, acquisition_id, (channel,))
    partition(dataset, acquisition_id, channel, space)
    if settings.target == "tracks":
        rows = dataset.tracks.query(acquisition_id=acquisition_id, channel=channel, space=space)
        rows = dataset.tracks.query(track_id=np.unique(rows["track_id"]), space=space)
    else:
        rows = dataset.localizations.query(acquisition_id=acquisition_id, channel=channel, space=space)
    if settings.target == "tracks":
        from .staging import dependency_token

        expected = dependency_token(
            dataset,
            {(acquisition_id, channel)} | set(zip(rows["acquisition_id"], rows["channel"], strict=True)),
            components={"localizations", "tracks", "transforms"},
        )
    if settings.metric not in rows or rows[settings.metric].dtype.kind not in "iuf":
        raise ValueError("Select an available numeric metric")
    values = np.asarray(rows[settings.metric], dtype=float)
    if settings.metric in rows.validity:
        values = np.where(rows.validity[settings.metric], values, np.nan)
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
    finish(dataset, expected, progress, cancel_event)
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
