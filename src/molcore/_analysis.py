"""Shared snapshot validation and partition selection for analysis APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._logging import logged
from .files import report
from .revisions import StateToken
from .staging import state_token
from .types import CoordinateSpace, StaleResultError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .types import CancelEvent, Dataset, ProgressCallback
    from .types.acquisition import ChannelView


@logged
def partition(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    space: CoordinateSpace | str,
    ids: ArrayLike | None = None,
) -> tuple[ChannelView, dict[str, np.ndarray]]:
    """Partition.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        space: Coordinate space: raw channel pixels, registered reference or drift-corrected reference.
        ids: Stable entity IDs; None selects the available partition.

    Returns:
        Computed tuple[ChannelView, dict[str, NumPy array]] value.
    """
    view = dataset.acquisitions[acquisition_id].channels[channel]
    rows = dataset.localizations.query(
        acquisition_id=acquisition_id, channel=channel, space=space, ids=ids
    )
    order = np.argsort(rows["id"], kind="stable")
    return view, {key: values[order] for key, values in rows.items()}


@logged
def finish(
    dataset: Dataset,
    expected: StateToken | None,
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
) -> None:
    """Finish.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        expected: Source revision token used to reject stale results.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.
    """
    report(progress, cancel_event, 1, 1, "Analysis complete")
    if state_token(dataset) != expected:
        raise StaleResultError("Analysis inputs changed during computation")


@logged
def start(
    dataset: Dataset,
    space: CoordinateSpace | str,
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
) -> tuple[StateToken, CoordinateSpace]:
    """Start.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        space: Coordinate space: raw channel pixels, registered reference or drift-corrected reference.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed tuple[tuple, CoordinateSpace] value.
    """
    report(progress, cancel_event, 0, 1, "Computing analysis")
    return state_token(dataset), CoordinateSpace(space)
