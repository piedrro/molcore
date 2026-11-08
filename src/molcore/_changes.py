"""Mutation bookkeeping shared by the model's public mutators."""

from __future__ import annotations

from typing import Any

import numpy as np


def versions(dataset: Any) -> dict[str, Any]:
    """Capture small revision markers without reading pixels or result columns.

    Args:
        dataset: Owner of the component stores.

    Returns:
        Component markers used to detect successful mutations.
    """
    result = {
        name: getattr(dataset, name).revision
        for name in (
            "localizations",
            "tracks",
            "observations",
            "photon_sequences",
            "bounding_boxes",
            "transforms",
        )
    }
    result["calibrations"] = tuple(((k, id(v)) for k, v in dataset.calibrations.items()))
    result["acquisitions"] = tuple(((k, id(v)) for k, v in dataset.acquisitions.items()))
    result["sequences"] = tuple(dataset.sequences)
    result["segmentations"] = tuple(((k, id(v)) for k, v in dataset.segmentations.items()))
    result["segmentation_outlines"] = tuple(dataset.segmentation_outlines.values())
    result["bactfit_cells"] = tuple(dataset.bactfit_cells.values())
    result["metadata"] = dataset._metadata_revision
    result["processing"] = dataset._processing_revision
    return result


def mutation_scope(
    receiver: Any,
    *,
    acquisition: Any = None,
    channel: str = None,
    ids: Any = None,
    tracks: Any = None,
    record: Any = None,
    segmentation_id: Any = None,
) -> Any:
    """Resolve an explicitly declared mutation selection before records disappear.

    Args:
        receiver: Receiver.
        acquisition: Acquisition.
        channel: Channel.
        ids: Ids.
        tracks: Tracks.
        record: Record.
        segmentation_id: Segmentation id.


    Returns:
        The computed value.
    """
    dataset = getattr(receiver, "dataset", receiver)
    if segmentation_id is not None:
        record = dataset.segmentations[segmentation_id]
    if record is not None:
        acquisition = record.acquisition_id
        channel = record.channel
        if acquisition is not None and channel is not None:
            previous = (dataset.segmentations if receiver is dataset else getattr(receiver, "_data", {})).get(
                record.id
            )
            selected = {(acquisition, channel)}
            if previous is not None and previous.acquisition_id is not None and previous.channel is not None:
                selected.add((previous.acquisition_id, previous.channel))
            return selected
    if tracks is not None:
        members = [dataset.tracks._members.get(int(value), []) for value in np.atleast_1d(tracks)]
        ids = np.concatenate(members) if members else []
    if ids is not None:
        data = dataset.localizations._data
        if not data:
            return set()
        selected = np.isin(data["id"], ids)
        return set(
            zip(data["acquisition_id"][selected].tolist(), data["channel"][selected].tolist(), strict=True)
        )
    selected_acquisitions = None if acquisition is None else set(np.atleast_1d(acquisition).tolist())
    selected_channels = None if channel is None else set(np.atleast_1d(channel).tolist())
    return {
        (a, c)
        for a, value in dataset.acquisitions.items()
        if selected_acquisitions is None or a in selected_acquisitions
        for c in value.channels
        if selected_channels is None or c in selected_channels
    }
