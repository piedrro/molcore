"""Mutation bookkeeping shared by the model's public mutators."""

from __future__ import annotations

import inspect
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
    result["calibrations"] = tuple((k, id(v)) for k, v in dataset.calibrations.items())
    result["acquisitions"] = tuple((k, id(v)) for k, v in dataset.acquisitions.items())
    result["sequences"] = tuple(dataset.sequences)
    result["segmentations"] = tuple(
        (k, id(v)) for k, v in dataset.segmentations.items()
    )
    result["segmentation_outlines"] = tuple(dataset.segmentation_outlines.values())
    result["bactfit_cells"] = tuple(dataset.bactfit_cells.values())
    result["metadata"] = dataset._metadata_revision
    return result


def partitions(
    dataset: Any, function: Any, args: tuple, kwargs: dict
) -> set[tuple[str, str]]:
    """Resolve the mutation's input identifiers before deletion or replacement.

    Args:
        dataset: Model containing the referenced entities.
        function: Public mutator whose signature describes the selection.
        args: Positional arguments including the receiver.
        kwargs: Named arguments.

    Returns:
        Conservatively affected acquisition/channel pairs.
    """
    values = dict(inspect.signature(function).bind(*args, **kwargs).arguments)
    values.update(values.get("values", {}))
    if "segmentation_id" in values and function.__name__ == "remove_segmentation":
        mask = dataset.segmentations[values["segmentation_id"]]
        return {(mask.acquisition_id, mask.channel)}
    if "segmentation" in values:
        mask = values["segmentation"]
        selected = {(mask.acquisition_id, mask.channel)}
        old = dataset.segmentations.get(mask.id)
        if old is not None:
            selected.add((old.acquisition_id, old.channel))
        return selected
    if args[0] is dataset.photon_sequences:
        if "ids" in values:
            return {
                (sequence.acquisition_id, sequence.channel)
                for sequence in dataset.photon_sequences._data.values()
            }
        sequence = values.get("sequence")
        if sequence is None and "id" in values:
            sequence = dataset.photon_sequences._data[values["id"]]
        if sequence is not None:
            return {(sequence.acquisition_id, sequence.channel)}
    if args[0] is dataset.transforms and "calibration" in values:
        name = values["calibration"].name
        return {
            key
            for key, selected in dataset.transforms._selected.items()
            if selected == name
        }
    acquisition = values.get("acquisition_id")
    for key in ("acquisition", "segmentation", "sequence"):
        item = values.get(key)
        if item is not None:
            acquisition = getattr(
                item, "acquisition_id", getattr(item, "id", acquisition)
            )
    if acquisition is not None:
        acquisitions = set(np.atleast_1d(acquisition).tolist())
        channel = values.get("channel")
        channels = None if channel is None else set(np.atleast_1d(channel).tolist())
        # Optical selection may bind an entire physical emission group.
        if args[0] is dataset.transforms:
            channels = None
        return {
            (a, c)
            for a in acquisitions
            if a in dataset.acquisitions
            for c in dataset.acquisitions[a].channels
            if channels is None or c in channels
        }
    ids = values.get(
        "localization_ids", values.get("localization_id", values.get("ids"))
    )
    track_ids = values.get("track_ids", values.get("track_id"))
    if track_ids is not None:
        members = [
            dataset.tracks._members.get(int(i), []) for i in np.atleast_1d(track_ids)
        ]
        ids = np.concatenate(members) if members else []
    if ids is not None and args[0] is not dataset.observations:
        data = dataset.localizations._data
        if data:
            selected = np.isin(data["id"], ids)
            return set(
                zip(
                    data["acquisition_id"][selected].tolist(),
                    data["channel"][selected].tolist(),
                    strict=True,
                )
            )
    return {(a, c) for a, value in dataset.acquisitions.items() for c in value.channels}
