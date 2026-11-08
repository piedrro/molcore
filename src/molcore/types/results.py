"""Transaction receipts independent of numerical result objects."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np

from ..immutable import immutable_record


@immutable_record
class ResultIdentity:
    """Identity and declared writes shared by detached scientific results."""

    result_id: str = field(default_factory=lambda: str(uuid4()), kw_only=True, compare=False, repr=False)
    execution: dict[str, object] = field(default_factory=dict, kw_only=True, compare=False)

    @property
    def write_scope(self) -> WriteScope:
        """Describe intended publication independently of a live model.

        Returns:
            The computed value.
        """
        return result_write_scope(self)


@immutable_record
class WriteScope:
    component: str
    mode: Literal["append", "partition", "selected", "preview"]
    partitions: tuple[tuple[str, str], ...] = ()
    selected_ids: np.ndarray | None = None


def result_write_scope(result: Any) -> Any:
    """Describe a built-in result's intended writes without reading a live model.

    Args:
        result: Result.


    Returns:
        The computed value.
    """
    name = type(result).__name__
    partitions = tuple(getattr(getattr(result, "expected", None), "partitions", ()))
    if name == "Segmentation":
        return WriteScope(
            "segmentations",
            "partition",
            tuple(
                sorted(
                    {(result.acquisition_id, result.channel)}
                    | ({result.target_partition} if result.target_partition is not None else set())
                )
            ),
        )
    if name == "LocalisationResult":
        return WriteScope(
            "localizations",
            "preview" if result.frame_range is not None or result.source_space != "raw" else "partition",
            ((result.acquisition_id, result.channel),),
        )
    if name in {"TrackingResult", "SequenceTrackingResult"}:
        ids = result.selected_ids
        return WriteScope("tracks", "partition" if ids is None else "selected", partitions, ids)
    if name in {"PhotometryResult", "PhotonSequenceResult"}:
        return WriteScope(
            "observations" if name == "PhotometryResult" else "photon_sequences", "append", partitions
        )
    if name in {"DriftResult", "AlignmentResult", "DatasetAlignmentResult"}:
        return WriteScope("transforms", "partition", partitions)
    return WriteScope("analysis", "preview", partitions)


@immutable_record
class ApplyReceipt:
    result_id: str
    revision: int
    identifiers: dict[str, object] = field(default_factory=dict)

    @property
    def localization_ids(self) -> np.ndarray | None:
        """Localization ids.

        Returns:
            The computed value.
        """
        return cast(np.ndarray | None, self.identifiers.get("localization_ids"))

    @property
    def track_ids(self) -> np.ndarray | None:
        """Track ids.

        Returns:
            The computed value.
        """
        return cast(np.ndarray | None, self.identifiers.get("track_ids"))

    @property
    def observation_ids(self) -> np.ndarray | None:
        """Observation ids.

        Returns:
            The computed value.
        """
        return cast(np.ndarray | None, self.identifiers.get("observation_ids"))

    @property
    def sequence_ids(self) -> tuple[str, ...] | None:
        """Sequence ids.

        Returns:
            The computed value.
        """
        return cast(tuple[str, ...] | None, self.identifiers.get("sequence_ids"))
