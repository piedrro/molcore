"""Detached, immutable localization query values with explicit scientific context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._shared import CoordinateSpace
from .column_batch import ColumnBatch as ColumnBatch
from .column_batch import ObservationBatch as ObservationBatch


@dataclass(frozen=True)
class ColumnSchema:
    """Actual NumPy dtype and optional declared physical unit of a column."""

    dtype: np.dtype
    unit: str | None = None
    nullable: bool = False


class LocalizationBatch(ColumnBatch):
    dataset_id: str
    space: CoordinateSpace
    original: bool
    sequence_id: str | None
    reference: str | None
    reference_context: Any
    transform_revision: int
    "Immutable localization columns with coordinate and frame context."

    def __init__(
        self,
        columns: Any,
        *,
        dataset_id: str,
        space: CoordinateSpace | str = "raw",
        original: bool = False,
        sequence_id: str | None = None,
        reference: str | None = None,
        reference_context: Any = None,
        transform_revision: int = 0,
        schema: Any = None,
        validity: Any = None,
        n_rows: int | None = None,
    ) -> None:
        super().__init__(
            columns,
            schema=schema,
            validity=validity,
            n_rows=n_rows,
            dataset_id=dataset_id,
            space=CoordinateSpace(space),
            original=bool(original),
            sequence_id=sequence_id,
            reference=reference,
            reference_context=reference_context or {},
            transform_revision=int(transform_revision),
        )

    @property
    def frame_context(self) -> Literal["camera", "sequence"]:
        """Frame context.

        Returns:
            The computed value.
        """
        return "camera" if self.sequence_id is None else "sequence"


class TrackMembershipBatch(LocalizationBatch):
    """Localization rows with stable track memberships."""
