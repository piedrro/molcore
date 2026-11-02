"""Detached, immutable localization query values with explicit scientific context."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ._shared import CoordinateSpace


@dataclass(frozen=True)
class ColumnSchema:
    """Actual NumPy dtype and optional declared physical unit of a column."""

    dtype: np.dtype
    unit: str | None = None


@dataclass(frozen=True, init=False, eq=False)
class LocalizationBatch(Mapping[str, np.ndarray]):
    """Immutable column mapping. ``len(batch)`` counts columns; ``n_rows`` counts rows.

    ``frame`` always means camera frame. Sequence queries additionally expose
    ``sequence_frame``; ``frame_context`` identifies the query's range semantics.
    Unknown measurement units remain ``None`` rather than being guessed.
    """

    _columns: Mapping[str, np.ndarray]
    dataset_id: str
    space: CoordinateSpace
    original: bool
    sequence_id: str | None
    reference: str | None
    reference_context: Mapping[tuple[str, str], str | None]
    transform_revision: int
    schema: Mapping[str, ColumnSchema]

    def __init__(
        self,
        columns: Mapping[str, ArrayLike],
        *,
        dataset_id: str,
        space: CoordinateSpace | str = CoordinateSpace.RAW,
        original: bool = False,
        sequence_id: str | None = None,
        reference: str | None = None,
        reference_context: Mapping[tuple[str, str], str | None] | None = None,
        transform_revision: int = 0,
        schema: Mapping[str, ColumnSchema] | None = None,
    ) -> None:
        prepared = {}
        for name, values in columns.items():
            array = np.asarray(values)
            if array.ndim != 1 or array.dtype.hasobject:
                raise ValueError(
                    "Batch columns must be one-dimensional non-object arrays"
                )
            # A bytes backing buffer prevents callers from re-enabling writes.
            prepared[name] = np.frombuffer(array.tobytes(), dtype=array.dtype)
        if len({len(value) for value in prepared.values()}) > 1:
            raise ValueError("Batch columns must have equal lengths")
        declared = schema or {}
        if set(declared) - prepared.keys():
            raise ValueError("Schema names must identify batch columns")
        resolved = {}
        for name, value in prepared.items():
            default_unit = (
                "pixel"
                if name in {"x", "y"}
                else "frame" if name in {"frame", "sequence_frame"} else None
            )
            spec = declared.get(name, ColumnSchema(value.dtype, default_unit))
            if np.dtype(spec.dtype) != value.dtype:
                raise ValueError(f"Schema dtype does not match column {name!r}")
            resolved[name] = ColumnSchema(value.dtype, spec.unit)
        for name, attribute in {
            "_columns": MappingProxyType(prepared),
            "dataset_id": dataset_id,
            "space": CoordinateSpace(space),
            "original": bool(original),
            "sequence_id": sequence_id,
            "reference": reference,
            "reference_context": MappingProxyType(dict(reference_context or {})),
            "transform_revision": int(transform_revision),
            "schema": MappingProxyType(resolved),
        }.items():
            object.__setattr__(self, name, attribute)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._columns[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    @property
    def n_rows(self) -> int:
        """Number of localizations in this result.

        Returns:
            Row count, including zero for empty queries.
        """
        return len(next(iter(self._columns.values()))) if self._columns else 0

    @property
    def frame_context(self) -> Literal["camera", "sequence"]:
        """Timeline used to interpret the query's frame range.

        Returns:
            Camera or sequence timeline; the frame column itself stays camera-relative.
        """
        return "camera" if self.sequence_id is None else "sequence"

    def to_columns(self) -> dict[str, np.ndarray]:
        """Return independent, writable arrays for mutable algorithms.

        Returns:
            Dictionary of copied columns.
        """
        return {name: values.copy() for name, values in self.items()}

    def to_records(self) -> np.ndarray:
        """Return a detached writable structured array, preserving column dtypes.

        Returns:
            NumPy structured array with one record per localization.
        """
        result = np.empty(
            self.n_rows, dtype=[(name, values.dtype) for name, values in self.items()]
        )
        for name, values in self.items():
            result[name] = values
        return result

    def __copy__(self) -> LocalizationBatch:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> LocalizationBatch:
        return self
