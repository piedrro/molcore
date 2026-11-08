"""Immutable projected columns and explicit per-column validity."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Self

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .batches import ColumnSchema
from typing import Any

import numpy as np

from ..immutable import ImmutableObject, freeze, immutable_array


class ColumnBatch(ImmutableObject, Mapping[str, np.ndarray]):
    """len counts columns; n_rows counts rows, including an empty projection."""

    _columns: Mapping[str, np.ndarray]
    schema: Mapping[str, ColumnSchema]
    validity: Mapping[str, np.ndarray]
    n_rows: int

    def __init__(
        self,
        columns: Mapping[str, ArrayLike],
        *,
        schema: Mapping[str, ColumnSchema] | None = None,
        validity: Mapping[str, ArrayLike] | None = None,
        n_rows: int | None = None,
        **context: Any,
    ) -> None:
        from .batches import ColumnSchema

        prepared = {key: immutable_array(value) for key, value in columns.items()}
        if any(value.ndim != 1 for value in prepared.values()):
            raise ValueError("Columns must be one-dimensional")
        lengths = {len(value) for value in prepared.values()}
        count = next(iter(lengths), 0) if n_rows is None else n_rows
        if len(lengths) > 1 or count < 0 or (lengths and lengths != {count}):
            raise ValueError("Invalid column lengths")
        declared = schema or {}
        if set(declared) - prepared.keys():
            raise ValueError("Schema names must identify batch columns")
        masks = {}
        for name, values in (validity or {}).items():
            mask = immutable_array(values)
            if name not in prepared or mask.dtype.kind != "b" or mask.shape != (count,):
                raise ValueError("Validity must be boolean and match a column")
            masks[name] = mask
        resolved = {}
        for name, values in prepared.items():
            unit = "pixel" if name in {"x", "y"} else "frame" if name in {"frame", "sequence_frame"} else None
            spec = declared.get(name, ColumnSchema(values.dtype, unit, name in masks))
            if np.dtype(spec.dtype) != values.dtype:
                raise ValueError(f"Schema dtype does not match {name!r}")
            resolved[name] = ColumnSchema(values.dtype, spec.unit, spec.nullable or name in masks)
        self._columns, self.schema, self.validity = (freeze(prepared), freeze(resolved), freeze(masks))
        self.n_rows, self._context = (int(count), freeze(context))
        for name, value in context.items():
            setattr(self, name, value)
        self._seal()

    def __getitem__(self, name: str) -> np.ndarray:
        return self._columns[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def project(self, names: Iterable[str] | None) -> Self:
        """Project.

        Args:
            names: Names.


        Returns:
            The computed value.
        """
        if names is None:
            return self
        names = tuple(names)
        if len(set(names)) != len(names) or set(names) - self.keys():
            raise ValueError("Select distinct existing columns")
        return type(self)(
            {name: self[name] for name in names},
            schema={name: self.schema[name] for name in names},
            validity={name: self.validity[name] for name in names if name in self.validity},
            n_rows=self.n_rows,
            **self._context,
        )

    def to_columns(self) -> dict[str, np.ndarray]:
        """To columns.

        Returns:
            The computed value.
        """
        return {key: value.copy() for key, value in self.items()}

    def to_records(self) -> np.ndarray:
        """To records.

        Returns:
            The computed value.
        """
        result = np.empty(self.n_rows, dtype=[(key, value.dtype) for key, value in self.items()])
        for key, value in self.items():
            result[key] = value
        return result

    def __deepcopy__(self, memo: Any) -> Any:
        return self


class ObservationBatch(ColumnBatch):
    """Channel measurements with independent row and column validity."""
