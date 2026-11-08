"""Indexed NumPy tables and validated localization/track editing."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .._changes import mutation_scope
from .._locking import model_mutated
from .._logging import logged
from ..immutable import freeze, freeze_metadata, immutable_array
from ._shared import CoordinateSpace
from .batches import ColumnSchema, LocalizationBatch, ObservationBatch, TrackMembershipBatch

if TYPE_CHECKING:
    from ..types import AcquisitionSequence, Dataset


@logged
def identifiers(values: Any, *, nullable: bool = False) -> np.ndarray:
    """Normalize identifiers without signed/unsigned floating-point promotion.

    Args:
        values: Input values to validate, convert or display.
        nullable: Nullable used by this operation.

    Returns:
        Computed NumPy array value.
    """
    array = np.atleast_1d(np.asarray(values, dtype=object))
    if array.ndim != 1:
        raise ValueError("Identifiers must be one-dimensional")
    result = []
    for value in array:
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ValueError("Identifiers must be integers")
        value = int(value)
        if nullable and value == -1:
            value = 0
        if value < (0 if nullable else 1) or value > np.iinfo(np.uint64).max:
            raise ValueError("Identifiers must fit positive uint64; zero denotes no anchor")
        result.append(value)
    return np.array(result, dtype=np.uint64)


@logged
def columns(values: Any) -> dict[str, np.ndarray]:
    """Broadcast scalar fields into detached, one-dimensional columns.

    Args:
        values: Input values to validate, convert or display.

    Returns:
        Detached named result columns.
    """
    arrays = {k: np.atleast_1d(v) for k, v in values.items()}
    lengths = [len(v) for v in arrays.values() if len(v) != 1]
    n = max(lengths) if lengths else 1 if arrays else 0
    result = {}
    for name, value in arrays.items():
        if value.ndim != 1 or len(value) not in (1, n):
            raise ValueError("All table columns must be 1D with compatible lengths")
        if value.dtype.kind == "O":
            if not all(isinstance(v, str) for v in value):
                raise ValueError("Object-valued columns are not supported; use typed columns")
            value = value.astype(str)
        result[name] = np.repeat(value, n) if len(value) == 1 and n != 1 else value.copy()
    return (result, n)


@logged
def integers(values: Any, name: str) -> np.ndarray:
    """Integers.

    Args:
        values: Input values to validate, convert or display.
        name: Identifier or field name used to address this value.

    Returns:
        Computed NumPy array value.
    """
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError(f"{name} must contain nonnegative integers")
    return values


def partition_keys(rows: Any, dataset: Dataset) -> set[tuple[str, str]]:
    """Find represented image partitions without constructing per-row Python strings.

    Args:
        rows: Named localization columns containing acquisition_id and channel.
        dataset: Model declaring the available acquisition/channel partitions.

    Returns:
        The acquisition/channel pairs actually represented by rows.
    """
    acquisitions, channels = rows["acquisition_id"], rows["channel"]
    if not len(acquisitions):
        return set()
    candidates = [(a, c) for a, value in dataset.acquisitions.items() for c in value.channels]
    if len(candidates) <= 16:
        represented, covered = set(), np.zeros(len(acquisitions), dtype=bool)
        for acquisition, channel in candidates:
            selected = (acquisitions == acquisition) & (channels == channel)
            if selected.any():
                represented.add((acquisition, channel))
                covered |= selected
        if covered.all():
            return represented
    # Many small partitions or invalid external rows retain the general path;
    # unknown keys must reach the caller's usual validation rather than disappear.
    return {(str(a), str(c)) for a, c in zip(acquisitions, channels, strict=True)}


class ColumnTable:
    """Base for result tables with detached query output and cached indexes."""

    def __init__(self, dataset: Dataset) -> None:
        """Initialise the object and validate its configuration.

        Args:
            dataset: Scientific model supplying acquisitions and result tables.
        """
        self.dataset = dataset
        self._data, self._indexes = ({}, {})
        self._validity, self._schema = ({}, {})
        self._next_id = 1
        self.revision = 0

    def __len__(self) -> int:
        """Return the number of available items.

        Returns:
            Computed int value.
        """
        return len(self._data.get("id", ()))

    def _index(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Index.

        Args:
            name: Identifier or field name used to address this value.

        Returns:
            Computed dict[Any, NumPy array] value.
        """
        if name not in self._indexes:
            values = self._data[name]
            order = np.argsort(values, kind="stable")
            self._indexes[name] = (values[order], order)
        return self._indexes[name]

    def _where(self, _candidate: Any = None, **selectors: Any) -> np.ndarray:
        """Where.

        Args:
            _candidate:  candidate used by this operation.
            **selectors: Additional named options for this operation.

        Returns:
            Computed NumPy array value.
        """
        candidates = _candidate
        for name, values in sorted(selectors.items(), key=lambda item: item[0] != "id"):
            if values is None:
                continue
            if name not in self._data:
                return np.array([], dtype=int)
            values = identifiers(values) if name == "id" else np.atleast_1d(values)
            if candidates is not None:
                candidates = candidates[np.isin(self._data[name][candidates], values)]
                continue
            sorted_values, order = self._index(name)
            ranges = []
            for value in values:
                lo = np.searchsorted(sorted_values, value, side="left")
                hi = np.searchsorted(sorted_values, value, side="right")
                ranges.append(order[lo:hi])
            selected = np.concatenate(ranges) if ranges else np.array([], dtype=int)
            candidates = np.unique(selected)
        return np.arange(len(self)) if candidates is None else np.sort(candidates)

    def _range(
        self, frame_range: tuple[int, int] | None, sequence: AcquisitionSequence | None = None
    ) -> np.ndarray | None:
        """Range.

        Args:
            frame_range: Optional half-open camera-frame interval.
            sequence: Sequence used by this operation.

        Returns:
            Computed NumPy array value.
        """
        if frame_range is None:
            return None
        start, stop = frame_range
        if stop < start:
            raise ValueError("Frame range must be increasing")
        if not self._data:
            return np.array([], dtype=int)
        if sequence is None:
            frames, order = self._index("frame")
        else:
            key = ("sequence", sequence.id)
            if key not in self._indexes:
                selected = self._where(acquisition_id=list(sequence.offsets))
                frames = self._data["frame"][selected].astype(np.int64, copy=True)
                for acquisition_id, offset in sequence.offsets.items():
                    frames[self._data["acquisition_id"][selected] == acquisition_id] += offset
                order = np.argsort(frames, kind="stable")
                self._indexes[key] = (frames[order], selected[order])
            frames, order = self._indexes[key]
        lo, hi = np.searchsorted(frames, [start, stop])
        return order[lo:hi]

    def _append(self, incoming: dict[str, np.ndarray]) -> Any:
        """Append.

        Args:
            incoming: Validated named columns to append to the table.

        Returns:
            Result retaining the input format or backend-specific representation.
        """
        n, old_n = (len(incoming["id"]), len(self))
        combined, validity = ({}, {})
        for name in self._data.keys() | incoming.keys():
            old, new = (self._data.get(name), incoming.get(name))
            template = old if new is None else new
            old_valid = (
                self._validity.get(name, np.ones(old_n, dtype=bool))
                if old is not None
                else np.zeros(old_n, dtype=bool)
            )
            new_valid = np.ones(n, dtype=bool) if new is not None else np.zeros(n, dtype=bool)

            def missing(count: int, template: np.ndarray = template) -> np.ndarray:
                """Missing.

                Args:
                    count: Count.


                Returns:
                    The computed value.
                """
                if template.dtype.kind in "fc":
                    return np.full(count, np.nan, dtype=template.dtype)
                return np.zeros(count, dtype=template.dtype)

            if old is None:
                old = missing(old_n)
            if new is None:
                new = missing(n)
            dtype = np.result_type(old.dtype, new.dtype)
            if old.dtype.kind in "iu" and new.dtype.kind in "iu" and (dtype.kind == "f"):
                if all(not len(value) or value.max() <= np.iinfo(np.int64).max for value in (old, new)):
                    dtype = np.dtype("int64")
                elif all(not len(value) or value.min() >= 0 for value in (old, new)):
                    dtype = np.dtype("uint64")
                else:
                    raise ValueError("Integer columns require a lossless common dtype")
                old, new = (old.astype(dtype), new.astype(dtype))
            combined[name] = immutable_array(np.concatenate([old, new]))
            if not old_valid.all() or not new_valid.all():
                validity[name] = immutable_array(np.concatenate([old_valid, new_valid]))
        self._data, self._validity = (combined, validity)
        self._schema = {
            name: ColumnSchema(combined[name].dtype, spec.unit, name in validity)
            for name, spec in self._schema.items()
            if name in combined
        }
        self._indexes.clear()
        self.revision += 1

    @model_mutated(scope=lambda self, name, *, unit=None: mutation_scope(self))
    def set_column_schema(self, name: str, *, unit: str | None = None) -> None:
        """Declare the physical unit of an existing column.

        Args:
            name: Name.
            unit: Unit.
        """
        if name not in self._data:
            raise ValueError("Unknown column")
        if unit is not None and (not isinstance(unit, str)):
            raise TypeError("Unit must be a string or None")
        self._schema = {
            **self._schema,
            name: ColumnSchema(self._data[name].dtype, unit, name in self._validity),
        }
        self.revision += 1

    def iter_batches(
        self, *, batch_size: int = 65536, columns: Any = None, **selectors: Any
    ) -> Iterator[LocalizationBatch | ObservationBatch]:
        """Query captured state with bounded output allocations.

        Args:
            batch_size: Batch size.
            columns: Columns.
            selectors: Selectors.

        Yields:
            Immutable projected batches from captured table state.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        table = getattr(
            self.dataset.snapshot()._model,
            "localizations" if isinstance(self, LocalizationTable) else "observations",
        )
        ids = table.query(columns=("id",), **selectors)["id"]
        selectors.pop("ids", None)
        for start in range(0, len(ids), batch_size):
            yield table.query(ids=ids[start : start + batch_size], columns=columns, **selectors)

    def add_batches(self, batches: Any) -> Any:
        """Combine input chunks once and publish atomically.

        Args:
            batches: Batches.


        Returns:
            The computed value.
        """
        from ..staging import transact

        chunks = list(batches)
        if not chunks:
            return np.array([], dtype=np.uint64)
        required = (
            {"acquisition_id", "channel", "frame", "x", "y"}
            if isinstance(self, LocalizationTable)
            else {"acquisition_id", "channel", "frame", "intensity", "molecule_id"}
        )
        if any(required - chunk.keys() for chunk in chunks):
            raise ValueError("Every batch must contain required columns")
        names = set().union(*(chunk.keys() for chunk in chunks))
        if "id" in names and any("id" not in chunk for chunk in chunks):
            raise ValueError("Supply explicit IDs in every batch, or none")
        counts = [
            chunk.n_rows
            if hasattr(chunk, "n_rows")
            else max((len(np.atleast_1d(value)) for value in chunk.values()), default=0)
            for chunk in chunks
        ]
        units = {}
        for chunk in chunks:
            for name, spec in getattr(chunk, "schema", {}).items():
                previous = units.get(name)
                if previous is not None and spec.unit is not None and (previous != spec.unit):
                    raise ValueError("Batch column units do not match")
                if spec.unit is not None:
                    units[name] = spec.unit
        combined, masks = ({}, {})
        for name in names:
            present = [np.atleast_1d(chunk[name]) for chunk in chunks if name in chunk]
            dtype = np.result_type(*(part.dtype for part in present))
            if all(part.dtype.kind in "iu" for part in present) and dtype.kind == "f":
                if all(not part.size or part.max() <= np.iinfo(np.int64).max for part in present):
                    dtype = np.dtype("int64")
                elif all(not part.size or part.min() >= 0 for part in present):
                    dtype = np.dtype("uint64")
                else:
                    raise ValueError("Integer columns require a lossless common dtype")
            values, valid = ([], [])
            for chunk, count in zip(chunks, counts, strict=True):
                array = np.atleast_1d(chunk[name]) if name in chunk else np.zeros(count, dtype=dtype)
                if len(array) == 1 and count != 1:
                    array = np.repeat(array, count)
                if array.ndim != 1 or len(array) != count:
                    raise ValueError("Batch column lengths do not match")
                values.append(array.astype(dtype, copy=False))
                valid.append(
                    np.asarray(
                        getattr(chunk, "validity", {}).get(name, np.full(count, name in chunk)), dtype=bool
                    )
                )
            combined[name] = np.concatenate(values)
            masks[name] = np.concatenate(valid)
            if name in required and (not masks[name].all()):
                raise ValueError("Required columns cannot contain null values")

        def commit(model: Any) -> Any:
            """Commit.

            Args:
                model: Model.


            Returns:
                The computed value.
            """
            table = model.localizations if isinstance(self, LocalizationTable) else model.observations
            for name, unit in units.items():
                if name in table._schema and table._schema[name].unit not in (None, unit):
                    raise ValueError("Column unit conflicts with the existing table")
            ids = table.add(**combined)
            offset = len(table) - len(ids)
            for name, mask in masks.items():
                old = table._validity.get(name, np.ones(len(table), dtype=bool)).copy()
                old[offset:] &= mask
                if not old.all():
                    table._validity[name] = immutable_array(old)
            for name, unit in units.items():
                table._schema[name] = ColumnSchema(table._data[name].dtype, unit, name in table._validity)
            return ids

        return transact(self.dataset, commit)

    def _ids(self, incoming: dict[str, np.ndarray], n: Any) -> np.ndarray:
        """Ids.

        Args:
            incoming: Validated named columns to append to the table.
            n: Number of rows being allocated or processed.

        Returns:
            Computed NumPy array value.
        """
        ids = incoming.get("id", np.arange(self._next_id, self._next_id + n, dtype=np.uint64))
        ids = identifiers(ids)
        existing = self._data.get("id", np.array([], dtype=np.uint64))
        if len(np.unique(ids)) != n or np.any(ids < self._next_id) or np.intersect1d(ids, existing).size:
            raise ValueError("IDs must be unique, monotonically allocated and never reused")
        return ids

    def _frames(self, incoming: dict[str, np.ndarray]) -> None:
        """Frames.

        Args:
            incoming: Validated named columns to append to the table.
        """
        integers(incoming["frame"], "Camera frames")
        for acquisition_id, channel in set(zip(incoming["acquisition_id"], incoming["channel"], strict=True)):
            acquisition = self.dataset.acquisitions.get(str(acquisition_id))
            if acquisition is None or channel not in acquisition.channels:
                raise ValueError("Unknown acquisition or channel")
            selected = (incoming["acquisition_id"] == acquisition_id) & (incoming["channel"] == channel)
            frames = incoming["frame"][selected]
            if not np.isin(frames, acquisition.channels[channel].source_frames).all():
                raise ValueError("Frame does not measure the specified channel")


class LocalizationTable(ColumnTable):
    """The active localizations with immutable originals and editable positions."""

    def _edit_rows(self, ids: ArrayLike | None) -> tuple[np.ndarray, np.ndarray]:
        """Edit rows.

        Args:
            ids: Stable entity IDs; None selects the available partition.

        Returns:
            Computed NumPy array value.
        """
        ids = identifiers(ids)
        lookup = {int(value): index for index, value in enumerate(self._data.get("id", []))}
        if len(np.unique(ids)) != len(ids) or any(int(i) not in lookup for i in ids):
            raise ValueError("Expected distinct existing localization IDs")
        return (ids, np.array([lookup[int(i)] for i in ids], dtype=int))

    @model_mutated(scope=lambda self, ids, **values: mutation_scope(self, ids=ids))
    @logged
    def update_measurements(self, ids: ArrayLike, **values: ArrayLike) -> None:
        """Atomically update numeric measurement columns in supplied ID order.

        Scalars broadcast. New columns are float64 with NaN in unassigned rows.
        Structural fields and manual coordinate overrides cannot be measurements.
        Dependent observations are invalidated; positions and IDs are preserved.

        Args:
            ids: Stable entity IDs; None selects the available partition.
            **values: Additional named options for this operation.
        """
        ids, selected = self._edit_rows(ids)
        reserved = {
            "id",
            "x",
            "y",
            "frame",
            "acquisition_id",
            "channel",
            "track_id",
            "localization_id",
            "sequence_frame",
            "molecule_id",
            "timestamp",
            "cycle_index",
        }
        prepared, updated_validity = ({}, {})
        for name, value in values.items():
            if not isinstance(name, str) or not name or name in reserved or name.startswith(("edit_", "_")):
                raise ValueError("Structural columns cannot be updated as measurements")
            array = np.asarray(value)
            if array.dtype.kind not in "biuf" or array.ndim > 1:
                raise ValueError("Measurements must be real numeric scalars or 1D arrays")
            if array.ndim and len(array) != len(ids):
                raise ValueError("Measurement lengths must match IDs")
            old = self._data.get(name)
            if old is not None and old.dtype.kind not in "biuf":
                raise ValueError("Existing column is not numeric")
            dtype = np.result_type(old.dtype, array.dtype) if old is not None else array.dtype
            if (
                old is not None
                and old.dtype.kind in "iu"
                and (array.dtype.kind in "iu")
                and (dtype.kind == "f")
            ):
                if all(not value.size or value.max() <= np.iinfo(np.int64).max for value in (old, array)):
                    dtype = np.dtype("int64")
                elif all(not value.size or value.min() >= 0 for value in (old, array)):
                    dtype = np.dtype("uint64")
                else:
                    raise ValueError("Integer measurements require a lossless common dtype")
            column = (
                old.astype(dtype, copy=True)
                if old is not None
                else np.full(len(self), np.nan, dtype=dtype)
                if dtype.kind in "fc"
                else np.zeros(len(self), dtype=dtype)
            )
            valid = (
                self._validity.get(name, np.ones(len(self), dtype=bool)).copy()
                if old is not None
                else np.zeros(len(self), dtype=bool)
            )
            valid[selected] = True
            updated_validity[name] = valid
            column[selected] = array
            prepared[name] = column
        if not len(ids) or not prepared:
            return
        self._data.update(prepared)
        for name, valid in updated_validity.items():
            if valid.all():
                self._validity.pop(name, None)
            else:
                self._validity[name] = immutable_array(valid)
            if name in self._schema:
                old_spec = self._schema[name]
                self._schema[name] = ColumnSchema(
                    self._data[name].dtype, old_spec.unit, old_spec.nullable or name in self._validity
                )
        self._indexes.clear()
        self.revision += 1
        tracks = [
            t for t, members in self.dataset.tracks._members.items() if np.intersect1d(ids, members).size
        ]
        self.dataset.observations._invalidate(ids=ids, tracks=tracks)

    @model_mutated(scope=lambda self, ids, *, x, y: mutation_scope(self, ids=ids))
    @logged
    def move_many(self, ids: ArrayLike, *, x: ArrayLike, y: ArrayLike) -> None:
        """Atomically set effective raw positions in ID order, preserving originals.

        Args:
            ids: Stable entity IDs; None selects the available partition.
            x: X used by this operation.
            y: Y used by this operation.
        """
        ids, selected = self._edit_rows(ids)
        prepared = []
        for value in (x, y):
            array = np.asarray(value, dtype=float)
            if array.ndim > 1 or (array.ndim and len(array) != len(ids)) or (not np.isfinite(array).all()):
                raise ValueError("Supply finite scalar positions or one position per ID")
            prepared.append(array)
        if not len(ids):
            return
        for name, array in zip(("edit_x", "edit_y"), prepared, strict=True):
            column = self._data[name].copy()
            column[selected] = array
            self._data[name] = column
        self.revision += 1
        self.dataset._changed(ids)

    @model_mutated(
        scope=lambda self, *, acquisition_id, channel, frame, x, y, id=None, **measurements: mutation_scope(
            self, acquisition=acquisition_id, channel=channel
        )
    )
    @logged
    def add(
        self,
        *,
        acquisition_id: ArrayLike,
        channel: ArrayLike,
        frame: ArrayLike,
        x: ArrayLike,
        y: ArrayLike,
        id: ArrayLike | None = None,
        **measurements: Any,
    ) -> np.ndarray:
        """Validate and append localizations atomically.

        Args:
            acquisition_id: Scalar or per-row acquisition identifiers.
            channel: Scalar or per-row logical channel names.
            frame: Original integer camera frame indices.
            x: Raw-channel horizontal coordinates in pixels.
            y: Raw-channel vertical coordinates in pixels.
            id: Optional explicit stable IDs; omitted values are allocated.
            **measurements: Additional scalar or per-row measurement columns.

        Returns:
            Allocated localization IDs as a NumPy array.

        Raises:
            ValueError: Required fields, references, coordinates or IDs are invalid.
        """
        values = dict(measurements, acquisition_id=acquisition_id, channel=channel, frame=frame, x=x, y=y)
        if id is not None:
            values["id"] = id
        required = {"acquisition_id", "channel", "frame", "x", "y"}
        if not required <= values.keys() or {"edit_x", "edit_y"} & values.keys():
            raise ValueError("Supply acquisition_id, channel, frame, x and y; edit fields are internal")
        if "id" in values:
            values["id"] = identifiers(values["id"])
        incoming, n = columns(values)
        self._frames(incoming)
        for name in ("x", "y"):
            incoming[name] = incoming[name].astype(float)
            if not np.isfinite(incoming[name]).all():
                raise ValueError("Coordinates must be finite")
        incoming["id"] = self._ids(incoming, n)
        incoming["edit_x"], incoming["edit_y"] = (np.full(n, np.nan), np.full(n, np.nan))
        self._append(incoming)
        self._next_id = max(self._next_id, int(incoming["id"].max()) + 1) if n else self._next_id
        if self is self.dataset.localizations:
            self.dataset.photon_sequences._localizations_changed(incoming["acquisition_id"])
        return incoming["id"].copy()

    @logged
    def query(
        self,
        *,
        acquisition_id: str | None = None,
        channel: str | None = None,
        frame_range: tuple[int, int] | None = None,
        ids: ArrayLike | None = None,
        columns: tuple[str, ...] | None = None,
        sequence_id: str | None = None,
        space: CoordinateSpace | str = "corrected",
        original: bool = False,
        segmentation_id: str | None = None,
        label: str | None = None,
    ) -> LocalizationBatch:
        """Return immutable columns matching acquisition, frame, ID or mask filters.

        Args:
            acquisition_id: Acquisition identifier or identifiers.
            channel: Logical channel name or names.
            frame_range: Half-open camera-frame range, sequence-relative if selected.
            ids: Localization identifiers.
            sequence_id: Explicit acquisition sequence defining the timeline.
            space: Corrected by default, applying selected registration and drift;
                raw and registered views are available explicitly.
            original: Use imported positions instead of manual position overrides.
            segmentation_id: Optional segmentation for label filtering.
            label: Label or labels to retain; defaults to all foreground labels.

        Returns:
            Immutable localization batch with detached columns and query context.
        """
        space = CoordinateSpace(space)
        if sequence_id is not None and acquisition_id is not None:
            raise ValueError("Select either acquisition_id or sequence_id")
        sequence = self.dataset.sequences[sequence_id] if sequence_id is not None else None
        selected = self._where(
            _candidate=self._range(frame_range, sequence),
            acquisition_id=list(sequence.offsets) if sequence else acquisition_id,
            channel=channel,
            id=ids,
        )

        def batch(result: dict[str, np.ndarray]) -> LocalizationBatch:
            """Attach the query's coordinate and frame context.

            Args:
                result: Selected detached columns.

            Returns:
                Immutable scientific query result.
            """
            partitions = partition_keys(result, self.dataset)
            value = LocalizationBatch(
                result,
                dataset_id=self.dataset.id,
                space=space,
                original=original,
                sequence_id=sequence_id,
                reference=sequence.reference if sequence else None,
                reference_context={
                    (str(a), str(c)): self.dataset.transforms._selected.get((a, c))
                    if space != CoordinateSpace.RAW
                    else None
                    for a, c in partitions
                },
                transform_revision=self.dataset.transforms.revision,
                schema={k: self._schema[k] for k in result if k in self._schema},
                validity={k: v[selected] for k, v in self._validity.items() if k in result},
            )
            return value.project(columns)

        if not self._data:
            empty = {
                name: np.array([], dtype=dtype)
                for name, dtype in (
                    ("id", "uint64"),
                    ("acquisition_id", "U1"),
                    ("channel", "U1"),
                    ("frame", "int64"),
                    ("x", "float64"),
                    ("y", "float64"),
                )
            }
            if sequence is not None:
                empty["sequence_frame"] = np.array([], dtype=np.int64)
            return batch(empty)
        coordinate_columns = (
            {"x", "y"}
            if columns is None or {"x", "y"} & set(columns) or segmentation_id is not None
            else set()
        )
        result = {
            k: v[selected]
            for k, v in self._data.items()
            if not k.startswith("edit_")
            and (
                columns is None
                or k in columns
                or k in {"id", "acquisition_id", "channel", "frame"} | coordinate_columns
            )
        }
        if not original:
            for name in coordinate_columns:
                edits = self._data[f"edit_{name}"][selected]
                result[name] = np.where(np.isnan(edits), result[name], edits)
        if sequence:
            result["sequence_frame"] = result["frame"] + np.array(
                [sequence.offsets[a] for a in result["acquisition_id"]], dtype=int
            )
        if segmentation_id is not None:
            labels = self.dataset.segmentation_labels(segmentation_id, result)
            keep = labels > 0 if label is None else np.isin(labels, np.atleast_1d(label))
            selected = selected[keep]
            result = {k: v[keep] for k, v in result.items()}
        if space != "raw" and coordinate_columns:
            result["x"], result["y"] = self.dataset.transforms.coordinates(result, space=space)
        return batch(result)

    @model_mutated(scope=lambda self, localization_id, *, x, y: mutation_scope(self, ids=[localization_id]))
    @logged
    def move(self, localization_id: int | np.integer, *, x: float, y: float) -> None:
        """Set a manual position without modifying imported coordinates.

        Args:
            localization_id: Existing localization ID.
            x: New horizontal channel-pixel coordinate.
            y: New vertical channel-pixel coordinate.
        """
        selected = self._where(id=[localization_id])
        if len(selected) != 1 or not np.isfinite([x, y]).all():
            raise ValueError("Expected an existing localization and finite coordinates")
        for name, value in (("edit_x", x), ("edit_y", y)):
            column = self._data[name].copy()
            column[selected] = value
            self._data[name] = column
        self.revision += 1
        self.dataset._changed([localization_id])

    @model_mutated(scope=lambda self, ids: mutation_scope(self, ids=ids))
    @logged
    def delete(self, ids: ArrayLike) -> None:
        """Delete localizations and remove their incident track memberships.

        Args:
            ids: Existing localization identifiers to delete atomically.
        """
        ids = identifiers(ids)
        selected = self._where(id=ids)
        if len(selected) != len(np.unique(ids)):
            raise ValueError("Unknown localization ID")
        self.dataset._changed(ids)
        keep = np.ones(len(self), dtype=bool)
        keep[selected] = False
        self._data = {k: v[keep] for k, v in self._data.items()}
        self._validity = {k: v[keep] for k, v in self._validity.items()}
        self._indexes.clear()
        self.revision += 1
        self.dataset.tracks._remove_localizations(ids)

    @model_mutated(
        scope=lambda self, acquisition_id, channel, **values: mutation_scope(
            self, acquisition=acquisition_id, channel=channel
        )
    )
    @logged
    def replace(self, acquisition_id: str, channel: str, **values: Any) -> np.ndarray:
        """Replace one active partition, preserving every other partition.

        Args:
            acquisition_id: Acquisition partition to replace.
            channel: Logical channel partition to replace.
            **values: New frame, x, y and optional measurement columns.

        Returns:
            New localization IDs.
        """
        staged = LocalizationTable(self.dataset)
        staged._next_id = self._next_id
        ids = staged.add(acquisition_id=acquisition_id, channel=channel, **values)
        if np.intersect1d(ids, self._data.get("id", np.array([], dtype=np.uint64))).size:
            raise ValueError("Replacement IDs must not already exist")
        old = self.query(acquisition_id=acquisition_id, channel=channel, space="raw")["id"]
        keep = ~np.isin(self._data.get("id", np.array([], dtype=np.uint64)), old)
        merged = LocalizationTable(self.dataset)
        merged._data = {k: v[keep] for k, v in self._data.items()}
        merged._validity = {k: v[keep] for k, v in self._validity.items()}
        merged._append(staged._data)
        self.delete(old)
        self.dataset.observations._invalidate_partition(acquisition_id, channel)
        self._data = merged._data
        self._validity = merged._validity
        self._schema = {
            name: ColumnSchema(self._data[name].dtype, spec.unit, name in self._validity)
            for name, spec in self._schema.items()
            if name in self._data
        }
        self._indexes.clear()
        self.revision += 1
        self._next_id = staged._next_id
        if self is self.dataset.localizations:
            self.dataset.photon_sequences._localizations_changed([acquisition_id])
        return ids


class ObservationTable(ColumnTable):
    """Channel measurements anchored to molecules and optional results."""

    @model_mutated(scope=lambda self, **values: mutation_scope(self))
    @logged
    def add(self, **values: Any) -> np.ndarray:
        """Append channel measurements with source timing and optional anchors.

        Args:
            **values: molecule_id, acquisition_id, channel, frame, intensity,
                optional localization_id/track_id and typed measurement columns.

        Returns:
            New observation identifiers.
        """
        if not {"molecule_id", "acquisition_id", "channel", "frame", "intensity"} <= values.keys():
            raise ValueError("Missing observation fields")
        for name in ("id", "localization_id", "track_id"):
            if name in values:
                values[name] = identifiers(values[name], nullable=name != "id")
        incoming, n = columns(values)
        self._frames(incoming)
        for name in ("localization_id", "track_id"):
            incoming.setdefault(name, np.zeros(n, dtype=np.uint64))
        known = self.dataset.localizations._data.get("id", np.array([], dtype=np.uint64))
        if not np.isin(incoming["localization_id"][incoming["localization_id"] != 0], known).all():
            raise ValueError("Unknown localization anchor")
        known_tracks = np.asarray(list(self.dataset.tracks._members), dtype=np.uint64)
        if not np.isin(incoming["track_id"][incoming["track_id"] != 0], known_tracks).all():
            raise ValueError("Unknown track anchor")
        incoming["id"] = self._ids(incoming, n)
        incoming["valid"] = np.ones(n, dtype=bool)
        self._append(incoming)
        self._next_id = max(self._next_id, int(incoming["id"].max()) + 1) if n else self._next_id
        return incoming["id"].copy()

    @logged
    def query(
        self,
        *,
        molecule_id: Any = None,
        acquisition_id: str | None = None,
        channel: str | None = None,
        frame_range: tuple[int, int] | None = None,
        ids: ArrayLike | None = None,
        include_stale: bool = False,
        columns: tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return detached active measurements, optionally including stale rows.

        Args:
            molecule_id: Molecule identifiers to select.
            acquisition_id: Source acquisition identifiers.
            channel: Logical channel names.
            frame_range: Half-open original camera-frame range.
            ids: Observation identifiers.
            include_stale: Include rows invalidated by edits or corrections.

        Returns:
            Dictionary of NumPy columns, including a validity column.
        """
        selected = self._where(
            _candidate=self._range(frame_range),
            molecule_id=molecule_id,
            acquisition_id=acquisition_id,
            channel=channel,
            id=ids,
        )
        if not self._data:
            return ObservationBatch(
                {
                    name: np.array([], dtype=dtype)
                    for name, dtype in (
                        ("id", "uint64"),
                        ("molecule_id", "U1"),
                        ("acquisition_id", "U1"),
                        ("channel", "U1"),
                        ("frame", "int64"),
                        ("intensity", "float64"),
                        ("localization_id", "uint64"),
                        ("track_id", "uint64"),
                        ("valid", "bool"),
                    )
                }
            ).project(columns)
        if not include_stale:
            selected = selected[self._data["valid"][selected]]
        result = {k: v[selected] for k, v in self._data.items() if columns is None or k in columns}
        return ObservationBatch(
            result,
            n_rows=len(selected),
            schema={k: self._schema[k] for k in result if k in self._schema},
            validity={k: v[selected] for k, v in self._validity.items() if k in result},
        ).project(columns)

    def _invalidate(self, ids: ArrayLike | None = (), tracks: ArrayLike | None = ()) -> None:
        """Invalidate.

        Args:
            ids: Stable entity IDs; None selects the available partition.
            tracks: Track IDs to invalidate, or whether this operation selects whole tracks.
        """
        self.dataset.photon_sequences._invalidate(ids=ids, tracks=tracks)
        if self._data:
            mask = np.isin(self._data["localization_id"], identifiers(ids)) | np.isin(
                self._data["track_id"], identifiers(tracks)
            )
            column = self._data["valid"].copy()
            column[mask] = False
            self._data["valid"] = column
            self.revision += 1

    def _invalidate_partition(self, acquisition_id: str, channel: str | None = None) -> None:
        """Invalidate partition.

        Args:
            acquisition_id: Source acquisition identifier.
            channel: Logical source channel name.
        """
        if self._data:
            mask = self._data["acquisition_id"] == acquisition_id
            if channel is not None:
                mask &= self._data["channel"] == channel
            column = self._data["valid"].copy()
            column[mask] = False
            self._data["valid"] = column
            self.revision += 1


class TrackTable:
    """Track membership without duplicated coordinates."""

    def __init__(self, dataset: Dataset) -> None:
        """Initialise the object and validate its configuration.

        Args:
            dataset: Scientific model supplying acquisitions and result tables.
        """
        self.dataset, self._members, self._metadata = (dataset, {}, {})
        self._next_id, self.revision = (1, 0)

    @property
    def metadata(self) -> Any:
        """Metadata.

        Returns:
            The computed value.
        """
        from ..immutable import freeze

        return freeze(self._metadata)

    def __len__(self) -> int:
        """Return the number of available items.

        Returns:
            Computed int value.
        """
        return len(self._members)

    def _validate(
        self, ids: ArrayLike | None, *, sequence_id: str | None = None, segmentation_id: str | None = None
    ) -> np.ndarray:
        """Validate.

        Args:
            ids: Stable entity IDs; None selects the available partition.
            sequence_id: Optional sequence identifier defining acquisition order and frame offsets.
            segmentation_id: Optional segmentation identifier restricting the selected objects.

        Returns:
            Computed NumPy array value.
        """
        ids = identifiers(ids)
        if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("A track requires distinct localization IDs")
        rows = self.dataset.localizations.query(ids=ids, space="raw")
        if len(rows["id"]) != len(ids) or len(np.unique(rows["channel"])) != 1:
            raise ValueError("Track localizations must exist and share a channel")
        acquisitions = np.unique(rows["acquisition_id"])
        timeline = rows["frame"]
        if sequence_id is not None:
            sequence = self.dataset.sequences[sequence_id]
            if not set(acquisitions) <= sequence.offsets.keys():
                raise ValueError("Track lies outside its sequence")
            timeline = rows["frame"] + np.array([sequence.offsets[a] for a in rows["acquisition_id"]])
        if len(acquisitions) > 1:
            if sequence_id is None or not sequence.allow_cross_boundary_tracks:
                raise ValueError("Cross-boundary tracks require an explicitly enabled sequence")
        if len(np.unique(timeline)) != len(timeline):
            raise ValueError("Track has multiple localizations at the same camera frame")
        if segmentation_id is not None:
            labels = self.dataset.segmentation_labels(segmentation_id, rows)
            if len(np.unique(labels)) != 1 or labels[0] == 0:
                raise ValueError("Track must remain inside one segmentation label")
        return rows["id"][np.argsort(timeline, kind="stable")].astype(np.uint64)

    @model_mutated(
        scope=lambda self, localization_ids, *, sequence_id=None, segmentation_id=None, metadata=None: (
            mutation_scope(self, ids=localization_ids)
        )
    )
    @logged
    def add(
        self,
        localization_ids: ArrayLike,
        *,
        sequence_id: str | None = None,
        segmentation_id: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Create a validated track from currently unassigned localizations.

        Args:
            localization_ids: Ordered or unordered localization IDs.
            sequence_id: Required for tracks crossing recording boundaries.
            segmentation_id: Optional mask constraining the track to one object.
            metadata: Tracking parameters and algorithm provenance.

        Returns:
            Stable integer track ID.
        """
        ids = self._validate(localization_ids, sequence_id=sequence_id, segmentation_id=segmentation_id)
        if any(np.intersect1d(ids, members).size for members in self._members.values()):
            raise ValueError("Localization already belongs to a track")
        parameters = freeze_metadata(metadata or {})
        track = self._next_id
        self._next_id += 1
        self._members[track] = ids
        self._metadata[track] = freeze(
            {"sequence_id": sequence_id, "segmentation_id": segmentation_id, "parameters": parameters}
        )
        self.revision += 1
        return track

    @logged
    def query(
        self,
        *,
        track_id: Any = None,
        frame_range: Any = None,
        space: CoordinateSpace | str = CoordinateSpace.CORRECTED,
        sequence_id: str | None = None,
        acquisition_id: str = None,
        channel: str = None,
        columns: Any = None,
        ids: Any = None,
    ) -> Any:
        """Join stable memberships to current positions, ordered by track and frame.

        Args:
            track_id: Track id.
            frame_range: Frame range.
            space: Space.
            sequence_id: Sequence id.
            acquisition_id: Acquisition id.
            channel: Channel.
            columns: Columns.
            ids: Ids.


        Returns:
            The computed value.
        """
        space = CoordinateSpace(space)
        tracks = list(self._members) if track_id is None else identifiers(track_id)
        local_columns = None if columns is None else tuple(set(columns) - {"track_id"} | {"id", "frame"})
        chunks, masks, context = ([], [], None)
        reference_context = {}
        partition_ids = None
        if sequence_id is not None and acquisition_id is not None:
            partition_ids = self.dataset.localizations.query(
                acquisition_id=acquisition_id, channel=channel, space="raw", columns=("id",)
            )["id"]
        for track in tracks:
            if int(track) not in self._members:
                raise ValueError("Unknown track ID")
            members = self._members[int(track)]
            if partition_ids is not None:
                members = members[np.isin(members, partition_ids)]
            if ids is not None:
                members = members[np.isin(members, ids)]
            batch = self.dataset.localizations.query(
                ids=members,
                frame_range=frame_range,
                space=space,
                sequence_id=sequence_id,
                acquisition_id=None if sequence_id is not None else acquisition_id,
                channel=channel,
                columns=local_columns,
            )
            order = np.argsort(batch.get("sequence_frame", batch["frame"]), kind="stable")
            values = {key: value[order] for key, value in batch.items()}
            values["track_id"] = np.full(batch.n_rows, track, dtype=np.uint64)
            chunks.append(values)
            masks.append({key: value[order] for key, value in batch.validity.items()})
            context = batch
            reference_context.update(batch.reference_context)
        if not chunks:
            context = self.dataset.localizations.query(
                ids=[], space=space, sequence_id=sequence_id, columns=local_columns
            )
            result = dict(context)
            result["track_id"] = np.array([], dtype=np.uint64)
        else:
            result = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
        validity = {
            key: np.concatenate(
                [
                    mask.get(key, np.ones(len(chunk["id"]), dtype=bool))
                    for mask, chunk in zip(masks, chunks, strict=True)
                ]
            )
            for key in set().union(*(mask.keys() for mask in masks))
        }
        return TrackMembershipBatch(
            result,
            dataset_id=self.dataset.id,
            space=space,
            sequence_id=sequence_id,
            reference=context.reference,
            reference_context=reference_context,
            transform_revision=self.dataset.transforms.revision,
            schema={key: value for key, value in context.schema.items() if key in result},
            validity=validity,
        ).project(columns)

    def iter_batches(
        self, *, batch_size: int = 65536, columns: Any = None, **selectors: Any
    ) -> Iterator[TrackMembershipBatch]:
        """Iterate captured memberships with bounded per-result row counts.

        Args:
            batch_size: Batch size.
            columns: Columns.
            selectors: Selectors.

        Yields:
            Immutable projected track-membership batches.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        table = self.dataset.snapshot()._model.tracks
        tracks = (
            list(table._members) if selectors.get("track_id") is None else identifiers(selectors["track_id"])
        )
        selectors.pop("track_id", None)
        selected_ids = selectors.pop("ids", None)
        for track in tracks:
            ids = table.query(track_id=track, ids=selected_ids, columns=("id",), **selectors)["id"]
            for start in range(0, len(ids), batch_size):
                yield table.query(
                    track_id=track, ids=ids[start : start + batch_size], columns=columns, **selectors
                )

    @model_mutated(
        scope=lambda self, acquisition_id, channel, memberships, *, ids=None, segmentation_id=None, metadata=None: (
            mutation_scope(self, acquisition=acquisition_id, channel=channel)
        )
    )
    @logged
    def replace(
        self,
        acquisition_id: str,
        channel: str,
        memberships: list[ArrayLike],
        *,
        ids: ArrayLike | None = None,
        segmentation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Replace wholly contained partition tracks atomically, retaining localization IDs.

        Memberships is an iterable of nonempty localization-ID arrays. Spanning
        tracks intersecting the scope are rejected, including empty replacements.
        Explicit IDs restrict replacement to that exact scope; an empty selection
        replaces nothing. Unknown, duplicate and out-of-partition IDs are rejected.

        Args:
            acquisition_id: Source acquisition identifier.
            channel: Logical source channel name.
            memberships: Memberships used by this operation.
            ids: Stable localization IDs defining the scope; None selects the partition.
            segmentation_id: Optional segmentation identifier restricting the selected objects.
            metadata: Additional structured provenance and acquisition metadata.

        Returns:
            Computed NumPy array value.
        """
        from copy import deepcopy

        if channel not in self.dataset.acquisitions[acquisition_id].channels:
            raise ValueError("Unknown channel")
        partition = self.dataset.localizations.query(
            acquisition_id=acquisition_id, channel=channel, space="raw"
        )["id"]
        if ids is not None:
            selected = identifiers(ids)
            if len(np.unique(selected)) != len(selected):
                raise ValueError("Selected localization IDs must be distinct")
            if not np.isin(selected, partition).all():
                raise ValueError("Selected localization IDs must belong to the partition")
            partition = selected
        old = []
        for track, members in self._members.items():
            inside = np.isin(members, partition)
            if inside.any():
                if not inside.all():
                    raise ValueError("Partition intersects a spanning track")
                old.append(track)
        staged = deepcopy(self, {id(self.dataset): self.dataset})
        for track in old:
            del staged._members[track], staged._metadata[track]
        result = []
        for members in memberships:
            members = identifiers(members)
            if not np.isin(members, partition).all():
                raise ValueError("Track members must belong to the selected partition")
            result.append(staged.add(members, segmentation_id=segmentation_id, metadata=metadata))
        self._members, self._metadata, self._next_id = (staged._members, staged._metadata, staged._next_id)
        self.revision += 1
        self.dataset.observations._invalidate(tracks=old)
        return np.asarray(result, dtype=np.uint64)

    @model_mutated(scope=lambda self, track_id: mutation_scope(self, tracks=[track_id]))
    @logged
    def delete(self, track_id: int) -> None:
        """Delete a track without deleting its localizations.

        Args:
            track_id: Existing track identifier.
        """
        track_id = int(track_id)
        if track_id not in self._members:
            raise ValueError("Unknown track ID")
        self.dataset.observations._invalidate(tracks=[track_id])
        del self._members[track_id], self._metadata[track_id]
        self.revision += 1

    @model_mutated(scope=lambda self, track_id, ids: mutation_scope(self, tracks=[track_id]))
    @logged
    def retain_members(self, track_id: int, ids: ArrayLike) -> None:
        """Retain a track subset in its existing order without removing localizations.

        A nonempty subset preserves the track ID and metadata. An empty subset
        deletes the track. Changed memberships invalidate dependent observations
        and photon sequences, including access to their historical crops. Raw
        localization crops remain valid; track crops resolve current membership.
        Keeping every member is a no-op, regardless of input order.

        Args:
            track_id: Existing track identifier.
            ids: Distinct existing members to retain; order does not reorder the track.

        Raises:
            ValueError: The track is unknown or selected IDs are duplicated,
                malformed, unknown, or not members of this track.
        """
        track_id = int(track_id)
        if track_id not in self._members:
            raise ValueError("Unknown track ID")
        selected = identifiers(ids)
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Retained localization IDs must be distinct")
        members = self._members[track_id]
        if not np.isin(selected, members).all():
            raise ValueError("Retained localization IDs must belong to the track")
        if not len(selected):
            self.delete(track_id)
            return
        if len(selected) == len(members):
            return
        self._members[track_id] = members[np.isin(members, selected)]
        self.dataset.observations._invalidate(tracks=[track_id])
        self.revision += 1

    @model_mutated(scope=lambda self, track_id, *, frame: mutation_scope(self, tracks=[track_id]))
    @logged
    def split(self, track_id: int, *, frame: int) -> int:
        """Split at a camera frame, using sequence frames for crossing tracks.

        Args:
            track_id: Existing track identifier.
            frame: First frame of the new track.

        Returns:
            New track identifier; the original retains the earlier members.
        """
        meta = self._metadata[track_id]
        rows = self.query(track_id=track_id, sequence_id=meta["sequence_id"], space="raw")
        timeline = rows.get("sequence_frame", rows["frame"])
        earlier, later = (rows["id"][timeline < frame], rows["id"][timeline >= frame])
        if not len(earlier) or not len(later):
            raise ValueError("Split must leave members on both sides")
        for members in (earlier, later):
            self._validate(members, sequence_id=meta["sequence_id"], segmentation_id=meta["segmentation_id"])
        self._members[track_id] = earlier
        new = self.add(
            later,
            sequence_id=meta["sequence_id"],
            segmentation_id=meta["segmentation_id"],
            metadata=meta["parameters"],
        )
        self.dataset.observations._invalidate(tracks=[track_id])
        return new

    @model_mutated(scope=lambda self, track_ids: mutation_scope(self, tracks=track_ids))
    @logged
    def merge(self, track_ids: ArrayLike) -> int:
        """Merge compatible tracks, keeping the first track's identifier.

        Args:
            track_ids: At least two distinct existing track identifiers.

        Returns:
            Identifier of the retained track.
        """
        tracks = [int(t) for t in track_ids]
        if len(set(tracks)) != len(tracks) or len(tracks) < 2:
            raise ValueError("Merge needs at least two distinct tracks")
        meta = self._metadata[tracks[0]]
        ids = np.concatenate([self._members[t] for t in tracks])
        validated = self._validate(
            ids, sequence_id=meta["sequence_id"], segmentation_id=meta["segmentation_id"]
        )
        for track in tracks[1:]:
            self.delete(track)
        self._members[tracks[0]] = validated
        self.dataset.observations._invalidate(tracks=tracks)
        self.revision += 1
        return tracks[0]

    @model_mutated(scope=lambda self, localization_ids, track_id: mutation_scope(self, ids=localization_ids))
    @logged
    def reassign(self, localization_ids: ArrayLike, track_id: int) -> None:
        """Move memberships atomically to an existing compatible track.

        Args:
            localization_ids: Existing localization identifiers to move.
            track_id: Destination track identifier.
        """
        ids = identifiers(localization_ids)
        meta = self._metadata[track_id]
        combined = np.union1d(self._members[track_id], ids)
        validated = self._validate(
            combined, sequence_id=meta["sequence_id"], segmentation_id=meta["segmentation_id"]
        )
        self._remove_localizations(ids)
        self._members[track_id], self._metadata[track_id] = (validated, meta)
        self.dataset.observations._invalidate(tracks=[track_id])
        self.revision += 1

    def _remove_localizations(self, ids: ArrayLike | None) -> None:
        """Remove localizations.

        Args:
            ids: Stable entity IDs; None selects the available partition.
        """
        for track in list(self._members):
            members = self._members[track]
            keep = ~np.isin(members, ids)
            if not keep.all():
                self.dataset.observations._invalidate(tracks=[track])
                self._members[track] = members[keep]
                if not keep.any():
                    del self._members[track], self._metadata[track]
                self.revision += 1
