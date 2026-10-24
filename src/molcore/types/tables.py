"""Indexed NumPy tables and validated localization/track editing."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def identifiers(values, *, nullable=False):
    """Normalize identifiers without signed/unsigned floating-point promotion."""
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


def columns(values):
    """Broadcast scalar fields into detached, one-dimensional columns."""
    arrays = {k: np.atleast_1d(v) for k, v in values.items()}
    lengths = [len(v) for v in arrays.values() if len(v) != 1]
    n = max(lengths) if lengths else (1 if arrays else 0)
    result = {}
    for name, value in arrays.items():
        if value.ndim != 1 or len(value) not in (1, n):
            raise ValueError("All table columns must be 1D with compatible lengths")
        if value.dtype.kind == "O":
            if not all(isinstance(v, str) for v in value):
                raise ValueError("Object-valued columns are not supported; use typed columns")
            value = value.astype(str)
        result[name] = np.repeat(value, n) if len(value) == 1 and n != 1 else value.copy()
    return result, n


def integers(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError(f"{name} must contain nonnegative integers")
    return values


class ColumnTable:
    """Base for result tables with detached query output and cached indexes."""

    def __init__(self, experiment):
        self.experiment = experiment
        self._data, self._indexes = {}, {}
        self._next_id = 1
        self.revision = 0

    def __len__(self):
        return len(self._data.get("id", ()))

    def _index(self, name):
        if name not in self._indexes:
            values = self._data[name]
            order = np.argsort(values, kind="stable")
            self._indexes[name] = (values[order], order)
        return self._indexes[name]

    def _where(self, _candidate=None, **selectors):
        candidates = _candidate
        # Start with stable IDs where possible, then filter the small candidate set.
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

    def _range(self, frame_range, sequence=None):
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
                self._indexes[key] = frames[order], selected[order]
            frames, order = self._indexes[key]
        lo, hi = np.searchsorted(frames, [start, stop])
        return order[lo:hi]

    def _append(self, incoming):
        n, old_n = len(incoming["id"]), len(self)
        combined = {}
        for name in self._data.keys() | incoming.keys():
            old, new = self._data.get(name), incoming.get(name)
            template = old if new is None else new

            def missing(count, name=name, template=template):
                if name in ("localization_id", "track_id"):
                    return np.zeros(count, dtype=np.uint64)
                if template.dtype.kind in "US":
                    return np.full(count, "", dtype=template.dtype)
                if template.dtype.kind == "b":
                    return np.ones(count, dtype=bool)
                return np.full(count, np.nan)

            if old is None:
                old = new[:0] if old_n == 0 else missing(old_n)
            if new is None:
                new = old[:0] if n == 0 else missing(n)
            combined[name] = np.concatenate([old, new])
        self._data = combined
        self._indexes.clear()
        self.revision += 1

    def _ids(self, incoming, n):
        ids = incoming.get("id", np.arange(self._next_id, self._next_id + n, dtype=np.uint64))
        ids = identifiers(ids)
        existing = self._data.get("id", np.array([], dtype=np.uint64))
        if len(np.unique(ids)) != n or np.any(ids < self._next_id) or np.intersect1d(ids, existing).size:
            raise ValueError("IDs must be unique, monotonically allocated and never reused")
        return ids

    def _frames(self, incoming):
        integers(incoming["frame"], "Camera frames")
        for acquisition_id, channel in set(zip(incoming["acquisition_id"], incoming["channel"], strict=True)):
            acquisition = self.experiment.acquisitions.get(str(acquisition_id))
            if acquisition is None or channel not in acquisition.channels:
                raise ValueError("Unknown acquisition or channel")
            selected = (incoming["acquisition_id"] == acquisition_id) & (incoming["channel"] == channel)
            frames = incoming["frame"][selected]
            if not np.isin(frames, acquisition.channels[channel].source_frames).all():
                raise ValueError("Frame does not measure the specified channel")


class LocalizationTable(ColumnTable):
    """The active localizations with immutable originals and editable positions."""

    def add(self, **values: Any) -> np.ndarray:
        """Validate and append localizations atomically.

        Args:
            **values: Scalar or 1D acquisition_id, channel, frame, x, y columns,
                optional id values, and additional typed measurements.

        Returns:
            Allocated localization IDs as a NumPy array.

        Raises:
            ValueError: Required fields, references, coordinates or IDs are invalid.
        """
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
        incoming["edit_x"], incoming["edit_y"] = np.full(n, np.nan), np.full(n, np.nan)
        self._append(incoming)
        self._next_id = max(self._next_id, int(incoming["id"].max()) + 1) if n else self._next_id
        return incoming["id"].copy()

    def query(
        self,
        *,
        acquisition_id=None,
        channel=None,
        frame_range=None,
        ids=None,
        sequence_id=None,
        space="raw",
        original=False,
        segmentation_id=None,
        label=None,
    ) -> dict[str, np.ndarray]:
        """Return detached columns matching acquisition, frame, ID or mask filters.

        Args:
            acquisition_id: Acquisition identifier or identifiers.
            channel: Logical channel name or names.
            frame_range: Half-open camera-frame range, sequence-relative if selected.
            ids: Localization identifiers.
            sequence_id: Explicit acquisition sequence defining the timeline.
            space: One of raw, registered or corrected.
            original: Use imported positions instead of manual position overrides.
            segmentation_id: Optional segmentation for label filtering.
            label: Label or labels to retain; defaults to all foreground labels.

        Returns:
            Dictionary of detached NumPy columns, including effective x and y.
        """
        from ._shared import CoordinateSpace

        space = CoordinateSpace(space)
        if sequence_id is not None and acquisition_id is not None:
            raise ValueError("Select either acquisition_id or sequence_id")
        sequence = self.experiment.sequences[sequence_id] if sequence_id is not None else None
        selected = self._where(
            _candidate=self._range(frame_range, sequence),
            acquisition_id=list(sequence.offsets) if sequence else acquisition_id,
            channel=channel,
            id=ids,
        )
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
            return empty
        result = {k: v[selected].copy() for k, v in self._data.items() if not k.startswith("edit_")}
        if not original:
            for name in ("x", "y"):
                edits = self._data[f"edit_{name}"][selected]
                result[name] = np.where(np.isnan(edits), result[name], edits)
        if sequence:
            result["sequence_frame"] = result["frame"] + np.array(
                [sequence.offsets[a] for a in result["acquisition_id"]], dtype=int
            )
        if segmentation_id is not None:
            labels = self.experiment.segmentation_labels(segmentation_id, result)
            keep = labels > 0 if label is None else np.isin(labels, np.atleast_1d(label))
            result = {k: v[keep] for k, v in result.items()}
        if space != "raw":
            result["x"], result["y"] = self.experiment.transforms.coordinates(result, space=space)
        return result

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
        self._data["edit_x"][selected], self._data["edit_y"][selected] = x, y
        self.revision += 1
        self.experiment._changed([localization_id])

    def delete(self, ids: ArrayLike) -> None:
        """Delete localizations and remove their incident track memberships.

        Args:
            ids: Existing localization identifiers to delete atomically.
        """
        ids = identifiers(ids)
        selected = self._where(id=ids)
        if len(selected) != len(np.unique(ids)):
            raise ValueError("Unknown localization ID")
        self.experiment._changed(ids)
        keep = np.ones(len(self), dtype=bool)
        keep[selected] = False
        self._data = {k: v[keep] for k, v in self._data.items()}
        self._indexes.clear()
        self.revision += 1
        self.experiment.tracks._remove_localizations(ids)

    def replace(self, acquisition_id: str, channel: str, **values: Any) -> np.ndarray:
        """Replace one active partition, preserving every other partition.

        Args:
            acquisition_id: Acquisition partition to replace.
            channel: Logical channel partition to replace.
            **values: New frame, x, y and optional measurement columns.

        Returns:
            New localization IDs.
        """
        staged = LocalizationTable(self.experiment)
        staged._next_id = self._next_id
        ids = staged.add(acquisition_id=acquisition_id, channel=channel, **values)
        if np.intersect1d(ids, self._data.get("id", np.array([], dtype=np.uint64))).size:
            raise ValueError("Replacement IDs must not already exist")
        old = self.query(acquisition_id=acquisition_id, channel=channel)["id"]
        keep = ~np.isin(self._data.get("id", np.array([], dtype=np.uint64)), old)
        merged = LocalizationTable(self.experiment)
        merged._data = {k: v[keep] for k, v in self._data.items()}
        # Dtype compatibility is checked before deleting results or dependents.
        merged._append(staged._data)
        self.delete(old)
        self.experiment.observations._invalidate_partition(acquisition_id, channel)
        self._data = merged._data
        self._indexes.clear()
        self.revision += 1
        self._next_id = staged._next_id
        return ids


class ObservationTable(ColumnTable):
    """Channel measurements anchored to molecules and optional results."""

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
        known = self.experiment.localizations._data.get("id", np.array([], dtype=np.uint64))
        if not np.isin(incoming["localization_id"][incoming["localization_id"] != 0], known).all():
            raise ValueError("Unknown localization anchor")
        known_tracks = np.asarray(list(self.experiment.tracks._members), dtype=np.uint64)
        if not np.isin(incoming["track_id"][incoming["track_id"] != 0], known_tracks).all():
            raise ValueError("Unknown track anchor")
        incoming["id"] = self._ids(incoming, n)
        incoming["valid"] = np.ones(n, dtype=bool)
        self._append(incoming)
        self._next_id = max(self._next_id, int(incoming["id"].max()) + 1) if n else self._next_id
        return incoming["id"].copy()

    def query(
        self,
        *,
        molecule_id=None,
        acquisition_id=None,
        channel=None,
        frame_range=None,
        ids=None,
        include_stale=False,
    ):
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
            return {
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
        if not include_stale:
            selected = selected[self._data["valid"][selected]]
        return {k: v[selected].copy() for k, v in self._data.items()}

    def _invalidate(self, ids=(), tracks=()):
        if self._data:
            mask = np.isin(self._data["localization_id"], identifiers(ids)) | np.isin(
                self._data["track_id"], identifiers(tracks)
            )
            self._data["valid"][mask] = False
            self.revision += 1

    def _invalidate_partition(self, acquisition_id, channel=None):
        if self._data:
            mask = self._data["acquisition_id"] == acquisition_id
            if channel is not None:
                mask &= self._data["channel"] == channel
            self._data["valid"][mask] = False
            self.revision += 1


class TrackTable:
    """Track membership without duplicated coordinates."""

    def __init__(self, experiment):
        self.experiment, self._members, self.metadata = experiment, {}, {}
        self._next_id, self.revision = 1, 0

    def __len__(self):
        return len(self._members)

    def _validate(self, ids, *, sequence_id=None, segmentation_id=None):
        ids = identifiers(ids)
        if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("A track requires distinct localization IDs")
        rows = self.experiment.localizations.query(ids=ids)
        if len(rows["id"]) != len(ids) or len(np.unique(rows["channel"])) != 1:
            raise ValueError("Track localizations must exist and share a channel")
        acquisitions = np.unique(rows["acquisition_id"])
        timeline = rows["frame"]
        if sequence_id is not None:
            sequence = self.experiment.sequences[sequence_id]
            if not set(acquisitions) <= sequence.offsets.keys():
                raise ValueError("Track lies outside its sequence")
            timeline = rows["frame"] + np.array([sequence.offsets[a] for a in rows["acquisition_id"]])
        if len(acquisitions) > 1:
            if sequence_id is None or not sequence.allow_cross_boundary_tracks:
                raise ValueError("Cross-boundary tracks require an explicitly enabled sequence")
        if len(np.unique(timeline)) != len(timeline):
            raise ValueError("Track has multiple localizations at the same camera frame")
        if segmentation_id is not None:
            labels = self.experiment.segmentation_labels(segmentation_id, rows)
            if len(np.unique(labels)) != 1 or labels[0] == 0:
                raise ValueError("Track must remain inside one segmentation label")
        return rows["id"][np.argsort(timeline, kind="stable")].astype(np.uint64)

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
        track = self._next_id
        self._next_id += 1
        self._members[track] = ids
        self.metadata[track] = {
            "sequence_id": sequence_id,
            "segmentation_id": segmentation_id,
            "parameters": dict(metadata or {}),
        }
        self.revision += 1
        return track

    def query(
        self, *, track_id=None, frame_range=None, space="raw", sequence_id=None
    ) -> dict[str, np.ndarray]:
        """Join memberships to current localization coordinates on demand.

        Args:
            track_id: Track ID or IDs; omitted selects all tracks.
            frame_range: Half-open camera-frame range.
            space: Requested coordinate space.
            sequence_id: Optional sequence defining frame offsets.

        Returns:
            Localization columns plus track_id, sorted by track and frame.
        """
        tracks = list(self._members) if track_id is None else np.atleast_1d(track_id)
        chunks = []
        for track in tracks:
            if int(track) not in self._members:
                raise ValueError("Unknown track ID")
            rows = self.experiment.localizations.query(
                ids=self._members[int(track)], frame_range=frame_range, space=space, sequence_id=sequence_id
            )
            rows["track_id"] = np.full(len(rows["id"]), track, dtype=np.uint64)
            order = np.argsort(rows.get("sequence_frame", rows["frame"]), kind="stable")
            chunks.append({k: v[order] for k, v in rows.items()})
        if not chunks:
            rows = self.experiment.localizations.query(ids=[])
            rows["track_id"] = np.array([], dtype=np.uint64)
            return rows
        return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    def delete(self, track_id: int) -> None:
        """Delete a track without deleting its localizations.

        Args:
            track_id: Existing track identifier.
        """
        track_id = int(track_id)
        if track_id not in self._members:
            raise ValueError("Unknown track ID")
        self.experiment.observations._invalidate(tracks=[track_id])
        del self._members[track_id], self.metadata[track_id]
        self.revision += 1

    def split(self, track_id: int, *, frame: int) -> int:
        """Split at a camera frame, using sequence frames for crossing tracks.

        Args:
            track_id: Existing track identifier.
            frame: First frame of the new track.

        Returns:
            New track identifier; the original retains the earlier members.
        """
        meta = self.metadata[track_id]
        rows = self.query(track_id=track_id, sequence_id=meta["sequence_id"])
        timeline = rows.get("sequence_frame", rows["frame"])
        earlier, later = rows["id"][timeline < frame], rows["id"][timeline >= frame]
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
        self.experiment.observations._invalidate(tracks=[track_id])
        return new

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
        meta = self.metadata[tracks[0]]
        ids = np.concatenate([self._members[t] for t in tracks])
        validated = self._validate(
            ids, sequence_id=meta["sequence_id"], segmentation_id=meta["segmentation_id"]
        )
        for track in tracks[1:]:
            self.delete(track)
        self._members[tracks[0]] = validated
        self.experiment.observations._invalidate(tracks=tracks)
        self.revision += 1
        return tracks[0]

    def reassign(self, localization_ids: ArrayLike, track_id: int) -> None:
        """Move memberships atomically to an existing compatible track.

        Args:
            localization_ids: Existing localization identifiers to move.
            track_id: Destination track identifier.
        """
        ids = identifiers(localization_ids)
        meta = self.metadata[track_id]
        combined = np.union1d(self._members[track_id], ids)
        validated = self._validate(
            combined, sequence_id=meta["sequence_id"], segmentation_id=meta["segmentation_id"]
        )
        self._remove_localizations(ids)
        self._members[track_id], self.metadata[track_id] = validated, meta
        self.experiment.observations._invalidate(tracks=[track_id])
        self.revision += 1

    def _remove_localizations(self, ids):
        for track in list(self._members):
            members = self._members[track]
            keep = ~np.isin(members, ids)
            if not keep.all():
                self.experiment.observations._invalidate(tracks=[track])
                self._members[track] = members[keep]
                if not keep.any():
                    del self._members[track], self.metadata[track]
                self.revision += 1
