"""Frame-resolved photon sequences independent of spatial track memberships."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from .._locking import model_locked
from .._logging import logged

if TYPE_CHECKING:
    from .experiment import Experiment


class PhotonUnit(StrEnum):
    """Units recorded explicitly; no camera calibration is inferred."""

    PHOTONS = "photons"
    IMAGE_INTENSITY = "image_intensity"


def _immutable(values: np.ndarray) -> np.ndarray:
    """Copy numeric values to immutable byte-backed storage.

    Args:
        values: Validated numeric sample array.

    Returns:
        Detached array whose writable flag cannot be re-enabled.
    """
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True)
class PhotonSequence:
    """One molecule's ordered measurements in one acquisition and logical channel.

    This is a frame-resolved sequence, rather than individual photon arrival events.
    Negative estimates are retained; NaN represents unavailable measurements.

    Attributes:
        acquisition_id: Original acquisition identifier.
        channel: Measured logical channel.
        frames: Strictly increasing original camera-frame indices; gaps are retained.
        photons: Photon estimates or raw intensity values according to unit.
        id: Stable sequence identifier, generated when omitted.
        timestamps: Recorded seconds, or NaN for unknown times.
        positions: Optional (n, 2) raw-channel (x, y) crop centres; paired NaNs mean unknown.
        valid: Per-sample measurement validity, independent of crop pixel validity.
        unit: Explicit physical photon or image-intensity units.
        method: Measurement method, such as imported, aperture or psf_fit.
        molecule_id: Optional identifier linking sequences across channels.
        localization_id: Optional stationary seed anchor.
        track_id: Optional moving trajectory anchor.
        depends_on_transforms: Whether later registration/drift edits invalidate these measurements.
        metadata: Detached JSON-compatible measurement provenance.
        run_id: Optional measurement-run identifier shared across molecules/channels.
        measurements: Additional numeric per-sample metrics, such as signal/background statistics.
        localization_ids: Optional per-sample localisation provenance; zero denotes no anchor.
        depends_on_localizations: Whether neighbour-mask input changes invalidate these samples.
    """

    acquisition_id: str
    channel: str
    frames: NDArray[np.int64]
    photons: NDArray[np.float64]
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamps: NDArray[np.float64] | None = None
    positions: NDArray[np.float64] | None = None
    valid: NDArray[np.bool_] | None = None
    unit: PhotonUnit = PhotonUnit.PHOTONS
    method: str = "imported"
    molecule_id: str | None = None
    localization_id: int | None = None
    track_id: int | None = None
    depends_on_transforms: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    measurements: dict[str, NDArray] = field(default_factory=dict)
    localization_ids: NDArray[np.uint64] | None = None
    depends_on_localizations: bool = False

    def __post_init__(self) -> None:
        """Validate samples and detach immutable numeric arrays from caller storage."""
        if not isinstance(self.depends_on_transforms, bool):
            raise TypeError("depends_on_transforms must be a boolean")
        if not isinstance(self.depends_on_localizations, bool):
            raise TypeError("depends_on_localizations must be a boolean")
        if not all(
            isinstance(v, str) and v for v in (self.id, self.acquisition_id, self.channel, self.method)
        ):
            raise ValueError("Sequence, acquisition, channel and method identifiers must be nonempty strings")
        frames = np.asarray(self.frames)
        if frames.ndim != 1 or frames.dtype.kind not in "iu" or np.any(frames < 0):
            raise ValueError("Frames must be a one-dimensional nonnegative integer array")
        if np.any(frames > np.iinfo(np.int64).max):
            raise ValueError("Frames exceed the supported int64 range")
        frames = frames.astype(np.int64)
        if np.any(np.diff(frames) <= 0):
            raise ValueError("Frames must be strictly increasing and unique")
        values = np.asarray(self.photons)
        if values.dtype.kind not in "iuf" or values.shape != frames.shape:
            raise ValueError("Photon values must be numeric and match the frame array")
        values = values.astype(np.float64)
        if np.isinf(values).any():
            raise ValueError("Photon values may contain NaN, but not infinity")
        times = (
            np.full(len(frames), np.nan)
            if self.timestamps is None
            else np.asarray(self.timestamps, dtype=float)
        )
        if (
            times.shape != frames.shape
            or np.isinf(times).any()
            or np.any(np.diff(times[np.isfinite(times)]) <= 0)
        ):
            raise ValueError("Timestamps must match frames and increase where known")
        valid = np.isfinite(values) if self.valid is None else np.asarray(self.valid)
        if valid.shape != frames.shape or valid.dtype.kind != "b" or np.any(valid & ~np.isfinite(values)):
            raise ValueError("Validity must be boolean and cannot mark NaN measurements valid")
        positions = None if self.positions is None else np.asarray(self.positions, dtype=float)
        if positions is not None and (
            positions.shape != (len(frames), 2)
            or np.isinf(positions).any()
            or np.any(np.isnan(positions).any(axis=1) != np.isnan(positions).all(axis=1))
        ):
            raise ValueError("Positions must contain one finite (x,y) pair or paired NaNs per sample")
        for anchor in (self.localization_id, self.track_id):
            if anchor is not None and (
                isinstance(anchor, bool) or not isinstance(anchor, (int, np.integer)) or anchor <= 0
            ):
                raise ValueError("Optional anchors must be positive integer IDs")
        if self.molecule_id is not None and (not isinstance(self.molecule_id, str) or not self.molecule_id):
            raise ValueError("Molecule ID must be a nonempty string when supplied")
        if self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id):
            raise ValueError("Run ID must be a nonempty string when supplied")
        reserved = {
            "id",
            "sequence_id",
            "sample_index",
            "acquisition_id",
            "channel",
            "frame",
            "frames",
            "intensity",
            "photons",
            "timestamp",
            "timestamps",
            "x",
            "y",
            "positions",
            "valid",
            "unit",
            "run_id",
            "molecule_id",
            "localization_id",
            "track_id",
        }
        measurements = {}
        for name, samples in self.measurements.items():
            samples = np.asarray(samples)
            if not isinstance(name, str) or not name or name in reserved:
                raise ValueError("Measurement names must be nonempty and cannot shadow sequence fields")
            if samples.shape != frames.shape or samples.dtype.kind not in "biuf":
                raise ValueError("Measurements must be numeric sample arrays matching frames")
            measurements[name] = _immutable(samples)
        anchors = (
            np.full(len(frames), self.localization_id or 0, dtype=np.uint64)
            if self.localization_ids is None
            else np.asarray(self.localization_ids)
        )
        if anchors.shape != frames.shape or anchors.dtype.kind not in "iu" or np.any(anchors < 0):
            raise ValueError("Sample localisation IDs must be nonnegative integers matching frames")
        object.__setattr__(self, "localization_ids", _immutable(anchors.astype(np.uint64)))
        object.__setattr__(self, "measurements", measurements)
        metadata = deepcopy(self.metadata)
        json.dumps(metadata, allow_nan=False)
        for name, array in (("frames", frames), ("photons", values), ("timestamps", times), ("valid", valid)):
            object.__setattr__(self, name, _immutable(array))
        object.__setattr__(self, "positions", None if positions is None else _immutable(positions))
        object.__setattr__(self, "unit", PhotonUnit(self.unit))
        object.__setattr__(self, "metadata", metadata)

    def __deepcopy__(self, memo: dict[int, Any]) -> PhotonSequence:
        """Keep sample arrays immutable when cloning model state.

        Args:
            memo: Object identities retained by the deepcopy operation.

        Returns:
            Independent validated record with detached metadata.
        """
        result = replace(self)
        memo[id(self)] = result
        return result


class PhotonSequenceStore:
    """Own independently identified photon sequences and derived-result validity."""

    def __init__(self, experiment: Experiment) -> None:
        """Bind a sequence collection to its owning scientific model.

        Args:
            experiment: Owner providing acquisition and optional anchor bindings.
        """
        self.experiment = experiment
        self._data: dict[str, PhotonSequence] = {}
        self._stale: set[str] = set()
        self.revision = 0

    def __len__(self) -> int:
        """Return the number of stored sequences, including stale results.

        Returns:
            Number of independently stored records.
        """
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterate stable sequence identifiers in insertion order.

        Returns:
            Iterator of sequence IDs.
        """
        return iter(self._data)

    @model_locked
    def __getitem__(self, id: str) -> PhotonSequence:
        """Read a detached record, including its stale provenance if applicable.

        Args:
            id: Existing photon-sequence identifier.

        Returns:
            Validated snapshot with immutable numeric samples.
        """
        return replace(self._data[id])

    @model_locked
    @logged
    def add(self, sequence: PhotonSequence) -> PhotonSequence:
        """Validate source frames and anchors before adding a detached record.

        Args:
            sequence: Frame-resolved photon or image-intensity sequence.

        Returns:
            Stored sequence with recorded acquisition timestamps populated.

        Raises:
            ValueError: IDs, timing, source bindings or optional anchors are invalid.
        """
        if not isinstance(sequence, PhotonSequence):
            raise TypeError("Expected PhotonSequence")
        if sequence.id in self._data:
            raise ValueError("Photon-sequence ID already exists")
        acquisition = self.experiment.acquisitions[sequence.acquisition_id]
        view = acquisition.channels[sequence.channel]
        if not np.isin(sequence.frames, view.source_frames).all():
            raise ValueError("Sequence frames must belong to the measured logical channel")
        if sequence.localization_id is not None:
            rows = self.experiment.localizations.query(ids=[sequence.localization_id])
            if len(rows["id"]) != 1 or rows["acquisition_id"][0] != acquisition.id:
                raise ValueError("Localization anchor must belong to the sequence acquisition")
        if sequence.track_id is not None:
            rows = self.experiment.tracks.query(track_id=sequence.track_id, acquisition_id=acquisition.id)
            if not len(rows["id"]):
                raise ValueError("Track anchor must include the sequence acquisition")
        sample_ids = np.unique(sequence.localization_ids[sequence.localization_ids != 0])
        if len(sample_ids):
            anchors = self.experiment.localizations.query(ids=sample_ids)
            if len(anchors["id"]) != len(sample_ids) or np.any(anchors["acquisition_id"] != acquisition.id):
                raise ValueError("Sample localisation anchors must belong to the sequence acquisition")
        times = sequence.timestamps
        if acquisition.timestamps is not None:
            recorded = acquisition.timestamps[sequence.frames]
            known = np.isfinite(times)
            if not np.allclose(times[known], recorded[known], rtol=1e-12, atol=1e-12):
                raise ValueError("Sequence timestamps disagree with recorded camera timestamps")
            times = recorded
        sequence = replace(sequence, timestamps=times)
        self._data[sequence.id] = sequence
        self.revision += 1
        return self[sequence.id]

    @model_locked
    @logged
    def query(
        self,
        *,
        acquisition_id: str | None = None,
        channel: str | None = None,
        molecule_id: str | None = None,
        include_stale: bool = False,
    ) -> tuple[PhotonSequence, ...]:
        """Select detached sequences without constructing spatial localizations.

        Args:
            acquisition_id: Optional source acquisition selector.
            channel: Optional logical measurement channel selector.
            molecule_id: Optional shared molecule identifier.
            include_stale: Include records invalidated by anchor or transform edits.

        Returns:
            Matching sequence records in insertion order.
        """
        return tuple(
            replace(value)
            for key, value in self._data.items()
            if (include_stale or key not in self._stale)
            and (acquisition_id is None or value.acquisition_id == acquisition_id)
            and (channel is None or value.channel == channel)
            and (molecule_id is None or value.molecule_id == molecule_id)
        )

    @model_locked
    def is_stale(self, id: str) -> bool:
        """Check whether a sequence's scientific inputs were invalidated.

        Args:
            id: Existing sequence identifier.

        Returns:
            True when recomputation is required for active analysis.
        """
        if id not in self._data:
            raise KeyError(id)
        return id in self._stale

    @model_locked
    @logged
    def remove(self, id: str) -> None:
        """Delete one sequence and its saved crop descriptors.

        Args:
            id: Existing photon-sequence identifier.
        """
        del self._data[id]
        self._stale.discard(id)
        self.experiment.crops._photon_saved = {
            k: v for k, v in self.experiment.crops._photon_saved.items() if k[0] != id
        }
        self.revision += 1

    def _invalidate(
        self,
        *,
        ids: Any = (),
        tracks: Any = (),
        acquisition_id: str | None = None,
        channel: str | None = None,
    ) -> None:
        """Mark source-dependent measurements stale without changing their samples.

        Args:
            ids: Changed localization anchor IDs.
            tracks: Changed spatial track IDs.
            acquisition_id: Optional acquisition whose transforms or results changed.
            channel: Optional affected measurement channel.
        """
        changed = {
            key
            for key, value in self._data.items()
            if value.localization_id in ids
            or np.isin(value.localization_ids[value.localization_ids != 0], ids).any()
            or value.track_id in tracks
            or (
                acquisition_id is not None
                and value.depends_on_transforms
                and value.acquisition_id == acquisition_id
                and (channel is None or value.channel == channel)
            )
        }
        if changed - self._stale:
            self._stale.update(changed)
            self.revision += 1

    def _localizations_changed(self, acquisitions: Iterable[str]) -> None:
        """Invalidate derived neighbour-mask measurements in affected acquisitions.

        Args:
            acquisitions: Acquisitions whose localisation positions or membership changed.
        """
        acquisitions = set(acquisitions)
        changed = {
            key
            for key, value in self._data.items()
            if value.depends_on_localizations and value.acquisition_id in acquisitions
        }
        if changed - self._stale:
            self._stale.update(changed)
            self.revision += 1
