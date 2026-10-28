"""The storage-independent public dataset facade."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from .._locking import model_mutated
from .._logging import logged
from .._model_api import DatasetServices
from ._shared import ImagePolicy, ProcessingMetadata, Segmentation
from .acquisition import Acquisition, AcquisitionSequence
from .photons import PhotonSequenceStore
from .tables import LocalizationTable, ObservationTable, TrackTable

if TYPE_CHECKING:
    from types import TracebackType

    from numpy.typing import ArrayLike

    from .scientific import CancelEvent, ProgressCallback
    from .storage import SaveOptions


class Dataset(DatasetServices):
    """Own acquisitions, active results, calibrations and provenance.

    Args:
        id: Stable probing-round identifier; generated when omitted.
        name: Human-readable name; defaults to the identifier.
        metadata: Optional JSON-compatible dataset metadata.

    Attributes:
        acquisitions: Acquisitions keyed by stable identifier.
        sequences: Explicit continuous sequences keyed by identifier.
        localizations: Active indexed localization table.
        tracks: Editable localization-to-track memberships.
        observations: Channel measurements and their validity flags.
        photon_sequences: Independent frame-resolved photon/intensity sequences.
        segmentations: Static segmentation datasets keyed by identifier.
        processing: Analysis settings and installed software versions.
    """

    def __init__(
        self, *, id: str | None = None, name: str | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        """Initialise the object and validate its configuration.

        Args:
            id: Stable probing-round identifier; generated when omitted.
            name: Human-readable name; defaults to the identifier.
            metadata: Additional structured provenance and acquisition metadata.
        """
        self._lock = RLock()
        self._lineage = str(uuid4())
        self.revision = 0
        self._metadata_revision = 0
        self._partition_revisions = {}
        self._subscribers = {}
        self._mutation_depth = 0
        self._mutation_partitions = set()
        self.id = str(uuid4()) if id is None else id
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Dataset ID must be a nonempty string")
        self.name = self.id if name is None else name
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Dataset name must be a nonempty string")
        self.metadata = dict(metadata or {})
        self._acquisitions: dict[str, Acquisition] = {}
        self._sequences: dict[str, AcquisitionSequence] = {}
        self._segmentations: dict[str, Segmentation] = {}
        self.acquisitions = MappingProxyType(self._acquisitions)
        self.sequences = MappingProxyType(self._sequences)
        self.segmentations = MappingProxyType(self._segmentations)
        self.localizations = LocalizationTable(self)
        self.tracks = TrackTable(self)
        self.observations = ObservationTable(self)
        self.photon_sequences = PhotonSequenceStore(self)
        self.processing = ProcessingMetadata(software={"molcore": "0.1.0"})
        for package in ("numpy", "h5py", "tifffile", "picassosr", "torch", "trackpy", "cellpose"):
            try:
                self.processing.software[package] = version(package)
            except PackageNotFoundError:
                pass
        from ..crops import CropStore
        from ..transforms import TransformManager

        self.transforms = TransformManager(self)
        self.calibrations = self.transforms.calibrations
        self.crops = CropStore(self)

    @model_mutated
    @logged
    def add_acquisition(self, acquisition: Acquisition) -> Acquisition:
        """Register an acquisition without replacing an existing identifier.

        Args:
            acquisition: Acquisition to add.

        Returns:
            The registered acquisition.
        """
        if acquisition.id in self.acquisitions:
            raise ValueError("Acquisition ID already exists")
        self._acquisitions[acquisition.id] = acquisition
        return acquisition

    @logged
    def remove_acquisition(self, id: str, *, cascade: bool = False) -> None:
        """Remove an acquisition, explicitly cascading scientific dependents.

        Cascading removes affected sequences and whole spanning tracks, retaining
        other acquisitions' localizations and reusable named calibrations. Borrowed
        sources are not closed. Built-in sources reopen files per read; dropping
        the acquisition releases its source reference without disturbing other owners.

        Args:
            id: Optional stable identifier; None allocates a new identifier.
            cascade: Cascade used by this operation.
        """
        from ..staging import transact

        if id not in self.acquisitions:
            raise ValueError("Unknown acquisition")

        def remove(model: Dataset) -> None:
            """Remove.

            Args:
                model: Model or fitting method selected by the caller.
            """
            ids = model.localizations.query(acquisition_id=id, space="raw")["id"]
            sequences = {k for k, s in model.sequences.items() if id in s.offsets}
            tracks = [
                t
                for t, members in model.tracks._members.items()
                if np.intersect1d(ids, members).size or model.tracks.metadata[t]["sequence_id"] in sequences
            ]
            masks = [k for k, s in model.segmentations.items() if s.acquisition_id == id]
            observations = model.observations
            rows = observations._data
            dependent = (
                len(ids)
                or tracks
                or sequences
                or masks
                or any(s.acquisition_id == id for s in model.photon_sequences.query(include_stale=True))
                or np.any(rows.get("acquisition_id", np.array([])) == id)
                or id in model.transforms._drift
                or any(a == id for a, _ in model.transforms._selected)
                or any(
                    id in values for values in model.processing.settings.values() if isinstance(values, dict)
                )
                or any(
                    id in run.get("acquisitions", [])
                    for run in model.processing.settings.get("photometry", {}).values()
                )
            )
            if dependent and not cascade:
                raise ValueError("Acquisition has dependent state; pass cascade=True")
            for sequence in model.photon_sequences.query(acquisition_id=id, include_stale=True):
                model.photon_sequences.remove(sequence.id)
            if rows:
                keep = (
                    (rows["acquisition_id"] != id)
                    & ~np.isin(rows["localization_id"], ids)
                    & ~np.isin(rows["track_id"], tracks)
                )
                observations._data = {k: v[keep] for k, v in rows.items()}
                observations._indexes.clear()
                observations.revision += 1
            for track in tracks:
                model.tracks.delete(track)
            model.localizations.delete(ids)
            for key in sequences:
                del model._sequences[key]
            for key in masks:
                del model._segmentations[key]
            model.transforms._selected = {k: v for k, v in model.transforms._selected.items() if k[0] != id}
            model.transforms._drift.pop(id, None)
            model.transforms.revision += 1
            for stage, values in model.processing.settings.items():
                if not isinstance(values, dict):
                    continue
                if stage in ("photometry", "photon_sequences"):
                    for run_id, run in list(values.items()):
                        if id in run.get("acquisitions", []):
                            run["acquisitions"].remove(id)
                            if stage == "photon_sequences":
                                run["sequence_ids"] = [
                                    key
                                    for key in run.get("sequence_ids", [])
                                    if key in model.photon_sequences
                                ]
                            if not run["acquisitions"]:
                                del values[run_id]
                elif stage == "cellpose":
                    for mask_id in masks:
                        values.pop(mask_id, None)
                else:
                    values.pop(id, None)
            del model._acquisitions[id]

        transact(self, remove)

    @model_mutated
    @logged
    def add_sequence(
        self,
        acquisition_ids: Sequence[str],
        *,
        id: str | None = None,
        allow_cross_boundary_tracks: bool = False,
        reference: str | None = None,
    ) -> AcquisitionSequence:
        """Create an explicit sequence from registered acquisitions.

        Args:
            acquisition_ids: Ordered acquisition identifiers.
            id: Optional stable sequence identifier.
            allow_cross_boundary_tracks: Permit memberships across recordings.
            reference: Shared coordinate-reference identifier for crossing tracks.

        Returns:
            The registered AcquisitionSequence.
        """
        sequence = AcquisitionSequence(
            [self.acquisitions[a] for a in acquisition_ids],
            id=id,
            allow_cross_boundary_tracks=allow_cross_boundary_tracks,
            reference=reference,
        )
        if sequence.id in self.sequences:
            raise ValueError("Sequence ID already exists")
        self._sequences[sequence.id] = sequence
        return sequence

    @model_mutated
    @logged
    def add_segmentation(self, segmentation: Segmentation) -> None:
        """Add or replace a static segmentation and invalidate its dependents.

        Args:
            segmentation: Segmentation with acquisition and coordinate bindings.
        """
        acquisition = self.acquisitions.get(segmentation.acquisition_id)
        if acquisition is None or segmentation.channel not in acquisition.channels:
            raise ValueError("Segmentation references an unknown acquisition/channel")
        view = acquisition.channels[segmentation.channel]
        view.channel_index(segmentation.frame)
        if segmentation.labels.shape != view.shape[1:]:
            raise ValueError("Segmentation geometry must match its bound channel")
        previous = self._segmentations.get(segmentation.id)
        if previous is not None:
            self.observations._invalidate_partition(previous.acquisition_id)
        self._segmentations[segmentation.id] = segmentation
        self.observations._invalidate_partition(segmentation.acquisition_id)

    @logged
    def segmentation_labels(self, segmentation_id: str, rows: dict[str, np.ndarray]) -> np.ndarray:
        """Find object labels at effective localization coordinates.

        Args:
            segmentation_id: Registered segmentation identifier.
            rows: Localization columns in effective raw channel coordinates.

        Returns:
            Labels for each row, with zero outside the image or acquisition.
        """
        segmentation = self.segmentations[segmentation_id]
        labels = np.zeros(len(rows["id"]), dtype=segmentation.labels.dtype)
        if not len(labels):
            return labels
        x, y = self.transforms.coordinates(rows, space=segmentation.space)
        # Convert reference-space queries into the bound segmentation channel.
        if segmentation.space == "raw":
            x, y = self.transforms.into_channel(rows, segmentation.channel)
        xi, yi = np.floor(x + 0.5).astype(int), np.floor(y + 0.5).astype(int)
        inside = (
            (rows["acquisition_id"] == segmentation.acquisition_id)
            & (xi >= 0)
            & (yi >= 0)
            & (xi < segmentation.labels.shape[1])
            & (yi < segmentation.labels.shape[0])
        )
        labels[inside] = segmentation.labels[yi[inside], xi[inside]]
        return labels

    def _changed(self, ids: ArrayLike | None) -> None:
        """Changed.

        Args:
            ids: Stable entity IDs; None selects the available partition.
        """
        ids = np.asarray(ids, dtype=np.uint64)
        self.photon_sequences._localizations_changed(
            self.localizations.query(ids=ids, space="raw")["acquisition_id"]
        )
        tracks = [
            track for track, members in self.tracks._members.items() if np.intersect1d(members, ids).size
        ]
        self.observations._invalidate(ids=ids, tracks=tracks)
        self.crops.invalidate(ids)

    @logged
    def save(
        self,
        path: str | Path,
        *,
        layout: str = "hdf5",
        image_policy: str | ImagePolicy = "none",
        crop_width: int = 7,
        options: SaveOptions | None = None,
        progress: ProgressCallback | None = None,
        cancel_event: CancelEvent | None = None,
        publish_guard: Any = None,
    ) -> None:
        """Write a versioned, atomic dataset snapshot.

        Args:
            path: Destination file or directory.
            layout: Either hdf5 or directory.
            image_policy: ImagePolicy selecting no pixels, full movies, or any combination
                of localisation, track and photon-sequence crops. both retains its
                original localisation-plus-track meaning; all_crops includes all three.
            crop_width: Odd pixel width used when saving crops.
            options: Composable SaveOptions; replaces the legacy image policy.
            progress: Optional callback receiving structured save progress.
            cancel_event: Optional event requesting cancellation before atomic replacement.
        """
        from ..storage import save_dataset

        save_dataset(
            self,
            path,
            layout=layout,
            image_policy=image_policy,
            crop_width=crop_width,
            options=options,
            progress=progress,
            cancel_event=cancel_event,
            publish_guard=publish_guard,
        )

    @classmethod
    @logged
    def open(
        cls,
        path: str | Path,
        *,
        progress: ProgressCallback | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> Dataset:
        """Restore eager result tables and lazy image/crop descriptors.

        Args:
            path: Saved HDF5 file or dataset directory.

        Returns:
            An Dataset usable as a context manager.
        """
        from ..storage import open_dataset

        return open_dataset(path, progress=progress, cancel_event=cancel_event)

    @logged
    def close(self) -> None:
        """Release image-source resources and memory caches."""
        for acquisition in self.acquisitions.values():
            acquisition.close()

    def __enter__(self) -> Dataset:
        """Enter the managed resource lifetime.

        Returns:
            Computed Dataset value.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release managed resources on exit.

        Args:
            exc_type: Exc type used by this operation.
            exc_value: Exc value used by this operation.
            traceback: Traceback used by this operation.
        """
        self.close()
