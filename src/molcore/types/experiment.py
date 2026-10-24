"""The storage-independent public experiment facade."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from ._shared import ImagePolicy, ProcessingMetadata, Segmentation
from .acquisition import Acquisition, AcquisitionSequence
from .tables import LocalizationTable, ObservationTable, TrackTable


class Experiment:
    """Own acquisitions, active results, calibrations and provenance.

    Args:
        metadata: Optional JSON-compatible experiment metadata.

    Attributes:
        acquisitions: Acquisitions keyed by stable identifier.
        sequences: Explicit continuous sequences keyed by identifier.
        localizations: Active indexed localization table.
        tracks: Editable localization-to-track memberships.
        observations: Channel measurements and their validity flags.
        segmentations: Static segmentation datasets keyed by identifier.
        processing: Analysis settings and installed software versions.
    """

    def __init__(self, *, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = dict(metadata or {})
        self.acquisitions: dict[str, Acquisition] = {}
        self.sequences: dict[str, AcquisitionSequence] = {}
        self.segmentations: dict[str, Segmentation] = {}
        self.localizations = LocalizationTable(self)
        self.tracks = TrackTable(self)
        self.observations = ObservationTable(self)
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

    def add_acquisition(self, acquisition: Acquisition) -> Acquisition:
        """Register an acquisition without replacing an existing identifier.

        Args:
            acquisition: Acquisition to add.

        Returns:
            The registered acquisition.
        """
        if acquisition.id in self.acquisitions:
            raise ValueError("Acquisition ID already exists")
        self.acquisitions[acquisition.id] = acquisition
        return acquisition

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
        self.sequences[sequence.id] = sequence
        return sequence

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
        self.segmentations[segmentation.id] = segmentation
        self.observations._invalidate_partition(segmentation.acquisition_id)

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

    def _changed(self, ids):
        ids = np.asarray(ids, dtype=np.uint64)
        tracks = [
            track for track, members in self.tracks._members.items() if np.intersect1d(members, ids).size
        ]
        self.observations._invalidate(ids=ids, tracks=tracks)
        self.crops.invalidate(ids)

    def save(
        self,
        path: str | Path,
        *,
        layout: str = "hdf5",
        image_policy: str | ImagePolicy = "none",
        crop_width: int = 7,
    ) -> None:
        """Write a versioned, atomic experiment snapshot.

        Args:
            path: Destination file or directory.
            layout: Either hdf5 or directory.
            image_policy: none, localization_crops, track_crops, both, or full.
            crop_width: Odd pixel width used when saving crops.
        """
        from ..storage import save_experiment

        save_experiment(self, path, layout=layout, image_policy=image_policy, crop_width=crop_width)

    @classmethod
    def open(cls, path: str | Path) -> Experiment:
        """Restore eager result tables and lazy image/crop descriptors.

        Args:
            path: Saved HDF5 file or dataset directory.

        Returns:
            An Experiment usable as a context manager.
        """
        from ..storage import open_experiment

        return open_experiment(path)

    def close(self) -> None:
        """Release image-source resources and memory caches."""
        for acquisition in self.acquisitions.values():
            acquisition.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
