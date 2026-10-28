"""Fiducial registration between probing rounds, without changing local data."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from ._processing import executor
from .files import report
from .storage import _json
from .types._shared import Calibration, StaleResultError
from .types.dataset_alignment import DatasetAlignmentResult, DatasetAlignmentSettings
from .types.images import Source
from .types.tables import identifiers

if TYPE_CHECKING:
    from .types import Dataset, Experiment

_DEFAULT_SETTINGS = DatasetAlignmentSettings()


def _fingerprint(dataset: Dataset) -> str:
    """Portable spatial-input signature; image pixels are never read."""
    geometry = [
        (
            a.id,
            a.source.shape,
            a.source.dtype.str,
            a.pixel_size,
            a.mode,
            a.excitation_sequence,
            a.phase,
            {c: asdict(s) for c, s in a.channel_specs.items()},
        )
        for a in dataset.acquisitions.values()
    ]
    geometry_json = _json(geometry)
    key = (dataset.localizations.revision, dataset.transforms.revision, geometry_json)
    cached = getattr(dataset, "_alignment_fingerprint_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    digest = sha256()
    for key, values in sorted(dataset.localizations._data.items()):
        digest.update(key.encode())
        digest.update(values.dtype.str.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    digest.update(geometry_json.encode())
    for key, value in sorted(dataset.transforms._calibrations.items()):
        digest.update(key.encode())
        digest.update(value.matrix.tobytes())
    digest.update(_json(sorted((a, c, n) for (a, c), n in dataset.transforms._selected.items())).encode())
    for key, (frames, displacement) in sorted(dataset.transforms._drift.items()):
        digest.update(key.encode())
        digest.update(frames.tobytes())
        digest.update(displacement.tobytes())
    value = digest.hexdigest()
    dataset._alignment_fingerprint_cache = (key, value)
    return value


def _fiducials(dataset: Dataset, acquisition: str, channel: str, ids: Any) -> tuple:
    if channel not in dataset.acquisitions[acquisition].channels:
        raise ValueError("Unknown fiducial channel")
    rows = dataset.localizations.query(
        acquisition_id=acquisition, channel=channel, ids=ids, space="corrected"
    )
    if not len(rows["id"]):
        raise ValueError("No fiducial localisations in selected partition")
    if ids is not None:
        ids = identifiers(ids)
        if len(np.unique(ids)) != len(ids) or len(rows["id"]) != len(ids):
            raise ValueError("Fiducial IDs must be unique and belong to the selected partition")
        positions = {int(v): i for i, v in enumerate(rows["id"])}
        order = [positions[int(i)] for i in ids]
        rows = {k: v[order] for k, v in rows.items()}
    points = np.column_stack([rows["x"], rows["y"]])
    if not np.isfinite(points).all():
        raise ValueError("Fiducial coordinates must be finite")
    return points, rows["id"]


def _fit_job(payload: tuple, reporter: Any) -> tuple:
    from .alignment import fit_transform

    source, reference, source_ids, reference_ids, paired, settings = payload
    reporter.report(0, 1, "Matching probing-round fiducials")
    if not paired:
        from scipy.spatial import cKDTree

        distances, nearest = cKDTree(reference).query(source)
        _, reverse = cKDTree(source).query(reference)
        keep = (distances <= settings.match_radius) & (reverse[nearest] == np.arange(len(source)))
        source, source_ids = source[keep], source_ids[keep]
        reference, reference_ids = reference[nearest[keep]], reference_ids[nearest[keep]]
    reporter.check_cancelled()
    matrix, residuals, inliers = fit_transform(
        source,
        reference,
        model=settings.model,
        ransac_threshold=settings.ransac_threshold,
        max_trials=settings.max_trials,
    )
    reporter.report(1, 1, "Probing-round alignment complete")
    return matrix, residuals, inliers, source_ids, reference_ids


def estimate_dataset_alignment(
    experiment: Experiment,
    source_dataset_id: str,
    reference_dataset_id: str,
    *,
    source_acquisition_id: str,
    source_channel: str,
    reference_acquisition_id: str,
    reference_channel: str,
    source_ids: Any = None,
    reference_ids: Any = None,
    settings: DatasetAlignmentSettings = _DEFAULT_SETTINGS,
    num_cpus: int | None = None,
    cpu: Any = None,
    progress: Any = None,
    cancel_event: Any = None,
) -> DatasetAlignmentResult:
    """Fit selected fiducials in local corrected spaces; do not apply the result.

    Explicit ID lists are ordered pairs. Without them, a positive matching radius
    is required and matching uses mutual nearest neighbours. The selected anchor
    acquisition defines each dataset's local spatial reference; other acquisitions
    must already be registered into that same reference to share its round transform.

    Args:
        experiment: Experiment containing the probing rounds.
        source_dataset_id: Source probing-round identifier.
        reference_dataset_id: Reference probing-round identifier.
        source_acquisition_id: Acquisition supplying source fiducials.
        source_channel: Logical source fiducial channel.
        reference_acquisition_id: Acquisition supplying reference fiducials.
        reference_channel: Logical reference fiducial channel.
        source_ids: Optional ordered source localisation IDs.
        reference_ids: Optional ordered corresponding reference localisation IDs.
        settings: Validated model and fiducial matching settings.
        num_cpus: Worker count; None uses the executor default.
        cpu: Optional borrowed CPU executor.
        progress: Optional structured progress callback.
        cancel_event: Optional cooperative cancellation event.

    Returns:
        Result of this operation.
    """
    if source_dataset_id == reference_dataset_id:
        raise ValueError("Choose distinct source and reference datasets")
    if (source_ids is None) != (reference_ids is None):
        raise ValueError("Supply both ordered fiducial ID lists, or neither")
    paired = source_ids is not None
    if not paired and settings.match_radius is None:
        raise ValueError("Automatic matching requires an explicit positive match_radius")
    source_dataset = experiment.datasets[source_dataset_id]
    reference_dataset = experiment.datasets[reference_dataset_id]
    with ExitStack() as stack:
        for dataset in sorted((source_dataset, reference_dataset), key=lambda d: d.id):
            stack.enter_context(dataset._lock)
        fingerprints = (_fingerprint(source_dataset), _fingerprint(reference_dataset))
        source, source_ids = _fiducials(source_dataset, source_acquisition_id, source_channel, source_ids)
        reference, reference_ids = _fiducials(
            reference_dataset, reference_acquisition_id, reference_channel, reference_ids
        )
    if paired and len(source) != len(reference):
        raise ValueError("Paired fiducial lists must have equal length")
    with executor(cpu, num_cpus) as runner:
        matrix, residuals, inliers, source_ids, reference_ids = runner.map(
            _fit_job,
            [(source, reference, source_ids, reference_ids, paired, settings)],
            progress=progress,
            cancel_event=cancel_event,
        )[0]
    if fingerprints != (_fingerprint(source_dataset), _fingerprint(reference_dataset)):
        raise StaleResultError("Fiducial inputs changed during alignment")
    report(None, cancel_event)
    return DatasetAlignmentResult(
        source_dataset_id,
        reference_dataset_id,
        source_acquisition_id,
        source_channel,
        reference_acquisition_id,
        reference_channel,
        matrix,
        residuals,
        inliers,
        source_ids,
        reference_ids,
        fingerprints,
        settings,
    )


class DatasetAlignmentManager:
    """Experiment-level common surface view, separate from dataset-local analysis."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self._records: dict[str, DatasetAlignmentResult] = {}
        self.reference_dataset_id: str | None = None
        self.reference_acquisition_id: str | None = None
        self.reference_channel: str | None = None
        self.revision = 0

    @property
    def records(self) -> Any:
        """Detached registrations keyed by source dataset ID.

        Returns:
            Detached registrations keyed by source dataset ID.
        """
        return MappingProxyType(deepcopy(self._records))

    def estimate(
        self, source_dataset_id: str, reference_dataset_id: str, **options: Any
    ) -> DatasetAlignmentResult:
        """Estimate.

        Args:
            source_dataset_id: Source probing-round identifier.
            reference_dataset_id: Reference probing-round identifier.

        Returns:
            Detached registration, correspondence IDs and fit quality.
        """
        return estimate_dataset_alignment(self.experiment, source_dataset_id, reference_dataset_id, **options)

    def apply(self, results: Any) -> None:
        """Atomically install fresh results sharing one reference anchor.

        Args:
            results: Detached registrations sharing one reference anchor.
        """
        results = list(results)
        if not results:
            raise ValueError("No alignment results supplied")
        if len({r.source_dataset_id for r in results}) != len(results):
            raise ValueError("Duplicate source dataset in alignment batch")
        with self.experiment._lock, ExitStack() as stack:
            for dataset in sorted(self.experiment.datasets.values(), key=lambda d: d.id):
                stack.enter_context(dataset._lock)
            anchor = (
                results[0].reference_dataset_id,
                results[0].reference_acquisition_id,
                results[0].reference_channel,
            )
            if self._records and anchor != (
                self.reference_dataset_id,
                self.reference_acquisition_id,
                self.reference_channel,
            ):
                raise ValueError("Reset existing alignments before changing the reference")
            for result in results:
                if anchor != (
                    result.reference_dataset_id,
                    result.reference_acquisition_id,
                    result.reference_channel,
                ):
                    raise ValueError("All results must share one reference anchor")
                if result.source_dataset_id == result.reference_dataset_id:
                    raise ValueError("Source and reference datasets must differ")
                source = self.experiment.datasets[result.source_dataset_id]
                reference = self.experiment.datasets[result.reference_dataset_id]
                if result.fingerprints != (_fingerprint(source), _fingerprint(reference)):
                    raise StaleResultError("Alignment inputs changed; estimate a new preview")
                Calibration("validation", result.matrix)
                if not np.isfinite(result.residuals).all() or not np.any(result.inliers):
                    raise ValueError("Alignment result has no valid inliers")
            staged = deepcopy(self._records)
            staged.update({r.source_dataset_id: deepcopy(r) for r in results})
            self._records = staged
            self.reference_dataset_id, self.reference_acquisition_id, self.reference_channel = anchor
            self.revision += 1

    def reset(self, dataset_id: str | None = None) -> None:
        """Remove a source transform, or clear the full reference set.

        Args:
            dataset_id: Probing-round identifier.
        """
        with self.experiment._lock:
            if dataset_id is None:
                self._records.clear()
            else:
                self._records.pop(dataset_id, None)
            if not self._records:
                self.reference_dataset_id = self.reference_acquisition_id = self.reference_channel = None
            self.revision += 1

    def is_stale(self, dataset_id: str) -> bool:
        """Is stale.

        Args:
            dataset_id: Probing-round identifier.

        Returns:
            Result of this operation.
        """
        result = self._records[dataset_id]
        source = self.experiment.datasets.get(dataset_id)
        reference = self.experiment.datasets.get(result.reference_dataset_id)
        return (
            source is None
            or reference is None
            or result.fingerprints != (_fingerprint(source), _fingerprint(reference))
        )

    def _matrix(self, dataset_id: str) -> np.ndarray:
        if self.reference_dataset_id is None:
            raise ValueError("No dataset alignment reference selected")
        if dataset_id == self.reference_dataset_id:
            if any(self.is_stale(key) for key in self._records):
                raise StaleResultError("The common view contains stale alignments")
            return np.eye(3)
        if dataset_id not in self._records:
            raise ValueError("Dataset has no alignment to the selected reference")
        if self.is_stale(dataset_id):
            raise StaleResultError("Dataset alignment is stale; estimate a new alignment")
        return self._records[dataset_id].matrix.copy()

    def coordinates(self, dataset_id: str, rows: dict[str, np.ndarray], *, space: str = "raw") -> tuple:
        """Map rows from an explicitly stated local space into the common view.

        Args:
            dataset_id: Probing-round identifier.
            rows: Coordinate columns retaining acquisition, channel and camera-frame bindings.
            space: Input or output coordinate space selected explicitly by the caller.

        Returns:
            Common-view x and y coordinate arrays.
        """
        from .transforms import project_points

        dataset = self.experiment.datasets[dataset_id]
        x, y = dataset.transforms.convert_coordinates(rows, source_space=space, target_space="corrected")
        result = project_points(self._matrix(dataset_id), np.column_stack([x, y]))
        return result[:, 0], result[:, 1]

    def image(self, dataset_id: str, acquisition_id: str, channel: str, camera_frame: int) -> np.ndarray:
        """Resample a raw frame once into the reference canvas, on demand.

        Args:
            dataset_id: Probing-round identifier.
            acquisition_id: Original recording identifier.
            channel: Logical channel identifier.
            camera_frame: Original camera-frame index.

        Returns:
            Resampled frame in the reference canvas.
        """
        from scipy.ndimage import map_coordinates

        from .transforms import project_points

        dataset = self.experiment.datasets[dataset_id]
        matrix = self._matrix(dataset_id) @ dataset.transforms.matrix(acquisition_id, channel, camera_frame)
        source = dataset.acquisitions[acquisition_id].channels[channel]
        image = source[source.channel_index(camera_frame)]
        reference = self.experiment.datasets[self.reference_dataset_id]
        shape = (
            reference.acquisitions[self.reference_acquisition_id].channels[self.reference_channel].shape[1:]
        )
        y, x = np.indices(shape)
        raw = project_points(np.linalg.inv(matrix), np.column_stack([x.ravel(), y.ravel()]))
        valid = np.isfinite(raw).all(axis=1)
        raw[~valid] = -1e20
        pixels = map_coordinates(
            image.astype(float), [raw[:, 1], raw[:, 0]], order=1, mode="constant", cval=0
        )
        pixels[~valid] = 0
        return pixels.reshape(shape)

    def view(self, dataset_id: str, acquisition_id: str, channel: str) -> Source:
        """Lazy stack retaining the source round's camera-frame timeline.

        Args:
            dataset_id: Probing-round identifier.
            acquisition_id: Original recording identifier.
            channel: Logical channel identifier.

        Returns:
            Lazy stack retaining original source-frame indices.
        """
        self._matrix(dataset_id)
        return AlignedDatasetView(self, dataset_id, acquisition_id, channel)


class AlignedDatasetView(Source):
    """One-frame-at-a-time experiment view; original pixels stay untouched."""

    def __init__(
        self, manager: DatasetAlignmentManager, dataset_id: str, acquisition_id: str, channel: str
    ) -> None:
        self.manager, self.dataset_id, self.acquisition_id, self.channel = (
            manager,
            dataset_id,
            acquisition_id,
            channel,
        )
        self.source = manager.experiment.datasets[dataset_id].acquisitions[acquisition_id].channels[channel]
        reference = manager.experiment.datasets[manager.reference_dataset_id]
        canvas = reference.acquisitions[manager.reference_acquisition_id].channels[manager.reference_channel]
        self.shape, self.dtype = (len(self.source), *canvas.shape[1:]), np.dtype(float)

    def _frame(self, index: int) -> np.ndarray:
        return self.manager.image(
            self.dataset_id, self.acquisition_id, self.channel, int(self.source.source_frames[index])
        )
