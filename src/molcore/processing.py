"""Transactional application of detached processing results for CLI and Qt callers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from ._processing import backend_version, record
from .photometry import apply_photometry as apply_photometry
from .photon_sequences import apply_photon_sequences as apply_photon_sequences
from .staging import transact
from .tracking import apply_tracking as apply_tracking
from .types import Calibration, Dataset

if TYPE_CHECKING:
    from .types import AlignmentResult, CameraSettings, DriftResult, LocalisationResult, Segmentation


@logged
def apply_batch(
    dataset: Dataset,
    operations: Iterable[Callable[[Dataset], Any]],
    *,
    expected: tuple | None = None,
) -> list[Any]:
    """Run model-update callables on one staged model and commit all or none.

    Callables must restrict effects to the staged scientific model: external I/O
    and mutations to borrowed image arrays cannot be rolled back.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        operations: Model-update callables executed atomically against one staged dataset.
        expected: Source revision token used to reject stale results.

    Returns:
        Computed list[Any] value.
    """
    operations = tuple(operations)
    if not all(callable(operation) for operation in operations):
        raise ValueError("Batch operations must be callable")

    def commit(model: Dataset) -> list[Any]:
        """Commit.

        Args:
            model: Model or fitting method selected by the caller.

        Returns:
            Computed list[Any] value.
        """
        model._batch_source_token = model._transaction_source_token
        return [operation(model) for operation in operations]

    return transact(dataset, commit, expected=expected)


@logged
def apply_localisation(
    dataset: Dataset, result: Any, *, camera: CameraSettings | None, expected: tuple | None = None
) -> LocalisationResult:
    """Replace one complete fitted partition after validating the whole update.

    Args:
        dataset: Live model to update.
        result: Detached LocalisationResult; subset previews cannot be applied.
        camera: Explicit CameraSettings used to produce the result.
        expected: Optional state token captured before computation.

    Returns:
        Result containing the stable IDs allocated by the model.
    """
    if result.frame_range is not None:
        raise ValueError("Subset processing is a preview and cannot replace a partition")
    if result.failed_fits and not len(result.columns["frame"]):
        raise ValueError("No localisation fits converged; active results were preserved")

    def commit(model: Dataset) -> Any:
        """Commit.

        Args:
            model: Model or fitting method selected by the caller.

        Returns:
            Result returned after updating the staged model.
        """
        ids = model.localizations.replace(result.acquisition_id, result.channel, **result.columns)
        record(
            model,
            "localisation",
            result.acquisition_id,
            result.channel,
            {
                "settings": asdict(result.settings),
                "camera": asdict(camera),
                "backend": backend_version(),
                "camera_frames": "all logical-channel frames",
                "failed_fits": result.failed_fits,
            },
        )
        return replace(result, localization_ids=ids)

    return transact(dataset, commit, expected=expected)


@logged
def apply_drift(dataset: Dataset, results: Any, *, expected: tuple | None = None) -> dict[str, DriftResult]:
    """Apply one or more drift results atomically.

    Args:
        dataset: Live model receiving reference-coordinate displacements.
        results: Mapping from acquisition identifier to DriftResult.
        expected: Optional pre-computation state token.

    Returns:
        Mapping of results marked applied.
    """

    def commit(model: Dataset) -> Any:
        """Commit.

        Args:
            model: Model or fitting method selected by the caller.

        Returns:
            Result returned after updating the staged model.
        """
        for key, value in results.items():
            if key != value.acquisition_id:
                raise ValueError("Drift result key does not match its acquisition")
            model.transforms.set_drift(key, value.displacement, frames=value.frames)
            record(
                model,
                "undrifting",
                key,
                value.channel,
                {
                    "settings": asdict(value.settings),
                    "backend": backend_version(),
                    "frame_coverage": [int(value.frames[0]), int(value.frames[-1])],
                    "coordinates": "registered reference pixels",
                },
            )
        return {k: replace(v, applied=True) for k, v in results.items()}

    return transact(dataset, commit, expected=expected)


@logged
def apply_alignment(
    dataset: Dataset,
    result: Any,
    *,
    model: str = "affine",
    match_radius: float | None = None,
    name: str | None = None,
    expected: tuple | None = None,
) -> AlignmentResult:
    """Install a fresh named calibration from an alignment preview.

    Args:
        dataset: Live model receiving the selected calibration.
        result: Detached AlignmentResult.
        model: Translation, affine, or homography fit recorded as provenance.
        match_radius: Explicit automatic correspondence radius, when used.
        name: New calibration name; defaults to the acquisition/channel pair.
        expected: Optional pre-computation state token.

    Returns:
        Result containing its newly selected calibration name.
    """
    name = name or f"{result.acquisition_id}:{result.source_channel}->{result.target_channel}"

    def commit(staged: Any) -> Any:
        """Commit.

        Args:
            staged: Staged used by this operation.

        Returns:
            Result returned after updating the staged model.
        """
        if name in staged.calibrations:
            raise ValueError("Calibration name already exists; choose a new name")
        staged.transforms.add_calibration(
            Calibration(
                name,
                result.reference_matrix,
                {
                    "model": model,
                    "source_channel": result.source_channel,
                    "target_channel": result.target_channel,
                    "source_ids": result.source_ids.tolist(),
                    "target_ids": result.target_ids.tolist(),
                    "residuals": result.residuals.tolist(),
                    "inliers": result.inliers.tolist(),
                    "source_emission": staged.acquisitions[result.acquisition_id]
                    .channel_specs[result.source_channel]
                    .emission,
                    "source_shape": list(
                        staged.acquisitions[result.acquisition_id].channels[result.source_channel].shape[1:]
                    ),
                    "source_roi": list(
                        staged.acquisitions[result.acquisition_id].channels[result.source_channel].roi
                    ),
                    "target_emission": staged.acquisitions[result.acquisition_id]
                    .channel_specs[result.target_channel]
                    .emission,
                    "target_shape": list(
                        staged.acquisitions[result.acquisition_id].channels[result.target_channel].shape[1:]
                    ),
                    "target_roi": list(
                        staged.acquisitions[result.acquisition_id].channels[result.target_channel].roi
                    ),
                    "pixel_size_nm": staged.acquisitions[result.acquisition_id].pixel_size,
                },
            )
        )
        staged.transforms.select(result.acquisition_id, result.source_channel, name)
        record(
            staged,
            "alignment",
            result.acquisition_id,
            result.source_channel,
            {
                "model": model,
                "match_radius": match_radius,
                "target_channel": result.target_channel,
                "calibration_name": name,
                "rms_residual": float(np.sqrt(np.mean(result.residuals**2))),
            },
        )
        return replace(result, calibration_name=name)

    return transact(dataset, commit, expected=expected)


@logged
def apply_segmentations(
    dataset: Dataset, results: Any, *, expected: tuple | None = None
) -> dict[str, Segmentation]:
    """Validate and store every mask before exposing any part of a batch.

    Args:
        dataset: Live model receiving static masks and source provenance.
        results: Mapping of segmentation IDs to detached Segmentation objects.
        expected: Optional pre-computation state token.

    Returns:
        Applied segmentation mapping.
    """

    def commit(model: Dataset) -> Any:
        """Commit.

        Args:
            model: Model or fitting method selected by the caller.

        Returns:
            Result returned after updating the staged model.
        """
        previous = model.processing.settings.get("cellpose", {})
        if not isinstance(previous, dict):
            raise ValueError("Cellpose processing metadata must be a dictionary")
        for key, value in results.items():
            if key != value.id:
                raise ValueError("Segmentation ID does not match its result key")
            model.add_segmentation(deepcopy(value))
        cellpose_values = [v for v in results.values() if v.metadata.get("algorithm") == "cellpose"]
        if cellpose_values:
            model.processing.settings["cellpose"] = {
                **previous,
                **{v.id: deepcopy(v.metadata) for v in cellpose_values},
            }
            for package in ("cellpose", "torch"):
                model.processing.software[package] = cellpose_values[0].metadata[f"{package}_version"]
        return results

    return transact(dataset, commit, expected=expected)
