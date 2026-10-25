"""Model-based affine channel alignment without modifying raw coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from ._processing import check_snapshot, executor, snapshot
from .types._shared import Calibration
from .types.experiment import Experiment
from .types.processing import AlignmentResult
from .types.tables import identifiers

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .cpu import CpuExecutor, WorkerProgress
    from .types import CancelEvent, ProgressCallback


@logged
def _job(payload: tuple, reporter: WorkerProgress) -> tuple:
    """Job.

    Args:
        payload: Serializable worker inputs; large arrays remain bounded to a job or image chunk.
        reporter: Worker progress and cooperative cancellation reporter.

    Returns:
        Computed tuple value.
    """
    source, target, source_ids, target_ids, model, match_radius, matched = payload
    reporter.report(0, 1, "Matching channel fiducials")
    if not matched:
        from scipy.spatial import cKDTree

        distances, nearest = cKDTree(target).query(source)
        _, reverse = cKDTree(source).query(target)
        keep = (distances <= match_radius) & (reverse[nearest] == np.arange(len(source)))
        source, source_ids = source[keep], source_ids[keep]
        target, target_ids = target[nearest[keep]], target_ids[nearest[keep]]
    reporter.report(0.5, 1, "Estimating channel transform")
    if model == "translation":
        if not len(source):
            raise ValueError("Translation needs at least one fiducial pair")
        matrix = np.eye(3)
        matrix[:2, 2] = np.mean(target - source, axis=0)
    else:
        if len(source) < 3:
            raise ValueError("Affine alignment needs at least three non-collinear fiducial pairs")
        design = np.column_stack([source, np.ones(len(source))])
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < 3:
            raise ValueError("Affine fiducials are degenerate/collinear")
        matrix = np.vstack([coefficients.T, [0.0, 0.0, 1.0]])
    Calibration("validation", matrix)
    residuals = np.linalg.norm(source @ matrix[:2, :2].T + matrix[:2, 2] - target, axis=1)
    reporter.report(1, 1, "Channel alignment complete")
    return matrix, residuals, source_ids, target_ids


def _rows(
    experiment: Experiment, acquisition_id: str, channel: str, ids: ArrayLike | None
) -> dict[str, np.ndarray]:
    """Rows.

    Args:
        experiment: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        ids: Stable entity IDs; None selects the available partition.

    Returns:
        Detached named result columns.
    """
    rows = experiment.localizations.query(acquisition_id=acquisition_id, channel=channel, ids=ids)
    if len(rows["id"]) == 0:
        raise ValueError("No fiducial localizations in selected channel")
    if ids is not None:
        ids = identifiers(ids)
        if len(np.unique(ids)) != len(ids) or len(rows["id"]) != len(ids):
            raise ValueError("Fiducial IDs must exist, be unique, and belong to the selected channel")
        positions = {int(value): i for i, value in enumerate(rows["id"])}
        order = [positions[int(value)] for value in ids]
        rows = {k: v[order] for k, v in rows.items()}
    return np.column_stack([rows["x"], rows["y"]]), rows["id"]


@logged
def _run(
    experiment: Experiment,
    acquisition_id: str,
    source_channel: str,
    target_channel: str,
    *,
    source_ids: ArrayLike | None,
    target_ids: ArrayLike | None,
    model: str,
    match_radius: float,
    name: str,
    apply: bool,
    num_cpus: int | None,
    cpu: CpuExecutor | None,
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
) -> Any:
    """Run.

    Args:
        experiment: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        source_channel: Source channel used by this operation.
        target_channel: Logical target channel in the same acquisition.
        source_ids: Optional stable IDs restricting source localizations.
        target_ids: Optional stable IDs restricting target localizations.
        model: Model or fitting method selected by the caller.
        match_radius: Match radius used by this operation.
        name: Identifier or field name used to address this value.
        apply: Whether to commit successful results to the model.
        num_cpus: Worker process count; None uses the executor default.
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    if source_channel == target_channel or model not in ("translation", "affine"):
        raise ValueError("Choose distinct channels and translation or affine alignment")
    matched = source_ids is not None and target_ids is not None
    if (source_ids is None) != (target_ids is None):
        raise ValueError("Supply both source_ids and target_ids, or neither")
    if not matched and (match_radius is None or not np.isfinite(match_radius) or match_radius <= 0):
        raise ValueError("Automatic matching requires an explicit positive match_radius")
    acquisition = experiment.acquisitions[acquisition_id]
    if source_channel not in acquisition.channels or target_channel not in acquisition.channels:
        raise ValueError("Unknown alignment channel")
    expected = snapshot(experiment, acquisition_id)
    source, source_ids = _rows(experiment, acquisition_id, source_channel, source_ids)
    target, target_ids = _rows(experiment, acquisition_id, target_channel, target_ids)
    if matched and len(source) != len(target):
        raise ValueError("Matched fiducial lists must have equal length")
    payload = source, target, source_ids, target_ids, model, match_radius, matched
    with executor(cpu, num_cpus) as runner:
        matrix, residuals, source_ids, target_ids = runner.map(
            _job, [payload], progress=progress, cancel_event=cancel_event
        )[0]
    check_snapshot(experiment, acquisition_id, expected)
    reference_matrix = experiment.transforms._registration(acquisition_id, target_channel) @ matrix
    calibration_name = None
    if apply:
        calibration_name = name or f"{acquisition_id}:{source_channel}->{target_channel}"
        existing = experiment.calibrations.get(calibration_name)
        if existing is not None:
            raise ValueError(
                "Calibration name already exists; choose a new name to preserve other selections"
            )
        experiment.transforms.add_calibration(
            Calibration(
                calibration_name,
                reference_matrix,
                {
                    "model": model,
                    "source_channel": source_channel,
                    "target_channel": target_channel,
                    "source_ids": source_ids.tolist(),
                    "target_ids": target_ids.tolist(),
                    "residuals": residuals.tolist(),
                },
            )
        )
        experiment.transforms.select(acquisition_id, source_channel, calibration_name)
        experiment.processing.settings.setdefault("alignment", {}).setdefault(acquisition_id, {})[
            str(source_channel)
        ] = {
            "model": model,
            "match_radius": match_radius,
            "target_channel": target_channel,
            "calibration_name": calibration_name,
            "rms_residual": float(np.sqrt(np.mean(residuals**2))),
        }
    return AlignmentResult(
        acquisition_id,
        str(source_channel),
        str(target_channel),
        matrix,
        reference_matrix,
        residuals,
        source_ids,
        target_ids,
        calibration_name,
    )


@logged
def estimate_alignment(
    experiment: Experiment,
    acquisition_id: str,
    source_channel: str,
    target_channel: str,
    *,
    source_ids: ArrayLike | None = None,
    target_ids: ArrayLike | None = None,
    model: str = "affine",
    match_radius: float | None = None,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> AlignmentResult:
    """Fit channel correspondences without changing the calibration selection.

    Args:
        experiment: Experiment supplying raw effective localization coordinates.
        acquisition_id: Acquisition containing both logical channels.
        source_channel: Channel to transform.
        target_channel: Channel defining the matched target coordinates.
        source_ids: Optional ordered source fiducial localization IDs.
        target_ids: Corresponding target IDs in matching order.
        model: translation or affine.
        match_radius: Maximum mutual-nearest distance when IDs are not paired.
        num_cpus: Positive worker count or None for the CPU default.
        cpu: Optional reusable CpuExecutor.
        progress: Parent-process ProgressUpdate callback.
        cancel_event: Optional caller cancellation event.

    Returns:
        Source-to-target matrix, composed reference matrix and pair residuals.
    """
    return _run(
        experiment,
        acquisition_id,
        source_channel,
        target_channel,
        source_ids=source_ids,
        target_ids=target_ids,
        model=model,
        match_radius=match_radius,
        name=None,
        apply=False,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )


@logged
def align_channels(
    experiment: Experiment,
    acquisition_id: str,
    source_channel: str,
    target_channel: str,
    *,
    source_ids: ArrayLike | None = None,
    target_ids: ArrayLike | None = None,
    model: str = "affine",
    match_radius: float | None = None,
    name: str | None = None,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> AlignmentResult:
    """Estimate and select a named source-channel registration calibration.

    Args:
        experiment: Experiment to receive a non-destructive calibration.
        acquisition_id: Acquisition identifier.
        source_channel: Channel to align.
        target_channel: Channel defining correspondence coordinates.
        source_ids: Optional ordered source fiducial IDs.
        target_ids: Corresponding target fiducial IDs.
        model: translation or affine.
        match_radius: Required maximum matching distance for unpaired tables.
        name: New calibration name; existing names are not overwritten.
        num_cpus: Positive CPU count or None for the default.
        cpu: Optional reusable CpuExecutor.
        progress: Parent-process ProgressUpdate callback.
        cancel_event: Optional caller cancellation event.

    Returns:
        AlignmentResult including the selected calibration name.
    """
    return _run(
        experiment,
        acquisition_id,
        source_channel,
        target_channel,
        source_ids=source_ids,
        target_ids=target_ids,
        model=model,
        match_radius=match_radius,
        name=name,
        apply=True,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )
