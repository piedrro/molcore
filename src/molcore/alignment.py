"""Robust optical channel alignment without modifying raw coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from ._processing import check_snapshot, executor, snapshot
from .execution import ExecutionContext, _record_execution
from .snapshots import DatasetSnapshot, processor_model
from .transforms import project_points
from .types._shared import Calibration
from .types.dataset import Dataset
from .types.processing import AlignmentResult
from .types.tables import identifiers

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .cpu import CpuExecutor, ProgressReporter
    from .types import CancelEvent, ProgressCallback


def _homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit a normalized DLT homography, rejecting degenerate correspondences."""
    normalized, transforms = ([], [])
    for points in (source, target):
        center = points.mean(axis=0)
        radius = np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))
        if radius <= np.finfo(float).eps:
            raise ValueError("Homography fiducials are degenerate")
        scale = np.sqrt(2) / radius
        transform = np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]])
        transforms.append(transform)
        normalized.append(project_points(transform, points))
    src, dst = normalized
    x, y = src.T
    u, v = dst.T
    zero, one = (np.zeros(len(x)), np.ones(len(x)))
    design = np.vstack(
        [
            np.column_stack([-x, -y, -one, zero, zero, zero, u * x, u * y, u]),
            np.column_stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v]),
        ]
    )
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("Homography fiducials are degenerate/collinear")
    _, _, vectors = np.linalg.svd(design, full_matrices=True)
    matrix = np.linalg.solve(transforms[1], vectors[-1].reshape(3, 3) @ transforms[0])
    matrix /= matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else np.linalg.norm(matrix)
    return Calibration("validation", matrix).matrix.copy()


def _fit_model(source: np.ndarray, target: np.ndarray, model: str) -> np.ndarray:
    """Fit one model to an already selected set of correspondences."""
    if model == "homography":
        return _homography(source, target)
    matrix = np.eye(3)
    if model == "translation":
        matrix[:2, 2] = np.mean(target - source, axis=0)
    else:
        design = np.column_stack([source, np.ones(len(source))])
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < 3:
            raise ValueError("Affine fiducials are degenerate/collinear")
        matrix[:2] = coefficients.T
    return Calibration("validation", matrix).matrix.copy()


def fit_transform(
    source: ArrayLike,
    target: ArrayLike,
    *,
    model: str = "affine",
    ransac_threshold: float = 1.0,
    max_trials: int = 1000,
) -> tuple:
    """Fit paired 2D points with deterministic RANSAC.

    Args:
        source: Source fiducial coordinates, shape (n, 2).
        target: Corresponding reference coordinates in matching order.
        model: Translation, affine or homography model.
        ransac_threshold: Maximum accepted residual in reference pixels.
        max_trials: Maximum number of deterministic RANSAC samples.

    Returns:
        Fitted homogeneous matrix, pair residuals and inlier mask.
    """
    source, target = (np.asarray(source, dtype=float), np.asarray(target, dtype=float))
    if source.ndim != 2 or source.shape[1:] != (2,) or target.shape != source.shape:
        raise ValueError("Expected matching (n, 2) point arrays")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Fiducials must be finite")
    minimum = {"translation": 1, "affine": 3, "homography": 4}.get(model)
    if minimum is None or len(source) < minimum:
        raise ValueError("Choose translation (1+), affine (3+), or homography (4+) fiducial pairs")
    if (
        not np.isfinite(ransac_threshold)
        or ransac_threshold <= 0
        or (not isinstance(max_trials, int))
        or (max_trials < 1)
    ):
        raise ValueError("RANSAC threshold and trials must be positive")
    rng = np.random.default_rng(0)
    best, best_score = (None, (-1, -np.inf))
    for _ in range(max_trials if len(source) > minimum else 1):
        sample = rng.choice(len(source), minimum, replace=False)
        try:
            candidate = _fit_model(source[sample], target[sample], model)
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = np.linalg.norm(project_points(candidate, source) - target, axis=1)
        selected = np.isfinite(residuals) & (residuals <= ransac_threshold)
        score = (int(selected.sum()), -float(np.sum(residuals[selected] ** 2)))
        if score > best_score:
            best, best_score = (selected, score)
        if selected.all():
            break
    if best is None or best.sum() < minimum:
        raise ValueError(f"No nondegenerate {model} consensus")
    inliers = best
    for _ in range(5):
        matrix = _fit_model(source[inliers], target[inliers], model)
        residuals = np.linalg.norm(project_points(matrix, source) - target, axis=1)
        selected = np.isfinite(residuals) & (residuals <= ransac_threshold)
        if np.array_equal(selected, inliers) or selected.sum() < minimum:
            break
        inliers = selected
    matrix = _fit_model(source[inliers], target[inliers], model)
    residuals = np.linalg.norm(project_points(matrix, source) - target, axis=1)
    return (matrix, residuals, inliers)


@logged
def _job(payload: tuple, reporter: ProgressReporter) -> tuple:
    """Job.

    Args:
        payload: Serializable worker inputs; large arrays remain bounded to a job or image chunk.
        reporter: Worker progress and cooperative cancellation reporter.

    Returns:
        Computed tuple value.
    """
    source, target, source_ids, target_ids, model, match_radius, matched, threshold, trials = payload
    reporter.report(0, 1, "Matching channel fiducials")
    if not matched:
        from scipy.spatial import cKDTree

        distances, nearest = cKDTree(target).query(source)
        _, reverse = cKDTree(source).query(target)
        keep = (distances <= match_radius) & (reverse[nearest] == np.arange(len(source)))
        source, source_ids = (source[keep], source_ids[keep])
        target, target_ids = (target[nearest[keep]], target_ids[nearest[keep]])
    reporter.report(0.5, 1, "Estimating channel transform")
    matrix, residuals, inliers = fit_transform(
        source, target, model=model, ransac_threshold=threshold, max_trials=trials
    )
    reporter.report(1, 1, "Channel alignment complete")
    return (matrix, residuals, source_ids, target_ids, inliers)


def _rows(
    dataset: Dataset, acquisition_id: str, channel: str, ids: ArrayLike | None
) -> dict[str, np.ndarray]:
    """Rows.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        ids: Stable entity IDs; None selects the available partition.

    Returns:
        Detached named result columns.
    """
    rows = dataset.localizations.query(acquisition_id=acquisition_id, channel=channel, ids=ids, space="raw")
    if len(rows["id"]) == 0:
        raise ValueError("No fiducial localizations in selected channel")
    if ids is not None:
        ids = identifiers(ids)
        if len(np.unique(ids)) != len(ids) or len(rows["id"]) != len(ids):
            raise ValueError("Fiducial IDs must exist, be unique, and belong to the selected channel")
        positions = {int(value): i for i, value in enumerate(rows["id"])}
        order = [positions[int(value)] for value in ids]
        rows = {k: v[order] for k, v in rows.items()}
    return (np.column_stack([rows["x"], rows["y"]]), rows["id"])


@logged
def _run(
    dataset: Dataset,
    acquisition_id: str,
    source_channel: str,
    target_channel: str,
    *,
    source_ids: ArrayLike | None,
    target_ids: ArrayLike | None,
    model: str,
    match_radius: float,
    ransac_threshold: float,
    max_trials: int,
    name: str,
    num_cpus: int | None,
    cpu: CpuExecutor | None,
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
) -> Any:
    """Run.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
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
    if source_channel == target_channel or model not in ("translation", "affine", "homography"):
        raise ValueError("Choose distinct channels and translation, affine or homography alignment")
    matched = source_ids is not None and target_ids is not None
    if (source_ids is None) != (target_ids is None):
        raise ValueError("Supply both source_ids and target_ids, or neither")
    if not matched and (match_radius is None or not np.isfinite(match_radius) or match_radius <= 0):
        raise ValueError("Automatic matching requires an explicit positive match_radius")
    acquisition = dataset.acquisitions[acquisition_id]
    if source_channel not in acquisition.channels or target_channel not in acquisition.channels:
        raise ValueError("Unknown alignment channel")
    source_emission = acquisition.channel_specs[source_channel].emission
    target_emission = acquisition.channel_specs[target_channel].emission
    if source_emission is not None and source_emission == target_emission:
        raise ValueError("Choose distinct physical emission paths")
    if acquisition.mode in ("fret", "alex") and source_emission == "D" and (target_emission == "A"):
        raise ValueError("Donor emission is the optical reference; align acceptor into donor")
    expected = snapshot(dataset, acquisition_id)
    source, source_ids = _rows(dataset, acquisition_id, source_channel, source_ids)
    target, target_ids = _rows(dataset, acquisition_id, target_channel, target_ids)
    if matched and len(source) != len(target):
        raise ValueError("Matched fiducial lists must have equal length")
    payload = (
        source,
        target,
        source_ids,
        target_ids,
        model,
        match_radius,
        matched,
        ransac_threshold,
        max_trials,
    )
    with executor(cpu, num_cpus) as runner:
        matrix, residuals, source_ids, target_ids, inliers = runner.map(
            _job, [payload], progress=progress, cancel_event=cancel_event
        )[0]
    check_snapshot(dataset, acquisition_id, expected)
    reference_matrix = dataset.transforms._registration(acquisition_id, target_channel) @ matrix
    result = AlignmentResult(
        acquisition_id,
        str(source_channel),
        str(target_channel),
        matrix,
        reference_matrix,
        residuals,
        source_ids,
        target_ids,
        None,
        inliers,
        expected=expected,
        model=model,
        match_radius=match_radius,
        requested_name=name,
    )
    return result


@logged
@_record_execution
def estimate_alignment(
    dataset: DatasetSnapshot,
    acquisition_id: str,
    source_channel: str,
    target_channel: str,
    *,
    source_ids: ArrayLike | None = None,
    target_ids: ArrayLike | None = None,
    model: str = "affine",
    match_radius: float | None = None,
    ransac_threshold: float = 1.0,
    max_trials: int = 1000,
    context: ExecutionContext | None = None,
) -> AlignmentResult:
    """Fit channel correspondences without changing the calibration selection.

    Args:
        context: Execution policy, frame chunk size, progress and cancellation.
        dataset: Dataset supplying raw effective localization coordinates.
        acquisition_id: Acquisition containing both logical channels.
        source_channel: Channel to transform.
        target_channel: Channel defining the matched target coordinates.
        source_ids: Optional ordered source fiducial localization IDs.
        target_ids: Corresponding target IDs in matching order.
        model: Translation, affine, or homography fit.
        match_radius: Maximum mutual-nearest distance when IDs are not paired.
    Returns:
        Source-to-target matrix, composed reference matrix and pair residuals.
    """
    dataset = processor_model(dataset)
    context = context or ExecutionContext()
    context.check_cancelled()
    cpu, num_cpus, progress, cancel_event = (
        context.cpu,
        context.num_cpus,
        context.progress,
        context.cancel_event,
    )
    return _run(
        dataset,
        acquisition_id,
        source_channel,
        target_channel,
        source_ids=source_ids,
        target_ids=target_ids,
        model=model,
        match_radius=match_radius,
        ransac_threshold=ransac_threshold,
        max_trials=max_trials,
        name=None,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )


@logged
@_record_execution
def align_channels(
    dataset: DatasetSnapshot,
    acquisition_id: str,
    source_channel: str,
    target_channel: str,
    *,
    source_ids: ArrayLike | None = None,
    target_ids: ArrayLike | None = None,
    model: str = "affine",
    match_radius: float | None = None,
    ransac_threshold: float = 1.0,
    max_trials: int = 1000,
    name: str | None = None,
    context: ExecutionContext | None = None,
) -> AlignmentResult:
    """Estimate and select a named source-channel registration calibration.

    Args:
        context: Execution policy, frame chunk size, progress and cancellation.
        dataset: Dataset to receive a non-destructive calibration.
        acquisition_id: Acquisition identifier.
        source_channel: Channel to align.
        target_channel: Channel defining correspondence coordinates.
        source_ids: Optional ordered source fiducial IDs.
        target_ids: Corresponding target fiducial IDs.
        model: Translation, affine, or homography fit.
        match_radius: Required maximum matching distance for unpaired tables.
        name: New calibration name; existing names are not overwritten.
    Returns:
        AlignmentResult including the selected calibration name.
    """
    dataset = processor_model(dataset)
    context = context or ExecutionContext()
    context.check_cancelled()
    cpu, num_cpus, progress, cancel_event = (
        context.cpu,
        context.num_cpus,
        context.progress,
        context.cancel_event,
    )
    return _run(
        dataset,
        acquisition_id,
        source_channel,
        target_channel,
        source_ids=source_ids,
        target_ids=target_ids,
        model=model,
        match_radius=match_radius,
        ransac_threshold=ransac_threshold,
        max_trials=max_trials,
        name=name,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )
