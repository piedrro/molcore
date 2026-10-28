"""Headless Trackpy linking with camera-frame memory and atomic application."""

from __future__ import annotations

from dataclasses import asdict, replace
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

from ._logging import logged
from ._processing import executor
from .backends import BackendRegistry, ProcessorBackend, missing_backend, resolve_backend
from .cpu import CpuExecutor
from .execution import ExecutionContext
from .revisions import FreshnessToken
from .staging import transact
from .types import (
    CancelEvent,
    Dataset,
    ProgressCallback,
    TrackingResult,
    TrackingSettings,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .cpu import WorkerProgress

_DEFAULT_SETTINGS = TrackingSettings()


@logged
def _job(payload: tuple, reporter: WorkerProgress) -> tuple[np.ndarray, ...]:
    """Job.

    Args:
        payload: Serializable worker inputs; large arrays remain bounded to a job or image chunk.
        reporter: Worker progress and cooperative cancellation reporter.

    Returns:
        Computed tuple[NumPy array, ...] value.
    """
    import trackpy as tp

    rows, labels, settings = payload
    output = []
    groups = np.unique(labels[labels > 0])
    for group_index, label in enumerate(groups):
        subset = {name: values[labels == label] for name, values in rows.items()}
        if not len(subset["id"]):
            continue
        frames = subset["frame"]
        first, last = int(frames.min()), int(frames.max())
        frame_ids = {}

        def coordinates(
            first: int = first,
            last: int = last,
            frames: ArrayLike | None = frames,
            subset: dict[str, np.ndarray] = subset,
            frame_ids: ArrayLike = frame_ids,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Coordinates.

            Args:
                first: First used by this operation.
                last: Last used by this operation.
                frames: Original camera-frame indices for the selected samples.
                subset: Subset used by this operation.
                frame_ids: Frame ids used by this operation.

            Returns:
                Computed tuple[NumPy array, NumPy array] value.
            """
            for frame in range(first, last + 1):
                reporter.check_cancelled()
                indices = np.flatnonzero(frames == frame)
                frame_ids[frame] = subset["id"][indices]
                yield frame, np.column_stack([subset["x"][indices], subset["y"][indices]])

        trajectories = {}
        for frame, particles in tp.link_iter(
            coordinates(),
            search_range=settings.search_range,
            memory=settings.memory,
            link_strategy="recursive",
        ):
            reporter.check_cancelled()
            for particle, lid in zip(particles, frame_ids.pop(frame), strict=True):
                trajectories.setdefault(int(particle), []).append(lid)
            reporter.report(
                group_index + (frame - first + 1) / (last - first + 1), len(groups), "Linking camera frames"
            )
        output.extend(
            np.asarray(members, dtype=np.uint64)
            for members in trajectories.values()
            if len(members) >= settings.min_length
        )
    return tuple(output)


@logged
def track(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    *,
    settings: TrackingSettings = _DEFAULT_SETTINGS,
    segmentation_id: str | None = None,
    apply: bool = True,
    cpu: CpuExecutor | None = None,
    num_cpus: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
    context: ExecutionContext | None = None,
    backend: ProcessorBackend | str | None = None,
    registry: BackendRegistry | None = None,
) -> TrackingResult:
    """Link one acquisition/channel partition independently using Trackpy 0.7.

    Search range is in selected-space pixels. Empty camera frames consume memory,
    including other excitation frames. Segmentation labels constrain independent
    linking groups; background label zero is excluded. A full successful applying
    run replaces wholly contained partition tracks, preserving localization IDs.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        settings: Validated settings controlling this operation.
        segmentation_id: Optional segmentation identifier restricting the selected objects.
        apply: Whether to commit successful results to the model.
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        num_cpus: Worker process count; None uses the executor default.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed TrackingResult value.
    """
    options = (
        (context or ExecutionContext())
        .resolve(cpu=cpu, num_cpus=num_cpus, progress=progress, cancel_event=cancel_event)
        .kwargs()
    )
    cpu, num_cpus, progress, cancel_event = (
        options[k] for k in ("cpu", "num_cpus", "progress", "cancel_event")
    )
    selected_backend = resolve_backend("track", backend, registry=registry)
    if not isinstance(settings, TrackingSettings):
        raise TypeError("settings must be TrackingSettings")
    if channel not in dataset.acquisitions[acquisition_id].channels:
        raise ValueError("Unknown channel")
    expected = dataset.dependencies(acquisition_id, channel)
    rows = dataset.localizations.query(
        acquisition_id=acquisition_id, channel=channel, space=settings.space
    ).to_columns()
    labels = np.ones(len(rows["id"]), dtype=np.int64)
    if segmentation_id is not None:
        raw = dataset.localizations.query(acquisition_id=acquisition_id, channel=channel, space="raw")
        labels = dataset.segmentation_labels(segmentation_id, raw)
    from importlib.metadata import PackageNotFoundError

    try:
        installed = version("trackpy") if selected_backend is None else selected_backend.version
    except PackageNotFoundError as exc:
        raise missing_backend("trackpy") from exc
    if selected_backend is None and tuple(int(p) for p in installed.split(".")[:2]) != (0, 7):
        raise ImportError(f"MolCore supports Trackpy >=0.7,<0.8; found {installed}")
    with executor(cpu, num_cpus) as runner:
        output = runner.map(
            _job if selected_backend is None else selected_backend.compute,
            [(rows, labels, settings)],
            progress=progress,
            cancel_event=cancel_event,
        )
    expected.validate(dataset)
    result = TrackingResult(
        acquisition_id, channel, output[0], settings, expected, installed, segmentation_id
    )
    return apply_tracking(dataset, result) if apply else result


@logged
def apply_tracking(
    dataset: Dataset, result: TrackingResult, *, expected: FreshnessToken | None = None
) -> TrackingResult:
    """Apply detached memberships transactionally after checking the source state.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        result: Detached computation result to inspect or apply.
        expected: Source revision token used to reject stale results.

    Returns:
        Computed TrackingResult value.
    """

    def commit(model: Dataset) -> TrackingResult:
        """Commit.

        Args:
            model: Model or fitting method selected by the caller.

        Returns:
            Computed TrackingResult value.
        """
        ids = model.tracks.replace(
            result.acquisition_id,
            result.channel,
            list(result.memberships),
            segmentation_id=result.segmentation_id,
            metadata={"settings": asdict(result.settings), "backend": result.backend_version},
        )
        model.processing.settings.setdefault("tracking", {}).setdefault(result.acquisition_id, {})[
            result.channel
        ] = {
            "settings": asdict(result.settings),
            "segmentation_id": result.segmentation_id,
            "backend": result.backend_version,
        }
        return replace(result, track_ids=ids)

    return transact(dataset, commit, expected=result.expected if expected is None else expected)
