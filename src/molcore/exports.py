"""Explicit, streaming image exports; scientific snapshots retain raw pixels."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tifffile

from .files import atomic_files, report
from .storage import _json

if TYPE_CHECKING:
    from .types import Dataset, Experiment


def export_dataset_images(
    dataset: Dataset,
    path: str | Path,
    *,
    space: str = "corrected",
    experiment: Experiment | None = None,
    progress: Any = None,
    cancel_event: Any = None,
    publish_guard: Any = None,
) -> tuple[Path, ...]:
    """Stream every channel to TIFF, baking only explicitly selected transforms.

    ``space`` may be raw, registered, corrected, or aligned. Aligned exports
    require the containing experiment and include its round-to-surface mapping.
    The manifest retains original camera frames, channels and excitation metadata.
    All output files replace together or roll back, including cancellation.

    Args:
        dataset: Scientific state of one probing round.
        path: Snapshot file or output directory.
        space: Input or output coordinate space selected explicitly by the caller.
        experiment: Experiment containing the probing rounds.
        progress: Optional structured progress callback.
        cancel_event: Optional cooperative cancellation event.

    Returns:
        Result of this operation.
    """
    from .staging import experiment_state_token, state_token
    from .types import CoordinateSpace, StaleResultError

    if space == "aligned":
        if experiment is None or experiment.datasets.get(dataset.id) is not dataset:
            raise ValueError(
                "Aligned export requires the experiment containing this dataset"
            )
        expected = experiment_state_token(experiment)
    else:
        space = CoordinateSpace(space).value
        expected = state_token(dataset)
    path = Path(path).resolve()
    entries, sources, outputs = [], [], []
    for index, acquisition in enumerate(dataset.acquisitions.values()):
        for number, (channel, source) in enumerate(acquisition.channels.items()):
            safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(channel))[:50]
            relative = Path(f"acquisition-{index:04d}") / f"{number:04d}-{safe}.tif"
            entries.append(
                dict(
                    acquisition_id=acquisition.id,
                    channel=channel,
                    path=relative.as_posix() if len(source) else None,
                    camera_frames=source.source_frames,
                    channel_spec=vars(acquisition.channel_specs[channel]),
                    excitation_sequence=acquisition.excitation_sequence,
                    phase=acquisition.phase,
                    timestamps=(
                        None
                        if acquisition.timestamps is None
                        else acquisition.timestamps[source.source_frames]
                    ),
                    pixel_size=acquisition.pixel_size,
                )
            )
            if len(source):
                view = (
                    experiment.alignment.view(dataset.id, acquisition.id, channel)
                    if space == "aligned"
                    else dataset.transforms.view(acquisition.id, channel, space=space)
                )
                sources.append(view)
                outputs.append(path / relative)
    total = sum(len(source) for source in sources)
    if not total:
        raise ValueError("Dataset has no image frames to export")
    completed = 0
    report(progress, cancel_event, 0, total, "Exporting dataset images")
    with atomic_files(
        [*outputs, path / "manifest.json"], publish_guard=publish_guard
    ) as temporary:
        for destination, source in zip(temporary[:-1], sources, strict=True):
            with tifffile.TiffWriter(destination, bigtiff=True) as writer:
                for frame in range(len(source)):
                    report(
                        progress,
                        cancel_event,
                        completed,
                        total,
                        "Exporting dataset images",
                    )
                    writer.write(
                        source[frame],
                        contiguous=True,
                        photometric="minisblack",
                        metadata=None,
                    )
                    completed += 1
        manifest = dict(
            dataset_id=dataset.id,
            name=dataset.name,
            space=space,
            metadata=dataset.metadata,
            channels=entries,
            calibrations=[asdict(c) for c in dataset.calibrations.values()],
            selected_calibrations=[
                [a, c, name] for (a, c), name in dataset.transforms._selected.items()
            ],
            drift={
                key: dict(frames=frames, displacement=displacement)
                for key, (frames, displacement) in dataset.transforms._drift.items()
            },
            dataset_alignment=(
                asdict(experiment.alignment.records[dataset.id])
                if space == "aligned"
                and dataset.id != experiment.alignment.reference_dataset_id
                else None
            ),
        )
        temporary[-1].write_text(_json(manifest), encoding="utf-8")
        current = (
            experiment_state_token(experiment)
            if space == "aligned"
            else state_token(dataset)
        )
        if current != expected:
            raise StaleResultError(
                "Dataset changed during export; output was discarded"
            )
        report(progress, cancel_event, total, total, "Dataset images exported")
    return tuple(outputs)
