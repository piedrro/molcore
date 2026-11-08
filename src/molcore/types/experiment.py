"""A sequencing experiment grouping independent probing-round datasets."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..snapshots import ExperimentSnapshot
from .._locking import model_locked
from ._shared import Calibration
from .dataset import Dataset


class Experiment:
    """Own probing rounds and their spatial relationships on one surface.

    Dataset-level scientific APIs operate on individual entries in ``datasets``.
    Round order, oligo names and barcode information can be recorded in dataset
    metadata. Spatial alignment does not join timelines or establish molecule identity.
    """

    def __init__(self, *, metadata: dict[str, Any] | None = None) -> None:
        self._lock = RLock()
        from ..immutable import freeze_metadata

        self._metadata = freeze_metadata(metadata or {})
        self._captured = False
        self._datasets: dict[str, Dataset] = {}
        from ..dataset_alignment import DatasetAlignmentManager

        self.alignment = DatasetAlignmentManager(self)

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Immutable experiment metadata.

        Returns:
            Current experiment annotations.
        """
        return self._metadata

    @model_locked
    def update_metadata(self, values: Mapping[str, Any], *, replace: bool = False) -> None:
        """Validate and replace experiment annotations.

        Args:
            values: Metadata to merge, or all metadata when replacing.
            replace: Replace all existing annotations.
        """
        from ..immutable import freeze_metadata
        from ..storage import _json

        updated = dict(values) if replace else {**self.metadata, **values}
        _json(updated)
        self._metadata = freeze_metadata(updated)

    def snapshot(self) -> ExperimentSnapshot:
        """Capture all datasets and alignment state without copying pixel/table buffers.

        Returns:
            A read-only experiment snapshot.
        """
        from ..snapshots import ExperimentSnapshot
        from ..staging import clone_experiment

        return ExperimentSnapshot(clone_experiment(self))

    @property
    def datasets(self) -> Mapping[str, Dataset]:
        """Read-only registration mapping; dataset contents remain editable.

        Returns:
            Registered datasets keyed by stable identifier.
        """
        return MappingProxyType(self._datasets)

    @model_locked
    def add_dataset(self, dataset: Dataset) -> Dataset:
        """Register a dataset without replacing another round.

        Args:
            dataset: Scientific state of one probing round.

        Returns:
            Result of this operation.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("Expected a Dataset")
        if dataset.id in self._datasets:
            raise ValueError("Dataset ID already exists")
        self._datasets[dataset.id] = dataset
        return dataset

    @model_locked
    def remove_dataset(self, id: str, *, cascade: bool = False) -> Dataset:
        """Detach a round, explicitly clearing alignment dependents if requested.

        The returned dataset retains its image sources and can be used independently.

        Args:
            id: Stable dataset identifier.
            cascade: Explicitly clear dependent registrations when removing a round.

        Returns:
            Result of this operation.
        """
        if id not in self._datasets:
            raise ValueError("Unknown dataset")
        dependent = id in self.alignment.records or id == self.alignment.reference_dataset_id
        if dependent and (not cascade):
            raise ValueError("Dataset has alignment dependents; pass cascade=True")
        if id == self.alignment.reference_dataset_id:
            self.alignment.reset()
        else:
            self.alignment.reset(id)
        return self._datasets.pop(id)

    def save(self, path: str | Path, **options: Any) -> None:
        """Atomically save all probing rounds and alignment records.

        Args:
            path: Snapshot file or output directory.
        """
        from ..storage import save_experiment

        save_experiment(self, path, **options)

    def apply_channel_calibration(
        self, calibration: Calibration, *, dataset_ids: list[str] | None = None, emission: str = "A"
    ) -> None:
        """Select one optical calibration across compatible rounds, atomically.

        Args:
            calibration: Named emission-path registration with optional geometry provenance.
            dataset_ids: Selected rounds; None includes every registered dataset.
            emission: Physical source emission path, normally A for acceptor.
        """
        from ..staging import adopt_dataset, clone_dataset

        changes = []
        with self._lock, ExitStack() as stack:
            keys = list(self.datasets) if dataset_ids is None else list(dataset_ids)
            if not keys or len(set(keys)) != len(keys):
                raise ValueError("Choose one or more distinct datasets")
            for key in sorted(keys):
                stack.enter_context(self.datasets[key]._lock)
            staged, matched = ({}, 0)
            for key in keys:
                live = self.datasets[key]
                model = clone_dataset(live)
                selections = [
                    (
                        a.id,
                        next((c for c, spec in a.channel_specs.items() if spec.emission == emission), None),
                    )
                    for a in model.acquisitions.values()
                ]
                selections = [(a, c) for a, c in selections if c is not None]
                if not selections:
                    continue
                if calibration.name in model.calibrations:
                    import numpy as np

                    if not np.array_equal(model.calibrations[calibration.name].matrix, calibration.matrix):
                        raise ValueError("Calibration name already exists with a different matrix")
                model.transforms.add_calibration(deepcopy(calibration))
                for acquisition_id, channel in selections:
                    model.transforms.select(acquisition_id, channel, calibration.name)
                    model._processing.settings.setdefault("optical_alignment", {})[acquisition_id] = (
                        calibration.name
                    )
                    matched += 1
                staged[key] = model
            if not matched:
                raise ValueError("No acquisitions have the selected physical emission path")
            for key, model in staged.items():
                live = self.datasets[key]
                change = adopt_dataset(live, model)
                if change is not None:
                    changes.append((live, change))
        for live, change in changes:
            live._publish_change(change)

    @classmethod
    def open(cls, path: str | Path, **options: Any) -> Experiment:
        """Open an experiment with eager results and lazy image/crop sources.

        Args:
            path: Snapshot file or output directory.

        Returns:
            Result of this operation.
        """
        from ..storage import open_experiment

        return open_experiment(path, **options)

    def close(self) -> None:
        """Close sources belonging to all registered rounds."""
        for dataset in self._datasets.values():
            dataset.close()

    def __enter__(self) -> Experiment:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
