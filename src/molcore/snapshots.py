"""Read-only scientific views backed by captured immutable model buffers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

from .revisions import DependencyToken
from .types.batches import LocalizationBatch, ObservationBatch, TrackMembershipBatch

if TYPE_CHECKING:
    from .types import (
        Acquisition,
        AcquisitionSequence,
        Calibration,
        Dataset,
        ProcessingMetadata,
        Segmentation,
    )
from typing import Any

from .immutable import ImmutableObject


class _ReadStore[BatchT](ImmutableObject):
    _read_methods = frozenset(
        {
            "query",
            "iter_batches",
            "get",
            "is_valid",
            "valid",
            "coordinates",
            "matrix",
            "between",
            "image",
            "frame_info",
            "camera_frames",
            "resolve",
            "sample",
            "localization",
            "track",
            "photon",
            "photon_sequence",
        }
    )

    def __init__(self, store: Any) -> None:
        self._store = store
        self._seal()

    def __getattr__(self, name: str) -> Any:
        if name in self._read_methods or name == "revision":
            return getattr(self._store, name)
        raise AttributeError(name)

    def __len__(self) -> Any:
        return len(self._store)

    def __iter__(self) -> Any:
        return iter(self._store)

    def __getitem__(self, key: Any) -> Any:
        return self._store[key]

    def query(self, **selectors: Any) -> BatchT:
        """Query captured state using the store's typed batch interface.

        Args:
            selectors: Named selectors accepted by the underlying store query.

        Returns:
            Immutable query output.
        """
        return self._store.query(**selectors)

    def iter_batches(
        self, *, batch_size: int = 65536, columns: tuple[str, ...] | None = None, **selectors: Any
    ) -> Iterator[BatchT]:
        """Iterate immutable batches from captured state.

        Args:
            batch_size: Maximum rows per output batch.
            columns: Optional ordered column projection.
            selectors: Named selectors accepted by the underlying store query.

        Returns:
            Iterator of immutable batches.
        """
        return self._store.iter_batches(batch_size=batch_size, columns=columns, **selectors)


class DatasetSnapshot(ImmutableObject):
    """Captured dataset state; query-only stores and borrowed lazy image sources.

    Private implementation attributes are not part of the read-only API.
    """

    localizations: _ReadStore[LocalizationBatch]
    observations: _ReadStore[ObservationBatch]
    tracks: _ReadStore[TrackMembershipBatch]
    metadata: Mapping[str, Any]
    acquisitions: Mapping[str, Acquisition]
    sequences: Mapping[str, AcquisitionSequence]
    segmentations: Mapping[str, Segmentation]
    calibrations: Mapping[str, Calibration]
    processing: ProcessingMetadata

    def __init__(self, model: Dataset) -> None:
        model._captured = True
        self._model = model
        self.id, self.name, self.revision = (model.id, model.name, model.revision)
        self.processing = model.processing
        for name in (
            "metadata",
            "acquisitions",
            "sequences",
            "segmentations",
            "segmentation_outlines",
            "bactfit_cells",
            "calibrations",
        ):
            setattr(self, name, getattr(model, name))
        for name in (
            "localizations",
            "observations",
            "tracks",
            "photon_sequences",
            "bounding_boxes",
            "transforms",
            "crops",
        ):
            setattr(self, name, _ReadStore(getattr(model, name)))
        self._seal()

    def dependencies(
        self,
        acquisition_id: str | None = None,
        channel: str | None = None,
        *,
        components: Iterable[str] = ("localizations", "tracks", "transforms", "segmentations"),
    ) -> DependencyToken:
        """Dependencies.

        Args:
            args: Args.
            kwargs: Kwargs.


        Returns:
            The computed value.
        """
        return self._model.dependencies(acquisition_id, channel, components=components)

    def segmentation_labels(self, *args: Any, **kwargs: Any) -> Any:
        """Segmentation labels.

        Args:
            args: Args.
            kwargs: Kwargs.


        Returns:
            The computed value.
        """
        return self._model.segmentation_labels(*args, **kwargs)

    def snapshot(self) -> DatasetSnapshot:
        """Snapshot.

        Returns:
            The computed value.
        """
        return self

    def close(self) -> None:
        """Snapshots never close sources borrowed from their live owner."""

    def __enter__(self) -> DatasetSnapshot:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def read_model(value: Any) -> Any:
    """Internal query implementation access; never expose it as a public mutator.

    Args:
        value: Value.


    Returns:
        The computed value.
    """
    return value._model if isinstance(value, DatasetSnapshot) else value


def processor_model(value: Any) -> Any:
    """Require a captured view at a scientific computation boundary.

    Args:
        value: Value.


    Returns:
        The computed value.
    """
    if isinstance(value, DatasetSnapshot):
        return value._model
    if getattr(value, "_captured", False):
        return value
    raise TypeError("Scientific processors require dataset.snapshot()")


class ExperimentSnapshot(ImmutableObject):
    """Read-only datasets and registrations captured from one experiment."""

    alignment: _ReadStore[Any]

    def __init__(self, model: Any) -> None:
        model._captured = True
        self._model = model
        self.metadata = model.metadata
        self.datasets = {key: DatasetSnapshot(value) for key, value in model.datasets.items()}
        self.alignment = _ReadStore(model.alignment)
        self._seal()

    def snapshot(self) -> ExperimentSnapshot:
        """Reuse this already captured view.

        Returns:
            This snapshot.
        """
        return self

    def close(self) -> None:
        """Never close sources borrowed from the experiment owner."""


def processor_experiment(value: Any) -> Any:
    """Access captured experiment state inside a scientific processor.

    Args:
        value: A read-only experiment snapshot.

    Returns:
        Internal captured state.
    """
    if isinstance(value, ExperimentSnapshot):
        return value._model
    if getattr(value, "_captured", False):
        return value
    raise TypeError("Scientific processors require experiment.snapshot()")
