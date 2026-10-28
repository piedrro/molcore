"""Small public model services shared by the Dataset facade."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any, Self

from ._locking import model_mutated
from .revisions import ChangeSet, DependencyToken

logger = logging.getLogger(__name__)
_UNSET = object()


class DatasetServices:
    """Subscriptions, explicit snapshots and result application for datasets."""

    def subscribe(self, callback: Callable[[ChangeSet], None]) -> Callable[[], None]:
        """Receive successful changes on the committing thread.

        Args:
            callback: Listener; exceptions are logged and do not undo commits.

        Returns:
            Idempotent function removing this subscription.
        """
        with self._lock:
            key = object()
            self._subscribers[key] = callback

        def unsubscribe() -> None:
            """Remove this subscription without affecting other listeners."""
            with self._lock:
                self._subscribers.pop(key, None)

        return unsubscribe

    def _record_change(self, components: frozenset[str], partitions: frozenset[tuple[str, str]]) -> ChangeSet:
        self.revision += 1
        for component in components:
            for acquisition, channel in partitions:
                key = component, acquisition, channel
                self._partition_revisions[key] = self._partition_revisions.get(key, 0) + 1
        return ChangeSet(self.revision, components, partitions)

    def _publish_change(self, change: ChangeSet) -> None:
        with self._lock:
            callbacks = tuple(self._subscribers.values())
        for callback in callbacks:
            try:
                callback(change)
            except Exception:
                logger.exception(
                    "Model change subscriber failed after revision %s committed", change.revision
                )

    def snapshot(self) -> Self:
        """Detach scientific state while borrowing immutable image sources.

        Returns:
            Independent model preserving the originating model's lineage.
        """
        from .staging import clone_dataset

        with self._lock:
            return clone_dataset(self)

    def dependencies(
        self,
        acquisition_id: str | None = None,
        channel: str | None = None,
        *,
        components: Iterable[str] = ("localizations", "tracks", "transforms", "segmentations"),
    ) -> DependencyToken:
        """Capture only revisions relevant to a scientific operation.

        Args:
            acquisition_id: One acquisition, or all currently registered acquisitions.
            channel: One logical channel, or all channels in the selected acquisitions.
            components: Result stores read by the operation; image bindings are always included.

        Returns:
            Named input dependency token validated by result application.
        """
        from .staging import dependency_token

        with self._lock:
            acquisitions = (
                self.acquisitions
                if acquisition_id is None
                else {acquisition_id: self.acquisitions[acquisition_id]}
            )
            selected = []
            for key, acquisition in acquisitions.items():
                channels = (
                    acquisition.channels if channel is None else {channel: acquisition.channels[channel]}
                )
                selected.extend((key, c) for c in channels)
            return dependency_token(self, selected, components=components)

    def apply[T](self, result: T) -> T:
        """Atomically apply one self-contained detached processing result.

        Args:
            result: Supported localisation, drift, alignment, tracking, photometry or segmentation result.

        Returns:
            Applied result including allocated stable identifiers where appropriate.
        """
        from .processing import apply_result

        return apply_result(self, result)

    def apply_all(self, results: Iterable[Any]) -> tuple[Any, ...]:
        """Apply independent results in one transaction or preserve all live state.

        Args:
            results: Detached results; earlier writes must not invalidate later inputs.

        Returns:
            Applied results in input order.
        """
        from .processing import apply_results

        return apply_results(self, results)

    @model_mutated
    def update_metadata(self, values: Mapping[str, Any], *, replace: bool = False) -> None:
        """Validate and copy JSON metadata before publishing one change.

        Args:
            values: Metadata entries to merge, or the full replacement mapping.
            replace: Replace rather than merge the existing metadata.
        """
        from .storage import _json

        updated = deepcopy(dict(values) if replace else {**self.metadata, **values})
        _json(updated)
        self.metadata.clear()
        self.metadata.update(updated)
        self._metadata_revision += 1

    @model_mutated
    def update_acquisition(
        self,
        acquisition_id: str,
        *,
        pixel_size: Any = _UNSET,
        timestamps: Any = _UNSET,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate acquisition calibration/timing changes and invalidate derived measurements.

        Args:
            acquisition_id: Existing acquisition to update.
            pixel_size: Nanometres per pixel; None clears the scale, omission retains it.
            timestamps: One finite time per camera frame; None clears timing, omission retains it.
            metadata: JSON metadata entries to merge into acquisition metadata.
        """
        from .storage import _json
        from .types import Acquisition, AcquisitionSequence

        old = self.acquisitions[acquisition_id]
        updated_metadata = deepcopy({**old.metadata, **(metadata or {})})
        _json(updated_metadata)
        replacement = Acquisition(
            old.source,
            id=old.id,
            channel_specs=old.channel_specs,
            excitation_sequence=old.excitation_sequence,
            phase=old.phase,
            mode=old.mode,
            pixel_size=old.pixel_size if pixel_size is _UNSET else pixel_size,
            timestamps=old.timestamps if timestamps is _UNSET else timestamps,
            metadata=updated_metadata,
        )
        sequences = {}
        for key, sequence in self.sequences.items():
            if acquisition_id in sequence.offsets:
                sequences[key] = AcquisitionSequence(
                    [replacement if a.id == acquisition_id else a for a in sequence.acquisitions],
                    id=key,
                    allow_cross_boundary_tracks=sequence.allow_cross_boundary_tracks,
                    reference=sequence.reference,
                )
        self._acquisitions[acquisition_id] = replacement
        self._sequences.update(sequences)
        self.observations._invalidate_partition(acquisition_id)
        self.photon_sequences._invalidate(acquisition_id=acquisition_id)
        self.crops.invalidate(self.localizations.query(acquisition_id=acquisition_id, space="raw")["id"])

    @model_mutated
    def remove_segmentation(self, segmentation_id: str) -> None:
        """Remove a mask and invalidate observations derived from its partition.

        Args:
            segmentation_id: Existing mask identifier.
        """
        segmentation = self._segmentations.pop(segmentation_id)
        self.observations._invalidate_partition(segmentation.acquisition_id)
