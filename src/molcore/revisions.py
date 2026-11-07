"""Named freshness contracts and structured model changes, independent of GUI code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Dataset


@dataclass(frozen=True)
class ChangeSet:
    """One successful model mutation or atomic transaction.

    Subscribers run on the committing thread; GUI adapters must marshal to their thread.
    """

    revision: int
    components: frozenset[str]
    partitions: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class StateToken:
    """Complete runtime model state for optimistic transaction conflict detection."""

    sources: tuple
    observations: int
    calibrations: tuple
    segmentations: tuple
    metadata: str
    processing: str
    acquisitions: tuple
    sequences: tuple
    photon_sequences: int
    lineage: str
    segmentation_outlines: tuple = ()
    bactfit_cells: tuple = ()
    bounding_boxes: int = 0

    @property
    def scientific_inputs(self) -> tuple:
        """Inputs excluding independently accumulated measurements and provenance.

        Returns:
            Named scientific state projected for atomic batch validation.
        """
        return (
            self.lineage,
            self.sources,
            self.calibrations,
            self.segmentations,
            self.segmentation_outlines,
            self.bactfit_cells,
            self.bounding_boxes,
            self.metadata,
            self.acquisitions,
            self.sequences,
        )


@dataclass(frozen=True)
class DependencyToken:
    """Revisions of explicit input partitions, plus their acquisition bindings."""

    lineage: str
    partitions: tuple[tuple[str, str], ...]
    components: frozenset[str]
    revisions: tuple[tuple[str, str, str, int], ...]
    bindings: tuple
    metadata: str

    def validate(self, dataset: Dataset) -> None:
        """Reject changed inputs or a result from another model lineage.

        Args:
            dataset: Live model or a snapshot of the original model.
        """
        from .staging import dependency_token
        from .types import StaleResultError

        try:
            current = dependency_token(
                dataset, self.partitions, components=self.components
            )
        except (KeyError, ValueError) as exc:
            raise StaleResultError("Result input partition no longer exists") from exc
        if current != self:
            raise StaleResultError(
                "Result inputs changed; compute a new preview before applying"
            )


FreshnessToken = StateToken | DependencyToken
