"""Persistent measurement regions and their revisioned identity."""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from typing import TYPE_CHECKING, Any
from uuid import uuid4
from datetime import datetime, timezone

from .._locking import model_locked, model_mutated

if TYPE_CHECKING:
    from .dataset import Dataset


@dataclass(frozen=True)
class BoundingBox:
    """One stationary measurement region in explicitly named pixel coordinates.

    Width and height describe the defining box. Signal apertures and background
    settings belong to an extraction run and can differ from this geometry.
    Raw boxes bind an acquisition/channel. Working boxes may be reused across
    aligned acquisitions and require explicitly supplied working pixels.
    """

    x: float
    y: float
    width: float
    height: float
    id: str = field(default_factory=lambda: str(uuid4()))
    acquisition_id: str | None = None
    channel: str | None = None
    space: str = "raw"
    reference: str | None = None
    revision: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            value = float(getattr(self, name))
            if not isfinite(value) or (name in ("width", "height") and value <= 0):
                raise ValueError("Box coordinates must be finite and dimensions positive")
            object.__setattr__(self, name, value)
        for name in ("id", "acquisition_id", "channel", "reference"):
            value = getattr(self, name)
            if (name == "id" or value is not None) and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Box {name} must be a nonempty string")
        if (self.acquisition_id is None) != (self.channel is None):
            raise ValueError("Supply a box acquisition and channel together")
        if self.space not in ("raw", "working"):
            raise ValueError("Box space must be raw or working")
        if self.space == "raw" and self.acquisition_id is None:
            raise ValueError("Raw boxes require an acquisition and channel")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("Box revision must be a nonnegative integer")
        metadata = deepcopy(dict(self.metadata))
        json.dumps(metadata, allow_nan=False)
        object.__setattr__(self, "metadata", metadata)
        for value in (self.created_at, self.updated_at):
            if value is not None:
                if not isinstance(value, str) or datetime.fromisoformat(value).tzinfo is None:
                    raise ValueError("Box timestamps must be ISO timestamps with a timezone")


class BoundingBoxStore:
    """Own editable boxes and invalidate sequences that depend on their revisions."""

    def __init__(self, dataset: Dataset) -> None:
        """Bind a box collection to its scientific model.

        Args:
            dataset: Owner of box definitions and dependent photon sequences.
        """
        self.dataset = dataset
        self._data: dict[str, BoundingBox] = {}
        self._deleted: set[str] = set()
        self.revision = 0

    @model_locked
    def __getitem__(self, id: str) -> BoundingBox:
        """Read an active definition as a detached record.

        Args:
            id: Stable box identifier.

        Returns:
            Definition with independently editable metadata.
        """
        if id in self._deleted:
            raise KeyError(id)
        return replace(self._data[id])

    @model_locked
    def __iter__(self) -> Iterator[str]:
        """Iterate active box IDs in insertion order.

        Returns:
            Iterator over stable active identifiers.
        """
        return iter(tuple(key for key in self._data if key not in self._deleted))

    @model_locked
    def __len__(self) -> int:
        """Count active boxes.

        Returns:
            Number of definitions available for extraction.
        """
        return len(self._data) - len(self._deleted)

    @model_locked
    def query(self, *, include_deleted: bool = False) -> tuple[BoundingBox, ...]:
        """Read detached definitions, optionally including deletion history.

        Args:
            include_deleted: Include the final revision of deleted definitions.

        Returns:
            Definitions in insertion order.
        """
        return tuple(
            replace(box) for key, box in self._data.items() if include_deleted or key not in self._deleted
        )

    @model_mutated
    def set(self, box: BoundingBox) -> BoundingBox:
        """Create or edit a definition, preserving identity and advancing its revision.

        Args:
            box: Validated geometry and metadata. The store manages its revision.

        Returns:
            Stored definition with its authoritative revision.
        """
        if not isinstance(box, BoundingBox):
            raise TypeError("Expected a BoundingBox")
        if box.acquisition_id is not None:
            self.dataset.acquisitions[box.acquisition_id].channels[box.channel]
        previous = self._data.get(box.id)
        if previous is not None and box.id not in self._deleted:
            before, after = asdict(previous), asdict(box)
            for key in ("revision", "created_at", "updated_at"):
                before.pop(key)
                after.pop(key)
            if before == after:
                return replace(previous)
        revision = 1 if previous is None else previous.revision + 1
        now = datetime.now(timezone.utc).isoformat()
        stored = replace(
            box,
            revision=revision,
            created_at=previous.created_at if previous is not None else box.created_at or now,
            updated_at=now,
        )
        self._data[box.id] = stored
        self._deleted.discard(box.id)
        self.dataset.photon_sequences._invalidate_bounding_boxes({box.id})
        self.revision += 1
        return replace(stored)

    @model_mutated
    def remove(self, id: str) -> None:
        """Delete an active definition while preserving its history and derived samples.

        Args:
            id: Stable box identifier whose dependent sequences become stale.
        """
        box = self[id]
        self._data[id] = replace(
            box, revision=box.revision + 1, updated_at=datetime.now(timezone.utc).isoformat()
        )
        self._deleted.add(id)
        self.dataset.photon_sequences._invalidate_bounding_boxes({id})
        self.revision += 1
