"""Explicit partition and sequence selections for scientific operations."""

from __future__ import annotations

from ..immutable import immutable_record


@immutable_record
class PartitionSelection:
    acquisition_id: str
    channel: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip() for value in (self.acquisition_id, self.channel)
        ):
            raise ValueError("Select a nonempty acquisition and channel")


@immutable_record
class SequenceSelection:
    sequence_id: str
    channel: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.sequence_id, self.channel)):
            raise ValueError("Select a nonempty sequence and channel")
