"""Static consumer contract, checked by mypy rather than unittest discovery."""

from typing import assert_type

from molcore import (
    ApplyReceipt,
    Dataset,
    DatasetSnapshot,
    ExecutionContext,
    LocalisationResult,
    LocalizationBatch,
)
from molcore.backends import BackendRegistry, TrackingBackend, TrackingOutput, TrackingRequest
from molcore.execution import Executor


def consume(
    dataset: Dataset, result: LocalisationResult, executor: Executor, backend: TrackingBackend
) -> None:
    assert_type(dataset.apply(result), ApplyReceipt)
    assert_type(dataset.localizations.query(), LocalizationBatch)
    assert_type(dataset.snapshot(), DatasetSnapshot)
    assert_type(dataset.snapshot().localizations.query(), LocalizationBatch)
    context = ExecutionContext(cpu=executor)
    context.check_cancelled()
    registry = BackendRegistry()
    registry.register("track", "custom", backend)
    assert_type(registry.resolve("track", "custom"), TrackingBackend)


def typed_dispatch(executor: Executor, backend: TrackingBackend, jobs: list[TrackingRequest]) -> None:
    assert_type(executor.map(backend.compute, jobs), list[TrackingOutput])
