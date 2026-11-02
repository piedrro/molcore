"""Static consumer contract, checked by mypy rather than unittest discovery."""

from typing import assert_type

from molcore import Dataset, ExecutionContext, LocalisationResult, LocalizationBatch
from molcore.backends import BackendRegistry, ProcessorBackend
from molcore.execution import Executor


def consume(
    dataset: Dataset,
    result: LocalisationResult,
    executor: Executor,
    backend: ProcessorBackend,
) -> None:
    assert_type(dataset.apply(result), LocalisationResult)
    assert_type(dataset.localizations.query(), LocalizationBatch)
    context = ExecutionContext(cpu=executor)
    context.check_cancelled()
    registry = BackendRegistry()
    registry.register("localise", "custom", backend)
    assert_type(registry.resolve("localise", "custom"), ProcessorBackend)
