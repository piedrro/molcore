"""Internal model snapshots, backend boundaries and executor ownership."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from ._logging import logged
from .cpu import CpuExecutor
from .revisions import DependencyToken

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .types import Dataset


@logged
def backend_version() -> str:
    """Backend version.

    Returns:
        Computed str value.
    """
    try:
        installed = version("picassosr")
    except PackageNotFoundError as exc:
        raise ImportError("Picasso is missing; install molcore[localisation]") from exc
    if tuple(int(part) for part in installed.split(".")[:2]) != (0, 11):
        raise ImportError(f"MolCore processing supports Picasso >=0.11,<0.12; found {installed}")
    return installed


@logged
def snapshot(dataset: Dataset, acquisition_id: str) -> DependencyToken:
    """Snapshot.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.

    Returns:
        Computed tuple value.
    """
    return dataset.dependencies(acquisition_id)


@logged
def check_snapshot(dataset: Dataset, acquisition_id: str, expected: DependencyToken) -> None:
    """Check snapshot.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        expected: Source revision token used to reject stale results.
    """
    expected.validate(dataset)


@contextmanager
@logged
def executor(cpu: CpuExecutor | None, num_cpus: int | None) -> Iterator[Any]:
    """Executor.

    Args:
        cpu: Optional borrowed CPU executor; ownership remains with the caller.
        num_cpus: Worker process count; None uses the executor default.

    Yields:
        Values produced by the iterator, retaining their original order.
    """
    if cpu is not None:
        if num_cpus is not None:
            raise ValueError("Set num_cpus on the supplied CpuExecutor, or omit cpu")
        yield cpu
    else:
        with CpuExecutor(num_cpus=num_cpus) as created:
            yield created


@logged
def record(dataset: Dataset, stage: Any, acquisition_id: str, channel: str, values: Any) -> None:
    """Record.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        stage: Stage used by this operation.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        values: Input values to validate, convert or display.
    """
    dataset.processing.settings.setdefault(stage, {}).setdefault(acquisition_id, {})[channel] = values
