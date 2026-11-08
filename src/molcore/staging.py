"""Detached model snapshots and atomic result transactions without GUI imports."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

from ._locking import model_locked
from ._logging import logged
from .revisions import ChangeSet, DependencyToken, FreshnessToken, StateToken
from .storage import _json
from .types import Dataset, Experiment, StaleResultError


@model_locked
@logged
def state_token(dataset: Dataset) -> StateToken:
    """Capture revisions and bindings without reading movie pixels.

    Args:
        dataset: Live model to protect from stale background work.

    Returns:
        Comparable token including result, source, calibration and mask bindings.
    """
    from .snapshots import read_model

    dataset = read_model(dataset)
    return StateToken(
        tuple(
            (
                (
                    a,
                    dataset.localizations.revision,
                    dataset.tracks.revision,
                    dataset.transforms.revision,
                    id(acquisition),
                )
                for a, acquisition in dataset.acquisitions.items()
            )
        ),
        dataset.observations.revision,
        tuple(((k, id(v)) for k, v in dataset.calibrations.items())),
        tuple(((k, id(v)) for k, v in dataset.segmentations.items())),
        dataset._metadata_revision,
        _json(vars(dataset.processing)),
        tuple(((k, id(v)) for k, v in dataset.acquisitions.items())),
        tuple(((k, id(v)) for k, v in dataset.sequences.items())),
        dataset.photon_sequences.revision,
        dataset._lineage,
        tuple(dataset.segmentation_outlines.values()),
        tuple(dataset.bactfit_cells.values()),
        dataset.bounding_boxes.revision,
        revision=dataset.revision,
    )


@model_locked
def dependency_token(
    dataset: Dataset, partitions: Iterable[tuple[str, str]], *, components: Iterable[str]
) -> DependencyToken:
    """Capture scoped revision counters and small acquisition descriptors.

    Args:
        dataset: Owner of the scientific inputs.
        partitions: Explicit acquisition/channel pairs used by the computation.
        components: Result stores read by the computation.

    Returns:
        Token that ignores edits to unrelated partitions.
    """
    selected = tuple(sorted(set(partitions)))
    components = frozenset(components) | {"acquisitions"}
    allowed = {
        "localizations",
        "tracks",
        "observations",
        "photon_sequences",
        "bounding_boxes",
        "transforms",
        "segmentations",
        "segmentation_outlines",
        "bactfit_cells",
        "acquisitions",
        "calibrations",
        "sequences",
        "metadata",
    }
    if components - allowed:
        raise ValueError(f"Unknown dependency components: {sorted(components - allowed)}")
    from .snapshots import read_model

    dataset = read_model(dataset)
    bindings = tuple(
        ((a, c, id(dataset.acquisitions[a]), dataset.acquisitions[a].channels[c].shape) for a, c in selected)
    )
    sequences = tuple(
        ((k, id(v)) for k, v in dataset.sequences.items() if any((a in v.offsets for a, _ in selected)))
    )
    revisions = tuple(
        (
            (component, a, c, dataset._partition_revisions.get((component, a, c), 0))
            for component in sorted(components)
            for a, c in selected
        )
    )
    return DependencyToken(
        dataset._lineage,
        selected,
        components,
        revisions,
        (bindings, sequences),
        dataset._metadata_revision if "metadata" in components else 0,
    )


@model_locked
@logged
def clone_dataset(dataset: Dataset) -> Dataset:
    """Copy scientific state while retaining borrowed lazy image descriptors.

    Args:
        dataset: Model whose result arrays and metadata should be detached.

    Returns:
        Independent model with shared image sources, never a full-movie copy.

    Notes:
        Owners must not modify borrowed image arrays during background jobs.
    """
    from .snapshots import read_model

    dataset = read_model(dataset)
    from threading import RLock
    from types import MappingProxyType

    result = Dataset.__new__(Dataset)
    result.__dict__ = dataset.__dict__.copy()
    result._lock = RLock()
    result._subscribers = {}
    result._mutation_depth = 0
    result._mutation_partitions = set()
    for name in ("acquisitions", "sequences", "segmentations", "segmentation_outlines", "bactfit_cells"):
        values = dict(getattr(dataset, name))
        setattr(result, "_" + name, values)
        setattr(result, name, MappingProxyType(values))
    memo = {id(dataset): result}

    def share(value: Any) -> None:
        """Share.

        Args:
            value: Value.
        """
        if isinstance(value, __import__("numpy").ndarray):
            memo[id(value)] = value
        elif isinstance(value, dict):
            for item in value.values():
                share(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                share(item)

    for name in ("localizations", "observations", "tracks", "transforms", "crops"):
        for value in vars(getattr(dataset, name)).values():
            if value is not dataset:
                share(value)
    for name in (
        "localizations",
        "tracks",
        "observations",
        "photon_sequences",
        "bounding_boxes",
        "transforms",
        "crops",
        "_processing",
    ):
        setattr(result, name, deepcopy(getattr(dataset, name), memo))
    result.calibrations = result.transforms.calibrations
    result._segmentation_outlines.update(dataset.segmentation_outlines)
    result._bactfit_cells.update(dataset.bactfit_cells)
    result._lineage = dataset._lineage
    result.revision = dataset.revision
    result._metadata_revision = dataset._metadata_revision
    result._partition_revisions = dataset._partition_revisions.copy()
    if hasattr(dataset, "_batch_source_token"):
        result._batch_source_token = dataset._batch_source_token
    return result


def experiment_state_token(experiment: Experiment) -> tuple:
    """Runtime freshness token for an entire experiment, without movie reads.

    Args:
        experiment: Experiment containing the probing rounds.

    Returns:
        Result of this operation.
    """
    with experiment._lock:
        return (
            _json(experiment.metadata),
            tuple(((key, id(dataset), state_token(dataset)) for key, dataset in experiment.datasets.items())),
            experiment.alignment.revision,
        )


def clone_experiment(experiment: Experiment) -> Experiment:
    """Detach every dataset and alignment while borrowing lazy image sources.

    Args:
        experiment: Experiment containing the probing rounds.

    Returns:
        Result of this operation.
    """
    with experiment._lock:
        result = Experiment(metadata=deepcopy(experiment.metadata))
        for dataset in experiment.datasets.values():
            with dataset._lock:
                result.add_dataset(clone_dataset(dataset))
        result.alignment._records = deepcopy(experiment.alignment._records)
        for name in ("reference_dataset_id", "reference_acquisition_id", "reference_channel", "revision"):
            setattr(result.alignment, name, getattr(experiment.alignment, name))
    return result


def _scientific_inputs(token: StateToken) -> tuple:
    """Scientific inputs.

    Args:
        token: Token used by this operation.

    Returns:
        Computed tuple value.
    """
    return token.scientific_inputs


def validate_expected(dataset: Dataset, expected: FreshnessToken | None) -> None:
    """Check a named global or scoped result token.

    Args:
        dataset: Model receiving the detached result.
        expected: Captured input dependencies, or None for caller-created results.
    """
    if isinstance(expected, DependencyToken):
        expected.validate(dataset)
    elif expected is not None:
        before = state_token(dataset)
        if before != expected:
            batch_source = getattr(dataset, "_batch_source_token", None)
            if batch_source != expected or before.scientific_inputs != expected.scientific_inputs:
                raise StaleResultError("Inputs changed; compute a new preview before applying")


def adopt_dataset(dataset: Dataset, staged: Dataset) -> ChangeSet | None:
    """Adopt validated staged state while preserving public store identities.

    Args:
        dataset: Locked live model receiving the state.
        staged: Independently validated model in the same lineage.

    Returns:
        One change to publish after the commit lock is released, or None for no change.
    """
    before, after = (state_token(dataset), state_token(staged))
    keys = {
        key
        for key, value in staged._partition_revisions.items()
        if dataset._partition_revisions.get(key, 0) != value
    }
    components = {key[0] for key in keys}
    partitions = {(key[1], key[2]) for key in keys}
    for field, component in (
        ("metadata", "metadata"),
        ("processing", "processing"),
        ("segmentations", "segmentations"),
        ("segmentation_outlines", "segmentation_outlines"),
        ("bactfit_cells", "bactfit_cells"),
        ("bounding_boxes", "bounding_boxes"),
        ("calibrations", "calibrations"),
        ("acquisitions", "acquisitions"),
        ("sequences", "sequences"),
    ):
        if getattr(before, field) != getattr(after, field):
            components.add(component)
    if before != after and (not components):
        components.update(("localizations", "tracks", "observations", "photon_sequences", "transforms"))
    if components and (not partitions):
        partitions = {(a, c) for a, value in dataset.acquisitions.items() for c in value.channels}
    for name in (
        "localizations",
        "tracks",
        "observations",
        "photon_sequences",
        "bounding_boxes",
        "transforms",
        "crops",
    ):
        target, source = (getattr(dataset, name), getattr(staged, name))
        values = vars(source).copy()
        values["dataset"] = dataset
        if name == "transforms":
            target._calibrations.clear()
            target._calibrations.update(source._calibrations)
            values["_calibrations"] = target._calibrations
        vars(target).clear()
        vars(target).update(values)
    for name in ("_acquisitions", "_sequences", "_segmentations", "_segmentation_outlines", "_bactfit_cells"):
        target = getattr(dataset, name)
        target.clear()
        target.update(getattr(staged, name))
    dataset._segmentation_revisions = staged._segmentation_revisions.copy()
    dataset._metadata = staged.metadata
    dataset._name = staged.name
    dataset._processing.settings.clear()
    dataset._processing.settings.update(staged._processing.settings)
    dataset._processing.software.clear()
    dataset._processing.software.update(staged._processing.software)
    dataset._metadata_revision = staged._metadata_revision
    dataset._processing_revision = staged._processing_revision
    dataset._partition_revisions = staged._partition_revisions.copy()
    if components:
        return dataset._record_change(frozenset(components), frozenset(partitions))
    return None


@logged
def transact[R](
    dataset: Dataset,
    operation: Callable[[Dataset], R],
    *,
    expected: FreshnessToken | None = None,
    _finish: Any = None,
) -> R:
    """Stage and atomically adopt a model update, rechecking freshness at commit.

    Model mutators and final adoption share a reentrant lock. Computation occurs
    outside the live lock. Concurrent supported edits are retained and cause the
    staged operation to fail. Direct dictionary/array mutation is not synchronized.
    Batch results may share the batch's original token while scientific inputs
    remain unchanged; independent observations/provenance may accumulate.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        operation: Operation used by this operation.
        expected: Source revision token used to reject stale results.

    Returns:
        Computed R value.
    """
    if getattr(dataset, "_is_staged", False):
        validate_expected(dataset, expected)
        value = operation(dataset)
        return _finish(value, dataset.revision) if _finish is not None else value
    with dataset._lock:
        before = state_token(dataset)
        validate_expected(dataset, expected)
        staged = clone_dataset(dataset)
        staged._transaction_source_token = before
        staged._is_staged = True
    value = operation(staged)
    with dataset._lock:
        if state_token(dataset) != before:
            raise StaleResultError("Live state changed during transaction; staged update was discarded")
        change = adopt_dataset(dataset, staged)
        if _finish is not None:
            value = _finish(value, dataset.revision)
        else:
            value = _finalize_receipts(value, dataset.revision)
    if change is not None:
        dataset._publish_change(change)
    return value


def _finalize_receipts(value: Any, revision: int) -> Any:
    """Bind nested application receipts to the actual outer commit revision."""
    from dataclasses import replace

    from .types.results import ApplyReceipt

    if isinstance(value, ApplyReceipt):
        return replace(value, revision=revision)
    if isinstance(value, list):
        return [_finalize_receipts(item, revision) for item in value]
    if isinstance(value, tuple):
        return tuple(_finalize_receipts(item, revision) for item in value)
    if isinstance(value, dict):
        return {key: _finalize_receipts(item, revision) for key, item in value.items()}
    return value
