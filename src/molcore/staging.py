"""Detached model snapshots and atomic result transactions without GUI imports."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict

from ._locking import model_locked
from ._logging import logged
from .revisions import ChangeSet, DependencyToken, FreshnessToken, StateToken
from .storage import _json
from .types import Acquisition, Dataset, Experiment, StaleResultError


@model_locked
@logged
def state_token(dataset: Dataset) -> StateToken:
    """Capture revisions and bindings without reading movie pixels.

    Args:
        dataset: Live model to protect from stale background work.

    Returns:
        Comparable token including result, source, calibration and mask bindings.
    """
    return StateToken(
        tuple(
            (
                a,
                dataset.localizations.revision,
                dataset.tracks.revision,
                dataset.transforms.revision,
                id(acquisition.source),
                tuple(
                    (name, view.roi, view.shape, view.source_frames.tobytes())
                    for name, view in acquisition.channels.items()
                ),
            )
            for a, acquisition in dataset.acquisitions.items()
        ),
        dataset.observations.revision,
        tuple(
            (k, v.matrix.tobytes(), _json(v.metadata))
            for k, v in dataset.calibrations.items()
        ),
        tuple(
            (
                k,
                s.acquisition_id,
                s.channel,
                s.frame,
                s.space,
                s.labels.dtype.str,
                s.labels.shape,
                s.labels.tobytes(),
                _json(s.metadata),
                s.source_path,
                s.mask_path,
                (
                    None
                    if s.source_image is None
                    else (
                        s.source_image.dtype.str,
                        s.source_image.shape,
                        s.source_image.tobytes(),
                    )
                ),
            )
            for k, s in dataset.segmentations.items()
        ),
        _json(dataset.metadata),
        _json(vars(dataset.processing)),
        tuple(
            (
                k,
                _json(a.metadata),
                a.pixel_size,
                _json({c: asdict(spec) for c, spec in a.channel_specs.items()}),
                a.mode,
                a.excitation_sequence,
                a.phase,
                None if a.timestamps is None else a.timestamps.tobytes(),
            )
            for k, a in dataset.acquisitions.items()
        ),
        tuple(
            (
                k,
                tuple(a.id for a in s.acquisitions),
                s.allow_cross_boundary_tracks,
                s.reference,
            )
            for k, s in dataset.sequences.items()
        ),
        dataset.photon_sequences.revision,
        dataset._lineage,
        tuple(dataset.segmentation_outlines.values()),
        tuple(dataset.bactfit_cells.values()),
    )


@model_locked
def dependency_token(
    dataset: Dataset,
    partitions: Iterable[tuple[str, str]],
    *,
    components: Iterable[str],
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
        raise ValueError(
            f"Unknown dependency components: {sorted(components - allowed)}"
        )
    bindings = []
    for acquisition_id, channel in selected:
        acquisition = dataset.acquisitions[acquisition_id]
        view = acquisition.channels[channel]
        bindings.append(
            (
                acquisition_id,
                channel,
                id(acquisition.source),
                view.shape,
                view.roi,
                view.source_frames.tobytes(),
                acquisition.pixel_size,
                acquisition.mode,
                acquisition.excitation_sequence,
                acquisition.phase,
                _json(asdict(acquisition.channel_specs[channel])),
                _json(acquisition.metadata),
                (
                    None
                    if acquisition.timestamps is None
                    else acquisition.timestamps.tobytes()
                ),
            )
        )
    sequences = tuple(
        (
            key,
            tuple(a.id for a in value.acquisitions),
            value.reference,
            value.allow_cross_boundary_tracks,
        )
        for key, value in dataset.sequences.items()
        if any(a in value.offsets for a, _ in selected)
    )
    revisions = tuple(
        (component, a, c, dataset._partition_revisions.get((component, a, c), 0))
        for component in sorted(components)
        for a, c in selected
    )
    return DependencyToken(
        dataset._lineage,
        selected,
        components,
        revisions,
        (tuple(bindings), sequences),
        _json(dataset.metadata),
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
    result = Dataset(
        id=dataset.id, name=dataset.name, metadata=deepcopy(dataset.metadata)
    )
    for acquisition in dataset.acquisitions.values():
        result.add_acquisition(
            Acquisition(
                acquisition.source,
                id=acquisition.id,
                channel_specs=deepcopy(acquisition.channel_specs),
                excitation_sequence=acquisition.excitation_sequence,
                phase=acquisition.phase,
                timestamps=acquisition.timestamps,
                pixel_size=acquisition.pixel_size,
                metadata=deepcopy(acquisition.metadata),
                mode=acquisition.mode,
            )
        )
    for sequence in dataset.sequences.values():
        result.add_sequence(
            [a.id for a in sequence.acquisitions],
            id=sequence.id,
            allow_cross_boundary_tracks=sequence.allow_cross_boundary_tracks,
            reference=sequence.reference,
        )
    memo = {id(dataset): result}
    for name in (
        "localizations",
        "tracks",
        "observations",
        "photon_sequences",
        "transforms",
        "crops",
        "processing",
    ):
        setattr(result, name, deepcopy(getattr(dataset, name), memo))
    result.calibrations = result.transforms.calibrations
    result._segmentations.update(deepcopy(dict(dataset.segmentations), memo))
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
            tuple(
                (key, id(dataset), state_token(dataset))
                for key, dataset in experiment.datasets.items()
            ),
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
        for name in (
            "reference_dataset_id",
            "reference_acquisition_id",
            "reference_channel",
            "revision",
        ):
            setattr(result.alignment, name, getattr(experiment.alignment, name))
    return result


def _scientific_inputs(token: StateToken) -> tuple:
    # Observations, photon sequences and provenance from independent processors are outputs.
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
            if (
                batch_source != expected
                or before.scientific_inputs != expected.scientific_inputs
            ):
                raise StaleResultError(
                    "Inputs changed; compute a new preview before applying"
                )


def adopt_dataset(dataset: Dataset, staged: Dataset) -> ChangeSet | None:
    """Adopt validated staged state while preserving public store identities.

    Args:
        dataset: Locked live model receiving the state.
        staged: Independently validated model in the same lineage.

    Returns:
        One change to publish after the commit lock is released, or None for no change.
    """
    before, after = state_token(dataset), state_token(staged)
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
        ("calibrations", "calibrations"),
        ("acquisitions", "acquisitions"),
        ("sequences", "sequences"),
    ):
        if getattr(before, field) != getattr(after, field):
            components.add(component)
    if before != after and not components:
        components.update(
            (
                "localizations",
                "tracks",
                "observations",
                "photon_sequences",
                "transforms",
            )
        )
    if components and not partitions:
        partitions = {
            (a, c) for a, value in dataset.acquisitions.items() for c in value.channels
        }
    for name in (
        "localizations",
        "tracks",
        "observations",
        "photon_sequences",
        "transforms",
        "crops",
    ):
        target, source = getattr(dataset, name), getattr(staged, name)
        values = vars(source).copy()
        values["dataset"] = dataset
        if name == "transforms":
            target._calibrations.clear()
            target._calibrations.update(source._calibrations)
            values["_calibrations"] = target._calibrations
        vars(target).clear()
        vars(target).update(values)
    for name in (
        "_acquisitions",
        "_sequences",
        "_segmentations",
        "_segmentation_outlines",
        "_bactfit_cells",
    ):
        target = getattr(dataset, name)
        target.clear()
        target.update(getattr(staged, name))
    dataset.metadata.clear()
    dataset.metadata.update(staged.metadata)
    dataset.processing.settings.clear()
    dataset.processing.settings.update(staged.processing.settings)
    dataset.processing.software.clear()
    dataset.processing.software.update(staged.processing.software)
    dataset._metadata_revision = staged._metadata_revision
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
    with dataset._lock:
        before = state_token(dataset)
        validate_expected(dataset, expected)
        staged = clone_dataset(dataset)
        staged._transaction_source_token = before
    value = operation(staged)
    with dataset._lock:
        if state_token(dataset) != before:
            raise StaleResultError(
                "Live state changed during transaction; staged update was discarded"
            )
        change = adopt_dataset(dataset, staged)
    if change is not None:
        dataset._publish_change(change)
    return value
