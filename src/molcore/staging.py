"""Detached model snapshots and atomic result transactions without GUI imports."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict

from ._locking import model_locked
from ._logging import logged
from .storage import _json
from .types import Acquisition, Dataset, Experiment, StaleResultError


@model_locked
@logged
def state_token(dataset: Dataset) -> tuple:
    """Capture revisions and bindings without reading movie pixels.

    Args:
        dataset: Live model to protect from stale background work.

    Returns:
        Comparable token including result, source, calibration and mask bindings.
    """
    return (
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
        tuple((k, v.matrix.tobytes(), _json(v.metadata)) for k, v in dataset.calibrations.items()),
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
                None
                if s.source_image is None
                else (s.source_image.dtype.str, s.source_image.shape, s.source_image.tobytes()),
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
            (k, tuple(a.id for a in s.acquisitions), s.allow_cross_boundary_tracks, s.reference)
            for k, s in dataset.sequences.items()
        ),
        dataset.photon_sequences.revision,
    )


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
    result = Dataset(id=dataset.id, name=dataset.name, metadata=deepcopy(dataset.metadata))
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
        "segmentations",
    ):
        setattr(result, name, deepcopy(getattr(dataset, name), memo))
    result.calibrations = result.transforms.calibrations
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
            tuple((key, id(dataset), state_token(dataset)) for key, dataset in experiment.datasets.items()),
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


def _scientific_inputs(token: tuple) -> tuple:
    # Observations, photon sequences and provenance from independent processors are outputs.
    """Scientific inputs.

    Args:
        token: Token used by this operation.

    Returns:
        Computed tuple value.
    """
    return token[0], token[2:5], token[6:8]


@logged
def transact[R](
    dataset: Dataset,
    operation: Callable[[Dataset], R],
    *,
    expected: tuple | None = None,
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
        if expected is not None and before != expected:
            batch_source = getattr(dataset, "_batch_source_token", None)
            if batch_source != expected or _scientific_inputs(before) != _scientific_inputs(expected):
                raise StaleResultError("Inputs changed; compute a new preview before applying")
        staged = clone_dataset(dataset)
        staged._transaction_source_token = before
    value = operation(staged)
    with dataset._lock:
        if state_token(dataset) != before:
            raise StaleResultError("Live state changed during transaction; staged update was discarded")
        for name in ("localizations", "tracks", "observations", "photon_sequences", "transforms", "crops"):
            item = getattr(staged, name)
            item.dataset = dataset
            setattr(dataset, name, item)
        dataset.calibrations = dataset.transforms.calibrations
        dataset.segmentations = staged.segmentations
        dataset.processing = staged.processing
        dataset.metadata = staged.metadata
        dataset.acquisitions = staged.acquisitions
        dataset.sequences = staged.sequences
    return value
