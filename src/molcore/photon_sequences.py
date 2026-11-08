"""Explicit conversion from observation runs to independent photon-sequence records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from .revisions import FreshnessToken
from .types.dataset import Dataset
from .types.photons import PhotonSequence, PhotonUnit

if TYPE_CHECKING:
    from .types.scientific import PhotonSequenceResult


@logged
def from_observations(
    dataset: Dataset,
    *,
    run_id: str,
    molecule_id: str | None = None,
    value_column: str = "intensity",
    unit: PhotonUnit = PhotonUnit.IMAGE_INTENSITY,
    method: str = "aperture",
) -> tuple[PhotonSequence, ...]:
    """Convert an active run without applying records or claiming photon calibration.

    Args:
        dataset: Owner supplying active observation measurements.
        run_id: Explicit measurement run, preventing accidental mixing of runs.
        molecule_id: Optional molecule selector within that run.
        value_column: Numeric measurement column to copy into sequence values.
        unit: Explicit units; original image intensity is the conservative default.
        method: Measurement method recorded as provenance.

    Returns:
        Detached per-acquisition/channel/molecule sequences ready for explicit add().
    """
    rows = dataset.observations.query(molecule_id=molecule_id)
    if "run_id" not in rows or not np.any(rows["run_id"] == run_id):
        raise ValueError("Unknown active observation run")
    if value_column not in rows or rows[value_column].dtype.kind not in "iuf":
        raise ValueError("Select an available numeric measurement column")
    mask = rows["run_id"] == run_id
    selected = {k: v[mask] for k, v in rows.items()}
    settings = dataset._processing.settings.get("photometry", {}).get(run_id, {}).get("settings", {})
    return from_columns(
        selected,
        run_id=run_id,
        value_column=value_column,
        unit=unit,
        method=method,
        depends_on_localizations=bool(settings.get("exclude_neighbours", False)),
    )


@logged
def from_columns(
    columns: Mapping[str, np.ndarray],
    *,
    run_id: str,
    value_column: str = "intensity",
    unit: PhotonUnit = PhotonUnit.IMAGE_INTENSITY,
    method: str = "aperture",
    metadata: dict[str, Any] | None = None,
    depends_on_localizations: bool = False,
) -> tuple[PhotonSequence, ...]:
    """Build primary sequence records directly from detached computation samples.

    Args:
        columns: Sample arrays, including molecule/acquisition/channel/frame provenance.
        run_id: Measurement-run identifier shared across resulting sequences.
        value_column: Primary numeric sample values.
        unit: Explicit measurement units, without inferred camera calibration.
        method: Scientific measurement method recorded as provenance.
        metadata: Additional run settings copied to each sequence.
        depends_on_localizations: Invalidate measurements after changes to neighbour-mask inputs.

    Returns:
        Detached sequence records containing their own metrics and sample anchors.
    """
    selected = columns
    required = {"acquisition_id", "channel", "molecule_id", "frame", value_column}
    if not required <= selected.keys() or selected[value_column].dtype.kind not in "iuf":
        raise ValueError("Missing or nonnumeric sequence sample fields")
    if len({len(v) for v in selected.values()}) > 1:
        raise ValueError("Sample columns must have equal lengths")
    reserved = {
        "id",
        "sequence_id",
        "sample_index",
        "acquisition_id",
        "channel",
        "frame",
        "frames",
        "intensity",
        "photons",
        "timestamp",
        "timestamps",
        "x",
        "y",
        "positions",
        "valid",
        "unit",
        "run_id",
        "molecule_id",
        "localization_id",
        "track_id",
    }
    groups = sorted(
        set(zip(selected["acquisition_id"], selected["channel"], selected["molecule_id"], strict=True))
    )
    result = []
    for acquisition, channel, molecule in groups:
        indexes = np.flatnonzero(
            (selected["acquisition_id"] == acquisition)
            & (selected["channel"] == channel)
            & (selected["molecule_id"] == molecule)
        )
        indexes = indexes[np.argsort(selected["frame"][indexes], kind="stable")]
        anchors = selected.get("localization_id", np.zeros(len(selected["frame"]), dtype=np.uint64))[indexes]
        localization_ids = np.unique(anchors)
        tracks = np.unique(
            selected.get("track_id", np.zeros(len(selected["frame"]), dtype=np.uint64))[indexes]
        )
        positions = (
            np.column_stack((selected["x"][indexes], selected["y"][indexes]))
            if "x" in selected and "y" in selected
            else None
        )
        measurements = {
            key: values[indexes]
            for key, values in selected.items()
            if key not in reserved and values.dtype.kind in "biuf"
        }
        for name in ("intensity", "photons"):
            if name in selected and name != value_column:
                preserved = "source_" + name
                if preserved in measurements and (
                    not np.array_equal(measurements[preserved], selected[name][indexes], equal_nan=True)
                ):
                    raise ValueError(
                        f"Measurement {preserved!r} conflicts with the preserved source quantity"
                    )
                measurements[preserved] = selected[name][indexes]
        result.append(
            PhotonSequence(
                acquisition_id=str(acquisition),
                channel=str(channel),
                frames=selected["frame"][indexes],
                photons=selected[value_column][indexes],
                positions=positions,
                timestamps=selected["timestamp"][indexes] if "timestamp" in selected else None,
                valid=selected["valid"][indexes] if "valid" in selected else None,
                molecule_id=str(molecule),
                unit=unit,
                method=method,
                run_id=run_id,
                depends_on_transforms=True,
                depends_on_localizations=depends_on_localizations,
                localization_ids=anchors,
                measurements=measurements,
                localization_id=int(localization_ids[0])
                if len(localization_ids) == 1 and localization_ids[0]
                else None,
                track_id=int(tracks[0]) if len(tracks) == 1 and tracks[0] else None,
                metadata={
                    **(metadata or {}),
                    "source_run_id": run_id,
                    "value_column": value_column,
                    **(
                        {"observation_ids": [int(v) for v in selected["id"][indexes]]}
                        if "id" in selected and selected["id"].dtype.kind in "iu"
                        else {}
                    ),
                },
            )
        )
    return tuple(result)


@logged
def to_columns(sequences: Sequence[PhotonSequence]) -> dict[str, np.ndarray]:
    """Project sequence-owned samples into transient columns for display/export.

    Args:
        sequences: Independent sequence records; no observation table is consulted.

    Returns:
        Detached sample columns including metrics, validity, units and stable identities.
    """
    metrics = sorted({name for sequence in sequences for name in sequence.measurements})
    pieces = []
    for sequence in sequences:
        n = len(sequence.frames)
        positions = sequence.positions if sequence.positions is not None else np.full((n, 2), np.nan)
        piece = dict(
            id=np.array([f"{sequence.id}:{i}" for i in range(n)]),
            sequence_id=np.full(n, sequence.id),
            sample_index=np.arange(n),
            acquisition_id=np.full(n, sequence.acquisition_id),
            channel=np.full(n, sequence.channel),
            molecule_id=np.full(n, sequence.molecule_id or sequence.id),
            run_id=np.full(n, sequence.run_id or str(sequence.metadata.get("source_run_id", sequence.id))),
            frame=sequence.frames.copy(),
            timestamp=sequence.timestamps.copy(),
            intensity=sequence.photons.copy(),
            photons=sequence.photons.copy(),
            valid=sequence.valid.copy(),
            unit=np.full(n, str(sequence.unit)),
            x=positions[:, 0].copy(),
            y=positions[:, 1].copy(),
            localization_id=sequence.localization_ids.copy(),
            track_id=np.full(n, sequence.track_id or 0, dtype=np.uint64),
        )
        for name in metrics:
            piece[name] = (
                sequence.measurements[name].copy() if name in sequence.measurements else np.full(n, np.nan)
            )
        pieces.append(piece)
    if not pieces:
        return {
            key: np.array([], dtype=dtype)
            for key, dtype in (
                ("id", str),
                ("sequence_id", str),
                ("sample_index", int),
                ("acquisition_id", str),
                ("channel", str),
                ("molecule_id", str),
                ("run_id", str),
                ("frame", int),
                ("timestamp", float),
                ("intensity", float),
                ("photons", float),
                ("valid", bool),
                ("unit", str),
                ("x", float),
                ("y", float),
                ("localization_id", np.uint64),
                ("track_id", np.uint64),
            )
        }
    return {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}


@logged
def _apply_photon_sequences(
    dataset: Dataset, result: PhotonSequenceResult, *, expected: FreshnessToken | None = None
) -> PhotonSequenceResult:
    """Atomically add a trace run directly to the primary photon-sequence collection.

    Args:
        dataset: Live scientific model receiving trace records.
        result: Detached stationary or moving trace output.
        expected: Optional caller-managed source-state token.

    Returns:
        Result with applied sequence IDs; tracks and observations are unchanged.
    """
    from .staging import transact

    def commit(model: Dataset) -> PhotonSequenceResult:
        """Commit every sequence or none, retaining named run provenance.

        Args:
            model: Detached transaction model.

        Returns:
            Trace result linked to its newly stored primary sequence records.
        """
        runs = model._processing.settings.setdefault("photon_sequences", {})
        if result.run_id in runs:
            raise ValueError("Photon-sequence run already applied")
        for sequence in result.sequences:
            model.photon_sequences.add(sequence)
        ids = tuple(sequence.id for sequence in result.sequences)
        runs[result.run_id] = dict(
            settings=asdict(result.settings),
            mode=result.mode,
            sequence_ids=list(ids),
            acquisitions=sorted({s.acquisition_id for s in result.sequences}),
            channels=sorted({s.channel for s in result.sequences}),
            units=sorted({str(s.unit) for s in result.sequences}),
            version=1,
        )
        return replace(result, sequence_ids=ids)

    return transact(dataset, commit, expected=result.expected if expected is None else expected)
