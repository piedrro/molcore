"""NumPy, Picasso and napari interchange without importing GUI libraries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import yaml


def to_records(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert equal-length named columns into a structured NumPy array.

    Args:
        columns: One-dimensional columns keyed by field name.

    Returns:
        Detached structured array preserving field order and scalar dtypes.

    Raises:
        ValueError: Columns have incompatible lengths or unsupported dimensions.
    """
    arrays = {k: np.asarray(v) for k, v in columns.items()}
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) > 1 or any(v.ndim != 1 or v.dtype.kind == "O" for v in arrays.values()):
        raise ValueError("Expected equally sized, typed 1D columns")
    result = np.empty(next(iter(lengths), 0), dtype=[(k, v.dtype) for k, v in arrays.items()])
    for name, values in arrays.items():
        result[name] = values
    return result


def from_records(records: np.ndarray) -> dict[str, np.ndarray]:
    """Detach named columns from a structured NumPy array.

    Args:
        records: One-dimensional structured array.

    Returns:
        Dictionary containing an independent array for each named field.
    """
    if records.ndim != 1 or records.dtype.names is None:
        raise ValueError("Expected a 1D structured array")
    return {name: records[name].copy() for name in records.dtype.names}


def write_picasso(path, columns: Mapping[str, np.ndarray], metadata: list[dict]) -> None:
    """Write Picasso's locs dataset and companion YAML metadata documents.

    Args:
        path: Destination localization HDF5 file.
        columns: Localization columns containing frame, x and y, plus measurements.
        metadata: Picasso metadata dictionaries, one per YAML document.

    Raises:
        ValueError: Required localization columns are missing.
    """
    if not {"frame", "x", "y"} <= columns.keys():
        raise ValueError("Picasso localizations require frame, x and y")
    path = Path(path)
    converted = {}
    for name, values in columns.items():
        values = np.asarray(values)
        converted[name] = np.char.encode(values, "utf-8") if values.dtype.kind == "U" else values
    records = to_records(converted)
    text = yaml.safe_dump_all(metadata, allow_unicode=True, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        file.create_dataset("locs", data=records)
    path.with_suffix(".yaml").write_text(text, encoding="utf-8")


def read_picasso(path) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Read Picasso localizations and optional companion YAML metadata.

    Args:
        path: Localization HDF5 file containing locs.

    Returns:
        Detached columns and metadata documents, preserving extra fields.
    """
    path = Path(path)
    with h5py.File(path, "r") as file:
        result = from_records(file["locs"][:])
    for name, values in result.items():
        if values.dtype.kind == "S":
            result[name] = np.char.decode(values, "utf-8")
    yaml_path = path.with_suffix(".yaml")
    metadata = list(yaml.safe_load_all(yaml_path.read_text(encoding="utf-8"))) if yaml_path.exists() else []
    return result, metadata


def _timeline(rows, frame_column):
    if frame_column == "frame" and len(np.unique(rows.get("acquisition_id", []))) > 1:
        raise ValueError("Multiple acquisitions require an explicit sequence_frame timeline")
    if frame_column not in rows:
        raise ValueError(f"Missing timeline column: {frame_column}")
    return rows[frame_column]


def napari_points(rows: Mapping[str, np.ndarray], *, frame_column: str = "frame") -> np.ndarray:
    """Convert a localization subset into napari (time, y, x) points.

    Args:
        rows: Already filtered localization columns in the desired coordinate space.
        frame_column: frame for one acquisition, or sequence_frame for a sequence.

    Returns:
        Floating-point array of shape (n, 3).
    """
    return np.column_stack([_timeline(rows, frame_column), rows["y"], rows["x"]]).astype(float)


def napari_tracks(rows: Mapping[str, np.ndarray], *, frame_column: str = "frame") -> np.ndarray:
    """Convert joined track rows into napari (track_id, time, y, x) data.

    Args:
        rows: Track query columns in the desired coordinate space.
        frame_column: Explicit acquisition or sequence timeline column.

    Returns:
        Floating-point array of shape (n, 4), sorted by track and time.
    """
    time = _timeline(rows, frame_column)
    if np.any(np.asarray(rows["track_id"], dtype=np.uint64) > 2**53):
        raise ValueError("Track identifiers exceed exact napari float representation")
    data = np.column_stack([rows["track_id"], time, rows["y"], rows["x"]]).astype(float)
    return data[np.lexsort((data[:, 1], data[:, 0]))]


def napari_labels(segmentation) -> np.ndarray:
    """Return detached label data suitable for a napari labels layer.

    Args:
        segmentation: Segmentation whose acquisition/space is selected by the caller.

    Returns:
        A copy of the 2D instance-label mask.
    """
    return segmentation.labels.copy()
