"""GUI-independent microscopy file operations with explicit frame conventions."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from concurrent.futures import CancelledError
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tifffile
import yaml

from ._logging import logged
from .interop import read_picasso, write_picasso
from .staging import transact
from .types import Acquisition, Calibration, ChannelSpec, Segmentation
from .types.images import CachedSource, FitsSource, HDF5Source, TiffSource
from .types.processing import ProgressUpdate

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from numpy.typing import ArrayLike

    from .types import CancelEvent, CoordinateSpace, Dataset, ImageSource, ProgressCallback


@logged
def report(
    progress: ProgressCallback | None,
    cancel_event: CancelEvent | None,
    completed: int = 0,
    total: int = 1,
    description: str = "Reading files",
) -> None:
    """Report I/O progress and check cancellation before and after callbacks.

    Args:
        progress: Optional callback accepting ProgressUpdate.
        cancel_event: Optional event exposing is_set().
        completed: Finished work units.
        total: Total work units.
        description: Current operation description.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("File operation cancelled")
    if progress is not None:
        progress(ProgressUpdate(completed, total, description))
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("File operation cancelled")


@contextmanager
@logged
def atomic_files(paths: Sequence[str | Path], *, publish_guard: Any = None) -> Iterator[Any]:
    """Stage related output files beside their destinations and roll back failures.

    Args:
        paths: Explicit destination file paths, all distinct.
        publish_guard: Optional context factory guarding final replacement of all output files.

    Yields:
        Temporary sibling Paths corresponding to each destination.
    """
    paths = [Path(p).resolve() for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Output destinations must be distinct")
    temps, backups, published = [], {}, []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix=".molcore-", suffix=path.suffix, dir=path.parent)
            os.close(fd)
            temps.append(Path(temp))
        yield temps
        with publish_guard() if publish_guard is not None else nullcontext():
            for path, temp in zip(paths, temps, strict=True):
                if path.exists():
                    fd, backup = tempfile.mkstemp(prefix=".molcore-backup-", dir=path.parent)
                    os.close(fd)
                    backup = Path(backup)
                    backup.unlink()
                    os.replace(path, backup)
                    backups[path] = backup
                os.replace(temp, path)
                published.append(path)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        for path, backup in backups.items():
            os.replace(backup, path)
        raise
    finally:
        for path in temps:
            path.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


@logged
def read_image(
    path: str | Path,
    *,
    dataset: str | None = None,
    hdu: int | str = 0,
    memory: bool = "lazy",
    cache_bytes: int = 256 * 1024**2,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> ImageSource:
    """Open a TIFF/FITS/HDF5 image with lazy, cached or RAM access.

    Args:
        path: Input image file.
        dataset: Required HDF5 dataset path.
        hdu: FITS image HDU index.
        memory: lazy, cache or ram.
        cache_bytes: Byte bound for cache mode.
        progress: Optional I/O progress callback.
        cancel_event: Optional cooperative cancellation event.

    Returns:
        A process-safe MolCore image source.
    """
    report(progress, cancel_event)
    suffix = Path(path).suffix.lower()
    if suffix in (".tif", ".tiff"):
        source = TiffSource(path)
    elif suffix in (".fits", ".fit", ".fts"):
        source = FitsSource(path, hdu)
    elif suffix in (".h5", ".hdf5", ".hdf"):
        if not dataset:
            raise ValueError("Select an HDF5 image dataset")
        source = HDF5Source(path, dataset)
    else:
        raise ValueError("Supported images are TIFF, FITS and HDF5")
    if memory == "cache":
        source = CachedSource(source, max_bytes=cache_bytes)
    elif memory == "ram":
        from .types.images import ArraySource

        pixels = np.empty(source.shape, source.dtype)
        for frame in range(len(source)):
            report(progress, cancel_event, frame, max(1, len(source)), "Loading image into RAM")
            pixels[frame] = source[frame]
        source = ArraySource(pixels)
    elif memory != "lazy":
        raise ValueError("Memory mode must be lazy, cache or ram")
    report(progress, cancel_event, 1, 1, "Image ready")
    return source


@logged
def image_acquisition(
    source: ImageSource,
    *,
    mode: str = "single",
    channel: str = "image",
    layout: str | None = "donor-left",
    phase: int = 0,
    pixel_size: float | None = None,
    id: str | None = None,
    channel_specs: Mapping[str, ChannelSpec] | None = None,
) -> Acquisition:
    """Build an acquisition with explicit single/FRET/ALEX channel geometry.

    Args:
        source: Original camera image source.
        mode: single, fret, alex or custom.
        channel: Single-channel display name.
        layout: donor-left, acceptor-left, donor-top or acceptor-top.
        phase: Acquisition-specific offset into donor/acceptor excitation.
        pixel_size: Optional pixel size in nanometres.
        id: Optional acquisition identifier.
        channel_specs: Explicit custom ChannelSpec mapping overriding presets.

    Returns:
        Acquisition retaining original camera pixels and frame identity.
    """
    excitation = ("D", "A") if mode == "alex" else ()
    if channel_specs is None:
        channel_specs = {channel: ChannelSpec()}
        if mode in ("fret", "alex"):
            y, x = source.shape[1:]
            horizontal = layout.endswith("left")
            if layout not in ("donor-left", "acceptor-left", "donor-top", "acceptor-top"):
                raise ValueError("Unknown donor/acceptor layout")
            if (x if horizontal else y) % 2:
                raise ValueError("Split dimension must be even; supply explicit custom ROIs otherwise")
            rois = (
                [(0, y, 0, x // 2), (0, y, x // 2, x)]
                if horizontal
                else [(0, y // 2, 0, x), (y // 2, y, 0, x)]
            )
            if layout.startswith("acceptor"):
                rois.reverse()
            if mode == "fret":
                channel_specs = {
                    name: ChannelSpec(roi, emission=em)
                    for name, roi, em in zip(("donor", "acceptor"), rois, ("D", "A"), strict=True)
                }
            else:
                channel_specs = {
                    ex + em: ChannelSpec(rois[i], excitation=ex, emission=em)
                    for ex in ("D", "A")
                    for i, em in enumerate(("D", "A"))
                }
    return Acquisition(
        source,
        id=id,
        mode=mode,
        channel_specs=channel_specs,
        excitation_sequence=excitation,
        phase=phase,
        pixel_size=pixel_size,
        metadata={"source_path": getattr(source, "path", None), "channel_layout": layout},
    )


@logged
def export_tiff(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    path: str | Path,
    *,
    space: CoordinateSpace | str = "raw",
    original: bool = False,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
    publish_guard: Any = None,
) -> None:
    """Stream original frames or one logical channel to TIFF plus provenance YAML.

    Args:
        dataset: Model supplying pixels.
        acquisition_id: Source acquisition.
        channel: Logical channel for channel export.
        path: Destination TIFF.
        space: raw, registered or corrected for channel export.
        original: Export the original camera stream instead of a logical channel.
        progress: Optional I/O progress callback.
        cancel_event: Optional cancellation event.
        publish_guard: Optional context factory validating live inputs before publishing output.
    """
    acquisition = dataset.acquisitions[acquisition_id]
    source = acquisition.source if original else dataset.transforms.view(acquisition_id, channel, space=space)
    frames = np.arange(len(source)) if original else acquisition.channels[channel].source_frames
    if not len(source):
        raise ValueError("Cannot export an empty image stack")
    metadata = {
        "acquisition_id": acquisition_id,
        "channel": None if original else channel,
        "space": "camera" if original else str(space),
        "camera_frames": frames.tolist(),
        "pixel_size_nm": acquisition.pixel_size,
    }
    if acquisition.timestamps is not None:
        metadata["timestamps"] = acquisition.timestamps[frames].tolist()
    path = Path(path)
    with atomic_files([path, path.with_suffix(path.suffix + ".yaml")], publish_guard=publish_guard) as (
        temp,
        info,
    ):
        with tifffile.TiffWriter(
            temp, bigtiff=np.prod(source.shape) * source.dtype.itemsize >= 2**32
        ) as writer:
            for index in range(len(source)):
                report(progress, cancel_event, index, len(source), "Exporting TIFF frames")
                writer.write(
                    source[index],
                    photometric="minisblack",
                    contiguous=True,
                    description=json.dumps(metadata) if index == 0 else None,
                    metadata=None,
                )
        info.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
        report(progress, cancel_event, len(source), len(source), "Publishing TIFF")


_POS_NAMES = {
    "FRAME": "frame",
    "XCENTER": "x",
    "YCENTER": "y",
    "BRIGHTNESS": "photons",
    "BG": "bg",
    "S_X": "sx",
    "S_Y": "sy",
}


@logged
def read_table(path: str | Path) -> dict[str, np.ndarray]:
    """Read Picasso, CSV or POS.OUT columns without guessing their frame mapping.

    Args:
        path: Localization/track file. POS.OUT frames remain one-based here.

    Returns:
        Detached columns including supported extra scalar measurements.
    """
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        return read_picasso(path)[0]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t" if path.name.lower().endswith(".pos.out") else ",")
        names = reader.fieldnames
        if not names or len(set(names)) != len(names):
            raise ValueError("Table requires distinct named columns")
        rows = list(reader)
    columns = {}
    for name in names:
        values = [row[name] for row in rows]
        if any(v is None for v in values):
            raise ValueError("Table contains missing cells")
        try:
            array = np.array(values, dtype=np.int64)
        except ValueError:
            try:
                array = np.array(values, dtype=float)
            except ValueError:
                array = np.array(values, dtype=str)
        columns[_POS_NAMES.get(name, name)] = array
    return columns


@logged
def write_table(
    path: str | Path, columns: Mapping[str, np.ndarray], *, metadata: Mapping[str, Any] | None = None
) -> None:
    """Write typed columns as CSV, POS.OUT or Picasso with atomic publication.

    Args:
        path: Output filename determining the format.
        columns: Columns in the caller's explicitly selected timeline.
        metadata: Picasso YAML metadata documents.
    """
    path = Path(path)
    from .interop import to_records

    to_records(columns)  # Validate lengths and scalar dtypes before touching files.
    if path.suffix.lower() in (".h5", ".hdf5"):
        with atomic_files([path, path.with_suffix(".yaml")]) as (data, info):
            write_picasso(data, columns, metadata or [])
            os.replace(data.with_suffix(".yaml"), info)
        return
    pos = path.name.lower().endswith(".pos.out")
    required = {"frame", "x", "y", "photons", "bg", "sx", "sy"} if pos else {"frame", "x", "y"}
    if not required <= columns.keys():
        raise ValueError(f"Missing format-required measurements: {sorted(required - columns.keys())}")
    converted = dict(columns)
    if pos:
        converted["frame"] = np.asarray(columns["frame"]) + 1
    reverse = {v: k for k, v in _POS_NAMES.items()}
    names = list(converted)
    with atomic_files([path]) as (temp,):
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t" if pos else ",")
            writer.writerow([reverse.get(n, n) if pos else n for n in names])
            writer.writerows(zip(*(converted[n] for n in names), strict=True))


@logged
def import_table(
    dataset: Dataset,
    acquisition_id: str,
    channel: str,
    columns: Mapping[str, ArrayLike],
    *,
    timeline: str = "camera",
    frame_base: int = 0,
    units: str = "pixels",
    tracks: bool = False,
    reuse_ids: bool = False,
) -> np.ndarray:
    """Map imported rows to original frames and apply them transactionally.

    Args:
        dataset: Target model.
        acquisition_id: Explicit target acquisition.
        channel: Explicit target channel.
        columns: frame/x/y plus scalar measurements and optional particle IDs.
        timeline: camera or channel indices.
        frame_base: Zero or one, explicitly selected by the caller.
        units: pixels or nm; nanometres require calibrated pixel size.
        tracks: Create track memberships from particle/track_id groups.
        reuse_ids: For track import, require existing localization_id anchors.

    Returns:
        Localization IDs for the imported rows.
    """
    if not {"frame", "x", "y"} <= columns.keys() or frame_base not in (0, 1):
        raise ValueError("Import requires frame/x/y and a zero/one frame base")
    columns = {k: np.asarray(v).copy() for k, v in columns.items()}
    frames = np.asarray(columns["frame"], dtype=float) - frame_base
    if not np.isfinite(frames).all() or np.any(frames < 0) or np.any(frames != np.floor(frames)):
        raise ValueError("Frames must be nonnegative integers after base conversion")
    frames = frames.astype(np.int64)
    acquisition = dataset.acquisitions[acquisition_id]
    view = acquisition.channels[channel]
    if timeline == "channel":
        if np.any(frames >= len(view)):
            raise ValueError("Channel frame outside the selected movie")
        frames = view.source_frames[frames]
    elif timeline != "camera":
        raise ValueError("Choose camera or channel frame mapping")
    columns["frame"] = frames
    if units == "nm":
        if acquisition.pixel_size is None:
            raise ValueError("Nanometre coordinates require a pixel calibration")
        for key in ("x", "y", "sx", "sy", "lpx", "lpy"):
            if key in columns:
                columns[key] = columns[key].astype(float) / acquisition.pixel_size
    elif units != "pixels":
        raise ValueError("Coordinate units must be pixels or nm")
    group = columns.get("particle", columns.get("track_id"))
    if tracks and group is None:
        raise ValueError("Track import requires a particle or track_id column")

    def commit(model: Dataset) -> Any:
        """Commit.

        Args:
            model: Model or fitting method selected by the caller.

        Returns:
            Result returned after updating the staged model.
        """
        if reuse_ids:
            if not tracks or "localization_id" not in columns:
                raise ValueError("Reusing tracks requires explicit localization_id anchors")
            ids = np.asarray(columns["localization_id"])
            for index, identifier in enumerate(ids):
                row = model.localizations.query(ids=[identifier], space="raw")
                if len(row["id"]) != 1 or any(
                    row[k][0] != value
                    for k, value in (
                        ("acquisition_id", acquisition_id),
                        ("channel", channel),
                        ("frame", frames[index]),
                        ("x", columns["x"][index]),
                        ("y", columns["y"][index]),
                    )
                ):
                    raise ValueError("Localization anchor conflicts with imported measurements")
        else:
            data = {
                k: v
                for k, v in columns.items()
                if k
                not in {
                    "id",
                    "localization_id",
                    "track_id",
                    "particle",
                    "acquisition_id",
                    "dataset",
                    "channel",
                    "override_x",
                    "override_y",
                    "has_override",
                    "original_x",
                    "original_y",
                }
            }
            if "id" in columns:
                data["source_localization_id"] = columns["id"]
            ids = model.localizations.add(acquisition_id=acquisition_id, channel=channel, **data)
        if tracks:
            for identifier in np.unique(group):
                model.tracks.add(ids[group == identifier], metadata={"source_track_id": str(identifier)})
        return ids

    return transact(dataset, commit)


@logged
def read_calibration(path: str | Path) -> np.ndarray:
    """Read an invertible affine or projective JSON/TXT/NPY calibration.

    Args:
        path: Matrix file; NumPy object loading is disabled.

    Returns:
        Validated homogeneous 3x3 matrix.
    """
    path = Path(path)
    value = (
        np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npy"
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if isinstance(value, dict):
        value = value["matrix"]
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0, 0, 1]])
    return Calibration("import", matrix).matrix.copy()


@logged
def write_calibration(path: str | Path, matrix: ArrayLike) -> None:
    """Write an affine or projective calibration as JSON or NumPy array.

    Args:
        path: Destination JSON/TXT/NPY file.
        matrix: Validated homogeneous 3x3 matrix.
    """
    matrix = Calibration("export", matrix).matrix
    with atomic_files([path]) as (temp,):
        if Path(path).suffix.lower() == ".npy":
            with temp.open("wb") as stream:
                np.save(stream, matrix, allow_pickle=False)
        else:
            temp.write_text(json.dumps(matrix.tolist()), encoding="utf-8")


@logged
def read_mask(
    path: str | Path,
    *,
    segmentation_id: str | None,
    acquisition_id: str,
    channel: str,
    frame: int | None,
    space: CoordinateSpace | str = "raw",
    source_image: ArrayLike | None = None,
) -> Segmentation:
    """Read a static label TIFF with explicit acquisition and coordinate bindings.

    Args:
        path: Two-dimensional label or binary mask TIFF.
        segmentation_id: Named result identifier.
        acquisition_id: Target acquisition.
        channel: Bound logical channel.
        frame: Source original camera frame.
        space: Coordinate system of the mask.
        source_image: Optional image used to create the mask.

    Returns:
        Detached validated Segmentation.
    """
    return Segmentation(
        segmentation_id,
        acquisition_id,
        tifffile.imread(path),
        source_image,
        channel=channel,
        frame=frame,
        space=space,
        mask_path=str(Path(path).resolve()),
    )
