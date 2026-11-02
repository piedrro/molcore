"""Versioned HDF5 snapshots with optional relative external image references."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from ._logging import logged
from .types._shared import Calibration, ChannelSpec, Crop, ImagePolicy, Segmentation
from .types.images import (
    CachedSource,
    FitsSource,
    HDF5Source,
    MissingSource,
    TiffSource,
)
from .types.photons import PhotonSequence
from .types.storage import SaveOptions

if TYPE_CHECKING:
    from .types import Dataset, Experiment, ImageSource
    from .types.scientific import CancelEvent, ProgressCallback

FORMAT_VERSION = 4

_SOURCE_CODECS: dict[str, tuple[type, Any, Any]] = {}


def register_image_source(
    kind: str, source_type: type, *, writer: Any, reader: Any
) -> None:
    """Register trusted descriptor callbacks; snapshot data never imports plugins.

    Writers return JSON-compatible metadata with an absolute ``path`` to a file
    or directory. Readers receive the metadata with its path resolved. Register
    stable namespaced kinds in each process that saves or opens these sources.

    Args:
        kind: Stable namespaced descriptor identifier.
        source_type: Exact image source class handled by this codec.
        writer: Callback returning JSON-compatible descriptor metadata.
        reader: Callback restoring a source from resolved descriptor metadata.
    """
    if not isinstance(kind, str) or not kind or not isinstance(source_type, type):
        raise ValueError("A nonempty kind and source type are required")
    if not callable(writer) or not callable(reader):
        raise TypeError("writer and reader must be callable")
    if kind in _SOURCE_CODECS:
        raise ValueError(f"Image source kind already registered: {kind}")
    _SOURCE_CODECS[kind] = (source_type, writer, reader)


register_image_source(
    "TiffSource",
    TiffSource,
    writer=lambda s: {"path": s.path},
    reader=lambda d: TiffSource(d["path"]),
)
register_image_source(
    "FitsSource",
    FitsSource,
    writer=lambda s: {"path": s.path, "hdu": s.hdu},
    reader=lambda d: FitsSource(d["path"], d["hdu"]),
)
register_image_source(
    "HDF5Source",
    HDF5Source,
    writer=lambda s: {"path": s.path, "dataset": s.dataset},
    reader=lambda d: HDF5Source(d["path"], d["dataset"]),
)


def _resolve_options(
    options: SaveOptions | None, policy: ImagePolicy, width: int, layout: str
) -> SaveOptions:
    """Translate legacy policies once into independent image choices."""
    if options is not None:
        if not isinstance(options, SaveOptions):
            raise TypeError("options must be SaveOptions")
        if policy != ImagePolicy.NONE or width != 7:
            raise ValueError("Do not combine options with image_policy or crop_width")
        if options.originals == "external" and layout != "directory":
            raise ValueError("External originals require directory layout")
        return options
    kinds = set()
    if policy in (
        ImagePolicy.LOCALIZATION_CROPS,
        ImagePolicy.BOTH,
        ImagePolicy.LOCALIZATION_AND_PHOTON_CROPS,
        ImagePolicy.ALL_CROPS,
    ):
        kinds.add("localization")
    if policy in (
        ImagePolicy.TRACK_CROPS,
        ImagePolicy.BOTH,
        ImagePolicy.TRACK_AND_PHOTON_CROPS,
        ImagePolicy.ALL_CROPS,
    ):
        kinds.add("track")
    if policy in (
        ImagePolicy.PHOTON_SEQUENCE_CROPS,
        ImagePolicy.LOCALIZATION_AND_PHOTON_CROPS,
        ImagePolicy.TRACK_AND_PHOTON_CROPS,
        ImagePolicy.ALL_CROPS,
    ):
        kinds.add("photon_sequence")
    return SaveOptions(
        originals=(
            "full"
            if policy == ImagePolicy.FULL
            else "external" if layout == "directory" else "none"
        ),
        crops=frozenset(kinds),
        crop_width=width,
    )


def _json(value: Any) -> str:
    """Json.

    Args:
        value: Value supplied by the caller or connected control.

    Returns:
        Computed str value.
    """

    def convert(item: Any) -> Any:
        """Convert.

        Args:
            item: Item used by this operation.

        Returns:
            JSON-compatible representation of the input value.
        """
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Metadata must be JSON-compatible, got {type(item).__name__}")

    return json.dumps(value, default=convert, ensure_ascii=False, allow_nan=False)


def _write_columns(group: Any, data: dict[str, np.ndarray]) -> None:
    # Numeric dataset names allow arbitrary user column names, including '/'.
    """Write columns.

    Args:
        group: Group used by this operation.
        data: Named arrays to read, combine or serialize.
    """
    for index, (name, values) in enumerate(data.items()):
        values = np.asarray(values)
        if values.dtype.kind == "U":
            ds = group.create_dataset(
                str(index), data=values.astype(object), dtype=h5py.string_dtype()
            )
        elif values.dtype.kind == "O":
            raise ValueError("Object-valued columns cannot be saved")
        else:
            ds = group.create_dataset(str(index), data=values)
        ds.attrs["name"], ds.attrs["numpy_dtype"] = name, values.dtype.str


def _read_columns(group: Any) -> dict[str, np.ndarray]:
    """Read columns.

    Args:
        group: Group used by this operation.

    Returns:
        Detached named result columns.
    """
    result = {}
    for ds in group.values():
        dtype = np.dtype(ds.attrs["numpy_dtype"])
        result[ds.attrs["name"]] = np.asarray(
            ds.asstr()[:] if dtype.kind == "U" else ds[:], dtype=dtype
        )
    return result


def _external(
    source: ImageSource, directory: Path | None, destination: Path
) -> dict[str, Any] | None:
    """External.

    Args:
        source: Image source providing lazy frame access.
        directory: Base directory for resolving external image references.
        destination: Resolved destination used for atomic output replacement.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    while isinstance(source, CachedSource):
        source = source.source
    codec = next(
        (
            (kind, writer)
            for kind, (typ, writer, _) in _SOURCE_CODECS.items()
            if type(source) is typ
        ),
        None,
    )
    if codec is None:
        return None
    kind, writer = codec
    spec = dict(writer(source))
    source_path = Path(spec["path"]).resolve()
    if source_path == destination:
        return None  # Do not leave references into the file being replaced.
    try:
        path = os.path.relpath(source_path, directory)
    except ValueError:
        path = str(
            source_path
        )  # Different Windows drive; still usable, not relocatable.
    spec.update(path=path, kind=kind)
    return spec


def _load_external(
    spec: dict[str, Any],
    directory: Path | None,
    shape: tuple[int, ...],
    dtype: np.dtype | str,
) -> ImageSource:
    """Load external.

    Args:
        spec: Spec used by this operation.
        directory: Base directory for resolving external image references.
        shape: Array dimensions in time, y, x order.
        dtype: NumPy-compatible element dtype.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    kind = spec["kind"]
    if kind not in _SOURCE_CODECS:
        raise ValueError(
            f"Unsupported external image source; register reader for {kind!r}"
        )
    path = (Path(directory or ".") / spec["path"]).resolve()
    if not path.exists():
        return MissingSource(shape, dtype, f"External images unavailable: {path}")
    source = _SOURCE_CODECS[kind][2]({**spec, "path": str(path)})
    if tuple(source.shape) != tuple(shape) or np.dtype(source.dtype) != np.dtype(dtype):
        source.close()
        raise ValueError(f"External image geometry/dtype changed: {path}")
    return source


def _write_crops(
    file: Any, dataset: Dataset, policy: frozenset[str], width: int
) -> list[int]:
    """Write crops.

    Args:
        file: File used by this operation.
        dataset: Scientific model supplying acquisitions and result tables.
        policy: Policy used by this operation.
        width: Odd crop width in camera pixels.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    ids = set()
    if "localization" in policy:
        ids.update(int(i) for i in dataset.localizations._data.get("id", []))
    if "track" in policy:
        for members in dataset.tracks._members.values():
            ids.update(int(i) for i in members)
    if not ids:
        return []
    ids = sorted(ids)
    group = file.create_group("crops")
    group.attrs["width"] = width
    # Different acquisitions may have different camera dtypes.
    dtype = np.result_type(*(a.source.dtype for a in dataset.acquisitions.values()))
    pixels = group.create_dataset(
        "pixels",
        (len(ids), width, width),
        dtype=dtype,
        chunks=(1, width, width),
        compression="gzip",
    )
    valid = group.create_dataset(
        "valid",
        (len(ids), width, width),
        dtype=bool,
        chunks=(1, width, width),
        compression="gzip",
    )
    frames, origins, acquisitions, channels = [], [], [], []
    for index, localization_id in enumerate(ids):
        crop = dataset.crops.localization(localization_id, width=width)
        pixels[index], valid[index] = crop.pixels, crop.valid
        frames.append(crop.camera_frame)
        origins.append(crop.origin)
        acquisitions.append(crop.acquisition_id)
        channels.append(crop.channel)
    group.create_dataset("ids", data=np.asarray(ids, dtype=np.uint64))
    group.create_dataset("frames", data=frames)
    group.create_dataset("origins", data=origins)
    group.create_dataset(
        "acquisitions",
        data=np.asarray(acquisitions, dtype=object),
        dtype=h5py.string_dtype(),
    )
    group.create_dataset(
        "channels", data=np.asarray(channels, dtype=object), dtype=h5py.string_dtype()
    )
    return ids


def _write_photon_sequences(file: h5py.File, dataset: Dataset) -> None:
    """Serialize independent sequence samples independently of image policies.

    Args:
        file: Staged destination HDF5 file.
        dataset: Owner providing photon-sequence records and validity.
    """
    root = file.create_group("photon_sequences")
    root.attrs["revision"] = dataset.photon_sequences.revision
    for index, sequence in enumerate(
        dataset.photon_sequences.query(include_stale=True)
    ):
        group = root.create_group(str(index))
        group.attrs["info"] = _json(
            dict(
                id=sequence.id,
                acquisition_id=sequence.acquisition_id,
                channel=sequence.channel,
                unit=sequence.unit,
                method=sequence.method,
                depends_on_transforms=sequence.depends_on_transforms,
                depends_on_localizations=sequence.depends_on_localizations,
                molecule_id=sequence.molecule_id,
                run_id=sequence.run_id,
                localization_id=sequence.localization_id,
                track_id=sequence.track_id,
                metadata=sequence.metadata,
                stale=dataset.photon_sequences.is_stale(sequence.id),
            )
        )
        for name in (
            "frames",
            "photons",
            "timestamps",
            "valid",
            "positions",
            "localization_ids",
        ):
            value = getattr(sequence, name)
            if value is not None:
                group.create_dataset(name, data=value)
        _write_columns(group.create_group("measurements"), sequence.measurements)


def _write_photon_crops(
    file: h5py.File,
    dataset: Dataset,
    policy: frozenset[str],
    width: int,
    cancel_event: CancelEvent | None,
) -> list[tuple[str, int]]:
    """Stream selected photon-sequence crops while retaining sample provenance.

    Args:
        file: Staged HDF5 output.
        dataset: Model supplying sequence positions and source pixels.
        policy: Selected microscopy-pixel inclusion policy.
        width: Odd crop width in channel pixels.
        cancel_event: Optional cooperative cancellation event.

    Returns:
        Sequence/sample keys in saved pixel-row order.
    """
    from .files import report

    if "photon_sequence" not in policy:
        return []
    sequences = dataset.photon_sequences.query(include_stale=True)
    for sequence in sequences:
        if sequence.positions is None:
            raise ValueError(f"Photon sequence {sequence.id!r} has no crop positions")
    count = sum(int(np.isfinite(s.positions).all(axis=1).sum()) for s in sequences)
    if not count:
        return []
    group = file.create_group("photon_crops")
    group.attrs["width"] = width
    dtype = np.result_type(
        *(dataset.acquisitions[s.acquisition_id].source.dtype for s in sequences)
    )
    pixels = group.create_dataset(
        "pixels",
        (count, width, width),
        dtype=dtype,
        chunks=(1, width, width),
        compression="gzip",
    )
    valid = group.create_dataset(
        "valid",
        (count, width, width),
        dtype=bool,
        chunks=(1, width, width),
        compression="gzip",
    )
    keys, frames, origins, acquisitions, channels, anchors = [], [], [], [], [], []
    for sequence in sequences:
        for index, crop in enumerate(
            dataset.crops.photon_sequence(sequence.id, width=width, include_stale=True)
        ):
            report(None, cancel_event)
            if crop is None:
                continue
            row = len(keys)
            pixels[row], valid[row] = crop.pixels, crop.valid
            keys.append((sequence.id, index))
            frames.append(crop.camera_frame)
            origins.append(crop.origin)
            acquisitions.append(crop.acquisition_id)
            channels.append(crop.channel)
            anchors.append(crop.localization_id or 0)
    for name, values in (
        ("sequence_ids", [key[0] for key in keys]),
        ("acquisitions", acquisitions),
        ("channels", channels),
    ):
        group.create_dataset(
            name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype()
        )
    for name, values in (
        ("sample_indices", [key[1] for key in keys]),
        ("frames", frames),
        ("origins", origins),
        ("ids", anchors),
    ):
        group.create_dataset(
            name,
            data=np.asarray(values, dtype=np.uint64 if name == "ids" else np.int64),
        )
    return keys


def _write_dataset_group(
    file: Any,
    dataset: Dataset,
    destination: Path,
    *,
    layout: str,
    policy: ImagePolicy,
    options: SaveOptions,
    strict_external: bool,
    crop_width: int,
    progress: Any,
    cancel_event: Any,
) -> tuple:
    """Write one dataset at an arbitrary HDF5 group without copying movie stacks."""
    from .files import report

    file.attrs["format"] = "molcore"
    file.attrs["format_version"] = FORMAT_VERSION
    file.attrs["metadata"] = _json(dataset.metadata)
    file.attrs["image_policy"] = policy.value
    file.attrs["image_storage"] = _json(
        dict(
            originals=options.originals,
            crops=sorted(options.crops),
            crop_width=options.crop_width,
        )
    )
    file.attrs["processing"] = _json(asdict(dataset.processing))
    acquisitions = file.create_group("acquisitions", track_order=True)
    for index, acquisition in enumerate(dataset.acquisitions.values()):
        report(None, cancel_event)
        group = acquisitions.create_group(str(index))
        info = {
            "id": acquisition.id,
            "shape": acquisition.source.shape,
            "dtype": acquisition.source.dtype.str,
            "mode": acquisition.mode,
            "pixel_size": acquisition.pixel_size,
            "metadata": acquisition.metadata,
            "excitation_sequence": acquisition.excitation_sequence,
            "phase": acquisition.phase,
            "channel_specs": {
                k: asdict(v) for k, v in acquisition.channel_specs.items()
            },
        }
        if acquisition.timestamps is not None:
            group.create_dataset("timestamps", data=acquisition.timestamps)
        if options.originals == "full":
            n, y, x = acquisition.source.shape
            images = group.create_dataset(
                "images",
                shape=(n, y, x),
                dtype=acquisition.source.dtype,
                maxshape=(None, y, x),
                chunks=(1, min(y, 256), min(x, 256)),
                compression="gzip",
            )
            for frame in range(n):
                report(
                    progress,
                    cancel_event,
                    frame,
                    max(1, n),
                    f"Saving images: {acquisition.id}",
                )
                images[frame] = acquisition.source[frame]
        elif options.originals == "external":
            info["external"] = _external(
                acquisition.source, destination.parent, destination
            )
            if strict_external and info["external"] is None:
                raise ValueError(
                    f"No external descriptor available for acquisition {acquisition.id!r}"
                )
        group.attrs["info"] = _json(info)
    file.attrs["sequences"] = _json(
        [
            {
                "id": s.id,
                "acquisition_ids": [a.id for a in s.acquisitions],
                "allow_cross_boundary_tracks": s.allow_cross_boundary_tracks,
                "reference": s.reference,
            }
            for s in dataset.sequences.values()
        ]
    )
    for name in ("localizations", "observations"):
        table = getattr(dataset, name)
        group = file.create_group(name)
        group.attrs["next_id"], group.attrs["revision"] = table._next_id, table.revision
        _write_columns(group, table._data)
    tracks = file.create_group("tracks")
    tracks.attrs["metadata"] = _json(dataset.tracks.metadata)
    tracks.attrs["next_id"] = dataset.tracks._next_id
    tracks.attrs["revision"] = dataset.tracks.revision
    memberships = [
        (t, int(i)) for t, ids in dataset.tracks._members.items() for i in ids
    ]
    tracks.create_dataset(
        "memberships", data=np.asarray(memberships, dtype=np.uint64).reshape(-1, 2)
    )
    transforms = file.create_group("transforms")
    transforms.attrs["revision"] = dataset.transforms.revision
    transforms.attrs["calibrations"] = _json(
        [asdict(c) for c in dataset.calibrations.values()]
    )
    transforms.attrs["selected"] = _json(
        [[a, c, name] for (a, c), name in dataset.transforms._selected.items()]
    )
    drift = transforms.create_group("drift")
    for index, (acquisition_id, (frames, displacement)) in enumerate(
        dataset.transforms._drift.items()
    ):
        group = drift.create_group(str(index))
        group.attrs["acquisition_id"] = acquisition_id
        group.create_dataset("frames", data=frames)
        group.create_dataset("displacement", data=displacement)
    segmentations = file.create_group("segmentations")
    for index, segmentation in enumerate(dataset.segmentations.values()):
        report(None, cancel_event)
        group = segmentations.create_group(str(index))
        info = {
            key: getattr(segmentation, key)
            for key in (
                "id",
                "acquisition_id",
                "channel",
                "space",
                "frame",
                "source_path",
                "mask_path",
                "metadata",
            )
        }
        group.attrs["info"] = _json(info)
        group.create_dataset("labels", data=segmentation.labels, compression="gzip")
        if segmentation.source_image is not None:
            group.create_dataset(
                "source", data=segmentation.source_image, compression="gzip"
            )
    saved_crop_ids = _write_crops(file, dataset, options.crops, crop_width)
    _write_photon_sequences(file, dataset)
    saved_photon_keys = _write_photon_crops(
        file, dataset, options.crops, crop_width, cancel_event
    )
    file.attrs["kind"] = "dataset"
    file.attrs["id"], file.attrs["name"] = dataset.id, dataset.name
    return saved_crop_ids, saved_photon_keys


@logged
def save_dataset(
    dataset: Dataset,
    path: str | Path,
    *,
    layout: str | None = "hdf5",
    image_policy: str = "none",
    options: SaveOptions | None = None,
    crop_width: int = 7,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
    publish_guard: Any = None,
) -> None:
    """Write an dataset snapshot without modifying the previous snapshot on failure.

    Args:
        dataset: Dataset to serialize.
        path: Destination file or directory.
        layout: hdf5 embeds selected content; directory may reference external images.
        image_policy: Raw-image inclusion policy.
        options: Independent original-image and crop choices, replacing legacy arguments.
        crop_width: Positive odd pixel width for saved crops.
        progress: Optional callback receiving structured save progress.
        cancel_event: Optional event requesting cancellation before atomic replacement.

    Raises:
        ValueError: Layout, metadata or crop settings are invalid.
        OSError: Writing or atomic replacement fails.
    """
    from .files import report

    report(progress, cancel_event, 0, 1, "Saving dataset")
    if layout not in ("hdf5", "directory"):
        raise ValueError("Layout must be hdf5 or directory")
    policy = ImagePolicy(image_policy)
    strict_external = options is not None
    options = _resolve_options(options, policy, crop_width, layout)
    crop_width = options.crop_width
    path = Path(path).resolve()
    destination = path / "dataset.h5" if layout == "directory" else path
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".molcore-", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        with h5py.File(temp_name, "w") as file:
            saved_crop_ids, saved_photon_keys = _write_dataset_group(
                file,
                dataset,
                destination,
                layout=layout,
                policy=policy,
                options=options,
                strict_external=strict_external,
                crop_width=crop_width,
                progress=progress,
                cancel_event=cancel_event,
            )
            file.flush()
        report(progress, cancel_event, 1, 1, "Committing dataset snapshot")
        with publish_guard() if publish_guard is not None else nullcontext():
            os.replace(temp_name, destination)
            # A replaced snapshot has new crop row indices and possibly no images.
            dataset.crops._saved = {
                key: value
                for key, value in dataset.crops._saved.items()
                if Path(value[0]).resolve() != destination
            }
            for index, localization_id in enumerate(saved_crop_ids):
                dataset.crops._saved[localization_id, crop_width] = (
                    str(destination),
                    index,
                )
            dataset.crops._photon_saved = {
                key: value
                for key, value in dataset.crops._photon_saved.items()
                if Path(value[0]).resolve() != destination
            }
            for index, (sequence_id, sample_index) in enumerate(saved_photon_keys):
                dataset.crops._photon_saved[sequence_id, sample_index, crop_width] = (
                    str(destination),
                    index,
                )
            for index, acquisition in enumerate(dataset.acquisitions.values()):
                source = acquisition.source
                while isinstance(source, CachedSource):
                    source = source.source
                if (
                    isinstance(source, HDF5Source)
                    and Path(source.path).resolve() == destination
                ):
                    acquisition.source = (
                        HDF5Source(destination, f"/acquisitions/{index}/images")
                        if options.originals == "full"
                        else MissingSource(source.shape, source.dtype)
                    )
    finally:
        if Path(temp_name).exists():
            Path(temp_name).unlink()


@logged
def open_dataset(
    path: str | Path,
    *,
    group: str = "/",
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> Dataset:
    """Open a supported snapshot, eagerly restoring tables and lazily reading pixels.

    Args:
        path: HDF5 snapshot file or dataset directory.

    Returns:
        Restored Dataset with process-safe image and crop descriptors.

    Raises:
        ValueError: The format or version is unsupported.
    """
    from .files import report
    from .types.acquisition import Acquisition
    from .types.dataset import Dataset

    report(progress, cancel_event, 0, 1, "Opening dataset")

    path = Path(path).resolve()
    if path.is_dir():
        path = path / "dataset.h5"
    with h5py.File(path, "r") as snapshot:
        file = snapshot[group]
        if file.attrs.get("kind", "dataset") != "dataset":
            raise ValueError(
                "Expected a Dataset snapshot; use Experiment.open for experiments"
            )
        if file.attrs.get("format") != "molcore" or file.attrs.get(
            "format_version"
        ) not in (
            1,
            2,
            3,
            FORMAT_VERSION,
        ):
            raise ValueError("Unsupported MolCore format/version")
        dataset = Dataset(
            id=file.attrs.get("id"),
            name=file.attrs.get("name"),
            metadata=json.loads(file.attrs["metadata"]),
        )
        processing = json.loads(file.attrs["processing"])
        dataset.processing.settings = processing["settings"]
        dataset.processing.software = processing["software"]
        for group in file["acquisitions"].values():
            report(None, cancel_event)
            info = json.loads(group.attrs["info"])
            if "images" in group:
                source = HDF5Source(path, group["images"].name)
            elif info.get("external"):
                source = _load_external(
                    info["external"], path.parent, info["shape"], info["dtype"]
                )
            else:
                source = MissingSource(info["shape"], info["dtype"])
            acquisition = Acquisition(
                source,
                id=info["id"],
                mode=info["mode"],
                channel_specs={
                    k: ChannelSpec(**v) for k, v in info["channel_specs"].items()
                },
                excitation_sequence=info["excitation_sequence"],
                phase=info["phase"],
                timestamps=group["timestamps"][:] if "timestamps" in group else None,
                pixel_size=info["pixel_size"],
                metadata=info["metadata"],
            )
            dataset.add_acquisition(acquisition)
        for sequence in json.loads(file.attrs["sequences"]):
            dataset.add_sequence(**sequence)
        for name in ("localizations", "observations"):
            table, group = getattr(dataset, name), file[name]
            table._data = _read_columns(group)
            table._next_id, table.revision = int(group.attrs["next_id"]), int(
                group.attrs["revision"]
            )
        tracks = file["tracks"]
        dataset.tracks.metadata = {
            int(k): v for k, v in json.loads(tracks.attrs["metadata"]).items()
        }
        membership = tracks["memberships"][:]
        for track in dataset.tracks.metadata:
            dataset.tracks._members[track] = membership[membership[:, 0] == track, 1]
        dataset.tracks._next_id = int(tracks.attrs["next_id"])
        dataset.tracks.revision = int(tracks.attrs["revision"])
        transforms = file["transforms"]
        for value in json.loads(transforms.attrs["calibrations"]):
            calibration = Calibration(**value)
            dataset.transforms._calibrations[calibration.name] = calibration
        dataset.transforms._selected = {
            (a, c): name for a, c, name in json.loads(transforms.attrs["selected"])
        }
        for group in transforms["drift"].values():
            dataset.transforms._drift[group.attrs["acquisition_id"]] = (
                group["frames"][:],
                group["displacement"][:],
            )
        dataset.transforms.revision = int(transforms.attrs["revision"])
        for group in file["segmentations"].values():
            segmentation = Segmentation(
                **json.loads(group.attrs["info"]),
                labels=group["labels"][:],
                source_image=group["source"][:] if "source" in group else None,
            )
            dataset._segmentations[segmentation.id] = segmentation
        if "crops" in file:
            width = int(file["crops"].attrs["width"])
            for index, localization_id in enumerate(file["crops/ids"][:]):
                dataset.crops._saved[int(localization_id), width] = (
                    str(path),
                    index,
                    file["crops"].name,
                )
        if "photon_sequences" in file:
            for group in file["photon_sequences"].values():
                info = json.loads(group.attrs["info"])
                stale = info.pop("stale", False)
                sequence = PhotonSequence(
                    **info,
                    frames=group["frames"][:],
                    photons=group["photons"][:],
                    timestamps=group["timestamps"][:],
                    valid=group["valid"][:],
                    positions=group["positions"][:] if "positions" in group else None,
                    localization_ids=(
                        group["localization_ids"][:]
                        if "localization_ids" in group
                        else None
                    ),
                    measurements=(
                        _read_columns(group["measurements"])
                        if "measurements" in group
                        else {}
                    ),
                )
                dataset.photon_sequences._data[sequence.id] = sequence
                if stale:
                    dataset.photon_sequences._stale.add(sequence.id)
            dataset.photon_sequences.revision = int(
                file["photon_sequences"].attrs["revision"]
            )
        if "photon_crops" in file:
            group = file["photon_crops"]
            width = int(group.attrs["width"])
            for index, (sequence_id, sample_index) in enumerate(
                zip(
                    group["sequence_ids"].asstr()[:],
                    group["sample_indices"][:],
                    strict=True,
                )
            ):
                dataset.crops._photon_saved[sequence_id, int(sample_index), width] = (
                    str(path),
                    index,
                    group.name,
                )
    report(progress, cancel_event, 1, 1, "Dataset opened")
    return dataset


@logged
def save_experiment(
    experiment: Experiment,
    path: str | Path,
    *,
    layout: str = "hdf5",
    image_policy: str = "none",
    options: SaveOptions | None = None,
    crop_width: int = 7,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
    publish_guard: Any = None,
) -> None:
    """Atomically save datasets and common-view registrations in one snapshot.

    Movie embedding always stores original camera pixels. Explicit TIFF exports
    are the route for writing resampled images.

    Args:
        experiment: Experiment containing the probing rounds.
        path: Snapshot file or output directory.
        layout: hdf5 or directory snapshot layout.
        image_policy: Microscopy pixels to embed; original frames are retained unchanged.
        options: Independent original-image and crop choices, replacing legacy arguments.
        crop_width: Positive odd width of stored crops.
        progress: Optional structured progress callback.
        cancel_event: Optional cooperative cancellation event.
    """
    from contextlib import ExitStack

    from .files import report

    if layout not in ("hdf5", "directory"):
        raise ValueError("Layout must be hdf5 or directory")
    policy = ImagePolicy(image_policy)
    strict_external = options is not None
    options = _resolve_options(options, policy, crop_width, layout)
    crop_width = options.crop_width
    destination = Path(path).resolve()
    if layout == "directory":
        destination /= "experiment.h5"
    report(progress, cancel_event, 0, 1, "Saving sequencing experiment")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".molcore-", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        with experiment._lock, ExitStack() as stack:
            for dataset in sorted(experiment.datasets.values(), key=lambda d: d.id):
                stack.enter_context(dataset._lock)
            with h5py.File(temporary, "w") as file:
                file.attrs["format"], file.attrs["format_version"] = (
                    "molcore",
                    FORMAT_VERSION,
                )
                file.attrs["kind"] = "experiment"
                file.attrs["metadata"] = _json(experiment.metadata)
                file.attrs["alignments"] = _json(
                    [asdict(r) for r in experiment.alignment._records.values()]
                )
                file.attrs["alignment_revision"] = experiment.alignment.revision
                datasets = file.create_group("datasets", track_order=True)
                for index, dataset in enumerate(experiment.datasets.values()):
                    _write_dataset_group(
                        datasets.create_group(str(index)),
                        dataset,
                        destination,
                        layout=layout,
                        policy=policy,
                        options=options,
                        strict_external=strict_external,
                        crop_width=crop_width,
                        progress=progress,
                        cancel_event=cancel_event,
                    )
                file.flush()
            report(progress, cancel_event, 1, 1, "Committing experiment snapshot")
            with publish_guard() if publish_guard is not None else nullcontext():
                os.replace(temporary, destination)
                restored = open_experiment(destination)
                for key, dataset in experiment.datasets.items():
                    saved = restored.datasets[key]
                    for name in ("_saved", "_photon_saved"):
                        cache = getattr(dataset.crops, name)
                        cache = {
                            k: v
                            for k, v in cache.items()
                            if Path(v[0]).resolve() != destination
                        }
                        cache.update(getattr(saved.crops, name))
                        setattr(dataset.crops, name, cache)
                    for aid, acquisition in dataset.acquisitions.items():
                        source = acquisition.source
                        while isinstance(source, CachedSource):
                            source = source.source
                        if (
                            isinstance(source, HDF5Source)
                            and Path(source.path).resolve() == destination
                        ):
                            acquisition.source = saved.acquisitions[aid].source
    finally:
        Path(temporary).unlink(missing_ok=True)


@logged
def open_experiment(
    path: str | Path,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> Experiment:
    """Restore one surface experiment and lazy nested dataset sources.

    Args:
        path: Snapshot file or output directory.
        progress: Optional structured progress callback.
        cancel_event: Optional cooperative cancellation event.

    Returns:
        Result of this operation.
    """
    from .files import report
    from .types.dataset_alignment import (
        DatasetAlignmentResult,
        DatasetAlignmentSettings,
    )
    from .types.experiment import Experiment

    report(progress, cancel_event, 0, 1, "Opening experiment")
    path = Path(path).resolve()
    if path.is_dir():
        path /= "experiment.h5"
    with h5py.File(path, "r") as file:
        if (
            file.attrs.get("format") != "molcore"
            or file.attrs.get("kind") != "experiment"
            or file.attrs.get("format_version") != FORMAT_VERSION
        ):
            raise ValueError(
                "Expected a supported Experiment snapshot; use Dataset.open for datasets"
            )
        experiment = Experiment(metadata=json.loads(file.attrs["metadata"]))
        for group in file["datasets"].values():
            experiment.add_dataset(
                open_dataset(path, group=group.name, cancel_event=cancel_event)
            )
        for value in json.loads(file.attrs["alignments"]):
            value["settings"] = DatasetAlignmentSettings(**value["settings"])
            value["fingerprints"] = tuple(value["fingerprints"])
            for key in (
                "matrix",
                "residuals",
                "inliers",
                "source_ids",
                "reference_ids",
            ):
                dtype = (
                    bool
                    if key == "inliers"
                    else np.uint64 if key.endswith("ids") else float
                )
                value[key] = np.asarray(value[key], dtype=dtype)
            result = DatasetAlignmentResult(**value)
            experiment.alignment._records[result.source_dataset_id] = result
            experiment.alignment.reference_dataset_id = result.reference_dataset_id
            experiment.alignment.reference_acquisition_id = (
                result.reference_acquisition_id
            )
            experiment.alignment.reference_channel = result.reference_channel
        experiment.alignment.revision = int(file.attrs["alignment_revision"])
    report(progress, cancel_event, 1, 1, "Experiment opened")
    return experiment


@logged
def read_crop(path: str | Path, index: int, group_name: str = "crops") -> Crop:
    """Read one saved crop without loading neighbouring crop pixels.

    Args:
        path: Snapshot containing a crop dataset.
        index: Row index in the saved crop dataset.
        group_name: Localisation or independent photon crop dataset group.

    Returns:
        Crop with raw-channel origin, validity and source provenance.
    """
    with h5py.File(path, "r") as file:
        group = file[group_name]
        return Crop(
            group["pixels"][index],
            group["valid"][index],
            group["acquisitions"].asstr()[index],
            group["channels"].asstr()[index],
            int(group["frames"][index]),
            tuple(int(v) for v in group["origins"][index]),
            int(group["ids"][index]) or None,
            group["sequence_ids"].asstr()[index] if "sequence_ids" in group else None,
            int(group["sample_indices"][index]) if "sample_indices" in group else None,
        )
