"""Versioned HDF5 snapshots with optional relative external image references."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from .crops import validate_width
from .types._shared import Calibration, ChannelSpec, Crop, ImagePolicy, Segmentation
from .types.images import CachedSource, FitsSource, HDF5Source, MissingSource, TiffSource

FORMAT_VERSION = 1


def _json(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Metadata must be JSON-compatible, got {type(item).__name__}")

    return json.dumps(value, default=convert, ensure_ascii=False, allow_nan=False)


def _write_columns(group, data):
    # Numeric dataset names allow arbitrary user column names, including '/'.
    for index, (name, values) in enumerate(data.items()):
        values = np.asarray(values)
        if values.dtype.kind == "U":
            ds = group.create_dataset(str(index), data=values.astype(object), dtype=h5py.string_dtype())
        elif values.dtype.kind == "O":
            raise ValueError("Object-valued columns cannot be saved")
        else:
            ds = group.create_dataset(str(index), data=values)
        ds.attrs["name"], ds.attrs["numpy_dtype"] = name, values.dtype.str


def _read_columns(group):
    result = {}
    for ds in group.values():
        dtype = np.dtype(ds.attrs["numpy_dtype"])
        result[ds.attrs["name"]] = np.asarray(ds.asstr()[:] if dtype.kind == "U" else ds[:], dtype=dtype)
    return result


def _external(source, directory, destination):
    while isinstance(source, CachedSource):
        source = source.source
    if not isinstance(source, (TiffSource, FitsSource, HDF5Source)):
        return None
    if Path(source.path).resolve() == destination:
        return None  # Do not leave references into the file being replaced.
    try:
        path = os.path.relpath(source.path, directory)
    except ValueError:
        path = source.path  # Different Windows drive; still usable, not relocatable.
    spec = {"path": path, "kind": type(source).__name__}
    if isinstance(source, FitsSource):
        spec["hdu"] = source.hdu
    if isinstance(source, HDF5Source):
        spec["dataset"] = source.dataset
    return spec


def _load_external(spec, directory, shape, dtype):
    path = (directory / spec["path"]).resolve()
    if not path.is_file():
        return MissingSource(shape, dtype, f"External images unavailable: {path}")
    match spec["kind"]:
        case "TiffSource":
            source = TiffSource(path)
        case "FitsSource":
            source = FitsSource(path, spec["hdu"])
        case "HDF5Source":
            source = HDF5Source(path, spec["dataset"])
        case _:
            raise ValueError("Unsupported external image source")
    if source.shape != tuple(shape) or source.dtype != np.dtype(dtype):
        raise ValueError(f"External image geometry/dtype changed: {path}")
    return source


def _write_crops(file, experiment, policy, width):
    ids = set()
    if policy in (ImagePolicy.LOCALIZATION_CROPS, ImagePolicy.BOTH):
        ids.update(int(i) for i in experiment.localizations._data.get("id", []))
    if policy in (ImagePolicy.TRACK_CROPS, ImagePolicy.BOTH):
        for members in experiment.tracks._members.values():
            ids.update(int(i) for i in members)
    if not ids:
        return []
    ids = sorted(ids)
    group = file.create_group("crops")
    group.attrs["width"] = width
    # Different acquisitions may have different camera dtypes.
    dtype = np.result_type(*(a.source.dtype for a in experiment.acquisitions.values()))
    pixels = group.create_dataset(
        "pixels", (len(ids), width, width), dtype=dtype, chunks=(1, width, width), compression="gzip"
    )
    valid = group.create_dataset(
        "valid", (len(ids), width, width), dtype=bool, chunks=(1, width, width), compression="gzip"
    )
    frames, origins, acquisitions, channels = [], [], [], []
    for index, localization_id in enumerate(ids):
        crop = experiment.crops.localization(localization_id, width=width)
        pixels[index], valid[index] = crop.pixels, crop.valid
        frames.append(crop.camera_frame)
        origins.append(crop.origin)
        acquisitions.append(crop.acquisition_id)
        channels.append(crop.channel)
    group.create_dataset("ids", data=np.asarray(ids, dtype=np.uint64))
    group.create_dataset("frames", data=frames)
    group.create_dataset("origins", data=origins)
    group.create_dataset(
        "acquisitions", data=np.asarray(acquisitions, dtype=object), dtype=h5py.string_dtype()
    )
    group.create_dataset("channels", data=np.asarray(channels, dtype=object), dtype=h5py.string_dtype())
    return ids


def save_experiment(experiment, path, *, layout="hdf5", image_policy="none", crop_width=7):
    """Write an experiment snapshot without modifying the previous snapshot on failure.

    Args:
        experiment: Experiment to serialize.
        path: Destination file or directory.
        layout: hdf5 embeds selected content; directory may reference external images.
        image_policy: Raw-image inclusion policy.
        crop_width: Positive odd pixel width for saved crops.

    Raises:
        ValueError: Layout, metadata or crop settings are invalid.
        OSError: Writing or atomic replacement fails.
    """
    if layout not in ("hdf5", "directory"):
        raise ValueError("Layout must be hdf5 or directory")
    policy = ImagePolicy(image_policy)
    validate_width(crop_width)
    path = Path(path).resolve()
    destination = path / "experiment.h5" if layout == "directory" else path
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".molcore-", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        with h5py.File(temp_name, "w") as file:
            file.attrs["format"] = "molcore"
            file.attrs["format_version"] = FORMAT_VERSION
            file.attrs["metadata"] = _json(experiment.metadata)
            file.attrs["image_policy"] = policy.value
            file.attrs["processing"] = _json(asdict(experiment.processing))
            acquisitions = file.create_group("acquisitions")
            for index, acquisition in enumerate(experiment.acquisitions.values()):
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
                    "channel_specs": {k: asdict(v) for k, v in acquisition.channel_specs.items()},
                }
                if acquisition.timestamps is not None:
                    group.create_dataset("timestamps", data=acquisition.timestamps)
                if policy == ImagePolicy.FULL:
                    n, y, x = acquisition.source.shape
                    dataset = group.create_dataset(
                        "images",
                        shape=(n, y, x),
                        dtype=acquisition.source.dtype,
                        maxshape=(None, y, x),
                        chunks=(1, min(y, 256), min(x, 256)),
                        compression="gzip",
                    )
                    for frame in range(n):
                        dataset[frame] = acquisition.source[frame]
                elif layout == "directory":
                    info["external"] = _external(acquisition.source, destination.parent, destination)
                group.attrs["info"] = _json(info)
            file.attrs["sequences"] = _json(
                [
                    {
                        "id": s.id,
                        "acquisition_ids": [a.id for a in s.acquisitions],
                        "allow_cross_boundary_tracks": s.allow_cross_boundary_tracks,
                        "reference": s.reference,
                    }
                    for s in experiment.sequences.values()
                ]
            )
            for name in ("localizations", "observations"):
                table = getattr(experiment, name)
                group = file.create_group(name)
                group.attrs["next_id"], group.attrs["revision"] = table._next_id, table.revision
                _write_columns(group, table._data)
            tracks = file.create_group("tracks")
            tracks.attrs["metadata"] = _json(experiment.tracks.metadata)
            tracks.attrs["next_id"] = experiment.tracks._next_id
            tracks.attrs["revision"] = experiment.tracks.revision
            memberships = [(t, int(i)) for t, ids in experiment.tracks._members.items() for i in ids]
            tracks.create_dataset("memberships", data=np.asarray(memberships, dtype=np.uint64).reshape(-1, 2))
            transforms = file.create_group("transforms")
            transforms.attrs["revision"] = experiment.transforms.revision
            transforms.attrs["calibrations"] = _json([asdict(c) for c in experiment.calibrations.values()])
            transforms.attrs["selected"] = _json(
                [[a, c, name] for (a, c), name in experiment.transforms._selected.items()]
            )
            drift = transforms.create_group("drift")
            for index, (acquisition_id, (frames, displacement)) in enumerate(
                experiment.transforms._drift.items()
            ):
                group = drift.create_group(str(index))
                group.attrs["acquisition_id"] = acquisition_id
                group.create_dataset("frames", data=frames)
                group.create_dataset("displacement", data=displacement)
            segmentations = file.create_group("segmentations")
            for index, segmentation in enumerate(experiment.segmentations.values()):
                group = segmentations.create_group(str(index))
                info = {k: v for k, v in vars(segmentation).items() if k not in ("labels", "source_image")}
                group.attrs["info"] = _json(info)
                group.create_dataset("labels", data=segmentation.labels, compression="gzip")
                if segmentation.source_image is not None:
                    group.create_dataset("source", data=segmentation.source_image, compression="gzip")
            saved_crop_ids = _write_crops(file, experiment, policy, crop_width)
            file.flush()
        os.replace(temp_name, destination)
        # A replaced snapshot has new crop row indices and possibly no images.
        experiment.crops._saved = {
            key: value
            for key, value in experiment.crops._saved.items()
            if Path(value[0]).resolve() != destination
        }
        for index, localization_id in enumerate(saved_crop_ids):
            experiment.crops._saved[localization_id, crop_width] = (str(destination), index)
        for index, acquisition in enumerate(experiment.acquisitions.values()):
            source = acquisition.source
            while isinstance(source, CachedSource):
                source = source.source
            if isinstance(source, HDF5Source) and Path(source.path).resolve() == destination:
                acquisition.source = (
                    HDF5Source(destination, f"/acquisitions/{index}/images")
                    if policy == ImagePolicy.FULL
                    else MissingSource(source.shape, source.dtype)
                )
    finally:
        if Path(temp_name).exists():
            Path(temp_name).unlink()


def open_experiment(path):
    """Open a supported snapshot, eagerly restoring tables and lazily reading pixels.

    Args:
        path: HDF5 snapshot file or dataset directory.

    Returns:
        Restored Experiment with process-safe image and crop descriptors.

    Raises:
        ValueError: The format or version is unsupported.
    """
    from .types.acquisition import Acquisition
    from .types.experiment import Experiment

    path = Path(path).resolve()
    if path.is_dir():
        path = path / "experiment.h5"
    with h5py.File(path, "r") as file:
        if file.attrs.get("format") != "molcore" or file.attrs.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported MolCore format/version")
        experiment = Experiment(metadata=json.loads(file.attrs["metadata"]))
        processing = json.loads(file.attrs["processing"])
        experiment.processing.settings = processing["settings"]
        experiment.processing.software = processing["software"]
        for group in file["acquisitions"].values():
            info = json.loads(group.attrs["info"])
            if "images" in group:
                source = HDF5Source(path, group["images"].name)
            elif info.get("external"):
                source = _load_external(info["external"], path.parent, info["shape"], info["dtype"])
            else:
                source = MissingSource(info["shape"], info["dtype"])
            acquisition = Acquisition(
                source,
                id=info["id"],
                mode=info["mode"],
                channel_specs={k: ChannelSpec(**v) for k, v in info["channel_specs"].items()},
                excitation_sequence=info["excitation_sequence"],
                phase=info["phase"],
                timestamps=group["timestamps"][:] if "timestamps" in group else None,
                pixel_size=info["pixel_size"],
                metadata=info["metadata"],
            )
            experiment.add_acquisition(acquisition)
        for sequence in json.loads(file.attrs["sequences"]):
            experiment.add_sequence(**sequence)
        for name in ("localizations", "observations"):
            table, group = getattr(experiment, name), file[name]
            table._data = _read_columns(group)
            table._next_id, table.revision = int(group.attrs["next_id"]), int(group.attrs["revision"])
        tracks = file["tracks"]
        experiment.tracks.metadata = {int(k): v for k, v in json.loads(tracks.attrs["metadata"]).items()}
        membership = tracks["memberships"][:]
        for track in experiment.tracks.metadata:
            experiment.tracks._members[track] = membership[membership[:, 0] == track, 1]
        experiment.tracks._next_id = int(tracks.attrs["next_id"])
        experiment.tracks.revision = int(tracks.attrs["revision"])
        transforms = file["transforms"]
        for value in json.loads(transforms.attrs["calibrations"]):
            calibration = Calibration(**value)
            experiment.transforms._calibrations[calibration.name] = calibration
        experiment.transforms._selected = {
            (a, c): name for a, c, name in json.loads(transforms.attrs["selected"])
        }
        for group in transforms["drift"].values():
            experiment.transforms._drift[group.attrs["acquisition_id"]] = (
                group["frames"][:],
                group["displacement"][:],
            )
        experiment.transforms.revision = int(transforms.attrs["revision"])
        for group in file["segmentations"].values():
            segmentation = Segmentation(
                **json.loads(group.attrs["info"]),
                labels=group["labels"][:],
                source_image=group["source"][:] if "source" in group else None,
            )
            experiment.segmentations[segmentation.id] = segmentation
        if "crops" in file:
            width = int(file["crops"].attrs["width"])
            for index, localization_id in enumerate(file["crops/ids"][:]):
                experiment.crops._saved[int(localization_id), width] = (str(path), index)
    return experiment


def read_crop(path, index):
    """Read one saved crop without loading neighbouring crop pixels.

    Args:
        path: Snapshot containing a crop dataset.
        index: Row index in the saved crop dataset.

    Returns:
        Crop with raw-channel origin, validity and source provenance.
    """
    with h5py.File(path, "r") as file:
        group = file["crops"]
        return Crop(
            group["pixels"][index],
            group["valid"][index],
            group["acquisitions"].asstr()[index],
            group["channels"].asstr()[index],
            int(group["frames"][index]),
            tuple(int(v) for v in group["origins"][index]),
            int(group["ids"][index]),
        )
