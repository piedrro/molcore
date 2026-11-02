"""Restricted, explicit migration of supported MolTrack scientific records."""

from __future__ import annotations

import builtins
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from .files import image_acquisition, import_table, read_image, report
from .interop import from_records
from .types import Dataset, Segmentation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .types import Acquisition, CancelEvent, ProgressCallback


class _RecordsOnly(pickle.Unpickler):
    @logged
    def find_class(self, module: str, name: str) -> Any:
        """Find class.

        Args:
            module: Module used by this operation.
            name: Identifier or field name used to address this value.

        Returns:
            Whitelisted class used by the restricted legacy unpickler.
        """
        if module == "builtins" and name in {"set", "frozenset", "slice", "complex"}:
            return getattr(builtins, name)
        if module in {
            "numpy",
            "numpy.core.multiarray",
            "numpy._core.multiarray",
            "numpy.core.numeric",
            "numpy._core.numeric",
        }:
            allowed = {
                "ndarray": np.ndarray,
                "dtype": np.dtype,
                "recarray": np.recarray,
                "_reconstruct": np._core.multiarray._reconstruct,
                "scalar": np._core.multiarray.scalar,
                "_frombuffer": np._core.numeric._frombuffer,
            }
            if name in allowed:
                return allowed[name]
        raise ValueError(
            f"Unsupported legacy Python object {module}.{name}; export plain tables/images first"
        )


@logged
def read_moltrack(path: str | Path) -> dict[str, Any]:
    """Decode plain dictionaries and NumPy records without loading plugin classes.

    Args:
        path: Legacy .moltrack project file.

    Returns:
        Project dictionary for a migration preview.

    Raises:
        ValueError: Unsupported Python objects or project structure are present.
    """
    with Path(path).open("rb") as stream:
        value = _RecordsOnly(stream).load()
    if not isinstance(value, dict):
        raise ValueError("Unsupported legacy project; expected a dictionary")
    return value


@logged
def migrate_moltrack(
    project: Any,
    *,
    acquisitions: Mapping[str, Acquisition] | None = None,
    channel_mapping: Mapping[tuple[str, str], str] | None = None,
    timeline: str = "channel",
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> tuple[Dataset, list[str]]:
    """Build a detached Dataset using explicit original-camera bindings.

    Args:
        project: Dictionary returned by read_moltrack.
        acquisitions: Dataset-name to Acquisition mapping supplied by the importer.
            If omitted, only original image paths with known single-channel layout
            can be recovered automatically; cropped channel stacks need binding.
        channel_mapping: Mapping of (legacy dataset, channel) to new channel name.
        timeline: Explicit legacy table timeline, camera or channel.
        progress: Optional progress callback.
        cancel_event: Optional cancellation event.

    Returns:
        Tuple of converted Dataset and human-readable migration report lines.
    """
    result, messages = Dataset(metadata={"import_format": "moltrack"}), []
    datasets = project.get("dataset_dict", {})
    bindings = dict(acquisitions or {})
    for name, data in datasets.items():
        report(progress, cancel_event, description=f"Binding legacy dataset {name}")
        if name not in bindings:
            path = data.get("path")
            mode = str(data.get("multichannel_mode", "")).lower()
            if (
                not isinstance(path, str)
                or not Path(path).is_file()
                or mode
                not in (
                    "single channel",
                    "single",
                    "single-channel",
                )
            ):
                raise ValueError(
                    f"Dataset {name!r} needs original images or an explicit lossless acquisition binding"
                )
            channels = tuple(data.get("images", {}))
            channel = (
                channels[0] if len(channels) == 1 else data.get("channel_name", "image")
            )
            bindings[name] = image_acquisition(
                read_image(path),
                channel=channel,
                pixel_size=data.get("pixel_size"),
                id=name,
            )
        result.add_acquisition(bindings[name])
    mapping = channel_mapping or {}
    for kind, key in (
        ("localisation_dict", "localisations"),
        ("tracking_dict", "tracks"),
    ):
        for dataset, channels in project.get(kind, {}).items():
            if dataset not in bindings:
                raise ValueError(
                    f"Missing acquisition binding for legacy dataset {dataset!r}"
                )
            for channel, data in channels.items():
                if key not in data:
                    continue
                records = data[key]
                columns = (
                    from_records(records)
                    if isinstance(records, np.ndarray)
                    else {k: np.asarray(v) for k, v in records.items()}
                )
                target = mapping.get((dataset, channel), channel)
                import_table(
                    result,
                    bindings[dataset].id,
                    target,
                    columns,
                    timeline=timeline,
                    tracks=kind == "tracking_dict",
                )
                messages.append(
                    f"Imported {len(columns['frame'])} {key}: {dataset}/{channel}"
                )
    mask = project.get("segmentations")
    if mask and len(bindings) == 1:
        acquisition = next(iter(bindings.values()))
        channel = next(iter(acquisition.channels))
        view = acquisition.channels[channel]
        labels = np.zeros(view.shape[1:], dtype=np.uint32)
        # Pixel centres and (y,x) polygons follow napari's shape convention.
        yy, xx = np.indices(labels.shape)
        for index, polygon in enumerate(mask.get("shapes", []), start=1):
            polygon = np.asarray(polygon, dtype=float)
            if polygon.ndim != 2 or polygon.shape[1] != 2:
                messages.append("Skipped non-2D legacy shape")
                continue
            inside = np.zeros(labels.shape, dtype=bool)
            previous = polygon[-1]
            for current in polygon:
                y0, x0 = previous
                y1, x1 = current
                if y1 != y0:
                    inside ^= ((y0 > yy) != (y1 > yy)) & (
                        xx < (x1 - x0) * (yy - y0) / (y1 - y0) + x0
                    )
                previous = current
            labels[inside] = index
        if len(view):
            source = project.get("segmentation_image", {}).get("data")
            if source is not None and np.asarray(source).shape != labels.shape:
                messages.append(
                    "Segmentation source geometry differs; source not bound to this mask"
                )
                source = None
            result.add_segmentation(
                Segmentation(
                    "legacy",
                    acquisition.id,
                    labels,
                    source,
                    channel=channel,
                    frame=int(view.source_frames[0]),
                )
            )
    elif mask:
        messages.append(
            "Legacy shapes need an explicit acquisition binding; masks were not imported"
        )
    for key in project.keys() - {
        "dataset_dict",
        "localisation_dict",
        "tracking_dict",
        "segmentations",
        "segmentation_image",
    }:
        messages.append(f"Not represented by MolCore: {key}")
    result.metadata["migration_report"] = messages
    report(progress, cancel_event, 1, 1, "Legacy migration staged")
    return result, messages
