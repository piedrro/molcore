"""Non-destructive registration, drift and image resampling."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from ._changes import mutation_scope
from ._locking import model_mutated
from ._logging import logged
from .types._shared import Calibration, CoordinateSpace
from .types.images import Source
from .types.tables import partition_keys

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike

    from .types import Dataset


def project_points(matrix: ArrayLike, points: ArrayLike) -> np.ndarray:
    """Apply a homogeneous transform, returning NaN at projective poles.

    Args:
        matrix: Homogeneous 3x3 coordinate mapping.
        points: Input coordinates with shape (..., 2).

    Returns:
        Projected coordinates with the same shape as points.
    """
    matrix, points = (np.asarray(matrix, dtype=float), np.asarray(points, dtype=float))
    if matrix.shape != (3, 3) or points.shape[-1:] != (2,):
        raise ValueError("Expected a 3x3 matrix and (..., 2) points")
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    mapped = homogeneous @ matrix.T
    scale = np.sum(np.abs(homogeneous * matrix[2]), axis=-1)
    valid = np.isfinite(mapped).all(axis=-1) & (np.abs(mapped[..., 2]) > 1e-12 * scale)
    result = np.full(points.shape, np.nan)
    np.divide(mapped[..., :2], mapped[..., 2, None], out=result, where=valid[..., None])
    return result


def project_jacobian(matrix: ArrayLike, points: ArrayLike) -> np.ndarray:
    """Evaluate local derivatives for uncertainty propagation.

    Args:
        matrix: Homogeneous 3x3 coordinate mapping.
        points: Raw coordinates with shape (..., 2).

    Returns:
        Local Jacobian matrices with shape (..., 2, 2).
    """
    matrix, points = (np.asarray(matrix, dtype=float), np.asarray(points, dtype=float))
    mapped = project_points(matrix, points)
    denominator = points @ matrix[2, :2] + matrix[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (matrix[:2, :2] - mapped[..., :, None] * matrix[2, :2]) / denominator[..., None, None]


class TransformManager:
    """Compose per-channel affine registrations and per-camera-frame drift.

    Args:
        dataset: Owner used to validate references and invalidate measurements.
    """

    def __init__(self, dataset: Dataset) -> None:
        """Initialise the object and validate its configuration.

        Args:
            dataset: Scientific model supplying acquisitions and result tables.
        """
        self.dataset = dataset
        self._calibrations, self._selected, self._drift = ({}, {}, {})
        self.revision = 0

    @property
    @logged
    def calibrations(self) -> Mapping[str, Calibration]:
        """Read-only mapping of named registration calibrations.

        Returns:
            Computed Mapping[str, Calibration] value.
        """
        return MappingProxyType(self._calibrations)

    def _changed(self, acquisition_id: str) -> None:
        """Changed.

        Args:
            acquisition_id: Source acquisition identifier.
        """
        self.revision += 1
        self.dataset.photon_sequences._invalidate(acquisition_id=acquisition_id)
        self.dataset.observations._invalidate_partition(acquisition_id)

    @model_mutated(scope=lambda self, calibration: mutation_scope(self))
    @logged
    def add_calibration(self, calibration: Calibration) -> None:
        """Store a named calibration without selecting it automatically.

        Args:
            calibration: Validated Calibration value with a 3x3 affine matrix.
        """
        self._calibrations[calibration.name] = Calibration(
            calibration.name, calibration.matrix, deepcopy(calibration.metadata)
        )
        self._refresh_selected_groups(calibration.name)
        for (acquisition, _), name in self._selected.items():
            if name == calibration.name:
                self._changed(acquisition)

    def _refresh_selected_groups(self, calibration_name: str) -> None:
        """Keep portable calibration provenance in sync with active path selections."""
        groups = {}
        for (acquisition_id, channel), selected_name in self._selected.items():
            if selected_name != calibration_name:
                continue
            spec = self.dataset.acquisitions[acquisition_id].channel_specs[channel]
            key = (acquisition_id, spec.emission, channel if spec.emission is None else "")
            groups.setdefault(key, []).append(channel)
        selected_groups = [
            {"acquisition_id": acquisition_id, "emission": emission, "channels": sorted(channels)}
            for (acquisition_id, emission, _), channels in sorted(
                groups.items(), key=lambda item: (item[0][0], item[0][1] or "", item[0][2])
            )
        ]
        old = self._calibrations[calibration_name]
        self._calibrations[calibration_name] = Calibration(
            old.name, old.matrix, {**old.metadata, "selected_groups": selected_groups}
        )

    @model_mutated(
        scope=lambda self, acquisition_id, channel, calibration_name: mutation_scope(
            self, acquisition=acquisition_id, channel=None
        )
    )
    @logged
    def select(self, acquisition_id: str, channel: str, calibration_name: str) -> None:
        """Choose registration into a shared reference coordinate system.

        Args:
            acquisition_id: Registered acquisition identifier.
            channel: Logical channel name.
            calibration_name: Existing named calibration.
        """
        if channel not in self.dataset.acquisitions[acquisition_id].channels:
            raise ValueError("Unknown channel")
        if calibration_name not in self._calibrations:
            raise ValueError("Unknown calibration")
        acquisition = self.dataset.acquisitions[acquisition_id]
        emission = acquisition.channel_specs[channel].emission
        channels = (
            [channel]
            if emission is None
            else [name for name, spec in acquisition.channel_specs.items() if spec.emission == emission]
        )
        calibration = self._calibrations[calibration_name]
        metadata = calibration.metadata
        source_emission = metadata.get("source_emission")
        if source_emission is not None and source_emission != emission:
            raise ValueError("Calibration emission path does not match selected channel")
        if (
            "source_emission" in metadata
            and source_emission is None
            and (metadata.get("source_channel", channel) != channel)
        ):
            raise ValueError("Custom channel calibration belongs to a different channel")
        geometry = metadata.get("source_shape")
        if geometry is not None and any(
            tuple(geometry) != acquisition.channels[c].shape[1:] for c in channels
        ):
            raise ValueError("Calibration source geometry does not match selected emission path")
        source_roi = metadata.get("source_roi")
        if source_roi is not None and any(tuple(source_roi) != acquisition.channels[c].roi for c in channels):
            raise ValueError("Calibration source geometry does not match selected emission path")
        if "pixel_size_nm" in metadata and metadata["pixel_size_nm"] != acquisition.pixel_size:
            raise ValueError("Calibration pixel scale does not match acquisition")
        target_emission = metadata.get("target_emission")
        target_channels = (
            [name for name, spec in acquisition.channel_specs.items() if spec.emission == target_emission]
            if target_emission is not None
            else [metadata["target_channel"]]
            if "target_channel" in metadata
            else []
        )
        if target_emission is not None and (not target_channels):
            raise ValueError("Calibration reference geometry is absent from acquisition")
        for target_name in target_channels:
            if target_name not in acquisition.channels:
                raise ValueError("Calibration reference channel is absent from acquisition")
            target_view = acquisition.channels[target_name]
            if (
                "target_shape" in metadata
                and tuple(metadata["target_shape"]) != target_view.shape[1:]
                or ("target_roi" in metadata and tuple(metadata["target_roi"]) != target_view.roi)
            ):
                raise ValueError("Calibration reference geometry does not match acquisition")
        for name in channels:
            previous = self._selected.get((acquisition_id, name))
            self._selected[acquisition_id, name] = calibration_name
            if previous is not None and previous != calibration_name:
                self._refresh_selected_groups(previous)
        self._refresh_selected_groups(calibration_name)
        self._changed(acquisition_id)

    @model_mutated(
        scope=lambda self, acquisition_id, displacement, *, frames=None: mutation_scope(
            self, acquisition=acquisition_id, channel=None
        )
    )
    @logged
    def set_drift(
        self, acquisition_id: str, displacement: ArrayLike, *, frames: ArrayLike | None = None
    ) -> None:
        """Set observed (dx, dy) drift in reference pixels at explicit frames.

        Args:
            acquisition_id: Registered acquisition identifier.
            displacement: Array of shape (n, 2) in reference-pixel coordinates.
            frames: Original camera frames; omitted means consecutive frames at zero.

        Raises:
            ValueError: Frames are duplicate, out of range, or displacements invalid.
        """
        n_frames = self.dataset.acquisitions[acquisition_id].source.shape[0]
        displacement = np.array(displacement, dtype=float, copy=True)
        frames = np.arange(len(displacement)) if frames is None else np.asarray(frames)
        if (
            displacement.shape != (len(frames), 2)
            or not np.isfinite(displacement).all()
            or frames.dtype.kind not in "iu"
            or (len(np.unique(frames)) != len(frames))
            or np.any((frames < 0) | (frames >= n_frames))
        ):
            raise ValueError("Invalid drift frames or displacements")
        order = np.argsort(frames)
        self._drift[acquisition_id] = (frames[order].copy(), displacement[order].copy())
        self._changed(acquisition_id)

    def _registration(self, acquisition_id: str, channel: str) -> np.ndarray:
        """Registration.

        Args:
            acquisition_id: Source acquisition identifier.
            channel: Logical source channel name.

        Returns:
            Computed NumPy array value.
        """
        name = self._selected.get((acquisition_id, channel))
        return np.eye(3) if name is None else self._calibrations[name].matrix.copy()

    def _displacement(self, acquisition_id: str, frames: ArrayLike | None) -> np.ndarray:
        """Displacement.

        Args:
            acquisition_id: Source acquisition identifier.
            frames: Original camera-frame indices for the selected samples.

        Returns:
            Computed NumPy array value.
        """
        if acquisition_id not in self._drift:
            return np.zeros((len(frames), 2))
        available, displacement = self._drift[acquisition_id]
        indices = np.searchsorted(available, frames)
        if (
            not len(available)
            or np.any(indices >= len(available))
            or np.any(available[np.minimum(indices, len(available) - 1)] != frames)
        ):
            raise ValueError("No drift supplied at requested camera frames; extrapolation is disabled")
        return displacement[indices]

    @logged
    def matrix(
        self,
        acquisition_id: str,
        channel: str,
        camera_frame: int,
        *,
        space: CoordinateSpace | str = "corrected",
    ) -> np.ndarray:
        """Return the raw-channel to requested-coordinate affine matrix.

        Args:
            acquisition_id: Acquisition identifier.
            channel: Logical channel name.
            camera_frame: Original camera frame.
            space: raw, registered or corrected.

        Returns:
            Detached 3x3 affine matrix; corrected subtracts reference-space drift.
        """
        space = CoordinateSpace(space)
        self.dataset.acquisitions[acquisition_id].channels[channel].channel_index(camera_frame)
        matrix = np.eye(3) if space == "raw" else self._registration(acquisition_id, channel)
        if space == "corrected":
            translation = np.eye(3)
            translation[:2, 2] = -self._displacement(acquisition_id, np.array([camera_frame]))[0]
            matrix = translation @ matrix
        return matrix

    @logged
    def coordinates(
        self, rows: dict[str, np.ndarray], *, space: CoordinateSpace | str = "corrected"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform effective raw localization coordinates without editing them.

        Args:
            rows: Dictionary with x, y, acquisition_id, channel and frame columns.
            space: raw, registered or corrected.

        Returns:
            Tuple of transformed x and y arrays.
        """
        space = CoordinateSpace(space)
        result = np.column_stack([rows["x"], rows["y"]]).astype(float)
        if space == "raw":
            return (result[:, 0], result[:, 1])
        for a, c in partition_keys(rows, self.dataset):
            selected = (rows["acquisition_id"] == a) & (rows["channel"] == c)
            matrix = self._registration(a, c)
            result[selected] = project_points(matrix, result[selected])
            if space == "corrected":
                result[selected] -= self._displacement(a, rows["frame"][selected])
        return (result[:, 0], result[:, 1])

    @logged
    def convert_coordinates(
        self,
        rows: dict[str, np.ndarray],
        *,
        source_space: CoordinateSpace = CoordinateSpace.RAW,
        target_space: CoordinateSpace = CoordinateSpace.RAW,
        target_channel: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert 1D position columns between spaces and optionally channels.

        Positions are bound to each row's original camera frame. Cross-channel
        conversion uses registration even when both spaces are raw. No image is
        resampled. Frames must exist in both source and target channels.

        Args:
            rows: Named measurement arrays retaining stable IDs and source-frame provenance.
            source_space: Source space used by this operation.
            target_space: Target space used by this operation.
            target_channel: Logical target channel in the same acquisition.

        Returns:
            Computed tuple[NumPy array, NumPy array] value.
        """
        source_space, target_space = (CoordinateSpace(source_space), CoordinateSpace(target_space))
        required = ("x", "y", "frame", "channel", "acquisition_id")
        arrays = {name: np.asarray(rows[name]) for name in required}
        n = len(arrays["x"])
        if any(v.ndim != 1 or len(v) != n for v in arrays.values()):
            raise ValueError("Coordinate columns must have matching 1D shapes")
        if arrays["frame"].dtype.kind not in "iu" or np.any(arrays["frame"] < 0):
            raise ValueError("Camera frames must be nonnegative integers")
        points = np.column_stack([arrays["x"], arrays["y"]]).astype(float)
        if not np.isfinite(points).all():
            raise ValueError("Coordinates must be finite")
        output = np.empty_like(points)
        for i in range(n):
            a, c, frame = (arrays["acquisition_id"][i], arrays["channel"][i], arrays["frame"][i])
            target = c if target_channel is None else target_channel
            source_matrix = self.matrix(a, c, frame, space=source_space)
            destination = self.matrix(a, target, frame, space=target_space)
            conversion = np.linalg.inv(source_matrix)
            if target != c:
                conversion = np.linalg.solve(
                    self._registration(a, target), self._registration(a, c) @ conversion
                )
            output[i] = project_points(destination @ conversion, points[i])
        return (output[:, 0], output[:, 1])

    @logged
    def into_channel(self, rows: dict[str, np.ndarray], channel: str) -> tuple[np.ndarray, np.ndarray]:
        """Map raw coordinates through the reference into another channel.

        Args:
            rows: Effective raw localization columns.
            channel: Target logical channel name.

        Returns:
            Target-channel x and y arrays at the same physical time.
        """
        x, y = self.coordinates(rows, space="registered")
        result = np.column_stack([x, y])
        for acquisition_id in np.unique(rows["acquisition_id"]):
            selected = rows["acquisition_id"] == acquisition_id
            matrix = np.linalg.inv(self._registration(acquisition_id, channel))
            result[selected] = project_points(matrix, result[selected])
        return (result[:, 0], result[:, 1])

    @logged
    def image(
        self,
        acquisition_id: str,
        channel: str,
        camera_frame: int,
        *,
        space: CoordinateSpace | str = "corrected",
        labels: bool = False,
    ) -> np.ndarray:
        """Resample only one requested frame using inverse coordinate mapping.

        Args:
            acquisition_id: Acquisition identifier.
            channel: Logical channel name.
            camera_frame: Original camera frame, not a channel index.
            space: raw, registered or corrected.
            labels: Use nearest-neighbour sampling instead of linear interpolation.

        Returns:
            Resampled 2D image, retaining the channel's output geometry.
        """
        view = self.dataset.acquisitions[acquisition_id].channels[channel]
        image = view[view.channel_index(camera_frame)]
        matrix = self.matrix(acquisition_id, channel, camera_frame, space=space)
        if np.array_equal(matrix, np.eye(3)):
            return image.copy()
        try:
            from scipy.ndimage import map_coordinates
        except ImportError as exc:
            raise ImportError("SciPy is missing; reinstall MolCore with its runtime dependencies") from exc
        y, x = np.indices(image.shape, dtype=float)
        inverse = np.linalg.inv(matrix)
        raw = project_points(inverse, np.column_stack([x.ravel(), y.ravel()]))
        raw[~np.isfinite(raw)] = -1
        return map_coordinates(
            image if labels else image.astype(float),
            [raw[:, 1], raw[:, 0]],
            order=0 if labels else 1,
            mode="constant",
            cval=0,
        ).reshape(image.shape)

    @logged
    def view(
        self, acquisition_id: str, channel: str, *, space: CoordinateSpace | str = "corrected"
    ) -> Source:
        """Create a lazy, Dask-compatible transformed channel view.

        Args:
            acquisition_id: Acquisition identifier.
            channel: Logical channel name.
            space: Coordinate space to resample into.

        Returns:
            Lazy image source with one-frame evaluation.
        """
        return TransformedView(self, acquisition_id, channel, space)


class TransformedView(Source):
    """Lazy frame resampling using a TransformManager."""

    def __init__(
        self, manager: TransformManager, acquisition_id: str, channel: str, space: CoordinateSpace | str
    ) -> None:
        """Initialise the object and validate its configuration.

        Args:
            manager: Manager used by this operation.
            acquisition_id: Source acquisition identifier.
            channel: Logical source channel name.
            space: Coordinate space: raw channel pixels, registered reference or drift-corrected reference.
        """
        self.manager, self.acquisition_id, self.channel = (manager, acquisition_id, channel)
        self.space = CoordinateSpace(space)
        self.source = manager.dataset.acquisitions[acquisition_id].channels[channel]
        self.shape, self.dtype = (self.source.shape, np.dtype(float))

    def _frame(self, index: int) -> np.ndarray:
        """Read one original image frame from the bound source.

        Args:
            index: Zero-based item or logical-channel frame index.

        Returns:
            Computed NumPy array value.
        """
        return self.manager.image(
            self.acquisition_id, self.channel, int(self.source.source_frames[index]), space=self.space
        )
