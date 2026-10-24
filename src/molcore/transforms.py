"""Non-destructive registration, drift and image resampling."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

import numpy as np

from .types._shared import Calibration, CoordinateSpace
from .types.images import Source


class TransformManager:
    """Compose per-channel affine registrations and per-camera-frame drift.

    Args:
        experiment: Owner used to validate references and invalidate measurements.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        self._calibrations, self._selected, self._drift = {}, {}, {}
        self.revision = 0

    @property
    def calibrations(self):
        """Read-only mapping of named registration calibrations."""
        return MappingProxyType(self._calibrations)

    def _changed(self, acquisition_id):
        self.revision += 1
        self.experiment.observations._invalidate_partition(acquisition_id)

    def add_calibration(self, calibration: Calibration) -> None:
        """Store a named calibration without selecting it automatically.

        Args:
            calibration: Validated Calibration value with a 3x3 affine matrix.
        """
        self._calibrations[calibration.name] = Calibration(
            calibration.name, calibration.matrix, deepcopy(calibration.metadata)
        )
        for (acquisition, _), name in self._selected.items():
            if name == calibration.name:
                self._changed(acquisition)

    def select(self, acquisition_id: str, channel: str, calibration_name: str) -> None:
        """Choose registration into a shared reference coordinate system.

        Args:
            acquisition_id: Registered acquisition identifier.
            channel: Logical channel name.
            calibration_name: Existing named calibration.
        """
        if channel not in self.experiment.acquisitions[acquisition_id].channels:
            raise ValueError("Unknown channel")
        if calibration_name not in self._calibrations:
            raise ValueError("Unknown calibration")
        self._selected[acquisition_id, channel] = calibration_name
        self._changed(acquisition_id)

    def set_drift(self, acquisition_id, displacement, *, frames=None):
        """Set observed (dx, dy) drift in reference pixels at explicit frames.

        Args:
            acquisition_id: Registered acquisition identifier.
            displacement: Array of shape (n, 2) in reference-pixel coordinates.
            frames: Original camera frames; omitted means consecutive frames at zero.

        Raises:
            ValueError: Frames are duplicate, out of range, or displacements invalid.
        """
        n_frames = self.experiment.acquisitions[acquisition_id].source.shape[0]
        displacement = np.array(displacement, dtype=float, copy=True)
        frames = np.arange(len(displacement)) if frames is None else np.asarray(frames)
        if (
            displacement.shape != (len(frames), 2)
            or not np.isfinite(displacement).all()
            or frames.dtype.kind not in "iu"
            or len(np.unique(frames)) != len(frames)
            or np.any((frames < 0) | (frames >= n_frames))
        ):
            raise ValueError("Invalid drift frames or displacements")
        order = np.argsort(frames)
        self._drift[acquisition_id] = (frames[order].copy(), displacement[order].copy())
        self._changed(acquisition_id)

    def _registration(self, acquisition_id, channel):
        name = self._selected.get((acquisition_id, channel))
        return np.eye(3) if name is None else self._calibrations[name].matrix.copy()

    def _displacement(self, acquisition_id, frames):
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

    def matrix(self, acquisition_id, channel, camera_frame, *, space="corrected"):
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
        self.experiment.acquisitions[acquisition_id].channels[channel].channel_index(camera_frame)
        matrix = np.eye(3) if space == "raw" else self._registration(acquisition_id, channel)
        if space == "corrected":
            matrix[:2, 2] -= self._displacement(acquisition_id, np.array([camera_frame]))[0]
        return matrix

    def coordinates(self, rows, *, space="corrected"):
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
            return result[:, 0], result[:, 1]
        for a, c in set(zip(rows["acquisition_id"], rows["channel"], strict=True)):
            selected = (rows["acquisition_id"] == a) & (rows["channel"] == c)
            matrix = self._registration(a, c)
            result[selected] = result[selected] @ matrix[:2, :2].T + matrix[:2, 2]
            if space == "corrected":
                result[selected] -= self._displacement(a, rows["frame"][selected])
        return result[:, 0], result[:, 1]

    def into_channel(self, rows, channel):
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
            result[selected] = result[selected] @ matrix[:2, :2].T + matrix[:2, 2]
        return result[:, 0], result[:, 1]

    def image(self, acquisition_id, channel, camera_frame, *, space="corrected", labels=False):
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
        view = self.experiment.acquisitions[acquisition_id].channels[channel]
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
        raw = inverse @ np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
        return map_coordinates(
            image if labels else image.astype(float),
            [raw[1], raw[0]],
            order=0 if labels else 1,
            mode="constant",
            cval=0,
        ).reshape(image.shape)

    def view(self, acquisition_id, channel, *, space="corrected"):
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

    def __init__(self, manager, acquisition_id, channel, space):
        self.manager, self.acquisition_id, self.channel = manager, acquisition_id, channel
        self.space = CoordinateSpace(space)
        self.source = manager.experiment.acquisitions[acquisition_id].channels[channel]
        self.shape, self.dtype = self.source.shape, np.dtype(float)

    def _frame(self, index):
        return self.manager.image(
            self.acquisition_id, self.channel, int(self.source.source_frames[index]), space=self.space
        )
