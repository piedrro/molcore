"""Shared value types and protocols; no implementation or I/O imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .._logging import logged


class ChannelName(StrEnum):
    """Conventional channel names; custom strings remain valid everywhere."""

    IMAGE = "image"
    DONOR = "donor"
    ACCEPTOR = "acceptor"
    DD = "DD"
    DA = "DA"
    AD = "AD"
    AA = "AA"


class AcquisitionMode(StrEnum):
    """Acquisition intent, independent of its explicit channel mapping."""

    SINGLE = "single"
    FRET = "fret"
    ALEX = "alex"
    CUSTOM = "custom"


class CoordinateSpace(StrEnum):
    """Coordinate views relative to channel pixels or the registered reference."""

    RAW = "raw"
    REGISTERED = "registered"
    CORRECTED = "corrected"


class ImagePolicy(StrEnum):
    """Microscopy pixel content included in an dataset snapshot."""

    NONE = "none"
    LOCALIZATION_CROPS = "localization_crops"
    TRACK_CROPS = "track_crops"
    BOTH = "both"
    PHOTON_SEQUENCE_CROPS = "photon_sequence_crops"
    LOCALIZATION_AND_PHOTON_CROPS = "localization_and_photon_crops"
    TRACK_AND_PHOTON_CROPS = "track_and_photon_crops"
    ALL_CROPS = "all_crops"
    FULL = "full"


class ImageUnavailableError(RuntimeError):
    """Requested pixels were not saved, or their source is missing."""


class StaleResultError(RuntimeError):
    """A derived result no longer matches its inputs."""


@dataclass(frozen=True)
class ChannelSpec:
    """Logical channel selection applied to original camera frames.

    Attributes:
        roi: Half-open (y0, y1, x0, x1) rectangle, or the full camera image.
        excitation: Excitation name to select from the acquisition sequence.
        frame_start: First candidate camera frame before excitation filtering.
        frame_step: Spacing between candidate camera frames.
    """

    roi: tuple[int, int, int, int] | None = None  # y0, y1, x0, x1
    excitation: str | None = None
    frame_start: int = 0
    frame_step: int = 1
    emission: str | None = None

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if self.frame_start < 0 or self.frame_step < 1:
            raise ValueError("Channel start/step must be nonnegative/positive")
        if self.emission is not None and (not isinstance(self.emission, str) or not self.emission):
            raise ValueError("Emission must be a nonempty physical path label or None")
        if self.roi is not None:
            object.__setattr__(self, "roi", tuple(self.roi))
            y0, y1, x0, x1 = self.roi
            if min(y0, x0) < 0 or y1 <= y0 or x1 <= x0:
                raise ValueError("ROI must have positive extent")


@dataclass(frozen=True)
class FrameInfo:
    """Provenance for one logical-channel image.

    Attributes:
        acquisition_id: Original acquisition identifier.
        camera_frame: Zero-based original camera frame.
        channel: Logical channel name.
        channel_frame: Logical-channel index in the requested view.
        timestamp: Recorded time in seconds, if supplied.
        sequence_frame: Original frame plus explicit acquisition-sequence offset.
        cycle_index: Excitation-cycle index within this acquisition.
    """

    acquisition_id: str
    camera_frame: int
    channel: str
    channel_frame: int
    timestamp: float | None = None
    sequence_frame: int | None = None
    cycle_index: int | None = None


@runtime_checkable
class ImageSource(Protocol):
    """Minimal image interface implemented by process-safe source descriptors.

    Attributes:
        shape: Three-dimensional (time, y, x) extent.
        dtype: NumPy pixel dtype.
    """

    shape: tuple[int, int, int]
    dtype: np.dtype

    def __getitem__(self, key: Any) -> NDArray:
        """Getitem.

        Args:
            key: NumPy-style frame and optional spatial selection.

        Returns:
            Computed NDArray value.
        """
        ...

    @logged
    def close(self) -> None:
        """Close."""
        ...


@dataclass(frozen=True)
class Calibration:
    """Named invertible raw-channel to reference-channel projective calibration.

    Attributes:
        name: Calibration identifier used for selection.
        matrix: Homogeneous 3x3 matrix operating on (x, y, 1).
        metadata: Dates, source paths, microscope identifiers and correction factors.
    """

    name: str
    matrix: NDArray = field(default_factory=lambda: np.eye(3))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        matrix = np.array(self.matrix, dtype=float, copy=True)
        object.__setattr__(self, "matrix", matrix)
        if (
            self.matrix.shape != (3, 3)
            or not np.isfinite(self.matrix).all()
            or np.linalg.matrix_rank(self.matrix) < 3
        ):
            raise ValueError("Registration must be an invertible 3x3 projective matrix")
        # Immutable backing bytes also prevent callers re-enabling writeability.
        object.__setattr__(self, "matrix", np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(3, 3))


@dataclass(frozen=True)
class Crop:
    """Raw-channel pixels and provenance for one molecule/frame measurement.

    Attributes:
        pixels: Square crop including zero padding outside the source image.
        valid: Boolean mask distinguishing source pixels from padding.
        acquisition_id: Source acquisition identifier.
        channel: Logical channel containing these pixels.
        camera_frame: Original camera frame.
        origin: Upper-left (x, y) crop origin in raw channel pixels.
        localization_id: Optional localization anchor.
        photon_sequence_id: Optional independent photon-sequence identifier.
        sample_index: Zero-based sample within the photon sequence, when applicable.
    """

    pixels: NDArray
    valid: NDArray
    acquisition_id: str
    channel: str
    camera_frame: int
    origin: tuple[int, int]  # x, y in channel space
    localization_id: int | None = None
    photon_sequence_id: str | None = None
    sample_index: int | None = None


@dataclass
class ProcessingMetadata:
    """Reproducibility settings and installed package versions.

    Attributes:
        settings: Stage-specific JSON-compatible processing parameters.
        software: Distribution names and versions, collected without importing them.
    """

    settings: dict[str, Any] = field(default_factory=dict)
    software: dict[str, str] = field(default_factory=dict)


@dataclass
class Segmentation:
    """Static segmentation source, labels and spatial binding.

    Attributes:
        id: Unique segmentation identifier.
        acquisition_id: Acquisition to which the mask applies.
        labels: Nonnegative instance labels; zero is background.
        source_image: Optional image used to produce the mask, saved by default.
        channel: Logical channel defining mask geometry.
        space: Coordinate system of the mask.
        frame: Source camera frame defining the spatial binding.
        source_path: Original segmentation image filename/path.
        mask_path: Imported mask filename/path, if applicable.
        metadata: Segmentation settings and provenance.
    """

    id: str
    acquisition_id: str
    labels: NDArray
    source_image: NDArray | None = None
    channel: str = "image"
    space: CoordinateSpace = CoordinateSpace.RAW
    frame: int = 0
    source_path: str | None = None
    mask_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        self.labels = np.array(self.labels, copy=True)
        if self.labels.ndim != 2 or self.labels.dtype.kind not in "biu" or np.any(self.labels < 0):
            raise ValueError("Labels must be a nonnegative integer 2D array")
        self.labels.flags.writeable = False
        self.space = CoordinateSpace(self.space)
        if self.source_image is not None:
            self.source_image = np.array(self.source_image, copy=True)
            if self.source_image.shape != self.labels.shape:
                raise ValueError("Segmentation source and mask shapes must agree")
            self.source_image.flags.writeable = False
