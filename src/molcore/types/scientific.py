"""Typed settings and detached results for shared scientific processing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .._logging import logged
from ..revisions import FreshnessToken
from ._shared import CoordinateSpace
from .processing import ProgressUpdate

if TYPE_CHECKING:
    from ..types import ProgressCallback
    from .photons import PhotonSequence


class CancelEvent(Protocol):
    """Cancellation interface accepted by processing APIs."""

    @logged
    def is_set(self) -> bool:
        """Is set.

        Returns:
            Computed bool value.
        """
        ...


ProgressCallback = Callable[[ProgressUpdate], None]


class ApertureShape(StrEnum):
    SQUARE = "square"
    CIRCLE = "circle"


class BackgroundMethod(StrEnum):
    LOCAL = "local"
    GLOBAL = "global"


class TraceMode(StrEnum):
    LOCALIZATIONS = "localizations"
    STATIONARY = "stationary"
    TRACK = "track"


class SpatialUnit(StrEnum):
    PIXELS = "pixels"
    NANOMETRES = "nm"


@dataclass(frozen=True)
class PhotometrySettings:
    """Apertures in target-channel pixels; circle radius is spot_size / 2.

    Background rings extend outside the aperture plus buffer. Both local and
    global summaries are always returned; background selects corrected intensity.
    Exclusion masks remove known signal apertures from background regions only.
    """

    spot_size: int = 3
    shape: ApertureShape = ApertureShape.SQUARE
    buffer: int = 1
    background_width: int = 1
    background: BackgroundMethod = BackgroundMethod.LOCAL
    exclude_neighbours: bool = True
    follow_drift: bool = True

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        for name, minimum in (("spot_size", 1), ("buffer", 0), ("background_width", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.spot_size % 2 != 1:
            raise ValueError("spot_size must be odd")
        if not isinstance(self.shape, ApertureShape) or not isinstance(
            self.background, BackgroundMethod
        ):
            raise TypeError("Use ApertureShape and BackgroundMethod enums")
        if not isinstance(self.exclude_neighbours, bool) or not isinstance(
            self.follow_drift, bool
        ):
            raise TypeError("Photometry flags must be booleans")


@dataclass(frozen=True)
class FixedApertureSettings:
    """Fixed mask geometry with separate window and masked-ring summaries."""

    spot_size: int = 3
    shape: ApertureShape = ApertureShape.SQUARE
    buffer: int = 1
    background_width: int = 1
    global_background: bool = False
    psf_box_size: int | None = None

    def __post_init__(self) -> None:
        for name, minimum in (("spot_size", 1), ("buffer", 0), ("background_width", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.shape, ApertureShape):
            raise TypeError("Use the ApertureShape enum")
        if not isinstance(self.global_background, bool):
            raise TypeError("global_background must be a boolean")
        if self.psf_box_size is not None and (
            isinstance(self.psf_box_size, bool)
            or not isinstance(self.psf_box_size, int)
            or self.psf_box_size < 3
            or self.psf_box_size % 2 != 1
        ):
            raise ValueError("PSF box size must be an odd integer of at least 3")


@dataclass(frozen=True)
class PhotometryResult:
    """Detached localisation-measurement columns, source state token and observation IDs.

    Intensity units are original image units summed over valid aperture pixels.
    NaN denotes unavailable numeric values; cycle_index -1 denotes no cycle.
    """

    columns: dict[str, np.ndarray]
    settings: PhotometrySettings
    mode: TraceMode
    run_id: str
    expected: FreshnessToken
    observation_ids: np.ndarray | None = None


@dataclass(frozen=True)
class PhotonSequenceResult:
    """Detached trace output whose primary records are photon sequences.

    Attributes:
        sequences: Per-acquisition/channel/molecule sequences with their sample metrics.
        settings: Aperture and background settings used for extraction.
        mode: Stationary-seed or moving-track extraction mode.
        run_id: Stable measurement-run identifier.
        expected: Source-state token used to reject stale application.
        sequence_ids: Applied sequence IDs, or None for an unapplied preview.
    """

    sequences: tuple[PhotonSequence, ...]
    settings: PhotometrySettings | FixedApertureSettings
    mode: TraceMode
    run_id: str
    expected: FreshnessToken
    sequence_ids: tuple[str, ...] | None = None

    @property
    def columns(self) -> dict[str, np.ndarray]:
        """Project sequence samples into detached columns for plotting and CSV export.

        Returns:
            Transient sample columns; no observation records are created.
        """
        from ..photon_sequences import to_columns

        return to_columns(self.sequences)


@dataclass(frozen=True)
class TrackingSettings:
    """Trackpy distances in selected coordinate pixels and explicit frame-clock memory.

    frame_clock="camera" counts every camera exposure, including other excitation
    channels. "channel" counts frames in the selected ChannelView; absent
    localizations consume memory, while other excitation frames do not.
    """

    search_range: float = 5.0
    memory: int = 0
    min_length: int = 1
    space: CoordinateSpace = CoordinateSpace.CORRECTED
    frame_clock: str = "camera"

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if not np.isfinite(self.search_range) or self.search_range <= 0:
            raise ValueError("search_range must be finite and positive")
        for name, minimum in (("memory", 0), ("min_length", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.frame_clock, str) or self.frame_clock not in (
            "camera",
            "channel",
        ):
            raise ValueError("frame_clock must be camera or channel")
        if not isinstance(self.space, CoordinateSpace):
            raise TypeError("Use CoordinateSpace enum")


@dataclass(frozen=True)
class TrackingResult:
    """Stable localization memberships independent of Trackpy particle labels.

    selected_ids is the exact replacement scope; None means the whole partition.
    An empty array selects nothing and never replaces unrelated tracks.
    """

    acquisition_id: str
    channel: str
    memberships: tuple[np.ndarray, ...]
    settings: TrackingSettings
    expected: FreshnessToken
    backend_version: str
    segmentation_id: str | None = None
    track_ids: np.ndarray | None = None
    selected_ids: np.ndarray | None = None


@dataclass(frozen=True)
class DiffusionFit:
    """Unweighted 2D linear fit; diffusion units are spatial_unit squared / second."""

    coefficient: float
    intercept: float
    slope: float
    r_squared: float
    residual_sum_squares: float
    lag_count: int


@dataclass(frozen=True)
class TrajectoryStatistics:
    """Per-step and MSD columns with explicit spatial units and seconds timing.

    Duration sums within-acquisition observed durations; no elapsed recording gap
    or displacement across acquisition boundaries is invented.
    """

    track_id: int
    space: CoordinateSpace
    unit: SpatialUnit
    steps: dict[str, np.ndarray]
    msd: dict[str, np.ndarray]
    duration: float
    path_length: float
    diffusion: DiffusionFit | None
