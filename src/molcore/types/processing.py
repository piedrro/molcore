"""Validated processing settings, detached results and CPU progress records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from .._logging import logged
from ._shared import CoordinateSpace


@dataclass(frozen=True)
class CellposeSettings:
    """Cellpose 4 inference parameters for static two-dimensional images.

    Attributes:
        pretrained_model: Built-in model name or path to Cellpose 4 weights.
        diameter: Optional object diameter in pixels; None keeps native scaling.
        normalize: Apply Cellpose's image normalization.
        flow_threshold: Maximum flow error accepted during mask construction.
        cellprob_threshold: Cell-probability threshold used to construct masks.
        min_size: Minimum object area in pixels; zero disables size filtering.
        batch_size: Network image-tile batch size, separate from process count.
        gpu: Explicitly request CUDA or MPS execution in one worker.
        use_bfloat16: Use reduced-precision model weights instead of float32.
    """

    pretrained_model: str | Path = "cpsam_v2"
    diameter: float | None = None
    normalize: bool = True
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    batch_size: int = 8
    gpu: bool = False
    use_bfloat16: bool = False

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if not isinstance(self.pretrained_model, (str, Path)) or not str(self.pretrained_model).strip():
            raise ValueError("Specify a Cellpose 4 model name or weight path")
        object.__setattr__(self, "pretrained_model", str(self.pretrained_model))
        if self.diameter is not None and (not np.isfinite(self.diameter) or self.diameter <= 0):
            raise ValueError("Diameter must be finite and positive")
        if not np.isfinite([self.flow_threshold, self.cellprob_threshold]).all() or self.flow_threshold < 0:
            raise ValueError("Invalid Cellpose thresholds")
        for name, minimum in (("min_size", 0), ("batch_size", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("normalize", "gpu", "use_bfloat16"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class SegmentationRequest:
    """Binding between one segmentation source frame and its model result.

    Attributes:
        acquisition_id: Source acquisition identifier.
        channel: Logical channel whose pixels should be segmented.
        frame: Original camera frame; None selects the first channel frame.
        segmentation_id: Result identifier; generated from acquisition/channel/frame.
        space: Coordinate space in which to sample the source image and store labels.
    """

    acquisition_id: str
    channel: str
    frame: int | None = None
    segmentation_id: str | None = None
    space: CoordinateSpace = CoordinateSpace.RAW

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if (
            not isinstance(self.acquisition_id, str)
            or not self.acquisition_id
            or not isinstance(self.channel, str)
            or not self.channel
        ):
            raise ValueError("Request requires acquisition and channel names")
        if self.frame is not None:
            if (
                isinstance(self.frame, (bool, np.bool_))
                or not isinstance(self.frame, (int, np.integer))
                or self.frame < 0
            ):
                raise ValueError("Frame must be a nonnegative camera-frame integer")
            object.__setattr__(self, "frame", int(self.frame))
        if self.segmentation_id is not None and (
            not isinstance(self.segmentation_id, str) or not self.segmentation_id
        ):
            raise ValueError("Segmentation ID must be a nonempty string")
        object.__setattr__(self, "space", CoordinateSpace(self.space))


class FitMethod(StrEnum):
    """Gaussian fitting objectives supported by the Picasso CPU backend."""

    LQ = "lq"
    MLE = "mle"


class DriftMethod(StrEnum):
    """Picasso drift estimators."""

    RCC = "rcc"
    AIM = "aim"


@dataclass(frozen=True)
class ProgressUpdate:
    """Parent-process progress notification.

    Attributes:
        completed: Completed work units, including fractional job progress.
        total: Total work units.
        description: Human-readable current processing stage.
    """

    completed: float
    total: float
    description: str = ""

    @property
    @logged
    def percent(self) -> float:
        """Completion percentage, monotonically increasing within one CPU call.

        Returns:
            Computed float value.
        """
        return 100.0 if self.total == 0 else 100 * self.completed / self.total


@dataclass(frozen=True)
class CameraSettings:
    """Explicit scalar camera conversion used when extracting fit spots.

    Attributes:
        baseline: Camera offset in ADU.
        sensitivity: Electrons per ADU.
        gain: Camera multiplication gain.
        em: Whether to use EMCCD localization-precision calculations.
    """

    baseline: float
    sensitivity: float
    gain: float
    em: bool = False

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if (
            not np.isfinite([self.baseline, self.sensitivity, self.gain]).all()
            or min(self.sensitivity, self.gain) <= 0
        ):
            raise ValueError("Camera settings must be finite with positive sensitivity and gain")


@dataclass(frozen=True)
class LocalisationSettings:
    """Net-gradient detection and Gaussian CPU fitting parameters.

    Attributes:
        min_net_gradient: Detection threshold in source-camera units.
        box_size: Odd fit/detection box width in channel pixels.
        fit_method: Least squares or Poisson maximum likelihood.
        chunk_size: Maximum channel frames submitted in one CPU job.
        tolerance: Backend optimizer tolerance.
        max_iterations: Maximum iterations per fitted spot.
    """

    min_net_gradient: float = 1000.0
    box_size: int = 7
    fit_method: FitMethod = FitMethod.LQ
    chunk_size: int = 16
    tolerance: float = 1e-3
    max_iterations: int = 1000

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        object.__setattr__(self, "fit_method", FitMethod(self.fit_method))
        for name in ("box_size", "chunk_size", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.box_size < 3 or self.box_size % 2 != 1:
            raise ValueError("Box size must be odd and at least three")
        if (
            not np.isfinite(self.min_net_gradient)
            or self.min_net_gradient < 0
            or not np.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            raise ValueError("Invalid threshold or optimizer tolerance")


@dataclass(frozen=True)
class DriftSettings:
    """RCC/AIM parameters expressed in reference-channel pixels and frames.

    Attributes:
        method: RCC or AIM estimator.
        segment_frames: Reference-channel frames per temporal segment.
        intersect_distance: AIM intersection distance in reference pixels.
        search_radius: AIM search radius in reference pixels.
    """

    method: DriftMethod = DriftMethod.RCC
    segment_frames: int = 1000
    intersect_distance: float = 0.15
    search_radius: float = 0.5

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        object.__setattr__(self, "method", DriftMethod(self.method))
        if (
            isinstance(self.segment_frames, bool)
            or not isinstance(self.segment_frames, int)
            or self.segment_frames < 1
        ):
            raise ValueError("Segment frames must be a positive integer")
        if (
            not np.isfinite([self.intersect_distance, self.search_radius]).all()
            or min(self.intersect_distance, self.search_radius) <= 0
        ):
            raise ValueError("AIM distances must be finite and positive")


@dataclass(frozen=True)
class DetectionResult:
    """Detached candidate positions with original camera-frame indices.

    Attributes:
        acquisition_id: Acquisition that supplied the camera pixels.
        channel: Logical source channel.
        columns: frame, x, y and net_gradient arrays.
        settings: Detection parameters.
        source_token: Identity of the image source used for detection.
        frame_range: Preview camera-frame selection, or None for the full channel.
    """

    acquisition_id: str
    channel: str
    columns: dict[str, np.ndarray]
    settings: LocalisationSettings
    source_token: int
    frame_range: tuple[int, int] | None = None


@dataclass(frozen=True)
class LocalisationResult:
    """Fitted measurement columns and optional active localization identifiers.

    Attributes:
        acquisition_id: Acquisition identifier.
        channel: Logical source channel.
        columns: Detached fitted columns in raw channel pixels.
        settings: Detection and fitting parameters.
        localization_ids: IDs allocated by a successful active-partition update.
        failed_fits: Candidates rejected for non-convergence or invalid fit values.
    """

    acquisition_id: str
    channel: str
    columns: dict[str, np.ndarray]
    settings: LocalisationSettings
    localization_ids: np.ndarray | None = None
    failed_fits: int = 0


@dataclass(frozen=True)
class DriftResult:
    """Estimated reference-space displacement at explicit camera frames.

    Attributes:
        acquisition_id: Acquisition identifier.
        channel: Reference logical channel used for estimation.
        frames: Actual camera frames within measured coverage.
        displacement: Corresponding (dx, dy) displacement array.
        settings: Estimator parameters.
        applied: Whether the result was committed to TransformManager.
    """

    acquisition_id: str
    channel: str
    frames: np.ndarray
    displacement: np.ndarray
    settings: DriftSettings
    applied: bool = False


@dataclass(frozen=True)
class AlignmentResult:
    """Source-to-target channel fit and composed reference calibration.

    Attributes:
        acquisition_id: Acquisition identifier.
        source_channel: Channel being aligned.
        target_channel: Channel defining correspondence coordinates.
        matrix: Source-to-target 3x3 homogeneous matrix.
        reference_matrix: Source-to-dataset-reference homogeneous matrix.
        residuals: Euclidean fit residual per matched pair, in target pixels.
        source_ids: IDs of matched source localizations.
        target_ids: IDs of matched target localizations.
        calibration_name: Name selected on application, if any.
    """

    acquisition_id: str
    source_channel: str
    target_channel: str
    matrix: np.ndarray
    reference_matrix: np.ndarray
    residuals: np.ndarray
    source_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint64))
    target_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint64))
    calibration_name: str | None = None
    inliers: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
