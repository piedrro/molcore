"""Settings and detached results for non-destructive exploratory analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._shared import CoordinateSpace


@dataclass(frozen=True)
class FilterSettings:
    """Inclusive finite numeric bounds; track reductions ignore non-finite members."""

    metric: str = "x"
    minimum: float = 0.0
    maximum: float = 1e12
    target: str = "localizations"
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if (
            not self.metric
            or not np.isfinite([self.minimum, self.maximum]).all()
            or self.minimum > self.maximum
        ):
            raise ValueError("Provide a metric and finite increasing bounds")
        if self.target not in ("localizations", "tracks"):
            raise ValueError("Target must be localizations or tracks")
        if self.aggregation not in ("mean", "median", "min", "max", "std"):
            raise ValueError("Unknown track aggregation")


@dataclass(frozen=True)
class ClusteringSettings:
    """DBSCAN Euclidean radius in selected-space pixels; samples are localizations."""

    radius: float = 0.2
    min_samples: int = 5

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Radius must be finite and positive")
        if (
            isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, int)
            or self.min_samples < 1
        ):
            raise ValueError("Minimum samples must be a positive integer")


@dataclass(frozen=True)
class ColocalisationSettings:
    """Inclusive Euclidean matching radius in common reference pixels."""

    radius: float = 5.0

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("Radius must be finite and nonnegative")


@dataclass(frozen=True)
class RenderSettings:
    """Output samples per coordinate pixel and Picasso blur width in those pixels."""

    oversampling: float = 5.0
    blur_method: str = "none"
    min_blur_width: float = 0.0

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if not np.isfinite(self.oversampling) or self.oversampling <= 0:
            raise ValueError("Oversampling must be finite and positive")
        if self.blur_method not in ("none", "smooth", "convolve", "gaussian", "gaussian_iso"):
            raise ValueError("Unknown Picasso blur method")
        if not np.isfinite(self.min_blur_width) or self.min_blur_width < 0:
            raise ValueError("Minimum blur width must be finite and nonnegative")


@dataclass(frozen=True)
class AnalysisResult:
    """Detached row columns bound to a source partition and model revision."""

    acquisition_id: str
    channel: str
    space: CoordinateSpace
    settings: object
    columns: dict[str, np.ndarray]
    expected: tuple


@dataclass(frozen=True)
class FilterResult(AnalysisResult):
    """Selected entity IDs and a distribution of finite candidate measurements."""

    selected_ids: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ClusteringResult(AnalysisResult):
    """Localization assignments including noise (-1), plus cluster centroids."""

    centroids: dict[str, np.ndarray]


@dataclass(frozen=True)
class ColocalisationResult(AnalysisResult):
    """Mutual pairs, distances and midpoints across two explicit channel sets."""

    target_channel: str
    unmatched_source: int
    unmatched_target: int


@dataclass(frozen=True)
class RenderResult(AnalysisResult):
    """Density image; origin is (x, y), spacing is in selected coordinate pixels."""

    image: np.ndarray
    origin: tuple[float, float]
    pixel_spacing: float
    count: int
