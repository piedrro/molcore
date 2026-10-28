"""Detached cross-round registration settings and results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetAlignmentSettings:
    """Fit corrected fiducial coordinates in reference-pixel units."""

    model: str = "affine"
    match_radius: float | None = None
    ransac_threshold: float = 1.0
    max_trials: int = 1000

    def __post_init__(self) -> None:
        if self.model not in ("translation", "affine", "homography"):
            raise ValueError("Choose translation, affine or homography")
        if self.match_radius is not None and (not np.isfinite(self.match_radius) or self.match_radius <= 0):
            raise ValueError("Matching radius must be finite and positive")
        if not np.isfinite(self.ransac_threshold) or self.ransac_threshold <= 0:
            raise ValueError("RANSAC threshold must be finite and positive")
        if isinstance(self.max_trials, bool) or not isinstance(self.max_trials, int) or self.max_trials < 1:
            raise ValueError("RANSAC trials must be a positive integer")


@dataclass(frozen=True)
class DatasetAlignmentResult:
    """Source-corrected to reference-corrected registration and provenance."""

    source_dataset_id: str
    reference_dataset_id: str
    source_acquisition_id: str
    source_channel: str
    reference_acquisition_id: str
    reference_channel: str
    matrix: np.ndarray
    residuals: np.ndarray
    inliers: np.ndarray
    source_ids: np.ndarray
    reference_ids: np.ndarray
    fingerprints: tuple[str, str]
    settings: DatasetAlignmentSettings

    @property
    def rms_residual(self) -> float:
        """RMS error of accepted correspondences, in reference pixels.

        Returns:
            RMS error of accepted fiducial pairs.
        """
        return float(np.sqrt(np.mean(self.residuals[self.inliers] ** 2)))
