"""Typed settings and results for display-only contrast estimation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from numbers import Real

import numpy as np

from .._logging import logged


class ContrastMethod(StrEnum):
    """Display limit estimator, applied to finite selected pixels."""

    PERCENTILE = "percentile"
    MIN_MAX = "min_max"


class ContrastScope(StrEnum):
    """Independent channel limits or common limits from pooled samples."""

    PER_CHANNEL = "per_channel"
    SHARED = "shared"


@dataclass(frozen=True)
class ContrastSettings:
    """Deterministic frame/pixel sampling and display limit settings.

    Percentile limits default to 1 and 99. At most 16 evenly spaced frames and
    65,536 evenly spaced finite, unmasked pixels per frame are sampled. None
    disables the corresponding cap. MIN_MAX processes every valid pixel in each
    selected frame, ignoring max_pixels_per_frame so outliers are not missed.
    """

    method: ContrastMethod = ContrastMethod.PERCENTILE
    percentiles: tuple[float, float] = (1.0, 99.0)
    max_frames: int | None = 16
    max_pixels_per_frame: int | None = 65_536

    def __post_init__(self) -> None:
        """Validate the immutable record after construction."""
        if not isinstance(self.method, ContrastMethod):
            raise TypeError("Use ContrastMethod enum")
        if (
            not isinstance(self.percentiles, tuple)
            or len(self.percentiles) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) for v in self.percentiles)
        ):
            raise ValueError("percentiles must be a pair of real numbers")
        low, high = self.percentiles
        if not np.isfinite([low, high]).all() or not 0 <= low < high <= 100:
            raise ValueError("Require 0 <= lower percentile < upper percentile <= 100")
        for name in ("max_frames", "max_pixels_per_frame"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")


@dataclass(frozen=True)
class ContrastResult:
    """Finite display limits and the sampling provenance for one input/channel.

    frames contains array frame indices or original camera frames for model calls.
    Counts describe this channel's input, including when limits are shared. No
    scientific measurements, source pixels or application display state are changed.
    """

    lower: float
    upper: float
    settings: ContrastSettings
    frames: tuple[int, ...]
    valid_pixel_count: int
    sample_count: int
    scope: ContrastScope = ContrastScope.PER_CHANNEL

    @property
    @logged
    def limits(self) -> tuple[float, float]:
        """Return limits directly suitable for a viewer's contrast_limits property.

        Returns:
            Computed tuple[float, float] value.
        """
        return self.lower, self.upper
