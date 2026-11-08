"""Editable segmentation outlines and detached bacterial cell fits."""

from math import isfinite
from operator import index

import numpy as np
from numpy.typing import ArrayLike

from ..immutable import immutable_record
from ._shared import CoordinateSpace

Points = tuple[tuple[float, float], ...]


def _points(values: ArrayLike, name: str, minimum: int) -> Points:
    coordinates = np.asarray(values, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < minimum
        or (not np.all(np.isfinite(coordinates)))
    ):
        raise ValueError(f"{name} requires at least {minimum} finite (x, y) points")
    return tuple(((float(x), float(y)) for x, y in coordinates))


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _frame(value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Frame must be a nonnegative integer")
    try:
        frame = index(value)
    except TypeError as exc:
        raise ValueError("Frame must be a nonnegative integer") from exc
    if frame < 0:
        raise ValueError("Frame must be a nonnegative integer")
    return frame


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@immutable_record
class SegmentationOutline:
    """One editable polygon in subpixel (x, y) order.

    Acquisition and channel are optional together. A bound frame is a source
    camera frame, while an unbound frame is local to the outline's image.
    None means the polygon applies across frames. Pixel size is in nm per pixel
    and can supply calibration for standalone outlines.
    """

    id: str
    vertices: Points
    acquisition_id: str | None = None
    channel: str | None = None
    frame: int | None = None
    space: CoordinateSpace = CoordinateSpace.RAW
    pixel_size: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.id, "Outline ID")
        if (self.acquisition_id is None) != (self.channel is None):
            raise ValueError("Supply acquisition and channel together")
        if self.acquisition_id is not None:
            _identifier(self.acquisition_id, "Acquisition ID")
        if self.channel is not None:
            _identifier(self.channel, "Channel")
        object.__setattr__(self, "vertices", _points(self.vertices, "Polygon", 3))
        object.__setattr__(self, "space", CoordinateSpace(self.space))
        if self.frame is not None:
            object.__setattr__(self, "frame", _frame(self.frame))
        if self.pixel_size is not None:
            object.__setattr__(self, "pixel_size", _positive(self.pixel_size, "Pixel size"))


@immutable_record
class BactFitCell:
    """Fitted cell geometry in subpixel (x, y) order, with radius in pixels.

    Polynomial coefficients use descending powers in BactFit's fitting basis.
    Linked fits use the outline's frame, coordinate space and calibration.
    Standalone fits can specify a local frame and pixel size in nm per pixel.
    """

    id: str
    polygon: Points
    midline: Points
    poles: Points
    polynomial_coefficients: tuple[float, ...]
    radius: float
    fit_error: float | None = None
    outline_id: str | None = None
    frame: int | None = None
    pixel_size: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.id, "Fit ID")
        if self.outline_id is not None:
            _identifier(self.outline_id, "Outline ID")
            if self.frame is not None or self.pixel_size is not None:
                raise ValueError("Linked fits inherit frame and calibration from their outline")
        object.__setattr__(self, "polygon", _points(self.polygon, "Polygon", 3))
        object.__setattr__(self, "midline", _points(self.midline, "Midline", 2))
        poles = _points(self.poles, "Poles", 2)
        if len(poles) != 2:
            raise ValueError("A fitted cell has exactly two poles")
        object.__setattr__(self, "poles", poles)
        coefficients = tuple(float(value) for value in self.polynomial_coefficients)
        if not coefficients or not all(map(isfinite, coefficients)):
            raise ValueError("Polynomial coefficients must be nonempty and finite")
        object.__setattr__(self, "polynomial_coefficients", coefficients)
        object.__setattr__(self, "radius", _positive(self.radius, "Radius"))
        if self.fit_error is not None:
            error = float(self.fit_error)
            if not isfinite(error) or error < 0:
                raise ValueError("Fit error must be finite and nonnegative")
            object.__setattr__(self, "fit_error", error)
        if self.frame is not None:
            object.__setattr__(self, "frame", _frame(self.frame))
        if self.pixel_size is not None:
            object.__setattr__(self, "pixel_size", _positive(self.pixel_size, "Pixel size"))
