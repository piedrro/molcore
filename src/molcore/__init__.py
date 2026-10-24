"""Shared, GUI-independent data foundation for single-molecule microscopy."""

from .types import (
    Acquisition,
    AcquisitionMode,
    AcquisitionSequence,
    Calibration,
    ChannelName,
    ChannelSpec,
    CoordinateSpace,
    Crop,
    Experiment,
    FrameInfo,
    ImagePolicy,
    ImageSource,
    ImageUnavailableError,
    ProcessingMetadata,
    Segmentation,
    StaleResultError,
)
from .types.images import ArraySource, CachedSource, FitsSource, HDF5Source, MissingSource, TiffSource

__version__ = "0.1.0"
__all__ = [
    "AcquisitionMode",
    "ChannelName",
    "Experiment",
    "Acquisition",
    "AcquisitionSequence",
    "ArraySource",
    "CachedSource",
    "FitsSource",
    "HDF5Source",
    "MissingSource",
    "TiffSource",
    "Calibration",
    "ChannelSpec",
    "CoordinateSpace",
    "Crop",
    "FrameInfo",
    "ImagePolicy",
    "ImageSource",
    "ImageUnavailableError",
    "ProcessingMetadata",
    "Segmentation",
    "StaleResultError",
]
