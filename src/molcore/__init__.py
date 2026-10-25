"""Shared, GUI-independent data foundation for single-molecule microscopy."""

import logging

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
from .types.analysis import (
    AnalysisResult,
    ClusteringResult,
    ClusteringSettings,
    ColocalisationResult,
    ColocalisationSettings,
    FilterResult,
    FilterSettings,
    RenderResult,
    RenderSettings,
)
from .types.contrast import ContrastMethod, ContrastResult, ContrastScope, ContrastSettings
from .types.images import ArraySource, CachedSource, FitsSource, HDF5Source, MissingSource, TiffSource
from .types.photons import PhotonSequence, PhotonSequenceStore, PhotonUnit
from .types.scientific import (
    ApertureShape,
    BackgroundMethod,
    DiffusionFit,
    PhotometryResult,
    PhotometrySettings,
    PhotonSequenceResult,
    SpatialUnit,
    TraceMode,
    TrackingResult,
    TrackingSettings,
    TrajectoryStatistics,
)

__version__ = "0.1.0"
logging.getLogger(__name__).addHandler(logging.NullHandler())
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

__all__ += [
    "ApertureShape",
    "BackgroundMethod",
    "DiffusionFit",
    "PhotometryResult",
    "PhotometrySettings",
    "SpatialUnit",
    "TraceMode",
    "TrackingResult",
    "TrackingSettings",
    "TrajectoryStatistics",
]


__all__ += ["ContrastMethod", "ContrastResult", "ContrastScope", "ContrastSettings"]
__all__ += ["PhotonSequence", "PhotonSequenceStore", "PhotonUnit"]
__all__ += ["PhotonSequenceResult"]
__all__ += [
    "AnalysisResult",
    "ClusteringResult",
    "ClusteringSettings",
    "ColocalisationResult",
    "ColocalisationSettings",
    "FilterResult",
    "FilterSettings",
    "RenderResult",
    "RenderSettings",
]
