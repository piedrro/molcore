"""Shared, GUI-independent data foundation for single-molecule microscopy."""

import logging

from .backends import BackendRegistry, ProcessorBackend
from .cpu import CpuExecutor
from .execution import ExecutionContext, Executor, SerialExecutor, ThreadExecutor
from .revisions import ChangeSet, DependencyToken, StateToken
from .snapshots import DatasetSnapshot, ExperimentSnapshot
from .types import (
    Acquisition,
    AcquisitionMode,
    AcquisitionSequence,
    AlignmentResult,
    BactFitCell,
    Calibration,
    CameraSettings,
    CellposeSettings,
    ChannelName,
    ChannelSpec,
    ColumnSchema,
    CoordinateSpace,
    Crop,
    Dataset,
    DatasetAlignmentResult,
    DatasetAlignmentSettings,
    DetectionResult,
    DriftMethod,
    DriftResult,
    DriftSettings,
    Experiment,
    FitMethod,
    FrameInfo,
    ImagePolicy,
    ImageSource,
    ImageUnavailableError,
    LocalisationResult,
    LocalisationSettings,
    LocalizationBatch,
    ProcessingMetadata,
    ProgressUpdate,
    SaveOptions,
    Segmentation,
    SegmentationOutline,
    SegmentationRequest,
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
from .types.batches import ColumnBatch, ObservationBatch, TrackMembershipBatch
from .types.boxes import BoundingBox, BoundingBoxStore
from .types.contrast import ContrastMethod, ContrastResult, ContrastScope, ContrastSettings
from .types.images import ArraySource, CachedSource, FitsSource, HDF5Source, MissingSource, TiffSource
from .types.photons import PhotonSequence, PhotonSequenceStore, PhotonUnit
from .types.results import ApplyReceipt, ResultIdentity, WriteScope
from .types.scientific import (
    ApertureShape,
    BackgroundMethod,
    DiffusionFit,
    PhotometryResult,
    PhotometrySettings,
    PhotonSequenceResult,
    SequenceTrackingResult,
    SpatialUnit,
    TraceMode,
    TrackingResult,
    TrackingSettings,
    TrajectoryStatistics,
)
from .types.selections import PartitionSelection, SequenceSelection

__version__ = "0.2.0"
logging.getLogger(__name__).addHandler(logging.NullHandler())
__all__ = [
    "CpuExecutor",
    "SequenceTrackingResult",
    "DatasetSnapshot",
    "ExperimentSnapshot",
    "ResultIdentity",
    "WriteScope",
    "ApplyReceipt",
    "PartitionSelection",
    "SequenceSelection",
    "ColumnBatch",
    "ObservationBatch",
    "TrackMembershipBatch",
    "AlignmentResult",
    "CameraSettings",
    "CellposeSettings",
    "ColumnSchema",
    "DetectionResult",
    "DriftMethod",
    "DriftResult",
    "DriftSettings",
    "FitMethod",
    "LocalisationResult",
    "LocalisationSettings",
    "LocalizationBatch",
    "ProgressUpdate",
    "SaveOptions",
    "SegmentationRequest",
    "AcquisitionMode",
    "ChannelName",
    "Dataset",
    "Experiment",
    "DatasetAlignmentResult",
    "DatasetAlignmentSettings",
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
    "SegmentationOutline",
    "BactFitCell",
    "StaleResultError",
]
__all__ += [
    "BackendRegistry",
    "ProcessorBackend",
    "ExecutionContext",
    "Executor",
    "SerialExecutor",
    "ThreadExecutor",
    "ChangeSet",
    "DependencyToken",
    "StateToken",
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
__all__ += ["BoundingBox", "BoundingBoxStore"]
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
