"""Public data models, shared records, and image-access protocols."""

from ._shared import (
    AcquisitionMode,
    Calibration,
    ChannelName,
    ChannelSpec,
    CoordinateSpace,
    Crop,
    FrameInfo,
    ImagePolicy,
    ImageSource,
    ImageUnavailableError,
    ProcessingMetadata,
    Segmentation,
    StaleResultError,
)
from .acquisition import Acquisition, AcquisitionSequence
from .analysis import (
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
from .batches import ColumnSchema, LocalizationBatch
from .boxes import BoundingBox, BoundingBoxStore
from .cells import BactFitCell, SegmentationOutline
from .contrast import ContrastMethod, ContrastResult, ContrastScope, ContrastSettings
from .dataset import Dataset
from .dataset_alignment import DatasetAlignmentResult, DatasetAlignmentSettings
from .experiment import Experiment
from .images import ArraySource, CachedSource, FitsSource, HDF5Source, MissingSource, TiffSource
from .photons import PhotonSequence, PhotonSequenceStore, PhotonUnit
from .processing import (
    AlignmentResult,
    CameraSettings,
    CellposeSettings,
    DetectionResult,
    DriftMethod,
    DriftResult,
    DriftSettings,
    FitMethod,
    LocalisationResult,
    LocalisationSettings,
    ProgressUpdate,
    SegmentationRequest,
)
from .scientific import (
    ApertureShape,
    BackgroundMethod,
    CancelEvent,
    DiffusionFit,
    PhotometryResult,
    PhotometrySettings,
    PhotonSequenceResult,
    ProgressCallback,
    SpatialUnit,
    TraceMode,
    TrackingResult,
    TrackingSettings,
    TrajectoryStatistics,
)
from .storage import SaveOptions
from .tables import LocalizationTable, ObservationTable, TrackTable

__all__ = [
    "BoundingBox",
    "BoundingBoxStore",
    "ColumnSchema",
    "LocalizationBatch",
    "SaveOptions",
    "CellposeSettings",
    "SegmentationRequest",
    "AlignmentResult",
    "CameraSettings",
    "DetectionResult",
    "DriftMethod",
    "DriftResult",
    "DriftSettings",
    "FitMethod",
    "LocalisationResult",
    "LocalisationSettings",
    "ProgressUpdate",
    "ArraySource",
    "CachedSource",
    "FitsSource",
    "HDF5Source",
    "MissingSource",
    "TiffSource",
    "AcquisitionMode",
    "ChannelName",
    "Dataset",
    "Experiment",
    "DatasetAlignmentResult",
    "DatasetAlignmentSettings",
    "Acquisition",
    "AcquisitionSequence",
    "LocalizationTable",
    "ObservationTable",
    "TrackTable",
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
    "ApertureShape",
    "BackgroundMethod",
    "CancelEvent",
    "DiffusionFit",
    "PhotometryResult",
    "PhotometrySettings",
    "ProgressCallback",
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
