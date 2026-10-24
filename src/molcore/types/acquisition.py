"""Camera acquisitions and explicitly concatenated logical channel views."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import numpy as np

from ._shared import AcquisitionMode, ChannelSpec, FrameInfo, ImageSource
from .images import Source


class ChannelView(Source):
    """Lazy spatial and temporal view of one acquisition channel.

    Args:
        acquisition: Acquisition containing the original camera stream.
        name: Logical channel identifier.
        spec: Spatial ROI and excitation selection.
    """

    def __init__(self, acquisition, name, spec):
        self.acquisition, self.name, self.spec = acquisition, name, spec
        self.roi = spec.roi or (0, acquisition.source.shape[1], 0, acquisition.source.shape[2])
        y0, y1, x0, x1 = self.roi
        if y1 > acquisition.source.shape[1] or x1 > acquisition.source.shape[2]:
            raise ValueError("Channel ROI extends beyond camera frame")
        frames = np.arange(spec.frame_start, acquisition.source.shape[0], spec.frame_step)
        if spec.excitation is not None:
            if spec.excitation not in acquisition.excitation_sequence:
                raise ValueError("Channel excitation missing from excitation sequence")
            excitation = np.asarray(acquisition.excitation_sequence)
            frames = frames[excitation[(frames + acquisition.phase) % len(excitation)] == spec.excitation]
        self.source_frames = frames
        self.source_frames.flags.writeable = False
        self.shape = (len(frames), y1 - y0, x1 - x0)
        self.dtype = acquisition.source.dtype

    def _frame(self, index):
        y0, y1, x0, x1 = self.roi
        return self.acquisition.source[int(self.source_frames[index])][y0:y1, x0:x1]

    def frame_info(self, index: int) -> FrameInfo:
        """Resolve a logical channel index to source-frame provenance.

        Args:
            index: Channel index, supporting Python negative indexing.

        Returns:
            FrameInfo retaining the original camera frame and timestamp.

        Raises:
            IndexError: The index lies outside this channel.
        """
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Channel index out of range")
        camera = int(self.source_frames[index])
        a = self.acquisition
        return FrameInfo(
            a.id,
            camera,
            self.name,
            index,
            None if a.timestamps is None else float(a.timestamps[camera]),
            cycle_index=(camera + a.phase) // len(a.excitation_sequence) if a.excitation_sequence else None,
        )

    def channel_index(self, camera_frame: int) -> int:
        """Find the logical index measuring an original camera frame.

        Args:
            camera_frame: Acquisition-local camera frame.

        Returns:
            Integer channel index.

        Raises:
            ValueError: The camera frame does not measure this channel.
        """
        index = int(np.searchsorted(self.source_frames, camera_frame))
        if index >= len(self) or self.source_frames[index] != camera_frame:
            raise ValueError(f"Camera frame {camera_frame} does not measure channel {self.name}")
        return index


class Acquisition:
    """One camera stream with explicit logical channel mappings.

    Args:
        source: Reopenable image source exposing (time, y, x).
        id: Stable identifier; generated when omitted.
        channel_specs: Mapping from channel names to ChannelSpec values.
        excitation_sequence: Repeating excitation names, such as ("D", "A").
        phase: Offset into the excitation sequence for the first camera frame.
        timestamps: Optional seconds values, one per original camera frame.
        pixel_size: Optional pixel size in nanometres.
        metadata: Acquisition-specific microscope settings and provenance.
        mode: Acquisition intent; does not override explicit channel mappings.

    Raises:
        ValueError: Geometry, timestamps, or calibration units are invalid.
    """

    def __init__(
        self,
        source: ImageSource,
        *,
        id=None,
        channel_specs=None,
        excitation_sequence=(),
        phase=0,
        timestamps=None,
        pixel_size=None,
        metadata=None,
        mode="custom",
    ):
        self.id = str(id or uuid4())
        self.source = source
        self.mode = AcquisitionMode(mode)
        self.excitation_sequence = tuple(excitation_sequence)
        self.phase = int(phase)
        self.pixel_size = pixel_size
        if pixel_size is not None and (not np.isfinite(pixel_size) or pixel_size <= 0):
            raise ValueError("Pixel size must be positive")
        self.timestamps = None if timestamps is None else np.array(timestamps, dtype=float, copy=True)
        if self.timestamps is not None:
            if self.timestamps.shape != (source.shape[0],) or not np.isfinite(self.timestamps).all():
                raise ValueError("Expected one finite timestamp per camera frame")
            self.timestamps.flags.writeable = False
        self.metadata = dict(metadata or {})
        self.channel_specs = dict(channel_specs or {"image": ChannelSpec()})
        self.channels = {name: ChannelView(self, name, spec) for name, spec in self.channel_specs.items()}

    def close(self):
        """Release resources owned by the underlying image source."""
        self.source.close()


class SequenceChannel(Source):
    """A lazy concatenation retaining acquisition-local camera provenance."""

    def __init__(self, sequence, name):
        self.sequence, self.name = sequence, name
        self.views = [a.channels[name] for a in sequence.acquisitions]
        self.ends = np.cumsum([len(view) for view in self.views])
        self.shape = (int(self.ends[-1]), *self.views[0].shape[1:])
        self.dtype = np.result_type(*(v.dtype for v in self.views))

    def _locate(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Sequence frame out of range")
        part = int(np.searchsorted(self.ends, index, side="right"))
        local = index - (int(self.ends[part - 1]) if part else 0)
        return part, local

    def _frame(self, index):
        part, local = self._locate(index)
        return self.views[part][local].astype(self.dtype, copy=False)

    def frame_info(self, index):
        """Return source identity and the sequence camera-frame offset.

        Args:
            index: Index in the concatenated logical channel.

        Returns:
            FrameInfo with both original and sequence camera-frame indices.
        """
        part, local = self._locate(index)
        info = self.views[part].frame_info(local)
        return replace(
            info,
            channel_frame=index,
            sequence_frame=info.camera_frame + self.sequence.offsets[info.acquisition_id],
        )


class AcquisitionSequence:
    """Explicit ordering of recordings without invented time continuity.

    Args:
        acquisitions: Ordered acquisitions with compatible logical geometry.
        id: Stable sequence identifier.
        allow_cross_boundary_tracks: Whether tracks may cross acquisitions.
        reference: Shared coordinate reference required for crossing tracks.

    Raises:
        ValueError: Acquisitions are incompatible or the reference is missing.
    """

    def __init__(self, acquisitions, *, id=None, allow_cross_boundary_tracks=False, reference=None):
        self.acquisitions = list(acquisitions)
        if not self.acquisitions or len({a.id for a in self.acquisitions}) != len(self.acquisitions):
            raise ValueError("Sequence needs distinct acquisitions")
        first = self.acquisitions[0]
        for a in self.acquisitions[1:]:
            if set(a.channels) != set(first.channels) or a.pixel_size != first.pixel_size:
                raise ValueError("Sequence channels and pixel scales must agree")
            if any(a.channels[c].shape[1:] != first.channels[c].shape[1:] for c in first.channels):
                raise ValueError("Sequence channel geometry must agree")
        if allow_cross_boundary_tracks and not reference:
            raise ValueError("Cross-boundary tracking requires a shared reference identifier")
        self.id = str(id or uuid4())
        self.allow_cross_boundary_tracks = bool(allow_cross_boundary_tracks)
        self.reference = reference
        self.offsets = {}
        offset = 0
        for a in self.acquisitions:
            self.offsets[a.id] = offset
            offset += a.source.shape[0]
        self.channels = {name: SequenceChannel(self, name) for name in first.channels}
