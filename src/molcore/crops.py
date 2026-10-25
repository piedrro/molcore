"""Random-access localization crops and measured-frame track sequences."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from .types._shared import Crop

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import ArrayLike

    from .types import Experiment


@logged
def validate_width(width: int) -> int:
    """Validate width.

    Args:
        width: Odd crop width in camera pixels.

    Returns:
        Computed int value.
    """
    if not isinstance(width, (int, np.integer)) or width < 1 or width % 2 != 1:
        raise ValueError("Crop width must be a positive odd integer")


class CropStore:
    """Resolve current raw-channel crops or lazily loaded saved crop records.

    Args:
        experiment: Owner providing current positions and image sources.
    """

    def __init__(self, experiment: Experiment) -> None:
        """Initialise the object and validate its configuration.

        Args:
            experiment: Scientific model supplying acquisitions and result tables.
        """
        self.experiment = experiment
        self._saved = {}
        self._photon_saved: dict[tuple[str, int, int], tuple[str, int]] = {}

    @logged
    def invalidate(self, ids: ArrayLike | None) -> None:
        """Discard saved crop descriptors affected by localization edits.

        Args:
            ids: Localization identifiers whose positions or existence changed.
        """
        ids = set(int(i) for i in ids)
        self._saved = {k: v for k, v in self._saved.items() if k[0] not in ids}

    def _extract(self, row: Any, width: int, camera_frame: int, localization_id: int | None) -> Crop:
        """Extract.

        Args:
            row: Row used by this operation.
            width: Odd crop width in camera pixels.
            camera_frame: Original camera-frame index, rather than logical channel index.
            localization_id: Localization id used by this operation.

        Returns:
            Result retaining the input format or backend-specific representation.
        """
        view = self.experiment.acquisitions[row["acquisition_id"]].channels[row["channel"]]
        image = view[view.channel_index(camera_frame)]
        half = width // 2
        x0, y0 = int(np.floor(row["x"] + 0.5)) - half, int(np.floor(row["y"] + 0.5)) - half
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(image.shape[1], x0 + width), min(image.shape[0], y0 + width)
        pixels = np.zeros((width, width), dtype=image.dtype)
        valid = np.zeros((width, width), dtype=bool)
        if x2 > x1 and y2 > y1:
            pixels[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0] = image[y1:y2, x1:x2]
            valid[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0] = True
        return Crop(
            pixels, valid, row["acquisition_id"], row["channel"], int(camera_frame), (x0, y0), localization_id
        )

    @logged
    def localization(self, localization_id: int, *, width: int = 7) -> Crop:
        """Read a raw crop centred on one effective localization position.

        Args:
            localization_id: Existing localization identifier.
            width: Positive odd crop width in channel pixels.

        Returns:
            Crop with a validity mask and original-frame provenance.

        Raises:
            ImageUnavailableError: Neither raw pixels nor a matching saved crop exists.
        """
        validate_width(width)
        rows = self.experiment.localizations.query(ids=[localization_id])
        if len(rows["id"]) != 1:
            raise ValueError("Unknown localization ID")
        key = (int(localization_id), width)
        if key in self._saved:
            from .storage import read_crop

            return read_crop(*self._saved[key])
        row = {k: v[0] for k, v in rows.items()}
        return self._extract(row, width, row["frame"], int(localization_id))

    @logged
    def track(self, track_id: int, *, width: int = 7) -> Iterator[Crop]:
        """Yield crops at measured track frames without gap interpolation.

        Args:
            track_id: Existing track identifier.
            width: Positive odd crop width.

        Yields:
            Crop values ordered by the track's acquisition/sequence timeline.
        """
        validate_width(width)
        meta = self.experiment.tracks.metadata[track_id]
        rows = self.experiment.tracks.query(track_id=track_id, sequence_id=meta["sequence_id"])
        for localization_id in rows["id"]:
            yield self.localization(localization_id, width=width)

    @logged
    def photon_sequence(
        self, sequence_id: str, *, width: int = 7, include_stale: bool = False
    ) -> Iterator[Crop | None]:
        """Yield one raw-channel crop per sequence sample without filling frame gaps.

        Args:
            sequence_id: Independent photon-sequence identifier.
            width: Positive odd crop width in channel pixels.
            include_stale: Explicitly permit historical crops of invalidated measurements.

        Yields:
            Crops retaining sequence/sample IDs, or None for unknown sample positions.

        Raises:
            ValueError: The sequence has no spatial crop centres.
            StaleResultError: Sequence inputs changed and include_stale is false.
            ImageUnavailableError: Neither source pixels nor matching saved crops exist.
        """
        from .storage import read_crop
        from .types._shared import StaleResultError

        validate_width(width)
        sequence = self.experiment.photon_sequences[sequence_id]
        if self.experiment.photon_sequences.is_stale(sequence_id) and not include_stale:
            raise StaleResultError(
                "Photon sequence is stale; recompute or explicitly request historical crops"
            )
        if sequence.positions is None:
            raise ValueError("Photon-sequence crops require explicit raw-channel positions")
        for index, frame in enumerate(sequence.frames):
            if not np.isfinite(sequence.positions[index]).all():
                yield None
                continue
            key = (sequence_id, index, width)
            if key in self._photon_saved:
                path, row = self._photon_saved[key]
                yield read_crop(path, row, group_name="photon_crops")
                continue
            x, y = sequence.positions[index]
            crop = self._extract(
                dict(acquisition_id=sequence.acquisition_id, channel=sequence.channel, x=x, y=y),
                width,
                int(frame),
                int(sequence.localization_ids[index]) or None,
            )
            yield replace(crop, photon_sequence_id=sequence_id, sample_index=index)

    @logged
    def stationary(self, localization_id: int, frames: ArrayLike | None, *, width: int = 7) -> Iterator[Crop]:
        """Sample explicitly requested camera frames at one fixed molecule position.

        Args:
            localization_id: Localization defining the stationary position/channel.
            frames: Original camera frames to sample; no interpolation is performed.
            width: Positive odd crop width.

        Yields:
            Crops with each requested original camera frame.
        """
        validate_width(width)
        rows = self.experiment.localizations.query(ids=[localization_id])
        if len(rows["id"]) != 1:
            raise ValueError("Unknown localization ID")
        row = {k: v[0] for k, v in rows.items()}
        for frame in frames:
            yield self._extract(row, width, frame, int(localization_id))
