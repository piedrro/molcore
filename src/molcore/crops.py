"""Random-access localization crops and measured-frame track sequences."""

from __future__ import annotations

import numpy as np

from .types._shared import Crop


def validate_width(width):
    if not isinstance(width, (int, np.integer)) or width < 1 or width % 2 != 1:
        raise ValueError("Crop width must be a positive odd integer")


class CropStore:
    """Resolve current raw-channel crops or lazily loaded saved crop records.

    Args:
        experiment: Owner providing current positions and image sources.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        self._saved = {}

    def invalidate(self, ids):
        """Discard saved crop descriptors affected by localization edits.

        Args:
            ids: Localization identifiers whose positions or existence changed.
        """
        ids = set(int(i) for i in ids)
        self._saved = {k: v for k, v in self._saved.items() if k[0] not in ids}

    def _extract(self, row, width, camera_frame, localization_id):
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

    def localization(self, localization_id, *, width=7):
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

    def track(self, track_id, *, width=7):
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

    def stationary(self, localization_id, frames, *, width=7):
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
