"""Worker-local image readers with explicit lifetime and bulk spatial access."""

from __future__ import annotations

from typing import Any

import numpy as np


class ImageReader:
    _handle: Any
    _data: Any
    _parent: ImageReader | None
    "Open one source locally; never pickle or share this reader between workers."

    def __init__(self, source: Any) -> None:
        self.source = source
        self._handle = self._data = self._parent = None
        self._closed = False
        from .types.images import FitsSource, HDF5Source, TiffSource

        try:
            if hasattr(source, "acquisition") and hasattr(source, "source_frames"):
                self._parent = open_reader(source.acquisition.source)
            elif isinstance(source, HDF5Source):
                import h5py

                self._handle = h5py.File(source.path, "r")
                self._data = self._handle[source.dataset]
            elif isinstance(source, TiffSource):
                import tifffile

                self._handle = tifffile.TiffFile(source.path)
            elif isinstance(source, FitsSource):
                from astropy.io import fits

                self._handle = fits.open(source.path, memmap=True)
                self._data = self._handle[source.hdu].section
        except FileNotFoundError as exc:
            from .types import ImageUnavailableError

            self.close()
            raise ImageUnavailableError(f"Missing image source: {getattr(source, 'path', '')}") from exc
        except BaseException:
            self.close()
            raise

    def read_frames(
        self, frames: Any, *, roi: tuple[int, int, int, int] | None = None, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Read original source indices in order, retaining duplicates and empty input.

        Args:
            frames: Frames.
            roi: Roi.
            out: Out.


        Returns:
            The computed value.
        """
        if self._closed:
            raise RuntimeError("Image reader is closed")
        from .types.images import ArraySource, FitsSource, HDF5Source, TiffSource, selection

        values = np.asarray(frames)
        if values.ndim == 1 and values.size == 0:
            values = np.array([], dtype=np.int64)
        indices, _, _ = selection(values, self.source.shape[0])
        height, width = self.source.shape[1:]
        bounds = (0, height, 0, width) if roi is None else tuple(roi)
        if len(bounds) != 4 or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in bounds
        ):
            raise ValueError("ROI must contain four integer bounds")
        y0, y1, x0, x1 = bounds
        if not (0 <= y0 <= y1 <= height and 0 <= x0 <= x1 <= width):
            raise ValueError("ROI extends outside the source")
        shape = (len(indices), y1 - y0, x1 - x0)
        if out is not None and (
            not isinstance(out, np.ndarray)
            or out.shape != shape
            or out.dtype != self.source.dtype
            or (not out.flags.writeable)
        ):
            raise ValueError("Output must be a writable array with matching shape and dtype")
        output = np.empty(shape, dtype=self.source.dtype) if out is None else out
        if not output.size:
            return output
        if self._parent is not None:
            sy0, _, sx0, _ = self.source.roi
            return self._parent.read_frames(
                self.source.source_frames[indices], roi=(sy0 + y0, sy0 + y1, sx0 + x0, sx0 + x1), out=output
            )
        if type(self.source) is ArraySource:
            output[:] = self.source.array[indices, y0:y1, x0:x1]
        elif isinstance(self.source, HDF5Source):
            if self._data.ndim == 2:
                output[:] = self._data[y0:y1, x0:x1]
            else:
                start = 0
                while start < len(indices):
                    stop = start + 1
                    while stop < len(indices) and indices[stop] == indices[stop - 1] + 1:
                        stop += 1
                    if output.flags.c_contiguous:
                        self._data.read_direct(
                            output,
                            source_sel=np.s_[indices[start] : indices[stop - 1] + 1, y0:y1, x0:x1],
                            dest_sel=np.s_[start:stop],
                        )
                    else:
                        output[start:stop] = self._data[indices[start] : indices[stop - 1] + 1, y0:y1, x0:x1]
                    start = stop
        elif isinstance(self.source, FitsSource):
            for index, frame in enumerate(indices):
                output[index] = (
                    self._data[int(frame), y0:y1, x0:x1]
                    if len(self._handle[self.source.hdu].shape) == 3
                    else self._data[y0:y1, x0:x1]
                )
        elif isinstance(self.source, TiffSource):
            if len(self._handle.pages) == self.source.shape[0]:
                for index, frame in enumerate(indices):
                    output[index] = self._handle.pages[int(frame)].asarray()[y0:y1, x0:x1]
            else:
                for index, frame in enumerate(indices):
                    output[index] = self.source[int(frame)][y0:y1, x0:x1]
        else:
            for index, frame in enumerate(indices):
                output[index] = self.source[int(frame)][y0:y1, x0:x1]
        return output

    def close(self) -> None:
        """Close reader-owned handles; never close the source descriptor."""
        if self._parent is not None:
            self._parent.close()
        if self._handle is not None:
            self._handle.close()
        self._closed = True

    def __enter__(self) -> Any:
        if self._closed:
            raise RuntimeError("Image reader is closed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def open_reader(source: Any) -> ImageReader:
    """Open a built-in or custom frame-addressable source using the fallback protocol.

    Args:
        source: Source.


    Returns:
        The computed value.
    """
    return ImageReader(source)
