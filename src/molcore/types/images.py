"""Reopenable image sources and byte-bounded process-local caching."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._logging import logged
from ._shared import ImageUnavailableError

if TYPE_CHECKING:
    import dask.array as da
    from numpy.typing import ArrayLike

    from ..types import ImageSource


def _shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Shape.

    Args:
        shape: Array dimensions in time, y, x order.

    Returns:
        Computed tuple[int, int, int] value.
    """
    shape = tuple(int(n) for n in shape)
    if len(shape) == 2:
        shape = (1, *shape)
    if len(shape) != 3 or min(shape) < 0 or min(shape[1:]) == 0:
        raise ValueError("Expected a 2D image or (time, y, x) stack")
    return shape


@logged
def selection(
    key: int | slice | ArrayLike, length: int
) -> tuple[np.ndarray, bool, tuple]:
    """Selection.

    Args:
        key: NumPy-style frame and optional spatial selection.
        length: Number of valid items in the indexed source.

    Returns:
        Computed NumPy array value.
    """
    parts = key if isinstance(key, tuple) else (key,)
    parts = parts or (slice(None),)
    ellipses = [i for i, part in enumerate(parts) if part is Ellipsis]
    if len(ellipses) > 1 or len(parts) > 3 or any(part is None for part in parts):
        raise IndexError(
            "Use at most three indices and one ellipsis; new axes are unsupported"
        )
    if ellipses:
        index = ellipses[0]
        parts = parts[:index] + (slice(None),) * (4 - len(parts)) + parts[index + 1 :]
    first = parts[0]
    scalar = isinstance(first, (int, np.integer))
    if scalar:
        index = int(first)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError("Frame index out of range")
        indices = np.array([index], dtype=np.int64)
    elif isinstance(first, slice):
        indices = np.arange(*first.indices(length), dtype=np.int64)
    else:
        indices = np.asarray(first)
        if indices.dtype.kind == "b":
            if indices.shape != (length,):
                raise IndexError("Boolean index length mismatch")
            indices = np.flatnonzero(indices)
        elif indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise IndexError("Expected integer, slice or 1D frame indices")
        indices = indices.astype(np.int64, copy=True)
        indices[indices < 0] += length
        if np.any((indices < 0) | (indices >= length)):
            raise IndexError("Frame index out of range")
    return indices, scalar, parts[1:]


class Source:
    """Common lazy indexing for stacks with frame-addressable pixels."""

    def __getitem__(self, key: int | slice | ArrayLike) -> np.ndarray:
        """Getitem.

        Args:
            key: NumPy-style frame and optional spatial selection.

        Returns:
            Computed NumPy array value.
        """
        indices, scalar, spatial = selection(key, self.shape[0])
        if scalar:
            return self._frame(int(indices[0]))[spatial or (...,)]
        frames = [self._frame(int(i)) for i in indices]
        data = (
            np.stack(frames)
            if frames
            else np.empty((0, *self.shape[1:]), dtype=self.dtype)
        )
        return data[(slice(None), *spatial)]

    @logged
    def close(self) -> None:
        """Sources own no persistent handles; each read closes its file."""

    @logged
    def materialize(self) -> ArraySource:
        """Load this stack into RAM.

        Returns:
            ArraySource containing the complete stack.
        """
        return ArraySource(self[:])

    @logged
    def as_dask(self, chunks: tuple[int, ...] | str | None = None) -> da.Array:
        """Expose deferred frame reads through Dask.

        Args:
            chunks: Dask chunk sizes; defaults to one full image per chunk.

        Returns:
            Dask array that does not read source pixels until computed.

        Raises:
            ImportError: The Dask runtime dependency is absent.
        """
        try:
            import dask.array as da
        except ImportError as exc:
            raise ImportError(
                "Dask is missing; reinstall MolCore with its runtime dependencies"
            ) from exc
        return da.from_array(
            self,
            chunks=chunks or (1, *self.shape[1:]),
            asarray=False,
            fancy=False,
            meta=np.empty((0, 0, 0), dtype=self.dtype),
        )

    @property
    @logged
    def ndim(self) -> int:
        """Number of array dimensions, always three.

        Returns:
            Computed int value.
        """
        return 3

    def __len__(self) -> int:
        """Return the number of available items.

        Returns:
            Computed int value.
        """
        return self.shape[0]


class ArraySource(Source):
    """Memory-backed image stack wrapping a NumPy array without copying it.

    Args:
        array: Real-valued 2D image or (time, y, x) NumPy-compatible stack.
    """

    def __init__(self, array: ArrayLike) -> None:
        """Initialise the object and validate its configuration.

        Args:
            array: Input pixel array; image stacks use time, y, x axis order.
        """
        array = np.asarray(array)
        self.shape = _shape(array.shape)
        if array.dtype.kind not in "biuf":
            raise ValueError("Image pixels must be real numbers")
        self.array = array.reshape(self.shape)
        self.dtype = array.dtype

    def _frame(self, index: int) -> np.ndarray:
        """Read one original image frame from the bound source.

        Args:
            index: Zero-based item or logical-channel frame index.

        Returns:
            Computed NumPy array value.
        """
        return self.array[index]


class TiffSource(Source):
    """Lazy TIFF reader that opens and closes the file for each requested frame.

    Args:
        path: TIFF path. Compressed stacks must have frame-addressable pages.

    Raises:
        ValueError: The first series is not a supported 2D time stack.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialise the object and validate its configuration.

        Args:
            path: Input or output filesystem path.
        """
        import tifffile

        self.path = str(Path(path).resolve())
        with tifffile.TiffFile(self.path) as file:
            series = file.series[0]
            self.shape = _shape(series.shape)
            self.dtype = np.dtype(series.dtype)
            self._paged = len(series.pages) == self.shape[0]
        if not self._paged:
            try:
                mapped = tifffile.memmap(self.path)
                del mapped
            except ValueError as exc:
                raise ValueError(
                    "TIFF must be frame-addressable or memory-mappable"
                ) from exc

    def _frame(self, index: int) -> np.ndarray:
        """Read one original image frame from the bound source.

        Args:
            index: Zero-based item or logical-channel frame index.

        Returns:
            Computed NumPy array value.
        """
        import tifffile

        if not Path(self.path).is_file():
            raise ImageUnavailableError(f"Missing TIFF source: {self.path}")
        if self._paged:
            with tifffile.TiffFile(self.path) as file:
                return file.series[0].pages[index].asarray()
        mapped = tifffile.memmap(self.path).reshape(self.shape)
        return np.array(mapped[index], copy=True)


class FitsSource(Source):
    """Lazy FITS section reader supporting scaled camera pixels.

    Args:
        path: FITS file path.
        hdu: Image HDU index or name.

    Raises:
        ImportError: The Astropy runtime dependency is absent.
    """

    def __init__(self, path: str | Path, hdu: int | str = 0) -> None:
        """Initialise the object and validate its configuration.

        Args:
            path: Input or output filesystem path.
            hdu: Hdu used by this operation.
        """
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise ImportError(
                "Astropy is missing; reinstall MolCore with its runtime dependencies"
            ) from exc
        self.path, self.hdu = str(Path(path).resolve()), hdu
        with fits.open(self.path, memmap=False) as file:
            header = file[hdu].header
            self.shape = _shape(
                tuple(header[f"NAXIS{i}"] for i in range(header["NAXIS"], 0, -1))
            )
            self.dtype = np.dtype(
                file[hdu].section[(slice(0, 1),) * header["NAXIS"]].dtype
            )

    def _frame(self, index: int) -> np.ndarray:
        """Read one original image frame from the bound source.

        Args:
            index: Zero-based item or logical-channel frame index.

        Returns:
            Computed NumPy array value.
        """
        from astropy.io import fits

        if not Path(self.path).is_file():
            raise ImageUnavailableError(f"Missing FITS source: {self.path}")
        with fits.open(self.path, memmap=False) as file:
            hdu = file[self.hdu]
            return hdu.section[index] if hdu.header["NAXIS"] == 3 else hdu.section[:]


class HDF5Source(Source):
    """Reopenable HDF5 image descriptor, safe to pass to spawned workers.

    Args:
        path: HDF5 file path.
        dataset: Absolute dataset name containing a 2D image or 3D time stack.
    """

    def __init__(self, path: str | Path, dataset: str | None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            path: Input or output filesystem path.
            dataset: Dataset used by this operation.
        """
        import h5py

        self.path, self.dataset = str(Path(path).resolve()), dataset
        with h5py.File(self.path, "r") as file:
            self.shape = _shape(file[dataset].shape)
            self.dtype = file[dataset].dtype

    def _frame(self, index: int) -> np.ndarray:
        """Read one original image frame from the bound source.

        Args:
            index: Zero-based item or logical-channel frame index.

        Returns:
            Computed NumPy array value.
        """
        import h5py

        if not Path(self.path).is_file():
            raise ImageUnavailableError(f"Missing HDF5 source: {self.path}")
        with h5py.File(self.path, "r") as file:
            data = file[self.dataset]
            return data[index] if data.ndim == 3 else data[:]


class MissingSource(Source):
    """Image geometry retained when pixels are unavailable.

    Args:
        shape: Original image extent.
        dtype: Original pixel dtype.
        reason: Explanation included when a pixel read is attempted.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | str,
        reason: str = "Raw images were not saved",
    ) -> None:
        """Initialise the object and validate its configuration.

        Args:
            shape: Array dimensions in time, y, x order.
            dtype: NumPy-compatible element dtype.
            reason: Reason used by this operation.
        """
        self.shape, self.dtype, self.reason = _shape(shape), np.dtype(dtype), reason

    def _frame(self, index: int) -> np.ndarray:
        """Read one original image frame from the bound source.

        Args:
            index: Zero-based item or logical-channel frame index.

        Returns:
            Computed NumPy array value.
        """
        raise ImageUnavailableError(self.reason)


class CachedSource(Source):
    """Byte-bounded LRU cache discarded when transferred between processes.

    Args:
        source: Underlying image source.
        max_bytes: Maximum cached pixel bytes; oversized frames bypass the cache.
    """

    def __init__(self, source: ImageSource, max_bytes: int = 64 * 1024 * 1024) -> None:
        """Initialise the object and validate its configuration.

        Args:
            source: Image source providing lazy frame access.
            max_bytes: Max bytes used by this operation.
        """
        if max_bytes < 0:
            raise ValueError("Cache budget must be nonnegative")
        self.source, self.max_bytes = source, int(max_bytes)
        self.shape, self.dtype = source.shape, source.dtype
        self._cache, self._pid = OrderedDict(), os.getpid()

    @property
    @logged
    def cached_bytes(self) -> int:
        """Number of pixel bytes currently held by this process's cache.

        Returns:
            Computed int value.
        """
        return sum(frame.nbytes for frame in self._cache.values())

    def _frame(self, index: int) -> np.ndarray:
        """Read one original image frame from the bound source.

        Args:
            index: Zero-based item or logical-channel frame index.

        Returns:
            Computed NumPy array value.
        """
        if self._pid != os.getpid():
            self._cache.clear()
            self._pid = os.getpid()
        if index in self._cache:
            frame = self._cache.pop(index)
        else:
            frame = np.array(self.source[index], copy=True)
            if frame.nbytes > self.max_bytes:
                return frame
            while self._cache and self.cached_bytes + frame.nbytes > self.max_bytes:
                self._cache.popitem(last=False)
            frame.flags.writeable = False
        self._cache[index] = frame
        return frame

    def __getstate__(self) -> dict[str, Any]:
        """Getstate.

        Returns:
            Computed dict[str, Any] value.
        """
        return {**self.__dict__, "_cache": OrderedDict()}

    @logged
    def close(self) -> None:
        """Close."""
        self._cache.clear()
        self.source.close()
