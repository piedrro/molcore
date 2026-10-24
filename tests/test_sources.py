import multiprocessing
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import tifffile

from molcore import ArraySource, CachedSource, FitsSource, HDF5Source, TiffSource


def read_worker(source):
    return int(source[2, 1, 1])


class CountingSource(ArraySource):
    def __init__(self, frames):
        super().__init__(frames)
        self.reads = []

    def _frame(self, index):
        self.reads.append(index)
        return super()._frame(index)


class SourceTests(unittest.TestCase):
    def test_cache_is_bounded_and_drops_pixels_when_pickled(self):
        source = CountingSource(np.zeros((5, 3, 3), dtype=np.uint16))
        cached = CachedSource(source, max_bytes=18)
        cached[0]
        cached[0]
        self.assertEqual(source.reads, [0])
        cached[2]
        self.assertLessEqual(cached.cached_bytes, 18)
        cached[0]
        self.assertEqual(source.reads, [0, 2, 0])
        self.assertEqual(pickle.loads(pickle.dumps(cached)).cached_bytes, 0)

    def test_tiff_hdf5_lazy_readers_and_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            movie = np.arange(4 * 5 * 5, dtype=np.uint16).reshape(4, 5, 5)
            tiff = Path(tmp) / "movie.tif"
            hdf5 = Path(tmp) / "movie.h5"
            tifffile.imwrite(tiff, movie, compression="deflate", photometric="minisblack")
            with h5py.File(hdf5, "w") as file:
                file.create_dataset("frames", data=movie, chunks=(1, 5, 5))
            sources = [TiffSource(tiff), HDF5Source(hdf5, "frames")]
            for source in sources:
                np.testing.assert_array_equal(source[::-2, 1:3, 2:4], movie[::-2, 1:3, 2:4])
                with multiprocessing.get_context("spawn").Pool(1) as pool:
                    self.assertEqual(pool.apply(read_worker, (source,)), 56)

    def test_dask_reads_only_requested_frames(self):
        try:
            __import__("dask.array")
        except ImportError:
            self.skipTest("Dask extra not installed")
        source = CountingSource(np.arange(5 * 3 * 3).reshape(5, 3, 3))
        lazy = source.as_dask()
        source.reads.clear()
        self.assertEqual(int(lazy[3, 0, 0].compute(scheduler="synchronous")), 27)
        self.assertEqual(source.reads, [3])

    def test_fits_sections_preserve_values(self):
        try:
            from astropy.io import fits
        except ImportError:
            self.skipTest("FITS extra not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "movie.fits"
            movie = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
            fits.writeto(path, movie)
            source = FitsSource(path)
            np.testing.assert_array_equal(source[2], movie[2])

    def test_import_does_not_load_gui_or_optional_analysis_engines(self):
        script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'napari','qtpy','picasso','trackpy','cellpose','torch','scipy','dask','astropy'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from molcore import Experiment
import molcore.cellpose
Experiment()
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
