"""Explicit exports bake alignment while preserving source movie pixels."""

import threading
import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

from molcore import ArraySource, Calibration, Dataset
from molcore.files import image_acquisition


class DatasetExportTests(unittest.TestCase):
    def test_cancelled_export_keeps_previous_output_files(self):
        from molcore.exports import export_dataset_images

        dataset = Dataset()
        dataset.add_acquisition(
            image_acquisition(ArraySource(np.ones((3, 4, 4))), id="movie")
        )
        with TemporaryDirectory() as directory:
            paths = export_dataset_images(dataset, directory)
            expected = {
                path: path.read_bytes()
                for path in (*paths, Path(directory) / "manifest.json")
            }
            cancel = threading.Event()

            def progress(update):
                if update.completed == 1:
                    cancel.set()

            with self.assertRaises(CancelledError):
                export_dataset_images(
                    dataset, directory, progress=progress, cancel_event=cancel
                )
            self.assertEqual({path: path.read_bytes() for path in expected}, expected)

    def test_export_bakes_both_alex_acceptor_channels_only_in_output(self):
        from molcore.exports import export_dataset_images

        pixels = np.zeros((4, 12, 24), dtype=np.uint16)
        pixels[:, 5, 4] = 100
        pixels[:, 5, 12 + 6] = 100
        original = pixels.copy()
        dataset = Dataset(id="round-1")
        dataset.add_acquisition(
            image_acquisition(ArraySource(pixels), id="movie", mode="alex")
        )
        dataset.transforms.add_calibration(
            Calibration("optics", np.array([[1, 0, -2], [0, 1, 0], [0, 0, 1]]))
        )
        dataset.transforms.select("movie", "DA", "optics")
        with TemporaryDirectory() as directory:
            paths = export_dataset_images(dataset, directory)
            self.assertEqual(len(paths), 4)
            for path in paths:
                stack = tifffile.imread(path)
                self.assertEqual(stack.shape, (2, 12, 12))
                np.testing.assert_array_equal(np.argmax(stack[:, 5], axis=-1), [4, 4])
            self.assertTrue((Path(directory) / "manifest.json").is_file())
        np.testing.assert_array_equal(pixels, original)
