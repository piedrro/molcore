import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import tifffile

from molcore import (
    Acquisition,
    AcquisitionMode,
    ArraySource,
    Calibration,
    Dataset,
    ImageUnavailableError,
    Segmentation,
    TiffSource,
)


def fixture(source=None):
    exp = Dataset(metadata={"title": "molecules α"})
    exp.add_acquisition(
        Acquisition(
            source or ArraySource(np.arange(3 * 5 * 5).reshape(3, 5, 5)),
            id="a",
            mode=AcquisitionMode.SINGLE,
            timestamps=[0.0, 0.1, 0.2],
        )
    )
    ids = exp.localizations.add(
        acquisition_id="a", channel="image", frame=[0, 2], x=[1.0, 2.0], y=[1.0, 2.0], photons=[100.0, 200.0]
    )
    exp.tracks.add(ids)
    exp.observations.add(
        molecule_id="molecule",
        acquisition_id="a",
        channel="image",
        frame=0,
        intensity=100,
        localization_id=int(ids[0]),
    )
    exp.add_segmentation(
        Segmentation(
            "cells",
            "a",
            np.ones((5, 5), dtype=np.uint16),
            source_image=np.full((5, 5), 9),
            source_path="original.tif",
        )
    )
    exp.transforms.add_calibration(Calibration("identity"))
    exp.transforms.select("a", "image", "identity")
    exp.processing.settings["localization"] = {"box": 7}
    return exp, ids


class StorageTests(unittest.TestCase):
    def test_same_path_crop_save_rebinds_in_memory_descriptors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crops.h5"
            exp = Dataset()
            movie = np.zeros((3, 5, 5), dtype=np.uint16)
            movie[:, :, :] = np.array([10, 20, 30])[:, None, None]
            exp.add_acquisition(Acquisition(ArraySource(movie), id="a"))
            ids = exp.localizations.add(acquisition_id="a", channel="image", frame=[0, 1, 2], x=2, y=2)
            exp.save(path, image_policy="localization_crops", crop_width=3)
            with Dataset.open(path) as restored:
                restored.localizations.delete([ids[0]])
                restored.save(path, image_policy="localization_crops", crop_width=3)
                for index in (1, 2):
                    crop = restored.crops.localization(ids[index], width=3)
                    self.assertEqual(crop.localization_id, int(ids[index]))
                    self.assertEqual(int(crop.pixels[1, 1]), (index + 1) * 10)

    def test_all_policies_and_layouts(self):
        for layout in ("hdf5", "directory"):
            for policy in ("none", "localization_crops", "track_crops", "both", "full"):
                with self.subTest(layout=layout, policy=policy), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / ("saved.h5" if layout == "hdf5" else "saved")
                    exp, ids = fixture()
                    exp.localizations.move(ids[0], x=2, y=1)
                    exp.save(path, layout=layout, image_policy=policy, crop_width=3)
                    with Dataset.open(path) as restored:
                        self.assertEqual(restored.metadata["title"], "molecules α")
                        self.assertEqual(restored.acquisitions["a"].mode, AcquisitionMode.SINGLE)
                        self.assertEqual(restored.localizations.query()["x"].tolist(), [2, 2])
                        self.assertEqual(restored.localizations.query(original=True)["x"].tolist(), [1, 2])
                        self.assertEqual(restored.tracks.query()["frame"].tolist(), [0, 2])
                        self.assertEqual(restored.segmentations["cells"].source_image[0, 0], 9)
                        self.assertEqual(restored.processing.settings["localization"]["box"], 7)
                        if policy == "full":
                            self.assertEqual(int(restored.acquisitions["a"].channels["image"][2, 2, 2]), 62)
                        else:
                            with self.assertRaises(ImageUnavailableError):
                                restored.acquisitions["a"].channels["image"][0]
                        if policy in ("localization_crops", "track_crops", "both", "full"):
                            crop = restored.crops.localization(ids[0], width=3)
                            self.assertEqual(int(crop.pixels[1, 1]), 7)
                        if policy != "full":
                            with self.assertRaises(ImageUnavailableError):
                                restored.crops.localization(ids[0], width=5)

    def test_directory_relocation_and_missing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "original"
            root.mkdir()
            tifffile.imwrite(
                root / "movie.tif", np.ones((3, 5, 5), dtype=np.uint16), photometric="minisblack"
            )
            exp, _ = fixture(TiffSource(root / "movie.tif"))
            exp.save(root / "analysis", layout="directory")
            relocated = Path(tmp) / "relocated"
            shutil.copytree(root, relocated)
            with Dataset.open(relocated / "analysis") as restored:
                self.assertEqual(int(restored.acquisitions["a"].source[1, 0, 0]), 1)
            (relocated / "movie.tif").unlink()
            with Dataset.open(relocated / "analysis") as restored:
                self.assertEqual(len(restored.localizations), 2)
                with self.assertRaises(ImageUnavailableError):
                    restored.acquisitions["a"].source[0]

    def test_failed_save_preserves_previous_file_and_rejects_newer_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "saved.h5"
            exp, _ = fixture()
            exp.save(path)
            original = path.read_bytes()
            with patch("os.replace", side_effect=OSError("simulated replacement failure")):
                with self.assertRaises(OSError):
                    exp.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
            with h5py.File(path, "a") as file:
                file.attrs["format_version"] = 999
            with self.assertRaises(ValueError):
                Dataset.open(path)
