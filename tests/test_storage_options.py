"""Composable image storage and registered external source round trips."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from molcore import Acquisition, ArraySource, Dataset, PhotonSequence, storage


class StorageOptionsTests(unittest.TestCase):
    def model(self):
        model = Dataset()
        model.add_acquisition(
            Acquisition(ArraySource(np.ones((2, 9, 9), dtype=np.uint16)), id="a")
        )
        ids = model.localizations.add(
            acquisition_id="a", channel="image", frame=[0, 1], x=[4, 4], y=[4, 4]
        )
        model.tracks.add(ids[:1])
        model.photon_sequences.add(
            PhotonSequence(
                id="p",
                acquisition_id="a",
                channel="image",
                frames=np.array([0]),
                photons=np.array([1.0]),
                positions=np.array([[4.0, 4.0]]),
            )
        )
        return model

    def test_all_combinations_and_full_images(self):
        self.assertTrue(
            hasattr(storage, "SaveOptions"), "Composable SaveOptions is missing"
        )
        with tempfile.TemporaryDirectory() as directory:
            for originals in ("none", "full"):
                for flags in itertools.product((False, True), repeat=3):
                    kinds = frozenset(
                        k
                        for k, selected in zip(
                            ("localization", "track", "photon_sequence"),
                            flags,
                            strict=True,
                        )
                        if selected
                    )
                    path = Path(directory) / "snapshot.h5"
                    storage.save_dataset(
                        self.model(),
                        path,
                        options=storage.SaveOptions(
                            originals=originals, crops=kinds, crop_width=3
                        ),
                    )
                    with h5py.File(path) as file:
                        self.assertEqual(
                            "images" in file["acquisitions/0"], originals == "full"
                        )
                        self.assertEqual("crops" in file, flags[0] or flags[1])
                        self.assertEqual("photon_crops" in file, flags[2])
                        if flags[0] or flags[1]:
                            self.assertEqual(
                                len(file["crops/ids"]), 2 if flags[0] else 1
                            )

    def test_registered_source_and_unknown_reader(self):
        self.assertTrue(
            hasattr(storage, "register_image_source"), "Source registration is missing"
        )

        class CustomSource(ArraySource):
            pass

        def writer(source):
            return {"path": source.path, "label": "custom"}

        def reader(descriptor):
            source = CustomSource(np.load(descriptor["path"]))
            source.path = descriptor["path"]
            return source

        storage.register_image_source(
            "test.custom", CustomSource, writer=writer, reader=reader
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pixels.npy"
            pixels = np.ones((2, 9, 9), dtype=np.uint16)
            np.save(path, pixels)
            source = CustomSource(pixels)
            source.path = str(path)
            model = Dataset()
            model.add_acquisition(Acquisition(source, id="a"))
            target = Path(directory) / "snapshot"
            storage.save_dataset(
                model,
                target,
                layout="directory",
                options=storage.SaveOptions(originals="external"),
            )
            with storage.open_dataset(target) as loaded:
                np.testing.assert_array_equal(
                    loaded.acquisitions["a"].source[0], pixels[0]
                )
            np.save(path, pixels.astype(np.float32))
            with self.assertRaisesRegex(ValueError, "geometry/dtype changed"):
                storage.open_dataset(target)
            np.save(path, pixels)
            with h5py.File(target / "dataset.h5", "r+") as file:
                group = file["acquisitions/0"]
                info = json.loads(group.attrs["info"])
                self.assertEqual(Path(info["external"]["path"]), Path("../pixels.npy"))
                info["external"]["kind"] = "unregistered.plugin"
                group.attrs["info"] = json.dumps(info)
            with self.assertRaisesRegex(ValueError, "unregistered.plugin"):
                storage.open_dataset(target)

    def test_invalid_options_and_conflicting_legacy_arguments(self):
        for kwargs in ({"originals": "bad"}, {"crops": {"bad"}}, {"crop_width": 2}):
            with self.assertRaises(ValueError):
                storage.SaveOptions(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.h5"
            with self.assertRaises(ValueError):
                storage.save_dataset(
                    self.model(),
                    path,
                    options=storage.SaveOptions(),
                    image_policy="full",
                )
            with self.assertRaises(ValueError):
                storage.save_dataset(
                    self.model(),
                    path,
                    options=storage.SaveOptions(originals="external"),
                )

    def test_explicit_none_omits_legacy_directory_references(self):
        import tifffile

        from molcore import TiffSource

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pixels.tif"
            tifffile.imwrite(source, np.ones((2, 9, 9), dtype=np.uint16))
            model = Dataset()
            model.add_acquisition(Acquisition(TiffSource(source), id="a"))
            target = Path(directory) / "snapshot"
            storage.save_dataset(
                model, target, layout="directory", options=storage.SaveOptions()
            )
            with h5py.File(target / "dataset.h5") as file:
                self.assertNotIn(
                    "external", json.loads(file["acquisitions/0"].attrs["info"])
                )

    def test_explicit_external_rejects_sources_without_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "external descriptor"):
                storage.save_dataset(
                    self.model(),
                    directory,
                    layout="directory",
                    options=storage.SaveOptions(originals="external"),
                )

    def test_experiment_full_images_and_crops(self):
        from molcore import Experiment

        experiment = Experiment()
        model = self.model()
        experiment.add_dataset(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.h5"
            storage.save_experiment(
                experiment,
                path,
                options=storage.SaveOptions(
                    originals="full", crops={"track", "photon_sequence"}, crop_width=3
                ),
            )
            loaded = storage.open_experiment(path)
            restored = loaded.datasets[model.id]
            np.testing.assert_array_equal(
                restored.acquisitions["a"].source[0], model.acquisitions["a"].source[0]
            )
            self.assertEqual(len(restored.crops._saved), 1)
            self.assertEqual(len(restored.crops._photon_saved), 1)
            restored.close()
