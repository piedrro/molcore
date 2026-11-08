"""Full snapshots retain explicit missing sources while embedding available pixels."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from molcore import (
    Acquisition,
    ArraySource,
    ChannelSpec,
    Dataset,
    ImageUnavailableError,
    PhotonSequence,
    Segmentation,
)
from molcore.types.images import CachedSource, MissingSource
from molcore.types.storage import SaveOptions


def missing_model(source=None):
    model = Dataset(id="stable-dataset", name="source-free", metadata={"title": "analysis"})
    model.add_acquisition(
        Acquisition(
            MissingSource((5, 7, 9), "uint16") if source is None else source,
            id="missing",
            mode="alex",
            phase=1,
            excitation_sequence=("D", "A"),
            channel_specs={
                "D": ChannelSpec((0, 7, 0, 5), excitation="D"),
                "A": ChannelSpec((0, 7, 5, 9), excitation="A"),
            },
            timestamps=np.arange(5) * 0.2,
            pixel_size=110,
            metadata={"origin": "old"},
        )
    )
    model.add_segmentation(
        Segmentation("cells", "missing", np.ones((7, 5), dtype=np.uint16), channel="D", frame=1)
    )
    ids = model.localizations.add(
        id=[17, 31], acquisition_id="missing", channel="D", frame=[1, 3], x=2, y=3, photons=[100, 200]
    )
    track = model.tracks.add(ids, metadata={"import": "retained"})
    model.observations.add(
        id=[9],
        molecule_id="m",
        acquisition_id="missing",
        channel="D",
        frame=[1],
        intensity=[100],
        localization_id=[17],
        track_id=track,
    )
    model.photon_sequences.add(
        PhotonSequence(
            id="photons",
            acquisition_id="missing",
            channel="D",
            frames=[1, 3],
            photons=[100, 200],
            track_id=track,
        )
    )
    model.update_processing({"tracking": {"algorithm": "retained"}})
    return model


class MissingStorageTests(unittest.TestCase):
    def assert_missing_round_trip(self, model):
        self.assertEqual(model.id, "stable-dataset")
        self.assertEqual(model.name, "source-free")
        self.assertEqual(model.metadata, {"title": "analysis"})
        acquisition = model.acquisitions["missing"]
        self.assertIsInstance(acquisition.source, MissingSource)
        self.assertEqual(acquisition.source.shape, (5, 7, 9))
        self.assertEqual(acquisition.source.dtype, np.dtype("uint16"))
        self.assertEqual(acquisition.mode, "alex")
        self.assertEqual(acquisition.phase, 1)
        self.assertEqual(acquisition.excitation_sequence, ("D", "A"))
        self.assertEqual(acquisition.pixel_size, 110)
        self.assertEqual(acquisition.metadata, {"origin": "old"})
        self.assertEqual(acquisition.channels["D"].shape, (2, 7, 5))
        self.assertEqual(acquisition.channels["A"].shape, (3, 7, 4))
        self.assertEqual(acquisition.channels["A"].roi, (0, 7, 5, 9))
        np.testing.assert_array_equal(acquisition.channels["D"].source_frames, [1, 3])
        np.testing.assert_array_equal(acquisition.timestamps, np.arange(5) * 0.2)
        np.testing.assert_array_equal(model.localizations.query(space="raw")["id"], [17, 31])
        np.testing.assert_array_equal(model.localizations.query(space="raw")["photons"], [100, 200])
        np.testing.assert_array_equal(model.tracks.query()["track_id"], [1, 1])
        self.assertEqual(model.tracks.metadata[1]["parameters"], {"import": "retained"})
        np.testing.assert_array_equal(model.observations.query()["id"], [9])
        np.testing.assert_array_equal(model.photon_sequences["photons"].photons, [100, 200])
        np.testing.assert_array_equal(model.segmentations["cells"].labels, np.ones((7, 5)))
        self.assertEqual(model.processing.settings["tracking"], {"algorithm": "retained"})
        with self.assertRaises(ImageUnavailableError):
            acquisition.source[0]

    def test_mixed_full_save_embeds_available_pixels_and_preserves_missing_results(self):
        pixels = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
        for layout in ("hdf5", "directory"):
            for arguments in ({"image_policy": "full"}, {"options": SaveOptions(originals="full")}):
                with (
                    self.subTest(layout=layout, arguments=arguments),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    model = missing_model()
                    source = CachedSource(ArraySource(pixels))
                    model.add_acquisition(Acquisition(source, id="available", timestamps=[0, 1, 2]))
                    path = Path(directory) / ("mixed.h5" if layout == "hdf5" else "mixed")
                    model.save(path, layout=layout, **arguments)
                    with Dataset.open(path) as restored:
                        self.assert_missing_round_trip(restored)
                        np.testing.assert_array_equal(restored.acquisitions["available"].source[:], pixels)
                        self.assertEqual(restored.acquisitions["available"].source.dtype, np.dtype("float32"))
                    target = path if layout == "hdf5" else path / "dataset.h5"
                    with h5py.File(target) as snapshot:
                        self.assertNotIn("images", snapshot["acquisitions/0"])
                        np.testing.assert_array_equal(snapshot["acquisitions/1/images"][:], pixels)

    def test_all_missing_sources_can_be_saved_full(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.h5"
            model = missing_model()
            model.add_acquisition(Acquisition(MissingSource((2, 3, 4), "int32"), id="second"))
            model.save(path, options=SaveOptions(originals="full"))
            with Dataset.open(path) as restored:
                self.assert_missing_round_trip(restored)
                self.assertEqual(restored.acquisitions["second"].source.shape, (2, 3, 4))
                self.assertEqual(restored.acquisitions["second"].source.dtype, np.dtype("int32"))
                with self.assertRaises(ImageUnavailableError):
                    restored.acquisitions["second"].source[0]

    def test_nested_cached_missing_source_can_be_saved_full(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached.h5"
            model = missing_model(CachedSource(CachedSource(MissingSource((5, 7, 9), "uint16"))))
            model.save(path, image_policy="full")
            with Dataset.open(path) as restored:
                self.assert_missing_round_trip(restored)

    def test_imagefree_project_can_be_resaved_full_at_the_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resaved.h5"
            missing_model().save(path, image_policy="none")
            with Dataset.open(path) as loaded:
                loaded.save(path, image_policy="full")
            with Dataset.open(path) as restored:
                self.assert_missing_round_trip(restored)

    def test_real_pixel_read_failures_preserve_previous_file_and_cleanup(self):
        for error in (OSError("bad pixel read"), ImageUnavailableError("unavailable external source")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "previous.h5"
                original = missing_model()
                original.save(path)
                before = path.read_bytes()

                class FailingSource(ArraySource):
                    def _frame(self, index, failure=error):
                        if index == 1:
                            raise failure
                        return super()._frame(index)

                original.add_acquisition(Acquisition(FailingSource(np.ones((3, 4, 6))), id="bad"))
                with self.assertRaisesRegex(type(error), str(error)):
                    original.save(path, image_policy="full")
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(list(Path(directory).glob(".molcore-*.tmp")), [])
                with Dataset.open(path) as restored:
                    self.assert_missing_round_trip(restored)
                    self.assertNotIn("bad", restored.acquisitions)

    def test_missing_crop_policy_remains_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ImageUnavailableError):
                missing_model().save(Path(directory) / "crops.h5", image_policy="localization_crops")
