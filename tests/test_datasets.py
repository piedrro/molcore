"""Independent probing-round state within one sequencing experiment."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import molcore


class DatasetTests(unittest.TestCase):
    def test_nested_photon_crops_remain_readable_after_same_path_rewrite(self):
        experiment = molcore.Experiment()
        dataset = molcore.Dataset(id="round-1")
        dataset.add_acquisition(
            molcore.Acquisition(
                molcore.ArraySource(np.arange(128).reshape(2, 8, 8)), id="movie"
            )
        )
        dataset.localizations.add(
            acquisition_id="movie", channel="image", frame=0, x=[2, 3], y=[3, 4]
        )
        dataset.photon_sequences.add(
            molcore.PhotonSequence(
                id="trace",
                acquisition_id="movie",
                channel="image",
                frames=np.array([0, 1]),
                photons=np.array([10.0, 20.0]),
                positions=np.array([[2.0, 3.0], [2.0, 3.0]]),
            )
        )
        experiment.add_dataset(dataset)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.h5"
            experiment.save(path, image_policy="all_crops", crop_width=3)
            with molcore.Experiment.open(path) as restored:
                model = restored.datasets["round-1"]
                self.assertEqual(
                    [
                        c.pixels[1, 1]
                        for c in model.crops.photon_sequence("trace", width=3)
                    ],
                    [26, 90],
                )
                model.localizations.delete([1])
                restored.save(path, image_policy="all_crops", crop_width=3)
                self.assertEqual(model.crops.localization(2, width=3).pixels[1, 1], 35)
                self.assertEqual(
                    [
                        c.pixels[1, 1]
                        for c in model.crops.photon_sequence("trace", width=3)
                    ],
                    [26, 90],
                )

    def test_shared_optical_calibration_aligns_both_alex_paths_in_every_round(self):
        from molcore.files import image_acquisition

        experiment = molcore.Experiment()
        for key in ("first", "second"):
            dataset = molcore.Dataset(id=key)
            dataset.add_acquisition(
                image_acquisition(
                    molcore.ArraySource(np.zeros((4, 10, 20))), id="movie", mode="alex"
                )
            )
            experiment.add_dataset(dataset)
        calibration = molcore.Calibration(
            "shared-optics", np.array([[1, 0, -2], [0, 1, 1], [0, 0, 1]])
        )
        experiment.apply_channel_calibration(calibration)
        for dataset in experiment.datasets.values():
            np.testing.assert_array_equal(
                dataset.transforms.matrix("movie", "DA", 0, space="registered"),
                calibration.matrix,
            )
            np.testing.assert_array_equal(
                dataset.transforms.matrix("movie", "AA", 1, space="registered"),
                calibration.matrix,
            )
            np.testing.assert_array_equal(
                dataset.transforms.matrix("movie", "DD", 0, space="registered"),
                np.eye(3),
            )
            np.testing.assert_array_equal(
                dataset.transforms.matrix("movie", "AD", 1, space="registered"),
                np.eye(3),
            )

    def test_rounds_keep_overlapping_local_ids_independent(self):
        self.assertTrue(
            hasattr(molcore, "Dataset"), "Probing rounds need a Dataset model"
        )
        experiment = molcore.Experiment()
        for name in ("round-1", "round-2"):
            dataset = molcore.Dataset(id=name)
            dataset.add_acquisition(
                molcore.Acquisition(
                    molcore.ArraySource(np.zeros((2, 8, 8))), id="movie"
                )
            )
            dataset.localizations.add(
                acquisition_id="movie", channel="image", frame=0, x=2, y=3
            )
            experiment.add_dataset(dataset)
        first, second = experiment.datasets.values()
        first.localizations.move(1, x=4, y=5)
        self.assertEqual(second.localizations.query(space="raw")["x"].tolist(), [2])

    def test_experiment_round_trip_keeps_full_sources_and_saved_crops_lazy(self):
        experiment = molcore.Experiment(metadata={"surface": "one"})
        for name in ("round-1", "round-2"):
            dataset = molcore.Dataset(
                id=name, name="Round " + name[-1], metadata={"barcode": name}
            )
            dataset.add_acquisition(
                molcore.Acquisition(
                    molcore.ArraySource(np.arange(128).reshape(2, 8, 8)), id="movie"
                )
            )
            dataset.localizations.add(
                acquisition_id="movie", channel="image", frame=0, x=2, y=3
            )
            experiment.add_dataset(dataset)
        with TemporaryDirectory() as directory:
            for policy in ("full", "localization_crops"):
                path = Path(directory) / (policy + ".h5")
                experiment.save(path, image_policy=policy, crop_width=3)
                with molcore.Experiment.open(path) as restored:
                    self.assertEqual(list(restored.datasets), ["round-1", "round-2"])
                    dataset = restored.datasets["round-2"]
                    self.assertEqual(dataset.name, "Round 2")
                    self.assertEqual(
                        dataset.crops.localization(1, width=3).pixels[1, 1], 26
                    )
                    if policy == "full":
                        self.assertIsInstance(
                            dataset.acquisitions["movie"].source, molcore.HDF5Source
                        )
                        self.assertEqual(
                            dataset.acquisitions["movie"].channels["image"][1][3, 2], 90
                        )


def rounds():
    experiment = molcore.Experiment()
    for name, dx in (("reference", 0), ("second", 2)):
        dataset = molcore.Dataset(id=name)
        pixels = np.zeros((3, 20, 20))
        pixels[:, 7, 6 + dx] = [10, 20, 30]
        dataset.add_acquisition(
            molcore.Acquisition(molcore.ArraySource(pixels), id="movie")
        )
        dataset.localizations.add(
            acquisition_id="movie",
            channel="image",
            frame=0,
            x=np.array([4, 12, 5, 13]) + dx,
            y=[4, 5, 12, 13],
        )
        experiment.add_dataset(dataset)
    return experiment


class RoundAlignmentTests(unittest.TestCase):
    def estimate(self, experiment):
        return experiment.alignment.estimate(
            "second",
            "reference",
            source_acquisition_id="movie",
            source_channel="image",
            reference_acquisition_id="movie",
            reference_channel="image",
            source_ids=[1, 2, 3, 4],
            reference_ids=[1, 2, 3, 4],
            num_cpus=1,
        )

    def test_alignment_maps_all_frames_without_changing_dataset_coordinates(self):
        experiment = rounds()
        result = self.estimate(experiment)
        experiment.alignment.apply([result])
        rows = experiment.datasets["second"].localizations.query(space="raw")
        x, y = experiment.alignment.coordinates("second", rows)
        np.testing.assert_allclose(x, [4, 12, 5, 13], atol=1e-10)
        np.testing.assert_allclose(y, [4, 5, 12, 13], atol=1e-10)
        np.testing.assert_array_equal(rows["x"], [6, 14, 7, 15])
        view = experiment.alignment.view("second", "movie", "image")
        np.testing.assert_allclose(view[:, 7, 6], [10, 20, 30], atol=1e-10)
        np.testing.assert_array_equal(
            experiment.datasets["second"].acquisitions["movie"].source[:, 7, 8],
            [10, 20, 30],
        )

    def test_edit_rejects_preview_and_saved_alignment(self):
        experiment = rounds()
        result = self.estimate(experiment)
        experiment.datasets["reference"].localizations.move(1, x=8, y=8)
        with self.assertRaises(molcore.StaleResultError):
            experiment.alignment.apply([result])
        self.assertEqual(dict(experiment.alignment.records), {})

    def test_removal_requires_cascade_and_reset_leaves_local_data(self):
        experiment = rounds()
        experiment.alignment.apply([self.estimate(experiment)])
        with self.assertRaises(ValueError):
            experiment.remove_dataset("reference")
        experiment.remove_dataset("reference", cascade=True)
        self.assertEqual(dict(experiment.alignment.records), {})
        self.assertEqual(
            len(experiment.datasets["second"].localizations.query()["id"]), 4
        )

    def test_alignment_survives_reopening_and_becomes_stale_on_input_edit(self):
        experiment = rounds()
        experiment.alignment.apply([self.estimate(experiment)])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.h5"
            experiment.save(path, image_policy="full")
            with molcore.Experiment.open(path) as restored:
                self.assertFalse(restored.alignment.is_stale("second"))
                self.assertAlmostEqual(
                    restored.alignment.image("second", "movie", "image", 2)[7, 6], 30
                )
                restored.datasets["reference"].localizations.move(1, x=5, y=5)
                self.assertTrue(restored.alignment.is_stale("second"))
                with self.assertRaises(molcore.StaleResultError):
                    restored.alignment.image("second", "movie", "image", 0)


if __name__ == "__main__":
    unittest.main()
