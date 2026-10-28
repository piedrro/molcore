import unittest

import numpy as np

from molcore import Acquisition, ArraySource, Calibration, Dataset, Segmentation


class IntegrityTests(unittest.TestCase):
    def test_empty_partition_replacement_clears_active_results(self):
        self.exp.localizations.add(acquisition_id="a", channel="image", frame=0, x=1, y=1)
        self.exp.localizations.replace(
            "a",
            "image",
            frame=np.array([], dtype=np.uint64),
            x=np.array([], dtype=float),
            y=np.array([], dtype=float),
        )
        self.assertEqual(len(self.exp.localizations), 0)

    def setUp(self):
        self.exp = Dataset()
        self.exp.add_acquisition(Acquisition(ArraySource(np.zeros((5, 7, 7))), id="a"))

    def test_large_ids_remain_distinct_when_selected_by_python_integers(self):
        ids = self.exp.localizations.add(
            acquisition_id="a",
            channel="image",
            frame=[0, 1],
            x=[1.0, 2.0],
            y=[1.0, 2.0],
            id=np.array([2**53, 2**53 + 1], dtype=np.uint64),
        )
        selected = self.exp.localizations.query(ids=[int(ids[1])])
        self.assertEqual(selected["id"].tolist(), [2**53 + 1])
        self.exp.localizations.move(int(ids[1]), x=3, y=3)
        self.assertEqual(self.exp.localizations.query()["x"].tolist(), [1, 3])

    def test_reassign_accepts_python_integer_ids(self):
        ids = self.exp.localizations.add(acquisition_id="a", channel="image", frame=[0, 1, 2], x=1, y=1)
        track = self.exp.tracks.add(ids[:2])
        self.exp.tracks.reassign([int(ids[2])], track)
        self.assertEqual(self.exp.tracks.query(track_id=track)["frame"].tolist(), [0, 1, 2])

    def test_unanchored_observations_do_not_round_large_localization_anchors(self):
        ids = self.exp.localizations.add(
            acquisition_id="a",
            channel="image",
            frame=[0, 1],
            x=1,
            y=1,
            id=np.array([2**53, 2**53 + 1], dtype=np.uint64),
        )
        self.exp.observations.add(molecule_id="m", acquisition_id="a", channel="image", frame=0, intensity=1)
        self.exp.observations.add(
            molecule_id="n", acquisition_id="a", channel="image", frame=1, intensity=2, localization_id=ids[1]
        )
        rows = self.exp.observations.query()
        self.assertEqual(int(rows["localization_id"][1]), 2**53 + 1)
        self.exp.localizations.move(int(ids[1]), x=2, y=2)
        self.assertEqual(self.exp.observations.query()["molecule_id"].tolist(), ["m"])

    def test_failed_split_does_not_mutate_original_track(self):
        labels = np.ones((7, 7), dtype=np.uint16)
        labels[6, 6] = 0
        self.exp.add_segmentation(Segmentation("mask", "a", labels))
        ids = self.exp.localizations.add(acquisition_id="a", channel="image", frame=[0, 1, 2], x=1, y=1)
        track = self.exp.tracks.add(ids, segmentation_id="mask")
        self.exp.localizations.move(ids[2], x=6, y=6)
        with self.assertRaises(ValueError):
            self.exp.tracks.split(track, frame=2)
        self.assertEqual(self.exp.tracks.query(track_id=track)["id"].tolist(), ids.tolist())

    def test_numpy_ellipsis_indexes_spatial_axes(self):
        movie = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        source = ArraySource(movie)
        np.testing.assert_array_equal(source[..., 2], movie[..., 2])

    def test_deleted_explicit_ids_cannot_be_reused(self):
        ids = self.exp.localizations.add(acquisition_id="a", channel="image", frame=0, x=1, y=1)
        self.exp.localizations.delete(ids)
        with self.assertRaises(ValueError):
            self.exp.localizations.add(acquisition_id="a", channel="image", frame=1, x=1, y=1, id=ids[0])

    def test_partition_replacement_keeps_other_acquisitions_and_validates_first(self):
        self.exp.add_acquisition(Acquisition(ArraySource(np.zeros((5, 7, 7))), id="b"))
        ids = self.exp.localizations.add(acquisition_id=["a", "b"], channel="image", frame=0, x=1, y=1)
        with self.assertRaises(ValueError):
            self.exp.localizations.replace("a", "image", frame=99, x=1, y=1)
        self.assertEqual(self.exp.localizations.query()["id"].tolist(), ids.tolist())
        self.exp.localizations.replace("a", "image", frame=2, x=2, y=2)
        self.assertEqual(self.exp.localizations.query(acquisition_id="b")["id"].tolist(), [int(ids[1])])

    def test_incompatible_measurement_replacement_is_atomic(self):
        self.exp.add_acquisition(Acquisition(ArraySource(np.zeros((5, 7, 7))), id="b"))
        ids = self.exp.localizations.add(
            acquisition_id=["a", "b"],
            channel="image",
            frame=0,
            x=1,
            y=1,
            quality=np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
        )
        with self.assertRaises((ValueError, TypeError)):
            self.exp.localizations.replace("a", "image", frame=1, x=2, y=2, quality=1.0)
        self.assertEqual(self.exp.localizations.query()["id"].tolist(), ids.tolist())

    def test_selected_calibration_cannot_be_mutated_through_registry(self):
        self.exp.transforms.add_calibration(Calibration("identity"))
        self.exp.transforms.select("a", "image", "identity")
        with self.assertRaises(ValueError):
            self.exp.calibrations["identity"].matrix[0, 2] = 3

    def test_sequence_frame_ranges_rebuild_after_membership_changes(self):
        self.exp.add_acquisition(Acquisition(ArraySource(np.zeros((5, 7, 7))), id="b"))
        self.exp.add_sequence(["a", "b"], id="s")
        ids = self.exp.localizations.add(
            acquisition_id=["a", "b", "b"], channel="image", frame=[4, 0, 2], x=1, y=1
        )
        rows = self.exp.localizations.query(sequence_id="s", frame_range=(5, 8))
        self.assertEqual(rows["id"].tolist(), ids[1:].tolist())
        self.assertEqual(rows["sequence_frame"].tolist(), [5, 7])
        self.exp.localizations.delete([ids[1]])
        self.assertEqual(
            self.exp.localizations.query(sequence_id="s", frame_range=(5, 8))["id"].tolist(), [int(ids[2])]
        )

    def test_empty_results_keep_their_query_schema(self):
        self.exp.add_sequence(["a"], id="s")
        self.assertEqual(self.exp.localizations.query(sequence_id="s")["sequence_frame"].shape, (0,))
        self.assertEqual(self.exp.observations.query(frame_range=(0, 1))["intensity"].shape, (0,))
