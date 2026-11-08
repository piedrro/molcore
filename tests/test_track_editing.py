"""Track-only retention and rolling statistics preserve stable scientific inputs."""

import unittest

import numpy as np
from scientific_helpers import compute_scenario

from molcore import Acquisition, ArraySource, Dataset, PhotonSequence, StaleResultError
from molcore.staging import state_token
from molcore.trajectory import trajectory_statistics
from molcore.types import CoordinateSpace, SpatialUnit


class TrackEditingTests(unittest.TestCase):
    def setUp(self):
        self.model = Dataset()
        self.images = np.arange(5 * 9 * 9).reshape(5, 9, 9)
        self.model.add_acquisition(
            Acquisition(ArraySource(self.images), id="a", timestamps=[0, 0.2, 0.7, 1.5, 2.5], pixel_size=100)
        )
        self.ids = self.model.localizations.add(
            acquisition_id="a", channel="image", frame=np.arange(5), x=[0, 1, 3, 6, 10], y=0
        )
        self.track = self.model.tracks.add(self.ids, metadata={"origin": "import"})
        self.other = self.model.localizations.add(acquisition_id="a", channel="image", frame=[0], x=2, y=2)

    def add_dependents(self):
        observations = self.model.observations.add(
            molecule_id="m",
            acquisition_id="a",
            channel="image",
            frame=[0, 1],
            intensity=[10, 20],
            track_id=self.track,
        )
        self.model.photon_sequences.add(
            PhotonSequence(
                id="dependent",
                acquisition_id="a",
                channel="image",
                frames=[0, 1],
                photons=[10, 20],
                positions=[[0, 0], [1, 0]],
                track_id=self.track,
            )
        )
        return observations

    def test_retain_arbitrary_subset_preserves_track_id_metadata_and_source_rows(self):
        self.model.tracks.retain_members(self.track, self.ids[[4, 1, 3]])
        np.testing.assert_array_equal(self.model.tracks.query(track_id=self.track)["id"], self.ids[[1, 3, 4]])
        self.assertEqual(self.model.tracks.metadata[self.track]["parameters"], {"origin": "import"})
        np.testing.assert_array_equal(
            self.model.localizations.query(space="raw")["id"], np.r_[self.ids, self.other]
        )
        np.testing.assert_array_equal(
            self.model.localizations.query(ids=self.ids, space="raw")["frame"], np.arange(5)
        )
        self.assertEqual(len(self.model.tracks), 1)
        self.assertEqual(
            [crop.camera_frame for crop in compute_scenario(self.model.crops.track, self.track, width=3)],
            [1, 3, 4],
        )
        self.assertEqual(self.model.crops.localization(self.ids[0], width=3).camera_frame, 0)

    def test_membership_change_invalidates_dependent_measurements_and_sequence_crops(self):
        observations = self.add_dependents()
        self.model.photon_sequences.add(
            PhotonSequence(id="independent", acquisition_id="a", channel="image", frames=[0], photons=[5])
        )
        self.model.tracks.retain_members(self.track, self.ids[1:])
        self.assertEqual(len(self.model.observations.query(ids=observations)["id"]), 0)
        self.assertFalse(self.model.observations.query(ids=observations, include_stale=True)["valid"].any())
        self.assertTrue(self.model.photon_sequences.is_stale("dependent"))
        self.assertFalse(self.model.photon_sequences.is_stale("independent"))
        with self.assertRaises(StaleResultError):
            list(self.model.crops.photon_sequence("dependent", width=3))
        self.assertEqual(
            len(list(self.model.crops.photon_sequence("dependent", width=3, include_stale=True))), 2
        )

    def test_invalid_retain_scope_does_not_partially_mutate_or_invalidate(self):
        self.add_dependents()
        before = state_token(self.model)
        for ids in ([self.ids[0], self.ids[0]], [999], self.other, [1.5], [[1]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.model.tracks.retain_members(self.track, ids)
            self.assertEqual(state_token(self.model), before)
        with self.assertRaises(ValueError):
            self.model.tracks.retain_members(999, [])
        self.assertEqual(state_token(self.model), before)

    def test_empty_retain_deletes_track_only_and_invalidates_dependents(self):
        self.add_dependents()
        self.model.tracks.retain_members(self.track, [])
        self.assertEqual(len(self.model.tracks), 0)
        self.assertNotIn(self.track, self.model.tracks.metadata)
        self.assertEqual(len(self.model.localizations), 6)
        self.assertTrue(self.model.photon_sequences.is_stale("dependent"))
        self.assertEqual(len(self.model.observations.query()["id"]), 0)

    def test_retaining_all_members_is_a_noop(self):
        self.add_dependents()
        before = state_token(self.model)
        self.model.tracks.retain_members(self.track, self.ids[::-1])
        self.assertEqual(state_token(self.model), before)
        self.assertFalse(self.model.photon_sequences.is_stale("dependent"))

    def test_selected_statistics_use_original_frames_timestamps_and_units(self):
        before = state_token(self.model)
        result = trajectory_statistics(
            self.model.snapshot(),
            self.track,
            ids=self.ids[[4, 1, 3]],
            space=CoordinateSpace.RAW,
            unit=SpatialUnit.NANOMETRES,
            frame_interval=99,
        )
        self.assertEqual(result.track_id, self.track)
        np.testing.assert_array_equal(result.steps["frame_start"], [1, 3])
        np.testing.assert_array_equal(result.steps["frame_end"], [3, 4])
        np.testing.assert_allclose(result.steps["elapsed_time"], [1.3, 1])
        np.testing.assert_allclose(result.steps["distance"], [500, 400])
        np.testing.assert_allclose(result.msd["lag"], [1, 1.3, 2.3])
        np.testing.assert_allclose(result.msd["msd"], [160000, 250000, 810000])
        self.assertAlmostEqual(result.duration, 2.3)
        self.assertAlmostEqual(result.path_length, 900)
        self.assertEqual(state_token(self.model), before)
        np.testing.assert_array_equal(self.model.acquisitions["a"].source[:], self.images)

    def test_invalid_statistics_scope_is_rejected_without_mutation(self):
        before = state_token(self.model)
        for ids in ([self.ids[0], self.ids[0]], [999], self.other, [1.5], [[1]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                trajectory_statistics(self.model.snapshot(), self.track, ids=ids)
            self.assertEqual(state_token(self.model), before)

    def test_default_empty_and_single_member_statistics(self):
        full = trajectory_statistics(self.model.snapshot(), self.track)
        self.assertAlmostEqual(full.duration, 2.5)
        self.assertAlmostEqual(full.path_length, 10)
        for ids in ([], self.ids[:1]):
            with self.subTest(ids=ids):
                result = trajectory_statistics(self.model.snapshot(), self.track, ids=ids)
                self.assertEqual(len(result.steps["distance"]), 0)
                self.assertEqual(len(result.msd["count"]), 0)
                self.assertEqual(result.duration, 0)
                self.assertEqual(result.path_length, 0)
