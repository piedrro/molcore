"""Independent photon sequences retain provenance, crop pixels and snapshot safety."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from molcore import Acquisition, ArraySource, Experiment, PhotonSequence, PhotonUnit
from molcore.photon_sequences import from_observations
from molcore.staging import clone_experiment, state_token, transact


class PhotonSequenceTests(unittest.TestCase):
    def setUp(self):
        self.model = Experiment()
        self.images = np.arange(5 * 9 * 9, dtype=np.uint16).reshape(5, 9, 9)
        self.model.add_acquisition(
            Acquisition(ArraySource(self.images), id="a", timestamps=np.arange(5) * 0.2)
        )

    def sequence(self, **kwargs):
        values = dict(
            id="p",
            acquisition_id="a",
            channel="image",
            frames=np.array([0, 2, 4]),
            photons=np.array([10.0, np.nan, -1.0]),
            positions=np.array([[0.0, 0.0], [4.0, 4.0], [5.0, 5.0]]),
        )
        values.update(kwargs)
        return PhotonSequence(**values)

    def test_separate_model_and_detached_arrays(self):
        sequence = self.sequence()
        saved = self.model.photon_sequences.add(sequence)
        self.assertEqual(len(self.model.localizations), 0)
        self.assertEqual(len(self.model.photon_sequences), 1)
        self.assertIs(saved.unit, PhotonUnit.PHOTONS)
        np.testing.assert_allclose(saved.timestamps, [0.0, 0.4, 0.8])
        self.assertFalse(saved.photons.flags.writeable)
        with self.assertRaises(ValueError):
            saved.photons.setflags(write=True)
        copy = self.model.photon_sequences["p"]
        copy.metadata["changed"] = True
        self.assertNotIn("changed", self.model.photon_sequences["p"].metadata)

    def test_sequence_crops_preserve_sample_frames_and_padding(self):
        self.model.photon_sequences.add(self.sequence())
        crops = list(self.model.crops.photon_sequence("p", width=3))
        self.assertEqual([c.camera_frame for c in crops], [0, 2, 4])
        self.assertEqual([c.sample_index for c in crops], [0, 1, 2])
        self.assertEqual(crops[0].photon_sequence_id, "p")
        self.assertEqual(crops[0].valid.sum(), 4)
        self.assertEqual(crops[1].pixels[1, 1], self.images[2, 4, 4])

    def test_every_crop_combination_round_trips(self):
        ids = self.model.localizations.add(
            acquisition_id="a", channel="image", frame=[0, 2], x=[4.0, 4.0], y=[4.0, 4.0]
        )
        track = self.model.tracks.add(ids)
        self.model.photon_sequences.add(self.sequence())
        policies = {
            "localization_crops": (True, True, False),
            "track_crops": (True, True, False),
            "both": (True, True, False),
            "photon_sequence_crops": (False, False, True),
            "localization_and_photon_crops": (True, True, True),
            "track_and_photon_crops": (True, True, True),
            "all_crops": (True, True, True),
        }
        with tempfile.TemporaryDirectory() as directory:
            for policy, (_, _, photons) in policies.items():
                path = Path(directory) / f"{policy}.h5"
                self.model.save(path, image_policy=policy, crop_width=3)
                with Experiment.open(path) as loaded:
                    np.testing.assert_allclose(
                        loaded.photon_sequences["p"].photons, [10.0, np.nan, -1.0], equal_nan=True
                    )
                    if photons:
                        crops = list(loaded.crops.photon_sequence("p", width=3))
                        self.assertEqual(crops[1].pixels[1, 1], self.images[2, 4, 4])
                        self.assertIsNone(crops[1].localization_id)
                    else:
                        from molcore import ImageUnavailableError

                        with self.assertRaises(ImageUnavailableError):
                            list(loaded.crops.photon_sequence("p", width=3))
                    if policy != "photon_sequence_crops":
                        self.assertEqual(len(list(loaded.crops.track(track, width=3))), 2)

    def test_clone_transaction_and_anchored_invalidation(self):
        lid = int(
            self.model.localizations.add(acquisition_id="a", channel="image", frame=[0], x=[4.0], y=[4.0])[0]
        )
        before = state_token(self.model)
        self.model.photon_sequences.add(self.sequence(localization_id=lid))
        self.assertNotEqual(before, state_token(self.model))
        cloned = clone_experiment(self.model)
        self.assertFalse(cloned.photon_sequences["p"].photons.flags.writeable)
        transact(self.model, lambda model: model.photon_sequences.add(self.sequence(id="q")))
        self.assertEqual(len(self.model.photon_sequences), 2)
        self.model.localizations.move(lid, x=5.0, y=4.0)
        self.assertTrue(self.model.photon_sequences.is_stale("p"))
        self.assertEqual([s.id for s in self.model.photon_sequences.query()], ["q"])

    def test_conversion_preserves_intensity_units_and_actual_positions(self):
        self.model.observations.add(
            molecule_id=["m", "m"],
            acquisition_id="a",
            channel="image",
            frame=[0, 2],
            intensity=[10.0, 20.0],
            x=[1.0, 2.0],
            y=[3.0, 4.0],
            run_id="r",
        )
        sequences = from_observations(self.model, run_id="r")
        self.assertEqual(len(sequences), 1)
        self.assertIs(sequences[0].unit, PhotonUnit.IMAGE_INTENSITY)
        np.testing.assert_allclose(sequences[0].positions, [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(len(self.model.photon_sequences), 0)

    def test_validate_frames_lengths_and_timing(self):
        for fields in (
            dict(frames=[2, 0, 4]),
            dict(photons=[1.0]),
            dict(photons=[1.0, np.inf, 2.0]),
            dict(positions=[[1.0, 2.0]]),
            dict(frames=[0.0, 2.0, 4.0]),
        ):
            with self.assertRaises((TypeError, ValueError)):
                self.sequence(**fields)
        with self.assertRaises(ValueError):
            self.model.photon_sequences.add(self.sequence(frames=[0, 2, 5]))
        with self.assertRaises(ValueError):
            self.model.photon_sequences.add(self.sequence(timestamps=[0.0, 0.1, 0.2]))

    def test_acquisition_removal_checks_photon_dependents(self):
        self.model.photon_sequences.add(self.sequence())
        with self.assertRaises(ValueError):
            self.model.remove_acquisition("a")
        self.model.remove_acquisition("a", cascade=True)
        self.assertEqual(len(self.model.photon_sequences), 0)

    def test_missing_positions_preserve_sample_indices(self):
        self.model.photon_sequences.add(self.sequence(positions=[[0.0, 0.0], [np.nan, np.nan], [5.0, 5.0]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.h5"
            self.model.save(path, image_policy="photon_sequence_crops", crop_width=3)
            with Experiment.open(path) as loaded:
                crops = list(loaded.crops.photon_sequence("p", width=3))
                self.assertIsNone(crops[1])
                self.assertEqual(crops[2].sample_index, 2)

    def test_version_one_files_still_open(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.h5"
            self.model.save(path)
            with h5py.File(path, "r+") as stream:
                stream.attrs["format_version"] = 1
                del stream["photon_sequences"]
            with Experiment.open(path) as loaded:
                self.assertEqual(len(loaded.photon_sequences), 0)

    def test_crop_only_snapshot_can_be_resaved_without_movies(self):
        self.model.photon_sequences.add(self.sequence())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photons.h5"
            self.model.save(path, image_policy="photon_sequence_crops", crop_width=3)
            with Experiment.open(path) as loaded:
                loaded.save(path, image_policy="photon_sequence_crops", crop_width=3)
                self.assertEqual(
                    list(loaded.crops.photon_sequence("p", width=3))[1].pixels[1, 1], self.images[2, 4, 4]
                )
                from molcore import ImageUnavailableError

                with self.assertRaises(ImageUnavailableError):
                    list(loaded.crops.photon_sequence("p", width=5))

    def test_alternating_channels_preserve_original_camera_frames(self):
        from molcore import ChannelSpec

        model = Experiment()
        model.add_acquisition(
            Acquisition(
                ArraySource(self.images),
                id="a",
                excitation_sequence=("D", "A"),
                channel_specs={"DD": ChannelSpec(excitation="D")},
                timestamps=np.arange(5) * 0.2,
            )
        )
        model.photon_sequences.add(self.sequence(channel="DD"))
        self.assertEqual([c.camera_frame for c in model.crops.photon_sequence("p", width=3)], [0, 2, 4])
        with self.assertRaises(ValueError):
            model.photon_sequences.add(self.sequence(id="bad", channel="DD", frames=[0, 1, 4]))

    def test_independent_sequences_survive_unrelated_model_changes(self):
        from molcore import Calibration, Segmentation

        self.model.photon_sequences.add(self.sequence())
        self.model.add_segmentation(Segmentation("mask", "a", np.ones((9, 9), dtype=int)))
        self.assertFalse(self.model.photon_sequences.is_stale("p"))
        self.model.transforms.add_calibration(Calibration("cal", np.eye(3)))
        self.model.transforms.select("a", "image", "cal")
        self.assertFalse(self.model.photon_sequences.is_stale("p"))

    def test_transform_dependent_sequences_are_explicit(self):
        from molcore import Calibration, Segmentation

        self.model.photon_sequences.add(self.sequence(depends_on_transforms=True))
        self.model.add_segmentation(Segmentation("mask", "a", np.ones((9, 9), dtype=int)))
        self.assertFalse(self.model.photon_sequences.is_stale("p"))
        self.model.transforms.add_calibration(Calibration("cal", np.eye(3)))
        self.model.transforms.select("a", "image", "cal")
        self.assertTrue(self.model.photon_sequences.is_stale("p"))

    def test_stationary_trace_writes_primary_sequences_without_observations(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import stationary_traces
        from molcore.types import PhotonSequenceResult

        seed = self.model.localizations.add(acquisition_id="a", channel="image", frame=[0], x=[4.0], y=[4.0])
        result = stationary_traces(self.model, seed, cpu=InlineCpu())
        self.assertIsInstance(result, PhotonSequenceResult)
        self.assertEqual(len(result.sequence_ids), 1)
        self.assertEqual(len(self.model.observations), 0)
        self.assertEqual(len(self.model.localizations), 1)
        self.assertEqual(len(self.model.tracks), 0)
        sequence = self.model.photon_sequences[result.sequence_ids[0]]
        self.assertIn("signal_sum", sequence.measurements)
        np.testing.assert_allclose(sequence.photons, result.columns["intensity"])
        self.assertIs(sequence.unit, PhotonUnit.IMAGE_INTENSITY)

    def test_moving_trace_keeps_track_as_localization_memberships(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import track_traces

        ids = self.model.localizations.add(
            acquisition_id="a", channel="image", frame=[0, 2, 4], x=[2.0, 3.0, 4.0], y=[4.0, 4.0, 4.0]
        )
        track = self.model.tracks.add(ids)
        result = track_traces(self.model, [track], cpu=InlineCpu())
        self.assertEqual(len(self.model.observations), 0)
        np.testing.assert_array_equal(self.model.tracks.query(track_id=track)["id"], ids)
        sequence = self.model.photon_sequences[result.sequence_ids[0]]
        self.assertEqual(sequence.track_id, track)
        np.testing.assert_array_equal(sequence.localization_ids, ids)
        crops = list(self.model.crops.photon_sequence(sequence.id, width=3))
        self.assertEqual([crop.localization_id for crop in crops], list(ids))

    def test_trace_metrics_and_sample_anchors_round_trip(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import apply_photometry, stationary_traces

        seed = self.model.localizations.add(acquisition_id="a", channel="image", frame=[0], x=[4.0], y=[4.0])
        preview = stationary_traces(self.model, seed, cpu=InlineCpu(), apply=False)
        self.assertEqual(len(self.model.photon_sequences), 0)
        applied = apply_photometry(self.model, preview)
        with self.assertRaises(ValueError):
            apply_photometry(self.model, preview, expected=state_token(self.model))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "primary.h5"
            self.model.save(path, image_policy="photon_sequence_crops", crop_width=3)
            with Experiment.open(path) as loaded:
                stored = loaded.photon_sequences[applied.sequence_ids[0]]
                np.testing.assert_allclose(
                    stored.measurements["signal_sum"], preview.sequences[0].measurements["signal_sum"]
                )
                np.testing.assert_array_equal(stored.localization_ids, np.repeat(seed, 5))
                self.assertEqual(len(loaded.observations), 0)

    def test_independent_trace_runs_apply_in_one_batch(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import apply_photometry, stationary_traces
        from molcore.processing import apply_batch

        seeds = self.model.localizations.add(
            acquisition_id="a", channel="image", frame=[0, 0], x=[3.0, 5.0], y=[4.0, 4.0]
        )
        first = stationary_traces(self.model, seeds[:1], cpu=InlineCpu(), apply=False)
        second = stationary_traces(self.model, seeds[1:], cpu=InlineCpu(), apply=False)
        apply_batch(
            self.model,
            [lambda model: apply_photometry(model, first), lambda model: apply_photometry(model, second)],
        )
        self.assertEqual(len(self.model.photon_sequences), 2)
        self.assertEqual(len(self.model.observations), 0)

    def test_sample_metrics_are_detached_and_immutable(self):
        sequence = self.sequence(measurements={"signal_sum": np.array([12.0, 13.0, 14.0])})
        self.model.photon_sequences.add(sequence)
        copy = self.model.photon_sequences["p"]
        with self.assertRaises(ValueError):
            copy.measurements["signal_sum"].setflags(write=True)
        copy.measurements["signal_sum"] = np.zeros(3)
        np.testing.assert_array_equal(
            self.model.photon_sequences["p"].measurements["signal_sum"], [12.0, 13.0, 14.0]
        )
        with self.assertRaises(ValueError):
            self.sequence(measurements={"signal_sum": np.ones(2)})
        with self.assertRaises(ValueError):
            self.sequence(measurements={"frame": np.ones(3)})

    def test_legacy_trace_preview_is_applied_to_primary_sequence_model(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import apply_photometry, stationary_traces
        from molcore.types import PhotometryResult

        seed = self.model.localizations.add(acquisition_id="a", channel="image", frame=[0], x=[4.0], y=[4.0])
        preview = stationary_traces(self.model, seed, cpu=InlineCpu(), apply=False)
        legacy = PhotometryResult(
            preview.columns, preview.settings, preview.mode, preview.run_id, preview.expected
        )
        result = apply_photometry(self.model, legacy)
        self.assertEqual(len(self.model.observations), 0)
        self.assertEqual(len(result.sequence_ids), 1)

    def test_alternate_primary_preserves_original_intensity(self):
        from molcore.photon_sequences import to_columns

        self.model.observations.add(
            molecule_id=["m", "m"],
            acquisition_id="a",
            channel="image",
            frame=[0, 2],
            intensity=[10.0, 20.0],
            signal_sum=[15.0, 25.0],
            run_id="r",
        )
        sequence = from_observations(self.model, run_id="r", value_column="signal_sum")[0]
        np.testing.assert_array_equal(sequence.measurements["source_intensity"], [10.0, 20.0])
        np.testing.assert_array_equal(to_columns([sequence])["source_intensity"], [10.0, 20.0])

    def test_neighbour_changes_invalidate_only_dependent_sequences(self):
        from test_photometry_trajectory import InlineCpu

        from molcore import PhotometrySettings
        from molcore.photometry import stationary_traces

        seeds = self.model.localizations.add(
            acquisition_id="a", channel="image", frame=[0, 0], x=[3.0, 5.0], y=[4.0, 4.0]
        )
        dependent = stationary_traces(self.model, seeds[:1], cpu=InlineCpu())
        independent = stationary_traces(
            self.model, seeds[:1], cpu=InlineCpu(), settings=PhotometrySettings(exclude_neighbours=False)
        )
        self.model.localizations.move(int(seeds[1]), x=6.0, y=4.0)
        self.assertTrue(self.model.photon_sequences.is_stale(dependent.sequence_ids[0]))
        self.assertFalse(self.model.photon_sequences.is_stale(independent.sequence_ids[0]))

    def test_adding_neighbours_invalidates_background_dependent_traces(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import stationary_traces

        seed = self.model.localizations.add(acquisition_id="a", channel="image", frame=[0], x=[4.0], y=[4.0])
        result = stationary_traces(self.model, seed, cpu=InlineCpu())
        self.model.localizations.add(acquisition_id="a", channel="image", frame=[0], x=[6.0], y=[4.0])
        self.assertTrue(self.model.photon_sequences.is_stale(result.sequence_ids[0]))


if __name__ == "__main__":
    unittest.main()
