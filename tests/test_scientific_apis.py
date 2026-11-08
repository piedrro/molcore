"""Identity, rollback and conversion contracts for scientific editing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scientific_helpers import compute_scenario

from molcore import Acquisition, ArraySource, CoordinateSpace, Dataset, StaleResultError
from molcore.staging import state_token


def fixture():
    exp = Dataset()
    exp.add_acquisition(Acquisition(ArraySource(np.ones((5, 11, 11))), id="a"))
    ids = exp.localizations.add(
        acquisition_id="a", channel="image", frame=[0, 2, 4], x=[3.0, 4.0, 5.0], y=[4.0, 4.0, 4.0]
    )
    return (exp, ids)


class EditingTests(unittest.TestCase):
    def test_measurements_identity_persistence_and_rollback(self):
        exp, ids = fixture()
        track = exp.tracks.add(ids)
        exp.observations.add(
            molecule_id="m", acquisition_id="a", channel="image", frame=0, intensity=1.0, track_id=track
        )
        exp.localizations.update_measurements(ids[[2, 0]], signal=[20.0, 10.0])
        np.testing.assert_equal(exp.localizations.query()["signal"], [10.0, np.nan, 20.0])
        self.assertEqual(len(exp.observations.query()["id"]), 0)
        token = state_token(exp)
        for values in ({"x": [1.0, 2.0]}, {"signal": [1.0, "bad"]}):
            with self.assertRaises(ValueError):
                exp.localizations.update_measurements(ids[:2], **values)
            self.assertEqual(state_token(exp), token)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.h5"
            exp.save(path)
            with Dataset.open(path) as restored:
                np.testing.assert_equal(restored.localizations.query()["id"], ids)
                np.testing.assert_equal(restored.localizations.query()["signal"], [10.0, np.nan, 20.0])

    def test_batch_and_bulk_positions(self):
        from molcore.processing import apply_batch

        exp, ids = fixture()
        token = state_token(exp)
        with self.assertRaises(ValueError):
            apply_batch(
                exp,
                [
                    lambda e: e.localizations.move_many(ids, x=7.0, y=8.0),
                    lambda e: e.localizations.move_many([999], x=0.0, y=0.0),
                ],
            )
        self.assertEqual(state_token(exp), token)
        exp.localizations.move_many(ids, x=7.0, y=8.0)
        np.testing.assert_equal(exp.localizations.query(original=True)["x"], [3.0, 4.0, 5.0])
        with self.assertRaises(StaleResultError):
            apply_batch(exp, [], expected=token)

    def test_track_partition_replacement(self):
        exp, ids = fixture()
        old = exp.tracks.add(ids)
        with self.assertRaises(ValueError):
            exp.tracks.replace("a", "image", [ids[:2], ids[1:]])
        self.assertEqual(list(exp.tracks.metadata), [old])
        tracks = exp.tracks.replace("a", "image", [ids[:2]])
        self.assertGreater(tracks[0], old)
        np.testing.assert_equal(exp.localizations.query()["id"], ids)
        self.assertEqual(len(exp.tracks.query(acquisition_id="a", channel="image")["id"]), 2)

    def test_removal_cascades_without_closing_borrowed_source(self):
        exp, ids = fixture()
        exp.tracks.add(ids)
        with self.assertRaises(ValueError):
            exp.remove_acquisition("a")
        exp.remove_acquisition("a", cascade=True)
        self.assertEqual(len(exp.localizations), 0)
        self.assertEqual(len(exp.tracks), 0)
        self.assertFalse(exp.acquisitions)

    def test_bulk_conversions(self):
        exp, ids = fixture()
        view = exp.acquisitions["a"].channels["image"]
        np.testing.assert_equal(view.camera_frames([4, 0, 2]), [4, 0, 2])
        np.testing.assert_equal(view.channel_indices([4, 0, 2]), [4, 0, 2])
        exp.transforms.set_drift("a", [[1.0, 2.0]] * 5)
        rows = exp.localizations.query(space="raw")
        converted = exp.transforms.convert_coordinates(
            rows, source_space=CoordinateSpace.RAW, target_space=CoordinateSpace.CORRECTED
        )
        rows = rows.to_columns()
        rows["x"], rows["y"] = converted
        x, y = exp.transforms.convert_coordinates(
            rows, source_space=CoordinateSpace.CORRECTED, target_space=CoordinateSpace.RAW
        )
        np.testing.assert_allclose(x, [3.0, 4.0, 5.0])
        np.testing.assert_allclose(y, 4.0)


class ScientificEdgeTests(unittest.TestCase):
    def test_processing_metadata_is_a_removal_dependency(self):
        exp, ids = fixture()
        exp.localizations.delete(ids)
        exp.update_processing({"localisation": {"a": {"image": {"settings": {}}}}})
        with self.assertRaises(ValueError):
            exp.remove_acquisition("a")
        exp.remove_acquisition("a", cascade=True)
        self.assertNotIn("a", exp.processing.settings["localisation"])

    def test_shared_source_and_spanning_dependencies(self):
        from unittest.mock import Mock, patch

        exp, ids = fixture()
        source = exp.acquisitions["a"].source
        patcher = patch.object(type(source), "close", Mock())
        patcher.start()
        self.addCleanup(patcher.stop)
        exp.add_acquisition(Acquisition(source, id="b"))
        other = exp.localizations.add(acquisition_id="b", channel="image", frame=0, x=2.0, y=3.0)
        exp.add_sequence(["a", "b"], id="s", allow_cross_boundary_tracks=True, reference="r")
        track = exp.tracks.add(np.r_[ids, other], sequence_id="s")
        exp.observations.add(
            molecule_id="m", acquisition_id="b", channel="image", frame=0, intensity=1.0, track_id=track
        )
        with self.assertRaises(ValueError):
            exp.tracks.replace("a", "image", [])
        exp.remove_acquisition("a", cascade=True)
        np.testing.assert_equal(exp.localizations.query()["id"], other)
        self.assertEqual(len(exp.observations.query(include_stale=True)["id"]), 0)
        self.assertFalse(exp.sequences)
        source.close.assert_not_called()
        np.testing.assert_allclose(exp.acquisitions["b"].channels["image"][0], 1.0)

    def test_timestamp_changes_reject_preview(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import measure_localizations

        exp, ids = fixture()
        preview = compute_scenario(measure_localizations, exp, ids, apply=False, cpu=InlineCpu())
        exp.update_acquisition("a", timestamps=np.arange(5, dtype=float))
        with self.assertRaises(StaleResultError):
            exp.apply(preview)

    def test_empty_frame_conversions(self):
        exp, _ = fixture()
        view = exp.acquisitions["a"].channels["image"]
        self.assertEqual(view.camera_frames([]).dtype.kind, "i")
        self.assertEqual(view.channel_indices([]).size, 0)

    def test_duplicate_ids_and_float_frames_are_rejected(self):
        exp, ids = fixture()
        with self.assertRaises(ValueError):
            exp.localizations.update_measurements([ids[0], ids[0]], signal=1.0)
        with self.assertRaises(ValueError):
            exp.tracks.query(track_id=[1.5])
        rows = exp.localizations.query()
        rows = rows.to_columns()
        rows["frame"] = rows["frame"].astype(float)
        with self.assertRaises(ValueError):
            exp.transforms.convert_coordinates(rows)


class QueryIsolationTests(unittest.TestCase):
    def test_partition_filter_runs_before_coordinate_transforms(self):
        exp, ids = fixture()
        exp.tracks.add(ids)
        exp.add_acquisition(Acquisition(ArraySource(np.ones((5, 11, 11))), id="b"))
        other = exp.localizations.add(acquisition_id="b", channel="image", frame=[0, 4], x=2.0, y=3.0)
        exp.tracks.add(other)
        exp.transforms.set_drift("b", [[0.0, 0.0]], frames=[0])
        rows = exp.tracks.query(acquisition_id="a", space=CoordinateSpace.CORRECTED)
        np.testing.assert_equal(rows["id"], ids)


class TransactionReviewTests(unittest.TestCase):
    def test_batch_applies_detached_results_and_metadata(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import measure_localizations
        from molcore.processing import apply_batch
        from molcore.tracking import track

        exp, ids = fixture()
        photometry = compute_scenario(measure_localizations, exp, ids, cpu=InlineCpu(), apply=False)
        linking = compute_scenario(track, exp, "a", "image", cpu=InlineCpu(), apply=False)
        result = apply_batch(
            exp,
            [
                lambda e: e.apply(photometry),
                lambda e: e.apply(linking),
                lambda e: e.update_metadata({"test": "persist"}),
            ],
            expected=state_token(exp),
        )
        self.assertEqual(len(result[0].observation_ids), 3)
        self.assertGreater(len(result[1].track_ids), 0)
        self.assertEqual(exp.metadata["test"], "persist")

    def test_batch_cannot_apply_result_after_input_edit(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import measure_localizations
        from molcore.processing import apply_batch

        exp, ids = fixture()
        preview = compute_scenario(measure_localizations, exp, ids, cpu=InlineCpu(), apply=False)
        before = state_token(exp)
        with self.assertRaises(StaleResultError):
            apply_batch(
                exp, [lambda e: e.localizations.move(ids[0], x=8.0, y=8.0), lambda e: e.apply(preview)]
            )
        self.assertEqual(state_token(exp), before)

    def test_concurrent_edit_during_staged_operation_is_not_lost(self):
        from molcore.processing import apply_batch

        exp, ids = fixture()

        def operation(staged):
            staged.localizations.move(ids[1], x=9.0, y=9.0)
            exp.localizations.move(ids[0], x=8.0, y=8.0)

        with self.assertRaises(StaleResultError):
            apply_batch(exp, [operation])
        np.testing.assert_allclose(exp.localizations.query()["x"], [8.0, 4.0, 5.0])

    def test_removal_cleans_mask_provenance(self):
        from molcore import Segmentation

        exp, _ = fixture()
        segmentation = Segmentation(
            id="cells",
            acquisition_id="a",
            labels=np.ones((11, 11), dtype=int),
            metadata={"algorithm": "cellpose", "cellpose_version": "4.2", "torch_version": "2.8"},
        )
        exp.apply_all({"cells": segmentation}.values())
        exp.remove_acquisition("a", cascade=True)
        self.assertNotIn("cells", exp.processing.settings["cellpose"])

    def test_track_id_validation_with_existing_track(self):
        exp, ids = fixture()
        exp.tracks.add(ids)
        for value in (True, 1.5):
            with self.assertRaises(ValueError):
                exp.tracks.query(track_id=value)


class MappingAndBoundaryTests(unittest.TestCase):
    def test_affine_cross_channel_round_trip_and_alex_frame_rejection(self):
        from molcore import Calibration, ChannelSpec

        exp = Dataset()
        exp.add_acquisition(
            Acquisition(
                ArraySource(np.ones((4, 11, 11))),
                id="a",
                channel_specs={"d": ChannelSpec(), "a": ChannelSpec()},
            )
        )
        exp.transforms.add_calibration(
            Calibration("a_to_ref", [[2.0, 0.2, 1.0], [0.1, 1.5, -2.0], [0.0, 0.0, 1.0]])
        )
        exp.transforms.select("a", "a", "a_to_ref")
        exp.transforms.set_drift("a", [[1.0, 2.0]] * 4)
        ids = exp.localizations.add(acquisition_id="a", channel="d", frame=[0, 2], x=[2.0, 5.0], y=[3.0, 7.0])
        rows = exp.localizations.query(ids=ids)
        x, y = exp.transforms.convert_coordinates(
            rows, target_channel="a", target_space=CoordinateSpace.CORRECTED
        )
        target = dict(rows, channel=np.array(["a", "a"]), x=x, y=y)
        x, y = exp.transforms.convert_coordinates(
            target, source_space=CoordinateSpace.CORRECTED, target_channel="d"
        )
        np.testing.assert_allclose(x, rows["x"])
        np.testing.assert_allclose(y, rows["y"])
        alternating = Acquisition(
            ArraySource(np.ones((4, 11, 11))),
            id="alternating",
            channel_specs={"d": ChannelSpec(excitation="d"), "a": ChannelSpec(excitation="a")},
            excitation_sequence=("d", "a"),
        )
        view = alternating.channels["a"]
        np.testing.assert_equal(view.camera_frames([1, 0]), [3, 1])
        np.testing.assert_equal(view.channel_indices([3, 1]), [1, 0])
        with self.assertRaises(ValueError):
            view.channel_indices([0])

    def test_statistics_never_bridge_acquisitions(self):
        from molcore.trajectory import trajectory_statistics

        exp, ids = fixture()
        exp.add_acquisition(Acquisition(ArraySource(np.ones((5, 11, 11))), id="b"))
        other = exp.localizations.add(
            acquisition_id="b", channel="image", frame=[0, 2], x=[50.0, 51.0], y=0.0
        )
        exp.add_sequence(["a", "b"], id="s", reference="r", allow_cross_boundary_tracks=True)
        tid = exp.tracks.add(np.r_[ids, other], sequence_id="s")
        result = trajectory_statistics(exp.snapshot(), tid, frame_interval=0.5)
        self.assertEqual(len(result.steps["distance"]), 3)
        self.assertAlmostEqual(result.path_length, 3.0)
        self.assertAlmostEqual(result.duration, 3.0)
        self.assertEqual(sum(result.msd["count"]), 4)

    def test_clone_tokens_preserve_mask_and_timing_state(self):
        from molcore import Segmentation
        from molcore.staging import clone_dataset

        exp, _ = fixture()
        exp.add_segmentation(Segmentation(id="mask", acquisition_id="a", labels=np.ones((11, 11), dtype=int)))
        self.assertEqual(state_token(exp), state_token(clone_dataset(exp)))
