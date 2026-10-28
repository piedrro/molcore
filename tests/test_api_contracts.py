"""Downstream-facing mutation, snapshot and detached-result contracts."""

import unittest
from unittest.mock import patch

import numpy as np

from molcore import Acquisition, ArraySource, Calibration, Dataset, Experiment, Segmentation, StaleResultError
from molcore.staging import clone_dataset, state_token, transact
from molcore.types import CameraSettings, LocalisationResult, LocalisationSettings


def model():
    result = Dataset()
    for name in ("a", "b"):
        result.add_acquisition(Acquisition(ArraySource(np.zeros((2, 7, 7))), id=name))
        result.localizations.add(acquisition_id=name, channel="image", frame=[0], x=[2.0], y=[2.0])
    return result


class MutationContractTests(unittest.TestCase):
    def test_acquisition_updates_validate_and_invalidate_only_their_partition(self):
        dataset = model()
        related = dataset.dependencies("a", "image")
        unrelated = dataset.dependencies("b", "image")
        dataset.update_acquisition("a", pixel_size=100, timestamps=[0.0, 0.1], metadata={"camera": "one"})
        self.assertEqual(dataset.acquisitions["a"].pixel_size, 100)
        unrelated.validate(dataset)
        with self.assertRaises(StaleResultError):
            related.validate(dataset)
        before = state_token(dataset)
        with self.assertRaises(ValueError):
            dataset.update_acquisition("a", timestamps=[0.0])
        self.assertEqual(state_token(dataset), before)

    def test_experiment_publishes_only_after_all_datasets_are_committed(self):
        first, second = model(), model()
        experiment = Experiment()
        experiment.add_dataset(first)
        experiment.add_dataset(second)
        seen = []

        def on_first(change):
            seen.append(second.transforms._selected.get(("a", "image")))
            second.localizations.move(1, x=5, y=5)

        first.subscribe(on_first)
        experiment.apply_channel_calibration(Calibration("shared"), emission=None)
        self.assertEqual(seen, ["shared"])
        self.assertEqual(second.localizations.query(ids=[1], space="raw")["x"].tolist(), [5.0])

    def test_transaction_preserves_public_store_and_collection_references(self):
        dataset = model()
        table, acquisitions = dataset.localizations, dataset.acquisitions
        transact(dataset, lambda staged: staged.localizations.move(1, x=3, y=4))
        self.assertIs(dataset.localizations, table)
        self.assertIs(dataset.acquisitions, acquisitions)
        self.assertEqual(table.query(ids=[1], space="raw")["x"].tolist(), [3.0])

    def test_collections_are_read_only(self):
        dataset = model()
        with self.assertRaises(TypeError):
            dataset.acquisitions["bad"] = dataset.acquisitions["a"]

    def test_events_describe_mutation_and_unsubscribe(self):
        dataset = model()
        events = []
        unsubscribe = dataset.subscribe(events.append)
        dataset.localizations.move(1, x=3, y=4)
        self.assertEqual(len(events), 1)
        self.assertIn("localizations", events[0].components)
        self.assertIn(("a", "image"), events[0].partitions)
        unsubscribe()
        dataset.localizations.move(1, x=4, y=4)
        self.assertEqual(len(events), 1)

    def test_failed_transaction_has_no_event_or_mutation(self):
        dataset = model()
        events = []
        dataset.subscribe(events.append)

        def fail(staged):
            staged.localizations.move(1, x=3, y=4)
            raise ValueError("reject")

        with self.assertRaises(ValueError):
            transact(dataset, fail)
        self.assertEqual(events, [])
        self.assertEqual(dataset.localizations.query(ids=[1], space="raw")["x"].tolist(), [2.0])

    def test_subscriber_failure_does_not_turn_commit_into_failure(self):
        dataset = model()

        def broken(change):
            raise RuntimeError("view closed")

        dataset.subscribe(broken)
        dataset.localizations.move(1, x=3, y=4)
        self.assertEqual(dataset.localizations.query(ids=[1], space="raw")["x"].tolist(), [3.0])


class RevisionContractTests(unittest.TestCase):
    def test_mask_removal_does_not_invalidate_another_acquisition(self):
        dataset = model()
        dataset.add_segmentation(Segmentation("mask", "a", np.ones((7, 7), dtype=int)))
        expected = dataset.dependencies("b", "image", components={"segmentations"})
        dataset.remove_segmentation("mask")
        expected.validate(dataset)

    def test_photometry_rejects_change_between_row_read_and_token_capture(self):
        from test_processor_backends import InlineExecutor

        from molcore.photometry import measure_localizations

        dataset = model()
        query = dataset.localizations.query
        moved = False

        def concurrent_read(*args, **kwargs):
            nonlocal moved
            rows = query(*args, **kwargs)
            if not moved:
                moved = True
                dataset.localizations.move(1, x=3, y=3)
            return rows

        with patch.object(dataset.localizations, "query", side_effect=concurrent_read):
            with self.assertRaises(StaleResultError):
                measure_localizations(dataset, [1], cpu=InlineExecutor(), apply=False)

    def test_named_state_and_clone_lineage(self):
        dataset = model()
        token = state_token(dataset)
        self.assertTrue(hasattr(token, "scientific_inputs"))
        self.assertEqual(token, state_token(clone_dataset(dataset)))

    def test_scoped_dependencies_ignore_unrelated_partition(self):
        dataset = model()
        expected = dataset.dependencies("a", "image", components={"localizations"})
        dataset.localizations.move(2, x=3, y=4)
        expected.validate(dataset)
        dataset.localizations.move(1, x=3, y=4)
        with self.assertRaises(StaleResultError):
            expected.validate(dataset)

    def test_scoped_dependencies_reject_foreign_dataset(self):
        dataset = model()
        expected = dataset.dependencies("a", "image", components={"localizations"})
        with self.assertRaises(StaleResultError):
            expected.validate(model())

    def test_snapshot_retains_lineage_and_detaches_results(self):
        dataset = model()
        snapshot = dataset.snapshot()
        expected = snapshot.dependencies("a", "image", components={"localizations"})
        expected.validate(dataset)
        snapshot.localizations.move(1, x=3, y=4)
        self.assertEqual(dataset.localizations.query(ids=[1], space="raw")["x"].tolist(), [2.0])


class ResultContractTests(unittest.TestCase):
    def test_batch_emits_once_and_rolls_back_on_later_failure(self):
        dataset = model()
        events = []
        dataset.subscribe(events.append)
        first = LocalisationResult(
            "a",
            "image",
            {"frame": np.array([0]), "x": np.array([3.0]), "y": np.array([4.0])},
            LocalisationSettings(),
            camera=CameraSettings(0, 1, 1),
            expected=dataset.dependencies("a", "image", components={"localizations"}),
        )
        with self.assertRaises(TypeError):
            dataset.apply_all([first, object()])
        self.assertEqual(events, [])
        self.assertEqual(dataset.localizations.query(ids=[1], space="raw")["x"].tolist(), [2.0])
        dataset.apply_all([first])
        self.assertEqual(len(events), 1)

    def test_apply_localisation_uses_captured_provenance(self):
        dataset = model()
        result = LocalisationResult(
            "a",
            "image",
            {"frame": np.array([1]), "x": np.array([3.0]), "y": np.array([4.0])},
            LocalisationSettings(),
            camera=CameraSettings(0, 1, 1),
            backend_version="custom-1",
            expected=dataset.dependencies("a", "image", components={"localizations"}),
        )
        applied = dataset.apply(result)
        self.assertIsNotNone(applied.localization_ids)
        self.assertEqual(dataset.localizations.query(acquisition_id="a")["frame"].tolist(), [1])
        self.assertEqual(dataset.processing.settings["localisation"]["a"]["image"]["backend"], "custom-1")
        with self.assertRaises(StaleResultError):
            dataset.apply(result)

    def test_partial_preview_cannot_replace_partition(self):
        dataset = model()
        result = LocalisationResult(
            "a",
            "image",
            {"frame": np.array([0]), "x": np.array([3.0]), "y": np.array([4.0])},
            LocalisationSettings(),
            frame_range=(0, 1),
            camera=CameraSettings(0, 1, 1),
        )
        with self.assertRaisesRegex(ValueError, "preview"):
            dataset.apply(result)


if __name__ == "__main__":
    unittest.main()
