"""Ownership and application contracts for the redesigned API."""

import unittest

import numpy as np

from molcore import (
    Acquisition,
    ArraySource,
    CameraSettings,
    Dataset,
    LocalisationResult,
    LocalisationSettings,
)


def dataset():
    model = Dataset(id="test", metadata={"nested": {"values": [1]}})
    model.add_acquisition(Acquisition(ArraySource(np.zeros((3, 8, 8))), id="a"))
    model.localizations.add(acquisition_id="a", channel="image", frame=[0, 1], x=[2, 3], y=[4, 5])
    return model


class OwnershipTests(unittest.TestCase):
    def test_track_metadata_updates_detach_and_preserve_memberships(self):
        model = dataset()
        track = model.tracks.add([1, 2], metadata={"particle": 1})
        captured = model.snapshot()
        previous_metadata = model.tracks.metadata[track]
        values = {"particle": 3, "nested": [1]}
        model.tracks.update_metadata(track, values)
        values["nested"].append(2)
        self.assertEqual(model.tracks.metadata[track]["parameters"]["nested"], (1,))
        self.assertEqual(previous_metadata["parameters"]["particle"], 1)
        np.testing.assert_array_equal(captured.tracks.query()["id"], [1, 2])
        np.testing.assert_array_equal(model.tracks.query()["id"], [1, 2])
        before = model.revision
        with self.assertRaises(TypeError):
            model.tracks.update_metadata(track, {"invalid": object()})
        self.assertEqual(model.revision, before)

    def test_source_rebinding_validates_geometry_and_preserves_captured_pixels(self):
        model = dataset()
        captured = model.snapshot()
        source = ArraySource(np.ones((3, 8, 8)))
        model.update_acquisition("a", source=source)
        self.assertIs(model.acquisitions["a"].source, source)
        self.assertEqual(captured.acquisitions["a"].source[0].sum(), 0)
        before = model.revision
        with self.assertRaises(ValueError):
            model.update_acquisition("a", source=ArraySource(np.ones((3, 9, 8))))
        self.assertEqual(model.revision, before)

    def test_bulk_insertion_preserves_large_mixed_integer_columns(self):
        model = dataset()

        def chunk(value):
            return dict(acquisition_id="a", channel="image", frame=[0], x=[1.0], y=[2.0], quality=value)

        model.localizations.add_batches(
            [
                chunk(np.array([2**63 + 1], dtype=np.uint64)),
                chunk(np.array([1], dtype=np.int64)),
            ]
        )
        values = model.localizations.query(columns=("quality",))["quality"]
        self.assertEqual(values.dtype, np.dtype("uint64"))
        self.assertEqual(int(values[-2]), 2**63 + 1)
        before = len(model.localizations)
        with self.assertRaises(ValueError):
            model.localizations.add_batches(
                [
                    chunk(np.array([2**63 + 1], dtype=np.uint64)),
                    chunk(np.array([-1], dtype=np.int64)),
                ]
            )
        self.assertEqual(len(model.localizations), before)

    def test_track_metadata_does_not_borrow_mutable_caller_values(self):
        model = dataset()
        metadata = {"nested": {"values": [1]}}
        track = model.tracks.add([1, 2], metadata=metadata)
        metadata["nested"]["values"][0] = 99
        self.assertEqual(model.tracks.metadata[track]["parameters"]["nested"]["values"], (1,))

    def test_serialization_preserves_result_and_pixel_immutability(self):
        import pickle

        result = LocalisationResult(
            "a",
            "image",
            {"frame": np.array([0]), "x": np.array([1.0]), "y": np.array([2.0])},
            LocalisationSettings(),
        )
        restored = pickle.loads(pickle.dumps(result))
        self.assertEqual(restored.result_id, result.result_id)
        with self.assertRaises(ValueError):
            restored.columns["x"].setflags(write=True)
        source = pickle.loads(pickle.dumps(ArraySource(np.ones((2, 3, 3)))))
        with self.assertRaises(ValueError):
            source[0].setflags(write=True)

    def test_metadata_rejects_objects_that_cannot_be_frozen_and_persisted(self):
        with self.assertRaises(TypeError):
            Dataset(metadata={"mutable": object()})
        model = dataset()
        with self.assertRaises(TypeError):
            model.update_metadata({"mutable": object()})
        self.assertNotIn("mutable", model.metadata)
        with self.assertRaises(TypeError):
            model.tracks.add([1, 2], metadata={"mutable": object()})
        self.assertEqual(len(model.tracks), 0)

    def test_dataset_identity_is_stable_and_renaming_is_notified(self):
        model = dataset()
        snapshot = model.snapshot()
        with self.assertRaises(AttributeError):
            model.id = "different"
        changes = []
        model.subscribe(changes.append)
        model.rename("New name")
        self.assertEqual(model.name, "New name")
        self.assertEqual(snapshot.name, "test")
        self.assertEqual(len(changes), 1)

    def test_array_source_owns_immutable_pixels_by_default(self):
        pixels = np.zeros((2, 3, 4))
        source = ArraySource(pixels)
        pixels[:] = 9
        self.assertEqual(float(source[0].sum()), 0)
        with self.assertRaises(ValueError):
            source[0].setflags(write=True)

    def test_explicit_borrowing_retains_external_pixels(self):
        pixels = np.zeros((2, 3, 4))
        source = ArraySource.borrow(pixels)
        pixels[:] = 9
        self.assertEqual(float(source[0].sum()), 108)

    def test_snapshot_has_no_live_mutation_or_save_interface(self):
        model = dataset()
        snapshot = model.snapshot()
        for name in ("apply", "save", "add_acquisition", "update_metadata"):
            self.assertFalse(hasattr(snapshot, name), name)
        self.assertFalse(hasattr(snapshot.localizations, "move"))
        with self.assertRaises(TypeError):
            snapshot.metadata["nested"]["values"][0] = 2
        model.localizations.move(1, x=7, y=6)
        self.assertEqual(snapshot.localizations.query()["x"].tolist(), [2, 3])
        self.assertEqual(model.localizations.query()["x"].tolist(), [7, 3])

    def test_snapshot_shares_unchanged_columns(self):
        model = dataset()
        snapshot = model.snapshot()
        self.assertTrue(
            np.shares_memory(model.localizations._data["x"], snapshot._model.localizations._data["x"])
        )
        model.localizations.move(1, x=6, y=7)
        self.assertTrue(
            np.shares_memory(model.localizations._data["x"], snapshot._model.localizations._data["x"])
        )

    def test_result_recursively_detaches_and_freezes_payload(self):
        columns = {"frame": np.array([0]), "x": np.array([1.0]), "y": np.array([2.0])}
        result = LocalisationResult("a", "image", columns, LocalisationSettings())
        columns["x"][0] = 99
        self.assertEqual(result.columns["x"].tolist(), [1])
        with self.assertRaises(TypeError):
            result.columns["extra"] = np.array([1])
        with self.assertRaises(ValueError):
            result.columns["x"].setflags(write=True)

    def test_model_records_and_metadata_are_read_only(self):
        model = dataset()
        with self.assertRaises(TypeError):
            model.metadata["extra"] = 1
        with self.assertRaises(AttributeError):
            model.acquisitions["a"].pixel_size = 3
        model.update_metadata({"extra": 1})
        model.update_acquisition("a", pixel_size=100)
        self.assertEqual(model.metadata["extra"], 1)
        self.assertEqual(model.acquisitions["a"].pixel_size, 100)


class TransactionTests(unittest.TestCase):
    def test_transaction_detects_supported_edits_even_if_values_are_restored(self):
        from molcore import Segmentation, StaleResultError
        from molcore.staging import transact

        model = dataset()

        def commit(staged):
            model.add_segmentation(Segmentation("temporary", "a", np.ones((8, 8), dtype=np.int32)))
            model.remove_segmentation("temporary")
            staged.localizations.move(1, x=7, y=7)

        with self.assertRaises(StaleResultError):
            transact(model, commit)
        self.assertEqual(model.localizations.query()["x"].tolist(), [2, 3])

    def test_nested_application_returns_a_receipt_for_the_outer_commit(self):
        from molcore import ApplyReceipt
        from molcore.processing import apply_batch

        model = dataset()
        result = self.result(model)
        receipts = apply_batch(model, [lambda staged: staged.apply(result)])
        self.assertIsInstance(receipts[0], ApplyReceipt)
        self.assertEqual(receipts[0].revision, model.revision)

    def result(self, model):
        return LocalisationResult(
            "a",
            "image",
            {"frame": np.array([0]), "x": np.array([1.0]), "y": np.array([2.0])},
            LocalisationSettings(),
            camera=CameraSettings(0, 1, 1),
            expected=model.dependencies("a", "image"),
        )

    def test_receipt_is_separate_from_immutable_result_and_publishes_once(self):
        model = dataset()
        result = self.result(model)
        changes = []
        model.subscribe(changes.append)
        receipt = model.apply(result)
        self.assertEqual(receipt.result_id, result.result_id)
        self.assertEqual(receipt.revision, model.revision)
        self.assertEqual(receipt.localization_ids.tolist(), [3])
        self.assertIsNone(result.localization_ids)
        self.assertEqual(len(changes), 1)

    def test_batch_failure_preserves_state_and_notifications(self):
        model = dataset()
        result = self.result(model)
        changes = []
        model.subscribe(changes.append)
        with self.assertRaises(__import__("molcore").StaleResultError):
            model.apply_all([result, result])
        self.assertEqual(model.localizations.query()["id"].tolist(), [1, 2])
        self.assertEqual(changes, [])

    def test_apply_uses_one_staging_boundary(self):
        from unittest.mock import patch

        from molcore.staging import clone_dataset

        model = dataset()
        result = self.result(model)
        with patch("molcore.staging.clone_dataset", wraps=clone_dataset) as clone:
            model.apply(result)
            self.assertEqual(clone.call_count, 1)


class BatchTests(unittest.TestCase):
    def test_track_query_can_filter_a_partition_on_a_sequence_clock(self):
        model = dataset()
        model.add_acquisition(Acquisition(ArraySource(np.zeros((3, 8, 8))), id="b"))
        other = model.localizations.add(acquisition_id="b", channel="image", frame=[0, 1], x=[2, 3], y=[4, 5])
        model.add_sequence(["a", "b"], id="s", allow_cross_boundary_tracks=True, reference="r")
        model.tracks.add(np.array([1, 2, *other], dtype=np.uint64), sequence_id="s")
        batch = model.tracks.query(sequence_id="s", acquisition_id="b", columns=("id", "sequence_frame"))
        self.assertEqual(batch["id"].tolist(), [3, 4])
        self.assertEqual(batch["sequence_frame"].tolist(), [3, 4])

    def test_bulk_insertion_preserves_schema_and_validity(self):
        model = dataset()
        model.localizations.add(acquisition_id="a", channel="image", frame=[2], x=[1], y=[1], count=[4])
        model.localizations.set_column_schema("count", unit="count")
        target = Dataset()
        target.add_acquisition(model.acquisitions["a"])
        batch = model.localizations.query(space="raw")
        target.localizations.add_batches([batch])
        actual = target.localizations.query(space="raw")
        self.assertEqual(actual.schema["count"].unit, "count")
        self.assertEqual(actual.validity["count"].tolist(), [False, False, True])

    def test_integer_measurement_updates_preserve_precision_and_nullability(self):
        model = dataset()
        model.localizations.update_measurements([1], count=np.array([2**63 + 3], dtype=np.uint64))
        actual = model.localizations.query(space="raw")
        self.assertEqual(actual["count"].dtype, np.dtype("uint64"))
        self.assertEqual(int(actual["count"][0]), 2**63 + 3)
        self.assertEqual(actual.validity["count"].tolist(), [True, False])

    def test_projection_and_iteration_preserve_order_and_context(self):
        model = dataset()
        batch = model.localizations.query(columns=("id", "x"), space="raw")
        self.assertEqual(set(batch), {"id", "x"})
        self.assertEqual(batch.n_rows, 2)
        self.assertEqual(batch.space, "raw")
        batches = list(model.localizations.iter_batches(columns=("id", "x"), batch_size=1))
        self.assertEqual([part["id"].tolist() for part in batches], [[1], [2]])
        self.assertEqual(model.localizations.query(columns=()).n_rows, 2)
        with self.assertRaises(ValueError):
            list(model.localizations.iter_batches(batch_size=0))

    def test_missing_integer_measurements_preserve_exact_values_and_validity(self):
        model = dataset()
        value = np.uint64(2**63 + 3)
        model.localizations.add(acquisition_id="a", channel="image", frame=[2], x=[1], y=[1], counter=[value])
        batch = model.localizations.query(space="raw", columns=("counter",))
        self.assertEqual(batch["counter"].dtype, np.dtype("uint64"))
        self.assertEqual(int(batch["counter"][2]), 2**63 + 3)
        self.assertEqual(batch.validity["counter"].tolist(), [False, False, True])

    def test_bulk_insertion_is_atomic_on_late_invalid_input(self):
        model = dataset()
        before = model.revision
        with self.assertRaises(ValueError):
            model.localizations.add_batches(
                [
                    {"acquisition_id": ["a"], "channel": ["image"], "frame": [0], "x": [1], "y": [2]},
                    {"acquisition_id": ["a"], "channel": ["image"], "frame": [999], "x": [1], "y": [2]},
                ]
            )
        self.assertEqual(model.revision, before)
        self.assertEqual(len(model.localizations), 2)

    def test_nullable_columns_and_units_round_trip(self):
        import tempfile
        from pathlib import Path

        model = dataset()
        model.localizations.add(
            acquisition_id="a",
            channel="image",
            frame=[2],
            x=[1],
            y=[1],
            counter=np.array([2**63 + 3], dtype=np.uint64),
        )
        model.localizations.set_column_schema("counter", unit="count")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.h5"
            model.save(path)
            with Dataset.open(path) as restored:
                batch = restored.localizations.query(columns=("counter",))
                self.assertEqual(batch.schema["counter"].unit, "count")
                self.assertEqual(int(batch["counter"][2]), 2**63 + 3)
                self.assertEqual(batch.validity["counter"].tolist(), [False, False, True])

    def test_missing_measurements_do_not_match_numeric_filters(self):
        from molcore import PartitionSelection
        from molcore.filtering import filter_results
        from molcore.types.analysis import FilterSettings

        model = dataset()
        model.localizations.add(acquisition_id="a", channel="image", frame=[2], x=[1], y=[1], count=[0])
        result = filter_results(
            model.snapshot(), PartitionSelection("a", "image"), settings=FilterSettings("count", 0, 0)
        )
        self.assertEqual(result.selected_ids.tolist(), [3])


class ProcessingBoundaryTests(unittest.TestCase):
    def test_tracking_is_detached_with_typed_request_and_explicit_selection(self):
        from molcore import ExecutionContext, PartitionSelection
        from molcore.backends import TrackingOutput, TrackingRequest
        from molcore.execution import SerialExecutor
        from molcore.tracking import track

        class Backend:
            version = "test-1"

            def compute(self, request, reporter):
                if not isinstance(request, TrackingRequest):
                    raise TypeError("Expected named tracking inputs")
                return TrackingOutput((request.rows["id"],))

        model = dataset()
        result = track(
            model.snapshot(),
            PartitionSelection("a", "image"),
            backend=Backend(),
            context=ExecutionContext(cpu=SerialExecutor(), chunk_size=7),
        )
        self.assertEqual(result.execution["chunk_size"], 7)
        self.assertEqual(result.execution["executor"], "SerialExecutor")
        self.assertEqual(len(model.tracks), 0)
        receipt = model.apply(result)
        self.assertEqual(receipt.track_ids.tolist(), [1])
        self.assertEqual(len(model.tracks), 1)
        with self.assertRaises(TypeError):
            track(model.snapshot(), PartitionSelection("a", "image"), apply=True)


class ImageReaderTests(unittest.TestCase):
    def test_cached_pixels_and_source_binding_are_read_only(self):
        from molcore import CachedSource

        source = CachedSource(ArraySource(np.ones((2, 3, 3))))
        with self.assertRaises(ValueError):
            source[0].setflags(write=True)
        with self.assertRaises(AttributeError):
            source.source = ArraySource(np.zeros((2, 3, 3)))

    def test_bulk_reads_preserve_duplicates_roi_and_output_buffer(self):
        source = ArraySource(np.arange(3 * 4 * 5).reshape(3, 4, 5))
        output = np.empty((3, 2, 3), dtype=source.dtype)
        with source.open_reader() as reader:
            actual = reader.read_frames([2, 0, 2], roi=(1, 3, 1, 4), out=output)
            self.assertIs(actual, output)
            self.assertEqual(actual[:, 0, 0].tolist(), [46, 6, 46])
            self.assertEqual(reader.read_frames([], roi=(1, 3, 1, 4)).shape, (0, 2, 3))
        with self.assertRaises(RuntimeError):
            reader.read_frames([0])

    def test_channel_reader_pushes_frame_and_spatial_selection_to_source(self):
        from molcore import ChannelSpec

        model = Dataset()
        pixels = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        model.add_acquisition(
            Acquisition(
                ArraySource(pixels),
                id="a",
                channel_specs={"image": ChannelSpec(roi=(1, 5, 2, 6), frame_step=2)},
            )
        )
        with model.acquisitions["a"].channels["image"].open_reader() as reader:
            actual = reader.read_frames([1, 0, 1], roi=(1, 3, 1, 3))
        self.assertEqual(actual[:, 0, 0].tolist(), [75, 15, 75])

    def test_unsigned_out_of_range_frames_cannot_wrap_to_last_frame(self):
        source = ArraySource(np.ones((2, 3, 3)))
        with source.open_reader() as reader, self.assertRaises(IndexError):
            reader.read_frames(np.array([2**64 - 1], dtype=np.uint64))


class ExtensionValidationTests(unittest.TestCase):
    def test_independent_segmentation_targets_apply_together(self):
        from molcore import ExecutionContext, Segmentation, SegmentationRequest
        from molcore.cellpose import segment_many
        from molcore.execution import SerialExecutor

        class Backend:
            version = "test"

            def compute(self, request, reporter):
                return Segmentation(
                    request.segmentation_id,
                    request.request.acquisition_id,
                    np.ones(request.image.shape, dtype=np.int32),
                    request.image,
                    channel=request.request.channel,
                    frame=request.frame,
                )

        model = dataset()
        results = segment_many(
            model.snapshot(),
            [
                SegmentationRequest("a", "image", frame=0, segmentation_id="m0"),
                SegmentationRequest("a", "image", frame=1, segmentation_id="m1"),
            ],
            backend=Backend(),
            context=ExecutionContext(cpu=SerialExecutor()),
        )
        receipts = model.apply_all(results.values())
        self.assertEqual(len(receipts), 2)
        self.assertEqual(set(model.segmentations), {"m0", "m1"})

    def test_segmentation_does_not_overwrite_a_target_created_after_capture(self):
        from molcore import ExecutionContext, PartitionSelection, Segmentation, StaleResultError
        from molcore.cellpose import segment
        from molcore.execution import SerialExecutor

        class Backend:
            version = "test"

            def compute(self, request, reporter):
                return Segmentation(
                    request.segmentation_id,
                    request.request.acquisition_id,
                    np.ones(request.image.shape, dtype=np.int32),
                    request.image,
                    channel=request.request.channel,
                    frame=request.frame,
                )

        model = dataset()
        model.add_acquisition(Acquisition(ArraySource(np.zeros((3, 8, 8))), id="b"))
        result = segment(
            model.snapshot(),
            PartitionSelection("a", "image"),
            segmentation_id="mask",
            backend=Backend(),
            context=ExecutionContext(cpu=SerialExecutor()),
        )
        model.add_segmentation(Segmentation("mask", "b", np.ones((8, 8), dtype=np.int32)))
        with self.assertRaises(StaleResultError):
            model.apply(result)

    def test_segmentation_preview_protects_an_existing_replacement_target(self):
        from molcore import ExecutionContext, PartitionSelection, Segmentation, StaleResultError
        from molcore.cellpose import segment
        from molcore.execution import SerialExecutor

        class Backend:
            version = "test"

            def compute(self, request, reporter):
                return Segmentation(
                    request.segmentation_id,
                    request.request.acquisition_id,
                    np.ones(request.image.shape, dtype=np.int32),
                    request.image,
                    channel=request.request.channel,
                    frame=request.frame,
                )

        model = dataset()
        model.add_acquisition(Acquisition(ArraySource(np.zeros((3, 8, 8))), id="b"))
        model.add_segmentation(Segmentation("mask", "b", np.zeros((8, 8), dtype=np.int32)))
        result = segment(
            model.snapshot(),
            PartitionSelection("a", "image"),
            segmentation_id="mask",
            backend=Backend(),
            context=ExecutionContext(cpu=SerialExecutor()),
        )
        model.add_segmentation(Segmentation("mask", "b", np.ones((8, 8), dtype=np.int32)))
        with self.assertRaises(StaleResultError):
            model.apply(result)

    def test_rebinding_a_segmentation_invalidates_both_partitions(self):
        from molcore import Segmentation, StaleResultError

        model = dataset()
        model.add_acquisition(Acquisition(ArraySource(np.zeros((3, 8, 8))), id="b"))
        model.add_segmentation(Segmentation("mask", "b", np.zeros((8, 8), dtype=np.int32)))
        token = model.dependencies("b", "image", components={"segmentations"})
        model.add_segmentation(Segmentation("mask", "a", np.ones((8, 8), dtype=np.int32)))
        with self.assertRaises(StaleResultError):
            token.validate(model)

    def test_working_source_localisation_requires_explicit_coordinate_conversion(self):
        from molcore import ExecutionContext, PartitionSelection
        from molcore.backends import LocalisationOutput
        from molcore.execution import SerialExecutor
        from molcore.localisation import localise

        class Backend:
            version = "test"

            def compute(self, request, reporter):
                columns = {
                    name: np.ones(len(request.frames))
                    for name in (
                        "x",
                        "y",
                        "net_gradient",
                        "photons",
                        "bg",
                        "sx",
                        "sy",
                        "lpx",
                        "lpy",
                        "ellipticity",
                        "fit_state",
                        "iterations",
                    )
                }
                columns["frame"] = request.frames
                return LocalisationOutput(columns, 0)

        model = dataset()
        result = localise(
            model.snapshot(),
            PartitionSelection("a", "image"),
            camera=CameraSettings(0, 1, 1),
            source=ArraySource(np.ones((3, 8, 8))),
            backend=Backend(),
            context=ExecutionContext(cpu=SerialExecutor()),
        )
        with self.assertRaises(ValueError):
            model.apply(result)

    def test_extension_handler_returns_allocations_through_the_same_boundary(self):
        from molcore import ResultIdentity
        from molcore.immutable import immutable_record
        from molcore.processing import register_result_handler

        @immutable_record
        class Extra(ResultIdentity):
            value: int

        def commit(result, staged):
            staged.update_metadata({"extra": result.value})
            return {"custom_ids": np.array([7], dtype=np.uint64)}

        register_result_handler(Extra, commit)
        model = dataset()
        result = Extra(3)
        receipt = model.apply(result)
        self.assertEqual(model.metadata["extra"], 3)
        self.assertEqual(receipt.identifiers["custom_ids"].tolist(), [7])
        self.assertEqual(receipt.result_id, result.result_id)

    def test_custom_array_source_can_maintain_reader_instrumentation(self):
        class CountingSource(ArraySource):
            def __init__(self, pixels):
                super().__init__(pixels)
                self.reads = []

            def _frame(self, index):
                self.reads.append(index)
                return super()._frame(index)

        source = CountingSource(np.ones((2, 3, 3)))
        with source.open_reader() as reader:
            reader.read_frames([1])
        self.assertEqual(source.reads, [1])

    def test_segmentation_backend_cannot_change_request_binding(self):
        from molcore import ExecutionContext, PartitionSelection, Segmentation
        from molcore.cellpose import segment
        from molcore.execution import SerialExecutor

        class Backend:
            version = "test"

            def compute(self, request, reporter):
                return Segmentation(
                    "different-id",
                    request.request.acquisition_id,
                    np.ones(request.image.shape, dtype=np.int32),
                    request.image,
                    channel=request.request.channel,
                    frame=request.frame,
                )

        model = dataset()
        with self.assertRaises(ValueError):
            segment(
                model.snapshot(),
                PartitionSelection("a", "image"),
                backend=Backend(),
                context=ExecutionContext(cpu=SerialExecutor()),
            )


if __name__ == "__main__":
    unittest.main()
