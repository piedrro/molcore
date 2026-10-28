"""Real processor boundaries with alternative numerical workers."""

import threading
import unittest
from concurrent.futures import CancelledError
from unittest.mock import patch

import numpy as np

from molcore import Acquisition, ArraySource, Dataset, Segmentation, cellpose, localisation, tracking
from molcore.backends import BackendRegistry
from molcore.execution import ExecutionContext
from molcore.types import CameraSettings, LocalisationSettings, TrackingSettings


class InlineExecutor:
    def map(self, function, jobs, *, total=None, progress=None, cancel_event=None):
        class Reporter:
            def check_cancelled(self):
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError()

            def report(self, *args):
                self.check_cancelled()

        reporter = Reporter()
        reporter.check_cancelled()
        result = [function(job, reporter) for job in jobs]
        if progress:
            progress(None)
        reporter.check_cancelled()
        return result


class LocalBackend:
    version = "local-custom-1"

    def compute(self, payload, reporter):
        images, frames, settings, camera, candidates, fitting = payload
        reporter.check_cancelled()
        columns = {
            "frame": frames,
            "x": np.full(len(frames), 4.0),
            "y": np.full(len(frames), 4.0),
            "net_gradient": np.ones(len(frames)),
        }
        if fitting:
            columns.update(
                {
                    name: np.ones(len(frames))
                    for name in (
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
            )
        return columns, 0


class TrackBackend:
    version = "track-custom-1"

    def compute(self, payload, reporter):
        rows, labels, settings = payload
        reporter.check_cancelled()
        return (rows["id"].copy(),)


class SegmentBackend:
    version = "segment-custom-1"

    def compute(self, payload, reporter):
        request, frame, identifier, image, path, settings, model_ref, model_stat = payload
        reporter.check_cancelled()
        return Segmentation(
            identifier,
            request.acquisition_id,
            np.ones(image.shape, dtype=np.int32),
            image.copy(),
            channel=request.channel,
            frame=frame,
            space=request.space,
        )


class ProcessorBackendTests(unittest.TestCase):
    def setUp(self):
        self.model = Dataset()
        self.model.add_acquisition(Acquisition(ArraySource(np.ones((3, 9, 9))), id="a"))
        self.execution = ExecutionContext(cpu=InlineExecutor())
        self.camera = CameraSettings(0, 1, 1)

    def test_localisation_custom_backend_captures_provenance_and_applies(self):
        software = self.model.processing.software.copy()
        registry = BackendRegistry()
        registry.register("localise", "custom", LocalBackend())
        with patch.object(localisation, "backend_version", side_effect=AssertionError("builtin dependency")):
            result = localisation.localise(
                self.model,
                "a",
                "image",
                camera=self.camera,
                context=self.execution,
                backend="custom",
                registry=registry,
                apply=False,
            )
        self.assertEqual(result.backend_version, LocalBackend.version)
        self.assertIs(result.camera, self.camera)
        self.assertIsNotNone(result.expected)
        self.assertEqual(len(self.model.localizations), 0)
        from molcore.processing import apply_localisation

        applied = apply_localisation(self.model, result)
        self.assertEqual(len(applied.localization_ids), 3)
        self.assertEqual(self.model.processing.software, software)

    def test_detection_can_fit_after_model_clone(self):
        detected = localisation.detect(
            self.model,
            "a",
            "image",
            settings=LocalisationSettings(box_size=3),
            context=self.execution,
            backend=LocalBackend(),
        )
        from molcore.staging import clone_dataset

        cloned = clone_dataset(self.model)
        result = localisation.fit(
            cloned, detected, camera=self.camera, context=self.execution, backend=LocalBackend(), apply=False
        )
        self.assertEqual(result.columns["frame"].tolist(), [0, 1, 2])

    def test_tracking_custom_backend_works_without_builtin_version(self):
        software = self.model.processing.software.copy()
        self.model.localizations.add(acquisition_id="a", channel="image", frame=[0, 1, 2], x=4, y=4)
        with patch.object(tracking, "version", side_effect=AssertionError("builtin dependency")):
            result = tracking.track(
                self.model,
                "a",
                "image",
                settings=TrackingSettings(min_length=1),
                context=self.execution,
                backend=TrackBackend(),
            )
        self.assertEqual(result.backend_version, TrackBackend.version)
        self.assertEqual(len(result.track_ids), 1)
        self.assertEqual(self.model.processing.software, software)

    def test_custom_segmentation_bypasses_model_resolution(self):
        with (
            patch.object(cellpose, "_backend_version", side_effect=AssertionError("builtin dependency")),
            patch.object(cellpose, "_resolve_model", side_effect=AssertionError("builtin model")),
        ):
            result = cellpose.segment(
                self.model,
                "a",
                "image",
                context=self.execution,
                backend=SegmentBackend(),
                segmentation_id="custom",
            )
        self.assertEqual(result.id, "custom")
        self.assertIsNotNone(result.expected)
        np.testing.assert_array_equal(self.model.segmentations["custom"].labels, result.labels)

    def test_cancelled_context_stops_before_backend_resolution(self):
        event = threading.Event()
        event.set()
        with patch.object(localisation, "backend_version", side_effect=AssertionError("builtin dependency")):
            with self.assertRaises(CancelledError):
                localisation.localise(
                    self.model, "a", "image", camera=self.camera, context=ExecutionContext(cancel_event=event)
                )

    def test_context_conflict_rejected(self):
        with self.assertRaises(ValueError):
            tracking.track(
                self.model, "a", "image", context=self.execution, num_cpus=2, backend=TrackBackend()
            )
