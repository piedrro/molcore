"""Cellpose adapter tests use controlled inference, never pretrained downloads."""

import inspect
import tempfile
import threading
import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from unittest.mock import patch

import numpy as np

from molcore import (
    Acquisition,
    ArraySource,
    ChannelSpec,
    Dataset,
    Segmentation,
    StaleResultError,
    cellpose,
)
from molcore.types import CellposeSettings, ProgressUpdate, SegmentationRequest


class InlineCpu:
    """Test utility executing the real adapter with a controlled model boundary."""

    num_cpus = 1

    def map(self, function, jobs, *, total=None, progress=None, cancel_event=None):
        jobs = list(jobs)
        results = []

        class Reporter:
            def check_cancelled(self):
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError()

            def report(self, completed, count, description=""):
                self.check_cancelled()
                if progress:
                    progress(
                        ProgressUpdate(
                            len(results) + (completed / count if count else 1), len(jobs), description
                        )
                    )

        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        for payload in jobs:
            results.append(function(payload, Reporter()))
        return results


class ControlledModel:
    calls = []
    evaluations = []
    bad_mask = False
    fail_on_image = None

    def __init__(self, **kwargs):
        self.calls.append(kwargs)
        self.device = kwargs["device"]
        self.pretrained_model = kwargs["pretrained_model"]

    def eval(self, image, **kwargs):
        self.evaluations.append((image.copy(), kwargs))
        if self.fail_on_image is not None and image[0, 0] == self.fail_on_image:
            raise RuntimeError("inference failed")
        if kwargs.get("progress"):
            kwargs["progress"].setValue(50)
        mask = np.zeros(image.shape, dtype=np.int32)
        mask[1:3, 1:3] = 7
        image[:] = 0  # Backend preprocessing must not overwrite source provenance.
        return (mask[:-1] if self.bad_mask else mask), [], np.zeros(256)


class CellposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from cellpose import models

        cls.torch = torch
        cls.models = models
        cls.constructor = models.CellposeModel
        cls.original_threads = torch.get_num_threads()

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.original_threads)

    def setUp(self):
        cellpose._MODEL_CACHE.clear()
        ControlledModel.calls, ControlledModel.evaluations = [], []
        ControlledModel.bad_mask, ControlledModel.fail_on_image = False, None
        self.exp = Dataset()
        self.image = np.arange(2 * 5 * 5, dtype=np.uint16).reshape(2, 5, 5)
        self.exp.add_acquisition(Acquisition(ArraySource(self.image), id="a"))
        self.old = Segmentation("cells", "a", np.ones((5, 5), dtype=np.uint16))
        self.exp.add_segmentation(self.old)
        self.factory = patch.object(self.models, "CellposeModel", ControlledModel)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.addCleanup(cellpose._MODEL_CACHE.clear)

    def test_segment_preserves_image_and_backend_options_and_metadata(self):
        old_picasso = self.exp.processing.software.get("picassosr")
        updates = []
        settings = CellposeSettings(
            diameter=25, normalize=False, flow_threshold=0.3, cellprob_threshold=-1, min_size=2, batch_size=4
        )
        result = cellpose.segment(
            self.exp,
            "a",
            "image",
            frame=1,
            segmentation_id="cells",
            settings=settings,
            cpu=InlineCpu(),
            progress=updates.append,
        )
        self.assertIs(self.exp.segmentations["cells"], result)
        self.assertEqual(result.frame, 1)
        self.assertEqual(result.labels[1, 1], 7)
        np.testing.assert_array_equal(result.source_image, self.image[1])
        self.assertEqual(result.source_image.dtype, np.dtype("uint16"))
        self.assertEqual(self.exp.processing.software.get("picassosr"), old_picasso)
        self.assertIn("cellpose", self.exp.processing.software)
        self.assertEqual(updates[-1].percent, 100)
        self.assertTrue(any(0 < update.percent < 100 for update in updates))
        kwargs = ControlledModel.calls[0]
        inspect.signature(self.constructor).bind_partial(**kwargs)
        inspect.signature(self.constructor.eval).bind(
            None, ControlledModel.evaluations[0][0], **ControlledModel.evaluations[0][1]
        )
        self.assertEqual(kwargs["pretrained_model"], "cpsam_v2")
        self.assertFalse(kwargs["gpu"])
        self.assertNotIn("model_type", kwargs)
        evaluation = ControlledModel.evaluations[0][1]
        self.assertEqual(evaluation["diameter"], 25)
        self.assertEqual(evaluation["cellprob_threshold"], -1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segmentation.h5"
            self.exp.save(path)
            with Dataset.open(path) as restored:
                np.testing.assert_array_equal(restored.segmentations["cells"].source_image, self.image[1])
                self.assertEqual(
                    restored.segmentations["cells"].metadata["settings"]["pretrained_model"], "cpsam_v2"
                )

    def test_preview_and_model_cache_and_camera_frame_mapping(self):
        self.exp.add_acquisition(
            Acquisition(
                ArraySource(np.zeros((5, 5, 5))),
                id="alex",
                excitation_sequence=("D", "A"),
                channel_specs={"AA": ChannelSpec(excitation="A")},
            )
        )
        preview = cellpose.segment(
            self.exp, "alex", "AA", segmentation_id="preview", apply=False, cpu=InlineCpu()
        )
        self.assertEqual(preview.frame, 1)
        self.assertNotIn("preview", self.exp.segmentations)
        cellpose.segment(self.exp, "a", "image", apply=False, cpu=InlineCpu())
        self.assertEqual(len(ControlledModel.calls), 1)

    def test_worker_failure_and_invalid_masks_do_not_apply_any_batch_result(self):
        self.exp.add_acquisition(Acquisition(ArraySource(np.full((1, 5, 5), 99)), id="b"))
        ControlledModel.fail_on_image = 99
        requests = [
            SegmentationRequest("a", "image", segmentation_id="cells"),
            SegmentationRequest("b", "image", segmentation_id="other"),
        ]
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            cellpose.segment_many(self.exp, requests, cpu=InlineCpu())
        self.assertIs(self.exp.segmentations["cells"], self.old)
        self.assertNotIn("other", self.exp.segmentations)
        ControlledModel.fail_on_image, ControlledModel.bad_mask = None, True
        with self.assertRaises(ValueError):
            cellpose.segment(self.exp, "a", "image", segmentation_id="cells", cpu=InlineCpu())
        self.assertIs(self.exp.segmentations["cells"], self.old)

    def test_cancellation_and_stale_target_or_source_preserve_user_edits(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(CancelledError):
            cellpose.segment(
                self.exp, "a", "image", segmentation_id="cells", cpu=InlineCpu(), cancel_event=stop
            )
        edited = []
        replacement = Segmentation("cells", "a", np.full((5, 5), 2, dtype=np.uint16))

        def progress(update):
            if not edited:
                self.exp.add_segmentation(replacement)
                edited.append(True)

        with self.assertRaises(StaleResultError):
            cellpose.segment(
                self.exp, "a", "image", segmentation_id="cells", cpu=InlineCpu(), progress=progress
            )
        self.assertIs(self.exp.segmentations["cells"], replacement)
        edited.clear()

        def change_image(update):
            if not edited:
                self.image[0, 0, 0] = 111
                edited.append(True)

        with self.assertRaises(StaleResultError):
            cellpose.segment(
                self.exp, "a", "image", segmentation_id="cells", cpu=InlineCpu(), progress=change_image
            )
        self.assertIs(self.exp.segmentations["cells"], replacement)

    def test_invalid_settings_model_names_and_duplicate_ids_are_rejected(self):
        for kwargs in (
            {"batch_size": 0},
            {"diameter": -1},
            {"min_size": -1},
            {"flow_threshold": float("nan")},
        ):
            with self.assertRaises(ValueError):
                CellposeSettings(**kwargs)
        with self.assertRaises(ValueError):
            cellpose.segment(
                self.exp, "a", "image", settings=CellposeSettings(pretrained_model="cyto3"), cpu=InlineCpu()
            )
        with self.assertRaises(ValueError):
            cellpose.segment_many(
                self.exp, [SegmentationRequest("a", "image", segmentation_id="same")] * 2, cpu=InlineCpu()
            )
        self.assertEqual(ControlledModel.calls, [])

    def test_gpu_pool_limit_and_unavailable_gpu_are_explicit(self):
        cpu = InlineCpu()
        cpu.num_cpus = 2
        with self.assertRaises(ValueError):
            cellpose.segment(self.exp, "a", "image", settings=CellposeSettings(gpu=True), cpu=cpu)
        with (
            patch.object(self.torch.cuda, "is_available", return_value=False),
            patch.object(self.torch.backends.mps, "is_available", return_value=False),
        ):
            with self.assertRaises(RuntimeError):
                cellpose.segment(self.exp, "a", "image", settings=CellposeSettings(gpu=True), cpu=InlineCpu())

    def test_real_spawned_backend_rejects_invalid_local_weights_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.weights"
            path.write_bytes(b"not a valid Cellpose checkpoint")
            with self.assertRaises(ValueError):
                cellpose.segment(
                    self.exp,
                    "a",
                    "image",
                    segmentation_id="cells",
                    settings=CellposeSettings(pretrained_model=str(path)),
                    num_cpus=1,
                )
        self.assertIs(self.exp.segmentations["cells"], self.old)

    def test_real_cellpose_eval_preprocessing_and_mask_construction_without_weights(self):
        model = object.__new__(self.constructor)
        model.device = self.torch.device("cpu")
        model.gpu, model.backbone, model.pretrained_model = False, "sam_vitl", "cpsam_v2"

        def flows(image, **kwargs):
            shape = image.shape[:3]
            return (
                np.zeros((2, *shape), dtype=np.float32),
                np.full(shape, -1, dtype=np.float32),
                np.zeros((shape[0], 256), dtype=np.float32),
            )

        with (
            patch.object(self.models, "CellposeModel", return_value=model),
            patch.object(model, "_run_net", side_effect=flows),
        ):
            result = cellpose.segment(self.exp, "a", "image", frame=1, apply=False, cpu=InlineCpu())
        self.assertEqual(result.labels.shape, (5, 5))
        self.assertEqual(int(result.labels.max()), 0)
        np.testing.assert_array_equal(result.source_image, self.image[1])

    def test_dino_requires_its_upstream_backend_without_silent_fallback(self):
        with patch("molcore.cellpose.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, "dinov3"):
                cellpose.segment(
                    self.exp,
                    "a",
                    "image",
                    segmentation_id="cells",
                    settings=CellposeSettings(pretrained_model="cpdino"),
                    cpu=InlineCpu(),
                )
        self.assertEqual(ControlledModel.calls, [])
        self.assertIs(self.exp.segmentations["cells"], self.old)
