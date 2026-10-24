import importlib.util
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
    Calibration,
    ChannelSpec,
    Experiment,
    StaleResultError,
    alignment,
    localisation,
    undrifting,
)
from molcore.cpu import CpuExecutor
from molcore.types.processing import CameraSettings, DriftSettings, LocalisationSettings


def spot_experiment(alex=False):
    y, x = np.indices((24, 24))
    image = 20 + 3000 / (2 * np.pi * 1.2**2) * np.exp(-((x - 11.3) ** 2 + (y - 10.7) ** 2) / (2 * 1.2**2))
    exp = Experiment()
    exp.add_acquisition(
        Acquisition(
            ArraySource(np.repeat(image[None], 5, axis=0).astype(np.float32)),
            id="a",
            excitation_sequence=("D", "A") if alex else (),
            channel_specs={"DD": ChannelSpec(excitation="D")} if alex else None,
            pixel_size=100,
        )
    )
    return exp


class ProcessingCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = CpuExecutor(num_cpus=2)

    @classmethod
    def tearDownClass(cls):
        cls.cpu.close()


@unittest.skipUnless(importlib.util.find_spec("picasso"), "Picasso processing extra not installed")
class LocalisationTests(ProcessingCase):
    def test_nonconverged_fits_do_not_clear_active_partition(self):
        exp = spot_experiment()
        ids = exp.localizations.add(acquisition_id="a", channel="image", frame=0, x=1, y=1)
        with self.assertRaisesRegex(ValueError, "converge"):
            localisation.localise(
                exp,
                "a",
                "image",
                camera=CameraSettings(0, 1, 1),
                settings=LocalisationSettings(min_net_gradient=10, max_iterations=1, chunk_size=5),
                cpu=self.cpu,
            )
        self.assertEqual(exp.localizations.query()["id"].tolist(), ids.tolist())

    def test_detection_and_both_fits_keep_alex_camera_frames(self):
        exp = spot_experiment(alex=True)
        settings = LocalisationSettings(min_net_gradient=10, box_size=7, chunk_size=2)
        detections = localisation.detect(exp, "a", "DD", settings=settings, cpu=self.cpu)
        self.assertEqual(detections.columns["frame"].tolist(), [0, 2, 4])
        self.assertEqual(len(exp.localizations), 0)
        for method in ("lq", "mle"):
            fitted = localisation.fit(
                exp,
                detections,
                camera=CameraSettings(0, 1, 1),
                settings=LocalisationSettings(min_net_gradient=10, fit_method=method, chunk_size=2),
                cpu=self.cpu,
            )
            np.testing.assert_allclose(fitted.columns["x"], 11.3, atol=0.12)
            np.testing.assert_allclose(fitted.columns["y"], 10.7, atol=0.12)
            self.assertEqual(exp.localizations.query()["frame"].tolist(), [0, 2, 4])
            self.assertEqual(len(fitted.localization_ids), 3)
            self.assertEqual(fitted.failed_fits, 0)
        self.assertIn("localisation", exp.processing.settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed.h5"
            exp.save(path)
            with Experiment.open(path) as restored:
                settings = restored.processing.settings["localisation"]["a"]["DD"]
                self.assertEqual(settings["settings"]["fit_method"], "mle")
                self.assertEqual(settings["camera"]["baseline"], 0)
                self.assertEqual(settings["failed_fits"], 0)

    def test_empty_fit_can_replace_partition_and_preview_does_not_mutate(self):
        exp = spot_experiment()
        ids = exp.localizations.add(acquisition_id="a", channel="image", frame=0, x=1, y=1)
        settings = LocalisationSettings(min_net_gradient=1e12, chunk_size=5)
        preview = localisation.localise(
            exp,
            "a",
            "image",
            camera=CameraSettings(0, 1, 1),
            settings=settings,
            frame_range=(0, 1),
            apply=False,
            cpu=self.cpu,
        )
        self.assertEqual(len(preview.columns["frame"]), 0)
        self.assertEqual(exp.localizations.query()["id"].tolist(), ids.tolist())
        localisation.localise(
            exp, "a", "image", camera=CameraSettings(0, 1, 1), settings=settings, cpu=self.cpu
        )
        self.assertEqual(len(exp.localizations), 0)

    def test_cancelled_and_stale_runs_do_not_overwrite_active_results(self):
        exp = spot_experiment()
        ids = exp.localizations.add(acquisition_id="a", channel="image", frame=0, x=1, y=1)
        stop = threading.Event()
        stop.set()
        with self.assertRaises(CancelledError):
            localisation.localise(
                exp, "a", "image", camera=CameraSettings(0, 1, 1), cpu=self.cpu, cancel_event=stop
            )
        changed = []

        def progress(update):
            if not changed:
                exp.localizations.move(ids[0], x=2, y=2)
                changed.append(True)

        with self.assertRaises(StaleResultError):
            localisation.localise(
                exp, "a", "image", camera=CameraSettings(0, 1, 1), cpu=self.cpu, progress=progress
            )
        self.assertEqual(exp.localizations.query()["id"].tolist(), ids.tolist())
        self.assertEqual(exp.localizations.query()["x"].tolist(), [2])


@unittest.skipUnless(importlib.util.find_spec("picasso"), "Picasso processing extra not installed")
class UndriftingTests(ProcessingCase):
    def test_rcc_rejects_fewer_than_four_backend_segments_before_starting_workers(self):
        exp = Experiment()
        exp.add_acquisition(Acquisition(ArraySource(np.zeros((34, 8, 8))), id="a"))
        exp.localizations.add(acquisition_id="a", channel="image", frame=np.arange(34), x=3, y=3)
        with patch("molcore.undrifting.executor", side_effect=AssertionError("Worker must not start")):
            with self.assertRaises(ValueError):
                undrifting.estimate_drift(
                    exp, "a", "image", settings=DriftSettings(segment_frames=10), num_cpus=1
                )

    def test_batch_undrift_uses_multiple_jobs_and_preserves_selection(self):
        rng = np.random.default_rng(13)
        points = rng.uniform(10, 50, (20, 2))
        exp = Experiment()
        for name in ("a", "b"):
            exp.add_acquisition(
                Acquisition(ArraySource(np.zeros((60, 64, 64), dtype=np.uint16)), id=name, pixel_size=100)
            )
            frame = np.repeat(np.arange(60), 20)
            positions = np.tile(points, (60, 1)) + np.column_stack([frame * 0.01, frame * 0.005])
            exp.localizations.add(
                acquisition_id=name, channel="image", frame=frame, x=positions[:, 0], y=positions[:, 1]
            )
        updates = []
        results = undrifting.undrift_many(
            exp,
            {"a": "image", "b": "image"},
            settings=DriftSettings(segment_frames=10),
            cpu=self.cpu,
            progress=updates.append,
        )
        self.assertEqual(set(results), {"a", "b"})
        self.assertTrue(all(result.applied for result in results.values()))
        self.assertTrue(all(update.total == 2 for update in updates))

    def test_rcc_and_aim_use_models_and_preserve_raw_coordinates(self):
        rng = np.random.default_rng(37)
        points = rng.uniform(10, 52, (30, 2))
        frame = np.repeat(np.arange(80), len(points))
        drift = np.column_stack([np.linspace(0, 1.2, 80), np.linspace(0, -0.8, 80)])
        positions = np.tile(points, (80, 1)) + np.repeat(drift, len(points), axis=0)
        positions += rng.normal(0, 0.02, positions.shape)
        for method in ("rcc", "aim"):
            with self.subTest(method=method):
                exp = Experiment()
                exp.add_acquisition(
                    Acquisition(
                        ArraySource(np.zeros((160, 64, 64), dtype=np.uint16)),
                        id="a",
                        pixel_size=100,
                        excitation_sequence=("D", "A"),
                        channel_specs={"DD": ChannelSpec(excitation="D"), "AA": ChannelSpec(excitation="A")},
                    )
                )
                exp.localizations.add(
                    acquisition_id="a", channel="DD", frame=2 * frame, x=positions[:, 0], y=positions[:, 1]
                )
                before = exp.localizations.query(original=True)["x"].copy()
                updates = []
                result = undrifting.undrift(
                    exp,
                    "a",
                    "DD",
                    settings=DriftSettings(method=method, segment_frames=10),
                    cpu=self.cpu,
                    progress=updates.append,
                )
                self.assertTrue(result.applied)
                self.assertEqual((int(result.frames[0]), int(result.frames[-1])), (0, 158))
                np.testing.assert_allclose(
                    result.displacement[-1] - result.displacement[0], [1.2, -0.8], atol=0.4
                )
                np.testing.assert_array_equal(exp.localizations.query(original=True)["x"], before)
                self.assertEqual(updates[-1].percent, 100)
                self.assertTrue(any(0 < u.percent < 100 for u in updates))


class AlignmentTests(unittest.TestCase):
    def test_affine_alignment_composes_target_reference_and_is_non_destructive(self):
        exp = Experiment()
        exp.add_acquisition(
            Acquisition(
                ArraySource(np.zeros((1, 30, 30))),
                id="a",
                channel_specs={"source": ChannelSpec(), "target": ChannelSpec()},
            )
        )
        points = np.array([[3.0, 3.0], [12.0, 4.0], [4.0, 15.0], [13.0, 14.0]])
        matrix = np.array([[1.05, 0.1, 2], [-0.05, 0.95, 1], [0, 0, 1.0]])
        target = points @ matrix[:2, :2].T + matrix[:2, 2]
        src_ids = exp.localizations.add(
            acquisition_id="a", channel="source", frame=0, x=points[:, 0], y=points[:, 1]
        )
        dst_ids = exp.localizations.add(
            acquisition_id="a", channel="target", frame=0, x=target[:, 0], y=target[:, 1]
        )
        target_ref = np.array([[1.0, 0, 5], [0, 1, -2], [0, 0, 1]])
        exp.transforms.add_calibration(Calibration("target-reference", target_ref))
        exp.transforms.select("a", "target", "target-reference")
        result = alignment.align_channels(
            exp,
            "a",
            "source",
            "target",
            source_ids=src_ids[::-1],
            target_ids=dst_ids[::-1],
            name="channel-calibration",
            num_cpus=1,
        )
        np.testing.assert_allclose(result.matrix, matrix, atol=1e-10)
        np.testing.assert_allclose(result.reference_matrix, target_ref @ matrix, atol=1e-10)
        np.testing.assert_allclose(result.residuals, 0, atol=1e-10)
        np.testing.assert_array_equal(exp.localizations.query(channel="source")["x"], points[:, 0])
        np.testing.assert_allclose(
            exp.localizations.query(channel="source", space="registered")["x"], target[:, 0] + 5
        )

    def test_translation_matching_and_degenerate_affine_rejection(self):
        exp = Experiment()
        exp.add_acquisition(
            Acquisition(
                ArraySource(np.zeros((1, 20, 20))),
                id="a",
                channel_specs={"s": ChannelSpec(), "t": ChannelSpec()},
            )
        )
        exp.localizations.add(acquisition_id="a", channel="s", frame=0, x=[2.0, 7.0], y=[2.0, 7.0])
        exp.localizations.add(acquisition_id="a", channel="t", frame=0, x=[2.2, 7.2], y=[2.3, 7.3])
        result = alignment.estimate_alignment(
            exp, "a", "s", "t", model="translation", match_radius=1, num_cpus=1
        )
        np.testing.assert_allclose(result.matrix[:2, 2], [0.2, 0.3])
        with self.assertRaises(ValueError):
            alignment.estimate_alignment(exp, "a", "s", "t", match_radius=1, num_cpus=1)
