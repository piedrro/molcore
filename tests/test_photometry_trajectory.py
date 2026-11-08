"""Numerical contracts for shared photometry and trajectory analysis."""

import threading
import unittest
from concurrent.futures import CancelledError
from unittest.mock import patch

import numpy as np
from scientific_helpers import compute_scenario

from molcore import Acquisition, ArraySource, Calibration, ChannelSpec, Dataset, StaleResultError
from molcore.cpu import CpuExecutor
from molcore.photometry import measure_localizations, stationary_traces, track_traces
from molcore.tracking import track
from molcore.trajectory import trajectory_statistics
from molcore.types import (
    ApertureShape,
    BackgroundMethod,
    CoordinateSpace,
    PhotometrySettings,
    SpatialUnit,
    TrackingSettings,
)


class InlineCpu:
    def map(self, fn, jobs, **kwargs):

        class Reporter:
            def report(self, *args):
                pass

            def check_cancelled(self):
                pass

        if kwargs.get("cancel_event") is not None and kwargs["cancel_event"].is_set():
            raise CancelledError()
        result = [fn(job, Reporter()) for job in jobs]
        callback = kwargs.get("progress")
        if callback:
            callback(None)
        return result


def fixture(movie=None):
    if movie is None:
        movie = np.full((5, 15, 15), 10.0, dtype=float)
        movie[:, 6:9, 6:9] = 20.0
    exp = Dataset()
    exp.add_acquisition(Acquisition(ArraySource(movie), id="a", pixel_size=100.0))
    ids = exp.localizations.add(
        acquisition_id="a", channel="image", frame=[0, 2, 4], x=[7.0, 7.0, 7.0], y=[7.0, 7.0, 7.0]
    )
    return (exp, ids)


class PhotometryTests(unittest.TestCase):
    def test_default_photometry_matches_borrowed_executor_without_processes(self):
        exp, ids = fixture()
        exp.transforms.set_drift("a", np.column_stack([np.arange(5), np.zeros(5)]))
        expected = compute_scenario(stationary_traces, exp, ids, apply=False, cpu=InlineCpu())
        updates = []

        def progress(update):
            self.assertIs(threading.current_thread(), threading.main_thread())
            updates.append(update.percent)

        with patch("molcore.photometry.executor", side_effect=AssertionError("process pool")):
            actual = compute_scenario(stationary_traces, exp, ids, apply=False, num_cpus=2, progress=progress)
        columns = actual.columns
        for name, values in expected.columns.items():
            if name not in {"id", "sequence_id", "run_id"}:
                np.testing.assert_array_equal(columns[name], values)
        self.assertEqual(updates[-1], 100)
        self.assertEqual(updates, sorted(updates))
        stop = threading.Event()

        def cancel(update):
            if update.completed > 0:
                stop.set()

        with self.assertRaises(CancelledError):
            compute_scenario(stationary_traces, exp, ids, num_cpus=2, progress=cancel, cancel_event=stop)
        self.assertEqual(len(exp.photon_sequences), 0)

    def test_exact_metrics_and_runs(self):
        exp, ids = fixture()
        result = compute_scenario(measure_localizations, exp, ids, cpu=InlineCpu())
        np.testing.assert_allclose(result.columns["signal_sum"], 180.0)
        np.testing.assert_allclose(result.columns["local_background_mean"], 10.0)
        np.testing.assert_allclose(result.columns["intensity"], 90.0)
        self.assertEqual(len(exp.observations), 3)
        again = compute_scenario(measure_localizations, exp, ids, cpu=InlineCpu())
        self.assertNotEqual(result.run_id, again.run_id)
        self.assertEqual(len(exp.observations), 6)
        self.assertIsNotNone(result.observation_ids)

    def test_drift_following_and_measured_track_gaps(self):
        exp, ids = fixture()
        exp.transforms.set_drift("a", np.column_stack([np.arange(5), np.zeros(5)]))
        result = compute_scenario(stationary_traces, exp, ids[:1], apply=False, cpu=InlineCpu())
        np.testing.assert_allclose(result.columns["x"], [7.0, 8.0, 9.0, 10.0, 11.0])
        tid = exp.tracks.add(ids)
        result = compute_scenario(track_traces, exp, [tid], apply=False, cpu=InlineCpu())
        np.testing.assert_equal(result.columns["frame"], [0, 2, 4])

    def test_edges_nonfinite_negative_background_and_empty(self):
        exp, ids = fixture()
        exp.localizations.move_many(ids[:1], x=0.0, y=0.0)
        settings = PhotometrySettings(background=BackgroundMethod.GLOBAL)
        result = compute_scenario(
            measure_localizations, exp, ids[:1], settings=settings, apply=False, cpu=InlineCpu()
        )
        self.assertEqual(result.columns["signal_count"][0], 4)
        self.assertLess(result.columns["intensity"][0], 0.0)
        result = compute_scenario(measure_localizations, exp, [], apply=False, cpu=InlineCpu())
        self.assertEqual(len(result.columns["frame"]), 0)

    def test_stale_and_cancelled_preserve_active_data(self):
        exp, ids = fixture()

        def progress(_):
            exp.localizations.move(ids[0], x=6.0, y=7.0)

        with self.assertRaises(StaleResultError):
            compute_scenario(measure_localizations, exp, ids, cpu=InlineCpu(), progress=progress)
        self.assertEqual(len(exp.observations), 0)
        event = threading.Event()
        event.set()
        with self.assertRaises(CancelledError):
            compute_scenario(stationary_traces, exp, ids, cpu=InlineCpu(), cancel_event=event)
        self.assertEqual(len(exp.observations), 0)

    def test_stationary_alternating_channels_registration(self):
        exp = Dataset()
        specs = {"d": ChannelSpec(excitation="d"), "a": ChannelSpec(excitation="a")}
        exp.add_acquisition(
            Acquisition(
                ArraySource(np.ones((4, 15, 15))), id="a", channel_specs=specs, excitation_sequence=("d", "a")
            )
        )
        exp.transforms.add_calibration(
            Calibration("offset", [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        exp.transforms.select("a", "a", "offset")
        ids = exp.localizations.add(acquisition_id="a", channel="d", frame=0, x=7.0, y=7.0)
        result = compute_scenario(stationary_traces, exp, ids, channels=["a"], apply=False, cpu=InlineCpu())
        np.testing.assert_equal(result.columns["frame"], [1, 3])
        np.testing.assert_allclose(result.columns["x"], 5.0)

    def test_real_spawn(self):
        exp, ids = fixture()
        with CpuExecutor(num_cpus=1) as cpu:
            result = compute_scenario(measure_localizations, exp, ids, cpu=cpu, apply=False)
        np.testing.assert_allclose(result.columns["intensity"], 90.0)


class TrajectoryTests(unittest.TestCase):
    def test_actual_camera_timing_and_units(self):
        exp, ids = fixture()
        exp.localizations.move_many(ids, x=[1.0, 3.0, 5.0], y=1.0)
        tid = exp.tracks.add(ids)
        result = trajectory_statistics(
            exp.snapshot(),
            tid,
            frame_interval=0.5,
            unit=SpatialUnit.NANOMETRES,
            space=CoordinateSpace.RAW,
            fit_lag_range=(0.5, 2.5),
        )
        np.testing.assert_allclose(result.steps["elapsed_time"], 1.0)
        np.testing.assert_allclose(result.steps["speed"], 200.0)
        np.testing.assert_equal(result.steps["gap_frames"], 1)
        np.testing.assert_allclose(result.msd["lag"], [1.0, 2.0])
        np.testing.assert_allclose(result.msd["msd"], [40000.0, 160000.0])
        self.assertAlmostEqual(result.path_length, 400.0)
        self.assertIsNotNone(result.diffusion)

    def test_missing_and_irregular_timing(self):
        exp, ids = fixture()
        tid = exp.tracks.add(ids)
        with self.assertRaises(ValueError):
            trajectory_statistics(exp.snapshot(), tid)
        exp.update_acquisition("a", timestamps=np.array([0.0, 0.2, 0.7, 0.8, 2.0]))
        result = trajectory_statistics(exp.snapshot(), tid, lag_bin_edges=[0.0, 1.0, 3.0])
        np.testing.assert_equal(result.msd["count"], [1, 2])

    def test_tracking_gaps_and_partition_application(self):
        exp, ids = fixture()
        settings = TrackingSettings(search_range=2.0, memory=1, min_length=3, space=CoordinateSpace.RAW)
        result = compute_scenario(track, exp, "a", "image", settings=settings, cpu=InlineCpu())
        self.assertEqual(len(result.memberships), 1)
        np.testing.assert_equal(result.memberships[0], ids)
        self.assertEqual(len(exp.tracks), 1)
        with self.assertRaises(ValueError):
            TrackingSettings(search_range=-1.0)


class NumericalEdgeTests(unittest.TestCase):
    def test_nonfinite_and_circle_aperture(self):
        exp, ids = fixture(np.full((5, 15, 15), np.nan))
        result = compute_scenario(measure_localizations, exp, ids[:1], apply=False, cpu=InlineCpu())
        self.assertEqual(result.columns["signal_count"][0], 0)
        self.assertTrue(np.isnan(result.columns["intensity"][0]))
        exp, ids = fixture(np.ones((5, 15, 15)))
        result = compute_scenario(
            measure_localizations,
            exp,
            ids[:1],
            settings=PhotometrySettings(spot_size=5, shape=ApertureShape.CIRCLE),
            apply=False,
            cpu=InlineCpu(),
        )
        self.assertEqual(result.columns["signal_count"][0], 21)

    def test_neighbour_background_exclusion(self):
        exp, ids = fixture()
        source = exp.acquisitions["a"].source[:].copy()
        source[:, 6:9, 9:12] = 1000.0
        exp, ids = fixture(source)
        exp.localizations.add(acquisition_id="a", channel="image", frame=0, x=10.0, y=7.0)
        masked = compute_scenario(measure_localizations, exp, ids[:1], cpu=InlineCpu(), apply=False)
        unmasked = compute_scenario(
            measure_localizations,
            exp,
            ids[:1],
            settings=PhotometrySettings(exclude_neighbours=False),
            cpu=InlineCpu(),
            apply=False,
        )
        self.assertAlmostEqual(masked.columns["local_background_mean"][0], 10.0)
        self.assertGreater(unmasked.columns["local_background_mean"][0], 10.0)

    def test_missing_drift_and_pixels(self):
        import tempfile
        from pathlib import Path

        from molcore import ImageUnavailableError

        exp, ids = fixture()
        exp.transforms.set_drift("a", [[0.0, 0.0]], frames=[0])
        with self.assertRaises(ValueError):
            compute_scenario(stationary_traces, exp, ids[:1], cpu=InlineCpu())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no_pixels.h5"
            exp.save(path)
            with Dataset.open(path) as restored:
                with self.assertRaises(ImageUnavailableError):
                    compute_scenario(measure_localizations, restored, ids, cpu=InlineCpu())

    def test_tracking_segmentation_empty_and_cancellation(self):
        from molcore import Segmentation

        exp, ids = fixture()
        labels = np.zeros((15, 15), dtype=np.int32)
        labels[5:10, 5:10] = 1
        exp.add_segmentation(
            Segmentation(
                id="cells",
                labels=labels,
                source_image=np.ones((15, 15)),
                acquisition_id="a",
                channel="image",
                frame=0,
            )
        )
        with CpuExecutor(num_cpus=1) as cpu:
            result = compute_scenario(
                track,
                exp,
                "a",
                "image",
                segmentation_id="cells",
                settings=TrackingSettings(memory=1, min_length=3),
                cpu=cpu,
            )
        np.testing.assert_equal(result.memberships[0], ids)
        event = threading.Event()
        event.set()
        with self.assertRaises(CancelledError):
            compute_scenario(track, exp, "a", "image", cpu=InlineCpu(), cancel_event=event)
        result = compute_scenario(
            track, exp, "a", "image", settings=TrackingSettings(memory=0, min_length=2), cpu=InlineCpu()
        )
        self.assertEqual(len(result.memberships), 0)
        self.assertEqual(len(exp.tracks), 0)

    def test_invalid_timing_calibration_and_bins(self):
        exp, ids = fixture()
        tid = exp.tracks.add(ids)
        exp.update_acquisition("a", timestamps=np.array([0.0, 1.0, 0.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            trajectory_statistics(exp.snapshot(), tid)
        exp.update_acquisition("a", timestamps=None)
        exp.update_acquisition("a", pixel_size=None)
        with self.assertRaises(ValueError):
            trajectory_statistics(exp.snapshot(), tid, frame_interval=1.0, unit=SpatialUnit.NANOMETRES)
        with self.assertRaises(ValueError):
            trajectory_statistics(exp.snapshot(), tid, frame_interval=1.0, lag_bin_edges=[0.0, 1.0, 1.0])

    def test_photometry_run_persistence(self):
        import tempfile
        from pathlib import Path

        exp, ids = fixture()
        result = compute_scenario(measure_localizations, exp, ids, cpu=InlineCpu())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photometry.h5"
            exp.save(path)
            with Dataset.open(path) as restored:
                columns = restored.observations.query()
                np.testing.assert_equal(columns["localization_id"], ids)
                self.assertTrue(all(columns["run_id"] == result.run_id))
                self.assertIn(result.run_id, restored.processing.settings["photometry"])
