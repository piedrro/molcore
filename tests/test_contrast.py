"""Display-only autocontrast numerical and lazy sampling contracts."""

import threading
import unittest
from concurrent.futures import CancelledError

import numpy as np

from molcore import Acquisition, ArraySource, ChannelSpec, Dataset
from molcore.contrast import autocontrast, autocontrast_channels
from molcore.types import ContrastMethod, ContrastScope, ContrastSettings


class ContrastTests(unittest.TestCase):
    def test_percentiles_and_unchanged_pixels(self):
        image = np.arange(100, dtype=float).reshape(10, 10)
        before = image.copy()
        result = autocontrast(image)
        np.testing.assert_allclose([result.lower, result.upper], [0.99, 98.01])
        np.testing.assert_equal(image, before)
        self.assertEqual(result.valid_pixel_count, 100)
        self.assertEqual(result.sample_count, 100)

    def test_masks_nonfinite_and_minmax(self):
        image = np.array([[np.nan, -5.0, 1.0], [np.inf, 8.0, 100.0]])
        mask = np.array([[True, True, True], [True, True, False]])
        result = autocontrast(
            image, mask=mask, settings=ContrastSettings(method=ContrastMethod.MIN_MAX)
        )
        self.assertEqual((result.lower, result.upper), (-5.0, 8.0))
        self.assertEqual(result.valid_pixel_count, 3)
        with self.assertRaises(ValueError):
            autocontrast(image, mask=np.ones((1, 1), dtype=bool))
        with self.assertRaises(TypeError):
            autocontrast(image, mask=np.ones_like(image))

    def test_empty_and_constant_images(self):
        for image in (np.empty((0, 2)), np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):
                autocontrast(image)
        for value in (0.0, -100.0, 100.0, np.finfo(float).max, -np.finfo(float).max):
            result = autocontrast(np.full((2, 2), value))
            self.assertTrue(np.isfinite([result.lower, result.upper]).all())
            self.assertLess(result.lower, result.upper)
            self.assertLessEqual(result.lower, value)
            self.assertGreaterEqual(result.upper, value)

    def test_settings_use_enums_and_validate_limits(self):
        with self.assertRaises(TypeError):
            ContrastSettings(method="percentile")
        for kwargs in (
            {"percentiles": (99.0, 1.0)},
            {"percentiles": (-1.0, 99.0)},
            {"max_frames": 0},
            {"max_pixels_per_frame": True},
        ):
            with self.assertRaises(ValueError):
                ContrastSettings(**kwargs)

    def test_deterministic_sampling(self):
        movie = np.arange(20 * 10 * 10, dtype=float).reshape(20, 10, 10)
        settings = ContrastSettings(max_frames=3, max_pixels_per_frame=7)
        first = autocontrast(movie, settings=settings)
        second = autocontrast(movie, settings=settings)
        self.assertEqual(first, second)
        self.assertEqual(first.frames, (0, 9, 19))
        self.assertEqual(first.sample_count, 21)
        self.assertEqual(first.valid_pixel_count, 300)
        exact = autocontrast(
            movie,
            settings=ContrastSettings(
                method=ContrastMethod.MIN_MAX, max_frames=None, max_pixels_per_frame=1
            ),
        )
        self.assertEqual((exact.lower, exact.upper), (0.0, 1999.0))

    def test_channels_alex_frames_and_shared_limits(self):
        movie = np.stack([np.full((3, 3), v) for v in (0.0, 100.0, 10.0, 110.0)])
        exp = Dataset()
        exp.add_acquisition(
            Acquisition(
                ArraySource(movie),
                id="a",
                channel_specs={
                    "d": ChannelSpec(excitation="d"),
                    "a": ChannelSpec(excitation="a"),
                },
                excitation_sequence=("d", "a"),
            )
        )
        settings = ContrastSettings(method=ContrastMethod.MIN_MAX)
        result = autocontrast_channels(exp, "a", settings=settings)
        self.assertEqual((result["d"].lower, result["d"].upper), (0.0, 10.0))
        self.assertEqual(result["d"].frames, (0, 2))
        self.assertEqual(result["a"].frames, (1, 3))
        shared = autocontrast_channels(
            exp, "a", settings=settings, scope=ContrastScope.SHARED
        )
        for limits in shared.values():
            self.assertEqual((limits.lower, limits.upper), (0.0, 110.0))
        with self.assertRaises(ValueError):
            autocontrast_channels(exp, "a", frame_range=(4, 5))

    def test_lazy_reads_and_cancellation(self):
        class CountingSource(ArraySource):
            def __init__(self, data):
                super().__init__(data)
                self.reads = []

            def _frame(self, index):
                self.reads.append(index)
                return super()._frame(index)

        source = CountingSource(np.ones((20, 3, 3)))
        exp = Dataset()
        exp.add_acquisition(Acquisition(source, id="a"))
        autocontrast_channels(exp, "a", settings=ContrastSettings(max_frames=3))
        self.assertEqual(source.reads, [0, 9, 19])
        event = threading.Event()
        event.set()
        with self.assertRaises(CancelledError):
            autocontrast_channels(exp, "a", cancel_event=event)
        self.assertEqual(source.reads, [0, 9, 19])


class ContrastEdgeTests(unittest.TestCase):
    def test_shared_percentiles_pool_pixels(self):
        exp = Dataset()
        exp.add_acquisition(
            Acquisition(
                ArraySource(np.array([[[0.0, 0.0, 100.0, 100.0]]])),
                id="a",
                channel_specs={
                    "d": ChannelSpec(roi=(0, 1, 0, 2)),
                    "a": ChannelSpec(roi=(0, 1, 2, 4)),
                },
            )
        )
        from molcore.staging import state_token

        before = state_token(exp)
        result = autocontrast_channels(
            exp,
            "a",
            scope=ContrastScope.SHARED,
            settings=ContrastSettings(percentiles=(25.0, 75.0)),
        )
        self.assertEqual(result["d"].limits, (0.0, 100.0))
        self.assertEqual(result["a"].limits, (0.0, 100.0))
        self.assertEqual(state_token(exp), before)

    def test_range_and_time_dependent_mask(self):
        exp = Dataset()
        exp.add_acquisition(
            Acquisition(ArraySource(np.arange(12).reshape(3, 2, 2)), id="a")
        )
        mask = np.zeros((3, 2, 2), dtype=bool)
        mask[1:, 0, 0] = True
        result = autocontrast_channels(
            exp,
            "a",
            frame_range=(1, 3),
            masks={"image": mask},
            settings=ContrastSettings(method=ContrastMethod.MIN_MAX),
        )["image"]
        self.assertEqual(result.frames, (1, 2))
        self.assertEqual(result.limits, (4.0, 8.0))
        self.assertEqual(result.sample_count, 2)

    def test_extreme_percentiles_and_constant_without_warnings(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for value in (np.finfo(float).max, -np.finfo(float).max):
                result = autocontrast(np.full((2, 2), value))
                self.assertTrue(np.isfinite(result.limits).all())
            result = autocontrast(
                np.array([[-np.finfo(float).max, np.finfo(float).max]])
            )
            self.assertTrue(np.isfinite(result.limits).all())
            self.assertLess(result.lower, result.upper)

    def test_progress_and_callback_cancellation(self):
        updates = []
        image = np.ones((3, 2, 2))
        autocontrast(image, progress=lambda update: updates.append(update.percent))
        self.assertEqual(updates[0], 0.0)
        self.assertEqual(updates[-1], 100.0)
        event = threading.Event()
        with self.assertRaises(CancelledError):
            autocontrast(image, cancel_event=event, progress=lambda _: event.set())

    def test_stale_channel_input_rejected(self):
        from molcore import StaleResultError

        exp = Dataset()
        exp.add_acquisition(Acquisition(ArraySource(np.ones((3, 2, 2))), id="a"))

        def progress(_):
            exp.metadata["edited"] = True

        with self.assertRaises(StaleResultError):
            autocontrast_channels(exp, "a", progress=progress)


class PercentilePrecisionTests(unittest.TestCase):
    def test_large_outlier_preserves_tiny_percentiles(self):
        image = np.r_[np.arange(1, 101) * 1e-100, 1e308].reshape(1, -1)
        result = autocontrast(image)
        expected = np.percentile(image, (1.0, 99.0))
        np.testing.assert_allclose(result.limits, expected, rtol=1e-14, atol=0.0)

    def test_safe_interpolation_across_opposite_extremes(self):
        largest = np.finfo(float).max
        settings = ContrastSettings(percentiles=(25.0, 75.0))
        with np.errstate(over="raise", invalid="raise"):
            result = autocontrast(np.array([[-largest, largest]]), settings=settings)
        np.testing.assert_allclose(result.limits, [-largest / 2, largest / 2])
