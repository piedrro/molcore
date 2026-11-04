"""Coordinate-only DBSCAN shares native geometry without synthetic datasets."""

import threading
import unittest
from concurrent.futures import CancelledError

import numpy as np

from molcore import Acquisition, ArraySource, Dataset, StaleResultError, clustering
from molcore.staging import state_token
from molcore.types.analysis import ClusteringSettings


class ClusterPointsTests(unittest.TestCase):
    def setUp(self):
        self.settings = ClusteringSettings(radius=0.3, min_samples=2)

    def test_labels_and_centroids_preserve_input_order_and_noise(self):
        points = np.array(
            [
                [10, 10],
                [0, 0],
                [10.1, 10.1],
                [0.1, 0.1],
                [np.nan, 2],
                [2, np.inf],
                [5, 5],
            ],
            dtype=np.float32,
        )
        points.flags.writeable = False
        before = points.copy()
        labels, centroids = clustering.cluster_points(points, settings=self.settings)
        np.testing.assert_array_equal(labels, [0, 1, 0, 1, -1, -1, -1])
        np.testing.assert_array_equal(centroids["cluster"], [0, 1])
        np.testing.assert_array_equal(centroids["count"], [2, 2])
        np.testing.assert_allclose(centroids["x"], [10.05, 0.05], rtol=1e-6)
        np.testing.assert_allclose(centroids["y"], [10.05, 0.05], rtol=1e-6)
        self.assertEqual(labels.dtype, np.dtype("int64"))
        self.assertEqual(centroids["x"].dtype, np.dtype("float64"))
        labels[:] = 99
        centroids["x"][:] = 99
        np.testing.assert_array_equal(points, before)

    def test_float_conversion_and_repeated_calls_are_detached(self):
        points = [["0", "0"], ["0.1", "0.1"]]
        labels, centroids = clustering.cluster_points(points, settings=self.settings)
        np.testing.assert_array_equal(labels, [0, 0])
        np.testing.assert_allclose(centroids["x"], [0.05])
        labels[:] = -1
        centroids["count"][:] = 0
        repeated, centers = clustering.cluster_points(points, settings=self.settings)
        np.testing.assert_array_equal(repeated, [0, 0])
        np.testing.assert_array_equal(centers["count"], [2])
        self.assertEqual(points, [["0", "0"], ["0.1", "0.1"]])

    def test_empty_and_all_nonfinite_inputs(self):
        for points, expected in (
            (np.empty((0, 2)), []),
            ([[np.nan, 1], [0, np.inf]], [-1, -1]),
        ):
            with self.subTest(points=points):
                labels, centroids = clustering.cluster_points(
                    points, settings=self.settings
                )
                np.testing.assert_array_equal(labels, expected)
                self.assertEqual(set(centroids), {"cluster", "count", "x", "y"})
                self.assertTrue(all(len(values) == 0 for values in centroids.values()))

    def test_bad_shape_and_non_numeric_inputs_are_rejected(self):
        for points in (
            [],
            [1, 2],
            1,
            [[1]],
            [[1, 2, 3]],
            np.empty((2, 2, 2)),
            [["bad", 1]],
        ):
            with self.subTest(points=points), self.assertRaises(ValueError):
                clustering.cluster_points(points, settings=self.settings)

    def test_helper_matches_native_stable_id_order_without_model_changes(self):
        model = Dataset()
        model.add_acquisition(Acquisition(ArraySource(np.ones((1, 12, 12))), id="a"))
        model.localizations.add(
            acquisition_id="a",
            channel="image",
            id=[40, 10, 30, 20],
            frame=0,
            x=[10, 0, 10.1, 0.1],
            y=[10, 0, 10.1, 0.1],
        )
        before = state_token(model)
        updates = []
        native = clustering.cluster_localizations(
            model,
            "a",
            "image",
            settings=self.settings,
            progress=updates.append,
        )
        labels, centroids = clustering.cluster_points(
            np.column_stack([native.columns["x"], native.columns["y"]]),
            settings=self.settings,
        )
        np.testing.assert_array_equal(native.columns["id"], [10, 20, 30, 40])
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])
        np.testing.assert_array_equal(native.columns["cluster"], labels)
        for field in centroids:
            np.testing.assert_array_equal(native.centroids[field], centroids[field])
        self.assertEqual(state_token(model), before)
        self.assertEqual([update.percent for update in updates], [0, 100])

    def test_native_cancellation_and_freshness_survive_extraction(self):
        model = Dataset()
        model.add_acquisition(Acquisition(ArraySource(np.ones((1, 9, 9))), id="a"))
        ids = model.localizations.add(
            acquisition_id="a", channel="image", frame=0, x=1, y=1
        )
        event = threading.Event()
        event.set()
        with self.assertRaises(CancelledError):
            clustering.cluster_localizations(model, "a", "image", cancel_event=event)

        def edit(update):
            if update.percent == 100:
                model.localizations.move(ids[0], x=2, y=2)

        with self.assertRaises(StaleResultError):
            clustering.cluster_localizations(model, "a", "image", progress=edit)
