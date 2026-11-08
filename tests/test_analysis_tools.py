"""Numerical contracts for the detached shared analysis tools."""

import unittest

import numpy as np
from scientific_helpers import compute_scenario

from molcore import Acquisition, ArraySource, ChannelSpec, Dataset
from molcore.clustering import cluster_localizations
from molcore.colocalisation import colocalise
from molcore.filtering import filter_results
from molcore.rendering import render_localizations
from molcore.types.analysis import ClusteringSettings, ColocalisationSettings, FilterSettings, RenderSettings


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.model = Dataset()
        self.model.add_acquisition(
            Acquisition(
                ArraySource(np.zeros((3, 12, 12))),
                id="a",
                channel_specs={"a": ChannelSpec(), "b": ChannelSpec()},
            )
        )
        self.ids = self.model.localizations.add(
            acquisition_id="a",
            channel="a",
            frame=[0, 1, 2],
            x=[2.0, 2.1, 9.0],
            y=[2.0, 2.0, 9.0],
            photons=[1.0, 2.0, np.nan],
        )
        self.other = self.model.localizations.add(
            acquisition_id="a", channel="b", frame=[0, 1], x=[3.0, 8.0], y=[2.0, 8.0]
        )

    def test_inclusive_filter_detached_and_finite(self):
        result = compute_scenario(
            filter_results, self.model, "a", "a", settings=FilterSettings("photons", 1.0, 2.0)
        )
        np.testing.assert_array_equal(result.selected_ids, self.ids[:2])
        self.assertEqual(len(self.model.localizations), 5)
        with self.assertRaises(ValueError):
            result.columns["x"][:] = 0
        self.assertEqual(self.model.localizations.query(ids=self.ids)["x"][0], 2.0)

    def test_track_aggregation(self):
        track = self.model.tracks.add(self.ids[:2])
        result = compute_scenario(
            filter_results,
            self.model,
            "a",
            "a",
            settings=FilterSettings("photons", 1.5, 1.5, target="tracks"),
        )
        np.testing.assert_array_equal(result.selected_ids, [track])

    def test_whole_track_reduction_includes_other_acquisitions(self):
        model = Dataset()
        ids = []
        for acquisition, photons in (("first", 1.0), ("second", 9.0)):
            model.add_acquisition(Acquisition(ArraySource(np.zeros((1, 4, 4))), id=acquisition))
            ids.extend(
                model.localizations.add(
                    acquisition_id=acquisition,
                    channel="image",
                    frame=[0],
                    x=[1.0],
                    y=[1.0],
                    photons=[photons],
                )
            )
        model.add_sequence(["first", "second"], id="seq", allow_cross_boundary_tracks=True, reference="r")
        track = model.tracks.add(ids, sequence_id="seq")
        result = compute_scenario(
            filter_results,
            model,
            "first",
            "image",
            settings=FilterSettings("photons", 5.0, 5.0, target="tracks"),
        )
        np.testing.assert_array_equal(result.selected_ids, [track])
        self.assertEqual(len(result.columns["id"]), 2)

    def test_mutual_matching_inclusive_euclidean(self):
        result = compute_scenario(
            colocalise, self.model, "a", "a", "b", settings=ColocalisationSettings(radius=0.9)
        )
        np.testing.assert_array_equal(result.columns["source_id"], self.ids[1:2])
        np.testing.assert_array_equal(result.columns["target_id"], self.other[:1])
        self.assertEqual(result.unmatched_source, 2)

    def test_matching_ties_use_stable_ids(self):
        model = Dataset()
        model.add_acquisition(
            Acquisition(
                ArraySource(np.zeros((1, 4, 4))),
                id="a",
                channel_specs={"a": ChannelSpec(), "b": ChannelSpec()},
            )
        )
        ids = model.localizations.add(
            acquisition_id="a", channel="a", frame=[0, 0], x=[0.0, 2.0], y=[1.0, 1.0]
        )
        model.localizations.add(acquisition_id="a", channel="b", frame=[0], x=[1.0], y=[1.0])
        result = compute_scenario(
            colocalise, model, "a", "a", "b", settings=ColocalisationSettings(radius=1.0)
        )
        np.testing.assert_array_equal(result.columns["source_id"], ids[:1])

    def test_dbscan_noise_and_ids(self):
        result = compute_scenario(
            cluster_localizations,
            self.model,
            "a",
            "a",
            settings=ClusteringSettings(radius=0.2, min_samples=2),
        )
        np.testing.assert_array_equal(result.columns["id"], self.ids)
        np.testing.assert_array_equal(result.columns["cluster"], [0, 0, -1])
        np.testing.assert_allclose(result.centroids["x"], [2.05])

    def test_render_geometry_without_calibration(self):
        result = compute_scenario(
            render_localizations, self.model, "a", "a", settings=RenderSettings(oversampling=2.0), space="raw"
        )
        self.assertEqual(result.image.shape, (24, 24))
        self.assertEqual(result.pixel_spacing, 0.5)
        self.assertEqual(result.origin, (0.0, 0.0))
        self.assertEqual(result.count, 3)

    def test_invalid_settings(self):
        for factory in (
            lambda: ClusteringSettings(radius=0),
            lambda: ClusteringSettings(min_samples=0),
            lambda: ColocalisationSettings(radius=-1),
            lambda: RenderSettings(oversampling=0),
            lambda: FilterSettings("x", 2, 1),
        ):
            with self.assertRaises((ValueError, TypeError)):
                factory()

    def test_empty_partitions_and_cancellation(self):
        import threading
        from concurrent.futures import CancelledError

        self.model.localizations.delete(self.ids)
        self.assertEqual(len(compute_scenario(cluster_localizations, self.model, "a", "a").columns["id"]), 0)
        self.assertEqual(len(compute_scenario(colocalise, self.model, "a", "a", "b").columns["source_id"]), 0)
        self.assertEqual(compute_scenario(render_localizations, self.model, "a", "a").count, 0)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            compute_scenario(filter_results, self.model, "a", "a", cancel_event=cancel)

    def test_raw_channels_with_different_rois_require_registration(self):
        model = Dataset()
        model.add_acquisition(
            Acquisition(
                ArraySource(np.zeros((1, 12, 12))),
                id="a",
                channel_specs={"a": ChannelSpec((0, 12, 0, 6)), "b": ChannelSpec((0, 12, 6, 12))},
            )
        )
        with self.assertRaises(ValueError):
            compute_scenario(colocalise, model, "a", "a", "b", space="raw")

    def test_precision_blur_requires_uncertainty_measurements(self):
        with self.assertRaisesRegex(ValueError, "lpx"):
            compute_scenario(
                render_localizations, self.model, "a", "a", settings=RenderSettings(blur_method="gaussian")
            )


if __name__ == "__main__":
    unittest.main()
