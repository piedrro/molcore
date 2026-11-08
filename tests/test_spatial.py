import unittest

import numpy as np
from scientific_helpers import compute_scenario

from molcore import Acquisition, ArraySource, Calibration, Dataset, Segmentation


class SpatialTests(unittest.TestCase):
    def setUp(self):
        self.exp = Dataset()
        movie = np.zeros((3, 9, 9), dtype=np.uint16)
        movie[:, 3, 2] = [10, 20, 30]
        self.exp.add_acquisition(Acquisition(ArraySource(movie), id="a"))
        self.ids = self.exp.localizations.add(
            acquisition_id="a", channel="image", frame=[0, 1, 2], x=[2.0, 2.0, 2.0], y=[3.0, 3.0, 3.0]
        )

    def test_affine_drift_points_and_images_agree(self):
        self.exp.transforms.add_calibration(Calibration("shift", [[1, 0, 2], [0, 1, 1], [0, 0, 1]]))
        self.exp.transforms.select("a", "image", "shift")
        self.exp.transforms.set_drift("a", [[1, 0], [1, 0], [1, 0]])
        rows = self.exp.localizations.query(space="corrected")
        np.testing.assert_array_equal(rows["x"], [3, 3, 3])
        np.testing.assert_array_equal(rows["y"], [4, 4, 4])
        image = self.exp.transforms.image("a", "image", 1, space="corrected")
        self.assertEqual(image[4, 3], 20)
        self.assertEqual(image[3, 2], 0)
        np.testing.assert_array_equal(
            self.exp.localizations.query(original=True, space="raw")["x"], [2, 2, 2]
        )

    def test_default_display_queries_match_corrected_images_without_editing_raw_data(self):
        self.exp.transforms.add_calibration(Calibration("shift", [[1, 0, 2], [0, 1, 1], [0, 0, 1]]))
        self.exp.transforms.select("a", "image", "shift")
        self.exp.transforms.set_drift("a", [[0, 0], [1, 0], [2, 1]])
        track = self.exp.tracks.add(self.ids)
        for rows in (self.exp.localizations.query(), self.exp.tracks.query(track_id=track)):
            np.testing.assert_array_equal(rows["x"], [4, 3, 2])
            np.testing.assert_array_equal(rows["y"], [4, 4, 3])
            for frame, x, y in zip(rows["frame"], rows["x"], rows["y"], strict=True):
                image = self.exp.transforms.view("a", "image")[frame]
                self.assertEqual(image[int(y), int(x)], (frame + 1) * 10)
        self.exp.localizations.move(self.ids[0], x=3, y=3)
        self.assertEqual(self.exp.localizations.query(ids=self.ids[:1])["x"].tolist(), [5])
        np.testing.assert_array_equal(self.exp.localizations.query(space="raw")["x"], [3, 2, 2])
        np.testing.assert_array_equal(
            self.exp.localizations.query(original=True, space="raw")["x"], [2, 2, 2]
        )
        np.testing.assert_array_equal(self.exp.acquisitions["a"].channels["image"][:, 3, 2], [10, 20, 30])

    def test_raw_crops_and_mask_constraints_do_not_apply_display_corrections_twice(self):
        self.exp.transforms.add_calibration(Calibration("shift", [[1, 0, 2], [0, 1, 1], [0, 0, 1]]))
        self.exp.transforms.select("a", "image", "shift")
        self.exp.transforms.set_drift("a", [[1, 0], [1, 0], [1, 0]])
        labels = np.zeros((9, 9), dtype=np.uint16)
        labels[4, 3] = 7
        self.exp.add_segmentation(Segmentation("cells", "a", labels, space="corrected"))
        track = self.exp.tracks.add(self.ids, segmentation_id="cells")
        np.testing.assert_array_equal(self.exp.tracks.query(track_id=track)["x"], [3, 3, 3])
        self.assertEqual(self.exp.crops.localization(self.ids[1], width=3).pixels[1, 1], 20)

    def test_density_rendering_defaults_to_corrected_coordinates(self):
        from molcore.rendering import render_localizations

        self.exp.transforms.add_calibration(Calibration("shift", [[1, 0, 2], [0, 1, 1], [0, 0, 1]]))
        self.exp.transforms.select("a", "image", "shift")
        self.exp.transforms.set_drift("a", [[1, 0], [1, 0], [1, 0]])
        result = compute_scenario(render_localizations, self.exp, "a", "image")
        np.testing.assert_array_equal(result.columns["x"], [3, 3, 3])
        np.testing.assert_array_equal(result.columns["y"], [4, 4, 4])

    def test_masks_and_track_constraints(self):
        labels = np.zeros((9, 9), dtype=np.uint16)
        labels[2:5, 1:4] = 7
        self.exp.add_segmentation(Segmentation("cells", "a", labels, labels.copy()))
        self.assertEqual(
            self.exp.localizations.query(segmentation_id="cells", label=7)["id"].tolist(), self.ids.tolist()
        )
        track = self.exp.tracks.add(self.ids, segmentation_id="cells")
        self.exp.localizations.move(self.ids[-1], x=8, y=8)
        self.assertEqual(
            self.exp.localizations.query(segmentation_id="cells")["id"].tolist(), self.ids[:2].tolist()
        )
        with self.assertRaises(ValueError):
            self.exp.tracks.merge(
                [
                    track,
                    self.exp.tracks.add(
                        self.exp.localizations.add(acquisition_id="a", channel="image", frame=0, x=8, y=8)
                    ),
                ]
            )

    def test_crop_borders_stationary_sampling_and_saved_originals(self):
        self.exp.localizations.move(self.ids[0], x=0, y=0)
        crop = self.exp.crops.localization(self.ids[0], width=3)
        self.assertEqual(crop.origin, (-1, -1))
        self.assertEqual(int(crop.valid.sum()), 4)
        self.assertEqual(crop.pixels.shape, (3, 3))
        track = self.exp.tracks.add(self.ids[1:])
        crops = list(compute_scenario(self.exp.crops.track, track, width=3))
        self.assertEqual([c.camera_frame for c in crops], [1, 2])
        sampled = list(self.exp.crops.stationary(self.ids[1], [0, 2], width=3))
        self.assertEqual([int(c.pixels[1, 1]) for c in sampled], [10, 30])
        with self.assertRaises(ValueError):
            self.exp.crops.localization(self.ids[0], width=4)

    def test_drift_never_extrapolates(self):
        self.exp.transforms.set_drift("a", [[1.0, 0.0]], frames=[0])
        with self.assertRaises(ValueError):
            self.exp.localizations.query(ids=[self.ids[1]], space="corrected")
