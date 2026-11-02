"""Optical registration preserves physical paths, projective geometry and pixels."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

from molcore import ArraySource, Calibration, Dataset, alignment
from molcore.files import (
    export_tiff,
    image_acquisition,
    read_calibration,
    write_calibration,
)


def projected(matrix, points):
    values = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return values[:, :2] / values[:, 2, None]


class OpticalAlignmentTests(unittest.TestCase):
    def test_calibration_can_be_reselected_after_snapshot_round_trip(self):
        dataset = Dataset()
        dataset.add_acquisition(self.alex())
        dataset.transforms.add_calibration(
            Calibration(
                "optics",
                np.eye(3),
                {
                    "source_emission": "A",
                    "source_roi": (0, 16, 16, 32),
                    "source_shape": (16, 16),
                    "target_emission": "D",
                    "target_roi": (0, 16, 0, 16),
                    "target_shape": (16, 16),
                },
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.h5"
            dataset.save(path)
            with Dataset.open(path) as restored:
                restored.transforms.select("a", "DA", "optics")
                np.testing.assert_array_equal(
                    restored.transforms._registration("a", "AA"), np.eye(3)
                )

    def alex(self, identifier="a"):
        pixels = np.arange(6 * 16 * 32, dtype=np.uint16).reshape(6, 16, 32)
        return image_acquisition(ArraySource(pixels), id=identifier, mode="alex")

    def test_presets_and_named_selection_share_emission_across_excitation(self):
        model = Dataset()
        a = self.alex()
        model.add_acquisition(a)
        model.add_acquisition(self.alex("b"))
        self.assertEqual(
            {c: s.emission for c, s in a.channel_specs.items()},
            {"DD": "D", "DA": "A", "AD": "D", "AA": "A"},
        )
        matrix = np.array([[1, 0, 2], [0, 1, -1], [0, 0, 1.0]])
        model.transforms.add_calibration(Calibration("optics", matrix))
        model.transforms.select("a", "DA", "optics")
        for channel in ("DA", "AA"):
            for frame in a.channels[channel].source_frames:
                np.testing.assert_array_equal(
                    model.transforms.matrix("a", channel, int(frame)), matrix
                )
        for channel in ("DD", "AD"):
            np.testing.assert_array_equal(
                model.transforms._registration("a", channel), np.eye(3)
            )
        model.transforms.select("b", "AA", "optics")
        np.testing.assert_array_equal(model.transforms._registration("b", "DA"), matrix)
        self.assertEqual(
            model.calibrations["optics"].metadata["selected_groups"],
            [
                {"acquisition_id": "a", "emission": "A", "channels": ["AA", "DA"]},
                {"acquisition_id": "b", "emission": "A", "channels": ["AA", "DA"]},
            ],
        )

    def test_named_calibration_reuse_checks_reference_geometry_and_pixel_scale(self):
        pixels = np.zeros((4, 16, 32), dtype=np.uint16)
        model = Dataset()
        for name, layout, pixel_size in (
            ("a", "donor-left", 100.0),
            ("b", "donor-left", 100.0),
            ("reversed", "acceptor-left", 100.0),
            ("scaled", "donor-left", 120.0),
        ):
            model.add_acquisition(
                image_acquisition(
                    ArraySource(pixels),
                    id=name,
                    mode="alex",
                    layout=layout,
                    pixel_size=pixel_size,
                )
            )
        metadata = {
            "source_emission": "A",
            "source_shape": [16, 16],
            "source_roi": [0, 16, 16, 32],
            "target_emission": "D",
            "target_shape": [16, 16],
            "target_roi": [0, 16, 0, 16],
            "pixel_size_nm": 100.0,
        }
        model.transforms.add_calibration(Calibration("optics", np.eye(3), metadata))
        model.transforms.select("a", "DA", "optics")
        model.transforms.select("b", "AA", "optics")
        for name in ("reversed", "scaled"):
            with (
                self.subTest(acquisition=name),
                self.assertRaisesRegex(ValueError, "geometry|pixel"),
            ):
                model.transforms.select(name, "DA", "optics")
        self.assertNotIn(("reversed", "AA"), model.transforms._selected)
        self.assertNotIn(("scaled", "DA"), model.transforms._selected)

    def test_projective_points_drift_inverse_sampling_and_export(self):
        model = Dataset()
        a = self.alex()
        model.add_acquisition(a)
        original = a.source[:].copy()
        matrix = np.array([[1.05, 0.02, 1], [-0.03, 0.98, 2], [0.008, -0.004, 1]])
        model.transforms.add_calibration(Calibration("optics", matrix))
        model.transforms.select("a", "DA", "optics")
        model.transforms.set_drift("a", np.tile([0.5, 1.5], (6, 1)))
        points = np.array([[3.0, 4.0], [9.0, 8.0]])
        ids = model.localizations.add(
            acquisition_id="a",
            channel="DA",
            frame=[0, 2],
            x=points[:, 0],
            y=points[:, 1],
        )
        rows = model.localizations.query(ids=ids, space="corrected")
        expected = projected(matrix, points) - [0.5, 1.5]
        np.testing.assert_allclose(np.column_stack([rows["x"], rows["y"]]), expected)
        x, y = model.transforms.convert_coordinates(
            rows, source_space="corrected", target_space="raw"
        )
        np.testing.assert_allclose(np.column_stack([x, y]), points)
        corrected = np.eye(3)
        corrected[:2, 2] = [-0.5, -1.5]
        np.testing.assert_allclose(
            model.transforms.matrix("a", "DA", 0), corrected @ matrix
        )
        raw = a.channels["AA"][0]
        yy, xx = np.indices(raw.shape)
        inverse = projected(
            np.linalg.inv(corrected @ matrix), np.column_stack([xx.ravel(), yy.ravel()])
        )
        expected_image = map_coordinates(
            raw.astype(float), [inverse[:, 1], inverse[:, 0]], order=1, mode="constant"
        ).reshape(raw.shape)
        np.testing.assert_allclose(model.transforms.image("a", "AA", 1), expected_image)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "aligned.tif"
            export_tiff(model, "a", "AA", path, space="corrected")
            np.testing.assert_allclose(tifffile.imread(path)[0], expected_image)
            path = Path(folder) / "optics.json"
            write_calibration(path, matrix)
            np.testing.assert_allclose(read_calibration(path), matrix)
        np.testing.assert_array_equal(a.source[:], original)
        np.testing.assert_array_equal(a.channels["AA"].source_frames, [1, 3, 5])

    def test_homography_ransac_rejects_outliers_deterministically(self):
        model = Dataset()
        a = self.alex()
        model.add_acquisition(a)
        rng = np.random.default_rng(7)
        source = rng.uniform(1, 13, (24, 2))
        matrix = np.array([[1.04, 0.03, 0.4], [-0.02, 0.96, 0.8], [0.006, -0.004, 1]])
        target = projected(matrix, source)
        target[-5:] = rng.uniform(1, 13, (5, 2))
        source_ids = model.localizations.add(
            acquisition_id="a", channel="DA", frame=0, x=source[:, 0], y=source[:, 1]
        )
        target_ids = model.localizations.add(
            acquisition_id="a", channel="DD", frame=0, x=target[:, 0], y=target[:, 1]
        )
        options = dict(
            source_ids=source_ids, target_ids=target_ids, model="homography", num_cpus=1
        )
        result = alignment.estimate_alignment(model, "a", "DA", "DD", **options)
        np.testing.assert_allclose(result.matrix, matrix, atol=1e-9)
        np.testing.assert_array_equal(result.inliers, np.arange(24) < 19)
        np.testing.assert_array_equal(
            alignment.estimate_alignment(model, "a", "DA", "DD", **options).matrix,
            result.matrix,
        )

    def test_translation_and_affine_ransac_reject_outliers_deterministically(self):
        rng = np.random.default_rng(43)
        source = rng.uniform(2, 25, (30, 2))
        matrices = {
            "translation": np.array(
                [[1.0, 0.0, 2.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]
            ),
            "affine": np.array(
                [[1.03, 0.04, 2.0], [-0.02, 0.97, -3.0], [0.0, 0.0, 1.0]]
            ),
        }
        for name, matrix in matrices.items():
            with self.subTest(model=name):
                target = projected(matrix, source)
                target[-6:] += [100.0, -100.0]
                first, residuals, inliers = alignment.fit_transform(
                    source, target, model=name, ransac_threshold=0.1
                )
                np.testing.assert_allclose(first, matrix, atol=1e-10)
                np.testing.assert_array_equal(inliers, np.arange(30) < 24)
                self.assertTrue(np.all(residuals[~inliers] > 100))
                second, _, second_inliers = alignment.fit_transform(
                    source, target, model=name, ransac_threshold=0.1
                )
                np.testing.assert_array_equal(second, first)
                np.testing.assert_array_equal(second_inliers, inliers)

    def test_projective_poles_are_nan_and_jacobian_is_local(self):
        from molcore.transforms import project_jacobian, project_points

        matrix = np.array([[1.0, 0, 0], [0, 1, 0], [-0.1, 0, 1]])
        self.assertTrue(np.isnan(project_points(matrix, [[10.0, 2.0]])).all())
        point = np.array([[3.0, 4.0]])
        numerical = np.column_stack(
            [
                (projected(matrix, point + delta)[0] - projected(matrix, point)[0])
                / 1e-6
                for delta in ([1e-6, 0], [0, 1e-6])
            ]
        )
        np.testing.assert_allclose(
            project_jacobian(matrix, point)[0], numerical, rtol=1e-6
        )

    def test_projective_photometry_tracks_inverse_positions_and_original_frames(self):
        from molcore.photometry import stationary_traces

        model = Dataset()
        model.add_acquisition(self.alex())
        matrix = np.array([[1.05, 0.02, 1], [-0.03, 0.98, 2], [0.008, -0.004, 1]])
        model.transforms.add_calibration(Calibration("optics", matrix))
        model.transforms.select("a", "DA", "optics")
        model.transforms.set_drift(
            "a", np.column_stack([np.arange(6) * 0.2, np.zeros(6)])
        )
        ids = model.localizations.add(
            acquisition_id="a", channel="DD", frame=0, x=7.0, y=8.0
        )
        result = stationary_traces(model, ids, channels=["AA"], apply=False, num_cpus=1)
        frames = np.array([1, 3, 5])
        expected = projected(
            np.linalg.inv(matrix), np.column_stack([7 + frames * 0.2, np.full(3, 8)])
        )
        np.testing.assert_allclose(
            np.column_stack([result.columns["x"], result.columns["y"]]), expected
        )
        np.testing.assert_array_equal(result.columns["frame"], frames)

    def test_projective_rendering_propagates_position_dependent_uncertainty(self):
        from molcore.rendering import render_localizations
        from molcore.types import RenderSettings

        model = Dataset()
        model.add_acquisition(self.alex())
        matrix = np.array([[1.05, 0.02, 1], [-0.03, 0.98, 2], [0.008, -0.004, 1]])
        model.transforms.add_calibration(Calibration("optics", matrix))
        model.transforms.select("a", "DA", "optics")
        points = np.array([[3.0, 4.0], [9.0, 8.0]])
        model.localizations.add(
            acquisition_id="a",
            channel="DA",
            frame=0,
            x=points[:, 0],
            y=points[:, 1],
            lpx=0.2,
            lpy=0.4,
        )
        result = render_localizations(
            model,
            "a",
            "DA",
            space="registered",
            settings=RenderSettings(blur_method="gaussian"),
        )
        for index, point in enumerate(points):
            jacobian = np.column_stack(
                [
                    (
                        projected(matrix, [point + delta])[0]
                        - projected(matrix, [point])[0]
                    )
                    / 1e-6
                    for delta in ([1e-6, 0], [0, 1e-6])
                ]
            )
            expected = np.sqrt(jacobian**2 @ [0.2**2, 0.4**2])
            np.testing.assert_allclose(
                [result.columns["lpx"][index], result.columns["lpy"][index]],
                expected,
                rtol=1e-6,
            )
