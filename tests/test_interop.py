import tempfile
import unittest
from pathlib import Path

import numpy as np

from molcore import Acquisition, ArraySource, Dataset


class InteropTests(unittest.TestCase):
    def test_picasso_and_numpy_roundtrip_preserves_extra_columns(self):
        from molcore.interop import read_picasso, to_records, write_picasso

        values = {
            "frame": np.array([0, 2], dtype=np.uint32),
            "x": np.array([1.0, 2.0]),
            "y": np.array([3.0, 4.0]),
            "photons": np.array([10.0, 20.0]),
            "custom_measurement": np.array([7, 9], dtype=np.int16),
        }
        metadata = [{"Width": 9, "Height": 9, "Frames": 3}, {"method": "test"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "localizations.hdf5"
            write_picasso(path, values, metadata)
            restored, info = read_picasso(path)
            self.assertEqual(info, metadata)
            self.assertEqual(restored["custom_measurement"].dtype, np.dtype("int16"))
            np.testing.assert_array_equal(restored["custom_measurement"], [7, 9])
            np.testing.assert_array_equal(to_records(restored)["frame"], [0, 2])

    def test_napari_arrays_use_yx_and_explicit_sequence_timeline(self):
        from molcore.interop import napari_points, napari_tracks

        exp = Dataset()
        for name in ("a", "b"):
            exp.add_acquisition(Acquisition(ArraySource(np.zeros((3, 9, 9))), id=name))
        seq = exp.add_sequence(["a", "b"], id="s", allow_cross_boundary_tracks=True, reference="camera")
        ids = exp.localizations.add(
            acquisition_id=["a", "b"], channel="image", frame=[2, 0], x=[1.0, 2.0], y=[3.0, 4.0]
        )
        exp.tracks.add(ids, sequence_id=seq.id)
        rows = exp.localizations.query(sequence_id="s")
        np.testing.assert_array_equal(
            napari_points(rows, frame_column="sequence_frame"), [[2, 3, 1], [3, 4, 2]]
        )
        np.testing.assert_array_equal(
            napari_tracks(exp.tracks.query(sequence_id="s"), frame_column="sequence_frame"),
            [[1, 2, 3, 1], [1, 3, 4, 2]],
        )
        with self.assertRaises(ValueError):
            napari_points(exp.localizations.query())
