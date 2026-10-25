"""File interchange tests protect frame provenance and previous destinations."""

import pickle
import tempfile
import threading
import unittest
from concurrent.futures import CancelledError
from pathlib import Path

import numpy as np
import tifffile

from molcore import Acquisition, ArraySource, ChannelSpec, Experiment
from molcore.files import export_tiff, import_table, read_calibration, read_table, write_table
from molcore.legacy import read_moltrack


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = Experiment()
        self.model.add_acquisition(Acquisition(
            ArraySource(np.arange(5 * 6 * 8).reshape(5, 6, 8).astype('uint16')), id='a',
            channel_specs={'DD': ChannelSpec(excitation='D'), 'AA': ChannelSpec(excitation='A')},
            excitation_sequence=('D', 'A'), phase=1,
        ))

    def test_pos_out_one_based_channel_frame_import(self):
        path = self.root / 'locs.pos.out'
        path.write_text('FRAME\tXCENTER\tYCENTER\tBRIGHTNESS\n1\t3\t4\t100\n2\t4\t4\t200\n')
        data = read_table(path)
        import_table(self.model, 'a', 'DD', data, timeline='channel', frame_base=1)
        rows = self.model.localizations.query()
        np.testing.assert_array_equal(rows['frame'], [1, 3])
        np.testing.assert_array_equal(rows['photons'], [100, 200])

    def test_track_import_rolls_back_invalid_membership(self):
        rows = {'frame': np.array([0, 0]), 'x': np.array([2, 3]), 'y': np.array([2, 3]),
                'particle': np.array([7, 7])}
        with self.assertRaises(ValueError):
            import_table(self.model, 'a', 'AA', rows, timeline='channel', tracks=True)
        self.assertEqual(len(self.model.localizations), 0)

    def test_csv_extra_measurements_roundtrip(self):
        path = self.root / 'locs.csv'
        columns = {'frame': np.array([0]), 'x': np.array([2.5]), 'y': np.array([3.0]),
                   'note': np.array(['a,b']), 'photons': np.array([42.5])}
        write_table(path, columns)
        loaded = read_table(path)
        for name in columns:
            np.testing.assert_array_equal(loaded[name], columns[name])

    def test_tiff_stream_and_cancel_preserve_destination(self):
        path = self.root / 'movie.tif'
        export_tiff(self.model, 'a', 'DD', path)
        np.testing.assert_array_equal(tifffile.imread(path), self.model.acquisitions['a'].channels['DD'][:])
        previous = path.read_bytes()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            export_tiff(self.model, 'a', 'DD', path, cancel_event=cancel)
        self.assertEqual(path.read_bytes(), previous)

    def test_cancel_save_before_replace(self):
        path = self.root / 'exp.h5'
        self.model.save(path)
        previous = path.read_bytes()
        cancel = threading.Event()

        def progress(update):
            cancel.set()

        with self.assertRaises(CancelledError):
            self.model.save(path, image_policy='full', progress=progress, cancel_event=cancel)
        self.assertEqual(path.read_bytes(), previous)

    def test_reject_projective_and_unsafe_pickle(self):
        path = self.root / 'matrix.txt'
        path.write_text('[[1,0,0],[0,1,0],[0.1,0,1]]')
        with self.assertRaisesRegex(ValueError, 'affine|projective'):
            read_calibration(path)
        path = self.root / 'bad.moltrack'
        path.write_bytes(pickle.dumps(Path('unsafe')))
        with self.assertRaisesRegex(ValueError, 'Unsupported|unsupported'):
            read_moltrack(path)


if __name__ == '__main__':
    unittest.main()
