import pickle
import unittest

import numpy as np

from molcore import Acquisition, AcquisitionSequence, ArraySource, ChannelSpec, Experiment


class ChannelTests(unittest.TestCase):
    def test_standard_channel_enums_and_custom_names(self):
        from molcore import AcquisitionMode, ChannelName

        a = Acquisition(
            ArraySource(np.zeros((2, 3, 4))),
            mode=AcquisitionMode.FRET,
            channel_specs={ChannelName.DONOR: ChannelSpec(), "custom": ChannelSpec()},
        )
        self.assertEqual(a.mode, AcquisitionMode.FRET)
        self.assertIs(a.channels["donor"], a.channels[ChannelName.DONOR])
        self.assertIn("custom", a.channels)

    def test_alex_mapping_keeps_odd_tail_and_original_frames(self):
        frames = np.arange(5 * 4 * 6).reshape(5, 4, 6)
        acquisition = Acquisition(
            ArraySource(frames),
            id="a",
            pixel_size=100,
            channel_specs={
                "DD": ChannelSpec(roi=(0, 4, 0, 3), excitation="D"),
                "DA": ChannelSpec(roi=(0, 4, 3, 6), excitation="D"),
                "AA": ChannelSpec(roi=(0, 4, 3, 6), excitation="A"),
            },
            excitation_sequence=("D", "A"),
        )
        self.assertEqual(acquisition.channels["DD"].shape, (3, 4, 3))
        self.assertEqual(acquisition.channels["AA"].source_frames.tolist(), [1, 3])
        np.testing.assert_array_equal(acquisition.channels["DD"][2], frames[4, :, :3])
        np.testing.assert_array_equal(acquisition.channels["AA"][::-1], frames[[3, 1], :, 3:])
        self.assertEqual(acquisition.channels["AA"].frame_info(1).camera_frame, 3)
        with self.assertRaises(IndexError):
            acquisition.channels["AA"][2]

    def test_sequence_is_lazy_and_retains_boundaries(self):
        a = Acquisition(ArraySource(np.ones((2, 3, 3))), id="a", pixel_size=100)
        b = Acquisition(ArraySource(np.full((3, 3, 3), 2)), id="b", pixel_size=100)
        sequence = AcquisitionSequence([a, b], id="s")
        self.assertEqual(sequence.channels["image"].shape, (5, 3, 3))
        self.assertEqual(sequence.channels["image"].frame_info(2).acquisition_id, "b")
        self.assertEqual(sequence.channels["image"].frame_info(2).camera_frame, 0)
        self.assertEqual(sequence.channels["image"].frame_info(2).sequence_frame, 2)
        np.testing.assert_array_equal(sequence.channels["image"][1:4, 0, 0], [1, 2, 2])
        self.assertIsNone(sequence.channels["image"].frame_info(2).timestamp)
        with self.assertRaises(ValueError):
            AcquisitionSequence([a, Acquisition(ArraySource(np.zeros((2, 4, 4))))])

    def test_public_types_and_array_input_are_detached(self):
        from molcore.types import ChannelSpec, ImagePolicy

        self.assertEqual(ImagePolicy.FULL.value, "full")
        self.assertEqual(ChannelSpec().frame_step, 1)
        restored = pickle.loads(pickle.dumps(ArraySource(np.zeros((2, 3, 4)))))
        self.assertEqual(restored.shape, (2, 3, 4))


class TableTests(unittest.TestCase):
    def setUp(self):
        self.exp = Experiment()
        self.exp.add_acquisition(Acquisition(ArraySource(np.zeros((5, 9, 9))), id="a"))
        self.ids = self.exp.localizations.add(
            acquisition_id=["a"] * 3,
            channel=["image"] * 3,
            frame=[0, 1, 2],
            x=[1.0, 2.0, 3.0],
            y=[2.0, 3.0, 4.0],
            photons=[10.0, 20.0, 30.0],
        )

    def test_queries_edits_keep_originals_and_invalidate_observations(self):
        selected = self.exp.localizations.query(acquisition_id="a", frame_range=(1, 3))
        self.assertEqual(selected["id"].tolist(), self.ids[1:].tolist())
        self.exp.observations.add(
            molecule_id=[1],
            acquisition_id=["a"],
            channel=["image"],
            frame=[1],
            intensity=[20.0],
            localization_id=[int(self.ids[1])],
        )
        self.exp.localizations.move(self.ids[1], x=7, y=6)
        self.assertEqual(self.exp.localizations.query(ids=[self.ids[1]])["x"].tolist(), [7.0])
        self.assertEqual(self.exp.localizations.query(ids=[self.ids[1]], original=True)["x"].tolist(), [2.0])
        self.assertEqual(len(self.exp.observations.query()["id"]), 0)
        self.assertEqual(len(self.exp.observations.query(include_stale=True)["id"]), 1)
        selected["x"][0] = 100
        self.assertEqual(self.exp.localizations.query(ids=[self.ids[1]])["x"].tolist(), [7.0])

    def test_track_membership_split_merge_and_deletion(self):
        track = self.exp.tracks.add(self.ids)
        split = self.exp.tracks.split(track, frame=2)
        self.assertEqual(self.exp.tracks.query(track_id=split)["frame"].tolist(), [2])
        self.exp.tracks.merge([track, split])
        self.exp.localizations.delete([self.ids[1]])
        self.assertEqual(self.exp.tracks.query(track_id=track)["frame"].tolist(), [0, 2])
        with self.assertRaises(ValueError):
            self.exp.tracks.add([self.ids[0]])

    def test_invalid_edit_is_atomic_and_ids_are_not_reused(self):
        before = self.exp.localizations.query()
        with self.assertRaises(ValueError):
            self.exp.localizations.add(acquisition_id=["a"], channel=["image"], frame=[100], x=[0], y=[0])
        np.testing.assert_array_equal(self.exp.localizations.query()["id"], before["id"])
        self.exp.localizations.delete(self.ids)
        added = self.exp.localizations.add(acquisition_id="a", channel="image", frame=0, x=1, y=2)
        self.assertGreater(int(added[0]), int(self.ids[-1]))


if __name__ == "__main__":
    unittest.main()
