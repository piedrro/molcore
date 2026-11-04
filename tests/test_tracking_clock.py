"""Tracking memory can follow camera frames or the logical channel clock."""

import unittest

import numpy as np
from test_processor_backends import InlineExecutor

from molcore import Acquisition, ArraySource, ChannelSpec, Dataset, tracking
from molcore.types import TrackingSettings


def fixture(sequence, count):
    model = Dataset()
    model.add_acquisition(
        Acquisition(
            ArraySource(np.ones((count, 9, 9))),
            id="a",
            excitation_sequence=sequence,
            channel_specs={"D": ChannelSpec(excitation="D")},
        )
    )
    frames = model.acquisitions["a"].channels["D"].source_frames
    ids = model.localizations.add(
        acquisition_id="a",
        channel="D",
        frame=frames,
        x=4,
        y=4,
    )
    return model, ids


class TrackingClockTests(unittest.TestCase):
    def link(self, model, ids, **settings):
        return tracking.track(
            model,
            "a",
            "D",
            ids=ids,
            settings=TrackingSettings(**settings),
            cpu=InlineExecutor(),
            apply=False,
        )

    def test_regular_alex_channel_clock_links_without_camera_gap_memory(self):
        model, ids = fixture(("D", "A"), 6)
        result = self.link(model, ids, frame_clock="channel", memory=0)
        self.assertEqual(len(result.memberships), 1)
        np.testing.assert_array_equal(result.memberships[0], ids)
        np.testing.assert_array_equal(
            model.localizations.query(space="raw")["frame"], [0, 2, 4]
        )

    def test_irregular_excitation_channel_clock_counts_logical_frames(self):
        model, ids = fixture(("D", "D", "A"), 6)
        source_frames = model.acquisitions["a"].channels["D"].source_frames.copy()
        revision = model.revision
        result = self.link(model, ids, frame_clock="channel", memory=0)
        self.assertEqual(len(result.memberships), 1)
        np.testing.assert_array_equal(result.memberships[0], ids)
        np.testing.assert_array_equal(result.selected_ids, ids)
        np.testing.assert_array_equal(source_frames, [0, 1, 3, 4])
        np.testing.assert_array_equal(
            model.localizations.query(space="raw")["frame"], [0, 1, 3, 4]
        )
        np.testing.assert_array_equal(
            model.acquisitions["a"].channels["D"].source_frames, source_frames
        )
        self.assertEqual(model.revision, revision)
        self.assertEqual(len(model.tracks), 0)
        applied = tracking.apply_tracking(model, result)
        np.testing.assert_array_equal(
            model.tracks.query(track_id=applied.track_ids)["frame"], [0, 1, 3, 4]
        )
        self.assertEqual(
            model.processing.settings["tracking"]["a"]["D"]["settings"]["frame_clock"],
            "channel",
        )

    def test_channel_memory_one_bridges_one_missing_logical_frame(self):
        for sequence, count in ((("D", "A"), 6), (("D", "D", "A"), 6)):
            with self.subTest(sequence=sequence):
                model, ids = fixture(sequence, count)
                selected = ids[[0, 2]]
                without_memory = self.link(
                    model, selected, frame_clock="channel", memory=0
                )
                self.assertEqual(len(without_memory.memberships), 2)
                result = self.link(model, selected, frame_clock="channel", memory=1)
                self.assertEqual(len(result.memberships), 1)
                np.testing.assert_array_equal(result.memberships[0], selected)

    def test_default_camera_clock_keeps_camera_gap_behavior(self):
        for sequence, count, memberships in (
            (("D", "A"), 6, [[0], [1], [2]]),
            (("D", "D", "A"), 6, [[0, 1], [2, 3]]),
        ):
            with self.subTest(sequence=sequence):
                model, ids = fixture(sequence, count)
                result = self.link(model, ids, memory=0)
                self.assertEqual(len(result.memberships), len(memberships))
                for actual, indices in zip(
                    result.memberships, memberships, strict=True
                ):
                    np.testing.assert_array_equal(actual, ids[indices])

    def test_invalid_frame_clock_is_rejected(self):
        for value in ("", "logical", None, True, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrackingSettings(frame_clock=value)
