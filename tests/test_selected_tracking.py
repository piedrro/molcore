"""Explicit tracking scope preserves independent imported memberships."""

import threading
import unittest
from concurrent.futures import CancelledError
from dataclasses import replace

import numpy as np
from scientific_helpers import compute_scenario
from test_processor_backends import InlineExecutor, TrackBackend

from molcore import Acquisition, ArraySource, Dataset, StaleResultError, tracking
from molcore.backends import TrackingOutput
from molcore.types import TrackingSettings


class SelectedTrackingTests(unittest.TestCase):
    def setUp(self):
        self.model = Dataset()
        self.model.add_acquisition(Acquisition(ArraySource(np.ones((4, 9, 9))), id="a"))
        self.model.add_acquisition(Acquisition(ArraySource(np.ones((4, 9, 9))), id="b"))
        self.ids = self.model.localizations.add(
            acquisition_id="a", channel="image", frame=[0, 1, 2, 3], x=4, y=4
        )
        self.other = self.model.localizations.add(acquisition_id="b", channel="image", frame=[0], x=4, y=4)

    def run_tracking(self, ids, **kwargs):
        return compute_scenario(
            tracking.track,
            self.model,
            "a",
            "image",
            ids=ids,
            settings=TrackingSettings(),
            cpu=InlineExecutor(),
            backend=TrackBackend(),
            **kwargs,
        )

    def test_selected_preview_and_apply_preserve_disjoint_import(self):
        imported = self.model.tracks.add(self.ids[2:], metadata={"import": "user"})
        selected = self.model.tracks.add(self.ids[:2])
        preview = self.run_tracking(self.ids[:2], apply=False)
        np.testing.assert_array_equal(preview.memberships[0], self.ids[:2])
        np.testing.assert_array_equal(preview.selected_ids, self.ids[:2])
        self.assertEqual(set(self.model.tracks.query()["track_id"]), {selected, imported})
        applied = self.model.apply(preview)
        np.testing.assert_array_equal(self.model.tracks.query(track_id=imported)["id"], self.ids[2:])
        self.assertEqual(self.model.tracks.metadata[imported]["parameters"], {"import": "user"})
        self.assertEqual(set(self.model.tracks.query()["track_id"]), {imported, *applied.track_ids})

    def test_invalid_selection_is_rejected_before_computation(self):
        for ids in ([self.ids[0], self.ids[0]], [999], self.other, [[int(self.ids[0])]], [1.5]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.run_tracking(ids, apply=False)

    def test_partial_track_overlap_rejected_atomically(self):
        original = self.model.tracks.add(self.ids)
        preview = self.run_tracking(self.ids[:2], apply=False)
        revision = self.model.tracks.revision
        with self.assertRaises(ValueError):
            self.model.apply(preview)
        np.testing.assert_array_equal(self.model.tracks.query(track_id=original)["id"], self.ids)
        self.assertEqual(self.model.tracks.revision, revision)

    def test_empty_selection_leaves_all_tracks_untouched(self):
        original = self.model.tracks.add(self.ids)
        preview = self.run_tracking([], apply=False)
        self.assertEqual(preview.memberships, ())
        applied = self.model.apply(preview)
        self.assertEqual(len(applied.track_ids), 0)
        np.testing.assert_array_equal(self.model.tracks.query(track_id=original)["id"], self.ids)

    def test_backend_or_detached_memberships_cannot_escape_selected_scope(self):

        class EscapingBackend(TrackBackend):
            def compute(inner, payload, reporter):
                return TrackingOutput((self.ids.copy(),))

        with self.assertRaises(ValueError):
            compute_scenario(
                tracking.track,
                self.model,
                "a",
                "image",
                ids=self.ids[:2],
                apply=False,
                cpu=InlineExecutor(),
                backend=EscapingBackend(),
            )
        preview = self.run_tracking(self.ids[:2], apply=False)
        with self.assertRaises(ValueError):
            self.model.apply(replace(preview, memberships=(self.ids,)))
        self.assertEqual(len(self.model.tracks), 0)

    def test_scope_replacement_public_api_and_empty_memberships_clear_only_scope(self):
        selected = self.model.tracks.add(self.ids[:2])
        imported = self.model.tracks.add(self.ids[2:])
        self.model.tracks.replace("a", "image", [], ids=self.ids[:2])
        self.assertNotIn(selected, self.model.tracks.query()["track_id"])
        np.testing.assert_array_equal(self.model.tracks.query(track_id=imported)["id"], self.ids[2:])

    def test_selected_results_still_validate_freshness_and_cancellation(self):
        preview = self.run_tracking(self.ids[:2], apply=False)
        self.model.localizations.move_many(self.ids[:1], x=5, y=4)
        with self.assertRaises(StaleResultError):
            self.model.apply(preview)
        event = threading.Event()
        event.set()
        with self.assertRaises(CancelledError):
            self.run_tracking(self.ids[:2], apply=False, cancel_event=event)

    def test_selected_preview_rejects_source_change_during_computation(self):

        def edit(_):
            self.model.localizations.move_many(self.ids[:1], x=5, y=4)

        with self.assertRaises(StaleResultError):
            self.run_tracking(self.ids[:2], apply=False, progress=edit)
