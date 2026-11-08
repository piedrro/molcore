"""Drift estimators consume explicit stable localization selections."""

import unittest
from unittest.mock import patch

import numpy as np
from scientific_helpers import compute_scenario
from test_processor_backends import InlineExecutor

from molcore import Acquisition, ArraySource, Dataset, StaleResultError, undrifting
from molcore.types import DriftSettings


def estimate_from_x(payload, reporter):
    columns, info, settings = payload
    reporter.check_cancelled()
    return np.column_stack([columns["x"], np.zeros(info[0]["Frames"])])


class SelectedDriftTests(unittest.TestCase):
    def setUp(self):
        self.model = Dataset()
        self.model.add_acquisition(Acquisition(ArraySource(np.zeros((8, 9, 9))), id="a"))
        self.model.add_acquisition(Acquisition(ArraySource(np.zeros((8, 9, 9))), id="b"))
        self.ids = self.model.localizations.add(
            acquisition_id="a", channel="image", frame=np.arange(8), x=np.arange(8), y=2
        )
        self.imported = self.model.localizations.add(
            acquisition_id="a", channel="image", frame=np.arange(8), x=100, y=2
        )
        self.other = self.model.localizations.add(acquisition_id="b", channel="image", frame=[0], x=4, y=4)
        self.version = patch.object(undrifting, "backend_version", return_value="0.11.3")
        self.worker = patch.object(undrifting, "_job", side_effect=estimate_from_x)
        self.version.start()
        self.worker.start()
        self.addCleanup(self.version.stop)
        self.addCleanup(self.worker.stop)

    def estimate(self, ids, **kwargs):
        return compute_scenario(
            undrifting.estimate_drift,
            self.model,
            "a",
            "image",
            ids=ids,
            settings=DriftSettings(segment_frames=2),
            cpu=InlineExecutor(),
            **kwargs,
        )

    def test_estimation_uses_only_selected_registered_positions(self):
        result = self.estimate(self.ids)
        np.testing.assert_array_equal(result.displacement[:, 0], np.arange(8))
        np.testing.assert_array_equal(result.frames, np.arange(8))
        np.testing.assert_array_equal(self.model.localizations.query(ids=self.imported)["x"], [100] * 8)
        self.assertFalse(result.applied)

    def test_invalid_or_empty_selection_is_rejected(self):
        for ids in ([self.ids[0], self.ids[0]], [999], self.other, [], [1.5]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.estimate(ids)

    def test_selected_estimation_retains_source_freshness_validation(self):

        def edit(_):
            self.model.localizations.move(self.ids[0], x=1, y=2)

        with self.assertRaises(StaleResultError):
            self.estimate(self.ids, progress=edit)

    def test_undrift_applies_selected_estimate_at_acquisition_scope(self):
        result = compute_scenario(
            undrifting.undrift,
            self.model,
            "a",
            "image",
            ids=self.ids,
            settings=DriftSettings(segment_frames=2),
            cpu=InlineExecutor(),
        )
        self.assertTrue(result.applied)
        np.testing.assert_array_equal(
            self.model.localizations.query(ids=self.imported, space="corrected")["x"],
            [100, 99, 98, 97, 96, 95, 94, 93],
        )
