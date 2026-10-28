"""Embeddable widget workflows use typed settings and preserve preview boundaries."""

import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtWidgets import QApplication

from molcore import Acquisition, ArraySource, Dataset
from molcore.types import DetectionResult, LocalisationSettings
from molcore.ui import (
    ChannelAlignmentWidget,
    FileToolsWidget,
    LocalisationWidget,
    MolCoreToolsWidget,
    SegmentationWidget,
    UiSession,
    UndriftingWidget,
)


class WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        model = Dataset()
        model.add_acquisition(Acquisition(ArraySource(np.zeros((3, 9, 9))), id="a"))
        self.session = UiSession(model)
        self.addCleanup(self.session.shutdown)

    def pump(self):
        deadline = time.monotonic() + 5
        while self.session.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertFalse(self.session.busy)

    def test_all_widgets_embed_without_napari(self):
        for cls in (
            FileToolsWidget,
            LocalisationWidget,
            UndriftingWidget,
            SegmentationWidget,
            ChannelAlignmentWidget,
            MolCoreToolsWidget,
        ):
            widget = cls(self.session)
            self.assertIs(widget.session, self.session)
            widget.close()

    def test_full_detection_cached_and_settings_change_invalidates(self):
        widget = LocalisationWidget(self.session)

        def detect(model, acquisition_id, channel, **kwargs):
            self.assertEqual(kwargs["num_cpus"], 2)
            return DetectionResult(
                acquisition_id,
                channel,
                {
                    "frame": np.array([0]),
                    "x": np.array([4.0]),
                    "y": np.array([4.0]),
                    "net_gradient": np.array([1001.0]),
                },
                LocalisationSettings(),
                0,
            )

        self.session.num_cpus = 2
        with patch("molcore.localisation.detect", side_effect=detect):
            widget.detect_full()
            self.pump()
        self.assertIsNotNone(widget.detections)
        self.assertEqual(len(self.session.dataset.localizations), 0)
        widget.threshold.setValue(widget.threshold.value() + 1)
        self.assertIsNone(widget.detections)
        widget.close()

    def test_file_dialog_cancel_does_not_start_job(self):
        widget = FileToolsWidget(self.session)
        model = self.session.dataset
        with patch("qtpy.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")):
            widget.open_dataset()
        self.assertFalse(self.session.busy)
        self.assertIs(model, self.session.dataset)
        widget.close()


if __name__ == "__main__":
    unittest.main()
