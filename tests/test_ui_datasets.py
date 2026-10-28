"""Cross-round session selection and preview/apply integration."""

import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtWidgets import QApplication

from molcore import Acquisition, ArraySource, Dataset, Experiment
from molcore.ui import UiSession


class DatasetUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_session(self):
        experiment = Experiment()
        for name, dx in (("first", 0), ("second", 2)):
            dataset = Dataset(id=name)
            dataset.add_acquisition(Acquisition(ArraySource(np.zeros((2, 20, 20))), id="movie"))
            dataset.localizations.add(
                acquisition_id="movie",
                channel="image",
                frame=0,
                x=np.array([4, 12, 5, 13]) + dx,
                y=[4, 5, 12, 13],
            )
            experiment.add_dataset(dataset)
        return UiSession(experiment)

    def test_selection_operates_on_one_round_and_preserves_others(self):
        session = self.make_session()
        self.assertEqual(session.dataset_id, "first")
        session.select_dataset("second")
        session.dataset.localizations.move(1, x=10, y=10)
        self.assertEqual(session.experiment.datasets["first"].localizations.query()["x"][0], 4)
        self.assertEqual(session.acquisition_id, "movie")
        session.shutdown()

    def test_optical_widget_defaults_to_acceptor_and_applies_all_rounds(self):
        from molcore.files import image_acquisition
        from molcore.ui import ChannelAlignmentWidget

        experiment = Experiment()
        for name in ("first", "second"):
            dataset = Dataset(id=name)
            dataset.add_acquisition(
                image_acquisition(ArraySource(np.zeros((4, 20, 40))), id="movie", mode="alex")
            )
            for channel, dx in (("DA", 2), ("DD", 0)):
                dataset.localizations.add(
                    acquisition_id="movie",
                    channel=channel,
                    frame=0,
                    x=np.array([4, 12, 5, 13]) + dx,
                    y=[4, 5, 12, 13],
                )
            experiment.add_dataset(dataset)
        session = UiSession(experiment)
        widget = ChannelAlignmentWidget(session)
        self.assertTrue(
            hasattr(widget, "source"), "Optical source must be selectable independently of displayed channel"
        )
        self.assertEqual(widget.source.currentText(), "DA")
        self.assertEqual(widget.target.currentText(), "DD")
        widget.source_ids.setText("1 2 3 4")
        widget.target_ids.setText("5 6 7 8")
        widget.estimate()
        deadline = time.monotonic() + 10
        while session.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertIsNotNone(widget.preview, widget.status.text())
        widget.apply_preview()
        for dataset in experiment.datasets.values():
            self.assertAlmostEqual(dataset.transforms.matrix("movie", "AA", 1)[0, 2], -2)
        widget.close()
        session.shutdown()

    def test_empty_experiment_cannot_import_into_unregistered_round(self):
        from molcore.ui import FileToolsWidget

        session = UiSession(Experiment())
        widget = FileToolsWidget(session)
        widget.import_into_current.setChecked(True)
        with (
            patch("molcore.ui.files.QFileDialog.getOpenFileNames", return_value=([], "")),
            self.assertRaisesRegex(ValueError, "Select a dataset"),
        ):
            widget.import_images()
        widget.close()
        session.shutdown()

    def test_live_edit_rejects_io_before_publishing_files(self):
        from molcore.exports import export_dataset_images

        for operation in ("export", "save"):
            with self.subTest(operation=operation), TemporaryDirectory() as directory:
                session = self.make_session()
                path = Path(directory) / ("manifest.json" if operation == "export" else "experiment.h5")
                path.write_bytes(b"previous output")
                gate = threading.Event()
                errors = []
                session.error.connect(errors.append)

                def compute(
                    model, progress, cancel, publish_guard, gate=gate, operation=operation, path=path
                ):
                    gate.wait(3)
                    if operation == "export":
                        export_dataset_images(
                            model.datasets["first"], directory, experiment=model, publish_guard=publish_guard
                        )
                    else:
                        model.save(path, image_policy="full", publish_guard=publish_guard)

                session.submit(compute, experiment_scope=True, publication=True)
                session.dataset.localizations.move(1, x=9, y=9)
                gate.set()
                deadline = time.monotonic() + 10
                while session.busy and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.app.processEvents()
                self.assertTrue(errors)
                self.assertEqual(path.read_bytes(), b"previous output")
                session.shutdown()

    def test_save_before_replace_preserves_edits_after_publication(self):
        from molcore.ui import FileToolsWidget

        session = self.make_session()
        widget = FileToolsWidget(session)
        published, resume = threading.Event(), threading.Event()
        replacements = []
        original_submit = session.submit

        def submit(compute, **options):
            def paused(*args):
                result = compute(*args)
                published.set()
                resume.wait(3)
                return result

            return original_submit(paused, **options)

        with (
            TemporaryDirectory() as directory,
            patch.object(session, "submit", side_effect=submit),
            patch(
                "molcore.ui.files.QFileDialog.getSaveFileName",
                return_value=(str(Path(directory) / "saved.h5"), ""),
            ),
        ):
            widget.save_experiment(on_saved=lambda: replacements.append("replaced"))
            self.assertTrue(published.wait(3))
            session.dataset.localizations.move(1, x=9, y=9)
            resume.set()
            deadline = time.monotonic() + 10
            while session.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.app.processEvents()
            self.app.processEvents()
            self.assertFalse(replacements)
            self.assertTrue(session.dirty)
        widget.close()
        session.shutdown()

    def test_alignment_widget_preview_apply_and_reset(self):
        from molcore.ui import DatasetAlignmentWidget

        session = self.make_session()
        widget = DatasetAlignmentWidget(session)
        widget.reference_dataset.setCurrentText("first")
        widget.source_dataset.setCurrentText("second")
        widget.source_ids.setText("1 2 3 4")
        widget.reference_ids.setText("1 2 3 4")
        widget.estimate()
        deadline = time.monotonic() + 20
        while session.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertIsNotNone(widget.preview, widget.status.text())
        widget.apply_preview()
        self.assertIn("second", session.experiment.alignment.records)
        widget.reset_alignment()
        self.assertEqual(dict(session.experiment.alignment.records), {})
        widget.close()
        session.shutdown()
