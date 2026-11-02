"""Qt session tests run with a host-owned offscreen application."""

import os
import threading
import time
import unittest
from concurrent.futures import CancelledError

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtCore import QThread
from qtpy.QtWidgets import QApplication

from molcore import Acquisition, ArraySource, Dataset
from molcore.staging import clone_dataset
from molcore.ui import UiSession


def dataset():
    result = Dataset()
    result.add_acquisition(Acquisition(ArraySource(np.zeros((3, 9, 9))), id="a"))
    return result


class SessionTests(unittest.TestCase):
    def test_model_mutation_refreshes_session_and_replacement_unsubscribes(self):
        original = dataset()
        session = UiSession(original)
        changes = []
        session.results_changed.connect(lambda: changes.append(True))
        original.localizations.add(
            acquisition_id="a", channel="image", frame=[0], x=[2], y=[2]
        )
        self.app.processEvents()
        self.assertEqual(changes, [True])
        self.assertTrue(session.dirty)
        session.set_dataset(dataset())
        original.localizations.move(1, x=3, y=3)
        self.app.processEvents()
        self.assertEqual(changes, [True])
        self.assertFalse(session.dirty)
        session.shutdown()

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def pump(self, session):
        deadline = time.monotonic() + 10
        while session.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertFalse(session.busy)

    def test_clone_detaches_results_but_keeps_lazy_sources(self):
        original = dataset()
        original.localizations.add(
            acquisition_id="a", channel="image", frame=[0], x=[3], y=[3]
        )
        frozen = clone_dataset(original)
        self.assertIs(
            frozen.acquisitions["a"].source, original.acquisitions["a"].source
        )
        frozen.localizations.move(1, x=4, y=4)
        self.assertEqual(original.localizations.query()["x"][0], 3)
        self.assertIs(frozen.localizations.dataset, frozen)

    def test_background_computation_and_gui_commit(self):
        session = UiSession(dataset())
        seen = []

        def compute(model, progress, cancel):
            self.assertIsNot(QThread.currentThread(), self.app.thread())
            return 42

        def commit(model, result):
            seen.append((result, QThread.currentThread() is self.app.thread()))

        session.submit(compute, commit=commit)
        self.pump(session)
        self.assertEqual(seen, [(42, True)])
        session.shutdown()

    def test_external_edits_reject_completion(self):
        model = dataset()
        session = UiSession(model)
        gate = threading.Event()
        errors, commits = [], []
        session.error.connect(errors.append)
        session.submit(
            lambda m, p, c: gate.wait(2), commit=lambda m, r: commits.append(r)
        )
        model.localizations.add(
            acquisition_id="a", channel="image", frame=[0], x=[2], y=[2]
        )
        gate.set()
        self.pump(session)
        self.assertFalse(commits)
        self.assertTrue(errors)
        session.shutdown()

    def test_cancellation_and_single_job(self):
        session = UiSession(dataset())
        commits = []

        def compute(model, progress, cancel):
            cancel.wait(2)
            raise CancelledError()

        session.submit(compute, commit=lambda m, r: commits.append(r))
        with self.assertRaises(RuntimeError):
            session.submit(compute)
        session.cancel()
        self.pump(session)
        self.assertFalse(commits)
        session.shutdown()

    def test_borrowed_dataset_is_not_closed(self):
        model = dataset()
        calls = []
        model.close = lambda: calls.append("closed")
        session = UiSession(model)
        session.set_dataset(Dataset())
        session.shutdown()
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
