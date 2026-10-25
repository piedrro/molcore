"""Shared analysis widgets embed without napari and respect preview boundaries."""

import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtWidgets import QApplication

from molcore import Acquisition, ArraySource, Experiment
from molcore.ui import (
    ClusteringWidget,
    ColocalisationWidget,
    LocalisationRenderWidget,
    MolCoreToolsWidget,
    PhotometryWidget,
    ResultsFilterWidget,
    TraceViewerWidget,
    TrackingWidget,
    TrajectoryStatisticsWidget,
    UiSession,
)


class AnalysisWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        model = Experiment()
        model.add_acquisition(
            Acquisition(
                ArraySource(np.ones((3, 11, 11))), id="a", timestamps=[0.0, 0.1, 0.3], pixel_size=100.0
            )
        )
        self.ids = model.localizations.add(
            acquisition_id="a",
            channel="image",
            frame=[0, 1, 2],
            x=[4.0, 4.1, 4.2],
            y=[4.0, 4.0, 4.0],
            photons=[10.0, 20.0, 30.0],
        )
        self.session = UiSession(model)
        self.addCleanup(self.session.shutdown)

    def pump(self):
        deadline = time.monotonic() + 30
        while self.session.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertFalse(self.session.busy)

    def test_embed_all_tools_and_container(self):
        for cls in (
            TrackingWidget,
            PhotometryWidget,
            TraceViewerWidget,
            TrajectoryStatisticsWidget,
            ResultsFilterWidget,
            LocalisationRenderWidget,
            ColocalisationWidget,
            ClusteringWidget,
            MolCoreToolsWidget,
        ):
            widget = cls(self.session)
            self.assertIs(widget.session, self.session)
            widget.close()

    def test_filter_preview_then_selection_signal(self):
        widget = ResultsFilterWidget(self.session)
        widget.metric.setCurrentText("photons")
        widget.minimum.setValue(10)
        widget.maximum.setValue(20)
        widget.compute()
        self.pump()
        self.assertIsNotNone(widget.preview, widget.status.text())
        np.testing.assert_array_equal(widget.preview.selected_ids, self.ids[:2])
        selected = []
        widget.selection_requested.connect(selected.append)
        widget.select_preview()
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(self.session.experiment.localizations), 3)
        widget.minimum.setValue(11)
        self.assertIsNone(widget.preview)
        widget.close()

    def test_clustering_changes_no_source_rows(self):
        widget = ClusteringWidget(self.session)
        widget.settings_form.controls["min_samples"].setValue(2)
        widget.compute()
        self.pump()
        self.assertIsNotNone(widget.preview, widget.status.text())
        self.assertEqual(len(widget.preview.centroids["cluster"]), 1)
        self.assertEqual(len(self.session.experiment.localizations), 3)
        widget.close()

    def test_tracking_preview_requires_explicit_apply(self):
        from molcore.types import TrackingResult, TrackingSettings

        widget = TrackingWidget(self.session)

        def fake(model, a, c, **kwargs):
            self.assertFalse(kwargs["apply"])
            return TrackingResult(a, c, (self.ids,), TrackingSettings(), (), "0.7")

        with patch("molcore.tracking.track", side_effect=fake):
            widget.compute()
            self.pump()
        self.assertEqual(len(self.session.experiment.tracks.query()["id"]), 0)
        widget.apply_preview()
        self.assertEqual(len(self.session.experiment.tracks.query()["id"]), 3)
        self.assertIsNone(widget.preview)
        widget.close()

    def test_enum_forms_return_enum_instances(self):
        from molcore.types import ApertureShape, PhotometrySettings
        from molcore.ui._common import SettingsForm

        form = SettingsForm(PhotometrySettings)
        self.assertIs(form.value().shape, ApertureShape.SQUARE)
        form.close()

    def test_trace_normalisation_is_display_only_and_keeps_actual_time(self):
        model = self.session.experiment
        model.observations.add(
            molecule_id=["seed"] * 3,
            acquisition_id="a",
            channel="image",
            frame=[0, 1, 2],
            intensity=[2.0, 4.0, 6.0],
            timestamp=[0.0, 0.1, 0.3],
            localization_id=self.ids,
            signal_sum=[3.0, 5.0, 7.0],
            signal_count=[1.0, 1.0, 1.0],
            local_background_mean=[1.0, 1.0, 1.0],
            run_id="r",
        )
        widget = TraceViewerWidget(self.session)
        widget.source.setCurrentText("Observations")
        widget.compute()
        self.pump()
        self.assertIsNotNone(widget.preview, widget.status.text())
        widget.xaxis.setCurrentText("Time (s)")
        widget.normalise.setChecked(True)
        widget.redraw()
        np.testing.assert_allclose(widget._columns["display_value"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(model.observations.query()["intensity"], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(widget.plot.listDataItems()[0].xData, [0.0, 0.1, 0.3])
        widget.close()

    def test_trajectory_views_use_real_timing(self):
        self.session.experiment.tracks.add(self.ids)
        widget = TrajectoryStatisticsWidget(self.session)
        widget.compute()
        self.pump()
        self.assertIsNotNone(widget.preview, widget.status.text())
        widget.view_mode.setCurrentText("Track speed")
        widget.redraw()
        np.testing.assert_allclose(widget._columns["elapsed_time"], [0.1, 0.2])
        np.testing.assert_allclose(widget.plot.listDataItems()[0].xData, [0.1, 0.3])
        widget.view_mode.setCurrentText("Track MSD")
        widget.redraw()
        self.assertIn("msd", widget._columns)
        widget.close()

    def test_changed_selection_discards_running_preview(self):
        import threading

        from molcore.filtering import filter_results

        entered, release = threading.Event(), threading.Event()
        widget = ResultsFilterWidget(self.session)

        def slow(model, a, c, **kwargs):
            entered.set()
            release.wait(5)
            return filter_results(model, a, c)

        with patch("molcore.filtering.filter_results", side_effect=slow):
            widget.compute()
            self.assertTrue(entered.wait(5))
            self.session.select("a", "image", space="registered")
            release.set()
            self.pump()
        self.assertIsNone(widget.preview)
        self.assertEqual(widget.status.text(), "Inputs changed; compute a new preview")
        widget.close()

    def test_empty_model_and_cancelled_export_dialog(self):
        empty = UiSession()
        widget = MolCoreToolsWidget(empty)
        self.assertEqual(widget.tabs.count(), 13)
        widget.close()
        empty.shutdown()
        tool = ResultsFilterWidget(self.session)
        tool.compute()
        self.pump()
        with patch("qtpy.QtWidgets.QFileDialog.getSaveFileName", return_value=("", "")):
            tool.export_csv()
        self.assertFalse(self.session.busy)
        tool.close()

    def test_trace_host_selection_uses_anchor_channel(self):
        from molcore import ChannelSpec

        model = Experiment()
        model.add_acquisition(
            Acquisition(
                ArraySource(np.ones((1, 5, 5))),
                id="a",
                channel_specs={"donor": ChannelSpec(), "acceptor": ChannelSpec()},
            )
        )
        ids = model.localizations.add(acquisition_id="a", channel="donor", frame=[0], x=[2.0], y=[2.0])
        model.observations.add(
            molecule_id=["seed"],
            acquisition_id="a",
            channel="acceptor",
            frame=[0],
            intensity=[3.0],
            localization_id=ids,
            run_id="r",
        )
        session = UiSession(model)
        session.select("a", "acceptor")
        widget = TraceViewerWidget(session)
        widget.source.setCurrentText("Observations")
        widget.compute()
        deadline = time.monotonic() + 10
        while session.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        selections = []
        widget.selection_requested.connect(selections.append)
        widget.select_trace()
        self.assertEqual(selections[0]["channel"], "donor")
        np.testing.assert_array_equal(selections[0]["ids"], ids)
        widget.close()
        session.shutdown()

    def test_photometry_preview_and_explicit_apply(self):
        from molcore.types import PhotometryResult, PhotometrySettings, TraceMode

        widget = PhotometryWidget(self.session)
        rows = dict(
            molecule_id=np.array(["seed"] * 3),
            acquisition_id=np.array(["a"] * 3),
            channel=np.array(["image"] * 3),
            frame=np.arange(3),
            localization_id=self.ids,
            intensity=np.array([1.0, 2.0, 3.0]),
        )
        result = PhotometryResult(rows, PhotometrySettings(), TraceMode.LOCALIZATIONS, "test-run", ())
        with patch("molcore.photometry.measure_localizations", return_value=result) as function:
            widget.compute()
            self.pump()
            self.assertFalse(function.call_args.kwargs["apply"])
        self.assertIsNotNone(widget.preview, widget.status.text())
        self.assertEqual(len(self.session.experiment.observations), 0)
        widget.apply_preview()
        self.assertEqual(len(self.session.experiment.observations), 3)
        self.assertIsNone(widget.preview)
        widget.close()

    def test_render_export_includes_geometry(self):
        import json
        import tempfile
        from pathlib import Path

        import tifffile

        widget = LocalisationRenderWidget(self.session)
        widget.settings_form.controls["oversampling"].setValue(2.0)
        widget.compute()
        self.pump()
        self.assertIsNotNone(widget.preview, widget.status.text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "density.tif"
            with patch("qtpy.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(path), "")):
                widget.export_image()
            self.assertEqual(tifffile.imread(path).shape, (22, 22))
            metadata = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(metadata["pixel_spacing"], 0.5)
            self.assertEqual(metadata["space"], "raw")
        widget.close()

    def test_cancellation_discards_filter_result(self):
        import threading

        from molcore.filtering import filter_results

        entered = threading.Event()
        widget = ResultsFilterWidget(self.session)

        def slow(model, a, c, **kwargs):
            entered.set()
            while not kwargs["cancel_event"].is_set():
                time.sleep(0.001)
            return filter_results(model, a, c)

        with patch("molcore.filtering.filter_results", side_effect=slow):
            widget.compute()
            self.assertTrue(entered.wait(5))
            self.session.cancel()
            self.pump()
        self.assertIsNone(widget.preview)
        self.assertEqual(len(self.session.experiment.localizations), 3)
        widget.close()

    def test_trace_viewer_reads_independent_photon_sequences(self):
        from molcore import PhotonSequence

        self.session.experiment.photon_sequences.add(
            PhotonSequence(
                acquisition_id="a",
                channel="image",
                frames=np.array([0, 2]),
                photons=np.array([12.0, 18.0]),
                id="photon-trace",
            )
        )
        widget = TraceViewerWidget(self.session)
        widget.source.setCurrentText("Photon sequences")
        widget.compute()
        self.pump()
        self.assertIsNotNone(widget.preview, widget.status.text())
        np.testing.assert_array_equal(widget._columns["frame"], [0, 2])
        np.testing.assert_array_equal(widget._columns["display_value"], [12.0, 18.0])
        self.assertEqual(widget._columns["unit"][0], "photons")
        self.assertEqual(len(self.session.experiment.observations), 0)
        widget.close()

    def test_native_trace_apply_and_background_display_use_photon_model(self):
        from test_photometry_trajectory import InlineCpu

        from molcore.photometry import stationary_traces

        preview = stationary_traces(self.session.experiment, self.ids[:1], cpu=InlineCpu(), apply=False)
        widget = PhotometryWidget(self.session)
        widget.mode.setCurrentText("stationary")
        with patch("molcore.photometry.stationary_traces", return_value=preview):
            widget.compute()
            self.pump()
        self.assertIsNotNone(widget.preview, widget.status.text())
        widget.apply_preview()
        self.assertEqual(len(self.session.experiment.observations), 0)
        self.assertEqual(len(self.session.experiment.photon_sequences), 1)
        viewer = TraceViewerWidget(self.session)
        self.assertEqual(viewer.source.currentText(), "Photon sequences")
        viewer.compute()
        self.pump()
        viewer.background.setCurrentText("Raw signal sum")
        viewer.redraw()
        np.testing.assert_allclose(
            viewer._columns["display_value"], preview.sequences[0].measurements["signal_sum"]
        )
        viewer.close()
        widget.close()


if __name__ == "__main__":
    unittest.main()
