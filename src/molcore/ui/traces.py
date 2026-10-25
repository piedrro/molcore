"""Observation trace inspection without modifying measured values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QCheckBox

from .._logging import logged
from ..files import report
from ..staging import state_token
from ..types.analysis import AnalysisResult
from ._analysis import AnalysisTool, numeric_metrics, refresh_combo
from ._common import combo

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ..types import Experiment, ProgressCallback
    from .session import UiSession


def _photon_rows(model: Experiment, acquisition_id: str) -> dict[str, np.ndarray]:
    """Project sequences into display columns without storing observation records.

    Args:
        model: Source experiment supplying validated photon samples.
        acquisition_id: Selected source acquisition.

    Returns:
        Detached columns retaining sequence/sample identities, units and frames.
    """
    from ..photon_sequences import to_columns

    rows = to_columns(model.photon_sequences.query(acquisition_id=acquisition_id))
    rows["intensity"] = np.where(rows["valid"], rows["intensity"], np.nan)
    infos = [
        model.acquisitions[acquisition_id]
        .channels[str(channel)]
        .frame_info(model.acquisitions[acquisition_id].channels[str(channel)].channel_index(int(frame)))
        for channel, frame in zip(rows["channel"], rows["frame"], strict=True)
    ]
    rows["channel_frame"] = np.array([info.channel_frame for info in infos], dtype=np.int64)
    rows["cycle_index"] = np.array(
        [info.cycle_index if info.cycle_index is not None else -1 for info in infos], dtype=np.int64
    )
    return rows


class TraceViewerWidget(AnalysisTool):
    """Browse observations or independent photon sequences with channel/time traces."""

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.source = self.field("Trace data", combo(["Photon sequences", "Observations"]))
        self.run_id, self.molecule, self.metric = combo([]), combo([]), combo([])
        self.channels = combo(["All"])
        self.xaxis = combo(["Frame", "Time (s)"])
        self.background = combo(
            ["Stored corrected intensity", "Raw signal sum", "Local background", "Global background"]
        )
        self.normalise = QCheckBox()
        for label, control in (
            ("Run", self.run_id),
            ("Molecule / track", self.molecule),
            ("Channel", self.channels),
            ("Metric", self.metric),
            ("Horizontal axis", self.xaxis),
            ("Intensity display", self.background),
            ("Normalise display (0–1)", self.normalise),
        ):
            self.fields.addRow(label, control)
            signal = control.toggled if isinstance(control, QCheckBox) else control.currentIndexChanged
            signal.connect(lambda *_: self.invoke(self.redraw))
        self.run_id.currentIndexChanged.connect(self.refresh_molecules)
        self.button("Load measurement traces", self.compute)
        self.button("Export displayed trace rows…", self.export_csv)
        self.button("Request host selection", self.select_trace)

    @logged
    def compute(self) -> None:
        """Compute a detached result using the current controls."""
        session = self.session
        a, c, space = session.acquisition_id, session.channel, session.space
        source = self.source.currentText()

        def compute(model: Experiment, progress: ProgressCallback | None, cancel: Any) -> AnalysisResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            report(progress, cancel, 0, 1, "Loading measurement traces")
            rows = (
                _photon_rows(model, a)
                if source == "Photon sequences"
                else model.observations.query(acquisition_id=a)
            )
            report(progress, cancel, 1, 1, "Loaded measurement traces")
            return AnalysisResult(a, c, space, {}, rows, state_token(model))

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            self.ready(result, f"{len(result.columns['id'])} samples; stored measurements unchanged")
            rows = result.columns
            refresh_combo(self.run_id, np.unique(rows.get("run_id", np.array([""]))))
            refresh_combo(
                self.metric,
                [k for k in numeric_metrics(rows) if k not in ("frame", "track_id", "localization_id")],
            )
            if "intensity" in rows:
                self.metric.setCurrentText("intensity")
            refresh_combo(self.channels, ["All", *np.unique(rows["channel"])])
            self.refresh_molecules()

        self.run(compute, complete)

    @logged
    def refresh_molecules(self, *_: Any) -> None:
        """Refresh molecules.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        if self.preview is None:
            return
        rows = self.preview.columns
        mask = rows.get("run_id", np.full(len(rows["id"]), "")) == self.run_id.currentText()
        refresh_combo(self.molecule, np.unique(rows["molecule_id"][mask]))
        self.invoke(self.redraw)

    @logged
    def redraw(self) -> None:
        """Redraw the selected detached measurements without changing source data."""
        if self.preview is None or not self.molecule.currentText() or not self.metric.currentText():
            return
        self.require_preview()
        self.plot.clear()
        self._columns = None
        self.table.setRowCount(0)
        rows = self.preview.columns
        mask = rows["molecule_id"] == self.molecule.currentText()
        if "run_id" in rows:
            mask &= rows["run_id"] == self.run_id.currentText()
        if self.channels.currentText() != "All":
            mask &= rows["channel"] == self.channels.currentText()
        selected = {k: v[mask].copy() for k, v in rows.items()}
        y = np.asarray(selected[self.metric.currentText()], dtype=float).copy()
        if self.metric.currentText() == "intensity":
            mode = self.background.currentText()
            if mode != "Stored corrected intensity":
                required = ["signal_sum"] + (
                    [
                        "signal_count",
                        "local_background_mean" if mode == "Local background" else "global_background_mean",
                    ]
                    if mode != "Raw signal sum"
                    else []
                )
                if any(k not in selected for k in required):
                    raise ValueError("Selected observations lack the requested background measurements")
                y = selected["signal_sum"].astype(float).copy()
                if mode != "Raw signal sum":
                    y -= selected[required[-1]] * selected["signal_count"]
        xkey = "timestamp" if self.xaxis.currentText() == "Time (s)" else "frame"
        if xkey not in selected or not np.isfinite(selected[xkey]).all():
            raise ValueError("Recorded timestamps unavailable; select Frame")
        self.plot.clear()
        color = 0
        for acquisition in np.unique(selected["acquisition_id"]):
            for channel in np.unique(selected["channel"]):
                indexes = np.flatnonzero(
                    (selected["acquisition_id"] == acquisition) & (selected["channel"] == channel)
                )
                indexes = indexes[np.argsort(selected[xkey][indexes], kind="stable")]
                if not len(indexes):
                    continue
                values = y[indexes].copy()
                finite = np.isfinite(values)
                if self.normalise.isChecked() and finite.any():
                    values[finite] = (values[finite] - values[finite].min()) / max(
                        np.ptp(values[finite]), 1e-12
                    )
                x = selected[xkey][indexes]
                # Split curves at unmeasured logical frames, retaining sparse-frame provenance.
                frames = selected.get("channel_frame", selected["frame"])[indexes]
                breaks = np.flatnonzero(np.diff(frames) > 1) + 1
                px, py = np.insert(x.astype(float), breaks, np.nan), np.insert(values, breaks, np.nan)
                self.plot.plot(px, py, pen=pg.intColor(color), symbol="o", symbolSize=4, connect="finite")
                y[indexes] = values
                color += 1
        selected["display_value"] = y
        self.show_columns(selected)
        self.plot.setLabel("bottom", self.xaxis.currentText())
        units = (
            np.unique(selected["unit"])
            if "unit" in selected and self.metric.currentText() in ("intensity", "photons")
            else []
        )
        suffix = (
            " (normalised)"
            if self.normalise.isChecked()
            else (f" ({', '.join(units)})" if len(units) else "")
        )
        self.plot.setLabel("left", self.metric.currentText() + suffix)

    @logged
    def select_trace(self) -> None:
        """Select trace."""
        self.require_preview()
        if self._columns is None:
            raise ValueError("Select a measured trace")
        rows = self._columns
        tracks = np.unique(rows["track_id"][rows["track_id"] != 0])
        ids = tracks if len(tracks) else np.unique(rows["localization_id"][rows["localization_id"] != 0])
        self.request_selection(ids, "tracks" if len(tracks) else "localizations")
