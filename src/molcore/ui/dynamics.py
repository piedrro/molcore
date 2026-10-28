"""Tracking, aperture photometry and trajectory analysis widgets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtWidgets import QLineEdit

from .. import photometry, tracking
from .._logging import logged
from ..files import report
from ..staging import state_token
from ..trajectory import trajectory_statistics
from ..types import PhotometrySettings, PhotonSequenceResult, SpatialUnit, TrackingSettings
from ..types.analysis import AnalysisResult
from ._analysis import IdTool, refresh_combo
from ._common import SettingsForm, combo

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ..types import Dataset, PhotometryResult, ProgressCallback, TrackingResult
    from .session import UiSession


class TrackingWidget(IdTool):
    """Preview Trackpy memberships and explicitly apply a partition replacement."""

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned dataset, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.ids.hide()
        self.fields.labelForField(self.ids).hide()
        self.settings_form = SettingsForm(TrackingSettings)
        # Corrections are automatic; coordinate space is not a UI setting.
        space_control = self.settings_form.controls["space"]
        space_control.hide()
        self.settings_form.layout().labelForField(space_control).hide()
        self.form.addWidget(self.settings_form)
        self.settings_form.changed.connect(self.invalidate)
        self.segmentation = self.field("Segmentation constraint", combo(["None"]))
        session.dataset_changed.connect(self.refresh)
        session.results_changed.connect(self.refresh)
        session.selection_changed.connect(self.refresh)
        self.button("Preview tracks", self.compute)
        self.button("Apply tracks — replace selected partition", self.apply_preview)
        self.button("Export preview memberships…", self.export_csv)
        self.refresh()

    @logged
    def refresh(self, *_: Any) -> None:
        """Refresh controls from the current shared model and selection.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        refresh_combo(self.segmentation, ["None", *self.session.dataset.segmentations])
        self.settings_form.controls["space"].setCurrentText(str(self.session.space))

    @logged
    def compute(self) -> None:
        """Compute a detached result using the current controls."""
        session = self.session
        settings = replace(self.settings_form.value(), space=session.space)
        a, c, cpus = session.acquisition_id, session.channel, session.num_cpus
        segmentation = None if self.segmentation.currentText() == "None" else self.segmentation.currentText()

        def compute(model: Dataset, progress: ProgressCallback | None, cancel: Any) -> TrackingResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return tracking.track(
                model,
                a,
                c,
                settings=settings,
                segmentation_id=segmentation,
                apply=False,
                num_cpus=cpus,
                progress=progress,
                cancel_event=cancel,
            )

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            self.ready(result, f"{len(result.memberships)} tracks; active memberships unchanged")
            ids = np.concatenate(result.memberships) if result.memberships else np.array([], dtype=np.uint64)
            groups = (
                np.concatenate([np.full(len(v), i) for i, v in enumerate(result.memberships)])
                if len(ids)
                else ids
            )
            rows = session.dataset.localizations.query(ids=ids, space=settings.space).to_columns()
            membership = {int(v): i for i, members in enumerate(result.memberships) for v in members}
            rows["preview_track"] = np.array([membership[int(v)] for v in rows["id"]], dtype=np.int64)
            self.show_columns(rows)
            self.plot.clear()
            for group in np.unique(groups):
                mask = rows["preview_track"] == group
                order = np.argsort(rows["frame"][mask])
                self.plot.plot(rows["x"][mask][order], rows["y"][mask][order], symbol="o", symbolSize=4)

        self.run(compute, complete)

    @logged
    def apply_preview(self) -> None:
        """Apply the current preview after checking its source revision."""
        self.require_preview()
        tracking.apply_tracking(self.session.dataset, self.preview, expected=self.preview_token)
        self.session.changed()
        self.status.setText("Tracks applied")


class PhotometryWidget(IdTool):
    """Measure localizations, stationary seeds or moving tracks across channels."""

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned dataset, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.mode = self.field("Mode", combo(["localizations", "stationary", "track"]))
        self.channels = self.field("Measurement channels (comma separated; blank: source)", QLineEdit())
        self.settings_form = SettingsForm(PhotometrySettings)
        self.form.addWidget(self.settings_form)
        self.settings_form.changed.connect(self.invalidate)
        self.button("Preview aperture and background masks", self.preview_masks)
        self.button("Compute measurement preview", self.compute)
        self.button("Apply measurement run", self.apply_preview)
        self.button("Export measurement rows…", self.export_csv)

    @logged
    def preview_masks(self) -> None:
        """Preview masks."""
        settings = self.settings_form.value()
        session = self.session
        view = session.dataset.acquisitions[session.acquisition_id].channels[session.channel]
        image = view[view.channel_index(session.camera_frame)]
        ids = self.selected_ids(tracks=self.mode.currentText() == "track")
        rows = (
            session.dataset.tracks.query(track_id=ids, space="raw")
            if self.mode.currentText() == "track"
            else session.dataset.localizations.query(ids=ids, space="raw")
        )
        current = np.flatnonzero(
            (rows["frame"] == session.camera_frame) & (rows["channel"] == session.channel)
        )
        if not len(current):
            raise ValueError("Select an entity measured in the current frame to preview its aperture")
        index = current[0]
        # Match the backend's nearest-integer pixel-centre convention.
        x, y = int(np.floor(rows["x"][index] + 0.5)), int(np.floor(rows["y"][index] + 0.5))
        radius = settings.spot_size // 2 if settings.shape == "square" else settings.spot_size / 2
        yy, xx = np.indices(image.shape)
        distance = (
            np.maximum(abs(xx - x), abs(yy - y)) if settings.shape == "square" else np.hypot(xx - x, yy - y)
        )
        signal = distance <= radius
        ring = (distance > radius + settings.buffer) & (
            distance <= radius + settings.buffer + settings.background_width
        )
        if settings.exclude_neighbours:
            neighbours = session.dataset.localizations.query(
                acquisition_id=session.acquisition_id,
                channel=session.channel,
                frame_range=(session.camera_frame, session.camera_frame + 1),
                space="raw",
            )
            for nx, ny in zip(neighbours["x"], neighbours["y"], strict=True):
                nx, ny = np.floor(nx + 0.5), np.floor(ny + 0.5)
                d = (
                    np.maximum(abs(xx - nx), abs(yy - ny))
                    if settings.shape == "square"
                    else np.hypot(xx - nx, yy - ny)
                )
                ring &= d > radius
        import pyqtgraph as pg

        self.plot.clear()
        display = session.dataset.transforms.image(
            session.acquisition_id, session.channel, session.camera_frame, space=session.space
        )
        self.plot.addItem(pg.ImageItem(display, axisOrder="row-major"))
        matrix = session.dataset.transforms.matrix(
            session.acquisition_id, session.channel, session.camera_frame, space=session.space
        )
        for mask, color in ((signal, "r"), (ring, "b")):
            py, px = np.nonzero(mask)
            points = matrix @ np.stack([px, py, np.ones(len(px))])
            self.plot.plot(
                points[0], points[1], pen=None, symbol="s", symbolBrush=None, symbolPen=color, symbolSize=8
            )
        self.plot.setAspectLocked(True)
        self.status.setText("Red: aperture; blue: local background after neighbour exclusion")

    @logged
    def compute(self) -> None:
        """Compute a detached result using the current controls."""
        session, settings = self.session, self.settings_form.value()
        mode = self.mode.currentText()
        ids = self.selected_ids(tracks=mode == "track")
        if not len(ids):
            raise ValueError("No selected localizations or tracks")
        channels = [c.strip() for c in self.channels.text().split(",") if c.strip()] or None
        cpus = session.num_cpus
        function = {
            "localizations": photometry.measure_localizations,
            "stationary": photometry.stationary_traces,
            "track": photometry.track_traces,
        }[mode]

        def compute(
            model: Dataset, progress: ProgressCallback | None, cancel: Any
        ) -> PhotometryResult | PhotonSequenceResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return function(
                model,
                ids,
                channels=channels,
                settings=settings,
                apply=False,
                num_cpus=cpus,
                progress=progress,
                cancel_event=cancel,
            )

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            description = (
                f"{len(result.sequences)} photon sequences"
                if isinstance(result, PhotonSequenceResult)
                else f"{len(result.columns['frame'])} localisation measurements"
            )
            self.ready(result, description + "; measurement run not yet applied")
            self.show_columns(result.columns)

        self.run(compute, complete)

    @logged
    def apply_preview(self) -> None:
        """Apply the current preview after checking its source revision."""
        self.require_preview()
        photometry.apply_photometry(self.session.dataset, self.preview, expected=self.preview_token)
        self.session.changed()
        self.status.setText("Measurement run applied")


@dataclass(frozen=True)
class TrajectoryAnalysisResult(AnalysisResult):
    """Detached population summary and per-track step/MSD results."""

    statistics: tuple


class TrajectoryStatisticsWidget(IdTool):
    """Inspect actual-time trajectory statistics and population diffusion estimates."""

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned dataset, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.unit = self.field("Spatial unit", combo(["pixels", "nm"]))
        self.interval = self.field(
            "Fallback camera frame interval (s; blank: require timestamps)", QLineEdit()
        )
        self.fit_range = self.field("Diffusion fit lag bounds (seconds; blank: no fit)", QLineEdit())
        self.lag_bins = self.field("Optional lag bin edges (seconds)", QLineEdit())
        self.track = combo([])
        self.view_mode = combo(["Population diffusion", "Track MSD", "Track speed", "Track steps"])
        self.fields.addRow("Track", self.track)
        self.fields.addRow("Plot", self.view_mode)
        self.track.currentIndexChanged.connect(lambda: self.invoke(self.redraw))
        self.view_mode.currentIndexChanged.connect(lambda: self.invoke(self.redraw))
        self.button("Compute trajectory statistics", self.compute)
        self.button("Export displayed data…", self.export_csv)
        self.button("Request host track selection", self.select_track)

    @logged
    def compute(self) -> None:
        """Compute a detached result using the current controls."""
        session = self.session
        ids = self.selected_ids(tracks=True)
        if not len(ids):
            raise ValueError("No selected tracks")
        interval = float(self.interval.text()) if self.interval.text().strip() else None
        fit = tuple(float(v) for v in self.fit_range.text().replace(",", " ").split()) or None
        bins = [float(v) for v in self.lag_bins.text().replace(",", " ").split()] or None
        a, c, space, unit = (
            session.acquisition_id,
            session.channel,
            session.space,
            SpatialUnit(self.unit.currentText()),
        )

        def compute(
            model: Dataset, progress: ProgressCallback | None, cancel: Any
        ) -> TrajectoryAnalysisResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            results = []
            for index, track_id in enumerate(ids):
                report(progress, cancel, index, len(ids), "Trajectory statistics")
                results.append(
                    trajectory_statistics(
                        model,
                        int(track_id),
                        space=space,
                        unit=unit,
                        frame_interval=interval,
                        fit_lag_range=fit,
                        lag_bin_edges=bins,
                    )
                )
            report(progress, cancel, len(ids), len(ids), "Trajectory statistics complete")
            columns = {
                "track_id": ids.copy(),
                "duration": np.array([r.duration for r in results]),
                "path_length": np.array([r.path_length for r in results]),
                "diffusion": np.array([r.diffusion.coefficient if r.diffusion else np.nan for r in results]),
            }
            return TrajectoryAnalysisResult(
                a,
                c,
                space,
                dict(unit=unit, frame_interval=interval, fit_lag_range=fit, lag_bin_edges=bins),
                columns,
                state_token(model),
                tuple(results),
            )

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            self.ready(result, f"{len(result.statistics)} tracks; units: {unit}; elapsed time in seconds")
            refresh_combo(self.track, [r.track_id for r in result.statistics])
            self.redraw()

        self.run(compute, complete)

    @logged
    def redraw(self) -> None:
        """Redraw the selected detached measurements without changing source data."""
        if self.preview is None:
            return
        self.require_preview()
        result = self.preview
        self.plot.clear()
        if self.view_mode.currentText() == "Population diffusion":
            self.histogram(result.columns["diffusion"], f"Diffusion ({result.settings['unit']}²/s)")
            self.show_columns(result.columns)
            return
        track = next(r for r in result.statistics if str(r.track_id) == self.track.currentText())
        if self.view_mode.currentText() == "Track MSD":
            rows = track.msd
            self.plot.plot(rows["lag"], rows["msd"], symbol="o")
            self.plot.setLabel("bottom", "Lag (s)")
            self.plot.setLabel("left", f"MSD ({track.unit}²)")
        else:
            rows = track.steps
            # Step tables retain timing and acquisition provenance supplied by the backend.
            value = "speed" if self.view_mode.currentText() == "Track speed" else "distance"
            x = rows["time_end"]
            self.plot.plot(x, rows[value], symbol="o")
            self.plot.setLabel("bottom", "Time (s)")
            self.plot.setLabel("left", f"{value} ({track.unit}{'/s' if value == 'speed' else ''})")
        self.show_columns(rows)

    @logged
    def select_track(self) -> None:
        """Select track."""
        self.require_preview()
        self.request_selection([int(self.track.currentText())], "tracks")
