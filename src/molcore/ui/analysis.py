"""Filtering, density rendering, fiducial matching and DBSCAN Qt tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtWidgets import QDoubleSpinBox, QFileDialog

from .. import clustering, colocalisation, filtering, rendering
from .._logging import logged
from ..types.analysis import ClusteringSettings, ColocalisationSettings, FilterSettings, RenderSettings
from ._analysis import AnalysisTool, numeric_metrics, partition_rows, refresh_combo, settings_metadata
from ._common import SettingsForm, combo

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ..types import Experiment, ProgressCallback
    from ..types.analysis import ClusteringResult, ColocalisationResult, FilterResult, RenderResult
    from .session import UiSession


class ResultsFilterWidget(AnalysisTool):
    """Preview and export inclusive numeric subsets without deleting source data."""

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.target = self.field("Target", combo(["localizations", "tracks"]))
        self.metric = self.field("Numeric metric", combo([]))
        self.aggregation = self.field("Track aggregation", combo(["mean", "median", "min", "max", "std"]))
        self.minimum, self.maximum = QDoubleSpinBox(), QDoubleSpinBox()
        for control in (self.minimum, self.maximum):
            control.setDecimals(6)
            control.setRange(-1e12, 1e12)
        self.maximum.setValue(1e12)
        self.field("Minimum (inclusive)", self.minimum)
        self.field("Maximum (inclusive)", self.maximum)
        self.target.currentIndexChanged.connect(self.refresh)
        session.selection_changed.connect(self.refresh)
        session.results_changed.connect(self.refresh)
        session.experiment_changed.connect(self.refresh)
        self.button("Preview filter", self.compute)
        self.button("Request host selection", self.select_preview)
        self.button("Export selected rows…", self.export_csv)
        self.refresh()

    @logged
    def refresh(self, *_: Any) -> None:
        """Refresh controls from the current shared model and selection.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        rows = partition_rows(self.session, tracks=self.target.currentText() == "tracks")
        refresh_combo(self.metric, numeric_metrics(rows))
        self.aggregation.setEnabled(self.target.currentText() == "tracks")

    @logged
    def compute(self) -> None:
        """Compute a detached result using the current controls."""
        session = self.session
        a, c, space = session.acquisition_id, session.channel, session.space
        settings = FilterSettings(
            self.metric.currentText(),
            self.minimum.value(),
            self.maximum.value(),
            self.target.currentText(),
            self.aggregation.currentText(),
        )

        def compute(model: Experiment, progress: ProgressCallback | None, cancel: Any) -> FilterResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return filtering.filter_results(
                model, a, c, settings=settings, space=space, progress=progress, cancel_event=cancel
            )

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            self.ready(
                result, f"{len(result.selected_ids)} {settings.target} selected; source data unchanged"
            )
            self.show_columns(result.columns)
            self.histogram(result.values, settings.metric)

        self.run(compute, complete)

    @logged
    def select_preview(self) -> None:
        """Emit stable result IDs for selection by the embedding host."""
        self.require_preview()
        self.request_selection(self.preview.selected_ids, self.preview.settings.target)


class ClusteringWidget(AnalysisTool):
    """Inspect stable-ID DBSCAN assignments, noise and centroids."""

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.settings_form = SettingsForm(ClusteringSettings)
        self.form.addWidget(self.settings_form)
        self.settings_form.changed.connect(self.invalidate)
        self.button("Preview DBSCAN clusters", self.compute)
        self.button("Export assignments…", self.export_csv)
        self.button("Export centroids…", self.export_centroids)

    @logged
    def compute(self) -> None:
        """Compute a detached result using the current controls."""
        session, settings = self.session, self.settings_form.value()
        a, c, space = session.acquisition_id, session.channel, session.space

        def compute(model: Experiment, progress: ProgressCallback | None, cancel: Any) -> ClusteringResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return clustering.cluster_localizations(
                model, a, c, settings=settings, space=space, progress=progress, cancel_event=cancel
            )

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            rows = result.columns
            self.ready(
                result,
                f"{len(result.centroids['cluster'])} clusters; {(rows['cluster'] == -1).sum()} noise rows",
            )
            self.show_columns(rows)
            self.plot.clear()
            for label in np.unique(rows["cluster"]):
                mask = rows["cluster"] == label
                color = (140, 140, 140) if label < 0 else __import__("pyqtgraph").intColor(int(label))
                self.plot.plot(
                    rows["x"][mask], rows["y"][mask], pen=None, symbol="o", symbolSize=6, symbolBrush=color
                )
            self.plot.setLabel("bottom", "x (coordinate pixels)")
            self.plot.setLabel("left", "y (coordinate pixels)")

        self.run(compute, complete)

    @logged
    def export_centroids(self) -> None:
        """Export centroids."""
        from ._analysis import export_columns

        self.require_preview()
        path, _ = QFileDialog.getSaveFileName(self, "Export cluster centroids", "", "CSV (*.csv)")
        if path:
            export_columns(path, self.preview.centroids)


class ColocalisationWidget(AnalysisTool):
    """Preview mutual-nearest Euclidean fiducial pairs across two channel sets."""

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.target = self.field("Target channel", combo([]))
        self.settings_form = SettingsForm(ColocalisationSettings)
        self.form.addWidget(self.settings_form)
        self.settings_form.changed.connect(self.invalidate)
        session.selection_changed.connect(self.refresh)
        session.experiment_changed.connect(self.refresh)
        self.button("Preview colocalisation", self.compute)
        self.button("Export matched pairs…", self.export_csv)
        self.refresh()

    @logged
    def refresh(self, *_: Any) -> None:
        """Refresh controls from the current shared model and selection.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        session = self.session
        channels = (
            session.experiment.acquisitions[session.acquisition_id].channels if session.acquisition_id else {}
        )
        refresh_combo(self.target, [c for c in channels if c != session.channel])

    @logged
    def compute(self) -> None:
        """Compute a detached result using the current controls."""
        session, settings = self.session, self.settings_form.value()
        a, c, target, space = (
            session.acquisition_id,
            session.channel,
            self.target.currentText(),
            session.space,
        )
        if not target:
            raise ValueError("A second channel is required")

        def compute(
            model: Experiment, progress: ProgressCallback | None, cancel: Any
        ) -> ColocalisationResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return colocalisation.colocalise(
                model, a, c, target, settings=settings, space=space, progress=progress, cancel_event=cancel
            )

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            self.ready(
                result,
                f"{len(result.columns['source_id'])} pairs; unmatched: "
                f"{result.unmatched_source} source, {result.unmatched_target} target",
            )
            self.show_columns(result.columns)
            self.plot.clear()
            self.plot.plot(result.columns["x"], result.columns["y"], pen=None, symbol="+", symbolSize=10)
            self.plot.setLabel("bottom", "Midpoint x (coordinate pixels)")
            self.plot.setLabel("left", "Midpoint y (coordinate pixels)")

        self.run(compute, complete)


class LocalisationRenderWidget(AnalysisTool):
    """Picasso density preview with TIFF and JSON geometry export."""

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.settings_form = SettingsForm(RenderSettings)
        old = self.settings_form.controls["blur_method"]
        self.blur = combo(["none", "smooth", "convolve", "gaussian", "gaussian_iso"])
        self.settings_form.layout().replaceWidget(old, self.blur)
        old.deleteLater()
        self.settings_form.controls["blur_method"] = self.blur
        self.blur.currentTextChanged.connect(self.settings_form.changed)
        self.form.addWidget(self.settings_form)
        self.settings_form.changed.connect(self.invalidate)
        self.image_item = None
        self.button("Preview density image", self.compute)
        self.button("Export TIFF and geometry…", self.export_image)

    @logged
    def invalidate(self, *_: Any) -> None:
        """Invalidate.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        super().invalidate()
        self.image_item = None

    @logged
    def compute(self) -> None:
        """Compute a detached result using the current controls."""
        session, settings = self.session, self.settings_form.value()
        a, c, space = session.acquisition_id, session.channel, session.space

        def compute(model: Experiment, progress: ProgressCallback | None, cancel: Any) -> RenderResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return rendering.render_localizations(
                model, a, c, settings=settings, space=space, progress=progress, cancel_event=cancel
            )

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            import pyqtgraph as pg
            from qtpy.QtCore import QRectF

            self.ready(
                result,
                f"{result.count} localisations; {result.pixel_spacing:g} coordinate pixels/output pixel",
            )
            self.plot.clear()
            self.image_item = pg.ImageItem(result.image, axisOrder="row-major")
            self.plot.addItem(self.image_item)
            x, y = result.origin
            self.image_item.setRect(
                QRectF(
                    x,
                    y,
                    result.image.shape[1] * result.pixel_spacing,
                    result.image.shape[0] * result.pixel_spacing,
                )
            )
            self.plot.setAspectLocked(True)
            self.plot.autoRange()

        self.run(compute, complete)

    @logged
    def export_image(self) -> None:
        """Export image."""
        import tifffile

        from ..files import atomic_files

        self.require_preview()
        path, _ = QFileDialog.getSaveFileName(self, "Export density image", "", "TIFF (*.tif)")
        if path:
            result = self.preview
            metadata = settings_metadata(result)
            metadata.update(origin=result.origin, pixel_spacing=result.pixel_spacing, count=result.count)
            sidecar = Path(path).with_suffix(".json")
            with atomic_files([path, sidecar]) as (image_path, metadata_path):
                tifffile.imwrite(image_path, result.image, metadata=metadata)
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
