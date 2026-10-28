"""Fiducial registration previews for successive surface-probing rounds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QSignalBlocker, Signal
from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QLineEdit

from ..staging import experiment_state_token
from ..transforms import project_points
from ..types import DatasetAlignmentSettings, StaleResultError
from ._common import ToolWidget, combo

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ..types import Experiment, ProgressCallback
    from .session import UiSession


class DatasetAlignmentWidget(ToolWidget):
    """Preview fiducial pairs and install one round-to-reference registration."""

    result_ready = Signal(object)

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        super().__init__(session, parent)
        self.reference_dataset, self.source_dataset = combo([]), combo([])
        self.reference_acquisition, self.source_acquisition = combo([]), combo([])
        self.reference_channel, self.source_channel = combo([]), combo([])
        self.reference_ids, self.source_ids = QLineEdit(), QLineEdit()
        self.model = combo(["affine", "translation", "homography"])
        self.radius, self.threshold = QDoubleSpinBox(), QDoubleSpinBox()
        for control, value in ((self.radius, 5), (self.threshold, 1)):
            control.setRange(0.000001, 1e9)
            control.setDecimals(6)
            control.setValue(value)
        self.image_overlay = QCheckBox("Include image overlay (requires original pixels)")
        fields = QFormLayout()
        for label, control in (
            ("Reference round", self.reference_dataset),
            ("Reference acquisition", self.reference_acquisition),
            ("Reference channel", self.reference_channel),
            ("Source round", self.source_dataset),
            ("Source acquisition", self.source_acquisition),
            ("Source channel", self.source_channel),
            ("Paired source IDs (optional)", self.source_ids),
            ("Paired reference IDs (optional)", self.reference_ids),
            ("Transform model", self.model),
            ("Matching radius (pixels)", self.radius),
            ("Inlier threshold (reference pixels)", self.threshold),
            ("Preview", self.image_overlay),
        ):
            fields.addRow(label, control)
            for name in ("currentIndexChanged", "textChanged", "valueChanged", "toggled"):
                if hasattr(control, name):
                    getattr(control, name).connect(self.invalidate)
                    break
        self.form.addLayout(fields)
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.setMinimumHeight(220)
        self.form.addWidget(self.plot)
        self.source_dataset.currentIndexChanged.connect(lambda: self._partitions("source"))
        self.reference_dataset.currentIndexChanged.connect(lambda: self._partitions("reference"))
        self.source_acquisition.currentIndexChanged.connect(lambda: self._channels("source"))
        self.reference_acquisition.currentIndexChanged.connect(lambda: self._channels("reference"))
        session.experiment_changed.connect(self._rounds)
        session.dataset_changed.connect(self._rounds)
        session.results_changed.connect(self._rounds)
        self._rounds()
        self.button("Estimate dataset alignment preview", self.estimate)
        self.button("Apply dataset alignment", self.apply_preview)
        self.button("Reset selected source alignment", self.reset_alignment)
        self.button("Reset all dataset alignments", self.reset_all)

    def _rounds(self, *_: Any) -> None:
        keys = list(self.session.experiment.datasets)
        for kind in ("reference", "source"):
            control = getattr(self, kind + "_dataset")
            previous = control.currentData()
            blocker = QSignalBlocker(control)
            control.clear()
            for key in keys:
                control.addItem(self.session.experiment.datasets[key].name, key)
            default = keys[0 if kind == "reference" else min(1, len(keys) - 1)] if keys else None
            control.setCurrentIndex(control.findData(previous if previous in keys else default))
            del blocker
            self._partitions(kind)

    def _partitions(self, kind: str) -> None:
        dataset = self.session.experiment.datasets.get(getattr(self, kind + "_dataset").currentData())
        control = getattr(self, kind + "_acquisition")
        previous = control.currentText()
        blocker = QSignalBlocker(control)
        control.clear()
        if dataset is not None:
            control.addItems(list(dataset.acquisitions))
            if previous in dataset.acquisitions:
                control.setCurrentText(previous)
        del blocker
        self._channels(kind)

    def _channels(self, kind: str) -> None:
        dataset = self.session.experiment.datasets.get(getattr(self, kind + "_dataset").currentData())
        acquisition = getattr(self, kind + "_acquisition").currentText()
        control = getattr(self, kind + "_channel")
        previous = control.currentText()
        control.clear()
        if dataset is not None and acquisition in dataset.acquisitions:
            control.addItems(list(dataset.acquisitions[acquisition].channels))
            if previous in dataset.acquisitions[acquisition].channels:
                control.setCurrentText(previous)

    def estimate(self) -> None:
        """Estimate a detached registration and show matched fiducials."""
        source, reference = self.source_dataset.currentData(), self.reference_dataset.currentData()
        options = dict(
            source_acquisition_id=self.source_acquisition.currentText(),
            source_channel=self.source_channel.currentText(),
            reference_acquisition_id=self.reference_acquisition.currentText(),
            reference_channel=self.reference_channel.currentText(),
            source_ids=[int(v) for v in self.source_ids.text().replace(",", " ").split()] or None,
            reference_ids=[int(v) for v in self.reference_ids.text().replace(",", " ").split()] or None,
            settings=DatasetAlignmentSettings(
                self.model.currentText(), self.radius.value(), self.threshold.value()
            ),
            num_cpus=self.session.num_cpus,
        )
        expected = experiment_state_token(self.session.experiment)
        include_images = self.image_overlay.isChecked()

        def compute(experiment: Experiment, progress: ProgressCallback | None, cancel: Any) -> tuple:
            """Compute.

            Args:
                experiment: Experiment containing the probing rounds.
                progress: Optional structured progress callback.
                cancel: Cooperative cancellation event.

            Returns:
                Result of this operation.
            """
            result = experiment.alignment.estimate(
                source, reference, **options, progress=progress, cancel_event=cancel
            )
            images = None
            if include_images:
                experiment.alignment.reset()
                experiment.alignment.apply([result])

                def first_image(key: str, acquisition: str, channel: str) -> np.ndarray:
                    """First image.

                    Args:
                        key: Dataset identifier.
                        acquisition: Acquisition identifier.
                        channel: Logical channel identifier.

                    Returns:
                        Result of this operation.
                    """
                    view = experiment.datasets[key].acquisitions[acquisition].channels[channel]
                    if not len(view):
                        raise ValueError("Selected channel has no image frames")
                    return experiment.alignment.image(key, acquisition, channel, int(view.source_frames[0]))

                images = (
                    first_image(source, result.source_acquisition_id, result.source_channel),
                    first_image(reference, result.reference_acquisition_id, result.reference_channel),
                )
            return result, images

        def complete(value: tuple) -> None:
            """Complete.

            Args:
                value: Detached result and optional preview image overlays.
            """
            result, images = value
            self.ready(
                result,
                f"{int(result.inliers.sum())}/{len(result.inliers)} inliers; RMS {result.rms_residual:.4g} px\n{result.matrix}",
            )
            self._experiment_token = expected
            self.plot.clear()
            if images is not None:
                for image, opacity in zip(images, (0.5, 0.5), strict=True):
                    item = pg.ImageItem(image, axisOrder="row-major")
                    item.setOpacity(opacity)
                    self.plot.addItem(item)
            source_rows = self.session.experiment.datasets[source].localizations.query(
                ids=result.source_ids, space="corrected"
            )
            reference_rows = self.session.experiment.datasets[reference].localizations.query(
                ids=result.reference_ids, space="corrected"
            )
            points = project_points(result.matrix, np.column_stack([source_rows["x"], source_rows["y"]]))
            self.plot.plot(points[:, 0], points[:, 1], pen=None, symbol="o", symbolBrush=(0, 180, 255))
            self.plot.plot(
                reference_rows["x"], reference_rows["y"], pen=None, symbol="+", symbolPen=(255, 170, 0)
            )
            self.result_ready.emit(result)

        self.session.submit(compute, completed=complete, experiment_scope=True)

    def apply_preview(self) -> None:
        """Install a fresh preview without modifying dataset-local pixels."""
        if self.preview is None:
            raise ValueError("Estimate an alignment preview first")
        if self._experiment_token != experiment_state_token(self.session.experiment):
            self.invalidate()
            raise StaleResultError("Inputs changed; estimate a new alignment preview")
        self.session.experiment.alignment.apply([self.preview])
        self.session.changed()
        self.status.setText("Dataset alignment applied; original images unchanged")

    def reset_alignment(self) -> None:
        """Clear the selected source round registration."""
        self.session.experiment.alignment.reset(self.source_dataset.currentData())
        self.session.changed()

    def reset_all(self) -> None:
        """Clear every round-to-reference registration."""
        self.session.experiment.alignment.reset()
        self.session.changed()
