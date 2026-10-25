"""Small native Qt forms and shared session controls."""

from __future__ import annotations

from dataclasses import MISSING, fields
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtCore import QPointF, QSignalBlocker, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPixmap, QPolygonF
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .._logging import logged
from ..staging import state_token

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike
    from qtpy.QtCore import QEvent

    from ..types import Segmentation
    from .session import UiSession


@logged
def combo(values: Any) -> QComboBox:
    """Combo.

    Args:
        values: Input values to validate, convert or display.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    widget = QComboBox()
    widget.addItems(list(values))
    return widget


class SettingsForm(QWidget):
    changed = Signal()

    def __init__(self, record: type[Any], parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            record: Dataclass type defining the form fields and their defaults.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(parent)
        self.record, self.controls = record, {}
        form = QFormLayout(self)
        for field in fields(record):
            default = field.default
            if default is MISSING:
                control = QLineEdit()
                control.setPlaceholderText("Required camera calibration")
                control.textChanged.connect(self.changed)
            elif isinstance(default, Enum):
                control = combo([str(v) for v in type(default)])
                control.setCurrentText(str(default))
                control.currentTextChanged.connect(self.changed)
            elif isinstance(default, bool):
                control = QCheckBox()
                control.setChecked(default)
                control.toggled.connect(self.changed)
            elif isinstance(default, int):
                control = QSpinBox()
                control.setRange(0, 2_000_000_000)
                control.setValue(default)
                control.valueChanged.connect(self.changed)
            elif isinstance(default, float) or default is None:
                control = QDoubleSpinBox()
                control.setDecimals(6)
                control.setRange(-1e12, 1e12)
                if default is None:
                    control.setMinimum(0)
                    control.setSpecialValueText("Automatic")
                control.setValue(default or 0)
                control.valueChanged.connect(self.changed)
            else:
                control = QLineEdit(str(default))
                control.textChanged.connect(self.changed)
            self.controls[field.name] = control
            labels = {
                "baseline": "Baseline (ADU)",
                "sensitivity": "Sensitivity (electrons/ADU)",
                "gain": "Camera gain",
                "box_size": "Box width (odd pixels)",
                "segment_frames": "Frames per drift segment",
                "diameter": "Diameter (pixels)",
                "min_size": "Minimum area (pixels)",
                "pretrained_model": "Cellpose model / weights",
            }
            label = labels.get(field.name, field.name.replace("_", " ").capitalize())
            form.addRow(label, control)

    @logged
    def value(self) -> Any:
        """Value.

        Returns:
            Validated dataclass instance populated from the current controls.
        """
        values = {}
        for field in fields(self.record):
            control = self.controls[field.name]
            if isinstance(control, QCheckBox):
                value = control.isChecked()
            elif isinstance(control, QComboBox):
                value = control.currentText()
                if isinstance(field.default, Enum):
                    value = type(field.default)(value)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                value = control.value()
                if field.default is None and value == 0:
                    value = None
            else:
                value = control.text().strip()
                if field.default is MISSING:
                    value = float(value)
            values[field.name] = value
        return self.record(**values)

    @logged
    def populate(self, values: Any) -> None:
        """Populate.

        Args:
            values: Input values to validate, convert or display.
        """
        for name, value in values.items():
            control = self.controls.get(name)
            if control is None:
                continue
            if isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, QComboBox):
                control.setCurrentText(str(value))
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.setValue(value or 0)
            else:
                control.setText(str(value))


class SelectionPanel(QWidget):
    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(parent)
        self.session = session
        form = QFormLayout(self)
        self.acquisition, self.channel, self.sequence = QComboBox(), QComboBox(), QComboBox()
        self.space = combo(["raw", "registered", "corrected"])
        self.frame = QSpinBox()
        self.frame.setRange(0, 0)
        self.frame_label = QLabel()
        self.cpus = QSpinBox()
        self.cpus.setRange(0, 61 if __import__("os").name == "nt" else 4096)
        self.cpus.setSpecialValueText("Automatic")
        for label, control in (
            ("Acquisition", self.acquisition),
            ("Channel", self.channel),
            ("Sequence", self.sequence),
            ("Coordinates", self.space),
            ("Channel frame (zero-based)", self.frame),
            ("Source", self.frame_label),
            ("CPU processes", self.cpus),
        ):
            form.addRow(label, control)
        self.acquisition.currentTextChanged.connect(self._acquisition)
        self.channel.currentTextChanged.connect(self._selection)
        self.space.currentTextChanged.connect(self._selection)
        self.sequence.currentIndexChanged.connect(self._selection)
        self.frame.valueChanged.connect(self._frame)
        self.cpus.valueChanged.connect(lambda v: setattr(session, "num_cpus", v or None))
        session.selection_changed.connect(self.refresh)
        session.experiment_changed.connect(self.refresh)
        session.frame_changed.connect(self.frame_refresh)
        session.options_changed.connect(self.options_refresh)
        self.refresh()

    @logged
    def refresh(self, *_: Any) -> None:
        """Refresh controls from the current shared model and selection.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        blockers = [QSignalBlocker(w) for w in (self.acquisition, self.channel, self.space, self.sequence)]
        model = self.session.experiment
        self.acquisition.clear()
        self.acquisition.addItems(list(model.acquisitions))
        self.acquisition.setCurrentText(self.session.acquisition_id or "")
        self.channel.clear()
        if self.session.acquisition_id in model.acquisitions:
            self.channel.addItems(list(model.acquisitions[self.session.acquisition_id].channels))
        self.channel.setCurrentText(self.session.channel or "")
        self.space.setCurrentText(str(self.session.space))
        self.sequence.clear()
        self.sequence.addItem("Independent acquisition", None)
        for key, seq in model.sequences.items():
            if self.session.acquisition_id in seq.offsets:
                self.sequence.addItem(key, key)
        self.sequence.setCurrentIndex(max(0, self.sequence.findData(self.session.sequence_id)))
        del blockers
        self.options_refresh()
        self.frame_refresh()

    @logged
    def options_refresh(self) -> None:
        """Options refresh."""
        blocker = QSignalBlocker(self.cpus)
        self.cpus.setValue(self.session.num_cpus or 0)
        del blocker

    @logged
    def frame_refresh(self) -> None:
        """Frame refresh."""
        blocker = QSignalBlocker(self.frame)
        if self.session.acquisition_id is not None:
            view = self.session.experiment.acquisitions[self.session.acquisition_id].channels[
                self.session.channel
            ]
            self.frame.setRange(0, max(0, len(view) - 1))
            if len(view):
                index = view.channel_index(self.session.camera_frame)
                self.frame.setValue(index)
                info = view.frame_info(index)
                text = f"Camera frame {info.camera_frame}"
                if info.timestamp is not None:
                    text += f"; {info.timestamp:g} s"
                self.frame_label.setText(text)
        else:
            self.frame.setRange(0, 0)
            self.frame_label.setText("No acquisition selected")
        del blocker

    def _acquisition(self, name: str) -> None:
        """Acquisition.

        Args:
            name: Identifier or field name used to address this value.
        """
        if name:
            channel = next(iter(self.session.experiment.acquisitions[name].channels))
            self.session.select(name, channel, space=self.space.currentText())

    def _selection(self, *_: Any) -> None:
        """Selection.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        if self.acquisition.currentText() and self.channel.currentText():
            self.session.select(
                self.acquisition.currentText(),
                self.channel.currentText(),
                space=self.space.currentText(),
                sequence_id=self.sequence.currentData(),
            )

    def _frame(self, index: int) -> np.ndarray:
        """Read one original image frame from the bound source.

        Args:
            index: Zero-based item or logical-channel frame index.

        Returns:
            Computed NumPy array value.
        """
        if self.session.acquisition_id is not None:
            view = self.session.experiment.acquisitions[self.session.acquisition_id].channels[
                self.session.channel
            ]
            if len(view):
                self.session.set_frame(int(view.source_frames[index]))


class ToolWidget(QWidget):
    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(parent)
        self.session, self.preview, self.preview_token = session, None, None
        layout = QVBoxLayout(self)
        self.body = QWidget()
        self.form = QVBoxLayout(self.body)
        self.selection = SelectionPanel(session)
        self.form.addWidget(self.selection)
        layout.addWidget(self.body)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(130)
        layout.addWidget(self.summary)
        self.status, self.bar = QLabel("Ready"), QProgressBar()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)
        layout.addWidget(self.status)
        footer = QHBoxLayout()
        footer.addWidget(self.bar)
        footer.addWidget(self.cancel_button)
        layout.addLayout(footer)
        session.busy_changed.connect(self._busy)
        session.progress.connect(self._progress)
        session.error.connect(self.show_error)
        session.selection_changed.connect(self.invalidate)
        session.experiment_changed.connect(self.invalidate)
        session.results_changed.connect(self.invalidate)
        self._busy(session.busy)

    @logged
    def button(self, text: str, callback: Callable[..., Any]) -> QPushButton:
        """Button.

        Args:
            text: Text supplied by the control or caller.
            callback: Callable invoked by the control or operation.

        Returns:
            Computed QPushButton value.
        """
        button = QPushButton(text)
        button.clicked.connect(lambda _checked=False: self.invoke(callback))
        self.form.addWidget(button)
        return button

    @logged
    def invoke(self, callback: Callable[..., Any]) -> None:
        """Invoke.

        Args:
            callback: Callable invoked by the control or operation.
        """
        try:
            return callback()
        except Exception as exc:
            self.show_error(f"{type(exc).__name__}: {exc}")
            return None

    @logged
    def show_error(self, message: Any) -> None:
        """Show error.

        Args:
            message: Message used by this operation.
        """
        self.status.setText(message)
        self.status.setWordWrap(True)

    def _busy(self, busy: Any) -> None:
        """Busy.

        Args:
            busy: Busy used by this operation.
        """
        self.body.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        if busy:
            self.status.setText("Running")
            self.bar.setValue(0)
        elif self.status.text() in ("Running", "Cancelling…"):
            self.status.setText("Ready")

    def _progress(self, update: Any) -> None:
        """Progress.

        Args:
            update: Update used by this operation.
        """
        self.bar.setValue(round(update.percent))
        self.status.setText(update.description)

    def _cancel(self) -> None:
        """Cancel."""
        self.session.cancel()
        self.status.setText("Cancelling…")

    @logged
    def invalidate(self, *_: Any) -> None:
        """Invalidate.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        self.preview, self.preview_token = None, None

    @logged
    def ready(self, value: Any, text: str) -> None:
        """Ready.

        Args:
            value: Value supplied by the caller or connected control.
            text: Text supplied by the control or caller.
        """
        self.preview, self.preview_token = value, state_token(self.session.experiment)
        self.summary.setPlainText(text)
        self.status.setText("Preview ready; active results unchanged")
        self.session.preview_changed.emit(value)

    @logged
    def require_preview(self) -> None:
        """Require preview."""
        if self.preview is None:
            raise ValueError("Compute a preview first")
        if state_token(self.session.experiment) != self.preview_token:
            self.invalidate()
            raise ValueError("Inputs changed; compute a new preview")


class CurvePreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(parent)
        self.frames, self.values = np.array([]), np.empty((0, 2))
        self.setMinimumHeight(140)

    @logged
    def set_curve(self, frames: ArrayLike | None, values: Any) -> None:
        """Set curve.

        Args:
            frames: Original camera-frame indices for the selected samples.
            values: Input values to validate, convert or display.
        """
        self.frames, self.values = np.asarray(frames), np.asarray(values)
        self.update()

    @logged
    def paintEvent(self, event: QEvent) -> None:
        """Paintevent.

        Args:
            event: Event used by this operation.
        """
        painter = QPainter(self)
        if len(self.frames) < 2:
            painter.drawText(self.rect(), Qt.AlignCenter, "Drift preview: dx (red), dy (blue)")
            return
        values = self.values
        low, high = float(values.min()), float(values.max())
        x = 8 + (self.width() - 16) * (self.frames - self.frames.min()) / max(1, np.ptp(self.frames))
        for axis, color in enumerate(("#ef5350", "#42a5f5")):
            y = self.height() - 8 - (self.height() - 16) * (values[:, axis] - low) / max(1e-9, high - low)
            painter.setPen(QColor(color))
            painter.drawPolyline(QPolygonF([QPointF(float(a), float(b)) for a, b in zip(x, y, strict=True)]))


class MaskPreview(QLabel):
    @logged
    def show_mask(self, segmentation: Segmentation) -> None:
        """Show mask.

        Args:
            segmentation: Segmentation used by this operation.
        """
        labels = segmentation.labels
        source = segmentation.source_image
        gray = np.zeros(labels.shape) if source is None else np.asarray(source, dtype=float)
        low, high = np.nanpercentile(gray, [1, 99])
        gray = np.nan_to_num(np.clip((gray - low) / max(1e-9, high - low), 0, 1)) * 255
        rgb = np.repeat(gray[..., None], 3, axis=2)
        colors = np.stack([(labels * p) % 255 for p in (67, 131, 197)], axis=-1)
        rgb[labels > 0] = 0.4 * rgb[labels > 0] + 0.6 * colors[labels > 0]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(image).scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
