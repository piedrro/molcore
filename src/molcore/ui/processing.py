"""Embeddable scientific processing tools backed by MolCore models and CPU jobs."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
)

from .. import alignment, cellpose, localisation, undrifting
from .._logging import logged
from ..processing import apply_alignment, apply_drift, apply_localisation, apply_segmentations
from ..staging import state_token
from ..types.processing import (
    CameraSettings,
    CellposeSettings,
    DriftSettings,
    LocalisationSettings,
    SegmentationRequest,
)
from ._common import CurvePreview, MaskPreview, SettingsForm, ToolWidget, combo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qtpy.QtWidgets import QWidget

    from ..types import (
        AlignmentResult,
        DetectionResult,
        DriftResult,
        LocalisationResult,
        ProgressCallback,
        Segmentation,
    )
    from .session import UiSession


class LocalisationWidget(ToolWidget):
    """Detect and fit a shared acquisition/channel with calibrated camera settings.

    Args:
        session: Shared UiSession providing the selected acquisition and CPU count.
        parent: Optional host widget.
    """

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.detections, self._detection_token = None, None
        self.settings_form = SettingsForm(LocalisationSettings)
        self.threshold = self.settings_form.controls["min_net_gradient"]
        self.camera_form = SettingsForm(CameraSettings)
        self.form.addWidget(QLabel("Detection and fitting"))
        self.form.addWidget(self.settings_form)
        self.form.addWidget(QLabel("Camera calibration (required for fitting)"))
        self.form.addWidget(self.camera_form)
        self.settings_form.changed.connect(self.invalidate)
        self.camera_form.changed.connect(self.invalidate)
        self.button("Preview detection — current frame", self.preview_current)
        self.button("Detect — full channel", self.detect_full)
        self.button("Fit cached full-channel detections", self.fit_cached)
        self.button("Detect + fit — replace active partition", self.localise_full)
        session.selection_changed.connect(self._camera)
        self._camera()

    def _camera(self) -> None:
        # Clear previous calibration; never carry a different camera's values over.
        """Camera."""
        for name in ("baseline", "sensitivity", "gain"):
            self.camera_form.controls[name].clear()
        if self.session.acquisition_id is not None:
            acquisition = self.session.experiment.acquisitions[self.session.acquisition_id]
            self.camera_form.populate(acquisition.metadata.get("camera", {}))

    @logged
    def invalidate(self, *_: Any) -> None:
        """Invalidate.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        super().invalidate()
        self.detections, self._detection_token = None, None

    def _detect(self, preview: bool) -> Any:
        """Detect.

        Args:
            preview: Preview used by this operation.

        Returns:
            Result retaining the input format or backend-specific representation.
        """
        settings, session = self.settings_form.value(), self.session
        a, c, cpus = session.acquisition_id, session.channel, session.num_cpus
        frames = (session.camera_frame, session.camera_frame + 1) if preview else None

        def compute(model: str, progress: ProgressCallback | None, cancel: Any) -> DetectionResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return localisation.detect(
                model,
                a,
                c,
                settings=settings,
                frame_range=frames,
                num_cpus=cpus,
                progress=progress,
                cancel_event=cancel,
            )

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            self.ready(
                result,
                f"{len(result.columns['frame'])} candidates; "
                + ("current-frame preview" if preview else "full-channel detections cached for fitting"),
            )
            if not preview:
                self.detections, self._detection_token = result, state_token(session.experiment)

        session.submit(compute, completed=complete)

    @logged
    def preview_current(self) -> None:
        """Preview candidates at the selected camera frame without changing data."""
        self._detect(True)

    @logged
    def detect_full(self) -> None:
        """Cache full-channel detections for a subsequent independent fitting job."""
        self._detect(False)

    @logged
    def fit_cached(self) -> None:
        """Fit cached full-channel detections and atomically replace their partition."""
        if self.detections is None or self._detection_token != state_token(self.session.experiment):
            raise ValueError("Detect the full channel again before fitting")
        self._fit(self.detections)

    @logged
    def localise_full(self) -> None:
        """Detect and fit the full selected channel using the configured CPU count."""
        self._fit(None)

    def _fit(self, detections: Any) -> Any:
        """Fit.

        Args:
            detections: Detections used by this operation.

        Returns:
            Result retaining the input format or backend-specific representation.
        """
        settings, camera, session = self.settings_form.value(), self.camera_form.value(), self.session
        a, c, cpus = session.acquisition_id, session.channel, session.num_cpus

        def compute(model: str, progress: ProgressCallback | None, cancel: Any) -> LocalisationResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            options = dict(
                settings=settings,
                camera=camera,
                apply=False,
                num_cpus=cpus,
                progress=progress,
                cancel_event=cancel,
            )
            if detections is None:
                return localisation.localise(model, a, c, **options)
            view = model.acquisitions[a].channels[c]
            cached = replace(detections, source_token=hash((id(model.acquisitions[a].source), id(view))))
            return localisation.fit(model, cached, **options)

        session.submit(
            compute,
            commit=lambda model, value: apply_localisation(model, value, camera=camera),
            completed=lambda value: self.summary.setPlainText(
                f"Applied {len(value.columns['frame'])} localizations; {value.failed_fits} failed fits"
            ),
        )


class UndriftingWidget(ToolWidget):
    """Estimate RCC/AIM drift and explicitly apply a reference-space preview.

    Args:
        session: Shared UiSession with localization results.
        parent: Optional host widget.
    """

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.settings_form = SettingsForm(DriftSettings)
        self.form.addWidget(self.settings_form)
        self.curve = CurvePreview()
        self.form.addWidget(self.curve)
        self.settings_form.changed.connect(self.invalidate)
        self.button("Estimate drift preview", self.estimate)
        self.button("Apply preview", self.apply_preview)

    @logged
    def estimate(self) -> None:
        """Estimate reference-pixel displacement without altering raw coordinates."""
        session, settings = self.session, self.settings_form.value()
        a, c, cpus = session.acquisition_id, session.channel, session.num_cpus

        def compute(model: str, progress: ProgressCallback | None, cancel: Any) -> DriftResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return undrifting.estimate_drift(
                model, a, c, settings=settings, num_cpus=cpus, progress=progress, cancel_event=cancel
            )

        def complete(result: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                result: Detached computation result to inspect or apply.
            """
            self.ready(
                result,
                f"{result.settings.method.upper()} drift at {len(result.frames)} camera frames\n"
                f"Coverage: {result.frames[0]}–{result.frames[-1]}\nDisplacement in reference pixels",
            )
            self.curve.set_curve(result.frames, result.displacement)

        session.submit(compute, completed=complete)

    @logged
    def apply_preview(self) -> None:
        """Apply the current non-stale drift preview on the GUI thread."""
        self.require_preview()
        apply_drift(
            self.session.experiment, {self.preview.acquisition_id: self.preview}, expected=self.preview_token
        )
        self.session.changed()
        self.status.setText("Drift applied")


class ChannelAlignmentWidget(ToolWidget):
    """Fit and select named translation/affine channel calibrations.

    Args:
        session: Shared UiSession supplying source and target localization tables.
        parent: Optional host widget.
    """

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        form = QFormLayout()
        self.target = combo([])
        self.model = combo(["affine", "translation"])
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.000001, 1e9)
        self.radius.setDecimals(6)
        self.radius.setValue(5)
        self.source_ids, self.target_ids, self.name = QLineEdit(), QLineEdit(), QLineEdit()
        for label, control in (
            ("Reference channel", self.target),
            ("Model", self.model),
            ("Matching radius (pixels)", self.radius),
            ("Paired source IDs (optional)", self.source_ids),
            ("Paired reference IDs (optional)", self.target_ids),
            ("New calibration name", self.name),
        ):
            form.addRow(label, control)
        self.form.addLayout(form)
        for control in (self.target, self.model):
            control.currentTextChanged.connect(self.invalidate)
        self.radius.valueChanged.connect(self.invalidate)
        self.source_ids.textChanged.connect(self.invalidate)
        self.target_ids.textChanged.connect(self.invalidate)
        self.name.textChanged.connect(self.invalidate)
        session.selection_changed.connect(self._targets)
        self._targets()
        self.button("Estimate alignment preview", self.estimate)
        self.button("Apply named calibration", self.apply_preview)

    def _targets(self) -> None:
        """Targets."""
        previous = self.target.currentText()
        self.target.clear()
        if self.session.acquisition_id is not None:
            self.target.addItems(
                list(self.session.experiment.acquisitions[self.session.acquisition_id].channels)
            )
            if previous:
                self.target.setCurrentText(previous)
            if self.target.currentText() == self.session.channel and self.target.count() > 1:
                self.target.setCurrentIndex((self.target.currentIndex() + 1) % self.target.count())

    @logged
    def estimate(self) -> None:
        """Preview a channel fit using paired IDs or bounded mutual-nearest matches."""
        session = self.session
        a, c, target, cpus = (
            session.acquisition_id,
            session.channel,
            self.target.currentText(),
            session.num_cpus,
        )
        options = dict(model=self.model.currentText(), match_radius=self.radius.value())
        for key, control in (("source_ids", self.source_ids), ("target_ids", self.target_ids)):
            options[key] = [int(v) for v in control.text().replace(",", " ").split()] or None
        self._fit_options = dict(
            model=options["model"],
            match_radius=options["match_radius"],
            name=self.name.text().strip() or None,
        )

        def compute(model: str, progress: ProgressCallback | None, cancel: Any) -> AlignmentResult:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return alignment.estimate_alignment(
                model, a, c, target, **options, num_cpus=cpus, progress=progress, cancel_event=cancel
            )

        session.submit(
            compute,
            completed=lambda result: self.ready(
                result,
                f"{len(result.residuals)} pairs; RMS residual {np.sqrt(np.mean(result.residuals**2)):.4g} px\n"
                f"Source → reference matrix:\n{result.reference_matrix}",
            ),
        )

    @logged
    def apply_preview(self) -> None:
        """Apply a validated preview under its fresh calibration name."""
        self.require_preview()
        apply_alignment(
            self.session.experiment, self.preview, expected=self.preview_token, **self._fit_options
        )
        self.session.changed()
        self.status.setText("Calibration applied")


class SegmentationWidget(ToolWidget):
    """Preview and apply static Cellpose masks with explicit source bindings.

    Args:
        session: Shared UiSession providing images, CPU count and coordinate space.
        parent: Optional host widget.
    """

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned experiment, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.settings_form = SettingsForm(CellposeSettings)
        self.form.addWidget(self.settings_form)
        self.settings_form.changed.connect(self.invalidate)
        self.button("Choose local model weights…", self.choose_weights)
        self.form.addWidget(QLabel("GPU uses one worker. Cellpose may download built-in model weights."))
        self.name = QLineEdit()
        self.name.setPlaceholderText("Optional segmentation ID for a single result")
        self.name.textChanged.connect(self.invalidate)
        self.form.addWidget(self.name)
        self.targets = QListWidget()
        self.targets.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.targets.setMaximumHeight(120)
        self.targets.itemSelectionChanged.connect(self.invalidate)
        self.form.addWidget(QLabel("Batch targets (optional; each records its source frame)"))
        self.form.addWidget(self.targets)
        self.mask_preview = MaskPreview()
        self.form.addWidget(self.mask_preview)
        self.button("Preview current frame", self.preview_current)
        self.button("Preview selected batch", self.preview_batch)
        self.button("Apply preview / complete batch", self.apply_preview)
        session.selection_changed.connect(self._targets)
        session.frame_changed.connect(self.invalidate)
        session.experiment_changed.connect(self._targets)
        self._targets()

    def _targets(self, *_: Any) -> None:
        """Targets.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        self.targets.clear()
        for a, acquisition in self.session.experiment.acquisitions.items():
            for c, view in acquisition.channels.items():
                if len(view):
                    item = QListWidgetItem(f"{a} / {c} (first camera frame {view.source_frames[0]})")
                    item.setData(Qt.UserRole, (a, c, int(view.source_frames[0])))
                    self.targets.addItem(item)

    @logged
    def choose_weights(self) -> None:
        """Choose offline Cellpose 4 weights through the host's native file dialog."""
        path, _ = QFileDialog.getOpenFileName(self, "Cellpose 4 weights")
        if path:
            self.settings_form.controls["pretrained_model"].setText(path)

    @logged
    def preview_current(self) -> None:
        """Compute a detached mask at the selected channel/camera frame."""
        session = self.session
        request = SegmentationRequest(
            session.acquisition_id,
            session.channel,
            session.camera_frame,
            self.name.text().strip() or None,
            session.space,
        )
        self._segment([request])

    @logged
    def preview_batch(self) -> None:
        """Compute selected target masks, applying none until the whole batch succeeds."""
        requests = [
            SegmentationRequest(*item.data(Qt.UserRole), space=self.session.space)
            for item in self.targets.selectedItems()
        ]
        if not requests:
            raise ValueError("Select at least one batch target")
        self._segment(requests)

    def _segment(self, requests: Sequence[SegmentationRequest]) -> Any:
        """Segment.

        Args:
            requests: Requests used by this operation.

        Returns:
            Result retaining the input format or backend-specific representation.
        """
        session, settings = self.session, self.settings_form.value()
        cpus = 1 if settings.gpu else session.num_cpus

        def compute(model: str, progress: ProgressCallback | None, cancel: Any) -> dict[str, Segmentation]:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return cellpose.segment_many(
                model,
                requests,
                settings=settings,
                apply=False,
                num_cpus=cpus,
                progress=progress,
                cancel_event=cancel,
            )

        def complete(results: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                results: Detached results keyed by acquisition or segmentation identifier.
            """
            self.ready(
                results,
                "\n".join(
                    f"{s.id}: {len(np.unique(s.labels[s.labels > 0]))} objects; "
                    f"{s.acquisition_id}/{s.channel}, camera {s.frame}, {s.space}"
                    for s in results.values()
                ),
            )
            if results:
                self.mask_preview.show_mask(next(iter(results.values())))

        session.submit(compute, completed=complete)

    @logged
    def apply_preview(self) -> None:
        """Apply every mask in a non-stale successful preview atomically."""
        self.require_preview()
        apply_segmentations(self.session.experiment, self.preview, expected=self.preview_token)
        self.session.changed()
        self.status.setText("Segmentations applied")
