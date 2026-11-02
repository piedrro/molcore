"""Convenience container for shared tools; ownership remains with the host."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout, QWidget

from .analysis import (
    ClusteringWidget,
    ColocalisationWidget,
    LocalisationRenderWidget,
    ResultsFilterWidget,
)
from .dataset_alignment import DatasetAlignmentWidget
from .dynamics import PhotometryWidget, TrackingWidget, TrajectoryStatisticsWidget
from .files import FileToolsWidget
from .processing import (
    ChannelAlignmentWidget,
    LocalisationWidget,
    SegmentationWidget,
    UndriftingWidget,
)
from .traces import TraceViewerWidget

if TYPE_CHECKING:
    from .session import UiSession


class MolCoreToolsWidget(QWidget):
    """Embed all tools against one borrowed UiSession without creating an app."""

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned dataset, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(parent)
        self.session = session
        self.tabs = QTabWidget()
        self.widgets = {}
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        for title, cls in (
            ("Files", FileToolsWidget),
            ("Localisation", LocalisationWidget),
            ("Drift", UndriftingWidget),
            ("Alignment", ChannelAlignmentWidget),
            ("Dataset alignment", DatasetAlignmentWidget),
            ("Segmentation", SegmentationWidget),
            ("Tracking", TrackingWidget),
            ("Photometry", PhotometryWidget),
            ("Traces", TraceViewerWidget),
            ("Trajectory statistics", TrajectoryStatisticsWidget),
            ("Filter", ResultsFilterWidget),
            ("Render", LocalisationRenderWidget),
            ("Colocalisation", ColocalisationWidget),
            ("Clustering", ClusteringWidget),
        ):
            widget = cls(session)
            self.widgets[title] = widget
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            self.tabs.addTab(scroll, title)
