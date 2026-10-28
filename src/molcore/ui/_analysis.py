"""Plot, table, export and host integration helpers for detached analysis tools."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFileDialog, QFormLayout, QLineEdit, QTableWidget, QTableWidgetItem

from .._logging import logged
from ._common import ToolWidget

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from numpy.typing import ArrayLike
    from qtpy.QtWidgets import QWidget

    from .session import UiSession


@logged
def identifiers(text: str) -> list[int] | None:
    """Parse optional explicit stable IDs, never floating point IDs.

    Args:
        text: Text supplied by the control or caller.

    Returns:
        Computed list[int] | None value.
    """
    return [int(value) for value in text.replace(",", " ").split()] or None


@logged
def export_columns(path: str | Path, columns: Mapping[str, np.ndarray]) -> None:
    """Write every row, including headers for an empty result, as UTF-8 CSV.

    Args:
        path: Input or output filesystem path.
        columns: Named, equally sized detached measurement arrays.
    """
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


class AnalysisTool(ToolWidget):
    """Base for embeddable tools emitting detached results and host selections.

    Signals:
        result_ready: Detached result available for a host overlay or inspector.
        selection_requested: Mapping containing acquisition, channel, space,
            entity type and stable IDs. The host decides how to display it.
    """

    result_ready = Signal(object)
    selection_requested = Signal(object)

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned dataset, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        self._generation = 0
        super().__init__(session, parent)
        self.fields = QFormLayout()
        self.form.addLayout(self.fields)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(180)
        self.form.addWidget(self.plot)
        self.table = QTableWidget()
        self.table.setMaximumHeight(170)
        self.form.addWidget(self.table)
        self._columns = None

    @logged
    def field(self, label: str, control: QWidget) -> QWidget:
        """Field.

        Args:
            label: Visible label describing the control or plot axis.
            control: Qt control bound to this form field.

        Returns:
            Computed QWidget value.
        """
        self.fields.addRow(label, control)
        for signal in ("currentIndexChanged", "valueChanged", "textChanged", "toggled"):
            if hasattr(control, signal):
                getattr(control, signal).connect(self.invalidate)
                break
        return control

    @logged
    def invalidate(self, *_: Any) -> None:
        """Invalidate.

        Args:
            *_: Additional positional values supplied by the caller.
        """
        super().invalidate()
        self._generation += 1
        self._columns = None
        if hasattr(self, "plot"):
            self.plot.clear()
            self.table.setRowCount(0)

    @logged
    def run(self, compute: Callable[..., Any], complete: Any) -> None:
        """Discard completions if settings/selection changed during a job.

        Args:
            compute: Background callable receiving a detached model, progress callback and cancellation event.
            complete: Complete used by this operation.
        """
        generation = self._generation

        def deliver(result: Any) -> None:
            """Deliver.

            Args:
                result: Detached computation result to inspect or apply.
            """
            if generation != self._generation:
                self.status.setText("Inputs changed; compute a new preview")
                return
            if is_dataclass(result) and hasattr(result, "expected"):
                from ..staging import state_token

                result = replace(result, expected=state_token(self.session.dataset))
            complete(result)
            self.result_ready.emit(self.preview if self.preview is not None else result)

        self.session.submit(compute, completed=deliver)

    @logged
    def show_columns(self, columns: Mapping[str, np.ndarray]) -> None:
        """Show columns.

        Args:
            columns: Named, equally sized detached measurement arrays.
        """
        self._columns = columns
        names = list(columns)
        count = len(next(iter(columns.values()), []))
        self.table.setColumnCount(len(names))
        self.table.setHorizontalHeaderLabels(names)
        self.table.setRowCount(min(count, 500))
        for col, name in enumerate(names):
            for row, value in enumerate(columns[name][:500]):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self.table.setToolTip(f"Showing up to 500 of {count} rows; export includes all rows")

    @logged
    def export_csv(self) -> None:
        """Export all selected result rows through a native file dialog."""
        self.require_preview()
        if self._columns is None:
            raise ValueError("Compute a result first")
        path, _ = QFileDialog.getSaveFileName(self, "Export all result rows", "", "CSV (*.csv)")
        if path:
            export_columns(path, self._columns)

    @logged
    def request_selection(self, ids: ArrayLike, entity: str) -> None:
        """Request selection.

        Args:
            ids: Stable IDs of the source entities to select.
            entity: Either localizations or tracks.
        """
        model = self.session.dataset
        rows = (
            model.tracks.query(track_id=ids, space="raw")
            if entity == "tracks"
            else model.localizations.query(ids=ids, space="raw")
        )
        for acquisition, channel in sorted(set(zip(rows["acquisition_id"], rows["channel"], strict=True))):
            mask = (rows["acquisition_id"] == acquisition) & (rows["channel"] == channel)
            self.selection_requested.emit(
                dict(
                    acquisition_id=acquisition,
                    channel=channel,
                    space=self.session.space,
                    entity=entity,
                    ids=np.unique(rows["track_id" if entity == "tracks" else "id"][mask]),
                )
            )

    @logged
    def histogram(self, values: Any, label: str) -> None:
        """Histogram.

        Args:
            values: Input values to validate, convert or display.
            label: Visible label describing the control or plot axis.
        """
        self.plot.clear()
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if len(values):
            counts, edges = np.histogram(values, bins=min(40, max(1, int(np.sqrt(len(values))))))
            self.plot.plot(edges, counts, stepMode="center", fillLevel=0, brush=(70, 130, 200, 90))
        self.plot.setLabel("bottom", label)
        self.plot.setLabel("left", "Count")


@logged
def numeric_metrics(rows: dict[str, np.ndarray]) -> list[str]:
    """Numeric metrics.

    Args:
        rows: Named measurement arrays retaining stable IDs and source-frame provenance.

    Returns:
        Computed list[str] value.
    """
    return [k for k, v in rows.items() if v.dtype.kind in "iuf" and k not in ("id", "track_id")]


@logged
def refresh_combo(control: QWidget, values: Any) -> None:
    """Refresh combo.

    Args:
        control: Qt control bound to this form field.
        values: Input values to validate, convert or display.
    """
    from qtpy.QtCore import QSignalBlocker

    old = control.currentText()
    blocker = QSignalBlocker(control)
    control.clear()
    control.addItems([str(v) for v in values])
    if old in [str(v) for v in values]:
        control.setCurrentText(old)
    del blocker


@logged
def partition_rows(session: UiSession, *, tracks: bool = False) -> dict[str, np.ndarray]:
    """Partition rows.

    Args:
        session: Shared host-owned dataset, selection and background-job coordinator.
        tracks: Track IDs to invalidate, or whether this operation selects whole tracks.

    Returns:
        Detached named result columns.
    """
    if session.acquisition_id is None:
        return {}
    table = session.dataset.tracks if tracks else session.dataset.localizations
    return table.query(acquisition_id=session.acquisition_id, channel=session.channel, space=session.space)


class IdTool(AnalysisTool):
    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned dataset, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        self.ids = self.field("Stable IDs (blank: selected partition)", QLineEdit())

    @logged
    def selected_ids(self, *, tracks: bool = False) -> np.ndarray:
        """Selected ids.

        Args:
            tracks: Track IDs to invalidate, or whether this operation selects whole tracks.

        Returns:
            Computed NumPy array value.
        """
        rows = partition_rows(self.session, tracks=tracks)
        available = np.unique(rows.get("track_id" if tracks else "id", np.array([], dtype=np.uint64)))
        selected = identifiers(self.ids.text())
        if selected is not None and not np.isin(selected, available).all():
            raise ValueError("IDs must belong to the selected acquisition/channel")
        return available if selected is None else np.asarray(selected, dtype=np.uint64)


@logged
def settings_metadata(result: Any) -> dict[str, Any]:
    """JSON-compatible common export metadata.

    Args:
        result: Detached computation result to inspect or apply.

    Returns:
        Computed dict[str, Any] value.
    """
    return dict(
        acquisition_id=result.acquisition_id,
        channel=result.channel,
        space=str(result.space),
        settings=asdict(result.settings) if is_dataclass(result.settings) else result.settings,
    )
