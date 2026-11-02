"""Native file dialogs and complete model-aware import/export workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import tifffile
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
)

from .. import files
from .._logging import logged
from ..interop import read_picasso
from ..legacy import migrate_moltrack, read_moltrack
from ..processing import apply_segmentations
from ..staging import experiment_state_token, transact
from ..types import Calibration, ChannelSpec, Dataset, Experiment, ImagePolicy
from ._common import ToolWidget, combo

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike
    from qtpy.QtWidgets import QWidget

    from ..types import Acquisition, CoordinateSpace, ProgressCallback
    from .session import UiSession


class _ImageOptions(QDialog):
    def __init__(
        self, paths: Sequence[str | Path], parent: QWidget | None = None
    ) -> None:
        """Initialise the object and validate its configuration.

        Args:
            paths: Input or output filesystem paths.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(parent)
        self.setWindowTitle("Image acquisition mapping")
        layout = QVBoxLayout(self)
        info = QLabel("\n".join(str(p) for p in paths))
        info.setWordWrap(True)
        layout.addWidget(info)
        form = QFormLayout()
        self.mode = combo(["single", "fret", "alex", "custom"])
        self.channel = QLineEdit("image")
        self.layout = combo(
            ["donor-left", "acceptor-left", "donor-top", "acceptor-top"]
        )
        self.phase = QSpinBox()
        self.phase.setRange(0, 1)
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0, 1e9)
        self.pixel_size.setDecimals(6)
        self.pixel_size.setSpecialValueText("Unknown")
        self.memory = combo(["lazy", "cache", "ram"])
        self.cache = QSpinBox()
        self.cache.setRange(1, 1_000_000)
        self.cache.setValue(256)
        self.dataset = combo([])
        self.dataset.setEditable(True)
        if len(paths) == 1 and Path(paths[0]).suffix.lower() in (
            ".h5",
            ".hdf5",
            ".hdf",
        ):
            with h5py.File(paths[0], "r") as file:
                file.visititems(
                    lambda name, obj: (
                        self.dataset.addItem(name)
                        if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3)
                        else None
                    )
                )
        self.hdu = QSpinBox()
        self.hdu.setRange(0, 10000)
        self.specs = QPlainTextEdit()
        self.specs.setPlaceholderText(
            '{"image": {"roi": [0,100,0,100], "frame_start": 0, "frame_step": 1}}'
        )
        self.specs.setMaximumHeight(85)
        for label, control in (
            ("Mode", self.mode),
            ("Single channel name", self.channel),
            ("Spatial split", self.layout),
            ("ALEX phase (0=D first, 1=A first)", self.phase),
            ("Pixel size (nm)", self.pixel_size),
            ("Image access", self.memory),
            ("Cache limit (MiB)", self.cache),
            ("HDF5 dataset", self.dataset),
            ("FITS HDU", self.hdu),
            ("Custom channel specifications (JSON)", self.specs),
        ):
            form.addRow(label, control)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "Each file becomes an independent acquisition; sequences are created explicitly."
            )
        )
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @logged
    def options(self) -> dict[str, Any]:
        """Options.

        Returns:
            Validated settings read from the dialog.
        """
        specs = self.specs.toPlainText().strip()
        specs = (
            {k: ChannelSpec(**v) for k, v in json.loads(specs).items()}
            if specs
            else None
        )
        if self.mode.currentText() == "custom" and specs is None:
            raise ValueError("Custom mode requires explicit channel specifications")
        return dict(
            mode=self.mode.currentText(),
            channel=self.channel.text().strip(),
            layout=self.layout.currentText(),
            phase=self.phase.value(),
            pixel_size=self.pixel_size.value() or None,
            channel_specs=specs,
        )

    @logged
    def source_options(self) -> dict[str, Any]:
        """Source options.

        Returns:
            Image reader options read from the dialog.
        """
        return dict(
            dataset=self.dataset.currentText() or None,
            hdu=self.hdu.value(),
            memory=self.memory.currentText(),
            cache_bytes=self.cache.value() * 1024**2,
        )


class _TableMapping(QDialog):
    def __init__(
        self,
        columns: Mapping[str, np.ndarray],
        path: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the object and validate its configuration.

        Args:
            columns: Named, equally sized detached measurement arrays.
            path: Input or output filesystem path.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(parent)
        self.setWindowTitle("Preview table and frame mapping")
        self.columns = columns
        layout = QVBoxLayout(self)
        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        names = list(columns)
        preview.setPlainText(
            "\t".join(names)
            + "\n"
            + "\n".join(
                "\t".join(str(columns[k][i]) for k in names)
                for i in range(min(10, len(next(iter(columns.values()), []))))
            )
        )
        layout.addWidget(preview)
        form = QFormLayout()
        self.mapping = {}
        for canonical in ("frame", "x", "y"):
            control = combo(names)
            if canonical in names:
                control.setCurrentText(canonical)
            self.mapping[canonical] = control
            form.addRow(f"{canonical} column", control)
        self.timeline = combo(["channel", "camera"])
        self.base = combo(["0", "1"])
        self.base.setCurrentText("1" if str(path).lower().endswith(".pos.out") else "0")
        self.units = combo(["pixels", "nm"])
        self.tracks, self.reuse = QCheckBox(), QCheckBox()
        form.addRow("Input frame convention", self.timeline)
        form.addRow("First frame number", self.base)
        form.addRow("Coordinate units", self.units)
        form.addRow("Import as tracks (particle/track_id required)", self.tracks)
        form.addRow("Reuse explicit localization_id anchors", self.reuse)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "Maps to the selected acquisition/channel. Coordinates must be raw channel coordinates."
            )
        )
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @logged
    def values(self) -> tuple:
        """Values.

        Returns:
            Computed tuple value.
        """
        selected = {
            name: control.currentText() for name, control in self.mapping.items()
        }
        if len(set(selected.values())) != 3:
            raise ValueError("Frame, x and y must use distinct columns")
        renamed = {name: self.columns[source] for name, source in selected.items()}
        renamed.update(
            {
                k: v
                for k, v in self.columns.items()
                if k not in selected.values() and k not in renamed
            }
        )
        return renamed, dict(
            timeline=self.timeline.currentText(),
            frame_base=int(self.base.currentText()),
            units=self.units.currentText(),
            tracks=self.tracks.isChecked(),
            reuse_ids=self.reuse.isChecked(),
        )


class FileToolsWidget(ToolWidget):
    """Import, preview, migrate and save microscopy data through a shared session.

    Args:
        session: Shared UiSession whose owned model may be opened/replaced.
        parent: Optional host widget used to parent native dialogs.
    """

    def __init__(self, session: UiSession, parent: QWidget | None = None) -> None:
        """Initialise the object and validate its configuration.

        Args:
            session: Shared host-owned dataset, selection and background-job coordinator.
            parent: Optional Qt parent controlling widget lifetime.
        """
        super().__init__(session, parent)
        options = QFormLayout()
        self.layout_mode = combo(["hdf5", "directory"])
        self.image_policy = combo([v.value for v in ImagePolicy])
        self.crop_width = QSpinBox()
        self.crop_width.setRange(1, 10001)
        self.crop_width.setSingleStep(2)
        self.crop_width.setValue(7)
        self.export_source = combo(["channel", "original camera"])
        self.dataset_export_space = combo(["corrected", "registered", "raw", "aligned"])
        self.import_into_current = QCheckBox("Add recordings to selected dataset")
        self.export_timeline = combo(["channel", "camera"])
        self.export_tracks = QCheckBox()
        for label, control in (
            ("Dataset layout", self.layout_mode),
            ("Saved images", self.image_policy),
            ("Crop width (odd pixels)", self.crop_width),
            ("TIFF export source", self.export_source),
            ("Dataset image export space", self.dataset_export_space),
            ("Image import grouping", self.import_into_current),
            ("Table export timeline", self.export_timeline),
            ("Export track rows", self.export_tracks),
        ):
            options.addRow(label, control)
        self.form.addLayout(options)
        for title, callback in (
            ("New experiment", self.new_experiment),
            ("Open experiment…", self.open_experiment),
            ("Save experiment…", self.save_experiment),
            ("New dataset", self.new_dataset),
            ("Open dataset…", self.open_dataset),
            ("Save dataset…", self.save_dataset),
            ("Import images…", self.import_images),
            ("Create explicit sequence…", self.create_sequence),
            ("Export TIFF images…", self.export_images),
            ("Export all dataset images…", self.export_dataset_images),
            ("Preview localization / track import…", self.import_results),
            ("Apply table import preview", self.apply_table_preview),
            ("Export localizations / tracks…", self.export_results),
            ("Import label mask…", self.import_mask),
            ("Export label mask…", self.export_mask),
            ("Import affine calibration…", self.import_calibration),
            ("Export affine calibration…", self.export_calibration),
            ("Migrate .moltrack project…", self.import_legacy),
            ("Apply migrated project preview", self.apply_legacy_preview),
        ):
            self.button(title, callback)

    def _replace_action(self, action: Any) -> None:
        """Replace action.

        Args:
            action: Action used by this operation.
        """
        if self.session.dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved dataset",
                "Save the current dataset before replacing it?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Yes:
                self.save_experiment(on_saved=action)
                return
        action()

    def _queue_after_save(self, callback: Any, expected: tuple) -> None:
        if callback is None or not self.session.publication_current:
            return
        from qtpy.QtCore import QTimer

        def _finish() -> None:
            if experiment_state_token(self.session.experiment) != expected:
                self.session.dirty = True
                self.status.setText(
                    "New edits remain unsaved; the current experiment was retained"
                )
                return
            callback()

        QTimer.singleShot(0, _finish)

    @logged
    def new_experiment(self) -> None:
        """Replace the session with an empty sequencing experiment."""
        self._replace_action(
            lambda: self.session.set_experiment(Experiment(), owns_experiment=True)
        )

    @logged
    def open_experiment(self) -> None:
        """Open a full sequencing experiment, including round registrations."""
        if self.layout_mode.currentText() == "directory":
            path = QFileDialog.getExistingDirectory(self, "Open MolCore experiment")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open MolCore experiment", "", "MolCore (*.h5 *.hdf5)"
            )
        if path:

            def start() -> None:
                """Start."""
                self.session.submit(
                    lambda model, progress, cancel: Experiment.open(
                        path, progress=progress, cancel_event=cancel
                    ),
                    completed=lambda result: self.session.set_experiment(
                        result, owns_experiment=True
                    ),
                    experiment_scope=True,
                )

            self._replace_action(start)

    @logged
    def save_experiment(self, *, on_saved: Any = None) -> None:
        """Save every round and its experiment-level alignment records.

        Args:
            on_saved: Optional completion callback after a successful save.
        """
        layout, policy, width = (
            self.layout_mode.currentText(),
            self.image_policy.currentText(),
            self.crop_width.value(),
        )
        if width % 2 != 1:
            raise ValueError("Crop width must be odd")
        if layout == "directory":
            path = QFileDialog.getExistingDirectory(self, "Save experiment directory")
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save MolCore experiment", "", "MolCore (*.h5)"
            )
        if not path:
            return
        destination = (
            Path(path) / "experiment.h5" if layout == "directory" else Path(path)
        )
        if (
            destination.exists()
            and QMessageBox.question(
                self,
                "Replace experiment",
                f"Replace {destination}?",
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        def compute(
            model: Experiment,
            progress: ProgressCallback | None,
            cancel: Any,
            publish_guard: Any,
        ) -> Experiment:
            """Compute.

            Args:
                model: Detached dataset or experiment snapshot.
                progress: Optional structured progress callback.
                cancel: Cooperative cancellation event.

            Returns:
                Result of this operation.
            """
            model.save(
                path,
                layout=layout,
                image_policy=policy,
                crop_width=width,
                progress=progress,
                cancel_event=cancel,
                publish_guard=publish_guard,
            )
            return model

        def complete(saved: Experiment) -> None:
            """Complete.

            Args:
                saved: Saved snapshot with updated lazy source descriptors.
            """
            after_save_token = experiment_state_token(self.session.experiment)
            self.session.dirty = not self.session.publication_current
            self.session.results_changed.emit()
            self.status.setText(f"Saved {destination}")
            self._queue_after_save(on_saved, after_save_token)

        self.session.submit(
            compute, completed=complete, experiment_scope=True, publication=True
        )

    @logged
    def new_dataset(self) -> None:
        """Create a session-owned empty model after resolving unsaved work."""
        dataset = Dataset(name=f"Round {len(self.session.experiment.datasets) + 1}")
        self.session.experiment.add_dataset(dataset)
        self.session.select_dataset(dataset.id)
        self.session.changed()

    @logged
    def open_dataset(self) -> Dataset:
        """Open a snapshot asynchronously, preserving the current model on failure.

        Returns:
            Computed Dataset value.
        """
        if self.layout_mode.currentText() == "directory":
            path = QFileDialog.getExistingDirectory(self, "Open MolCore dataset")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open MolCore dataset", "", "MolCore (*.h5 *.hdf5)"
            )
        if not path:
            return

        def start() -> tuple[tuple, CoordinateSpace]:
            """Start.

            Returns:
                Computed tuple[tuple, CoordinateSpace] value.
            """
            self.session.submit(
                lambda model, progress, cancel: Dataset.open(
                    path, progress=progress, cancel_event=cancel
                ),
                completed=self._add_opened_dataset,
            )

        start()

    def _add_opened_dataset(self, dataset: Dataset) -> None:
        self.session.experiment.add_dataset(dataset)
        self.session.select_dataset(dataset.id)
        self.session.changed()

    @logged
    def save_dataset(self, *, on_saved: Any = None) -> None:
        """Save a detached snapshot using the chosen layout and image policy.

        Args:
            on_saved: Optional GUI callback after successful save and cleanup.
        """
        layout, policy, width = (
            self.layout_mode.currentText(),
            self.image_policy.currentText(),
            self.crop_width.value(),
        )
        if width % 2 != 1:
            raise ValueError("Crop width must be odd")
        if layout == "directory":
            path = QFileDialog.getExistingDirectory(
                self, "Save MolCore dataset directory"
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save MolCore dataset", "", "MolCore (*.h5)"
            )
        if not path:
            return
        destination = Path(path) / "dataset.h5" if layout == "directory" else Path(path)
        if (
            destination.exists()
            and QMessageBox.question(
                self,
                "Replace dataset",
                f"Replace {destination}?",
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        def compute(
            model: Dataset,
            progress: ProgressCallback | None,
            cancel: Any,
            publish_guard: Any,
        ) -> Dataset:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            model.save(
                path,
                layout=layout,
                image_policy=policy,
                crop_width=width,
                progress=progress,
                cancel_event=cancel,
                publish_guard=publish_guard,
            )
            return model

        def complete(saved: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                saved: Saved used by this operation.
            """
            after_save_token = experiment_state_token(self.session.experiment)
            if len(self.session.experiment.datasets) <= 1:
                self.session.dirty = not self.session.publication_current
            self.session.results_changed.emit()
            self.status.setText(f"Saved {destination}")
            self._queue_after_save(on_saved, after_save_token)

        self.session.submit(compute, completed=complete, publication=True)

    @logged
    def import_images(self) -> None:
        """Import independent TIFF/FITS/HDF5 acquisitions with explicit channel layouts."""
        if self.import_into_current.isChecked() and self.session.dataset_id is None:
            raise ValueError("Select a dataset before adding recordings")
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import original images",
            "",
            "Images (*.tif *.tiff *.fits *.fit *.h5 *.hdf5)",
        )
        if not paths:
            return
        dialog = _ImageOptions(paths, self)
        if dialog.exec() != QDialog.Accepted:
            return
        options, source_options = dialog.options(), dialog.source_options()
        into_current = self.import_into_current.isChecked()
        occupied = set(
            self.session.dataset.acquisitions
            if into_current
            else self.session.experiment.datasets
        )
        identifiers = []
        for path in paths:
            base, identifier, index = Path(path).stem, Path(path).stem, 2
            while identifier in occupied:
                identifier = f"{base}_{index}"
                index += 1
            identifiers.append(identifier)
            occupied.add(identifier)

        def compute(
            model: Dataset, progress: ProgressCallback | None, cancel: Any
        ) -> list[Acquisition]:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            return [
                files.image_acquisition(
                    files.read_image(
                        path, **source_options, progress=progress, cancel_event=cancel
                    ),
                    id=identifier,
                    **options,
                )
                for path, identifier in zip(paths, identifiers, strict=True)
            ]

        def commit(model: Dataset | Experiment, values: Any) -> None:
            # All acquisitions were constructed and validated before the first insertion.
            """Commit.

            Args:
                model: Model or fitting method selected by the caller.
                values: Input values to validate, convert or display.
            """
            if not into_current:
                datasets = [
                    Dataset(
                        id=a.id,
                        name=a.id,
                        metadata={"round_index": len(model.datasets) + i + 1},
                    )
                    for i, a in enumerate(values)
                ]
                if any(d.id in model.datasets for d in datasets):
                    raise ValueError("An imported dataset ID now exists")
                for dataset, acquisition in zip(datasets, values, strict=True):
                    dataset.add_acquisition(acquisition)
                for dataset in datasets:
                    model.add_dataset(dataset)
                return
            if any(a.id in model.acquisitions for a in values):
                raise ValueError("An imported acquisition ID now exists")
            for acquisition in values:
                model.add_acquisition(acquisition)

        def complete_import(values: list[Acquisition]) -> None:
            """Complete import.

            Args:
                values: Value supplied for this operation.
            """
            if not into_current:
                self.session.select_dataset(values[0].id)
            self.session.select(values[0].id, next(iter(values[0].channels)))

        self.session.submit(
            compute,
            commit=commit,
            completed=complete_import,
            experiment_scope=not into_current,
        )

    @logged
    def create_sequence(self) -> None:
        """Group explicitly selected acquisitions without merging their camera streams."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Create explicit sequence")
        layout = QVBoxLayout(dialog)
        choices = QListWidget()
        choices.setSelectionMode(QAbstractItemView.ExtendedSelection)
        choices.addItems(list(self.session.dataset.acquisitions))
        layout.addWidget(QLabel("Select recordings in their acquisition order:"))
        layout.addWidget(choices)
        name, reference, crossing = (
            QLineEdit(),
            QLineEdit(),
            QCheckBox("Allow cross-acquisition tracks"),
        )
        form = QFormLayout()
        form.addRow("Sequence name", name)
        form.addRow("Shared reference (required for crossing tracks)", reference)
        layout.addLayout(form)
        layout.addWidget(crossing)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.Accepted:
            identifiers = [item.text() for item in choices.selectedItems()]
            sequence = self.session.dataset.add_sequence(
                identifiers,
                id=name.text().strip() or None,
                allow_cross_boundary_tracks=crossing.isChecked(),
                reference=reference.text().strip() or None,
            )
            self.session.changed()
            self.session.select(
                identifiers[0], self.session.channel, sequence_id=sequence.id
            )

    @logged
    def export_dataset_images(self) -> None:
        """Explicitly write all selected-dataset channel stacks with chosen transforms."""
        from ..exports import export_dataset_images

        dataset_id = self.session.dataset_id
        if dataset_id is None:
            raise ValueError("Select a dataset to export")
        path = QFileDialog.getExistingDirectory(self, "Export dataset channel images")
        if not path:
            return
        space = self.dataset_export_space.currentText()
        if (
            any(Path(path).iterdir())
            and QMessageBox.question(
                self,
                "Export dataset images",
                "Replace matching dataset export files in this directory?",
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self.session.submit(
            lambda model, progress, cancel, publish_guard: export_dataset_images(
                model.datasets[dataset_id],
                path,
                space=space,
                experiment=model,
                progress=progress,
                cancel_event=cancel,
                publish_guard=publish_guard,
            ),
            experiment_scope=True,
            publication=True,
        )

    @logged
    def export_images(self) -> None:
        """Stream the selected camera/channel movie with explicit coordinate provenance."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export image movie", "", "TIFF (*.tif *.tiff)"
        )
        if path:
            a, c, space = (
                self.session.acquisition_id,
                self.session.channel,
                self.session.space,
            )
            original = self.export_source.currentText() == "original camera"
            self.session.submit(
                lambda model, progress, cancel, publish_guard: files.export_tiff(
                    model,
                    a,
                    c,
                    path,
                    space=space,
                    original=original,
                    progress=progress,
                    cancel_event=cancel,
                    publish_guard=publish_guard,
                ),
                publication=True,
            )

    @logged
    def import_results(self) -> None:
        """Read and preview explicit table-column and frame mappings before applying."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import localizations or tracks",
            "",
            "Tables (*.hdf5 *.h5 *.csv *.pos.out)",
        )
        if not path:
            return

        def complete(value: Any) -> None:
            """Present a successful detached computation on the GUI thread.

            Args:
                value: Value supplied by the caller or connected control.
            """
            columns, metadata = value
            dialog = _TableMapping(columns, path, self)
            if dialog.exec() == QDialog.Accepted:
                data, options = dialog.values()
                self.ready(
                    (data, options, metadata),
                    f"{len(data['frame'])} rows staged for {self.session.acquisition_id}/{self.session.channel}\n"
                    f"{options}; raw channel coordinates. Apply to import.",
                )

        def compute(
            model: Dataset, progress: ProgressCallback | None, cancel: Any
        ) -> Any:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.

            Returns:
                Detached result produced by this background computation.
            """
            files.report(progress, cancel)
            value = (
                read_picasso(path)
                if Path(path).suffix.lower() in (".h5", ".hdf5")
                else (files.read_table(path), [])
            )
            files.report(progress, cancel, 1, 1, "Table loaded")
            return value

        self.session.submit(compute, completed=complete)

    @logged
    def apply_table_preview(self) -> None:
        """Apply the staged table mapping through validated model operations."""
        self.require_preview()
        if not isinstance(self.preview, tuple) or len(self.preview) != 3:
            raise ValueError("Preview a localization/track table first")
        data, options, metadata = self.preview
        a, c = self.session.acquisition_id, self.session.channel

        def commit(model: Dataset) -> None:
            """Commit.

            Args:
                model: Model or fitting method selected by the caller.
            """
            files.import_table(model, a, c, data, **options)
            model.processing.settings.setdefault("table_import", {}).setdefault(a, {})[
                c
            ] = {
                "mapping": options,
                "metadata": metadata,
            }

        transact(self.session.dataset, commit, expected=self.preview_token)
        self.session.changed()
        self.status.setText("Table imported")

    @logged
    def export_results(self) -> None:
        """Export selected raw-coordinate rows with an explicit camera/channel timeline."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export localizations / tracks",
            "",
            "Picasso (*.hdf5);;CSV (*.csv);;POS.OUT (*.pos.out)",
        )
        if not path:
            return
        a, c = self.session.acquisition_id, self.session.channel
        tracks, timeline = (
            self.export_tracks.isChecked(),
            self.export_timeline.currentText(),
        )

        def compute(
            model: Dataset, progress: ProgressCallback | None, cancel: Any
        ) -> None:
            """Compute a detached result using the current controls.

            Args:
                model: Model or fitting method selected by the caller.
                progress: Optional callback receiving structured progress updates.
                cancel: Event requesting cooperative cancellation of this background job.
            """
            files.report(progress, cancel)
            rows = (
                model.tracks.query(space="raw")
                if tracks
                else model.localizations.query(
                    acquisition_id=a, channel=c, space="raw"
                ).to_columns()
            )
            if tracks:
                selected = (rows["acquisition_id"] == a) & (rows["channel"] == c)
                rows = {k: v[selected] for k, v in rows.items()}
            view = model.acquisitions[a].channels[c]
            if timeline == "channel":
                rows["frame"] = np.array(
                    [view.channel_index(int(f)) for f in rows["frame"]], dtype=np.int64
                )
            metadata = [
                {
                    "Frames": (
                        len(view)
                        if timeline == "channel"
                        else model.acquisitions[a].source.shape[0]
                    ),
                    "Height": view.shape[1],
                    "Width": view.shape[2],
                    "Pixelsize": model.acquisitions[a].pixel_size,
                    "MolCore timeline": timeline,
                    "MolCore camera frames": view.source_frames.tolist(),
                    "MolCore acquisition": a,
                    "MolCore channel": c,
                }
            ]
            files.report(progress, cancel, 0.8, 1, "Writing result table")
            files.write_table(path, rows, metadata=metadata)

        self.session.submit(compute)

    @logged
    def import_mask(self) -> None:
        """Import a static label TIFF bound to the current frame and coordinate space."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import label mask", "", "TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        a, c, frame, space = (
            self.session.acquisition_id,
            self.session.channel,
            self.session.camera_frame,
            self.session.space,
        )
        identifier, ok = QInputDialog.getText(
            self, "Mask identifier", "Segmentation name", text=Path(path).stem
        )
        if ok and identifier:

            def compute(
                model: Dataset, progress: ProgressCallback | None, cancel: Any
            ) -> Any:
                """Compute a detached result using the current controls.

                Args:
                    model: Model or fitting method selected by the caller.
                    progress: Optional callback receiving structured progress updates.
                    cancel: Event requesting cooperative cancellation of this background job.

                Returns:
                    Detached result produced by this background computation.
                """
                files.report(progress, cancel)
                source = model.transforms.image(a, c, frame, space=space)
                value = files.read_mask(
                    path,
                    segmentation_id=identifier,
                    acquisition_id=a,
                    channel=c,
                    frame=frame,
                    space=space,
                    source_image=source,
                )
                files.report(progress, cancel, 1, 1, "Mask read")
                return value

            self.session.submit(
                compute,
                commit=lambda model, value: apply_segmentations(
                    model, {value.id: value}
                ),
            )

    @logged
    def export_mask(self) -> None:
        """Export a selected static segmentation as an integer TIFF mask."""
        names = list(self.session.dataset.segmentations)
        if not names:
            raise ValueError("No segmentation masks are available")
        name, ok = QInputDialog.getItem(
            self, "Export mask", "Segmentation", names, 0, False
        )
        if not ok:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export label mask", "", "TIFF (*.tif)"
        )
        if path:

            def compute(
                model: Dataset, progress: ProgressCallback | None, cancel: Any
            ) -> None:
                """Compute a detached result using the current controls.

                Args:
                    model: Model or fitting method selected by the caller.
                    progress: Optional callback receiving structured progress updates.
                    cancel: Event requesting cooperative cancellation of this background job.
                """
                files.report(progress, cancel)
                with files.atomic_files([path]) as (temp,):
                    tifffile.imwrite(
                        temp, model.segmentations[name].labels, photometric="minisblack"
                    )
                    files.report(progress, cancel, 1, 1, "Publishing mask")

            self.session.submit(compute)

    @logged
    def import_calibration(self) -> None:
        """Import and select an affine calibration for the selected channel."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import affine matrix", "", "Matrix (*.json *.txt *.npy)"
        )
        if not path:
            return
        name, ok = QInputDialog.getText(
            self, "Calibration name", "New calibration name", text=Path(path).stem
        )
        if not ok or not name:
            return
        a, c = self.session.acquisition_id, self.session.channel

        def commit(model: Dataset, matrix: ArrayLike) -> None:
            """Commit.

            Args:
                model: Model or fitting method selected by the caller.
                matrix: Matrix used by this operation.
            """

            def apply(staged: Any) -> None:
                """Apply.

                Args:
                    staged: Staged used by this operation.
                """
                if name in staged.calibrations:
                    raise ValueError("Calibration name already exists")
                staged.transforms.add_calibration(
                    Calibration(name, matrix, {"source_path": path})
                )
                staged.transforms.select(a, c, name)

            transact(model, apply)

        self.session.submit(
            lambda model, progress, cancel: files.read_calibration(path), commit=commit
        )

    @logged
    def export_calibration(self) -> None:
        """Export a named affine matrix without changing active selections."""
        names = list(self.session.dataset.calibrations)
        if not names:
            raise ValueError("No named calibrations are available")
        name, ok = QInputDialog.getItem(
            self, "Export calibration", "Calibration", names, 0, False
        )
        if ok:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export affine matrix", "", "JSON (*.json);;NumPy (*.npy)"
            )
            if path:
                self.session.submit(
                    lambda model, progress, cancel: files.write_calibration(
                        path, model.calibrations[name].matrix
                    )
                )

    @logged
    def import_legacy(self) -> None:
        """Preview a restricted .moltrack migration with explicit original-image relinking."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Migrate legacy project", "", "MolTrack (*.moltrack)"
        )
        if not path:
            return

        def decoded(project: Any) -> None:
            """Decoded.

            Args:
                project: Decoded legacy project records to validate and migrate.
            """
            bindings, mapping = {}, {}
            for name, data in project.get("dataset_dict", {}).items():
                source_path = data.get("path")
                if not isinstance(source_path, str) or not Path(source_path).is_file():
                    source_path, _ = QFileDialog.getOpenFileName(
                        self,
                        f"Relink original camera stream: {name}",
                        "",
                        "Images (*.tif *.tiff *.fits *.h5 *.hdf5)",
                    )
                if not source_path:
                    self._legacy_results_only(project)
                    return
                dialog = _ImageOptions([source_path], self)
                dialog.setWindowTitle(f"Original camera mapping for {name}")
                channels = list(data.get("images", {}))
                if len(channels) == 1:
                    dialog.channel.setText(channels[0])
                if dialog.exec() != QDialog.Accepted:
                    return
                bindings[name] = (
                    source_path,
                    dialog.options(),
                    dialog.source_options(),
                )
                for kind in ("localisation_dict", "tracking_dict"):
                    for channel in project.get(kind, {}).get(name, {}):
                        mapping[name, channel] = channel
            timeline, ok = QInputDialog.getItem(
                self,
                "Legacy frame mapping",
                "Table frame convention",
                ["channel", "camera"],
                0,
                False,
            )
            if not ok:
                return

            def compute(
                model: Dataset, progress: ProgressCallback | None, cancel: Any
            ) -> tuple[Dataset, list[str]]:
                """Compute a detached result using the current controls.

                Args:
                    model: Model or fitting method selected by the caller.
                    progress: Optional callback receiving structured progress updates.
                    cancel: Event requesting cooperative cancellation of this background job.

                Returns:
                    Detached result produced by this background computation.
                """
                acquisitions = {
                    name: files.image_acquisition(
                        files.read_image(
                            path,
                            **source_options,
                            progress=progress,
                            cancel_event=cancel,
                        ),
                        id=name,
                        **options,
                    )
                    for name, (path, options, source_options) in bindings.items()
                }
                return migrate_moltrack(
                    project,
                    acquisitions=acquisitions,
                    channel_mapping=mapping,
                    timeline=timeline,
                    progress=progress,
                    cancel_event=cancel,
                )

            from qtpy.QtCore import QTimer

            QTimer.singleShot(
                0,
                lambda: self.invoke(
                    lambda: self.session.submit(
                        compute,
                        completed=lambda value: self.ready(
                            value,
                            "\n".join(value[1])
                            or "Migration staged; apply to replace the session",
                        ),
                    )
                ),
            )

        self.session.submit(
            lambda model, progress, cancel: read_moltrack(path), completed=decoded
        )

    def _legacy_results_only(self, project: Any) -> None:
        """Legacy results only.

        Args:
            project: Decoded legacy project records to validate and migrate.
        """
        targets = [
            (dataset, channel, key, value[key])
            for kind, key in (
                ("localisation_dict", "localisations"),
                ("tracking_dict", "tracks"),
            )
            for dataset, channels in project.get(kind, {}).items()
            for channel, value in channels.items()
            if key in value
        ]
        if not targets or self.session.acquisition_id is None:
            raise ValueError(
                "Original-camera binding is required; no selected acquisition is available for results-only import"
            )
        labels = [f"{a}/{c}: {key}" for a, c, key, _ in targets]
        selected, ok = QInputDialog.getItem(
            self,
            "Results-only legacy import",
            "Bind one table to current acquisition/channel",
            labels,
            0,
            False,
        )
        if ok:
            from ..interop import from_records

            _, _, key, data = targets[labels.index(selected)]
            columns = (
                from_records(data)
                if isinstance(data, np.ndarray)
                else {k: np.asarray(v) for k, v in data.items()}
            )
            dialog = _TableMapping(columns, "legacy.csv", self)
            dialog.tracks.setChecked(key == "tracks")
            if dialog.exec() == QDialog.Accepted:
                data, options = dialog.values()
                self.ready(
                    (data, options, []),
                    "Legacy table staged; apply table import preview to bind these results",
                )

    @logged
    def apply_legacy_preview(self) -> None:
        """Replace the current session with a successfully staged legacy migration."""
        self.require_preview()
        if not isinstance(self.preview, tuple) or not isinstance(
            self.preview[0], Dataset
        ):
            raise ValueError("Preview a complete legacy project migration first")
        model = self.preview[0]
        self._replace_action(lambda: self.session.set_dataset(model, owns_dataset=True))
