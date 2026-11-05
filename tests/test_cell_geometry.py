import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import molcore
from molcore import Acquisition, ArraySource, Dataset, Experiment, StaleResultError
from molcore.staging import state_token, transact

OUTLINE = ((0.25, 0.75), (4.0, 0.75), (4.0, 2.0), (0.25, 2.0))


class CellGeometryTests(unittest.TestCase):
    def test_geometry_survives_transactions_and_dataset_storage(self):
        model = Dataset()
        model.add_acquisition(Acquisition(ArraySource(np.zeros((3, 5, 5))), id="a"))
        outline = molcore.SegmentationOutline("cell-1", OUTLINE, acquisition_id="a", channel="image", frame=2)
        global_outline = molcore.SegmentationOutline("global", OUTLINE, pixel_size=105)
        fit = molcore.BactFitCell(
            "fit-1",
            OUTLINE,
            ((1, 1), (3, 1)),
            ((0, 1), (4, 1)),
            (0.1, 1.0),
            0.75,
            fit_error=0.125,
            outline_id="cell-1",
        )
        model.add_segmentation_outline(outline)
        model.add_segmentation_outline(global_outline)
        transact(model, lambda staged: staged.add_bactfit_cell(fit))

        with tempfile.TemporaryDirectory() as tmp:
            for layout in ("hdf5", "directory"):
                path = Path(tmp) / layout
                model.save(path, layout=layout)
                with Dataset.open(path) as reopened:
                    self.assertEqual(reopened.segmentation_outlines["cell-1"], outline)
                    self.assertEqual(reopened.segmentation_outlines["global"], global_outline)
                    self.assertEqual(reopened.bactfit_cells["fit-1"], fit)

    def test_replacing_outline_invalidates_only_its_fits(self):
        model = Dataset()
        model.add_segmentation_outline(molcore.SegmentationOutline("cell-1", OUTLINE))
        fit_values = (OUTLINE, ((1, 1), (3, 1)), ((0, 1), (4, 1)), (0.1, 1.0), 0.75)
        model.add_bactfit_cell(molcore.BactFitCell("linked", *fit_values, outline_id="cell-1"))
        model.add_bactfit_cell(molcore.BactFitCell("standalone", *fit_values))
        with self.assertRaises(ValueError):
            model.add_bactfit_cell(molcore.BactFitCell("bad", *fit_values, outline_id="missing"))
        snapshot = model.snapshot()
        expected = state_token(model)
        model.add_segmentation_outline(molcore.SegmentationOutline("cell-1", OUTLINE[:3]))
        self.assertEqual(set(model.bactfit_cells), {"standalone"})
        self.assertEqual(set(snapshot.bactfit_cells), {"linked", "standalone"})
        with self.assertRaises(StaleResultError):
            transact(model, lambda staged: None, expected=expected)

    def test_acquisition_removal_handles_bound_outlines_and_fits(self):
        model = Dataset()
        model.add_acquisition(Acquisition(ArraySource(np.zeros((3, 5, 5))), id="a"))
        model.add_segmentation_outline(
            molcore.SegmentationOutline("cell-1", OUTLINE, acquisition_id="a", channel="image")
        )
        model.add_bactfit_cell(
            molcore.BactFitCell(
                "linked", OUTLINE, ((1, 1), (3, 1)), ((0, 1), (4, 1)), (0.1, 1.0), 0.75, outline_id="cell-1"
            )
        )
        with self.assertRaises(ValueError):
            model.remove_acquisition("a")
        model.remove_acquisition("a", cascade=True)
        self.assertFalse(model.segmentation_outlines)
        self.assertFalse(model.bactfit_cells)

    def test_version_four_experiment_loads_without_vector_geometry(self):
        experiment = Experiment()
        experiment.add_dataset(Dataset(id="round"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.h5"
            experiment.save(path)
            with h5py.File(path, "r+") as file:
                file.attrs["format_version"] = 4
                group = file["datasets/0"]
                group.attrs["format_version"] = 4
                del group["segmentation_outlines"]
                del group["bactfit_cells"]
            with Experiment.open(path) as restored:
                dataset = restored.datasets["round"]
                self.assertFalse(dataset.segmentation_outlines)
                self.assertFalse(dataset.bactfit_cells)
