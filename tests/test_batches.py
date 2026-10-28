import inspect
import unittest
from collections.abc import Mapping

import numpy as np

from molcore import Acquisition, ArraySource, Dataset


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(id="round")
        for name in ("a", "b"):
            self.dataset.add_acquisition(Acquisition(ArraySource(np.zeros((3, 4, 4))), id=name))
        self.ids = self.dataset.localizations.add(
            acquisition_id=["a", "b"],
            channel="image",
            frame=[1, 0],
            x=[1, 2],
            y=[2, 3],
            photons=np.array([10, 20], dtype=np.float32),
        )

    def test_query_retains_context_and_detaches_mutable_conversions(self):
        batch = self.dataset.localizations.query(space="raw", original=True)
        self.assertIsInstance(batch, Mapping)
        self.assertEqual(batch.dataset_id, "round")
        self.assertEqual(batch.space, "raw")
        self.assertTrue(batch.original)
        self.assertEqual(batch.frame_context, "camera")
        self.assertEqual(batch.n_rows, 2)
        self.assertEqual(batch.schema["x"].unit, "pixel")
        self.assertEqual(batch.schema["photons"].dtype, np.dtype("float32"))
        self.assertIsNone(batch.schema["photons"].unit)
        with self.assertRaises(ValueError):
            batch["x"][0] = 9
        with self.assertRaises(ValueError):
            batch["x"].setflags(write=True)
        with self.assertRaises(TypeError):
            batch["new"] = np.zeros(2)
        mutable = batch.to_columns()
        mutable["x"][0] = 8
        records = batch.to_records()
        records["x"][0] = 7
        self.dataset.localizations.move(self.ids[0], x=6, y=5)
        self.assertEqual(batch["x"].tolist(), [1, 2])
        self.assertEqual(records["id"].dtype, np.dtype("uint64"))

    def test_sequence_and_empty_queries_keep_reference_context(self):
        self.dataset.add_sequence(["a", "b"], id="s", reference="shared")
        batch = self.dataset.localizations.query(sequence_id="s", space="raw")
        self.assertEqual(batch.sequence_id, "s")
        self.assertEqual(batch.reference, "shared")
        self.assertEqual(batch.frame_context, "sequence")
        self.assertEqual(batch["frame"].tolist(), [1, 0])
        self.assertEqual(batch["sequence_frame"].tolist(), [1, 3])
        empty = self.dataset.localizations.query(sequence_id="s", ids=[], space="raw")
        self.assertEqual(empty.n_rows, 0)
        self.assertEqual(empty.reference, "shared")
        self.assertEqual(empty.to_records().shape, (0,))

    def test_add_exposes_required_fields(self):
        signature = inspect.signature(self.dataset.localizations.add)
        for name in ("acquisition_id", "channel", "frame", "x", "y"):
            self.assertIn(name, signature.parameters)
            self.assertIs(signature.parameters[name].default, inspect.Parameter.empty)


if __name__ == "__main__":
    unittest.main()
