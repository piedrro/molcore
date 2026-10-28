"""Scientific commits are shared by CLI and GUI and atomic across batches."""

import unittest

import numpy as np

from molcore import Acquisition, ArraySource, Dataset, Segmentation
from molcore.processing import apply_segmentations


class TransactionTests(unittest.TestCase):
    def test_invalid_batch_does_not_apply_first_mask(self):
        model = Dataset()
        model.add_acquisition(Acquisition(ArraySource(np.zeros((2, 5, 5))), id="a"))
        first = Segmentation("one", "a", np.ones((5, 5), dtype=int))
        second = Segmentation("two", "a", np.ones((7, 7), dtype=int))
        with self.assertRaises(ValueError):
            apply_segmentations(model, {"one": first, "two": second})
        self.assertFalse(model.segmentations)


if __name__ == "__main__":
    unittest.main()
