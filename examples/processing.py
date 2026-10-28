"""Run model-based processing with configurable CPUs and parent-side progress."""

import numpy as np

from molcore import Acquisition, ArraySource, Dataset
from molcore.cpu import CpuExecutor
from molcore.localisation import localise
from molcore.types.processing import CameraSettings, LocalisationSettings


def progress(update):
    """Print progress emitted by the calling process.

    Args:
        update: Typed ProgressUpdate with percent and description.
    """
    print(f"{update.percent:5.1f}% {update.description}")


if __name__ == "__main__":
    y, x = np.indices((24, 24))
    image = (20 + 500 * np.exp(-((x - 11.3) ** 2 + (y - 10.7) ** 2) / (2 * 1.2**2))).astype(np.float32)
    dataset = Dataset()
    dataset.add_acquisition(Acquisition(ArraySource(np.repeat(image[None], 4, axis=0)), id="movie"))
    with CpuExecutor(num_cpus=2) as cpu:
        result = localise(
            dataset,
            "movie",
            "image",
            camera=CameraSettings(0, 1, 1),
            settings=LocalisationSettings(min_net_gradient=10, chunk_size=2),
            cpu=cpu,
            progress=progress,
        )
    print(f"Stored {len(result.localization_ids)} localizations")
