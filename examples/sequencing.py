"""Two probing rounds with optical misregistration and a moving surface."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from molcore import ArraySource, Dataset, ExecutionContext, Experiment
from molcore.alignment import align_channels
from molcore.exports import export_dataset_images
from molcore.files import image_acquisition


def main() -> None:
    points = np.array([[5, 5], [16, 6], [6, 17], [18, 18]])
    experiment = Experiment(metadata={"surface": "synthetic-slide"})
    for index, shift in enumerate((0, 2), start=1):
        dataset = Dataset(id=f"round-{index}", metadata={"round_index": index, "oligo": f"probe-{index}"})
        pixels = np.zeros((4, 32, 64), dtype=np.uint16)
        for x, y in points:
            pixels[:, y, x + shift] = 100
            pixels[:, y, 32 + x + shift + 1] = 100
        dataset.add_acquisition(image_acquisition(ArraySource(pixels), id="movie", mode="alex"))
        for channel, extra in (("DD", 0), ("DA", 1)):
            dataset.localizations.add(
                acquisition_id="movie",
                channel=channel,
                frame=0,
                x=points[:, 0] + shift + extra,
                y=points[:, 1],
            )
        experiment.add_dataset(dataset)
    reference = experiment.datasets["round-1"]
    optical = align_channels(
        reference.snapshot(),
        "movie",
        "DA",
        "DD",
        source_ids=[5, 6, 7, 8],
        target_ids=[1, 2, 3, 4],
        model="homography",
        context=ExecutionContext(num_cpus=1),
    )
    receipt = reference.apply(optical)
    experiment.apply_channel_calibration(reference.calibrations[receipt.identifiers["calibration_name"]])
    result = experiment.alignment.estimate(
        "round-2",
        "round-1",
        source_acquisition_id="movie",
        source_channel="DD",
        reference_acquisition_id="movie",
        reference_channel="DD",
        source_ids=[1, 2, 3, 4],
        reference_ids=[1, 2, 3, 4],
        context=ExecutionContext(num_cpus=1),
    )
    experiment.alignment.apply([result])
    with TemporaryDirectory() as directory:
        snapshot = Path(directory) / "sequencing.h5"
        experiment.save(snapshot, image_policy="full")
        with Experiment.open(snapshot) as restored:
            paths = export_dataset_images(
                restored.datasets["round-2"], Path(directory) / "export", space="aligned", experiment=restored
            )
            assert len(paths) == 4
            np.testing.assert_allclose(
                restored.alignment.image("round-2", "movie", "AA", 1)[5, 5], 100, atol=1e-08
            )
            print(f"Aligned {len(restored.datasets)} rounds; exported {len(paths)} channel stacks")
    experiment.close()


if __name__ == "__main__":
    main()
