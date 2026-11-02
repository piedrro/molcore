"""Small GUI-free workflows representing MolTrack and MolSeeQ adoption."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from molcore import (
    Acquisition,
    AcquisitionMode,
    ArraySource,
    ChannelName,
    ChannelSpec,
    Dataset,
)
from molcore.interop import napari_points, napari_tracks


def moltrack_example() -> Dataset:
    """Build the shared model from single-channel images and localization results.

    Returns:
        Dataset with measured-frame track membership and no GUI dependencies.
    """
    dataset = Dataset()
    dataset.add_acquisition(
        Acquisition(
            ArraySource(np.zeros((3, 16, 16), dtype=np.uint16)),
            id="tracking",
            mode=AcquisitionMode.SINGLE,
            pixel_size=100,
        )
    )
    ids = dataset.localizations.add(
        acquisition_id="tracking",
        channel=ChannelName.IMAGE,
        frame=[0, 1, 2],
        x=[4.0, 5.0, 6.0],
        y=[7.0, 7.0, 8.0],
        photons=[100.0, 95.0, 110.0],
    )
    dataset.tracks.add(ids, metadata={"algorithm": "external tracker"})
    assert napari_points(dataset.localizations.query()).shape == (3, 3)
    assert napari_tracks(dataset.tracks.query()).shape == (3, 4)
    return dataset


def molseeq_example() -> Dataset:
    """Build four ALEX channel views while retaining the original camera stream.

    Returns:
        Dataset with unequal donor/acceptor channel lengths and intensities.
    """
    dataset = Dataset()
    dataset.add_acquisition(
        Acquisition(
            ArraySource(np.zeros((5, 16, 32), dtype=np.uint16)),
            id="alex",
            mode=AcquisitionMode.ALEX,
            excitation_sequence=("D", "A"),
            channel_specs={
                ChannelName.DD: ChannelSpec((0, 16, 0, 16), excitation="D"),
                ChannelName.DA: ChannelSpec((0, 16, 16, 32), excitation="D"),
                ChannelName.AD: ChannelSpec((0, 16, 0, 16), excitation="A"),
                ChannelName.AA: ChannelSpec((0, 16, 16, 32), excitation="A"),
            },
        )
    )
    dataset.observations.add(
        molecule_id="stationary-1",
        acquisition_id="alex",
        channel=[ChannelName.DD, ChannelName.DA, ChannelName.AA],
        frame=[0, 0, 1],
        intensity=[100.0, 50.0, 80.0],
    )
    assert dataset.acquisitions["alex"].channels[ChannelName.DD].shape[0] == 3
    return dataset


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        for name, dataset in (
            ("tracking", moltrack_example()),
            ("alex", molseeq_example()),
        ):
            path = Path(directory) / f"{name}.h5"
            dataset.save(path, image_policy="full")
            with Dataset.open(path) as restored:
                print(name, list(restored.acquisitions))
