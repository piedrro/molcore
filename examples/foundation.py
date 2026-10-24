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
    Experiment,
)
from molcore.interop import napari_points, napari_tracks


def moltrack_example() -> Experiment:
    """Build the shared model from single-channel images and localization results.

    Returns:
        Experiment with measured-frame track membership and no GUI dependencies.
    """
    experiment = Experiment()
    experiment.add_acquisition(
        Acquisition(
            ArraySource(np.zeros((3, 16, 16), dtype=np.uint16)),
            id="tracking",
            mode=AcquisitionMode.SINGLE,
            pixel_size=100,
        )
    )
    ids = experiment.localizations.add(
        acquisition_id="tracking",
        channel=ChannelName.IMAGE,
        frame=[0, 1, 2],
        x=[4.0, 5.0, 6.0],
        y=[7.0, 7.0, 8.0],
        photons=[100.0, 95.0, 110.0],
    )
    experiment.tracks.add(ids, metadata={"algorithm": "external tracker"})
    assert napari_points(experiment.localizations.query()).shape == (3, 3)
    assert napari_tracks(experiment.tracks.query()).shape == (3, 4)
    return experiment


def molseeq_example() -> Experiment:
    """Build four ALEX channel views while retaining the original camera stream.

    Returns:
        Experiment with unequal donor/acceptor channel lengths and intensities.
    """
    experiment = Experiment()
    experiment.add_acquisition(
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
    experiment.observations.add(
        molecule_id="stationary-1",
        acquisition_id="alex",
        channel=[ChannelName.DD, ChannelName.DA, ChannelName.AA],
        frame=[0, 0, 1],
        intensity=[100.0, 50.0, 80.0],
    )
    assert experiment.acquisitions["alex"].channels[ChannelName.DD].shape[0] == 3
    return experiment


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        for name, experiment in (("tracking", moltrack_example()), ("alex", molseeq_example())):
            path = Path(directory) / f"{name}.h5"
            experiment.save(path, image_policy="full")
            with Experiment.open(path) as restored:
                print(name, list(restored.acquisitions))
