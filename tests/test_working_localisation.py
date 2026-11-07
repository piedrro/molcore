import numpy as np

from molcore import Acquisition, ArraySource, ChannelSpec, Dataset, LocalisationSettings
from molcore.localisation import detect


def test_detect_uses_working_pixels_and_camera_frames():
    dataset = Dataset()
    dataset.add_acquisition(
        Acquisition(
            ArraySource(np.zeros((4, 13, 13), dtype=np.uint16)),
            id="movie",
            channel_specs={"DD": ChannelSpec(excitation="D")},
            excitation_sequence=("D", "A"),
        )
    )
    working = np.zeros((2, 13, 13), dtype=np.uint16)
    working[1, 6, 7] = 200
    result = detect(
        dataset,
        "movie",
        "DD",
        source=ArraySource(working),
        settings=LocalisationSettings(box_size=5, min_net_gradient=10),
    )
    np.testing.assert_array_equal(result.columns["frame"], [2])
    np.testing.assert_array_equal(result.columns["x"], [7])
    np.testing.assert_array_equal(result.columns["y"], [6])
