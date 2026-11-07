from threading import Event

import numpy as np
import pytest

from molcore import Acquisition, ArraySource, ChannelSpec, Dataset, PhotometrySettings
from molcore.photometry import stationary_site_traces


def alex_dataset():
    dataset = Dataset()
    dataset.add_acquisition(
        Acquisition(
            ArraySource(np.arange(45).reshape(5, 3, 3)),
            id="movie",
            channel_specs={"DD": ChannelSpec(excitation="D")},
            excitation_sequence=("D", "A"),
        )
    )
    return dataset


def test_sites_use_working_pixels_and_original_camera_frames():
    dataset = alex_dataset()
    working = ArraySource(np.arange(27).reshape(3, 3, 3) + 100)
    result = stationary_site_traces(
        dataset,
        "movie",
        "DD",
        {"site": (1, 1)},
        source=working,
        settings=PhotometrySettings(spot_size=1, buffer=0),
    )

    sequence = dataset.photon_sequences[result.sequence_ids[0]]
    assert sequence.molecule_id == "site"
    np.testing.assert_array_equal(sequence.frames, [0, 2, 4])
    np.testing.assert_array_equal(sequence.measurements["signal_sum"], [104, 113, 122])
    assert sequence.positions is None
    assert len(dataset.localizations.query()["id"]) == 0


def test_sites_reject_a_mismatched_working_source():
    with pytest.raises(ValueError, match="shape"):
        stationary_site_traces(
            alex_dataset(),
            "movie",
            "DD",
            {"site": (1, 1)},
            source=ArraySource(np.zeros((2, 3, 3))),
        )


def test_cancelled_sites_do_not_publish():
    from concurrent.futures import CancelledError

    dataset = alex_dataset()
    stop = Event()
    stop.set()
    with pytest.raises(CancelledError):
        stationary_site_traces(dataset, "movie", "DD", {"site": (1, 1)}, cancel_event=stop)
    assert not dataset.photon_sequences.query()


def test_fixed_apertures_preserve_window_and_ring_at_the_image_edge():
    from molcore.photometry import FixedApertureSettings

    dataset = Dataset()
    dataset.add_acquisition(Acquisition(ArraySource(np.arange(1, 26).reshape(1, 5, 5)), id="movie"))
    result = stationary_site_traces(
        dataset,
        "movie",
        "image",
        {"edge": (0, 0)},
        settings=FixedApertureSettings(spot_size=1, buffer=0, global_background=True),
        apply=False,
    )
    metrics = result.sequences[0].measurements
    np.testing.assert_array_equal(metrics["signal_sum"], [1])
    np.testing.assert_array_equal(metrics["local_window_mean"], [4])
    np.testing.assert_array_equal(metrics["masked_local_ring_mean"], [5])
    np.testing.assert_array_equal(metrics["global_background_mean"], [13])
    np.testing.assert_array_equal(metrics["masked_global_background_mean"], [13.5])


def test_fixed_sites_keep_psf_amplitude_separate_from_aperture_intensity():
    from molcore.photometry import FixedApertureSettings

    yy, xx = np.mgrid[-5:6, -5:6]
    pixels = 10 + 100 * np.exp(-(xx**2 + yy**2) / 2)
    dataset = Dataset()
    dataset.add_acquisition(Acquisition(ArraySource(pixels), id="movie"))
    result = stationary_site_traces(
        dataset,
        "movie",
        "image",
        {"center": (5, 5), "edge": (0, 0)},
        settings=FixedApertureSettings(psf_box_size=7),
        apply=False,
    )
    center, edge = result.sequences
    assert center.measurements["psf_amplitude"][0] == pytest.approx(628, rel=0.15)
    assert center.photons[0] == center.measurements["signal_sum"][0]
    assert np.isnan(edge.measurements["psf_amplitude"][0])
