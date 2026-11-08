"""Display limits for arrays and lazy split-channel microscopy movies."""

import numpy as np

from molcore import (
    Acquisition,
    ArraySource,
    ChannelName,
    ChannelSpec,
    ContrastMethod,
    ContrastScope,
    ContrastSettings,
    Dataset,
)
from molcore.contrast import autocontrast, autocontrast_channels


def main() -> None:
    rng = np.random.default_rng(42)
    movie = rng.normal(100.0, 10.0, size=(4, 32, 64))
    movie[:, :, 32:] += 500.0
    original = movie.copy()
    frame = autocontrast(movie[0], settings=ContrastSettings(percentiles=(1.0, 99.0)))
    print("Frame display limits:", frame.limits)
    dataset = Dataset()
    dataset.add_acquisition(
        Acquisition(
            ArraySource(movie),
            id="recording",
            channel_specs={
                ChannelName.DONOR: ChannelSpec(roi=(0, 32, 0, 32)),
                ChannelName.ACCEPTOR: ChannelSpec(roi=(0, 32, 32, 64)),
            },
        )
    )
    per_channel = autocontrast_channels(dataset.snapshot(), "recording")
    shared = autocontrast_channels(
        dataset.snapshot(),
        "recording",
        scope=ContrastScope.SHARED,
        settings=ContrastSettings(method=ContrastMethod.MIN_MAX, max_frames=None),
    )
    for channel, limits in per_channel.items():
        print(channel, "independent:", limits.limits, "shared:", shared[channel].limits)
    assert shared[ChannelName.DONOR].limits == shared[ChannelName.ACCEPTOR].limits
    np.testing.assert_equal(movie, original)
    print("Original pixels preserved.")


if __name__ == "__main__":
    main()
