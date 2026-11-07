from concurrent.futures import CancelledError
from threading import Event

import numpy as np
import pytest

from molcore import ArraySource
from molcore.temporal import rolling_filter


@pytest.mark.parametrize(
    "method, expected",
    [
        ("mean", [5, 3, 13, 15]),
        ("median", [5, 0, 10, 15]),
    ],
)
def test_rolling_filter_truncates_endpoints_and_retains_dtype(method, expected):
    source = ArraySource(np.array([0, 10, 0, 30], dtype=np.uint16).reshape(4, 1, 1))
    result = rolling_filter(source, 1, method=method)
    assert result.dtype == source.dtype
    np.testing.assert_array_equal(result[:].ravel(), expected)
    np.testing.assert_array_equal(source[:].ravel(), [0, 10, 0, 30])


def test_rolling_filter_can_be_cancelled_between_frames():
    source = ArraySource(np.ones((3, 2, 2)))
    stop = Event()

    def progress(update):
        if update.completed == 1:
            stop.set()

    with pytest.raises(CancelledError):
        rolling_filter(source, 1, cancel_event=stop, progress=progress)
