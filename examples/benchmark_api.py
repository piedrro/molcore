"""Compare API latency and peak allocations without host-specific test thresholds.

This benchmark also runs against the previous API so the same workloads and
Python/NumPy environment can be used for a fair comparison.
"""

import argparse
import inspect
import json
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import h5py
import numpy as np
import tifffile

from molcore import Acquisition, ArraySource, Dataset
from molcore.staging import transact
from molcore.types.images import HDF5Source, TiffSource


def measure(function, repeats=5):
    timings, allocations = [], []
    result = None
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - started)
        allocations.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    return {"median_seconds": statistics.median(timings), "peak_bytes": max(allocations)}, result


def main(n=1_000_000):
    model = Dataset(id="benchmark")
    model.add_acquisition(Acquisition(ArraySource(np.zeros((100, 16, 16))), id="a"))
    model.localizations.add(
        acquisition_id="a", channel="image", frame=np.arange(n) % 100, x=np.ones(n), y=np.ones(n)
    )
    for start in range(0, 1000, 5):
        model.tracks.add(np.arange(start + 1, start + 6, dtype=np.uint64))
    # Warm imports and indexes; report steady-state operation costs.
    model.localizations.query(ids=[1])
    model.localizations.query(frame_range=(0, 1))
    model.snapshot()
    output = {"rows": n, "numpy": np.__version__}
    output["id_query"], _ = measure(lambda: model.localizations.query(ids=[1]))
    output["frame_query"], _ = measure(lambda: model.localizations.query(frame_range=(0, 1)))
    output["snapshot"], _ = measure(model.snapshot)
    output["small_commit"], _ = measure(
        lambda: transact(model, lambda staged: staged.localizations.move(1, x=2, y=3))
    )
    output["many_short_tracks"], tracks = measure(lambda: model.tracks.query(space="raw"))
    assert len(tracks["id"]) == 1000

    from molcore.photometry import measure_localizations

    pixels = np.full((20, 32, 32), 10.0)
    pixels[:, 6:9, 6:9] = 20.0
    small = Dataset()
    small.add_acquisition(Acquisition(ArraySource(pixels), id="a"))
    ids = small.localizations.add(acquisition_id="a", channel="image", frame=np.arange(20), x=7.0, y=7.0)
    if "apply" in inspect.signature(measure_localizations).parameters:

        def photometry():
            return measure_localizations(small, ids, apply=False, num_cpus=1)
    else:
        from molcore import ExecutionContext

        def photometry():
            return measure_localizations(small.snapshot(), ids, context=ExecutionContext(num_cpus=1))

    photometry()
    output["repeated_photometry"], measured = measure(photometry, repeats=3)
    output["photometry_intensity_sum"] = float(measured.columns["intensity"].sum())
    assert output["photometry_intensity_sum"] == 1800.0

    with tempfile.TemporaryDirectory() as directory:
        values = np.arange(64 * 64 * 64, dtype=np.uint16).reshape(64, 64, 64)
        path = Path(directory) / "images.h5"
        with h5py.File(path, "w") as file:
            file.create_dataset("images", data=values, chunks=(1, 64, 64), compression="gzip")
        hdf = HDF5Source(path, "images")
        output["compressed_hdf5_read"], result = measure(lambda: hdf[:24])
        np.testing.assert_array_equal(result, values[:24])
        tiff_path = Path(directory) / "images.tif"
        tifffile.imwrite(tiff_path, values, photometric="minisblack", compression="deflate")
        tiff = TiffSource(tiff_path)
        output["compressed_tiff_read"], result = measure(lambda: tiff[:24])
        np.testing.assert_array_equal(result, values[:24])
        hdf.close()
        tiff.close()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    main(parser.parse_args().rows)
