"""Headless scientific editing, traces, tracking and trajectory statistics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from molcore import (
    Acquisition,
    ApertureShape,
    ArraySource,
    BackgroundMethod,
    CoordinateSpace,
    Dataset,
    PhotometrySettings,
    SpatialUnit,
    TrackingSettings,
)
from molcore.cpu import CpuExecutor
from molcore.photometry import measure_localizations, stationary_traces, track_traces
from molcore.processing import apply_batch
from molcore.staging import state_token
from molcore.tracking import track
from molcore.trajectory import trajectory_statistics


def main() -> None:
    movie = np.full((5, 15, 15), 10.0, dtype=float)
    movie[:, 6:9, 6:9] = 20.0
    dataset = Dataset()
    dataset.add_acquisition(
        Acquisition(ArraySource(movie), id="movie", pixel_size=100.0, timestamps=np.arange(5) * 0.1)
    )
    ids = dataset.localizations.add(acquisition_id="movie", channel="image", frame=np.arange(5), x=7.0, y=7.0)
    token = state_token(dataset)
    apply_batch(
        dataset, [lambda model: model.localizations.update_measurements(ids, quality=1.0)], expected=token
    )
    photometry = PhotometrySettings(shape=ApertureShape.SQUARE, background=BackgroundMethod.LOCAL)
    with CpuExecutor(num_cpus=1, max_pending=2) as cpu:
        measurements = dataset.apply(
            measure_localizations(dataset.snapshot(), ids, settings=photometry, cpu=cpu, apply=False)
        )
        preview = track(
            dataset.snapshot(),
            "movie",
            "image",
            settings=TrackingSettings(search_range=2.0, min_length=3, space=CoordinateSpace.CORRECTED),
            cpu=cpu,
            apply=False,
        )
        linking = dataset.apply(preview)
        moving = track_traces(dataset, linking.track_ids, settings=photometry, cpu=cpu)
        stationary = stationary_traces(dataset, ids[:1], settings=photometry, cpu=cpu)
    statistics = trajectory_statistics(
        dataset, int(linking.track_ids[0]), unit=SpatialUnit.NANOMETRES, fit_lag_range=(0.1, 0.4)
    )
    np.testing.assert_allclose(measurements.columns["intensity"], 90.0)
    assert len(moving.columns["frame"]) == len(stationary.columns["frame"]) == 5
    assert statistics.path_length == 0.0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scientific.h5"
        dataset.save(path)
        with Dataset.open(path) as restored:
            assert len(restored.observations) == 5
            assert len(restored.photon_sequences) == 2
            assert sum(len(sequence.frames) for sequence in restored.photon_sequences.query()) == 10
            assert len(restored.tracks) == 1
    print("Photometry, stationary/moving traces, tracking, statistics and save/open verified.")


if __name__ == "__main__":
    main()
