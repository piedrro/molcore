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
    Experiment,
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
    experiment = Experiment()
    experiment.add_acquisition(
        Acquisition(ArraySource(movie), id="movie", pixel_size=100.0, timestamps=np.arange(5) * 0.1)
    )
    ids = experiment.localizations.add(
        acquisition_id="movie", channel="image", frame=np.arange(5), x=7.0, y=7.0
    )
    token = state_token(experiment)
    apply_batch(
        experiment, [lambda model: model.localizations.update_measurements(ids, quality=1.0)], expected=token
    )
    photometry = PhotometrySettings(shape=ApertureShape.SQUARE, background=BackgroundMethod.LOCAL)
    with CpuExecutor(num_cpus=1, max_pending=2) as cpu:
        measurements = measure_localizations(experiment, ids, settings=photometry, cpu=cpu)
        linking = track(
            experiment,
            "movie",
            "image",
            settings=TrackingSettings(search_range=2.0, min_length=3, space=CoordinateSpace.CORRECTED),
            cpu=cpu,
        )
        moving = track_traces(experiment, linking.track_ids, settings=photometry, cpu=cpu)
        stationary = stationary_traces(experiment, ids[:1], settings=photometry, cpu=cpu)
    statistics = trajectory_statistics(
        experiment, int(linking.track_ids[0]), unit=SpatialUnit.NANOMETRES, fit_lag_range=(0.1, 0.4)
    )
    np.testing.assert_allclose(measurements.columns["intensity"], 90.0)
    assert len(moving.columns["frame"]) == len(stationary.columns["frame"]) == 5
    assert statistics.path_length == 0.0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scientific.h5"
        experiment.save(path)
        with Experiment.open(path) as restored:
            assert len(restored.observations) == 15
            assert len(restored.tracks) == 1
    print("Photometry, stationary/moving traces, tracking, statistics and save/open verified.")


if __name__ == "__main__":
    main()
