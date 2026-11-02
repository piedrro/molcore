"""Run every shared Qt tool on synthetic data; --smoke checks offscreen rendering."""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from molcore import (
    Acquisition,
    ArraySource,
    ChannelSpec,
    Dataset,
    PhotonSequence,
    PhotonUnit,
)


def demo_dataset() -> Dataset:
    """Create moving emitters with stable IDs, a spatial track and a photon sequence.

    Returns:
        Two-channel dataset ready for all eight analysis tools.
    """
    rng = np.random.default_rng(7)
    frames = 12
    yy, xx = np.indices((31, 31))
    movie = rng.normal(20, 1, (frames, 31, 62))
    for frame in range(frames):
        spot = 100 * np.exp(-((xx - (10 + 0.15 * frame)) ** 2 + (yy - 14) ** 2) / 2)
        movie[frame, :, :31] += spot
        movie[frame, :, 31:] += 0.8 * spot
    model = Dataset()
    model.add_acquisition(
        Acquisition(
            ArraySource(movie),
            id="demo",
            pixel_size=100.0,
            timestamps=np.arange(frames) * 0.05,
            channel_specs={
                "donor": ChannelSpec((0, 31, 0, 31)),
                "acceptor": ChannelSpec((0, 31, 31, 62)),
            },
        )
    )
    ids = model.localizations.add(
        acquisition_id="demo",
        channel="donor",
        frame=np.arange(frames),
        x=10 + 0.15 * np.arange(frames),
        y=np.full(frames, 14.0),
        photons=np.linspace(100, 180, frames),
        lpx=np.full(frames, 0.1),
        lpy=np.full(frames, 0.1),
    )
    model.localizations.add(
        acquisition_id="demo",
        channel="acceptor",
        frame=np.arange(frames),
        x=10.1 + 0.15 * np.arange(frames),
        y=np.full(frames, 14.1),
        photons=np.linspace(80, 144, frames),
        lpx=np.full(frames, 0.1),
        lpy=np.full(frames, 0.1),
    )
    track = model.tracks.add(ids)
    model.photon_sequences.add(
        PhotonSequence(
            molecule_id="demo-track",
            acquisition_id="demo",
            channel="donor",
            frames=np.arange(frames),
            timestamps=np.arange(frames) * 0.05,
            photons=np.linspace(100, 180, frames),
            localization_ids=ids,
            track_id=track,
            run_id="demo",
            unit=PhotonUnit.IMAGE_INTENSITY,
            method="synthetic",
            positions=np.column_stack(
                (10 + 0.15 * np.arange(frames), np.full(frames, 14.0))
            ),
            measurements=dict(
                signal_sum=np.linspace(120, 200, frames),
                signal_count=np.ones(frames),
                local_background_mean=np.full(frames, 20.0),
                global_background_mean=np.full(frames, 20.0),
            ),
        )
    )
    return model


def main() -> int:
    """Launch the host-owned application or perform a short rendering check.

    Returns:
        Qt exit code, or zero after a successful smoke check.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from qtpy.QtWidgets import QApplication

    from molcore.ui import MolCoreToolsWidget, UiSession

    app = QApplication.instance() or QApplication(sys.argv[:1])
    session = UiSession(demo_dataset())
    session.select("demo", "donor")
    panel = MolCoreToolsWidget(session)
    panel.resize(980, 1000)
    panel.setWindowTitle("Molcore shared tools")
    panel.show()
    app.processEvents()
    if args.smoke:
        for index in range(panel.tabs.count()):
            panel.tabs.setCurrentIndex(index)
            app.processEvents()
            if panel.grab().isNull():
                raise RuntimeError("Widget failed to render")
        panel.close()
        session.shutdown()
        return 0
    app.setQuitOnLastWindowClosed(False)
    app.lastWindowClosed.connect(session.shutdown)
    session.shutdown_finished.connect(app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
