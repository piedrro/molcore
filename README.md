# MolCore

Shared, GUI-independent data foundations for napari-moltrack and napari-molseeq.
MolCore stores images, localizations, tracks, photon sequences, observations, segmentations,
calibrations and processing provenance. Processing modules provide
Picasso localisation and drift estimation, affine channel alignment, and Cellpose 3
segmentation. Shared Trackpy linking, aperture photometry, multichannel traces and trajectory statistics
are available as headless processors; presentation remains in the consuming applications.

## Installation

Python 3.12 or newer:

```bash
pip install .
# Or, for development:
uv sync --all-extras
```

The standard installation includes image reading, lazy arrays, image resampling,
and core data models. Install `molcore[localisation]`, `molcore[tracking]`,
`molcore[segmentation]`, or `molcore[clustering]` for selected scientific backends;
`molcore[all]` includes all scientific backends and optional UI libraries.
Ruff is kept in the development dependency group. MolCore's APIs remain independent
of napari and GUI code.

## Basic workflow

For multiple oligo probing rounds on one surface, use
[`Experiment` and `Dataset`](docs/datasets.md). Optical alignment shares one
calibration across FRET Acceptor or ALEX DA/AA channels; experiment-level
alignment corrects surface movement between rounds. Both use lazy views and
retain original images until an explicit transformed-image export.
Run `python examples/sequencing.py` for a two-round ALEX example.

Independent frame-resolved photon/intensity sequences and all combinations of
localisation, track and photon-sequence crop storage are described in
[photon sequences](docs/photon-sequences.md).

For embeddable Qt analysis tools, see [shared widgets](docs/shared-widgets.md) and
run `python examples/shared_widgets.py`. Optional UI dependencies are installed
with `pip install "molcore[ui]"`; the host supplies its Qt binding. See
[logging and type contracts](docs/logging.md) for diagnostics and library conventions.

```python
import numpy as np
from molcore import Acquisition, ChannelName, Dataset, TiffSource
from molcore.interop import napari_points

dataset = Dataset()
dataset.add_acquisition(Acquisition(
    TiffSource("movie.tif"), id="recording", pixel_size=100,
))
ids = dataset.localizations.add(
    acquisition_id="recording", channel=ChannelName.IMAGE,
    frame=[0, 1], x=[12., 13.], y=[20., 20.], photons=[100., 120.],
)
track_id = dataset.tracks.add(ids, metadata={"algorithm": "external tracker"})
subset = dataset.localizations.query(frame_range=(0, 2))
points = napari_points(subset)  # (time, y, x); no napari import
crop = dataset.crops.localization(ids[0], width=7)
dataset.save("dataset.h5", image_policy="full")

with Dataset.open("dataset.h5") as restored:
    image = restored.acquisitions["recording"].channels["image"][0]
```

See [examples/foundation.py](examples/foundation.py) for runnable single-channel
tracking and four-channel ALEX examples using synthetic data.

## Organisation and conventions

All public model classes, shared records and image sources live in
`src/molcore/types/`, including `types/images.py`. They are also exported from
`molcore`. Crop extraction, transform operations, snapshot storage and
interoperability live in separate implementation modules. Public APIs use
Google-style docstrings.

- Arrays use `(time, y, x)`; coordinates use `(x, y)` in channel pixels, with
  integer pixel centres. `pixel_size` is in nanometres; timestamps are seconds.
- `ChannelName` provides `IMAGE`, `DONOR`, `ACCEPTOR`, `DD`, `DA`, `AD`, and `AA`.
  Custom string channel names remain supported. For ALEX names, the first letter
  denotes excitation and the second emission.
- `AcquisitionMode` provides `SINGLE`, `FRET`, `ALEX`, and `CUSTOM`. It describes
  intent; explicit `ChannelSpec` mappings determine actual channel contents.
- Channel indices differ from camera frames. `view.frame_info(index)` preserves
  the camera frame, acquisition, recorded timestamp and excitation cycle.
- Localizations and observations use positive, monotonically allocated `uint64`
  IDs. Zero in an optional anchor column means no anchor. IDs are never reused.
- Queries return detached NumPy columns. Extra measurement columns are accepted.
  Localisation queries return immutable `LocalizationBatch` mappings with coordinate,
  frame and schema context; use `.to_columns()` for writable arrays.
  Missing numeric measurements become NaN when new rows omit them.
- Localization and track queries default to `space="corrected"`: manual edits,
  selected channel registration/alignment and drift correction apply automatically
  without modifying stored coordinates. Use `space="raw"` for effective channel
  coordinates or `query(original=True, space="raw")` for imported coordinates.
- Display images through `dataset.transforms.image(...)` or the lazy
  `dataset.transforms.view(...)`; both default to the same corrected space.
  Direct acquisition/channel sources retain raw pixels for scientific processing.
  The shared UI applies corrections automatically without coordinate controls.
- Registration maps raw channel pixels into reference pixels. Corrected views
  then subtract `(dx, dy)` drift in reference pixels at actual camera frames.
  Missing registration or drift acts as identity. Supplied drift is never extrapolated.

## Sequences, edits and masks

Use `dataset.add_sequence(["first", "second"], id="continuous")` to establish
recording order. This creates lazy concatenated channel views and a
`sequence_frame` query column, while keeping original camera frames and timestamps.
It does not invent elapsed time across recording gaps. Sequence channels must
have matching geometry and pixel scales; layouts and calibrations may differ.
Cross-boundary tracks additionally require `allow_cross_boundary_tracks=True`
and a shared `reference` identifier.

Localization edits and track operations validate before committing. Imported
coordinates remain available. Track membership never duplicates coordinates;
each localization belongs to at most one track, with one member per camera frame.
Replacing an acquisition/channel partition preserves independent partitions.
Edits invalidate anchored observations and saved crops; stale observations are
available only through `query(include_stale=True)`.

`Segmentation` holds a static source image, binary/instance labels, paths,
metadata and explicit acquisition/channel/coordinate bindings. Background and
out-of-bounds positions have label zero. Localization queries accept
`segmentation_id` and `label`; track creation accepts a segmentation constraint.

## Images, crops and saving

`ArraySource` wraps RAM arrays; `TiffSource`, `FitsSource`, and `HDF5Source` read
requested frames. TIFFs must be frame-addressable or memory-mappable.
`CachedSource(source, max_bytes=...)` adds a process-local LRU cache. Sources
support `materialize()` and `as_dask()`. File descriptors reopen for
each read, including in spawned processes.

`Dataset.save` accepts `layout="hdf5"` or `layout="directory"`, and:

| `image_policy` | Embedded microscopy pixels |
| --- | --- |
| `none` | None |
| `localization_crops` | Crops at active localization positions |
| `track_crops` | Crops at measured track members |
| `both` | Union of localization and track crops |
| `photon_sequence_crops` | Crops at photon-sequence sample positions |
| `localization_and_photon_crops` | Localisation and photon-sequence crops |
| `track_and_photon_crops` | Track and photon-sequence crops |
| `all_crops` | All three crop categories |
| `full` | Original camera frames, stored once |

Use `crop_width` to choose the saved odd pixel width. Track crops preserve measured
frames and gaps; `crops.stationary(localization_id, frames, width=...)` samples an
explicit list of camera frames at a fixed position. Crops retain origins, validity
masks, logical channels and original camera frames.

An HDF5 snapshot embeds its selected content. A directory contains
`dataset.h5` and may reference external originals by relative path. Moving
the dataset together with those originals preserves access. Missing originals
leave result tables and segmentation usable and raise `ImageUnavailableError`
when pixels are requested. Crops cannot reconstruct full movies or unsaved crop
widths. Segmentation source images are saved independently of movie policy.

Saves are single-writer snapshots replaced atomically after successful writing.
The versioned format uses JSON metadata and typed HDF5 datasets, never pickle.
MolCore writes format version 4. Standalone datasets can still read format versions 1–3;
experiment snapshots store multiple datasets and their alignment records together.
Processing settings and package-version manifests round-trip with the dataset.

## CPU processing

`molcore.cpu.CpuExecutor` runs importable functions in spawned processes. Set
`num_cpus` explicitly, or leave one available CPU free by default (capped at the
Windows process-pool limit of 61). `max_pending`
bounds submitted jobs and therefore the number of image chunks in flight. The
executor limits native numerical thread pools to one thread per worker to avoid
nested oversubscription. One CPU still executes in a child process.

Each CPU function receives `(payload, reporter)`. Call
`reporter.report(completed, total, description)` for sub-job progress and
`reporter.check_cancelled()` at cooperative cancellation boundaries. The caller's
`progress(update)` callback receives a `ProgressUpdate` with `percent`, `completed`,
`total` and `description`, in the calling process/thread. Callback exceptions and
worker exceptions propagate; cancellation accepts an event with `is_set()`.

Use an executor context manager and a main guard in standalone programs. See
[examples/processing.py](examples/processing.py) for a runnable workflow:

```python
from molcore.cpu import CpuExecutor
from molcore.localisation import localise
from molcore.types import CameraSettings, LocalisationSettings

def run_processing(dataset):
    with CpuExecutor(num_cpus=4) as cpu:
        return localise(
            dataset, "recording", "image",
            camera=CameraSettings(baseline=100, sensitivity=1, gain=1),
            settings=LocalisationSettings(min_net_gradient=1000, fit_method="mle"),
            cpu=cpu,
            progress=lambda update: print(update.percent, update.description),
        )
```

The camera values above are examples; supply the calibration for the actual
camera. `localisation.detect()` returns typed candidates without changing active
results; `fit()` fits them; `localise()` combines both stages. Full successful
runs replace the selected acquisition/channel partition. Rejected fits are counted
in `LocalisationResult.failed_fits` and processing metadata; if all detected
candidates fail, an applying run raises and preserves existing data. A `frame_range` is a
camera-frame preview and requires `apply=False`. Inputs and ALEX frame provenance
come from the existing MolCore models; in-memory image chunks are streamed rather
than sending an entire movie with every job.

`undrifting.estimate_drift()` computes a `DriftResult`; `undrift()` also stores it
in `TransformManager`. Choose RCC or AIM through `DriftSettings(method=...,
segment_frames=...)`. Both require at least four populated temporal segments.
Coordinates are estimated in registered reference pixels. Camera-frame drift is
interpolated only within observed reference-channel coverage; original pixels
and localization coordinates remain unchanged. RCC uses a one-pixel Gaussian
rendering kernel. `undrift_many(dataset, {acquisition_id: channel, ...},
num_cpus=4)` processes independent acquisitions concurrently. Each estimator's
temporal passes remain sequential inside its worker.

`alignment.estimate_alignment()` fits translation or affine correspondences;
`align_channels()` also selects a new named calibration. Pass paired localization
IDs, or an explicit `match_radius` for mutual-nearest matching. Affine fits need
three non-collinear pairs. Results include pair residuals and compose through the
target channel's existing registration. Homography and GPU processing are outside
this increment.

See [application integration contracts](docs/api-contracts.md) for stable model
handles, automatic change notifications, scoped input dependencies and the common
`dataset.snapshot()` / `dataset.apply(result)` workflow.

All processors accept `num_cpus`, a reusable `cpu` executor, `progress`, and
`cancel_event`. Workers return detached results; the caller validates model
revisions before committing. Failed, cancelled, and stale runs preserve active
results. Processing parameters, camera settings and backend versions persist in
the dataset. Core model imports remain independent of Picasso, Qt and napari.
The processing backend currently targets the verified Picasso 0.11 API series.

## Cellpose 3 segmentation

`molcore.cellpose.segment()` segments one model-bound 2D source frame and returns
the existing `Segmentation` type. It stores the exact source pixels, instance
labels, camera-frame/channel binding, coordinate space and backend provenance.
`segment_many()` accepts `SegmentationRequest` objects and applies results only
after every job and stale-input check succeeds.

```python
from molcore.cellpose import segment
from molcore.types import CellposeSettings

def segment_cells(dataset):
    return segment(
        dataset, "recording", "image", segmentation_id="cells",
        settings=CellposeSettings(pretrained_model="cyto3", min_size=15),
        num_cpus=1,
        progress=lambda update: print(update.percent, update.description),
    )
```

As with other spawned processing calls, use a main guard in standalone scripts.
The default frame is the first actual camera frame in the logical channel, so
alternating excitation channels retain their original frame identity. Pass
`space="registered"` or `"corrected"` to segment those coordinate views, and
`apply=False` for a detached preview. Source images remain saved even when raw
movie pixels are excluded from an dataset snapshot.

Cellpose uses one CPU worker by default to avoid replicating a large neural
network across all available CPUs. A reusable `CpuExecutor` amortizes model
loading; each worker caches one model. Explicit `num_cpus` controls batch
parallelism. `CellposeSettings(gpu=True)` requests CUDA or MPS and requires
exactly one worker; unavailable hardware raises an error rather than silently
switching to CPU. Progress is stage-based, and cancellation is cooperative at
backend checkpoints rather than interrupting native inference kernels.

Supported built-in names are `cyto3` (default), `cyto2`, `cyto`, and
`nuclei`. Existing local Cellpose 3 weight paths support offline use.
Invalid names/paths are rejected before Cellpose can silently substitute a model.
Built-in weights are downloaded by Cellpose on first use. The segmentation
extra requires Cellpose 3.0.x; SAM/DINO models from Cellpose 4 are unsupported.

See the [official Cellpose 3 model documentation](https://cellpose.readthedocs.io/en/v3.0.9/models.html).
Cellpose 4/Omnipose checkpoints, training and 3D segmentation are outside this
module's scope. Tests exercise real Cellpose interfaces and preprocessing/mask
construction with controlled network outputs; pretrained weights and GPU
inference are not downloaded or exercised automatically.

## Development and current limits

```bash
uv sync --all-extras
uv pip install PySide6  # Qt binding for the full UI test suite
uv run python -m unittest discover -s tests -v
uv run mypy
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run python examples/foundation.py
```

Tests run on Python 3.12 and 3.14. CI covers Windows and Linux. Version one supports
2D time series, static masks, affine registration and result tables in RAM.
Nonlinear transforms, dynamic segmentation, synchronized multi-camera streams,
full processing histories are outside this release.

## Shared scientific processing

Typed settings and enum choices now support aperture photometry, stationary and
moving multichannel traces, Trackpy linking and actual-time trajectory statistics.
Stable-ID measurement/position updates, atomic mixed updates, partition track
replacement, safe acquisition removal and bulk frame/coordinate conversions are
also available. See [the scientific API guide](docs/scientific-api.md) and
[the runnable headless example](examples/scientific.py).

Autocontrast is available through `molcore.contrast.autocontrast` for arrays and
`autocontrast_channels` for lazy acquisition channels. Typed percentile/min-max
settings, masks and per-channel/shared display limits preserve original pixels.
See [the scientific API guide](docs/scientific-api.md#display-autocontrast).
