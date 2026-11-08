# Independent photon sequences and crop storage

`PhotonSequence` is a separate frame-resolved data model. It does not require a
spatial track or a localisation in every frame. One sequence belongs to one
acquisition and logical channel; use a shared `molecule_id` to associate channels.
This model stores per-frame estimates, rather than individual photon-arrival events.

Tracks remain ordered memberships of existing localisations. Photon sequences own
their intensity samples and measurements; an optional track link records where a
trace was measured, without turning photon samples into track members.

`stationary_traces()` and `track_traces()` now return `PhotonSequenceResult`.
Compute from `dataset.snapshot()` and commit the detached result with
`dataset.apply(result)`. The returned receipt contains allocated sequence IDs. Localisation-only photometry continues to use
`PhotometryResult` and the observation table.

```python
from molcore.photometry import stationary_traces

result = stationary_traces(dataset.snapshot(), seed_localization_ids)
receipt = dataset.apply(result)
sequence = dataset.photon_sequences[receipt.sequence_ids[0]]
signal = sequence.measurements["signal_sum"]
background = sequence.measurements["local_background_mean"]
```

Each record has its own `run_id`, `measurements` arrays and optional per-sample
`localization_ids`. Numeric samples and their metrics are immutable. The result's
`columns` property is a temporary plotting/export view; it is not a second stored
observation representation.

When selecting an alternate primary quantity during legacy conversion, the
original intensity/photon columns are retained as `source_intensity` and
`source_photons` measurements. The export view's `intensity` and `photons` columns
refer to the chosen primary sequence values; `metadata['value_column']` records
that choice.

```python
import numpy as np
from molcore import PhotonSequence, PhotonUnit

sequence = PhotonSequence(
    id="molecule-1-donor",
    acquisition_id="recording",
    channel="donor",
    frames=np.array([0, 2, 4]),  # original camera frames, including ALEX gaps
    photons=np.array([120., np.nan, 98.]),
    positions=np.array([[12., 20.], [12., 20.], [12., 20.]]),
    molecule_id="molecule-1",
    unit=PhotonUnit.PHOTONS,
    method="psf_fit",
    metadata={"camera_calibration": "calibration-2026"},
)
dataset.photon_sequences.add(sequence)
```

`frames` must increase strictly and belong to the selected channel. The model
retains gaps, NaN measurements, negative background-corrected estimates and explicit
sample validity. Recorded timestamps are populated on addition; supplied times
must agree with recorded camera times. Unknown timestamps stay NaN. No sample or
elapsed time is invented.

`unit` distinguishes `PHOTONS` from `IMAGE_INTENSITY`. Setting a unit records the
meaning of supplied values; it does not calibrate camera pixels or run PSF fitting.
Numeric arrays are copied into immutable storage. Store access returns detached
metadata, so editing a retrieved object does not edit the dataset.

Optional `localization_id` and `track_id` link a sequence to its source anchor.
Anchor edits mark linked sequences stale. Set `depends_on_transforms=True` for
measurements derived using registration or drift; subsequent transform edits then
mark them stale. Converted observation runs set this flag. Independent imports
remain active when unrelated localisations, segmentations or transforms change.
Native traces using neighbour exclusion set `depends_on_localizations=True`:
localisation additions, moves or deletions in the acquisition then invalidate
their background-dependent measurements. Traces without neighbour exclusion
continue to depend on their own anchors and declared transforms only.
`query()` excludes stale records by default; `query(include_stale=True)` and
`is_stale(id)` expose their history. Removing an acquisition checks photon-sequence
dependencies and requires `cascade=True` to remove them.

Legacy observation runs can be explicitly converted without creating tracks or
additional localisations:

```python
from molcore.photon_sequences import from_observations

for sequence in from_observations(dataset, run_id="measurement-run"):
    dataset.photon_sequences.add(sequence)
```

The conversion copies actual source positions, frames and anchors. Its default
unit is image intensity because aperture photometry does not automatically produce
calibrated photon counts. Choose another numeric column and explicit unit when
converting calibrated or fitted-photon measurements. Conversion does not mutate the
dataset until the returned records are added.

## Crop policies

The sequence's optional `positions` array supplies a raw-channel `(x, y)` centre
for each sample. Fixed positions support stationary molecules; varying positions
support moving emitters. Cropping never requires synthetic localisations.

```python
crops = list(dataset.crops.photon_sequence("molecule-1-donor", width=7))
dataset.save("analysis.h5", image_policy="all_crops", crop_width=7)
```

| `image_policy` | Saved microscopy pixels |
| --- | --- |
| `none` | No movie or crop pixels |
| `localization_crops` | Localisation crops |
| `track_crops` | Crops at measured track members |
| `both` | Localisation and track crops, preserving the existing meaning |
| `photon_sequence_crops` | Photon-sequence sample crops |
| `localization_and_photon_crops` | Localisation and photon-sequence crops |
| `track_and_photon_crops` | Track and photon-sequence crops |
| `all_crops` | All three crop categories |
| `full` | Full original camera movies |

Photon-sequence sample values and provenance are saved under every policy. The
policy controls pixels only. Crop-only snapshots remain usable without movies,
including re-saving their existing crops at the same width.

Each photon crop retains `photon_sequence_id`, `sample_index`, source acquisition,
channel, original frame, raw origin and a per-pixel validity mask for edge padding.
Paired NaN positions yield `None` at that sample index; later samples keep their
original indices. A sequence without positions can store/export its measurements,
but requesting its crops raises a clear error. Unsaved widths require original
pixels. Stale sequence crops require explicit `include_stale=True`.

Snapshots use format version 7 for sample metrics, anchor arrays and nullable
table schemas. Only current-format files are accepted; see [migration](api-migration.md).

The trace viewer defaults to **Trace data → Photon sequences** and reads the primary model
directly. It does not insert photon samples into the observation or track tables.

## Explicit stationary sites

### Persistent bounding boxes

Use the dataset's bounding-box store when measurement sites must be edited,
saved or linked to sequence history.

```python
from molcore import BoundingBox, PartitionSelection
from molcore.photometry import FixedApertureSettings, bounding_box_traces

box = dataset.bounding_boxes.set(BoundingBox(
    x=42.5, y=19.0, width=7, height=7,
    acquisition_id="movie", channel="donor",
    metadata={"origin": "manual", "label": "molecule A"},
))
result = bounding_box_traces(
    dataset.snapshot(), PartitionSelection("movie", "donor"), [box.id],
    settings=FixedApertureSettings(spot_size=3, buffer=1, background_width=2),
)
receipt = dataset.apply(result)
sequence = dataset.photon_sequences[receipt.sequence_ids[0]]
print(sequence.bounding_box_id, sequence.bounding_box_revision)
print(sequence.metadata["bounding_box"])
print(sequence.metadata["extraction"])
dataset.save("measurement.h5", image_policy="none")
```

Each sequence retains the box definition used for extraction, its revision,
signal and background settings, source acquisition and channel, software
versions and creation time. Optional PSF settings and working-source provenance
are included. Box width and height describe the defining region. Signal and
background apertures are independently chosen for each extraction run.

Editing a definition with the same ID advances its revision and marks dependent
sequences stale. Shared background masks also depend on the other requested
boxes. Removing a box preserves its deletion history and the sequence's original
definition. Query stale sequences explicitly with `include_stale=True`.

Working-space boxes require an explicit working image source. Their coordinates
are recorded in that space. The caller can provide `source_metadata` describing
the pixel-processing history. MolSeeq stores shared selections in experiment
metadata, synchronizes their IDs into each round's bounding-box store, and keeps
working-image recipes in the round's processing metadata.

Snapshot format 6 preserves boxes, deletion history and sequence references under
every image policy. Formats 1–5 remain readable, with an empty box collection
when definitions were absent. Earlier readers reject format 6.

`molcore.photometry.stationary_site_traces()` takes a mapping of molecule IDs to
`(x, y)` centres for one acquisition and logical channel. The centres stay fixed
in the measured pixel space, and the results retain that channel's camera frames.
Use the optional `source` for a working image with identical shape and frame
ordering. Working-image coordinates are recorded separately from raw crop
centres.

`FixedApertureSettings` enables separate local-window and masked-ring summaries,
optional whole-frame backgrounds, and optional Picasso Gaussian fit metrics.
Aperture intensities and fitted amplitudes are stored separately. The fitted
amplitudes use the supplied image units.
