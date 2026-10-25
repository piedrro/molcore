# Independent photon sequences and crop storage

`PhotonSequence` is a separate frame-resolved data model. It does not require a
spatial track or a localisation in every frame. One sequence belongs to one
acquisition and logical channel; use a shared `molecule_id` to associate channels.
This model stores per-frame estimates, rather than individual photon-arrival events.

Tracks remain ordered memberships of existing localisations. Photon sequences own
their intensity samples and measurements; an optional track link records where a
trace was measured, without turning photon samples into track members.

`stationary_traces()` and `track_traces()` now return `PhotonSequenceResult`.
Their default applying runs write directly to `experiment.photon_sequences`.
`apply=False` returns detached sequence records; use `apply_photon_sequences()`
to commit them explicitly. Localisation-only photometry continues to use
`PhotometryResult` and the observation table.

```python
from molcore.photometry import stationary_traces

result = stationary_traces(experiment, seed_localization_ids)
sequence = experiment.photon_sequences[result.sequence_ids[0]]
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
experiment.photon_sequences.add(sequence)
```

`frames` must increase strictly and belong to the selected channel. The model
retains gaps, NaN measurements, negative background-corrected estimates and explicit
sample validity. Recorded timestamps are populated on addition; supplied times
must agree with recorded camera times. Unknown timestamps stay NaN. No sample or
elapsed time is invented.

`unit` distinguishes `PHOTONS` from `IMAGE_INTENSITY`. Setting a unit records the
meaning of supplied values; it does not calibrate camera pixels or run PSF fitting.
Numeric arrays are copied into immutable storage. Store access returns detached
metadata, so editing a retrieved object does not edit the experiment.

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

for sequence in from_observations(experiment, run_id="measurement-run"):
    experiment.photon_sequences.add(sequence)
```

The conversion copies actual source positions, frames and anchors. Its default
unit is image intensity because aperture photometry does not automatically produce
calibrated photon counts. Choose another numeric column and explicit unit when
converting calibrated or fitted-photon measurements. Conversion does not mutate the
experiment until the returned records are added.

## Crop policies

The sequence's optional `positions` array supplies a raw-channel `(x, y)` centre
for each sample. Fixed positions support stationary molecules; varying positions
support moving emitters. Cropping never requires synthetic localisations.

```python
crops = list(experiment.crops.photon_sequence("molecule-1-donor", width=7))
experiment.save("analysis.h5", image_policy="all_crops", crop_width=7)
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

Snapshots now use format version 3 for sample metrics and anchor arrays; version-1
and version-2 files remain readable. Earlier molcore versions reject version-3
snapshots rather than silently lose data.

The trace viewer defaults to **Trace data → Photon sequences** and reads the primary model
directly. It does not insert photon samples into the observation or track tables.
