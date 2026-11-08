# Sequencing datasets and alignment

An `Experiment` represents one surface followed through successive probing rounds.
Each `Dataset` is a round with its own images and scientific results. Record
`round_index`, oligo identities and barcode sequences in dataset metadata.
An `Acquisition` is one original camera stream; split recording fragments may share
a dataset and an explicit acquisition sequence. Separate rounds never share a timeline.

```python
from molcore import Dataset, Experiment

experiment = Experiment(metadata={"surface": "slide-1"})
experiment.add_dataset(Dataset(
    id="round-1", name="First probing round",
    metadata={"round_index": 1, "oligos": [{"name": "probe-A", "barcode": "ACGT"}]},
))
dataset = experiment.datasets["round-1"]
```

`Dataset` replaces the previous scientific `Experiment` facade. Pass a dataset to
localisation, tracking, photometry and other existing processing functions.
Localisation, molecule, track and acquisition IDs are local to each dataset;
cross-round references must include the dataset ID. Spatial alignment does not
establish molecular identity or decode barcodes.

## Optical alignment

Physical emission groups are explicit in `ChannelSpec.emission`. The FRET/ALEX
image-import presets populate these groups automatically, for either split layout.

| Mode | Reference emission group | Transformed emission group |
| --- | --- | --- |
| FRET | Donor | Acceptor |
| ALEX | DD and AD | DA and AA |

Fit using suitable fiducials in one representative channel from each group.
Selecting the calibration applies it to every channel in the source emission
group for every frame. Reuse a calibration across compatible rounds with
`experiment.apply_channel_calibration(calibration)`. Validation completes for
all selected rounds before any round changes. Custom channels without emission
metadata remain independent. See [channel alignment](channel-alignment.md).

## Movement between rounds

After optical alignment and local drift correction, estimate the movement of the
surface using stable fiducials or explicitly corresponding features. A fluorescent
reporter need not appear in every round.

```python
from molcore import ExecutionContext

result = experiment.alignment.estimate(
    "round-2", "round-1",
    source_acquisition_id="movie", source_channel="DD",
    reference_acquisition_id="movie", reference_channel="DD",
    source_ids=[1, 2, 3, 4], reference_ids=[1, 2, 3, 4],
    context=ExecutionContext(num_cpus=1),
)
experiment.alignment.apply([result])
aligned_movie = experiment.alignment.view("round-2", "movie", "DA")
aligned_frame = aligned_movie[0]
```

Ordered ID lists establish fiducial pairs. Without them, provide a positive
`DatasetAlignmentSettings.match_radius` for mutual nearest-neighbour matching.
Available models are translation, affine and projective homography; fitting uses
deterministic outlier rejection. Automatic matching assumes that corresponding
features are initially within the selected radius. Results retain paired IDs,
inliers, residuals and RMS error.

The selected acquisition/channel defines the dataset's local spatial reference.
Other acquisitions must already share that reference to use its round transform.
Each source round maps directly to one reference round. Reset the alignment set
before changing the reference. Edits to fiducials, channel geometry, registration
or drift make affected alignments stale; re-estimation is required.

`experiment.alignment.coordinates(dataset_id, rows, space="raw")` maps row
coordinates into the common surface view. State `space="corrected"` explicitly
when passing coordinates queried in the dataset's default corrected space.
Dataset-local queries, photometry and crops retain their local conventions.

## Images, snapshots and export

Aligned stacks are lazy. Each requested frame is resampled directly from its raw
channel into the reference canvas, composing optical registration, local drift
and round movement in one mapping. Original pixels and ALEX camera-frame indices
remain unchanged. Common-view images use the reference channel's canvas, with
zeros outside the source footprint.

`Dataset.save/open` work independently. `Experiment.save/open` include every
round and its alignment records. Snapshots support nested lazy movie and crop
sources. Full-image snapshots always store original camera pixels.

Use an explicit export to bake transformations:

```python
from molcore.exports import export_dataset_images

export_dataset_images(
    experiment.datasets["round-2"], "aligned-round-2",
    space="aligned", experiment=experiment,
)
```

Exports stream channel TIFF stacks and a manifest retaining the original frame
and excitation provenance. `raw`, `registered` and `corrected` exports also work
with standalone datasets. Cancellation, stale inputs and write failures roll back
the output file set.

`UiSession(experiment)` exposes a dataset selector across all shared tools. The
Dataset alignment panel provides fiducial preview, optional image overlays,
Apply and Reset. Files can save/open entire experiments, append datasets, or
import recording fragments into the selected dataset explicitly.

## Segmentation outlines and fitted cells

`Dataset.segmentations` stores integer label masks. For editable subpixel
polygons, use `add_segmentation_outline(SegmentationOutline(...))` and read
`Dataset.segmentation_outlines`. Vertices use `(x, y)` order. An outline can
apply across frames or specify one frame. If bound to an acquisition and
channel, the frame is an original camera-frame index.

`add_bactfit_cell(BactFitCell(...))` stores a fitted polygon, midline, two poles,
polynomial coefficients and radius, with an optional fit error. A fit can be
standalone or reference a source outline by ID. Replacing or removing a source
outline clears its linked fits. Both collections are saved in format version 7
independently of the snapshot's image policy.
