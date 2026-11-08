# Scientific processing APIs

Molcore owns scientific data, validation, transformations and processing. GUI
widgets, napari layers and legacy dictionary compatibility belong to consumers.
New finite settings use enums; arbitrary identifiers and measurement names remain
strings. Public processing inputs and outputs have type annotations.

## Editing and transactions

`localizations.update_measurements(ids, **columns)` updates real numeric columns
in the supplied stable-ID order. Scalars broadcast; arrays must match the ID count.
IDs must be unique and present. New columns are nullable elsewhere, preserving their native dtype and validity masks. Structural
columns cannot be edited this way. Updates invalidate observations anchored to
changed localizations or their tracks; image crops and positions stay intact.

`localizations.move_many(ids, x=..., y=...)` atomically edits effective raw-channel
positions, preserving imported originals and invalidating dependent crops and
observations. `query(original=True)` accesses imported coordinates.

`tracks.query(acquisition_id=..., channel=...)` filters members before coordinate
transforms, including when sequences are selected. A spanning track may therefore
return only its selected members. `tracks.replace(acquisition_id, channel,
memberships)` accepts arrays of existing localization IDs and returns fresh uint64
track IDs. Empty memberships clear the partition. Intersections with spanning
tracks, duplicate ownership, duplicate camera frames and invalid masks are rejected
before any active data changes.

`state_token`, `clone_dataset` and `transact` in `molcore.staging` support detached
work and atomic commits. Tokens now also cover timestamps, channel specifications,
excitation settings and sequence options. Borrowed image pixel arrays must not be
mutated during processing; tokens do not hash full movies.

`processing.apply_batch(dataset, operations, expected=token)` runs typed
model-update callables against one detached model and adopts all changes only on
success. Operations must restrict side effects to that model; external I/O and
borrowed image arrays cannot be rolled back. Use model edit methods in operations.
Batch operations may apply detached results sharing the original batch token while
their scientific inputs remain unchanged. Independent observation/provenance updates
can accumulate; edits to localizations, memberships, transforms, acquisition bindings,
masks or dataset metadata invalidate subsequent previews based on the earlier state.
Apply independent photometry results before replacing tracks; dependent processing
must be recomputed from the updated model.
Transactions recheck freshness immediately before adoption. Public model mutators
and adoption share a reentrant lock; direct dictionary or borrowed-array edits are
not synchronized and must not run concurrently with processing.
Scientific functions require `dataset.snapshot()` and return immutable detached
results. Use `PartitionSelection` or `SequenceSelection` where the signature calls
for a selection, and `context=ExecutionContext(...)` for execution policy.
`Dataset.apply(result)` returns an immutable `ApplyReceipt` with allocated IDs;
`apply_all` commits multiple independent results atomically. Operation-specific
apply helpers are private. See [integration contracts](api-contracts.md).

`Dataset.remove_acquisition(id, cascade=False)` rejects scientific dependents.
With `cascade=True`, it removes the acquisition's localizations, observations,
crops, masks, selected transforms, drift and processing records. Affected sequences
and whole spanning tracks are removed; surviving acquisitions' localizations remain.
Stale observations referencing deleted anchors are physically removed. Reusable
named calibrations remain. Shared or borrowed sources are never forcibly closed:
built-in readers open files per read, and removal releases the source reference.
Normal `Dataset.close()` still closes sources of remaining acquisitions.

## Photometry and traces

Use `measure_localizations`, `stationary_traces`, or `track_traces` from
`molcore.photometry`. Localisation measurements return `PhotometryResult`;
stationary and moving traces return `PhotonSequenceResult` containing primary
photon-sequence records. They accept explicit typed
arguments for `channels`, `settings`, optional half-open camera `frame_range`,
`context` for executor, CPU count, progress and cancellation.

`PhotometrySettings` defaults to a 3-pixel square signal aperture, 1-pixel excluded
buffer and 1-pixel local background ring. Use `ApertureShape.SQUARE` or `.CIRCLE`
and `BackgroundMethod.LOCAL` or `.GLOBAL`. Circle radius is `spot_size / 2`; square
half-width is `spot_size // 2`. Sizes must be positive odd integers. Aperture sizes
are defined in each target channel's pixels, not warped reference pixels.

Centres round with `floor(position + 0.5)`, matching crop extraction. Original image
pixels are never resampled. Regions clip to the image bounds and exclude nonfinite
pixels. Returned float64 columns include `signal_*`, `local_background_*`, and
`global_background_*` for sum, mean, median, min, max and population std, plus integer
pixel counts. Background sums describe actual background pixels, not an aperture
area estimate. `intensity = signal_sum - selected_background_mean * signal_count`.
Empty regions yield NaN metrics and count zero; negative intensities are retained.
Units are original image intensity units, without implicit camera conversion.

Neighbour exclusion defaults on and masks known signal apertures from both local
and global background regions. Neighbours comprise requested positions in that
frame and active localizations at the same camera frame, mapped across channels.
It does not infer unobserved neighbours or interpolate track positions. Disable
`exclude_neighbours` to obtain unmasked background estimates.

Stationary seeds define fixed molecules in corrected reference coordinates.
Drift is subtracted at the anchor frame, then added at each sampled frame before
inverse target registration. With `follow_drift=False`, the registered reference
position is fixed instead. An acquisition with no drift uses zero displacement;
once drift is supplied, every requested frame must have explicit coverage.

Stationary traces sample each target channel's actual camera frames, including
alternating excitation. Localization and moving-track measurements require the
target channel to measure the member's actual camera frame; unavailable target
frames are omitted. Moving tracks have no interpolation or extrapolation.

Results retain camera/channel frames, timestamp (NaN when absent), excitation cycle
(-1 when absent), acquisition, channel, raw aperture centre, molecule key and
localization/track anchors. No cross-recording elapsed time is invented. Molecule
keys are `localization:<id>` or `track:<id>`.

Applying localisation measurements appends observations and returns observation IDs.
Applying traces writes directly to `dataset.photon_sequences` and returns sequence
IDs; it creates no observation records or additional spatial localisations.
Each sequence owns its samples, raw crop centres, per-sample localisation references,
signal/background measurements and explicit units. `PhotonSequenceResult.columns` is
a transient display/export projection of those records. Run settings and provenance
are retained; earlier runs remain queryable until invalidated.
Computation always produces a detached preview; apply explicitly with `Dataset.apply`.
Save/open uses format 7, including measurement schemas and nullable validity masks. Raw channel pixels are required; incomplete crop snapshots cannot be
used to reconstruct traces.

One camera frame is streamed per job; `CpuExecutor.max_pending` bounds in-flight
image data. Final observation columns remain in RAM. All processors accept reusable
executors, report progress in the caller and cooperate with cancellation. Standalone
scripts using spawned workers require an `if __name__ == '__main__'` guard.

## Tracking and trajectory statistics

For explicitly concatenated recordings, use `molcore.tracking.track_sequence`
with an acquisition sequence that allows cross-boundary tracking. The default
channel clock counts logical frames across the sequence. Stored memberships
keep their acquisition-local camera frames and localisation IDs.

`tracking.track` uses Trackpy >=0.7,<0.8, loaded only when tracking is requested.
`TrackingSettings` defines search range in selected-space pixels, missing-camera-frame
memory, minimum measured length and `CoordinateSpace`. Empty camera frames,
including other excitation frames, consume memory. Segmentation labels form
independent linking groups and label zero is excluded. Each acquisition is linked
independently; explicit existing model APIs still support spanning tracks.
Applying full results atomically replaces the partition's tracks, including empty runs.

`trajectory.trajectory_statistics` returns `TrajectoryStatistics`: step columns,
MSD columns, within-acquisition duration, path length and optional `DiffusionFit`.
Choose `SpatialUnit.PIXELS` or `.NANOMETRES`; nanometres require positive pixel
calibration. Recorded timestamps take precedence; if absent, supply a positive
camera-frame interval in seconds. Channel row indices never substitute for camera
frames. Selected timestamps must be strictly increasing within each acquisition.

Steps include dx, dy, distance, elapsed seconds, speed and missing-camera-frame
count. Steps and MSD pairs never cross acquisition boundaries; duration sums each
acquisition's observed span and excludes unknown recording gaps. Every within-
acquisition position pair contributes to MSD grouped by actual positive elapsed
lag (12 significant digits), or explicit lag-bin edges. Bins are half-open except
for the final inclusive edge; reported lag is the mean actual lag in the bin.
Empty bins have count zero and NaN values. Returned pair counts are raw counts,
not estimates of statistically independent samples. Computation is quadratic in
track length and uses linear temporary storage per anchor position.

Diffusion fitting is opt-in via an explicit inclusive `fit_lag_range`. At least two
distinct populated lags are required. The unweighted 2D fit is `MSD = 4 D t + b`;
results include D, intercept, slope, R-squared, residual sum of squares and fitted
lag count. Negative D is reported without clipping. This is a numerical estimate,
not a motion-model selection or localization-error correction procedure.

`ChannelView.camera_frames(indices)` and `.channel_indices(frames)` provide ordered
vector conversions, rejecting unavailable frames and negative indices.
`transforms.convert_coordinates(rows, source_space=CoordinateSpace..., target_space=...,
target_channel=...)` converts finite 1D columns through raw/reference coordinates
without resampling. Cross-channel conversions require the camera frame to exist in
both channels; stationary trace sampling handles different channel times separately.
Corrected-space conversions require available drift values.

See `examples/scientific.py` for a complete headless workflow.

## Consumer migration and disabled controls

No moltrack or molseeq files are changed. Moltrack can replace metadata measurement
overlays with `update_measurements`, private staging with `transact`/`apply_batch`,
manual deletion with `remove_acquisition`, and linking/photometry adapters with the
new processors. Such migration needs application tests and remains separate.

Source inspection confirms these five controls are disabled together in moltrack's
backend UI. Their labels help distinguish the intended behaviour:

| Control | Assessment and next action |
| --- | --- |
| Localization window cropping | Label says remove localizations outside the FOV. Molcore already requires complete fitting boxes inside logical channel bounds; a separate FOV/window definition needs a concrete workflow before adding filtering. |
| Overlapping localization removal | Label refers to overlapping bounding boxes determined by Picasso box size. This is a separate spatial exclusion policy, not background masking or clustering; choose survivor/overlap rules before implementing it. |
| ROI border width | No processor setting currently expresses this extra exclusion margin. A caller-defined ROI-border filtering requirement is needed; logical image ROI bounds already exist. |
| Cellpose image inversion | No current CellposeSettings inversion option. Specify inversion range/dtype and whether it occurs before normalization before adding scientific preprocessing. |
| Minimise RAM usage | Localization already streams bounded image chunks. Integrate chunk size and executor pending-job limits into the application first; results remain in RAM and this is not disk-backed localization storage. |

These assessments establish current behaviour, not numerical equivalence to the
legacy controls. No disabled-control behaviour is added speculatively.

## Display autocontrast

`molcore.contrast.autocontrast(image, settings=..., mask=...)` returns a typed
`ContrastResult` for a 2D array or `(time, y, x)` movie. Use `.limits` directly as a
viewer's contrast limits. It never edits pixels or scientific observations.

```python
from molcore import ContrastMethod, ContrastScope, ContrastSettings
from molcore.contrast import autocontrast, autocontrast_channels

settings = ContrastSettings(
    method=ContrastMethod.PERCENTILE,
    percentiles=(1.0, 99.0),
    max_frames=16,
    max_pixels_per_frame=65_536,
)
limits = autocontrast(image, settings=settings).limits
results = autocontrast_channels(
    dataset.snapshot(), "recording", settings=settings,
    scope=ContrastScope.PER_CHANNEL,
)
# A consuming application can set layer.contrast_limits = results[channel].limits
```

`ContrastMethod.MIN_MAX` computes exact extrema within selected frames. Percentile
limits use NumPy's linear percentile interpolation. Nonfinite pixels are excluded;
optional boolean masks include pixels where True. A 2D mask applies to every frame;
a 3D mask must match the full logical-channel stack. Model calls accept masks by
channel name and an optional half-open camera `frame_range`.

Movie defaults sample at most 16 evenly spaced frames, including the endpoints
when two or more frames are selected. Percentile calculations sample at most
65,536 evenly spaced finite included pixels per frame. Set either cap to None to
use every frame/pixel. Sampling is deterministic and approximate; it may miss
rare intensity populations. MIN_MAX ignores the pixel cap and examines every valid
pixel in sampled frames; use `max_frames=None` for exact whole-movie extrema.
Lazy model calls read one selected frame at a time without materializing a movie.
Percentile sample storage grows with the selected frames and channel count.

`ContrastScope.PER_CHANNEL` calculates independent limits. `.SHARED` pools channel
pixel samples before calculating common limits, rather than averaging channel
percentiles. Each channel's result records its own original camera frames, settings,
valid-pixel count and sampled-pixel count, even when the display limits are shared.
Array results record array frame indices. Constant limits expand to a finite,
strictly increasing interval; no finite unmasked pixels or an empty frame selection
raises ValueError. An empty selected channel also raises in shared mode.

Both APIs accept typed progress and cancellation callbacks. Model calls reject
stale input state. Display limits remain caller-owned; they are not stored as
scientific processing results or applied to napari automatically.

A standalone demonstration is available in `examples/autocontrast.py`.
