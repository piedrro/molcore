# Optical channel alignment

Channel alignment registers physical emission paths while leaving camera pixels and raw localization positions intact. The donor emission path is the reference for FRET and ALEX presets. FRET acceptor positions map into donor coordinates. In ALEX, DA and AA are the same acceptor emission path, so selecting a calibration for either selects it for both; DD and AD remain on the donor reference path. This applies to every original camera frame of each channel.

`image_acquisition(..., mode="fret")` and `mode="alex"` assign `ChannelSpec.emission` from the preset. A custom `ChannelSpec` defaults to `emission=None`, making it an independent optical path unless the caller explicitly supplies a shared path label.

## Estimate and reuse a calibration

```python
from molcore.alignment import estimate_alignment
from molcore.processing import apply_alignment

preview = estimate_alignment(
    dataset, "camera-1", "DA", "DD",
    source_ids=acceptor_fiducials,
    target_ids=donor_fiducials,
    model="homography",  # also "translation" or "affine"
)
applied = apply_alignment(dataset, preview, model="homography", name="optics-2026")
dataset.transforms.select("camera-2", "AA", applied.calibration_name)
```

The fiducial ID lists must be paired in corresponding order. Without IDs, pass a positive `match_radius` for mutual nearest-neighbor matching. All three fit models use deterministic RANSAC (default one-pixel inlier threshold, 1,000 trials). `AlignmentResult` exposes the matrix, residual per match, inlier mask, and source and target IDs. A preview does not change the dataset; `apply_alignment` installs a named calibration and selects the source emission group.

Named calibrations can be selected on other acquisitions when emission path, source and reference geometry, and pixel scale agree. A fitted calibration stores those properties, its fiducial evidence, and `selected_groups` provenance. The latter lists each acquisition, emission path and logical channels currently using the calibration. Imported matrices without geometry metadata remain explicit user supplied calibrations.

## Coordinates and images

The registration matrix maps raw channel coordinates into reference coordinates. A homography divides by its third homogeneous coordinate; points at a projective pole become NaN. Per-camera-frame drift is subtracted *after* registration: `corrected = translation(-drift) @ registration`. Conversion into another channel uses the inverse of that channel's registration. Rendering propagates localization precision through the local projective Jacobian.

`dataset.transforms.view(acquisition_id, channel, space="corrected")` is a lazy stack. It samples one requested frame by inverse warping; the original source remains unchanged. `export_tiff(dataset, acquisition_id, channel, path, space="corrected")` explicitly writes the entire aligned channel stack and a YAML provenance sidecar containing original camera-frame indices. `space="raw"` exports unaligned channel pixels.
