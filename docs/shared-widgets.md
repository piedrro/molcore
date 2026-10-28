# Shared analysis widgets

Install the UI extras and a Qt binding, then launch the example:

```sh
pip install "molcore[ui]" PySide6
python examples/shared_widgets.py
```

Molcore does not create a QApplication, select a binding or import napari. Choose
your binding before importing `molcore.ui`. One session coordinates an dataset,
selections and a cancellable background job across all embedded widgets.

```python
from molcore.ui import MolCoreToolsWidget, TrackingWidget, UiSession

session = UiSession(dataset)  # borrows the dataset
tools = MolCoreToolsWidget(session, parent=host_widget)
# Or embed one tool:
tracking = TrackingWidget(session, parent=host_widget)
tracking.result_ready.connect(host.show_tracking_preview)
tracking.selection_requested.connect(host.select_entities)
session.select("recording", "donor", space="registered")
session.set_frame(0)  # original camera frame belonging to this channel
```

Call `session.changed()` after host-side model mutations. Replace an idle model
with `session.set_dataset(new_model)`. Call `session.shutdown()` during teardown;
keep the event loop running until `shutdown_finished` when cancelling an active
job. Closing a borrowed widget does not close its session.

## Tools

| Widget | Inputs and outputs |
| --- | --- |
| `TrackingWidget` | Linking settings and optional segmentation; preview memberships, then explicitly replace partition tracks. |
| `PhotometryWidget` | Localisation/stationary/track modes, channels, aperture and background; localisation measurements apply to observations, while traces apply directly to photon sequences. |
| `TraceViewerWidget` | Observations or independent photon sequences grouped by run and molecule; channel/metric, frame/time axes, background correction and display normalisation. |
| `TrajectoryStatisticsWidget` | Track IDs, units, timing fallback, lag bins and fit bounds; inspect speed, steps, MSD and population diffusion. |
| `ResultsFilterWidget` | Numeric metric, inclusive bounds and track aggregation; inspect distributions and export subsets. |
| `LocalisationRenderWidget` | Picasso blur, oversampling and minimum blur; export TIFF plus JSON geometry. |
| `ColocalisationWidget` | Two channels and Euclidean radius; inspect pairs, distances, midpoints and unmatched counts. |
| `ClusteringWidget` | DBSCAN radius and minimum localisation count; inspect assignments, noise (`-1`) and centroids. |

Blank ID inputs select the current partition; explicit IDs must belong to it.
Photometry apertures have odd sizes. Tracking memory counts camera frames,
including other excitation frames. Measurements retain actual frame and excitation
provenance. Missing timestamps require a frame axis or an explicit trajectory
frame interval. Physical units require calibration. Acquisition gaps never imply
elapsed time or displacement.

Changed settings, selection or model state invalidate previews. Changed inputs
during a job discard its completion. Export and Apply reject stale results. Tables
show at most 500 rows; exports include all rows.

Only explicit tracking and photometry Apply buttons update scientific tables.
Other analysis results are transient and exportable. Centroids and midpoints never
replace localisations. Display normalisation does not modify observations; trace
exports add `display_value` alongside original measurements.

Every new tool emits `result_ready(result)` with a detached preview; the session
also emits `preview_changed(result)`. Selection-capable tools emit
`selection_requested` with `acquisition_id`, `channel`, `space`, `entity` and copied
stable IDs. The host handles highlighting and navigation. The container exposes
tools in `tools.widgets`, keyed by tab title.

Selections resolve the anchors' source channels, which may differ from a measured
channel. A selection spanning source partitions emits one payload per partition.
Whole-track filtering reduces all members of a candidate track, including members
in other acquisitions of an explicitly linked sequence.

## Headless analysis

```python
from molcore.filtering import filter_results
from molcore.clustering import cluster_localizations
from molcore.colocalisation import colocalise
from molcore.rendering import render_localizations
from molcore.types import FilterSettings, ClusteringSettings, ColocalisationSettings, RenderSettings

selected = filter_results(dataset, "recording", "donor",
                          settings=FilterSettings("photons", 100, 1000))
clusters = cluster_localizations(dataset, "recording", "donor",
                                settings=ClusteringSettings(radius=1., min_samples=5))
pairs = colocalise(dataset, "recording", "donor", "acceptor",
                   settings=ColocalisationSettings(radius=2.), space="registered")
density = render_localizations(dataset, "recording", "donor",
                               settings=RenderSettings(oversampling=5.))
```

These functions do not import Qt. Results carry acquisition/channel, coordinate
space, settings and revision tokens. Colocalisation matches localisation sets
across frames for fiducial analysis, rather than asserting temporal coincidence.
Equal-distance ties use stable IDs. Raw ROIs must coincide; use a common registered
reference for different spatial views. DBSCAN samples count localisations rather
than distinct frames. Filters exclude non-finite values; track reductions ignore
non-finite members and reject tracks with no finite measurement.

Rendering delegates to Picasso. Sampling uses coordinate pixels, so physical
calibration is optional. Precision blur requires finite positive `lpx/lpy`;
registered variances propagate through the affine transform, assuming independent
raw x/y uncertainties. Results include `(x, y)` origin and coordinate-pixel spacing.
Transformed default bounds enclose transformed positions. Output is limited to
100 million pixels.

The shared tools use Euclidean matching and preserve source data. The plugin
repositories and their legacy panels are unchanged.

See [photon sequences](photon-sequences.md) for the independent data model and
localisation/track/photon crop-save combinations. Select **Trace data → Photon
sequences** in the trace viewer to inspect those records directly.
