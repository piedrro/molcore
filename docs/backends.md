# Execution and optional processing backends

Core imports do not load Picasso, Trackpy, Cellpose, scikit-learn or Qt. Install
`molcore[localisation]`, `[tracking]`, `[segmentation]`, `[clustering]`, `[ui]`, or
`[all]` as needed. The UI extra needs a host-provided Qt binding. SciPy remains core.

## Execution policy

Scientific processors take a single `context=ExecutionContext(...)` argument.
CPU count, executor, progress, cancellation and chunk size are execution policy;
scientific parameters remain in settings. Results record these separately.

```python
from molcore import CpuExecutor, ExecutionContext, PartitionSelection
from molcore.localisation import localise

with CpuExecutor(num_cpus=2, max_pending=4) as executor:
    result = localise(
        dataset.snapshot(), PartitionSelection("recording", "image"), camera=camera,
        context=ExecutionContext(cpu=executor, chunk_size=16, progress=on_progress),
    )
receipt = dataset.apply(result)
```

Contexts borrow executors and never close them. Choose `num_cpus` or `cpu`, not
both. `SerialExecutor`, `ThreadExecutor` and spawned `CpuExecutor` implement ordered
`imap(function, jobs, *, total=None, progress=None, cancel_event=None)` and collecting
`map(...)`. Pending jobs plus buffered completed results are bounded by
`max_pending`; slow earlier jobs do not allow unbounded input consumption.
Closing an iterator releases its pending work and the executor can be reused.

Workers receive `(payload, reporter)`. The reporter provides `report(completed,
total, description)` and `check_cancelled()`. Progress callbacks run on the caller
thread; callback and worker errors propagate. Cancellation is cooperative. CPU
jobs must be picklable and standalone programs require a main guard. CPU workers
limit native numerical thread pools; one CPU still uses a spawned process.

## Typed numerical workers

`ProcessorBackend[RequestT, OutputT]` is a structural protocol. Operation protocols
`LocalisationBackend`, `TrackingBackend` and `SegmentationBackend` bind the types.
`BackendRegistry.register` overloads check operation/backend pairing. Registration
does not import optional engines; configure registries before concurrent work.

```python
from molcore.backends import BackendRegistry, TrackingRequest, TrackingOutput
from molcore.execution import ProgressReporter

class SingleTrackBackend:
    version = "single-track-1"

    def compute(self, payload: TrackingRequest, reporter: ProgressReporter) -> TrackingOutput:
        reporter.check_cancelled()
        members = payload.rows["id"][payload.labels > 0]
        return TrackingOutput((members,) if len(members) >= payload.settings.min_length else ())

registry = BackendRegistry()
registry.register("track", "single", SingleTrackBackend())
# track(snapshot, selection, backend="single", registry=registry, context=context)
```

This example illustrates the interface, not a scientific linking algorithm.
Instances and bound methods must be picklable for process execution.

| Operation | Request | Output |
| --- | --- | --- |
| Localisation | `LocalisationRequest(images, frames, settings, camera, candidates, fitting)` | `LocalisationOutput(columns, failed_fit_count)` |
| Tracking | `TrackingRequest(rows, labels, settings)` | `TrackingOutput(memberships)` using stable localisation IDs |
| Segmentation | `SegmentationWorkerRequest(request, frame, segmentation_id, image, source_path, settings, model_ref, model_stat)` | `Segmentation` preserving the request binding and source pixels |

Workers receive immutable arrays. MolCore validates named output types, column
lengths/frame bindings, membership ownership and segmentation shape/binding before
returning scientific results. It owns selection, provenance, freshness and commit.
Custom workers remain responsible for numerical validity. Localisation detection
columns are `frame`, `x`, `y`, `net_gradient`; fitted outputs additionally include
`photons`, `bg`, `sx`, `sy`, `lpx`, `lpy`, `ellipticity`, `fit_state`, `iterations`.

## Bulk image access

`molcore.readers.open_reader(source)` opens one worker-local reader for a context.
`reader.read_frames(indices, roi=(y_slice, x_slice), out=buffer)` preserves order and
duplicates and supports empty selections. HDF5 uses hyperslabs and direct output
when possible; TIFF reuses an open handle. Channel readers push logical frame and
ROI mapping down to the source. Custom sources retain a frame-by-frame fallback.
Close readers with their context manager; do not share an open reader across
threads or processes.
