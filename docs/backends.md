# Execution and optional processing backends

The core installation supports models, storage, transforms and analysis without
loading Picasso, Trackpy, Cellpose or scikit-learn. Install only the processing
dependencies needed by your application:

| Extra | Dependency | Operations |
| --- | --- | --- |
| `molcore[localisation]` | Picasso | Detection, fitting, localisation and Picasso-based alignment/drift |
| `molcore[tracking]` | Trackpy | Tracking |
| `molcore[segmentation]` | Cellpose | Segmentation |
| `molcore[clustering]` | scikit-learn | DBSCAN clustering |
| `molcore[ui]` | QtPy, PyQtGraph | Shared UI widgets (install a Qt binding separately) |
| `molcore[all]` | All the above | Full installation |

SciPy remains a core dependency: coordinate transforms, spatial matching and
colocalisation use it independently of the optional processing libraries.

## Reusing execution policy

```python
from molcore.cpu import CpuExecutor
from molcore.execution import ExecutionContext
from molcore.localisation import localise
from molcore.tracking import track

def process(dataset, acquisition_id, channel, camera, on_progress, cancel_event):
    with CpuExecutor(num_cpus=2) as cpu:
        execution = ExecutionContext(cpu=cpu, progress=on_progress, cancel_event=cancel_event)
        localise(dataset, acquisition_id, channel, camera=camera, **execution.kwargs())
        track(dataset, acquisition_id, channel, **execution.kwargs())
```

Guard standalone calls with `if __name__ == "__main__":` when using the spawned
CPU executor. Contexts borrow executors and never close them. `num_cpus` and a
borrowed executor are mutually exclusive. `resolve(...)` fills unspecified
context fields and rejects duplicate configuration, including matching values;
`None` means unspecified. `kwargs()` checks cancellation before dispatch and
returns a fresh typed dictionary. Processors and worker reporters continue to
check cancellation during execution.

An alternative executor implements `Executor.map(function, jobs, *, total=None,
progress=None, cancel_event=None) -> list`. It returns results in input order,
supplies a reporter supporting `report(completed, total, description)` and
`check_cancelled()`, and honours cancellation. Callbacks execute on the calling
thread. The application owns the executor's lifetime and concurrency policy.

## Injecting numerical backends

`ProcessorBackend` is a structural protocol: a backend exposes a string
`version` and `compute(payload, reporter)`. A backend can be passed directly to
a processor or registered by operation and name in a caller-owned
`BackendRegistry`. Named selection uses the supplied registry; if omitted, it
uses the module's `backends` registry. Registration never imports a processing
library. Configure registries before concurrent processing begins.

```python
from molcore.backends import BackendRegistry
from molcore.tracking import track

class SingleTrackBackend:
    version = "single-track-1"

    def compute(self, payload, reporter):
        rows, labels, settings = payload
        reporter.check_cancelled()
        members = rows["id"][labels > 0].copy()
        reporter.report(1, 1, "Linked selected localizations")
        return (members,) if len(members) >= settings.min_length else ()

registry = BackendRegistry()
registry.register("track", "single", SingleTrackBackend())
# track(dataset, acquisition_id, channel, backend="single", registry=registry)
```

This small example intentionally groups all selected localizations together;
it demonstrates the contract rather than a scientific tracking algorithm.
Instances and bound methods must be picklable when used with `CpuExecutor`.
Passing an instance directly avoids shared registry configuration. Duplicate
names require `replace=True`; names are scoped to their operation.

Backends replace the numerical worker only. MolCore still owns source selection,
settings validation, progress dispatch, detached results and transactional model
application. A worker must not mutate its input arrays or the dataset. Current
payload contracts are operation-specific:

| Operation | Payload tuple | Return value |
| --- | --- | --- |
| `localise` | `(images, frames, settings, camera, candidates, fitting)` | `(columns, failed_fit_count)`; columns are named NumPy arrays using original camera frames; candidates may be `None`; fitting distinguishes detection from fitted results |
| `track` | `(rows, labels, settings)` | Tuple of stable localization-ID arrays, one per trajectory; zero labels are excluded and positive labels delimit independent groups |
| `segment` | `(request, frame, segmentation_id, image, source_path, settings, model_ref, model_stat)` | A `Segmentation` carrying the supplied IDs, channel, frame and coordinate space, nonnegative integer labels matching the input image, and a preserved source image |

Localisation columns for detection are `frame`, `x`, `y`, `net_gradient`.
Fitted results additionally include `photons`, `bg`, `sx`, `sy`, `lpx`, `lpy`,
`ellipticity`, `fit_state` and `iterations`. Empty results must retain these
columns with zero-length arrays. These worker payloads are deliberately explicit
but not yet separate typed domain request objects; the protocol guarantees the
worker boundary, not the scientific validity of arbitrary custom implementations.
