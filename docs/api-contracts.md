# Application integration contracts

MolCore keeps image access lazy and scientific computation independent of Qt.
The common workflow is to capture a snapshot, compute a detached result, then
apply it to the live dataset. Existing `apply=True` processor calls remain available.

```python
from molcore import CameraSettings, ExecutionContext, LocalisationSettings
from molcore.localisation import localise

snapshot = dataset.snapshot()
result = localise(
    snapshot, "recording", "image",
    camera=CameraSettings(baseline=100, sensitivity=1, gain=1),
    settings=LocalisationSettings(),
    context=ExecutionContext(num_cpus=4),
    apply=False,
)
# Inspect result.columns or display it before committing.
applied = dataset.apply(result)
print(applied.localization_ids)
```

Camera values are examples. Use the actual camera calibration. Spawned executors
require the usual Python main guard. Snapshots copy tables and metadata but share
lazy image sources: callers must not mutate borrowed image arrays during work.

## Result application

`Dataset.apply(result)` accepts localisation, drift, channel alignment, tracking,
photometry, photon-sequence and segmentation results. Results carry their source
dependencies and the provenance required to apply them. Partial localisation
previews cannot replace a full partition. Successful application returns the
corresponding result with allocated IDs or application metadata.

`Dataset.apply_all(results)` commits all results or none. Inputs are revalidated
before each application: results whose inputs are changed by an earlier result
must be recomputed. Independent measurements can accumulate. `processing.apply_batch`
remains available for custom model-update callables; external side effects cannot
be rolled back. Extension result classes can register a handler with
`processing.register_result_handler(ResultClass, handler)`. A handler receives
`(result, staged_dataset)` and must restrict effects to that model.

## Freshness and events

`StateToken` names the complete model state used for transaction conflict checks.
`DependencyToken` tracks explicit acquisition/channel partitions and named input
components. Built-in localisation, tracking, photometry, drift and segmentation
use scoped dependencies. Edits to unrelated acquisitions do not invalidate these
results. A changed input or a different dataset lineage raises `StaleResultError`.
Tokens are runtime objects, not portable persisted revision identifiers.

```python
token = dataset.dependencies("recording", "image", components={"localizations"})
token.validate(dataset)

unsubscribe = dataset.subscribe(lambda change: print(change.components, change.partitions))
dataset.localizations.move(localization_id, x=12, y=20)
unsubscribe()
```

Listeners receive one `ChangeSet` per successful outer mutation or transaction.
They run on the committing thread, after adoption. Listener exceptions are logged
and do not undo an already committed change. Qt `UiSession` marshals notifications
to its GUI thread, updates dirty state, and unsubscribes when replaced or shut down.

Table/store objects and the acquisition/sequence/segmentation collection views
retain identity across commits. Collection mappings are read-only. Use public
mutators, `update_metadata`, `update_acquisition`, `add_segmentation`,
`remove_segmentation` and transform methods to change model state.
`update_acquisition` validates pixel scale, timing, metadata and affected sequences
before making changes; source/layout replacement requires removing and re-adding
the acquisition with explicit dependency handling.

Legacy mutable record attributes and metadata dictionaries still exist. Direct
mutation is unsupported for synchronization and notifications. Scoped revision
counters require the public mutators; full transaction tokens retain conservative
checks of metadata and masks for legacy callers. Do not modify source pixels behind
MolCore's back while processing.

## Typed queries

Localisation queries return `LocalizationBatch`, an immutable mapping of NumPy
columns. It includes dataset identity, coordinate space, original/effective position
choice, sequence frame context, selected calibration names and transform revision.
`batch.schema` reports actual dtypes and known units; unknown measurement units are
`None`. `len(batch)` counts columns; `batch.n_rows` counts localizations.

```python
batch = dataset.localizations.query(space="corrected")
print(batch.space, batch.schema["x"].unit, batch.n_rows)
mutable_columns = batch.to_columns()
records = batch.to_records()
```

Use the conversions when a downstream algorithm needs writable arrays. Required
insertion fields are explicit keyword arguments: `acquisition_id`, `channel`,
`frame`, `x`, `y`; arbitrary extra measurement columns remain supported.

See [backend and execution interfaces](backends.md) for custom numerical workers
and installation extras, and [storage options](storage-options.md) for independent
original-image/crop choices and custom image-source codecs.

## Migration and validation

- Replace mutations of query dictionaries with `query(...).to_columns()`.
- Keep captured result tokens; GUI adapters no longer need to replace them with live tokens.
- Use `dataset.apply(result)` instead of rebuilding commit-time camera/settings provenance.
- Install `molcore[all]` for the previous full scientific dependency set, or selected extras.
- `python -m mypy` checks the new backend, execution, batch, revision and storage interfaces
  together with `tests/typing_consumer.py`. Existing numerical modules are not yet
  covered by whole-package strict static checking.
