# Application integration contracts

Scientific processors consume read-only snapshots and return detached, immutable
results. Applying a result is a separate, atomic operation on the live dataset.

```python
from molcore import ExecutionContext, LocalisationSettings, PartitionSelection
from molcore.localisation import localise

result = localise(
    dataset.snapshot(), PartitionSelection("recording", "image"),
    camera=camera, settings=LocalisationSettings(),
    context=ExecutionContext(num_cpus=4, chunk_size=16),
)
receipt = dataset.apply(result)
print(receipt.result_id, receipt.revision, receipt.localization_ids)
```

Use the actual camera calibration. Standalone scripts using spawned processes
require a Python main guard. Partial localisation previews and results from a
working image source cannot replace raw-channel localisation tables.

## Ownership

`DatasetSnapshot` exposes queries and immutable records, without model mutators,
save, or apply methods. Capturing it copies model containers and shares immutable
scientific buffers; later live edits replace buffers rather than changing them.
`Experiment.snapshot()` captures datasets for cross-dataset estimation.

Metadata, settings, source descriptors, masks, calibrations and scientific result
arrays reject mutation. Use `update_metadata`, `rename`, `update_processing`,
`update_acquisition`, table edit methods and transform mutators. Metadata must be
JSON-compatible with string keys and finite numbers. NumPy values are accepted.

`ArraySource(array)` owns an immutable copy. `ArraySource.borrow(array)` opts into
a borrowed, read-only view: its owner must keep the underlying pixels stable from
snapshot capture through computation and application. File sources similarly
require stable external files. Snapshots do not close the live model's sources.

## Applying results and extending transactions

`Dataset.apply(result)` returns an `ApplyReceipt`, containing the stable result ID,
committed dataset revision and allocated identifiers. It does not modify the result.
`Dataset.apply_all(results)` commits all results in one transaction or none.
Each result is checked against the staged state, so dependent work must be computed
again after its inputs change. Independent observations and masks can accumulate.

Every built-in result declares its input dependencies and `write_scope` separately.
`StateToken` checks the whole runtime model, including revision changes that restore
the same values. `DependencyToken` tracks selected components and partitions.
Unrelated partition edits do not invalidate scoped results. Changed inputs or
dataset lineage raise `StaleResultError`; tokens are not portable saved revisions.
Named segmentation targets additionally detect intervening replacement/deletion.

For extensions, derive a result from `ResultIdentity` and register a handler with
`processing.register_result_handler(ResultClass, handler)`. The handler receives
`(result, staged_dataset)` and returns an identifier mapping for the receipt.
Keep effects inside the staged model. `processing.apply_batch` also accepts custom
model-update callables. External I/O cannot be rolled back. Nested transactions
join the same staging boundary and receive the outer committed revision.

Listeners receive one `ChangeSet` after a successful outer mutation. They run on
the committing thread; listener exceptions are logged after adoption. Store and
collection handles retain identity across commits. `UiSession` marshals events to
the GUI thread and applies worker results to the live dataset.

## Batches, schemas and nulls

Localisation, observation and track-membership queries return immutable mappings:
`LocalizationBatch`, `ObservationBatch` and `TrackMembershipBatch`. `len(batch)`
counts columns and `batch.n_rows` counts rows. Localisation context includes
coordinate space, frame/reference information and transform revision.

```python
batch = dataset.localizations.query(columns=("id", "frame", "x", "y"))
for part in dataset.localizations.iter_batches(batch_size=65536, columns=("id", "x")):
    consume(part)
mutable = batch.to_columns()
```

Projection retains the selected columns' schema and validity. `batch.validity`
contains masks for nullable columns; absent masks mean all values are valid.
Missing integer, boolean and string values retain native dtypes. Floating
scientific values retain NaN conventions; consult validity for explicit nulls.
Use `to_columns()` or `to_records()` for writable conversions. Set physical units
with `table.set_column_schema(name, unit=...)`. `table.add_batches(chunks)` combines
validated chunks once and inserts atomically, preserving units and validity.

See [backend interfaces](backends.md), [migration](api-migration.md),
[performance](performance.md) and [storage options](storage-options.md).
