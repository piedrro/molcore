# Migrating to MolCore 0.2

This release deliberately breaks Python and saved-file compatibility. Update
moltrack/molseeq adapters before using it with those applications.

1. Compute from `dataset.snapshot()` (or `experiment.snapshot()` for experiment
   estimation). Use `PartitionSelection(acquisition_id, channel)` and
   `SequenceSelection(sequence_id, channel)` where accepted by the processor.
2. Remove `apply`, `cpu`, `num_cpus`, `progress` and `cancel_event` processor
   keywords. Pass execution options through `context=ExecutionContext(...)`.
   Move localisation chunk size from scientific settings into that context.
3. Apply detached results explicitly with `dataset.apply(result)` or
   `dataset.apply_all(results)`. Use the returned `ApplyReceipt` for allocated IDs
   and calibration names. Operation-specific apply helpers are private.
4. Replace direct metadata/record/source mutations with public model mutators.
   Query arrays and scientific records are immutable. Use `batch.to_columns()`
   when writable arrays are required. `ArraySource` copies by default; opt into
   `ArraySource.borrow` only when the external owner guarantees stable pixels.
5. Update custom workers to the typed request/output records in
   [backends](backends.md). Raw result tuples are no longer accepted.
6. Handle nullable measurement columns using `batch.validity` and native dtypes;
   do not assume missing integers were converted to floating NaN. Schemas/units
   round-trip through storage and bulk insertion.

MolCore writes and reads **format 7 only**, for both datasets and experiments.
Older files are rejected. Convert valuable files with an environment running the
older release before upgrading, exporting their scientific data for explicit
re-import; there is no automatic old-format reader in 0.2. Preserve original files.
Runtime dependency tokens and named target revisions are not portable identifiers.

See [integration contracts](api-contracts.md) and the runnable
[processing](../examples/processing.py), [scientific](../examples/scientific.py)
and [sequencing](../examples/sequencing.py) examples.
