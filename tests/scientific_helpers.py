"""Explicit snapshot/compute/apply harness for numerical regression fixtures.

Older numerical fixtures describe both computation and publication in one call.
This harness runs those scenarios through the new separate public boundaries;
new API contract tests exercise those boundaries directly.
"""

from dataclasses import fields, is_dataclass, replace

from molcore import DatasetSnapshot, ExecutionContext, PartitionSelection, SequenceSelection

_APPLY = {
    "fit",
    "localise",
    "track",
    "track_sequence",
    "measure_localizations",
    "stationary_traces",
    "stationary_site_traces",
    "bounding_box_traces",
    "track_traces",
    "align_channels",
    "undrift",
    "undrift_many",
    "segment_many",
    "segment",
}
_PARTITION = {
    "detect",
    "localise",
    "track",
    "stationary_site_traces",
    "bounding_box_traces",
    "estimate_drift",
    "undrift",
    "segment",
    "filter_results",
    "cluster_localizations",
    "colocalise",
    "render_localizations",
}


def compute_scenario(function, dataset_model, *args, **kwargs):
    """Run real computation, then explicitly publish when the fixture requests it."""
    name = function.__name__
    if function.__module__ not in {
        "molcore.localisation",
        "molcore.tracking",
        "molcore.photometry",
        "molcore.alignment",
        "molcore.undrifting",
        "molcore.cellpose",
        "molcore.filtering",
        "molcore.clustering",
        "molcore.colocalisation",
        "molcore.rendering",
        "molcore.contrast",
    }:
        return function(dataset_model, *args, **kwargs)
    publish = kwargs.pop("apply", name in _APPLY)
    execution = {
        key: kwargs.pop(key) for key in ("cpu", "num_cpus", "progress", "cancel_event") if key in kwargs
    }
    context = kwargs.pop("context", None)
    if context is not None:
        context = context.resolve(**execution)
    else:
        context = ExecutionContext(**execution)
    args = list(args)
    if name in _PARTITION or name == "track_sequence":
        first = "sequence_id" if name == "track_sequence" else "acquisition_id"
        cls = SequenceSelection if name == "track_sequence" else PartitionSelection
        if args and isinstance(args[0], (PartitionSelection, SequenceSelection)):
            pass
        elif len(args) >= 2:
            args[:2] = [cls(*args[:2])]
        else:
            args.insert(0, cls(kwargs.pop(first), kwargs.pop("channel")))
    snapshot = dataset_model if isinstance(dataset_model, DatasetSnapshot) else dataset_model.snapshot()
    result = function(snapshot, *args, context=context, **kwargs)
    values = list(result.values()) if isinstance(result, dict) else [result]
    for value in values:
        token = getattr(value, "expected", None)
        if token is not None and hasattr(token, "validate"):
            token.validate(dataset_model)
    if not publish:
        return result
    receipts = dataset_model.apply_all(values)

    def applied(value, receipt):
        if not is_dataclass(value):
            return value
        names = {field.name for field in fields(value)}
        changes = {key: item for key, item in receipt.identifiers.items() if key in names}
        return replace(value, **changes) if changes else value

    output = [applied(value, receipt) for value, receipt in zip(values, receipts, strict=True)]
    return dict(zip(result, output, strict=True)) if isinstance(result, dict) else output[0]
