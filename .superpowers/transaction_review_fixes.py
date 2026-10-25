from pathlib import Path
root = Path('src/molcore')
p = root / 'types/experiment.py'
s = p.read_text().replace('from typing import Any', 'from typing import Any\nfrom threading import RLock\n\nfrom .._locking import model_locked')
s = s.replace('        self.metadata = dict(metadata or {})', '        self._lock = RLock()\n        self.metadata = dict(metadata or {})', 1)
for name in ('add_acquisition', 'add_sequence', 'add_segmentation'):
    s = s.replace(f'    def {name}(', f'    @model_locked\n    def {name}(', 1)
# Removal uses transact; do not hold live lock across staging/operation, to allow concurrent edits to be rejected.
s = s.replace('                else:\n                    values.pop(id, None)', '                elif stage == "cellpose":\n                    for mask_id in masks:\n                        values.pop(mask_id, None)\n                else:\n                    values.pop(id, None)', 1)
p.write_text(s)
p = root / 'types/tables.py'
s = p.read_text().replace('from ._shared import CoordinateSpace', 'from .._locking import model_locked\nfrom ._shared import CoordinateSpace')
for name in ('update_measurements', 'move_many', 'add', 'move', 'delete', 'replace', 'split', 'merge', 'reassign'):
    s = s.replace(f'    def {name}(', f'    @model_locked\n    def {name}(')
p.write_text(s)
p = root / 'transforms.py'
s = p.read_text().replace('from .types._shared import', 'from ._locking import model_locked\nfrom .types._shared import', 1)
for name in ('add_calibration', 'select', 'set_drift'):
    s = s.replace(f'    def {name}(', f'    @model_locked\n    def {name}(', 1)
p.write_text(s)
p = root / 'staging.py'
s = p.read_text().replace('from ._processing import snapshot', 'from collections.abc import Callable\nfrom typing import TypeVar\n\nfrom ._locking import model_locked')
s = s.replace('def state_token(experiment):', '@model_locked\ndef state_token(experiment: Experiment) -> tuple:')
s = s.replace('        tuple((a, snapshot(experiment, a)) for a in experiment.acquisitions),', '''        tuple((a, experiment.localizations.revision, experiment.tracks.revision,
               experiment.transforms.revision, id(acquisition.source),
               tuple((name, view.roi, view.shape, view.source_frames.tobytes())
                     for name, view in acquisition.channels.items()))
              for a, acquisition in experiment.acquisitions.items()),''')
s = s.replace('        tuple((k, id(s), s.labels.tobytes(), _json(s.metadata)) for k, s in experiment.segmentations.items()),', '''        tuple((k, s.acquisition_id, s.channel, s.frame, s.space, s.labels.dtype.str,
               s.labels.shape, s.labels.tobytes(), _json(s.metadata), s.source_path, s.mask_path,
               None if s.source_image is None else (s.source_image.dtype.str, s.source_image.shape,
                                                    s.source_image.tobytes()))
              for k, s in experiment.segmentations.items()),''')
s = s.replace('    result.calibrations = result.transforms.calibrations\n    return result', '''    result.calibrations = result.transforms.calibrations
    if hasattr(experiment, "_batch_source_token"):
        result._batch_source_token = experiment._batch_source_token
    return result''', 1)
start = s.index('def transact(')
s = s[:start] + '''R = TypeVar("R")


def _scientific_inputs(token: tuple) -> tuple:
    # Observations and provenance produced by independent processors are outputs.
    return token[0], token[2:5], token[6:]


def transact(
    experiment: Experiment, operation: Callable[[Experiment], R], *, expected: tuple | None = None,
) -> R:
    """Stage and atomically adopt a model update, rechecking freshness at commit.

    Model mutators and final adoption share a reentrant lock. Computation occurs
    outside the live lock. Concurrent supported edits are retained and cause the
    staged operation to fail. Direct dictionary/array mutation is not synchronized.
    Batch results may share the batch's original token while scientific inputs
    remain unchanged; independent observations/provenance may accumulate.
    """
    with experiment._lock:
        before = state_token(experiment)
        if expected is not None and before != expected:
            batch_source = getattr(experiment, "_batch_source_token", None)
            if batch_source != expected or _scientific_inputs(before) != _scientific_inputs(expected):
                raise StaleResultError("Inputs changed; compute a new preview before applying")
        staged = clone_experiment(experiment)
        staged._transaction_source_token = before
    value = operation(staged)
    with experiment._lock:
        if state_token(experiment) != before:
            raise StaleResultError("Live state changed during transaction; staged update was discarded")
        for name in ("localizations", "tracks", "observations", "transforms", "crops"):
            item = getattr(staged, name)
            item.experiment = experiment
            setattr(experiment, name, item)
        experiment.calibrations = experiment.transforms.calibrations
        experiment.segmentations = staged.segmentations
        experiment.processing = staged.processing
        experiment.metadata = staged.metadata
        experiment.acquisitions = staged.acquisitions
        experiment.sequences = staged.sequences
    return value
'''
p.write_text(s)
p = root / 'processing.py'
s = p.read_text().replace('''    return transact(
        experiment, lambda model: [operation(model) for operation in operations], expected=expected
    )''', '''    def commit(model: Experiment) -> list[Any]:
        model._batch_source_token = model._transaction_source_token
        return [operation(model) for operation in operations]
    return transact(experiment, commit, expected=expected)''', 1)
p.write_text(s)
