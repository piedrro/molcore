"""Internal model snapshots, backend boundaries and executor ownership."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

from .cpu import CpuExecutor
from .types._shared import StaleResultError


def backend_version():
    try:
        installed = version("picassosr")
    except PackageNotFoundError as exc:
        raise ImportError("Picasso is missing; reinstall MolCore with its runtime dependencies") from exc
    if tuple(int(part) for part in installed.split(".")[:2]) != (0, 11):
        raise ImportError(f"MolCore processing supports Picasso >=0.11,<0.12; found {installed}")
    return installed


def snapshot(experiment, acquisition_id):
    acquisition = experiment.acquisitions[acquisition_id]
    return (
        experiment.localizations.revision,
        experiment.tracks.revision,
        experiment.transforms.revision,
        id(acquisition.source),
        tuple((name, id(view)) for name, view in acquisition.channels.items()),
    )


def check_snapshot(experiment, acquisition_id, expected):
    if snapshot(experiment, acquisition_id) != expected:
        raise StaleResultError("Processing inputs changed during computation; no result was applied")


@contextmanager
def executor(cpu, num_cpus):
    if cpu is not None:
        if num_cpus is not None:
            raise ValueError("Set num_cpus on the supplied CpuExecutor, or omit cpu")
        yield cpu
    else:
        with CpuExecutor(num_cpus=num_cpus) as created:
            yield created


def record(experiment, stage, acquisition_id, channel, values):
    experiment.processing.settings.setdefault(stage, {}).setdefault(acquisition_id, {})[channel] = values
    if "backend" in values:
        experiment.processing.software["picassosr"] = values["backend"]
