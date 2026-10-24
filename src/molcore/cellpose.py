"""Cellpose 4 inference using MolCore acquisitions, CPU tools and segmentations."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ._processing import check_snapshot, executor, snapshot
from .cpu import CpuExecutor
from .storage import _json
from .types import (
    CellposeSettings,
    CoordinateSpace,
    Experiment,
    ProgressUpdate,
    Segmentation,
    SegmentationRequest,
    StaleResultError,
)
from .types.images import CachedSource

_BUILTIN_MODELS = frozenset({"cpsam_v2", "cpsam", "cpdino", "cpdino-vitb"})
_MODEL_CACHE = {}


def _backend_version():
    try:
        installed = version("cellpose")
    except PackageNotFoundError as exc:
        raise ImportError("Cellpose is missing; reinstall MolCore with its runtime dependencies") from exc
    major, minor = (int(part) for part in installed.split(".")[:2])
    if major != 4 or minor < 2:
        raise ImportError(f"MolCore segmentation requires Cellpose >=4.2,<5; found {installed}")
    return installed


def _resolve_model(settings):
    candidate = Path(settings.pretrained_model).expanduser()
    if candidate.is_file():
        candidate = candidate.resolve()
        stat = candidate.stat()
        return str(candidate), (stat.st_size, stat.st_mtime_ns)
    if settings.pretrained_model not in _BUILTIN_MODELS:
        raise ValueError("Choose a Cellpose 4 SAM/DINO model name or an existing local weight file")
    return settings.pretrained_model, None


def _fingerprint(image):
    image = np.ascontiguousarray(image)
    return image.shape, image.dtype.str, hashlib.blake2b(image.view(np.uint8), digest_size=16).digest()


def _target_token(segmentation):
    if segmentation is None:
        return None
    return (
        id(segmentation),
        segmentation.acquisition_id,
        segmentation.channel,
        segmentation.frame,
        segmentation.space,
        segmentation.source_path,
        segmentation.mask_path,
        _fingerprint(segmentation.labels),
        None if segmentation.source_image is None else _fingerprint(segmentation.source_image),
        _json(segmentation.metadata),
    )


def _source(experiment, request, frame):
    view = experiment.acquisitions[request.acquisition_id].channels[request.channel]
    if request.space == CoordinateSpace.RAW:
        image = view[view.channel_index(frame)]
    else:
        image = experiment.transforms.image(
            request.acquisition_id, request.channel, frame, space=request.space
        )
    image = np.array(image, copy=True)
    if image.ndim != 2 or image.dtype.kind not in "biuf" or not np.isfinite(image).all():
        raise ValueError("Cellpose source must be a finite, real-valued 2D image")
    return image


def _load_model(model_ref, model_stat, settings):
    import torch
    from cellpose import models

    torch.set_num_threads(1)
    if settings.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            raise RuntimeError("GPU requested, but CUDA and MPS are unavailable")
    else:
        device = torch.device("cpu")
    key = model_ref, model_stat, str(device), settings.use_bfloat16
    if key not in _MODEL_CACHE:
        is_dino = model_ref in {"cpdino", "cpdino-vitb"}
        if model_stat is not None:
            try:
                is_dino = models.get_backbone(model_ref).startswith("dino")
            except Exception as exc:
                raise ValueError(f"Cannot read Cellpose 4 checkpoint: {model_ref}") from exc
        if is_dino and importlib.util.find_spec("dinov3") is None:
            raise ImportError(
                "DINO models require the upstream dinov3 package: pip install git+https://github.com/facebookresearch/dinov3"
            )
        _MODEL_CACHE.clear()  # Bound network memory to one model per worker.
        _MODEL_CACHE[key] = models.CellposeModel(
            gpu=settings.gpu, device=device, pretrained_model=model_ref, use_bfloat16=settings.use_bfloat16
        )
    return _MODEL_CACHE[key]


class _Progress:
    def __init__(self, reporter):
        self.reporter = reporter

    def setValue(self, value):
        self.reporter.report(0.2 + 0.7 * min(1.0, max(0.0, value / 100)), 1, "Cellpose inference")


def _job(payload, reporter):
    request, frame, segmentation_id, image, source_path, settings, model_ref, model_stat = payload
    reporter.report(0.05, 1, "Loading Cellpose model")
    model = _load_model(model_ref, model_stat, settings)
    reporter.report(0.2, 1, "Cellpose inference")
    # Preserve the exact input pixels even when the backend normalizes in place.
    source = np.array(image, copy=True)
    source.flags.writeable = False
    output = model.eval(
        np.array(source, dtype=np.float32, copy=True),
        batch_size=settings.batch_size,
        channel_axis=None,
        normalize=settings.normalize,
        diameter=settings.diameter,
        flow_threshold=settings.flow_threshold,
        cellprob_threshold=settings.cellprob_threshold,
        min_size=settings.min_size,
        do_3D=False,
        progress=_Progress(reporter),
    )
    reporter.report(0.95, 1, "Validating Cellpose labels")
    if not isinstance(output, (tuple, list)) or not output:
        raise ValueError("Cellpose returned an invalid inference result")
    labels = np.asarray(output[0])
    if labels.shape != source.shape or labels.dtype.kind not in "biu" or np.any(labels < 0):
        raise ValueError("Cellpose must return nonnegative integer labels matching the source image")
    import torch

    result = Segmentation(
        segmentation_id,
        request.acquisition_id,
        labels,
        source,
        channel=request.channel,
        space=request.space,
        frame=frame,
        source_path=source_path,
        metadata={
            "algorithm": "cellpose",
            "cellpose_version": _backend_version(),
            "torch_version": str(torch.__version__),
            "settings": asdict(settings),
            "device": str(model.device),
            "model_path": str(model.pretrained_model),
        },
    )
    reporter.report(1, 1, "Cellpose segmentation complete")
    return result


def segment_many(
    experiment: Experiment,
    requests: Sequence[SegmentationRequest],
    *,
    settings: CellposeSettings | None = None,
    apply: bool = True,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: Callable[[ProgressUpdate], None] | None = None,
    cancel_event=None,
) -> dict[str, Segmentation]:
    """Segment model-bound source frames and stage every result before application.

    Args:
        experiment: Experiment supplying images and receiving segmentations.
        requests: Channel/camera-frame bindings and optional result identifiers.
        settings: Cellpose 4 parameters; default model is cpsam_v2 on CPU.
        apply: Store successful results; False returns detached previews.
        num_cpus: Worker count, defaulting to one to bound network memory.
        cpu: Optional reusable executor; configure its CPU count on construction.
        progress: Parent-process callback receiving progress percentages and stages.
        cancel_event: Caller event for cooperative cancellation at stage boundaries.

    Returns:
        Segmentation objects keyed by identifier, preserving exact source pixels.

    Raises:
        StaleResultError: Inputs or target segmentations changed during inference.
        ValueError: Settings, model selection, masks or result bindings are invalid.
        ImportError: The requested Cellpose/DINO backend is unavailable.

    Notes:
        Built-in weights download through Cellpose on first use. Local weight
        paths support offline operation. GPU inference requires one worker.
        Native inference kernels finish before cooperative cancellation is checked.
    """
    _backend_version()
    settings = settings or CellposeSettings()
    model_ref, model_stat = _resolve_model(settings)
    configured = cpu.num_cpus if cpu is not None else (num_cpus if num_cpus is not None else 1)
    if settings.gpu and configured != 1:
        raise ValueError("GPU Cellpose inference requires exactly one worker")
    requests = tuple(requests)
    prepared, seen = [], set()
    for request in requests:
        if not isinstance(request, SegmentationRequest):
            raise TypeError("Use SegmentationRequest objects for batch segmentation")
        acquisition = experiment.acquisitions[request.acquisition_id]
        view = acquisition.channels[request.channel]
        if not len(view):
            raise ValueError("Cannot segment an empty channel")
        frame = int(view.source_frames[0]) if request.frame is None else request.frame
        view.channel_index(frame)
        identifier = request.segmentation_id or f"{request.acquisition_id}:{request.channel}:cellpose:{frame}"
        if identifier in seen:
            raise ValueError("Batch segmentation IDs must be unique")
        seen.add(identifier)
        source = acquisition.source
        while isinstance(source, CachedSource):
            source = source.source
        source_path = getattr(source, "path", None) or acquisition.metadata.get("source_path")
        prepared.append(
            (
                request,
                frame,
                identifier,
                source_path,
                snapshot(experiment, request.acquisition_id),
                _fingerprint(_source(experiment, request, frame)),
                _target_token(experiment.segmentations.get(identifier)),
            )
        )

    def jobs():
        for request, frame, identifier, source_path, _, fingerprint, _ in prepared:
            image = _source(experiment, request, frame)
            if _fingerprint(image) != fingerprint:
                raise StaleResultError("Segmentation source pixels changed before inference")
            yield request, frame, identifier, image, source_path, settings, model_ref, model_stat

    with executor(cpu, (configured if cpu is None else num_cpus)) as runner:
        values = runner.map(_job, jobs(), total=len(prepared), progress=progress, cancel_event=cancel_event)
    if _resolve_model(settings) != (model_ref, model_stat):
        raise StaleResultError("Cellpose weights changed during inference")
    for request, frame, identifier, _, expected, fingerprint, target in prepared:
        check_snapshot(experiment, request.acquisition_id, expected)
        if _fingerprint(_source(experiment, request, frame)) != fingerprint:
            raise StaleResultError("Segmentation source pixels changed during inference")
        if apply and _target_token(experiment.segmentations.get(identifier)) != target:
            raise StaleResultError("Target segmentation changed during inference")
    results = {value.id: value for value in values}
    if apply and results:
        previous = experiment.processing.settings.get("cellpose", {})
        if not isinstance(previous, dict):
            raise ValueError("Cellpose processing metadata must be a dictionary")
        metadata = {**previous, **{value.id: deepcopy(value.metadata) for value in values}}
        for value in values:
            experiment.add_segmentation(value)
        experiment.processing.settings["cellpose"] = metadata
        experiment.processing.software["cellpose"] = values[0].metadata["cellpose_version"]
        experiment.processing.software["torch"] = values[0].metadata["torch_version"]
    return results


def segment(
    experiment: Experiment,
    acquisition_id: str,
    channel: str,
    *,
    frame: int | None = None,
    segmentation_id: str | None = None,
    space: CoordinateSpace | str = CoordinateSpace.RAW,
    settings: CellposeSettings | None = None,
    apply: bool = True,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: Callable[[ProgressUpdate], None] | None = None,
    cancel_event=None,
) -> Segmentation:
    """Segment one MolCore channel frame with the Cellpose 4 backend.

    Args:
        experiment: Experiment containing the source acquisition.
        acquisition_id: Source acquisition identifier.
        channel: Logical channel to segment.
        frame: Actual camera frame; None selects the channel's first frame.
        segmentation_id: Optional stable name for storing/replacing the result.
        space: Raw, registered or corrected source/label coordinate space.
        settings: Cellpose model, thresholds, device and normalization settings.
        apply: Store the result in Experiment.segmentations when successful.
        num_cpus: CPU worker count; defaults to one.
        cpu: Optional reusable CpuExecutor to amortize model loading.
        progress: Parent-process ProgressUpdate callback.
        cancel_event: Optional caller cancellation event.

    Returns:
        Existing MolCore Segmentation type with source image and provenance.
    """
    request = SegmentationRequest(acquisition_id, channel, frame, segmentation_id, space)
    results = segment_many(
        experiment,
        [request],
        settings=settings,
        apply=apply,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
    )
    return next(iter(results.values()))
