"""Cellpose 3 inference using MolCore acquisitions, CPU tools and segmentations."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._logging import logged
from ._processing import check_snapshot, executor, snapshot
from .backends import BackendRegistry, ProcessorBackend, resolve_backend
from .cpu import CpuExecutor
from .execution import ExecutionContext
from .storage import _json
from .types import (
    CellposeSettings,
    CoordinateSpace,
    Dataset,
    ProgressUpdate,
    Segmentation,
    SegmentationRequest,
    StaleResultError,
)
from .types.images import CachedSource

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .cpu import WorkerProgress
    from .types import CancelEvent

_BUILTIN_MODELS = frozenset({"cyto3", "cyto2", "cyto", "nuclei"})
_MODEL_CACHE = {}


def _backend_version() -> str:
    """Backend version.

    Returns:
        Computed str value.
    """
    try:
        installed = version("cellpose")
    except PackageNotFoundError as exc:
        raise ImportError("Cellpose is missing; install molcore[segmentation]") from exc
    major = int(installed.split(".")[0])
    if major != 3 or int(installed.split(".")[1]) != 0:
        raise ImportError(
            f"MolCore segmentation requires Cellpose >=3.0,<3.1; found {installed}"
        )
    return installed


def _resolve_model(settings: CellposeSettings) -> tuple:
    """Resolve model.

    Args:
        settings: Validated settings controlling this operation.

    Returns:
        Computed tuple value.
    """
    candidate = Path(settings.pretrained_model).expanduser()
    if candidate.is_file():
        candidate = candidate.resolve()
        stat = candidate.stat()
        return str(candidate), (stat.st_size, stat.st_mtime_ns)
    if settings.pretrained_model not in _BUILTIN_MODELS:
        raise ValueError(
            "Choose a Cellpose 3 cyto3/cyto2/cyto/nuclei model or an existing local weight file"
        )
    return settings.pretrained_model, None


def _fingerprint(image: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
    """Fingerprint.

    Args:
        image: Two-dimensional source image.

    Returns:
        Computed str value.
    """
    image = np.ascontiguousarray(image)
    return (
        image.shape,
        image.dtype.str,
        hashlib.blake2b(image.view(np.uint8), digest_size=16).digest(),
    )


def _target_token(segmentation: Segmentation) -> tuple | None:
    """Target token.

    Args:
        segmentation: Segmentation used by this operation.

    Returns:
        Computed tuple value.
    """
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
        (
            None
            if segmentation.source_image is None
            else _fingerprint(segmentation.source_image)
        ),
        _json(segmentation.metadata),
    )


def _source(dataset: Dataset, request: SegmentationRequest, frame: int | None) -> Any:
    """Source.

    Args:
        dataset: Scientific model supplying acquisitions and result tables.
        request: Source-bound segmentation request, including channel and original camera frame.
        frame: Original camera-frame index for the requested source image.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    view = dataset.acquisitions[request.acquisition_id].channels[request.channel]
    if request.space == CoordinateSpace.RAW:
        image = view[view.channel_index(frame)]
    else:
        image = dataset.transforms.image(
            request.acquisition_id, request.channel, frame, space=request.space
        )
    image = np.array(image, copy=True)
    if (
        image.ndim != 2
        or image.dtype.kind not in "biuf"
        or not np.isfinite(image).all()
    ):
        raise ValueError("Cellpose source must be a finite, real-valued 2D image")
    return image


def _load_model(model_ref: Any, model_stat: Any, settings: CellposeSettings) -> Any:
    """Load model.

    Args:
        model_ref: Model ref used by this operation.
        model_stat: Model stat used by this operation.
        settings: Validated settings controlling this operation.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
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
    key = model_ref, model_stat, str(device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()  # Bound network memory to one model per worker.
        options = (
            {"model_type": model_ref}
            if model_stat is None
            else {"pretrained_model": model_ref}
        )
        try:
            _MODEL_CACHE[key] = models.CellposeModel(
                gpu=settings.gpu, device=device, **options
            )
        except Exception as exc:
            if model_stat is not None:
                raise ValueError(
                    f"Cannot read Cellpose 3 checkpoint: {model_ref}"
                ) from exc
            raise
    return _MODEL_CACHE[key]


class _Progress:
    def __init__(self, reporter: WorkerProgress) -> None:
        """Initialise the object and validate its configuration.

        Args:
            reporter: Worker progress and cooperative cancellation reporter.
        """
        self.reporter = reporter

    @logged
    def setValue(self, value: Any) -> None:
        """Setvalue.

        Args:
            value: Value supplied by the caller or connected control.
        """
        self.reporter.report(
            0.2 + 0.7 * min(1.0, max(0.0, value / 100)), 1, "Cellpose inference"
        )


@logged
def _job(payload: tuple, reporter: WorkerProgress) -> Any:
    """Job.

    Args:
        payload: Serializable worker inputs; large arrays remain bounded to a job or image chunk.
        reporter: Worker progress and cooperative cancellation reporter.

    Returns:
        Result retaining the input format or backend-specific representation.
    """
    (
        request,
        frame,
        segmentation_id,
        image,
        source_path,
        settings,
        model_ref,
        model_stat,
    ) = payload
    reporter.report(0.05, 1, "Loading Cellpose model")
    model = _load_model(model_ref, model_stat, settings)
    reporter.report(0.2, 1, "Cellpose inference")
    # Preserve the exact input pixels even when the backend normalizes in place.
    source = np.array(image, copy=True)
    source.flags.writeable = False
    inference = np.max(source) - source if settings.invert else source
    output = model.eval(
        np.array(inference, dtype=np.float32, copy=True),
        batch_size=settings.batch_size,
        channels=None,
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
    if (
        labels.shape != source.shape
        or labels.dtype.kind not in "biu"
        or np.any(labels < 0)
    ):
        raise ValueError(
            "Cellpose must return nonnegative integer labels matching the source image"
        )
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


@logged
def segment_many(
    dataset: Dataset,
    requests: Sequence[SegmentationRequest],
    *,
    settings: CellposeSettings | None = None,
    apply: bool = True,
    num_cpus: int | None = None,
    cpu: CpuExecutor | None = None,
    progress: Callable[[ProgressUpdate], None] | None = None,
    cancel_event: CancelEvent | None = None,
    context: ExecutionContext | None = None,
    backend: ProcessorBackend | str | None = None,
    registry: BackendRegistry | None = None,
) -> dict[str, Segmentation]:
    """Segment model-bound source frames and stage every result before application.

    Args:
        dataset: Dataset supplying images and receiving segmentations.
        requests: Channel/camera-frame bindings and optional result identifiers.
        settings: Cellpose 3 parameters; default model is cyto3 on CPU.
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
        ImportError: The requested Cellpose backend is unavailable.

    Notes:
        Built-in weights download through Cellpose on first use. Local weight
        paths support offline operation. GPU inference requires one worker.
        Native inference kernels finish before cooperative cancellation is checked.
    """
    options = (
        (context or ExecutionContext())
        .resolve(
            cpu=cpu, num_cpus=num_cpus, progress=progress, cancel_event=cancel_event
        )
        .kwargs()
    )
    cpu, num_cpus, progress, cancel_event = (
        options[k] for k in ("cpu", "num_cpus", "progress", "cancel_event")
    )
    selected_backend = resolve_backend("segment", backend, registry=registry)
    if selected_backend is None:
        _backend_version()
    settings = settings or CellposeSettings()
    model_ref, model_stat = (
        _resolve_model(settings) if selected_backend is None else (None, None)
    )
    configured = (
        getattr(cpu, "num_cpus", 1)
        if cpu is not None
        else (num_cpus if num_cpus is not None else 1)
    )
    if settings.gpu and configured != 1:
        raise ValueError("GPU Cellpose inference requires exactly one worker")
    requests = tuple(requests)
    prepared, seen = [], set()
    for request in requests:
        if not isinstance(request, SegmentationRequest):
            raise TypeError("Use SegmentationRequest objects for batch segmentation")
        acquisition = dataset.acquisitions[request.acquisition_id]
        view = acquisition.channels[request.channel]
        if not len(view):
            raise ValueError("Cannot segment an empty channel")
        frame = int(view.source_frames[0]) if request.frame is None else request.frame
        view.channel_index(frame)
        identifier = (
            request.segmentation_id
            or f"{request.acquisition_id}:{request.channel}:cellpose:{frame}"
        )
        if identifier in seen:
            raise ValueError("Batch segmentation IDs must be unique")
        seen.add(identifier)
        source = acquisition.source
        while isinstance(source, CachedSource):
            source = source.source
        source_path = getattr(source, "path", None) or acquisition.metadata.get(
            "source_path"
        )
        prepared.append(
            (
                request,
                frame,
                identifier,
                source_path,
                snapshot(dataset, request.acquisition_id),
                _fingerprint(_source(dataset, request, frame)),
                _target_token(dataset.segmentations.get(identifier)),
            )
        )

    def jobs() -> Iterator[Any]:
        """Jobs.

        Yields:
            Values produced by the iterator, retaining their original order.
        """
        for request, frame, identifier, source_path, _, fingerprint, _ in prepared:
            image = _source(dataset, request, frame)
            if _fingerprint(image) != fingerprint:
                raise StaleResultError(
                    "Segmentation source pixels changed before inference"
                )
            yield request, frame, identifier, image, source_path, settings, model_ref, model_stat

    with executor(cpu, (configured if cpu is None else num_cpus)) as runner:
        values = runner.map(
            _job if selected_backend is None else selected_backend.compute,
            jobs(),
            total=len(prepared),
            progress=progress,
            cancel_event=cancel_event,
        )
    if selected_backend is None and _resolve_model(settings) != (model_ref, model_stat):
        raise StaleResultError("Cellpose weights changed during inference")
    for request, frame, identifier, _, expected, fingerprint, target in prepared:
        check_snapshot(dataset, request.acquisition_id, expected)
        if _fingerprint(_source(dataset, request, frame)) != fingerprint:
            raise StaleResultError(
                "Segmentation source pixels changed during inference"
            )
        if apply and _target_token(dataset.segmentations.get(identifier)) != target:
            raise StaleResultError("Target segmentation changed during inference")
    for value, entry in zip(values, prepared, strict=True):
        value.expected = entry[4]
        if selected_backend is not None:
            value.metadata.update(
                {"algorithm": "external", "backend": selected_backend.version}
            )
    results = {value.id: value for value in values}
    if apply and results:
        from .processing import apply_segmentations

        return apply_segmentations(dataset, results)
    return results


@logged
def segment(
    dataset: Dataset,
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
    cancel_event: CancelEvent | None = None,
    context: ExecutionContext | None = None,
    backend: ProcessorBackend | str | None = None,
    registry: BackendRegistry | None = None,
) -> Segmentation:
    """Segment one MolCore channel frame with the Cellpose 3 backend.

    Args:
        dataset: Dataset containing the source acquisition.
        acquisition_id: Source acquisition identifier.
        channel: Logical channel to segment.
        frame: Actual camera frame; None selects the channel's first frame.
        segmentation_id: Optional stable name for storing/replacing the result.
        space: Raw, registered or corrected source/label coordinate space.
        settings: Cellpose model, thresholds, device and normalization settings.
        apply: Store the result in Dataset.segmentations when successful.
        num_cpus: CPU worker count; defaults to one.
        cpu: Optional reusable CpuExecutor to amortize model loading.
        progress: Parent-process ProgressUpdate callback.
        cancel_event: Optional caller cancellation event.

    Returns:
        Existing MolCore Segmentation type with source image and provenance.
    """
    request = SegmentationRequest(
        acquisition_id, channel, frame, segmentation_id, space
    )
    results = segment_many(
        dataset,
        [request],
        settings=settings,
        apply=apply,
        num_cpus=num_cpus,
        cpu=cpu,
        progress=progress,
        cancel_event=cancel_event,
        context=context,
        backend=backend,
        registry=registry,
    )
    return next(iter(results.values()))
