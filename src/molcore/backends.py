"""Explicit, injectable worker backends with no optional-library imports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, overload

import numpy as np


class LocalisationRequest(NamedTuple):
    images: np.ndarray | Sequence[np.ndarray]
    frames: np.ndarray
    settings: LocalisationSettings
    camera: CameraSettings
    candidates: Mapping[str, np.ndarray] | None
    fitting: bool


class LocalisationOutput(NamedTuple):
    columns: Mapping[str, np.ndarray]
    failed_fit_count: int


class TrackingRequest(NamedTuple):
    rows: Mapping[str, np.ndarray]
    labels: np.ndarray
    settings: TrackingSettings


class TrackingOutput(NamedTuple):
    memberships: tuple[np.ndarray, ...]


class SegmentationWorkerRequest(NamedTuple):
    request: SegmentationRequest
    frame: int
    segmentation_id: str
    image: np.ndarray
    source_path: str | None
    settings: CellposeSettings
    model_ref: Any
    model_stat: Any


class LocalisationBackend(Protocol):
    version: str

    def compute(self, payload: LocalisationRequest, reporter: ProgressReporter) -> LocalisationOutput:
        """Compute.

        Args:
            payload: Payload.
            reporter: Reporter.


        Returns:
            The computed value.
        """
        ...


class TrackingBackend(Protocol):
    version: str

    def compute(self, payload: TrackingRequest, reporter: ProgressReporter) -> TrackingOutput:
        """Compute.

        Args:
            payload: Payload.
            reporter: Reporter.


        Returns:
            The computed value.
        """
        ...


class SegmentationBackend(Protocol):
    version: str

    def compute(self, payload: SegmentationWorkerRequest, reporter: ProgressReporter) -> Segmentation:
        """Compute.

        Args:
            payload: Payload.
            reporter: Reporter.


        Returns:
            The computed value.
        """
        ...


if TYPE_CHECKING:
    from .execution import ProgressReporter
    from .types import (
        CameraSettings,
        CellposeSettings,
        LocalisationSettings,
        Segmentation,
        SegmentationRequest,
        TrackingSettings,
    )
Operation = Literal["localise", "track", "segment"]


class ProcessorBackend[RequestT, OutputT](Protocol):
    """Numerical worker implementation; model validation and application stay in MolCore.

    Instances passed to a process executor must be picklable. Payload and return
    schemas are operation-specific and documented in docs/backends.md.
    """

    @property
    def version(self) -> str:
        """Backend identifier recorded with its results.

        Returns:
            Stable implementation version.
        """
        ...

    def compute(self, payload: RequestT, reporter: ProgressReporter) -> OutputT:
        """Execute one numerical work item.

        Args:
            payload: Operation-specific detached inputs.
            reporter: Progress and cancellation channel.

        Returns:
            Operation-specific detached output.
        """
        ...


def _validate(backend: ProcessorBackend) -> None:
    if not callable(getattr(backend, "compute", None)) or not isinstance(
        getattr(backend, "version", None), str
    ):
        raise TypeError("A backend must expose compute(payload, reporter) and a string version")


class BackendRegistry:
    """Caller-owned registry keyed by operation and name; no implicit discovery.

    Register implementations before starting concurrent work. Use a distinct
    registry per application/session when shared global configuration is unwanted.
    """

    def __init__(self) -> None:
        self._backends: dict[tuple[Operation, str], ProcessorBackend] = {}

    @overload
    def register(
        self,
        operation: Literal["localise"],
        name: str,
        backend: LocalisationBackend,
        *,
        replace: bool = False,
    ) -> None:
        """Register one typed scientific worker before concurrent processing.

        Args:
            operation: Scientific operation served by this worker.
            name: Registered worker name.
            backend: Typed worker implementation.
            replace: Allow replacement of an existing registration.
        """
        ...

    @overload
    def register(
        self, operation: Literal["track"], name: str, backend: TrackingBackend, *, replace: bool = False
    ) -> None:
        """Register one typed scientific worker before concurrent processing.

        Args:
            operation: Scientific operation served by this worker.
            name: Registered worker name.
            backend: Typed worker implementation.
            replace: Allow replacement of an existing registration.
        """
        ...

    @overload
    def register(
        self, operation: Literal["segment"], name: str, backend: SegmentationBackend, *, replace: bool = False
    ) -> None:
        """Register one typed scientific worker before concurrent processing.

        Args:
            operation: Scientific operation served by this worker.
            name: Registered worker name.
            backend: Typed worker implementation.
            replace: Allow replacement of an existing registration.
        """
        ...

    def register(
        self, operation: Operation, name: str, backend: ProcessorBackend, *, replace: bool = False
    ) -> None:
        """Register one worker; duplicate registration requires explicit replacement.

        Args:
            operation: Processing family served by the worker.
            name: Application-defined backend name.
            backend: Numerical worker implementation.
            replace: Explicitly replace an existing registration.
        """
        if operation not in ("localise", "track", "segment"):
            raise ValueError(f"Unsupported backend operation: {operation!r}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Backend name must be a nonempty string")
        _validate(backend)
        key = (operation, name)
        if key in self._backends and (not replace):
            raise ValueError(f"Backend {name!r} is already registered for {operation}")
        self._backends[key] = backend

    @overload
    def resolve(self, operation: Literal["localise"], name: str) -> LocalisationBackend:
        """Resolve a worker registered for this scientific operation.

        Args:
            operation: Scientific operation served by this worker.
            name: Registered worker name.

        Returns:
            The computed value.
        """
        ...

    @overload
    def resolve(self, operation: Literal["track"], name: str) -> TrackingBackend:
        """Resolve a worker registered for this scientific operation.

        Args:
            operation: Scientific operation served by this worker.
            name: Registered worker name.

        Returns:
            The computed value.
        """
        ...

    @overload
    def resolve(self, operation: Literal["segment"], name: str) -> SegmentationBackend:
        """Resolve a worker registered for this scientific operation.

        Args:
            operation: Scientific operation served by this worker.
            name: Registered worker name.

        Returns:
            The computed value.
        """
        ...

    def resolve(self, operation: Operation, name: str) -> ProcessorBackend:
        """Return a registered worker or explain how to configure it.

        Args:
            operation: Processing family.
            name: Registered backend name.

        Returns:
            The registered numerical worker.
        """
        try:
            return self._backends[operation, name]
        except KeyError:
            raise LookupError(
                f"No {operation} backend named {name!r}; register it with BackendRegistry.register() or pass a backend instance directly"
            ) from None


backends = BackendRegistry()


def resolve_backend(
    operation: Operation, backend: ProcessorBackend | str | None, *, registry: BackendRegistry | None = None
) -> ProcessorBackend | None:
    """Resolve a named/injected worker; None selects the processor's built-in worker.

    Args:
        operation: Processing family.
        backend: Worker instance, registered name, or None.
        registry: Application registry; defaults to the module registry.

    Returns:
        Worker implementation, or None for the built-in implementation.
    """
    if backend is None:
        return None
    if isinstance(backend, str):
        backend = (backends if registry is None else registry).resolve(operation, backend)
    _validate(backend)
    return _TypedWorker(operation, backend)


class _TypedWorker:
    """Adapt scheduler payloads to typed extension requests and validate outputs."""

    def __init__(self, operation: Any, backend: Any) -> None:
        self.operation, self.backend = (operation, backend)
        self.version = backend.version

    def compute(self, payload: Any, reporter: Any) -> Any:
        """Compute.

        Args:
            payload: Payload.
            reporter: Reporter.


        Returns:
            The computed value.
        """
        from .immutable import freeze

        if self.operation == "track":
            request = TrackingRequest(*payload)
            request = request._replace(rows=freeze(request.rows), labels=freeze(request.labels))
            result = self.backend.compute(request, reporter)
            if not isinstance(result, TrackingOutput):
                raise TypeError("Tracking backends must return TrackingOutput")
            return result.memberships
        if self.operation == "localise":
            local_request = LocalisationRequest(*payload)
            local_request = local_request._replace(
                images=freeze(local_request.images),
                frames=freeze(local_request.frames),
                candidates=freeze(local_request.candidates),
            )
            result = self.backend.compute(local_request, reporter)
            if not isinstance(result, LocalisationOutput):
                raise TypeError("Localisation backends must return LocalisationOutput")
            required = {"frame", "x", "y", "net_gradient"}
            if local_request.fitting:
                required.update(
                    {"photons", "bg", "sx", "sy", "lpx", "lpy", "ellipticity", "fit_state", "iterations"}
                )
            arrays = {key: np.asarray(value) for key, value in result.columns.items()}
            if (
                required - arrays.keys()
                or any(value.ndim != 1 or value.dtype.hasobject for value in arrays.values())
                or len({len(value) for value in arrays.values()}) > 1
            ):
                raise ValueError("Localisation backend returned invalid columns")
            if (
                arrays["frame"].dtype.kind not in "iu"
                or not np.isin(arrays["frame"], local_request.frames).all()
                or (not np.isfinite(arrays["x"]).all())
                or (not np.isfinite(arrays["y"]).all())
            ):
                raise ValueError("Localisation backend returned invalid positions or frames")
            if (
                isinstance(result.failed_fit_count, bool)
                or not isinstance(result.failed_fit_count, (int, np.integer))
                or result.failed_fit_count < 0
            ):
                raise ValueError("Failed fit count must be a nonnegative integer")
            return (result.columns, result.failed_fit_count)
        from .types import Segmentation

        request_segment = SegmentationWorkerRequest(*payload)
        request_segment = request_segment._replace(image=freeze(request_segment.image))
        result = self.backend.compute(request_segment, reporter)
        if not isinstance(result, Segmentation):
            raise TypeError("Segmentation backends must return Segmentation")
        binding = (
            request_segment.segmentation_id,
            request_segment.request.acquisition_id,
            request_segment.request.channel,
            request_segment.frame,
            request_segment.request.space,
        )
        if (
            result.id,
            result.acquisition_id,
            result.channel,
            result.frame,
            result.space,
        ) != binding or result.labels.shape != request_segment.image.shape:
            raise ValueError("Segmentation backend changed its request binding")
        if (
            result.source_image is None
            or result.source_image.dtype != request_segment.image.dtype
            or (not np.array_equal(result.source_image, request_segment.image, equal_nan=True))
        ):
            raise ValueError("Segmentation backend must preserve the source image")
        return result


def missing_backend(distribution: str) -> ImportError:
    """Build an actionable optional-dependency error without importing the dependency.

    Args:
        distribution: Missing installed distribution name.

    Returns:
        ImportError explaining the appropriate installation extra.
    """
    extras = {
        "picassosr": "localisation",
        "trackpy": "tracking",
        "cellpose": "segmentation",
        "scikit-learn": "clustering",
    }
    extra = extras.get(distribution, "all")
    return ImportError(
        f"{distribution} is required for this operation; install it with pip install 'molcore[{extra}]'"
    )
