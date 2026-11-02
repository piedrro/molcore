"""Explicit, injectable worker backends with no optional-library imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from .cpu import WorkerProgress

Operation = Literal["localise", "track", "segment"]


class ProcessorBackend(Protocol):
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

    def compute(self, payload: Any, reporter: WorkerProgress) -> Any:
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
        raise TypeError(
            "A backend must expose compute(payload, reporter) and a string version"
        )


class BackendRegistry:
    """Caller-owned registry keyed by operation and name; no implicit discovery.

    Register implementations before starting concurrent work. Use a distinct
    registry per application/session when shared global configuration is unwanted.
    """

    def __init__(self) -> None:
        self._backends: dict[tuple[Operation, str], ProcessorBackend] = {}

    def register(
        self,
        operation: Operation,
        name: str,
        backend: ProcessorBackend,
        *,
        replace: bool = False,
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
        if key in self._backends and not replace:
            raise ValueError(f"Backend {name!r} is already registered for {operation}")
        self._backends[key] = backend

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
                f"No {operation} backend named {name!r}; register it with BackendRegistry.register() "
                "or pass a backend instance directly"
            ) from None


backends = BackendRegistry()


def resolve_backend(
    operation: Operation,
    backend: ProcessorBackend | str | None,
    *,
    registry: BackendRegistry | None = None,
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
        return (backends if registry is None else registry).resolve(operation, backend)
    _validate(backend)
    return backend


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
