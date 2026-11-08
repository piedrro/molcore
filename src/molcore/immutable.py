"""Owned immutable buffers and recursively immutable, pickle-safe values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Never, dataclass_transform

import numpy as np


class FrozenDict(dict):
    """JSON-compatible mapping that cannot be changed through its public API."""

    def _readonly(self, *args: Any, **kwargs: Any) -> Never:
        raise TypeError("Use a model mutator to replace immutable values")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __deepcopy__(self, memo: Any) -> Any:
        return self

    def __reduce__(self) -> Any:
        return (_restore_mapping, (dict(self),))


def _restore_mapping(values: dict[Any, Any]) -> FrozenDict:
    return FrozenDict({key: freeze(value) for key, value in values.items()})


def immutable_array(values: Any) -> np.ndarray:
    """Own a byte-backed numeric array, reusing already immutable buffers.

    Args:
        values: Values.


    Returns:
        The computed value.
    """
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("Immutable arrays cannot contain Python objects")
    base = array
    while isinstance(base.base, np.ndarray):
        base = base.base
    if isinstance(base.base, bytes) and (not array.flags.writeable):
        return array
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def freeze(value: Any) -> Any:
    """Detach and freeze nested metadata, arrays and containers.

    Args:
        value: Value.


    Returns:
        The computed value.
    """
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, np.ndarray):
        return immutable_array(value)
    if isinstance(value, Mapping):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    return value


def freeze_metadata(value: Mapping[str, Any]) -> FrozenDict:
    """Validate JSON annotations before detaching and recursively freezing them.

    Args:
        value: String-keyed JSON-compatible annotations, including NumPy values.

    Returns:
        Detached immutable metadata.
    """
    import json
    from pathlib import Path

    def _validate_keys(item: Any) -> None:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("Metadata keys must be strings")
            for child in item.values():
                _validate_keys(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                _validate_keys(child)

    def _convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Metadata must be JSON-compatible, got {type(item).__name__}")

    if not isinstance(value, Mapping):
        raise TypeError("Metadata must be a mapping")
    _validate_keys(value)
    json.dumps(value, default=_convert, allow_nan=False)
    return freeze(value)


@dataclass_transform(frozen_default=True)
def immutable_record(cls: Any) -> Any:
    """Frozen dataclass whose validated fields are recursively immutable.

    Returns:
        The computed value.
    """
    validate = getattr(cls, "__post_init__", None)

    def __post_init__(self: Any) -> None:
        if validate is not None:
            validate(self)
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            object.__setattr__(
                self,
                descriptor.name,
                freeze_metadata(value) if descriptor.name == "metadata" else freeze(value),
            )
        if "columns" in {descriptor.name for descriptor in fields(self)}:
            arrays = tuple(self.columns.values())
            if (
                any(not isinstance(value, np.ndarray) or value.ndim != 1 for value in arrays)
                or len({len(value) for value in arrays}) > 1
            ):
                raise ValueError("Result columns must be one-dimensional arrays of equal length")

    cls.__post_init__ = __post_init__
    cls.__deepcopy__ = lambda self, memo: self
    cls.__reduce__ = lambda self: (
        _restore_record,
        (type(self), {field.name: getattr(self, field.name) for field in fields(self)}),
    )
    return dataclass(frozen=True)(cls)


def _restore_record(cls: Any, values: dict[str, Any]) -> Any:
    record = cls.__new__(cls)
    for name, value in values.items():
        object.__setattr__(record, name, freeze_metadata(value) if name == "metadata" else freeze(value))
    return record


class ImmutableObject:
    """Construction-time writable record, sealed when initialization completes."""

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Use the dataset's update methods to replace this record")
        object.__setattr__(self, name, value)

    def _seal(self) -> None:
        for name, value in tuple(vars(self).items()):
            object.__setattr__(self, name, freeze(value))
        object.__setattr__(self, "_sealed", True)

    def __deepcopy__(self, memo: Any) -> Any:
        return self

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, freeze(value) if state.get("_sealed") else value)
