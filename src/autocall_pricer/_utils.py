"""Internal helpers: immutable arrays and value semantics for frozen dataclasses."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import numpy.typing as npt


def frozen_array(values: npt.ArrayLike) -> np.ndarray:
    """Float copy of ``values`` that cannot be modified in place.

    Copying protects the object from later changes of the caller's array;
    the read-only flag protects it from changes through its own attributes.
    """
    arr = np.array(values, dtype=float)
    arr.setflags(write=False)
    return arr


def _hashable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    return value


class ValueSemantics:
    """Equality and hashing by value for dataclasses holding NumPy arrays.

    The dataclass-generated ``__eq__`` compares tuples of fields, which raises
    on arrays ("truth value of an array is ambiguous"), and the generated
    ``__hash__`` fails on them. Subclasses must be declared with ``eq=False``.
    Fields declared with ``compare=False`` are ignored.
    """

    def _compared_values(self) -> tuple[Any, ...]:
        return tuple(getattr(self, f.name) for f in fields(self) if f.compare)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, ValueSemantics)
        for a, b in zip(self._compared_values(), other._compared_values()):
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
        return True

    def __hash__(self) -> int:
        return hash((self.__class__, *map(_hashable, self._compared_values())))
