
import importlib
from typing import Any, Callable
import numpy as np

class _HAL:
    def __init__(self) -> None:
        self._impl = None
        try:
            self._impl = importlib.import_module("hal_ext")  # C++ ext if present (built to site-packages)
        except Exception:
            from . import py_hal as impl
            self._impl = impl

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._impl.matmul(a, b)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return self._impl.relu(x)

    def add_bias(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._impl.add_bias(x, b)

hal = _HAL()
