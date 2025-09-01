
import numpy as np

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D arrays")
    return a @ b  # NumPy BLAS

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0, dtype=x.dtype)

def add_bias(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    if b.ndim != 1:
        raise ValueError("bias must be 1D")
    return x + b
