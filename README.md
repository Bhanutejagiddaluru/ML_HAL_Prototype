
# ML Infrastructure & Custom Hardware Integration — HAL Prototype

This repository is a **working prototype** of a hardware–software co‑design stack that bridges open‑source ML frameworks
(Pytorch) with a **mock custom accelerator** via a **Hardware Abstraction Layer (HAL)**. It is structured like a realistic
ML infra project: a C++ extension (optional) for the HAL fast path, a Python fallback, a tiny runtime/scheduler layer,
and framework adapters (PyTorch). It includes examples and a simple latency benchmark.

> ✅ The prototype runs **out of the box** using the Python fallback (no compilation).
> 🧩 If you compile the C++ extension, some ops get an extra speed‑up on multi‑core CPUs (simulating an accelerator fast path).
> 🔁 The design prepares you to swap the HAL implementation with a real device runtime (PCIe/NPU/FPGA, gRPC, etc.).

## Folder Structure

```
ML_HAL_Prototype/
├─ README.md
├─ requirements.txt
├─ requirements-full.txt
├─ pyproject.toml                 # build the C++ HAL extension via pybind11 (optional)
├─ images/
│  ├─ architecture.png
│  └─ dataflow.png
├─ scripts/
│  ├─ build_hal.sh               # Linux/macOS build
│  └─ build_hal.ps1              # Windows build (MSVC + CMake)
├─ src/
│  ├─ __init__.py
│  ├─ hal/
│  │  ├─ __init__.py
│  │  ├─ dispatcher.py           # chooses C++ extension if available, else Python fallback
│  │  ├─ py_hal.py               # pure-Python HAL ops (NumPy)
│  │  └─ cpp/
│  │     ├─ CMakeLists.txt
│  │     └─ hal_ext.cpp          # pybind11 C++ fast path (matmul)
│  ├─ backends/
│  │  ├─ __init__.py
│  │  └─ pytorch_backend.py      # HALLinear (custom autograd Function)
│  ├─ runtime/
│  │  ├─ __init__.py
│  │  ├─ scheduler.py            # trivial sequential scheduler + op fusion stub
│  │  └─ device.py               # device/placement registry
│  └─ distributed/
│     ├─ __init__.py
│     └─ launcher.py             # minimal torchrun-style launcher (optional)
├─ examples/
│  ├─ benchmark_inference.py     # measures end-to-end latency (< 2s targets small models)
│  └─ train_pytorch_mlp.py       # trains a tiny MLP using HALLinear
└─ tests/
   └─ test_hal_linear.py         # sanity checks for HALLinear forward/backward
```

## Quickstart

```bash
# 1) Create venv & install minimal deps
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) (Optional) Build the C++ HAL extension for extra speed
# Linux/macOS:
bash scripts/build_hal.sh
# Windows (x64 Native Tools Command Prompt for VS):
powershell -ExecutionPolicy Bypass -File scripts/build_hal.ps1

# 3) Run a latency benchmark
python examples/benchmark_inference.py

# 4) Train a tiny MLP on synthetic data using HALLinear
python examples/train_pytorch_mlp.py
```

### Expected Output (benchmark)
You should see a single-forward latency printed in milliseconds and an end-to-end batch latency. On most laptops,
for the default sizes, end-to-end latency should be well under **2 seconds**.

> Note: This is a **prototype**. The C++ fast path uses CPU + OpenMP to simulate an accelerator. Replace `hal_ext.cpp`
with your actual device driver calls (DMA, command queues, event waits, etc.), keeping the Python API stable.

## How it Works

- **HAL API** (`src/hal/dispatcher.py`): Defines a minimal API (`matmul`, `relu`) used by framework adapters.
- **C++ Extension** (`src/hal/cpp/hal_ext.cpp`): Implements `matmul` with multi-threading (simulated accelerator path).
- **Python Fallback** (`src/hal/py_hal.py`): Uses NumPy; always available.
- **PyTorch Backend** (`src/backends/pytorch_backend.py`): Defines `HALLinear`, a `torch.nn.Module` using a custom
  autograd `Function` that calls the HAL for the forward pass and uses vanilla PyTorch for the backward pass.
- **Runtime/Scheduler** (`src/runtime/*`): Stubs for op fusion/placement; shows where you'd add graph compilation.

## Swapping in a Real Accelerator

- Replace the C++ code in `hal_ext.cpp` with calls into your device runtime (e.g. `device_enqueue_gemm(...)`).
- Plumb host↔device copies (pinned memory, DMA) and stream synchronization.
- Expand the HAL API (conv, layernorm, softmax) and add layout/tiling helpers in `runtime/`.
- Keep the Python signatures the same so `pytorch_backend.py` doesn't change.

---

© 2025 – MIT License. For demo purposes only.
