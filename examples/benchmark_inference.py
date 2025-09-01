
import time
import numpy as np
import torch
from src.backends.pytorch_backend import HALLinear

def main():
    torch.manual_seed(0)
    device = torch.device('cpu')  # prototype uses CPU tensors

    B, IN, OUT = 64, 512, 512  # adjust sizes if needed
    x = torch.randn(B, IN, device=device)
    layer = HALLinear(IN, OUT).to(device).eval()

    # Warmup
    with torch.no_grad():
        _ = layer(x)

    # Measure single forward
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        y = layer(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()

    print(f"Single forward latency: {(t1 - t0)*1000:.2f} ms, output shape={tuple(y.shape)}")

    # End-to-end micro-batch
    batches = 8
    t0 = time.time()
    with torch.no_grad():
        for _ in range(batches):
            _ = layer(x)
    t1 = time.time()
    print(f"End-to-end latency for {batches} batches: {t1 - t0:.3f} s (target < 2s)")

if __name__ == "__main__":
    main()
