
import torch
from src.backends.pytorch_backend import HALLinear

def test_forward_backward():
    torch.manual_seed(0)
    layer = HALLinear(16, 8)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    y.sum().backward()
    # ensure grads exist
    assert layer.weight.grad is not None
    if layer.bias is not None:
        assert layer.bias.grad is not None
