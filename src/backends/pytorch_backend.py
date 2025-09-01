
import torch
from torch import nn
from torch.autograd import Function
import numpy as np

from ..hal.dispatcher import hal

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().contiguous().to('cpu').float().numpy()

def _from_numpy(x: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(x).to(like.device).to(like.dtype)

class HALMatMulFunction(Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        a_np = _to_numpy(a)
        b_np = _to_numpy(b)
        y_np = hal.matmul(a_np, b_np)
        y = _from_numpy(y_np, a)
        ctx.save_for_backward(a, b)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        # Use regular PyTorch ops for gradients in the prototype
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = grad_out @ b.t()
        if ctx.needs_input_grad[1]:
            grad_b = a.t() @ grad_out
        return grad_a, grad_b

class HALLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.use_bias:
            fan_in = self.weight.shape[1]
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = HALMatMulFunction.apply(x, self.weight.t())
        if self.bias is not None:
            # add bias via HAL for the demo
            y_np = hal.add_bias(y.detach().cpu().float().numpy(), self.bias.detach().cpu().float().numpy())
            y = torch.from_numpy(y_np).to(y.device).to(y.dtype)
        return y
