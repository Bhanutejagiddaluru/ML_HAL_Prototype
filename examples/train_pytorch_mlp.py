
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.backends.pytorch_backend import HALLinear

class TinyMLP(nn.Module):
    def __init__(self, d_in=128, d_hidden=256, d_out=10):
        super().__init__()
        self.net = nn.Sequential(
            HALLinear(d_in, d_hidden), nn.ReLU(),
            HALLinear(d_hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)

def main():
    torch.manual_seed(42)
    N, D_IN, D_OUT = 4096, 128, 10
    X = torch.randn(N, D_IN)
    y = torch.randint(0, D_OUT, (N,))

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = TinyMLP(d_in=D_IN, d_hidden=256, d_out=D_OUT)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}")
        for xb, yb in pbar:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

    print("Training complete. Running a quick inference...")
    with torch.no_grad():
        pred = model(torch.randn(1, D_IN)).softmax(dim=-1)
        print("Pred:", pred.numpy())

if __name__ == "__main__":
    main()
