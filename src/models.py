import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Jaringan saraf feed-forward; dipakai untuk model IDS maupun detektor."""

    def __init__(self, in_dim, hidden, out_dim=2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_classifier(model, x, y, epochs, lr, batch_size, seed):
    """Melatih classifier dengan cross-entropy berbobot kelas dan optimizer Adam."""
    torch.manual_seed(seed)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    counts = torch.bincount(y_t, minlength=2).float()
    weight = counts.sum() / (2 * counts)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        print(f"  epoch {epoch + 1:02d}/{epochs}  loss={total / len(x_t):.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, batch_size=4096):
    """Mengembalikan probabilitas kelas 1 untuk seluruh data."""
    x_t = torch.as_tensor(x, dtype=torch.float32)
    probs = [torch.softmax(model(x_t[i:i + batch_size]), dim=1)[:, 1] for i in range(0, len(x_t), batch_size)]
    return torch.cat(probs).numpy()
