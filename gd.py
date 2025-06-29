import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )
    def forward(self, x):
        return self.model(x)

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

def flatten_params(model):
    return torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])

def set_weights_from_flat(model, flat_vector):
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_vector[offset:offset + numel].view_as(p))
            offset += numel

torch.manual_seed(0)
w0 = flatten_params(model).detach()
d1 = torch.randn_like(w0); d1 /= torch.norm(d1)
d2 = torch.randn_like(w0); d2 /= torch.norm(d2)

alpha_range = np.linspace(-3.0, 3.0, 150)
beta_range = np.linspace(-3.0, 3.0, 150)
ALPHA, BETA = np.meshgrid(alpha_range, beta_range)

initial_w = flatten_params(model).detach()
LOSS_INIT = np.zeros_like(ALPHA)
for i in range(ALPHA.shape[0]):
    for j in range(ALPHA.shape[1]):
        w_test = initial_w + ALPHA[i, j] * d1 + BETA[i, j] * d2
        set_weights_from_flat(model, w_test)
        with torch.no_grad():
            LOSS_INIT[i, j] = criterion(model(X_train), y_train).item()

loss_min, loss_max = LOSS_INIT.min(), LOSS_INIT.max()

plt.ion()
for epoch in range(10):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

    w_curr = flatten_params(model).detach()
    LOSS = np.zeros_like(ALPHA)
    for i in range(ALPHA.shape[0]):
        for j in range(ALPHA.shape[1]):
            w_test = w_curr + ALPHA[i, j] * d1 + BETA[i, j] * d2
            set_weights_from_flat(model, w_test)
            with torch.no_grad():
                LOSS[i, j] = criterion(model(X_train), y_train).item()
    set_weights_from_flat(model, w_curr)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        LOSS,
        extent=[alpha_range[0], alpha_range[-1], beta_range[0], beta_range[-1]],
        origin='lower',
        cmap='turbo',
        aspect='auto',
        vmin=loss_min,
        vmax=loss_max,
        interpolation='nearest'
    )
    plt.colorbar(im, label='Loss')
    plt.plot(0, 0, 'ro', markersize=10, label=f'Loss = {loss.item():.4f}')
    plt.title(f"Epoch {epoch+1}")
    plt.xlabel("Alpha (direction d1)")
    plt.ylabel("Beta (direction d2)")
    plt.legend()
    plt.grid(False)
    plt.pause(0.15)
    plt.clf()

plt.ioff()
plt.close()

