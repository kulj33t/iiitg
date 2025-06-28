import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

SEED = 21
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

iris = load_iris()
X = iris.data[:, :2]
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

idx = 0
x0 = torch.tensor(X_scaled[idx], dtype=torch.float32, requires_grad=True)
target_class = torch.tensor([y[idx]])

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
criterion = nn.CrossEntropyLoss()

def generate_loss_surface(model, grid_x, grid_y, target_class):
    Xg, Yg = np.meshgrid(grid_x, grid_y)
    Z = np.zeros_like(Xg)

    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            pt = torch.tensor([[Xg[i, j], Yg[i, j]]], dtype=torch.float32)
            out = model(pt)
            loss = criterion(out, target_class)
            Z[i, j] = loss.item()

    return Xg, Yg, Z

def plot_moving_point(Xg, Yg, Z, point, loss_val, step):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Xg, Yg, Z, levels=40, cmap='turbo')
    plt.plot(point[0], point[1], 'ro', markersize=8, label=f'Loss: {loss_val:.4f}')
    plt.title(f"Gradient Descent - Step {step}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(contour, label='Loss')
    plt.xlim(Xg.min(), Xg.max())
    plt.ylim(Yg.min(), Yg.max())
    plt.grid(True)
    plt.legend()
    # plt.savefig(f"vid/step_{step:03d}.png")
    plt.show()
    plt.close()

def gradient_descent_moving_point(model, x_start, steps, lr, box_size, resolution):
    x = x_start.clone().detach().requires_grad_(True)
    center = x_start.detach().numpy()

    grid_x = np.linspace(center[0] - box_size, center[0] + box_size, resolution)
    grid_y = np.linspace(center[1] - box_size, center[1] + box_size, resolution)
    Xg, Yg, Z = generate_loss_surface(model, grid_x, grid_y, target_class)

    for step in range(steps):
        out = model(x.unsqueeze(0))
        loss = criterion(out, target_class)
        current_pos = x.clone().detach().numpy()
        plot_moving_point(Xg, Yg, Z, current_pos, loss.item(), step)

        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
        x.grad.zero_()

gradient_descent_moving_point(model, x_start=x0, steps=100, lr=0.05, box_size=1.0, resolution=100)


