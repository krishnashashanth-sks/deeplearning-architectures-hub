from pennylane import numpy as np # Keep pennylane's numpy for quantum operations
import torch
import matplotlib.pyplot as plt

# Visualize the decision boundary
def plot_decision_boundary(model, X, y, device, title='Decision Boundary'):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Convert meshgrid to tensor and send to device
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        Z = model(grid_tensor).cpu().numpy()
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[y.cpu().numpy().flatten() == 0, 0], X[y.cpu().numpy().flatten() == 0, 1], color='red', marker='o', label='Class 0')
    plt.scatter(X[y.cpu().numpy().flatten() == 1, 0], X[y.cpu().numpy().flatten() == 1, 1], color='blue', marker='x', label='Class 1')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
