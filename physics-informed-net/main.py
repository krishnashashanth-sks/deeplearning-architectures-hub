from model import PINN
from losses import *
import tensorflow as tf
import numpy as np
import maplotlib.pyplot as plt
from train import train_step

model = PINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# --- Training Parameters ---
n_epochs = 2000
n_collocation_points = 100

# Collocation points (where we enforce the physics)
x_collocation = tf.constant(np.random.rand(n_collocation_points, 1) * 10, dtype=tf.float32) # x from 0 to 10

# Initial condition point
x_initial = tf.constant([[0.0]], dtype=tf.float32)
y_initial = tf.constant([[1.0]], dtype=tf.float32)

# Store loss history
history = {'total_loss': [], 'physics_loss': [], 'ic_loss': []}

print("Starting training...")
for epoch in range(n_epochs):
    total_loss, p_loss, ic_loss = train_step(x_collocation, x_initial, y_initial, model, optimizer,physics_loss,initial_condition_loss)

    history['total_loss'].append(total_loss.numpy().item())
    history['physics_loss'].append(p_loss.numpy().item() if isinstance(p_loss.numpy(), np.ndarray) else p_loss.numpy())
    history['ic_loss'].append(ic_loss.numpy().item())

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss.numpy().item():.4f}, Physics Loss: {p_loss.numpy().item():.4f}, IC Loss: {ic_loss.numpy().item():.4f}")

print("Training finished.")

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(history['total_loss'], label='Total Loss')
plt.plot(history['physics_loss'], label='Physics Loss')
plt.plot(history['ic_loss'], label='Initial Condition Loss')
plt.title('Loss History During PINN Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log') # Use a log scale for better visualization of decreasing losses
plt.legend()
plt.grid(True)
plt.show()

# Generate points for plotting
x_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_analytical = np.exp(-x_test)
y_pinn = model(tf.constant(x_test, dtype=tf.float32)).numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_test, y_analytical, label='Analytical Solution: $e^{-x}$', color='red', linestyle='--')
plt.plot(x_test, y_pinn, label='PINN Solution', color='blue', alpha=0.7)
plt.scatter(x_initial, y_initial, color='green', marker='o', s=100, label='Initial Condition')
plt.title('PINN Solution vs. Analytical Solution for dy/dx + y = 0')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()