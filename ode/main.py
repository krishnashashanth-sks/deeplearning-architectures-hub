import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torchdiffeq import odeint
from train import train_model
from model import ODEFunc,TrueODE

# Define dimensions for the ODEFunc
input_dim = 2  # e.g., a 2D state [x, y]
hidden_dim = 50
output_dim = 2 # output should match input_dim

# Instantiate the ODEFunc neural network
func = ODEFunc(input_dim, hidden_dim, output_dim)

#  Define the true ODE function for a 2D spiral


# Instantiate the true ODE function
true_ode_function = TrueODE()

#  Create a list of initial conditions (x0s)
x0s = [
    torch.tensor([2., 0.], dtype=torch.float32).unsqueeze(0),  # Initial condition 1
    torch.tensor([-2., 1.], dtype=torch.float32).unsqueeze(0), # Initial condition 2
    torch.tensor([0., -3.], dtype=torch.float32).unsqueeze(0)  # Initial condition 3
]

#  Define a common set of time points (t_points)
t_points = torch.linspace(0., 20., 100) # From t=0 to t=20, with 100 points

#  For each initial condition, generate a true trajectory
true_trajectories = []

for i, x0 in enumerate(x0s):
    with torch.no_grad(): # No need to compute gradients for true data generation
        # odeint(func, initial_state, time_points)
        trajectory = odeint(true_ode_function, x0, t_points, method='dopri5')
    true_trajectories.append(trajectory)
    print(f"Trajectory {i+1} shape: {trajectory.shape}")

#  Organize the generated data
# We can keep them as a list of tensors for now.
# For training, we might concatenate them or sample batches.

# Let's just print the shapes and a few values to confirm
print(f"Number of initial conditions: {len(x0s)}")
print(f"Time points shape: {t_points.shape}")
print(f"Example initial condition shape: {x0s[0].shape}")
print(f"Example true trajectory shape: {true_trajectories[0].shape}")

print("First trajectory (first 5 points):\n", true_trajectories[0][:5])
print("Last trajectory (last 5 points):\n", true_trajectories[-1][-5:])


# Instantiate the MSELoss as the criterion
criterion = nn.MSELoss()
print("MSELoss (criterion) instantiated successfully.")

#  Instantiate the Adam optimizer
# 'func' is the ODEFunc instance created in a previous step
learning_rate = 0.01
optimizer = optim.Adam(func.parameters(), lr=learning_rate)
print(f"Adam optimizer instantiated successfully with learning rate: {learning_rate}.")

num_epochs = 500

train_model(num_epochs,x0s,true_trajectories,optimizer,func,t_points,criterion)

plt.figure(figsize=(12, 8))

# Iterate through each initial condition and its true trajectory
for i, x0 in enumerate(x0s):
    true_trajectory = true_trajectories[i]

    # Use the trained func to predict trajectories
    with torch.no_grad():
        pred_trajectory = odeint(func, x0, t_points, method='dopri5')

    # Convert tensors to numpy arrays for plotting
    true_traj_np = true_trajectory.squeeze().cpu().numpy()
    pred_traj_np = pred_trajectory.squeeze().cpu().numpy()

    # Plot true trajectory
    plt.plot(true_traj_np[:, 0], true_traj_np[:, 1], 'o--', label=f'True Trajectory {i+1}', alpha=0.7)
    # Plot predicted trajectory
    plt.plot(pred_traj_np[:, 0], pred_traj_np[:, 1], '-', label=f'Predicted Trajectory {i+1}', alpha=0.9)

plt.title('Neural ODE: True vs. Predicted Trajectories')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.grid(True)
plt.show()
