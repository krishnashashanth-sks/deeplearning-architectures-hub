from torchdiffeq import odeint

def train_model(num_epochs,x0s,true_trajectories,optimizer,func,t_points,criterion):
    print("Starting training loop...")

    # Store losses for potential plotting later
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        # Iterate through each initial condition and its true trajectory
        for i, x0 in enumerate(x0s):
            true_trajectory = true_trajectories[i]

            # 1. Zero the gradients
            optimizer.zero_grad()

            # 2. Forward pass: Use the ODE solver to predict trajectories
            # Ensure t_points is passed correctly
            pred_trajectory = odeint(func, x0, t_points, method='dopri5')

            # 3. Calculate loss: Compare predicted trajectories with ground truth
            # Reshape true_trajectory to match pred_trajectory (time_points, batch_size, state_dim)
            # The true_trajectory from generation is already (time_points, 1, state_dim)
            # pred_trajectory will also be (time_points, 1, state_dim) due to x0.unsqueeze(0)
            loss = criterion(pred_trajectory, true_trajectory)
            total_loss += loss.item()

            # 4. Backward pass: Compute gradients through the ODE solver
            loss.backward()

            # 5. Optimizer step: Update the neural network's parameters
            optimizer.step()

        average_loss = total_loss / len(x0s)
        epoch_losses.append(average_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0: # Print loss every 50 epochs or on the first epoch
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}')

    print("Training loop finished.")