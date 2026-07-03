import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import ReceptanceWeightedKV
# 2. Prepare Training Data
input_dim = 10
hidden_dim = 5 # Used for key_dim and value_dim
output_dim = 1
num_samples = 1000

X_train = torch.randn(num_samples, input_dim, dtype=torch.float32)
y_train = torch.randn(num_samples, output_dim, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Number of batches in train_loader: {len(train_loader)}")

# 3. Implement Training Loop
model = ReceptanceWeightedKV(input_dim, hidden_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

print(f"\nStarting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad() # Zero the parameter gradients

        outputs = model(batch_X) # Forward pass
        loss = criterion(outputs, batch_y) # Calculate loss

        loss.backward() # Backward pass
        optimizer.step() # Optimize and update weights

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

# 4. Generate and Load New Prediction Data
num_samples_predict = 50
X_predict = torch.randn(num_samples_predict, input_dim, dtype=torch.float32)

predict_dataset = TensorDataset(X_predict)
predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

print(f"\nShape of X_predict: {X_predict.shape}")
print(f"Number of batches in predict_loader: {len(predict_loader)}")

# 5. Generate Predictions
model.eval() # Set the model to evaluation mode

all_predictions = [] # List to store predictions from each batch

with torch.no_grad(): # Disable gradient calculations for predictions
    for batch_X_predict in predict_loader:
        # DataLoader for predict_dataset only yields X, so we need to unpack it correctly
        if isinstance(batch_X_predict, (list, tuple)):
            batch_X_predict = batch_X_predict[0]

        batch_predictions = model(batch_X_predict)
        all_predictions.append(batch_predictions)

final_predictions = torch.cat(all_predictions, dim=0)

print("\nGenerated Predictions (first 5 samples):")
print(final_predictions[:5])
print(f"Total number of predictions: {len(final_predictions)}")