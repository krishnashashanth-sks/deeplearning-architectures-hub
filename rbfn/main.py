from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from model import RBFN

# 2. Generate an array of 200 input features, X, by creating evenly spaced values between -5 and 5.
X = np.linspace(-5, 5, 200)

# 3. Generate the target variable, y, by applying a non-linear function to X and adding some random noise.
y = np.sin(X) * 2 + X * 0.5 + np.random.normal(0, 0.5, 200)

# 4. Reshape X to be a 2D array, which is typically required for scikit-learn models.
X = X.reshape(-1, 1)

# 5. Split the generated X and y into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Instantiate the RBFN model
# Let's choose a number of RBF neurons, e.g., 10
num_rbf_neurons = 10
rbfn = RBFN(num_rbf_neurons=num_rbf_neurons)

print(f"RBFN model instantiated with {num_rbf_neurons} RBF neurons.")

# 2. Train the model
rbfn.fit(X_train, y_train)

print("RBFN model trained successfully.")

# 3. Make predictions on the test set
y_pred = rbfn.predict(X_test)

# 4. Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error on the test set: {mse:.4f}")

# Optionally, visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual Test Data', alpha=0.7)
plt.scatter(X_test, y_pred, label='RBFN Predictions', alpha=0.7)
plt.title('RBF Network Predictions vs. Actual Data')
plt.xlabel('X_test')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()