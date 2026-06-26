
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from model import DeepONet
from layers import *

example_input_shape = 10 # Example: 10 sensor readings
example_output_dim = 32 # Example: Feature vector of dimension 32

branch_net = BranchNetwork(output_dim=example_output_dim, input_shape=example_input_shape)

example_trunk_input_shape = 2 # Example: 2D coordinates (x, y)
trunk_net = TrunkNetwork(output_dim=example_output_dim, input_shape=example_trunk_input_shape)

# Let's define a simple operator for demonstration:
# G(u)(x) = integral_0^1 u(s) * K(x, s) ds
# where u(s) is the input function and K(x,s) is a kernel function.
# For simplicity, we'll choose a specific kernel and discrete input functions.

# Operator: G(u)(x) = sum(u_i * K(x, s_i)) for discrete s_i
# Let K(x, s) = sin(pi * x * s)

# 1. Define the number of input functions (u) we want to generate
num_input_functions = 1000

# 2. Define the number of sensor points for the input function (branch input)
# This matches example_input_shape
sensor_points = np.linspace(0, 1, example_input_shape)

# 3. Define the domain for the output query points (trunk input)
# This matches example_trunk_input_shape for 1D output domain
output_query_points_1d = np.linspace(0, 1, 100) # Example: 100 query points for each input function

# If example_trunk_input_shape is 2, we would need 2D query points, e.g., meshgrid
# For this example, let's assume a 1D output domain for simplicity for now,
# and adapt if trunk_input_shape > 1 later for a more complex example.

# Generate random input functions (u) sampled at sensor_points
# Each u will be a vector of example_input_shape (e.g., 10) values
input_functions = np.random.rand(num_input_functions, example_input_shape)

# Define the kernel function
def kernel(x, s):
  return np.sin(np.pi * x * s)

# Initialize lists to store training data
branch_inputs = [] # List of input functions
trunk_inputs = [] # List of query points
operator_outputs = [] # List of true operator outputs

# Generate input-output pairs
for i in range(num_input_functions):
  u = input_functions[i]
  for x in output_query_points_1d:
    # For a fixed u and a specific x, compute the operator's output
    # G(u)(x) = sum(u_j * K(x, s_j)) for j=0 to example_input_shape-1
    # Note: K(x,s_j) is a vector of length example_input_shape, and u_j is also a vector
    # We need to perform an element-wise product and then sum.
    # For the simplified operator, let's assume a basic interaction
    # More precisely, a discrete integral approximation:
    # G(u)(x) = sum_{j=0}^{M-1} u(s_j) * K(x, s_j) * delta_s
    # where delta_s = sensor_points[1] - sensor_points[0]
    # For simplicity, let's assume delta_s is absorbed into the weights of DeepONet.
    # So, output is approx sum(u_j * K(x, s_j))

    # Calculate the true output of the operator
    # This part depends on how the operator is defined. Let's use the discrete sum form.
    # The sensor_points are the 's' values for the input function u.
    # The current 'x' is the output query point.

    # Create a vector of kernel values for the current x and all sensor_points
    kernel_values = kernel(x, sensor_points)

    # Element-wise multiplication of input function values and kernel values
    # This corresponds to u(s_j) * K(x, s_j)
    product = u * kernel_values

    # Sum the products to get the operator output for this (u, x) pair
    true_output = np.sum(product)

    branch_inputs.append(u)
    trunk_inputs.append([x]) # Trunk input needs to be 2D if example_trunk_input_shape is 2, but for 1D query points, it's [x]
    operator_outputs.append(true_output)

# Convert lists to TensorFlow tensors
X_branch = tf.constant(np.array(branch_inputs), dtype=tf.float32)
X_trunk = tf.constant(np.array(trunk_inputs), dtype=tf.float32)
Y_true = tf.constant(np.array(operator_outputs), dtype=tf.float32)

# Reshape Y_true to be (num_samples, 1)
Y_true = tf.reshape(Y_true, (-1, 1))

print(f"Shape of X_branch (input functions): {X_branch.shape}")
print(f"Shape of X_trunk (query points): {X_trunk.shape}")
print(f"Shape of Y_true (operator outputs): {Y_true.shape}")

deeponet_model = DeepONet(branch_net, trunk_net)

# Compile the model with the chosen optimizer and loss function
deeponet_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError()
)

history = deeponet_model.fit(
    x=(X_branch, X_trunk),
    y=Y_true,
    epochs=50, # You can adjust the number of epochs
    batch_size=256, # Adjust batch size as needed
    validation_split=0.2 # Use 20% of data for validation
)


# Extract training and validation loss from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a range of epochs for the x-axis
epochs = range(1, len(train_loss) + 1)

# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 1. Generate a new set of input functions for testing
num_test_input_functions = 100 # Use 100 test functions

# Assuming sensor_points, output_query_points_1d, and kernel are already defined from previous steps
# If not, they would need to be re-defined or imported from the previous context.
# For continuity, we assume they are accessible here.

# Generate random input functions (u) sampled at sensor_points for testing
test_input_functions = np.random.rand(num_test_input_functions, example_input_shape)

# Initialize lists to store test data
X_branch_test_list = []
X_trunk_test_list = []
Y_true_test_list = []

# 2. Create a test dataset
for i in range(num_test_input_functions):
  u_test = test_input_functions[i]
  for x_test in output_query_points_1d:
    kernel_values_test = kernel(x_test, sensor_points)
    product_test = u_test * kernel_values_test
    true_output_test = np.sum(product_test)

    X_branch_test_list.append(u_test)
    X_trunk_test_list.append([x_test]) # Trunk input needs to be 2D if example_trunk_input_shape is 2, but for 1D query points, it's [x]
    Y_true_test_list.append(true_output_test)

# 3. Convert lists to TensorFlow tensors
X_branch_test = tf.constant(np.array(X_branch_test_list), dtype=tf.float32)
X_trunk_test = tf.constant(np.array(X_trunk_test_list), dtype=tf.float32)
Y_true_test = tf.constant(np.array(Y_true_test_list), dtype=tf.float32)

# Reshape Y_true_test to be (num_samples, 1)
Y_true_test = tf.reshape(Y_true_test, (-1, 1))

print(f"Shape of X_branch_test (test input functions): {X_branch_test.shape}")
print(f"Shape of X_trunk_test (test query points): {X_trunk_test.shape}")
print(f"Shape of Y_true_test (test operator outputs): {Y_true_test.shape}")


# 4. Use the deeponet_model.predict() method with (X_branch_test, X_trunk_test) as input
predictions = deeponet_model.predict((X_branch_test, X_trunk_test))

# 5. Calculate the Mean Squared Error (MSE) between Y_true_test and the model's predictions
mse = mean_squared_error(Y_true_test.numpy(), predictions)

# 6. Print the calculated MSE
print(f"Mean Squared Error on test set: {mse}")

# 7. For a few example test input functions, select their corresponding data
num_examples_to_plot = 3

plt.figure(figsize=(15, 5 * num_examples_to_plot))

# Iterate through a few example test functions
for i in range(num_examples_to_plot):
    # Select a random index for an input function from the test set
    # Note: X_branch_test has shape (num_test_input_functions * len(output_query_points_1d), example_input_shape)
    # We need to pick a block of data corresponding to one input function.
    start_idx = i * len(output_query_points_1d)
    end_idx = (i + 1) * len(output_query_points_1d)

    # Extract the branch input, trunk inputs, and true outputs for this specific input function
    example_X_branch = X_branch_test[start_idx:end_idx]
    example_X_trunk = X_trunk_test[start_idx:end_idx]
    example_Y_true = Y_true_test[start_idx:end_idx]

    # Make predictions for this example input function across all query points
    example_predictions = deeponet_model.predict((example_X_branch, example_X_trunk))

    # 9. Plot the true operator output and the predicted operator output
    plt.subplot(num_examples_to_plot, 1, i + 1)
    plt.plot(example_X_trunk.numpy(), example_Y_true.numpy(), 'b-', label='True Output')
    plt.plot(example_X_trunk.numpy(), example_predictions, 'r--', label='Predicted Output')
    plt.title(f'Operator Output for Test Function {i+1}')
    plt.xlabel('Query Point (x)')
    plt.ylabel('Operator Value G(u)(x)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

print("Visual comparison of true vs. predicted operator outputs for example test functions displayed.")

# 1. Generate a single new, unseen input function (u)
# The 'example_input_shape' and 'sensor_points' are assumed to be available from previous cells.
new_input_function = np.random.rand(example_input_shape)

# 2. Define a set of query points (x) across the output domain
# The 'output_query_points_1d' is assumed to be available from previous cells.
# For consistency, we use the same query points as for training/testing visualization.
query_points_inference = output_query_points_1d

# 3. Prepare the branch input and trunk input tensors for the inference step.
# The branch input will be the single new input function (u) repeated for each query point.
num_query_points = len(query_points_inference)
branch_input_inference = tf.constant(np.tile(new_input_function, (num_query_points, 1)), dtype=tf.float32)

# The trunk input will be the set of query points.
# Reshape query_points_inference to match the expected input shape for the trunk network.
# (num_query_points, 1) if example_trunk_input_shape is 1
trunk_input_inference = tf.constant(query_points_inference.reshape(-1, 1), dtype=tf.float32)

# 4. Use the trained deeponet_model's .predict() method
predicted_output = deeponet_model.predict((branch_input_inference, trunk_input_inference))

# 5. Calculate the true operator output for this new input function
true_output_inference_list = []
for x_val in query_points_inference:
    kernel_values_inference = kernel(x_val, sensor_points)
    product_inference = new_input_function * kernel_values_inference
    true_output_inference = np.sum(product_inference)
    true_output_inference_list.append(true_output_inference)

true_output_inference = tf.constant(np.array(true_output_inference_list).reshape(-1, 1), dtype=tf.float32)

print(f"Shape of branch_input_inference: {branch_input_inference.shape}")
print(f"Shape of trunk_input_inference: {trunk_input_inference.shape}")
print(f"Shape of predicted_output: {predicted_output.shape}")
print(f"Shape of true_output_inference: {true_output_inference.shape}")

# 6. Plot the predicted operator output against the true operator output
plt.figure(figsize=(10, 6))
plt.plot(query_points_inference, true_output_inference.numpy(), 'b-', label='True Operator Output')
plt.plot(query_points_inference, predicted_output, 'r--', label='Predicted Operator Output')
plt.title('DeepONet Inference: True vs. Predicted Output for a New Input Function')
plt.xlabel('Query Point (x)')
plt.ylabel('Operator Value G(u)(x)')
plt.legend()
plt.grid(True)

# 7. Display the plot.
plt.show()