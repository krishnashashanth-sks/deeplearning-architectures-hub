from model import build_fractalnet_model
from tensorflow import keras
import numpy as np

# Example parameters for model instantiation
input_shape = (32, 32, 3)  # Example for CIFAR-10 like data
num_classes = 10           # Example for CIFAR-10 like classification
filters_list = [32, 64, 128] # Filters for each fractal block
num_columns_list = [3, 3, 3] # Number of columns for each fractal block
drop_path_prob_base = 0.2  # Base drop path probability

# Instantiate the model
fractal_net_model = build_fractalnet_model(
    input_shape=input_shape,
    num_classes=num_classes,
    filters_list=filters_list,
    num_columns_list=num_columns_list,
    drop_path_prob_base=drop_path_prob_base
)

# Display model summary
fractal_net_model.summary()

print("FractalNet model instantiated successfully.")

# Define training parameters
initial_learning_rate = 0.01
decay_rate = 0.9
decay_steps = 10000
epochs = 10
batch_size = 64

# Create a learning rate schedule (e.g., ExponentialDecay)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Define the optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
fractal_net_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy', # Assuming integer labels
    metrics=['accuracy']
)

input_shape_tuple = fractal_net_model.input_shape[1:]
num_samples = 1000
x_train = np.random.rand(num_samples, *input_shape_tuple).astype(np.float32)
y_train = np.random.randint(0, num_classes, num_samples).astype(np.int32)
x_val = np.random.rand(num_samples // 4, *input_shape_tuple).astype(np.float32)
y_val = np.random.randint(0, num_classes, num_samples // 4).astype(np.int32)

# Calculate steps_per_epoch
steps_per_epoch = len(x_train) // batch_size

# Train the model
history = fractal_net_model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    steps_per_epoch=steps_per_epoch
)


# --- Model Evaluation and Inference ---

print("Creating dummy test data for demonstration.")
x_test = np.random.rand(num_samples // 4, *input_shape_tuple).astype(np.float32)
y_test = np.random.randint(0, num_classes, num_samples // 4).astype(np.int32)

# Evaluate the model on the test data
print("\nEvaluating the model on test data...")
test_loss, test_accuracy = fractal_net_model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Perform model inference on a small batch of test samples
print("\nPerforming inference on a small batch of test data...")
num_inference_samples = 5
# Ensure we don't try to get more samples than available in x_test
actual_num_inference_samples = min(num_inference_samples, len(x_test))
inference_samples = x_test[:actual_num_inference_samples]
inference_predictions = fractal_net_model.predict(inference_samples)

print("Sample Predictions (softmax probabilities):")
for i, pred in enumerate(inference_predictions):
    predicted_class = np.argmax(pred)
    true_class = y_test[i]
    print(f"  Sample {i+1}: Predicted Class = {predicted_class}, True Class = {true_class}, Probabilities = {pred}")
