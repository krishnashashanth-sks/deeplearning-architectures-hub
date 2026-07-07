import numpy as np
import random

# ---  Data Preparation Function ---
def prepare_dummy_data(n_samples=100, n_timesteps=50, n_features=10):
    """
    Prepares dummy sequential data for training and testing.
    """
    X = np.random.rand(n_samples, n_timesteps, n_features)
    y = np.random.randint(0, 2, n_samples) # Binary classification example
    print(f"Prepared dummy data: X_shape={X.shape}, y_shape={y.shape}")
    return X, y, n_timesteps, n_features

# ---  Hyperparameter Search Space Definition ---
def define_search_space():
    """
    Defines the hyperparameter search space for IChOA.
    Continuous parameters are tuples (min_val, max_val).
    Discrete parameters are lists of choices.
    """
    search_space = {
        'filters1': [16, 32, 64],
        'kernel_size1': [3, 5],
        'pool_size1': [2],
        'dropout1': (0.1, 0.3),

        'filters2': [64, 128, 256],
        'kernel_size2': [3, 5],
        'pool_size2': [2],
        'dropout2': (0.2, 0.4),

        'lstm_units': [50, 100, 150],
        'dropout_lstm': (0.2, 0.4),

        'learning_rate': (0.0001, 0.01),
        'epochs': [5, 10],
        'batch_size': [16, 32, 64]
    }
    print("Hyperparameter search space defined.")
    return search_space

# --- Mapping Function for IChOA ---
def map_to_search_space(chimp_position, search_space):
    """
    Maps normalized chimp positions [0, 1] to actual hyperparameter values
    based on the defined search space (continuous or discrete).
    """
    hyperparameters = {}
    for param, values in search_space.items():
        if isinstance(values, tuple) and len(values) == 2: # Continuous range (min, max)
            min_val, max_val = values
            hyperparameters[param] = min_val + chimp_position[param] * (max_val - min_val)
        elif isinstance(values, list): # Discrete choices
            # For discrete choices, quantize the [0,1] value to select an option
            idx = min(int(chimp_position[param] * len(values)), len(values) - 1)
            hyperparameters[param] = values[idx]
    return hyperparameters

# ---  Chimp Initialization Function ---
def initialize_chimps(num_chimps, search_space_keys):
    """
    Initializes a population of chimps with random positions between 0 and 1.
    """
    chimps = []
    for _ in range(num_chimps):
        chimp_position = {key: random.random() for key in search_space_keys}
        chimps.append(chimp_position)
    return chimps

# ---  Generate Test Data Function ---
def generate_test_data(n_samples_test, n_timesteps, n_features):
    """
    Generates dummy test data with the specified shapes.
    """
    X_test = np.random.rand(n_samples_test, n_timesteps, n_features)
    y_test = np.random.randint(0, 2, n_samples_test)
    print(f"\nTest data generated: X_test_shape={X_test.shape}, y_test_shape={y_test.shape}")
    return X_test, y_test
