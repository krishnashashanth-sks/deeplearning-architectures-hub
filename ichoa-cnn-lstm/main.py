import numpy as np
from model import build_model
from tensorflow.keras.optimizers import Adam
from utils import prepare_dummy_data, define_search_space, generate_test_data, map_to_search_space
from ichoa_optim import run_ichoa_optimization
from evaluate import evaluate_and_infer_model

# Parameters for data generation and IChOA
N_SAMPLES_TRAIN = 100
N_TIMESTEPS = 50
N_FEATURES = 10
NUM_CHIMPS = 10
MAX_ITERATIONS = 5
N_SAMPLES_TEST = 20

# 1. Prepare Dummy Data
X_train, y_train, n_timesteps, n_features = prepare_dummy_data(N_SAMPLES_TRAIN, N_TIMESTEPS, N_FEATURES)

# 2. Define Search Space
search_space = define_search_space()

# --- Baseline Evaluation (Before Optimization) ---
# To prevent TypeErrors, we map a random initial position [0.5 across all keys] to get valid baseline parameters
baseline_normalized = {key: 0.5 for key in search_space.keys()}
baseline_hyperparameters = map_to_search_space(baseline_normalized, search_space)

print("\n--- Building Baseline Model ---")
baseline_model = build_model(baseline_hyperparameters, n_timesteps, n_features)

# Compile the baseline model
baseline_optimizer = Adam(learning_rate=baseline_hyperparameters['learning_rate'])
baseline_model.compile(optimizer=baseline_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

baseline_model.summary()

# Train baseline model with its assigned hyperparameters
base_epochs = int(baseline_hyperparameters['epochs'])
base_batch_size = int(baseline_hyperparameters['batch_size'])

history = baseline_model.fit(X_train, y_train, epochs=base_epochs, batch_size=base_batch_size, validation_split=0.2, verbose=0)
baseline_validation_accuracy = history.history['val_accuracy'][-1]
print(f"Before IChOA Optimization Validation Accuracy: {baseline_validation_accuracy:.4f}\n")


# 3. Run IChOA Optimization
best_hyperparameters = run_ichoa_optimization(X_train, y_train, n_timesteps, n_features, 
                                             search_space, NUM_CHIMPS, MAX_ITERATIONS)


# 4. Build and Train Final Model with Best Hyperparameters
if best_hyperparameters:
    print("\n--- Building Final Optimized Model ---")
    final_model = build_model(best_hyperparameters, n_timesteps, n_features)
    
    final_optimizer = Adam(learning_rate=best_hyperparameters['learning_rate'])
    final_model.compile(optimizer=final_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print("\nFinal Model Summary:")
    final_model.summary()

    # Extract optimized training execution settings
    final_epochs = int(best_hyperparameters['epochs'])
    final_batch_size = int(best_hyperparameters['batch_size'])

    print(f"\nTraining final model for {final_epochs} epochs with batch size {final_batch_size}...")
    final_history = final_model.fit(X_train, y_train, epochs=final_epochs, batch_size=final_batch_size, verbose=1)

    print("\nFinal Model Training History:")
    print(final_history.history)
    print("Final model training complete.")
    
    # 5. Generate Test Data
    X_test, y_test = generate_test_data(N_SAMPLES_TEST, n_timesteps, n_features)

    # 6. Evaluate and Perform Inference
    evaluate_and_infer_model(final_model, X_test, y_test)
else:
    print("\nFinal model was not built due to an error during optimization.")