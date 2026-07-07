import random
from utils import initialize_chimps,map_to_search_space
import numpy as np
from model import build_model
from tensorflow.keras.optimizers import Adam

# ---  IChOA Optimization Loop Function ---
def run_ichoa_optimization(X_train, y_train, n_timesteps, n_features, search_space, num_chimps, max_iterations):
    """
    Executes the IChOA algorithm to find optimal hyperparameters.
    """
    print("Starting IChOA optimization loop...")

    search_space_keys = list(search_space.keys())
    chimps = initialize_chimps(num_chimps, search_space_keys)

    best_overall_fitness = -float('inf')
    best_overall_hyperparameters = None

    # Initialize leader chimp positions and scores
    alph_chimp_pos = {key: 0.0 for key in search_space_keys}
    beta_chimp_pos = {key: 0.0 for key in search_space_keys}
    delta_chimp_pos = {key: 0.0 for key in search_space_keys}

    alpha_score = -float('inf')
    beta_score = -float('inf')
    delta_score = -float('inf')

    for iteration in range(max_iterations):
        print(f"\n--- IChOA Iteration {iteration + 1}/{max_iterations} ---")

        # Evaluate fitness for each chimp and identify leaders
        for i in range(num_chimps):
            current_chimp_position_normalized = chimps[i]
            current_hyperparams = map_to_search_space(current_chimp_position_normalized, search_space)

            try:
                model = build_model(current_hyperparams, n_timesteps, n_features)
                optimizer = Adam(learning_rate=current_hyperparams['learning_rate'])
                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                epochs=10
                batch_size=1

                # Train the model (verbose=0 to suppress output during optimization)
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

                # Evaluate the model and return the fitness score (e.g., validation accuracy)
                fitness = history.history['val_accuracy'][-1]

                # Update Alpha, Beta, Delta
                if fitness > alpha_score:
                    alpha_score = fitness
                    alph_chimp_pos = current_chimp_position_normalized.copy()
                # Using elif to ensure distinct leaders for Beta and Delta only if alpha is already set
                elif fitness > beta_score and fitness < alpha_score: # Ensure beta is worse than alpha but better than delta
                    beta_score = fitness
                    beta_chimp_pos = current_chimp_position_normalized.copy()
                elif fitness > delta_score and fitness < beta_score: # Ensure delta is worse than beta but better than omega (implicitly)
                    delta_score = fitness
                    delta_chimp_pos = current_chimp_position_normalized.copy()

                # Update overall best if this chimp is the best so far
                if fitness > best_overall_fitness:
                    best_overall_fitness = fitness
                    best_overall_hyperparameters = current_hyperparams.copy()

            except Exception as e:
                # print(f"  Error evaluating chimp {i+1}: {e}") # Optionally print errors
                fitness = -float('inf') # Assign a very low fitness on error

        print(f"  Alpha Score: {alpha_score:.4f}, Beta Score: {beta_score:.4f}, Delta Score: {delta_score:.4f}")

        # Update 'f' (chaotic map parameter for IChOA)
        f = 2 - iteration * (2 / max_iterations) # Linearly decreases from 2 to 0
        c = 2 * random.random() # Random value between 0 and 2 for 'c'

        # Update positions of chimps
        for i in range(num_chimps):
            for key in search_space_keys:
                r1 = random.random()
                # r2 is used in the original ChOA, but often simplified in implementation or combined into A
                # For this simplified IChOA, r2 might not be explicitly used in the A calculation, but good to have.
                # A = f * (2 * r1 - 1) is the common formulation for exploration/exploitation balance

                A = f * (2 * r1 - 1)

                # Calculate distance to Alpha, Beta, Delta (using their positions)
                D_alpha = abs(c * alph_chimp_pos[key] - chimps[i][key])
                X1 = alph_chimp_pos[key] - A * D_alpha

                D_beta = abs(c * beta_chimp_pos[key] - chimps[i][key])
                X2 = beta_chimp_pos[key] - A * D_beta

                D_delta = abs(c * delta_chimp_pos[key] - chimps[i][key])
                X3 = delta_chimp_pos[key] - A * D_delta

                # Final position update - average of the movements towards alpha, beta, delta
                chimps[i][key] = (X1 + X2 + X3) / 3

                # Ensure positions stay within [0, 1] bounds
                chimps[i][key] = np.clip(chimps[i][key], 0.0, 1.0)

    print(f"\nIChOA Optimization complete. Best overall validation accuracy: {best_overall_fitness:.4f}")
    print(f"Best hyperparameters found by IChOA: {best_overall_hyperparameters}")
    return best_overall_hyperparameters