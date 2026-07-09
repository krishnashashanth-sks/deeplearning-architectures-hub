from tensorflow import keras

def get_loss_function(action_type='continuous'):
    if action_type == 'continuous':
        return keras.losses.MeanSquaredError()
    elif action_type == 'discrete':
        # Placeholder for discrete action loss, assuming one-hot encoding if CategoricalCrossentropy
        # or integer labels if SparseCategoricalCrossentropy
        return keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        raise ValueError("action_type must be 'continuous' or 'discrete'")
