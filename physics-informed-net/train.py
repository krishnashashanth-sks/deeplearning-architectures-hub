import tensorflow as tf

def train_step(x_collocation, x_initial, y_initial, model, optimizer,physics_loss,initial_condition_loss):
    with tf.GradientTape() as tape:
        # Compute physics loss
        p_loss = tf.reduce_mean(physics_loss(x_collocation, model))

        # Compute initial condition loss
        ic_loss = initial_condition_loss(x_initial, y_initial, model)

        # Total loss
        total_loss = p_loss + ic_loss

    # Compute gradients and apply updates
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, p_loss, ic_loss