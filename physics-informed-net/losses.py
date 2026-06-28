import tensorflow as tf

def physics_loss(x, model):
    # Ensure x is a Tensor and track gradients
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y_pred = model(x)

    # Compute dy/dx
    dy_dx = tape.gradient(y_pred, x)
    del tape

    # Physics-informed residual: dy/dx + y_pred = 0
    return tf.square(dy_dx + y_pred)

def initial_condition_loss(x0, y0, model):
    # Predict y at x0
    y_pred_at_x0 = model(x0)
    # Loss is the squared difference from the true initial condition
    return tf.square(y_pred_at_x0 - y0)
