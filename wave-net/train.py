import tensorflow as tf

@tf.function
def train_step(model, inputs, targets,loss_fn,optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        # The targets should be integer labels for SparseCategoricalCrossentropy
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss