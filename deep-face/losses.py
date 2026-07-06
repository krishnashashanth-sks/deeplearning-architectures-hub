import tensorflow as tf

def triplet_loss(y_true, y_pred, alpha=0.2):
    # A dummy loss function for model compilation, actual triplet loss 
    # would calculate distances between embeddings from y_pred.
    return tf.reduce_mean(y_pred * 0.0)