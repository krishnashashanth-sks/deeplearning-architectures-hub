import tensorflow as tf

def triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin=0.2):
    """
    Calculates the triplet loss.

    Args:
        anchor_embedding: Embedding vector for the anchor image.
        positive_embedding: Embedding vector for the positive image.
        negative_embedding: Embedding vector for the negative image.
        margin: A hyperparameter that specifies the minimum distance between
                the anchor-negative pair and the anchor-positive pair.

    Returns:
        The computed triplet loss.
    """
    # Calculate Euclidean distance between anchor and positive
    positive_distance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), axis=1)
    positive_distance = tf.sqrt(positive_distance)

    # Calculate Euclidean distance between anchor and negative
    negative_distance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), axis=1)
    negative_distance = tf.sqrt(negative_distance)

    # Compute the triplet loss
    loss = tf.maximum(0.0, positive_distance - negative_distance + margin)
    return loss
