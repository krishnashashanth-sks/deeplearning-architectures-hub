import tensorflow as tf

# ---  Loss Functions ---
def infonce_loss(query_embeddings, positive_key_embeddings, negative_key_embeddings, temperature=0.1):
    query_embeddings = tf.math.l2_normalize(query_embeddings, axis=-1)
    positive_key_embeddings = tf.math.l2_normalize(positive_key_embeddings, axis=-1)
    negative_key_embeddings = tf.math.l2_normalize(negative_key_embeddings, axis=-1)

    pos_logits = tf.reduce_sum(query_embeddings * positive_key_embeddings, axis=-1)
    neg_logits = tf.einsum('bd,bnd->bn', query_embeddings, negative_key_embeddings)

    logits = tf.concat([tf.expand_dims(pos_logits, axis=1), neg_logits], axis=1)
    logits /= temperature

    labels = tf.zeros(tf.shape(logits)[0], dtype=tf.int32)

    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)


def masked_language_model_loss(predicted_logits, true_token_ids, mask_indices):
    masked_logits = tf.boolean_mask(predicted_logits, mask_indices)
    masked_true_token_ids = tf.boolean_mask(true_token_ids, mask_indices)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    loss = loss_object(masked_true_token_ids, masked_logits)

    return tf.reduce_mean(loss)


def masked_graph_reconstruction_loss(predicted_embeddings, true_embeddings, mask_indices):
    masked_predicted = tf.boolean_mask(predicted_embeddings, mask_indices)
    masked_true = tf.boolean_mask(true_embeddings, mask_indices)

    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_object(masked_true, masked_predicted)

    return tf.reduce_mean(loss)


def margin_based_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=1.0):
    pos_distance_squared = tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=-1)
    neg_distance_squared = tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings), axis=-1)
    loss = tf.maximum(0.0, margin + pos_distance_squared - neg_distance_squared)
    return tf.reduce_mean(loss)


def transe_triplet_loss(head_embeddings, relation_embeddings, positive_tail_embeddings, negative_tail_embeddings, margin=1.0):
    pos_score_squared = tf.reduce_sum(tf.square(head_embeddings + relation_embeddings - positive_tail_embeddings), axis=-1)
    neg_score_squared = tf.reduce_sum(tf.square(head_embeddings + relation_embeddings - negative_tail_embeddings), axis=-1)
    loss = tf.maximum(0.0, margin + pos_score_squared - neg_score_squared)
    return tf.reduce_mean(loss)
