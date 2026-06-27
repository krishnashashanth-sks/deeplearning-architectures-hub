import numpy as np

def evaluate_model(siamese_model, test_triplets):
    """
    Evaluates the Siamese model's ability to distinguish similar and dissimilar pairs.
    Args:
        siamese_model: The trained SiameseModel instance.
        test_triplets: A NumPy array of test triplets (anchor, positive, negative images).

    Returns:
        float: The accuracy (percentage of correctly identified triplets).
    """
    correct_predictions = 0
    total_predictions = 0

    # Extract the embedding model from the siamese_model for direct inference
    embedding_model = siamese_model.siamese_network.layers[0] # Assuming embedding_model is the first layer of siamese_network

    for i in range(test_triplets.shape[0]):
        anchor_image = np.expand_dims(test_triplets[i, 0], axis=0)
        positive_image = np.expand_dims(test_triplets[i, 1], axis=0)
        negative_image = np.expand_dims(test_triplets[i, 2], axis=0)

        anchor_embedding = embedding_model.predict(anchor_image, verbose=0)
        positive_embedding = embedding_model.predict(positive_image, verbose=0)
        negative_embedding = embedding_model.predict(negative_image, verbose=0)

        # Calculate Euclidean distances
        d_ap = np.sum(np.square(anchor_embedding - positive_embedding))
        d_an = np.sum(np.square(anchor_embedding - negative_embedding))

        if d_ap < d_an:
            correct_predictions += 1
        total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    return accuracy