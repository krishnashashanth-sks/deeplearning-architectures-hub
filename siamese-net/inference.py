import numpy as np

def inference_function(image,embedding_model):
    """
    Generates an embedding for a given image using the trained embedding_model.
    Args:
        image (np.array): A single input image (e.g., shape (28, 28, 1)).

    Returns:
        np.array: The embedding vector for the input image.
    """
    # Ensure the image has a batch dimension if it doesn't already
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0) # Add batch dimension
    embedding = embedding_model.predict(image)
    return embedding