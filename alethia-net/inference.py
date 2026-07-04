import numpy as np

def infer_on_image(model, image, class_names):
    """
    Performs inference on a single image using the provided model.

    
    Args:
        model: The trained Keras model.
        image: A single image (numpy array, shape e.g., (32, 32, 3)).
        class_names: A list of class names.

    Returns:
        The predicted class name and the probability of that class.
    """
    # Preprocess the image: normalize and add batch dimension
    processed_image = image.astype('float32') / 255.0
    processed_image = np.expand_dims(processed_image, axis=0) # Add batch dimension

    # Get predictions
    predictions = model.predict(processed_image)

    # Get the predicted class and its probability
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]

    return predicted_class_name, predicted_probability
