from PIL import Image
import numpy as np
from IPython.display import display
import tensorflow as tf

def predict_image_class(model, image_path, target_size=(128, 128)):
    # Load and resize the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)

    # Expand dimensions to create a batch of 1 image
    img_batch = np.expand_dims(img_array, axis=0)

    # Preprocess the image for the model (InceptionV3 preprocessing)
    img_preprocessed = tf.keras.applications.inception_v3.preprocess_input(img_batch)

    # Make prediction
    predictions = model.predict(img_preprocessed)
    predicted_class = np.argmax(predictions[0])

    print(f"Predicted class for {image_path}: {predicted_class}")
    display(img)

    return predicted_class