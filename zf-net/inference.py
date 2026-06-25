import numpy as np
from PIL import Image
import tensorflow as tf

def predict_image(model, image_path, img_height, img_width, class_names=None):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = np.array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Add batch dimension and normalize

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    if class_names:
        predicted_class_name = class_names[predicted_class_index]
        print(f"Predicted class: {predicted_class_name} (Index: {predicted_class_index})")
    else:
        print(f"Predicted class index: {predicted_class_index}")
    print(f"Confidence: {confidence:.2f}")
    return predicted_class_index, confidence, predictions