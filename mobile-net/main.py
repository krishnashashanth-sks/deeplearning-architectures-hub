import numpy as np
import matplotlib.pyplot as plt
from model import MobileNetV2
from dataset import ds_train,ds_test,ds_info
from utils import preprocess_image
import tensorflow as tf

model = MobileNetV2(input_shape=(224, 224, 3), num_classes=1000, alpha=1.0)
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Apply preprocessing to training and test datasets
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

EPOCHS = 10 # You can adjust the number of epochs as needed
history = model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test
)

loss, accuracy = model.evaluate(ds_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# Get a batch of test images and labels
for images, labels in ds_test.take(1):
    predictions = model.predict(images)

    num_samples_to_show = 5
    plt.figure(figsize=(15, 8))
    for i in range(num_samples_to_show):
        plt.subplot(1, num_samples_to_show, i + 1)

        # Un-normalize image for display (assuming it was normalized to [-1, 1])
        display_image = (images[i].numpy() + 1) / 2.0

        plt.imshow(display_image)

        predicted_label = np.argmax(predictions[i])
        true_label = labels[i].numpy()

        # Get class names from ds_info if available
        if 'label' in ds_info.features and hasattr(ds_info.features['label'], 'names'):
            class_names = ds_info.features['label'].names
            title_str = f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}"
        else:
            title_str = f"True: {true_label}\nPred: {predicted_label}"

        color = "green" if predicted_label == true_label else "red"
        plt.title(title_str, color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()