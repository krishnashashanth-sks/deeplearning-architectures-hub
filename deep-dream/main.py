import tensorflow as tf
from model import build_custom_inception_model
from utils import deepdream_step,deprocess_image
from PIL import Image
from inference import predict_image_class
import PIL
from IPython.display import display

custom_inception_model = build_custom_inception_model()
custom_inception_model.summary()

custom_inception_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Choose target layers from your custom_inception_model
target_layers = [
    custom_inception_model.get_layer(name).name for name in [
        'conv2d_2',
        'concatenate',
        'concatenate_1'
    ]
]

# Define the number of iterations and learning rate
iterations = 50
learning_rate = 0.01

# You can start with a random noise image or a specific image
# For simplicity, let's start with random noise
img = tf.random.uniform(minval=0.0, maxval=255.0, shape=(1, 256, 256, 3), dtype=tf.float32)
img = tf.keras.applications.inception_v3.preprocess_input(img)

print("Starting DeepDream...")
for i in range(iterations):
    img, loss = deepdream_step(img, custom_inception_model, target_layers, learning_rate=learning_rate)
    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss.numpy():.2f}")
        # Optional: display intermediate images
        # display(Image(data=PIL.Image.fromarray(deprocess_image(img[0].numpy()))))

print("DeepDream finished. Displaying final image:")
final_img = deprocess_image(img[0].numpy())
display(PIL.Image.fromarray(final_img))

# Save the final image
PIL.Image.fromarray(final_img).save("deepdream_custom_inception.png")

# InceptionV3 technically requires 75x75 minimum.
# 128x128 is a good balance. 256x256 is usually overkill for MNIST.
TARGET_SIZE = (128, 128)
BATCH_SIZE = 32

# Load MNIST dataset
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

def preprocess_mnist_images(image, label):
    # Process one image at a time (called by the dataset pipeline)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, TARGET_SIZE)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label

# 2. USE TF.DATA PIPELINE (The "Magic" for RAM)
# This keeps only BATCH_SIZE images in RAM at any given time.
train_ds = tf.data.Dataset.from_tensor_slices((x_train_mnist, y_train_mnist))
train_ds = train_ds.shuffle(10000).map(preprocess_mnist_images).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test_mnist, y_test_mnist))
test_ds = test_ds.map(preprocess_mnist_images).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

history_mnist = custom_inception_model.fit(
    x=train_ds,
    epochs=5, # You can increase epochs for better training
    # batch_size is defined in the dataset pipeline now
    validation_data=test_ds # Use the test_ds for validation directly
)

results_mnist = custom_inception_model.evaluate(test_ds)
print("\nMNIST Test Results:")
print(f"Loss: {results_mnist[0]:.4f}")
print(f"Accuracy: {results_mnist[1]:.4f}")

# Take a sample image from the raw MNIST test data
# We'll use the 5th image (index 4) as an example
sample_image_index = 4
raw_sample_image = x_test_mnist[sample_image_index]
actual_label = y_test_mnist[sample_image_index]

# Save the raw MNIST image temporarily as a PNG file
# This simulates loading an external image file
sample_image_filename = 'sample_mnist_digit.png'
Image.fromarray(raw_sample_image).save(sample_image_filename)

print(f"Saved a sample image to '{sample_image_filename}' (Actual label: {actual_label})")

# Perform inference using the custom function
predicted_digit = predict_image_class(custom_inception_model, sample_image_filename)
print(f"The actual label was: {actual_label}")