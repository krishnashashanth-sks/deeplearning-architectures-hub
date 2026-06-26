# Neccesart libraries for implementation of SegNet
# pip install fiftyone pycocotools tensorflow_datasets
from model import build_segnet
from dataset import dataset
from utils import *
import matplotlib.pyplot as plt

input_shape = (224, 224, 3)
# Define number of classes for segmentation (e.g., background + 2 specific objects)
num_classes = 3

# Build the SegNet model
segnet_model = build_segnet(input_shape, num_classes)

segnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create the TensorFlow dataset for training
batch_size = 4 # You can adjust this
train_tf_dataset = create_tf_dataset(dataset, batch_size=batch_size,input_shape=input_shape,num_classes=num_classes)

segnet_model.fit(
    train_tf_dataset,
    epochs=50, # Reduced for demonstration purposes; you'll likely need more
    verbose=1
)

# Get one batch from the training dataset
for images, masks in train_tf_dataset.take(1):
    break # We only need one batch

# Take the first image and its corresponding ground truth mask from the batch
sample_image = images[0]
sample_ground_truth_mask = masks[0]

# Make a prediction with the trained model
# The model expects a batch dimension, so we add it to the sample_image
predicted_mask_raw = segnet_model.predict(tf.expand_dims(sample_image, axis=0))
predicted_mask = predicted_mask_raw[0] # Remove the batch dimension

# Convert one-hot encoded masks to class labels (integer values)
ground_truth_class_map = tf.argmax(sample_ground_truth_mask, axis=-1).numpy()
predicted_class_map = tf.argmax(predicted_mask, axis=-1).numpy()

print(f"Shape of original image: {sample_image.shape}")
print(f"Shape of ground truth class map: {ground_truth_class_map.shape}")
print(f"Shape of predicted class map: {predicted_class_map.shape}")

# Visualize the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(sample_image.numpy())
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
# Using 'viridis' colormap for distinct classes, ensure num_classes is available
plt.imshow(ground_truth_class_map, cmap='viridis', vmin=0, vmax=num_classes - 1)
plt.title('Ground Truth Mask')
plt.colorbar(label='Class Label')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(predicted_class_map, cmap='viridis', vmin=0, vmax=num_classes - 1)
plt.title('Predicted Mask')
plt.colorbar(label='Class Label')
plt.axis('off')

plt.tight_layout()
plt.show()