import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from model import build_deepface_model,create_deepface_dataset
from losses import triplet_loss

# Instantiate the model
deepface_model = build_deepface_model()
print("DeepFace model built and summarized:")
deepface_model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
deepface_model.compile(optimizer=optimizer, loss=triplet_loss)
print("Model compiled with Adam optimizer and a placeholder triplet loss.")



# Create a dummy data directory and images for full pipeline demonstration
dummy_base_dir = "dummy_deepface_data"
os.makedirs(dummy_base_dir, exist_ok=True)

num_identities = 5
images_per_identity = 3
demo_image_paths = []
demo_labels = []

for i in range(num_identities):
    identity_label = f"person_{i:02d}"
    identity_dir = os.path.join(dummy_base_dir, identity_label)
    os.makedirs(identity_dir, exist_ok=True)
    for j in range(images_per_identity):
        # Create a simple colored square image
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        color = [int(i*50 % 255), int(j*80 % 255), 100]
        cv2.rectangle(dummy_img, (50, 50), (150, 150), color, -1)
        # Add 'eyes' for detection and alignment demo (adjust positions for variety)
        cv2.circle(dummy_img, (75, 80), 8, (255, 255, 255), -1) # Left eye
        cv2.circle(dummy_img, (125, 85), 8, (255, 255, 255), -1) # Right eye (slightly angled)
        
        dummy_path = os.path.join(identity_dir, f"img_{j:03d}.jpg")
        cv2.imwrite(dummy_path, dummy_img)
        demo_image_paths.append(dummy_path)
        demo_labels.append(identity_label)

demo_df = pd.DataFrame({'image_path': demo_image_paths, 'label': demo_labels})

print(f"Generated {len(demo_df)} dummy images for demonstration.")
print("First 5 dummy image paths:", demo_df['image_path'].tolist()[:5])

# Create the tf.data.Dataset
demo_batch_size = 2 # Small batch size for demonstration
dataset = create_deepface_dataset(demo_df, batch_size=demo_batch_size, augment=True)

# Iterate through one batch to demonstrate functionality
print("\n--- Demonstrating one batch from the DeepFace tf.data.Dataset ---")
for (anchors, positives, negatives) in dataset.take(1):
    print(f"Anchor batch shape: {anchors.shape}, dtype: {anchors.dtype}")
    print(f"Positive batch shape: {positives.shape}, dtype: {positives.dtype}")
    print(f"Negative batch shape: {negatives.shape}, dtype: {negatives.dtype}")

    # Display first triplet in the batch
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow((anchors[0].numpy() + 1) / 2) # Denormalize for display [-1, 1] -> [0, 1]
    plt.title('Anchor')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow((positives[0].numpy() + 1) / 2)
    plt.title('Positive')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow((negatives[0].numpy() + 1) / 2)
    plt.title('Negative')
    plt.axis('off')
    plt.show()
    break

print("Full DeepFace implementation consolidated and demonstrated in a single cell.")

