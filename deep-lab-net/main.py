import tensorflow as tf
import shutil
from model import DeepLabV3Plus
from metric import CustomMeanIoU
from utils import create_dummy_file,load_image_mask,augment_data

IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_CLASSES = 21
BATCH_SIZE = 8
EPOCHS = 10

TRAIN_IMG_DIR = 'temp_train_images'
TRAIN_MASK_DIR = 'temp_train_masks'
VAL_IMG_DIR = 'temp_val_images'
VAL_MASK_DIR = 'temp_val_masks'

num_train_samples = 100
num_val_samples = 20

# Create dummy directories
for directory in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR]:
    os.makedirs(directory, exist_ok=True)

# Create dummy image and mask files for training
train_image_paths = []
train_mask_paths = []
for i in range(num_train_samples):
    img_path = os.path.join(TRAIN_IMG_DIR, f'image_{i}.jpg')
    mask_path = os.path.join(TRAIN_MASK_DIR, f'mask_{i}.png')
    create_dummy_file(img_path, channels=3, is_mask=False)
    create_dummy_file(mask_path, is_mask=True)
    train_image_paths.append(img_path)
    train_mask_paths.append(mask_path)

# Create dummy image and mask files for validation
val_image_paths = []
val_mask_paths = []
for i in range(num_val_samples):
    img_path = os.path.join(VAL_IMG_DIR, f'image_{i}.jpg')
    mask_path = os.path.join(VAL_MASK_DIR, f'mask_{i}.png')
    create_dummy_file(img_path, channels=3, is_mask=False)
    create_dummy_file(mask_path, is_mask=True)
    val_image_paths.append(img_path)
    val_mask_paths.append(mask_path)

# Create tf.data.Dataset objects
train_path_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
val_path_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

train_dataset = train_path_dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=num_train_samples)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_path_dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# Instantiate the DeepLabV3+ model
deep_lab_model = DeepLabV3Plus(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    num_classes=NUM_CLASSES,
    output_stride=16, # Or 8, depending on desired backbone features
    aspp_filters=256,
    aspp_dilation_rates=[6, 12, 18]
)

# Define optimizer, loss, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [CustomMeanIoU(num_classes=NUM_CLASSES)]

# Compile the model
deep_lab_model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
print("DeepLabV3+ model compiled successfully.")

# Start training
print("Starting model training...")
history = deep_lab_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    verbose=1
)
print("Model training completed.")

dummy_inference_input = tf.random.normal((1, IMG_HEIGHT, IMG_WIDTH, 3))
print(f"Dummy inference input shape: {dummy_inference_input.shape}")

inference_output = deep_lab_model.predict(dummy_inference_input)

print(f"Inference output shape: {inference_output.shape}")

# Cleanup dummy directories (optional, but good practice)
shutil.rmtree(TRAIN_IMG_DIR)
shutil.rmtree(TRAIN_MASK_DIR)
shutil.rmtree(VAL_IMG_DIR)
shutil.rmtree(VAL_MASK_DIR)
print("Dummy directories and files cleaned up.")
