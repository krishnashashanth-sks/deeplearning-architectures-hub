import tensorflow as tf
import cv2
import numpy as np
import random
from mtcnn.mtcnn import MTCNN # Assumes mtcnn is installed

# Instantiate MTCNN detector once
face_detector = MTCNN()

def detect_face(image_path):
    # Function to read image, detect face, and return bounding box and keypoints
    try:
        # MTCNN expects RGB images
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error reading/converting image {image_path}: {e}. Skipping.")
        return None, None

    detections = face_detector.detect_faces(img_rgb)
    if detections:
        # Return the first detected face (assuming one main face per image for training)
        return img_rgb, detections[0]
    return img_rgb, None # Return original image and None if no face detected

def align_face(image, detection_result, target_size=(160, 160)):
    if not detection_result or 'keypoints' not in detection_result:
        return None

    keypoints = detection_result['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Desired eye positions for target_size (e.g., 160x160)
    desired_left_eye = (0.3 * target_size[0], 0.4 * target_size[1])
    desired_right_eye = (0.7 * target_size[0], 0.4 * target_size[1])

    desired_eye_distance = np.sqrt((desired_right_eye[0] - desired_left_eye[0])**2 + (desired_right_eye[1] - desired_left_eye[1])**2)
    current_eye_distance = np.sqrt(dx**2 + dy**2)
    scale = desired_eye_distance / current_eye_distance

    eyes_center_x = (left_eye[0] + right_eye[0]) / 2
    eyes_center_y = (left_eye[1] + right_eye[1]) / 2
    center = (eyes_center_x, eyes_center_y)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    desired_center_x = (desired_left_eye[0] + desired_right_eye[0]) / 2
    desired_center_y = (desired_left_eye[1] + desired_right_eye[1]) / 2

    tx = desired_center_x - (M[0,0]*center[0] + M[0,1]*center[1] + M[0,2])
    ty = desired_center_y - (M[1,0]*center[0] + M[1,1]*center[1] + M[1,2])

    M[0, 2] += tx
    M[1, 2] += ty

    aligned_image = cv2.warpAffine(image, M, target_size, flags=cv2.INTER_LINEAR)
    return aligned_image

def augment_image(image, seed=None):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0

    # Random horizontal flipping
    augmented_image_tensor = tf.image.random_flip_left_right(image_tensor, seed=seed)

    # Random brightness adjustment
    augmented_image_tensor = tf.image.random_brightness(augmented_image_tensor, max_delta=0.2, seed=seed)

    # Random contrast adjustment
    augmented_image_tensor = tf.image.random_contrast(augmented_image_tensor, lower=0.8, upper=1.2, seed=seed)

    # Random rotation (90-degree increments)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=seed)
    augmented_image_tensor = tf.image.rot90(augmented_image_tensor, k=k)

    augmented_image_tensor = tf.clip_by_value(augmented_image_tensor, 0.0, 1.0)

    return (augmented_image_tensor * 255).numpy().astype(np.uint8)

print("Data augmentation function `augment_image` defined.")

# --- 5. Normalize and Resize Images Function ---

def preprocess_face_for_model(image, target_size=(152, 152)):
    image_float = tf.image.convert_image_dtype(image, dtype=tf.float32)
    resized_image = tf.image.resize(image_float, target_size, method=tf.image.ResizeMethod.BILINEAR)
    normalized_image = (resized_image * 2.0) - 1.0
    return normalized_image.numpy()

print("Normalization and resizing function `preprocess_face_for_model` defined.")

# --- 6. Triplet Batch Generation Function ---

def generate_triplet_batch_paths(dataframe, batch_size):
    triplets = []
    identities = dataframe['label'].unique()

    if len(identities) < 2:
        print("Not enough identities to form triplets.")
        return []

    for _ in range(batch_size):
        anchor_row = dataframe.sample(n=1).iloc[0]
        anchor_path = anchor_row['image_path']
        anchor_identity = anchor_row['label']

        positive_candidates = dataframe[dataframe['label'] == anchor_identity]['image_path'].tolist()
        # Ensure positive is different from anchor if possible
        if len(positive_candidates) > 1:
            positive_path = random.choice([p for p in positive_candidates if p != anchor_path])
        else:
            positive_path = positive_candidates[0] # Take itself if no other option

        negative_identities_candidates = [id for id in identities if id != anchor_identity]
        if not negative_identities_candidates:
            print("Not enough distinct identities to form negative pairs.")
            return []
        negative_identity = random.choice(negative_identities_candidates)
        negative_candidates = dataframe[dataframe['label'] == negative_identity]['image_path'].tolist()
        negative_path = random.choice(negative_candidates)

        triplets.append((anchor_path, positive_path, negative_path))

    return triplets

@tf.function
def load_and_preprocess_single_image(image_path_tensor, augment=True):
    # Convert tensor path to string
    image_path = image_path_tensor.numpy().decode('utf-8')

    # Use tf.py_function for non-TensorFlow ops like cv2 and MTCNN
    # This part must be wrapped as it contains non-TF operations
    def _py_function_body(path, do_augment):
        img_rgb, detection = detect_face(path)
        if detection is None:
            # If no face is detected, create a black image to avoid pipeline breaking
            # or handle more gracefully (e.g., skip or retry)
            processed_img = np.zeros((152, 152, 3), dtype=np.float32)
        else:
            aligned_img = align_face(img_rgb, detection)
            if aligned_img is None:
                processed_img = np.zeros((152, 152, 3), dtype=np.float32)
            else:
                # Convert aligned image from BGR (if cv2 returned BGR) to RGB for TF augmentation
                # The align_face function currently uses the input image as-is, which is RGB here.
                # So aligned_img is RGB.
                if do_augment:
                    # Augment_image expects uint8 RGB
                    augmented_img = augment_image(aligned_img.astype(np.uint8))
                else:
                    augmented_img = aligned_img.astype(np.uint8)
                processed_img = preprocess_face_for_model(augmented_img)
        return processed_img

    # Wrap the python function for tf.data.Dataset
    processed_image = tf.py_function(
        func=_py_function_body,
        inp=[image_path_tensor, augment],
        Tout=tf.float32
    )
    processed_image.set_shape((152, 152, 3)) # Set shape explicitly
    return processed_image

def create_deepface_dataset(dataframe, batch_size, augment=True, shuffle_buffer_size=1024):
    # Create a TF Dataset from image paths and labels for triplet generation
    # This generator yields triplet paths, which are then loaded and processed
    def triplet_generator():
        while True:
            triplet_paths = generate_triplet_batch_paths(dataframe, batch_size=1) # Generate one triplet at a time
            if triplet_paths:
                yield triplet_paths[0][0], triplet_paths[0][1], triplet_paths[0][2]
            else:
                # If not enough identities, yield dummy data to prevent pipeline from breaking
                dummy_path = b'dummy_data/person_00/img_000.jpg'
                yield dummy_path, dummy_path, dummy_path

    dataset = tf.data.Dataset.from_generator(
        triplet_generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )

    # Shuffle the triplet paths
    dataset = dataset.shuffle(shuffle_buffer_size)

    # Map the preprocessing function to each image path in the triplet
    dataset = dataset.map(
        lambda a, p, n: (
            load_and_preprocess_single_image(a, augment=augment),
            load_and_preprocess_single_image(p, augment=augment),
            load_and_preprocess_single_image(n, augment=augment)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch the preprocessed triplets
    dataset = dataset.batch(batch_size)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset