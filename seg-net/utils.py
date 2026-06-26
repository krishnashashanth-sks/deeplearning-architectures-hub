import tensorflow as tf
from dataset import category_to_label

def process_coco_sample(sample,input_shape=(224, 224, 3),num_classes=3):
    # Load image
    img_path = sample.filepath
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Get original image dimensions for mask creation
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]

    # Create an empty mask for the current image. Initialize with background (0)
    # The mask will have integer labels [0, 1, 2, ...]
    mask = tf.zeros((height, width), dtype=tf.int32)

    # Iterate through ground_truth detections to create the segmentation mask
    # Note: FiftyOne stores masks as `Segmentation` objects within `Detection`
    if sample.ground_truth is not None and sample.ground_truth.detections is not None:
        # Sort detections by area (smaller objects first) to ensure larger objects aren't fully occluded
        # Or, process in a fixed order if certain classes should take precedence.
        # For simplicity here, we'll just draw them in the order they appear.
        for det in sample.ground_truth.detections:
            if det.mask is not None and det.label in category_to_label:
                class_id = category_to_label[det.label]

                # Get bounding box coordinates [x, y, width, height] in relative coordinates
                # Convert to absolute pixel coordinates
                x, y, w, h = det.bounding_box
                x_abs = tf.cast(x * tf.cast(width, tf.float32), tf.int32)
                y_abs = tf.cast(y * tf.cast(height, tf.float32), tf.int32)
                w_abs = tf.cast(w * tf.cast(width, tf.float32), tf.int32)
                h_abs = tf.cast(h * tf.cast(height, tf.float32), tf.int32)

                # Create a temporary full-size mask for the current detection
                det_mask_full_image = tf.zeros((height, width), dtype=tf.int32)

                # FiftyOne's det.mask is a boolean array relative to its bounding box
                # We need to place it correctly onto a full-sized mask
                object_mask = tf.cast(det.mask, tf.int32) * class_id

                # Pad the object_mask to match the full image size
                # This is a bit tricky with tf.pad if the mask doesn't align perfectly with the bbox
                # A more robust way is to slice and assign, but tf.tensor_scatter_nd_update
                # or tf.pad are alternatives. Let's try slicing first.

                # Ensure object_mask dimensions match the bbox dimensions after casting
                # If det.mask is (h_bbox, w_bbox), we need to ensure our target slice is also (h_bbox, w_bbox)
                # FiftyOne's det.mask has dimensions corresponding to the cropped mask

                # Create a tensor to hold the object_mask at its correct position
                # This is an alternative to scatter_nd_update which can be complex.
                # Let's rebuild the mask with tf.where to avoid shape issues directly.

                # Instead of tf.maximum, we will build the mask by assigning class_id to positive pixels
                # on the full-image mask. This approach avoids direct shape mismatch during tf.maximum.

                # Create a boolean mask of the same size as the full image, for the current object
                bbox_mask = tf.pad(object_mask,
                                   [[y_abs, height - (y_abs + tf.shape(object_mask)[0])],
                                    [x_abs, width - (x_abs + tf.shape(object_mask)[1])]],
                                   "CONSTANT", constant_values=0)

                # Now, bbox_mask has the shape of the original image, and only the object's pixels
                # within its original bounding box are non-zero (equal to class_id).
                # We can use tf.maximum to overlay this onto the main mask.
                # tf.maximum ensures that if objects overlap, the one with the higher class_id wins.
                mask = tf.maximum(mask, bbox_mask)

    # Resize image and mask to model's input shape
    img = tf.image.resize(img, (input_shape[0], input_shape[1]))
    # For masks, use 'nearest_neighbor' interpolation to preserve class labels
    mask = tf.image.resize(tf.expand_dims(mask, -1), (input_shape[0], input_shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(tf.squeeze(mask, -1), tf.int32) # Remove channel dimension and cast to int32

    # One-hot encode the mask
    # Ensure the mask values are within [0, num_classes-1]
    mask = tf.clip_by_value(mask, 0, num_classes - 1)
    mask_one_hot = tf.one_hot(mask, depth=num_classes)

    return img, mask_one_hot

def create_tf_dataset(fiftyone_dataset, batch_size=4, shuffle=True,input_shape=(224, 224, 3),num_classes=3):
    # Get a list of FiftyOne samples
    samples_list = list(fiftyone_dataset)

    # Create a tf.data.Dataset from this list
    # We'll use tf.py_function to wrap the Python `process_coco_sample` function
    # so it can be used within a TensorFlow graph.
    output_signature = (
        tf.TensorSpec(shape=(input_shape[0], input_shape[1], input_shape[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(input_shape[0], input_shape[1], num_classes), dtype=tf.float32)
    )

    tf_dataset = tf.data.Dataset.from_generator(
        lambda: (process_coco_sample(s) for s in samples_list),
        output_signature=output_signature
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(samples_list) * 2)

    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset