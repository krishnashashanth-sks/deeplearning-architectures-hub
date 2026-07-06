from losses import face_classification_loss,bbox_regression_loss,rnet_face_classification_loss,rnet_bbox_regression_loss,onet_face_classification_loss,onet_bbox_regression_loss,onet_landmark_regression_loss 
import tensorflow as tf
from utils import hard_sample_mining_onet,hard_sample_mining_rnet
import numpy as np
from data_generators import pnet_data_generator

def train_pnet(pnet_model, annotations, epochs=1, batch_size=32, steps_per_epoch=100):
    print("\n--- Training P-Net ---")

    # Re-create the dataset each time to ensure generator starts fresh
    pnet_train_dataset = tf.data.Dataset.from_generator(
        lambda: pnet_data_generator(annotations, batch_size=batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 12, 12, 3), dtype=tf.float32),
            {
                'flatten_cls': tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                'flatten_reg': tf.TensorSpec(shape=(None, 5), dtype=tf.float32)
            }
        )
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    pnet_model.compile(
        optimizer=optimizer,
        loss={'flatten_cls': face_classification_loss, 'flatten_reg': bbox_regression_loss},
        loss_weights={'flatten_cls': 1.0, 'flatten_reg': 0.5}
    )

    # Train the P-Net
    history = pnet_model.fit(
        pnet_train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1
    )
    print("P-Net training complete.")
    return pnet_model, history

def train_rnet(rnet_model, pnet_model, annotations, epochs=1, batch_size=32):
    print("\n--- Training R-Net ---")

    # 1. Perform Hard Sample Mining for R-Net
    print("Generating R-Net hard samples via P-Net inference...")
    rnet_images, rnet_labels, rnet_bbox_targets = hard_sample_mining_rnet(
        annotations,
        pnet_model,
        rnet_input_size=(24, 24),
        pnet_cls_threshold=0.6,
        pnet_nms_threshold=0.5
    )
    print(f"R-Net hard sample mining collected {len(rnet_images)} samples.")

    if len(rnet_images) == 0:
        print("No R-Net samples collected, skipping training.")
        return rnet_model, None

    # Prepare R-Net data for training
    cls_true_targets = []
    for label in rnet_labels:
        if label == 0: # Non-face
            cls_true_targets.append(np.array([label, 1, 0], dtype=np.float32))
        elif label == 1: # Face
            cls_true_targets.append(np.array([label, 0, 1], dtype=np.float32))
        else: # Partial (label -1)
            cls_true_targets.append(np.array([label, 0, 0], dtype=np.float32))
    cls_true_targets = np.array(cls_true_targets)

    bbox_true_targets = np.hstack([rnet_labels[:, np.newaxis], rnet_bbox_targets])
    bbox_true_targets = bbox_true_targets.astype(np.float32)

    rnet_train_dataset = tf.data.Dataset.from_tensor_slices(
        (rnet_images, {'cls_output': cls_true_targets, 'bbox_output': bbox_true_targets})
    ).shuffle(buffer_size=len(rnet_images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Compile R-Net model (ensure `rnet_face_classification_loss` and `rnet_bbox_regression_loss` are accessible)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    rnet_model.compile(
        optimizer=optimizer,
        loss={'cls_output': rnet_face_classification_loss, 'bbox_output': rnet_bbox_regression_loss},
        loss_weights={'cls_output': 1.0, 'bbox_output': 0.5}
    )

    # Train the R-Net
    steps_per_epoch = max(1, len(rnet_images) // batch_size)
    history = rnet_model.fit(
        rnet_train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1
    )
    print("R-Net training complete.")
    return rnet_model, history

def train_onet(onet_model, pnet_model, rnet_model, annotations, epochs=1, batch_size=32):
    print("\n--- Training O-Net ---")

    # 1. Perform Hard Sample Mining for O-Net
    print("Generating O-Net hard samples via P-Net and R-Net inference...")
    onet_images, onet_labels, onet_bbox_targets, onet_landmark_targets = hard_sample_mining_onet(
        annotations,
        pnet_model,
        rnet_model,
        onet_input_size=(48, 48),
        pnet_cls_threshold=0.6,
        pnet_nms_threshold=0.5,
        rnet_cls_threshold=0.01, # Lenient to ensure samples pass
        rnet_nms_threshold=0.5  # Lenient to ensure samples pass
    )
    print(f"O-Net hard sample mining collected {len(onet_images)} samples.")

    if len(onet_images) == 0:
        print("No O-Net samples collected, skipping training.")
        return onet_model, None

    # Prepare O-Net data for training
    cls_true_targets = []
    for label in onet_labels:
        if label == 0: # Non-face
            cls_true_targets.append(np.array([label, 1, 0], dtype=np.float32))
        elif label == 1: # Face
            cls_true_targets.append(np.array([label, 0, 1], dtype=np.float32))
        else: # Partial (label -1)
            cls_true_targets.append(np.array([label, 0, 0], dtype=np.float32))
    cls_true_targets = np.array(cls_true_targets)

    bbox_true_targets = np.hstack([onet_labels[:, np.newaxis], onet_bbox_targets])
    bbox_true_targets = bbox_true_targets.astype(np.float32)

    landmark_true_targets = np.hstack([onet_labels[:, np.newaxis], onet_landmark_targets])
    landmark_true_targets = landmark_true_targets.astype(np.float32)

    onet_train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            onet_images,
            {
                'cls_output': cls_true_targets,
                'bbox_output': bbox_true_targets,
                'landmark_output': landmark_true_targets
            }
        )
    ).shuffle(buffer_size=len(onet_images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Compile O-Net model (ensure all three loss functions are accessible)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    onet_model.compile(
        optimizer=optimizer,
        loss={
            'cls_output': onet_face_classification_loss,
            'bbox_output': onet_bbox_regression_loss,
            'landmark_output': onet_landmark_regression_loss
        },
        loss_weights={
            'cls_output': 1.0,
            'bbox_output': 0.5,
            'landmark_output': 1.0
        }
    )

    # Train the O-Net
    steps_per_epoch = max(1, len(onet_images) // batch_size)
    history = onet_model.fit(
        onet_train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1
    )
    print("O-Net training complete.")
    return onet_model, history