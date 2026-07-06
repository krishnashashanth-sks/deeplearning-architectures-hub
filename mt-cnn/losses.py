import tensorflow as tf

def face_classification_loss(y_true, y_pred):
    y_true_cls = tf.cast(y_true[:, 0], tf.float32)
    valid_mask = tf.where(tf.not_equal(y_true_cls, -1.0))
    y_true_filtered = tf.gather_nd(y_true[:, 1:], valid_mask)
    y_pred_filtered = tf.gather_nd(y_pred, valid_mask)
    loss_fn_cls = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn_cls(y_true_filtered, y_pred_filtered)
    return tf.cond(tf.greater(tf.shape(valid_mask)[0], 0), lambda: tf.reduce_mean(loss), lambda: 0.0)

def bbox_regression_loss(y_true, y_pred):
    y_true_cls = tf.cast(y_true[:, 0], tf.float32)
    valid_mask = tf.where(tf.not_equal(y_true_cls, 0.0))
    y_true_filtered = tf.gather_nd(y_true[:, 1:], valid_mask)
    y_pred_filtered = tf.gather_nd(y_pred, valid_mask)
    loss_fn_reg = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn_reg(y_true_filtered, y_pred_filtered)
    return tf.cond(tf.greater(tf.shape(valid_mask)[0], 0), lambda: tf.reduce_mean(loss), lambda: 0.0)

def rnet_face_classification_loss(y_true, y_pred):
    # y_true[:, :1] contains the actual raw labels (0 for non-face, 1 for face, -1 for partial/ignored)
    # y_pred contains the probabilities for non-face and face

    # Ensure y_true is float32 for calculations
    y_true_cls_raw = tf.cast(y_true[:, 0], tf.float32) # Extract raw label

    # Create a mask for valid classification samples (labels 0 and 1)
    valid_mask = tf.where(tf.not_equal(y_true_cls_raw, -1.0)) # Exclude partial samples (label -1)

    # Apply the mask to true labels (one-hot encoded) and predictions
    y_true_filtered = tf.gather_nd(y_true[:, 1:], valid_mask) # Actual one-hot labels for classification
    y_pred_filtered = tf.gather_nd(y_pred, valid_mask)

    # Calculate binary cross-entropy loss
    loss_fn_cls = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn_cls(y_true_filtered, y_pred_filtered)

    # If no valid samples, return 0 loss to avoid NaN
    return tf.cond(
        tf.greater(tf.shape(valid_mask)[0], 0),
        lambda: tf.reduce_mean(loss),
        lambda: 0.0
    )

def rnet_bbox_regression_loss(y_true, y_pred):
    # y_true[:, :1] contains the actual raw labels (0 for non-face, 1 for face, -1 for partial/ignored)
    # y_true[:, 1:] contains the actual bounding box targets
    # y_pred contains the predicted bounding box offsets

    # Ensure y_true is float32 for calculations
    y_true_cls_raw = tf.cast(y_true[:, 0], tf.float32) # Extract raw label

    # Create a mask for valid regression samples (labels 1 for face, -1 for partial)
    valid_mask = tf.where(tf.not_equal(y_true_cls_raw, 0.0)) # Exclude non-face samples (label 0)

    # Apply the mask to true labels (bbox targets) and predictions
    y_true_filtered = tf.gather_nd(y_true[:, 1:], valid_mask) # Actual bbox targets
    y_pred_filtered = tf.gather_nd(y_pred, valid_mask)

    # Calculate L2 loss (Mean Squared Error)
    loss_fn_reg = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn_reg(y_true_filtered, y_pred_filtered)

    # If no valid samples, return 0 loss to avoid NaN
    return tf.cond(
        tf.greater(tf.shape(valid_mask)[0], 0),
        lambda: tf.reduce_mean(loss),
        lambda: 0.0
    )

def onet_face_classification_loss(y_true, y_pred):
    # y_true[:, :1] contains the actual raw labels (0 for non-face, 1 for face, -1 for partial/ignored)
    # y_pred contains the probabilities for non-face and face

    # Ensure y_true is float32 for calculations
    y_true_cls_raw = tf.cast(y_true[:, 0], tf.float32) # Extract raw label

    # Create a mask for valid classification samples (labels 0 and 1)
    valid_mask = tf.where(tf.not_equal(y_true_cls_raw, -1.0)) # Exclude partial samples (label -1)

    # Apply the mask to true labels (one-hot encoded) and predictions
    y_true_filtered = tf.gather_nd(y_true[:, 1:], valid_mask) # Actual one-hot labels for classification
    y_pred_filtered = tf.gather_nd(y_pred, valid_mask)

    # Calculate binary cross-entropy loss
    loss_fn_cls = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn_cls(y_true_filtered, y_pred_filtered)

    # If no valid samples, return 0 loss to avoid NaN
    return tf.cond(
        tf.greater(tf.shape(valid_mask)[0], 0),
        lambda: tf.reduce_mean(loss),
        lambda: 0.0
    )

def onet_bbox_regression_loss(y_true, y_pred):
    # y_true[:, :1] contains the actual raw labels (0 for non-face, 1 for face, -1 for partial/ignored)
    # y_true[:, 1:] contains the actual bounding box targets
    # y_pred contains the predicted bounding box offsets

    # Ensure y_true is float32 for calculations
    y_true_cls_raw = tf.cast(y_true[:, 0], tf.float32) # Extract raw label

    # Create a mask for valid regression samples (labels 1 for face, -1 for partial)
    valid_mask = tf.where(tf.not_equal(y_true_cls_raw, 0.0)) # Exclude non-face samples (label 0)

    # Apply the mask to true labels (bbox targets) and predictions
    y_true_filtered = tf.gather_nd(y_true[:, 1:], valid_mask) # Actual bbox targets
    y_pred_filtered = tf.gather_nd(y_pred, valid_mask)

    # Calculate L2 loss (Mean Squared Error)
    loss_fn_reg = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn_reg(y_true_filtered, y_pred_filtered)

    # If no valid samples, return 0 loss to avoid NaN
    return tf.cond(
        tf.greater(tf.shape(valid_mask)[0], 0),
        lambda: tf.reduce_mean(loss),
        lambda: 0.0
    )

def onet_landmark_regression_loss(y_true, y_pred):
    # y_true[:, :1] contains the actual raw labels (0 for non-face, 1 for face, -1 for partial/ignored)
    # y_true[:, 1:] contains the actual landmark targets
    # y_pred contains the predicted landmark offsets

    # Ensure y_true is float32 for calculations
    y_true_cls_raw = tf.cast(y_true[:, 0], tf.float32) # Extract raw label

    # Create a mask for valid landmark regression samples (only positive labels, i.e., 1)
    valid_mask = tf.where(tf.equal(y_true_cls_raw, 1.0))

    # Apply the mask to true labels (landmark targets) and predictions
    y_true_filtered = tf.gather_nd(y_true[:, 1:], valid_mask) # Actual landmark targets
    y_pred_filtered = tf.gather_nd(y_pred, valid_mask)

    # Calculate L2 loss (Mean Squared Error)
    loss_fn_lm = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn_lm(y_true_filtered, y_pred_filtered)

    # If no valid samples, return 0 loss to avoid NaN
    return tf.cond(
        tf.greater(tf.shape(valid_mask)[0], 0),
        lambda: tf.reduce_mean(loss),
        lambda: 0.0
    )
