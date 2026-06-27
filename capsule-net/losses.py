from tensorflow.keras import backend as K

# --- 1. Define the margin loss function ---
def margin_loss(y_true, y_pred):
    # y_true: one-hot encoded labels (batch_size, num_classes)
    # y_pred: digit_capsules_output (batch_size, num_classes, dim_digit_capsule)

    # Calculate the length of each digit capsule vector
    # (batch_size, num_classes)
    L = K.sqrt(K.sum(K.square(y_pred), axis=-1, keepdims=False))

    # Define margin parameters
    m_plus = 0.9
    m_minus = 0.1
    lambda_param = 0.5

    # Calculate positive margin loss: max(0, m_plus - L)^2
    # y_true here acts as an indicator for presence of class j
    loss_pos = K.square(K.maximum(0., m_plus - L))

    # Calculate negative margin loss: max(0, L - m_minus)^2
    # (1 - y_true) acts as an indicator for absence of class j
    loss_neg = K.square(K.maximum(0., L - m_minus))

    # Combine losses
    # Total margin loss for each sample: sum over all classes
    # The paper uses a sum over j, so we sum over the num_classes axis
    margin_loss = y_true * loss_pos + lambda_param * (1 - y_true) * loss_neg
    margin_loss = K.sum(margin_loss, axis=-1) # Sum over classes for each sample

    return margin_loss # Returns loss per batch element

# --- 2. Define the reconstruction loss function using MSE ---
def reconstruction_loss(y_true, y_pred):
    # y_true: original input images (batch_size, H, W, C)
    # y_pred: reconstructed images (batch_size, H, W, C)

    # Flatten the images for MSE calculation
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # Calculate Mean Squared Error
    recon_loss = K.mean(K.square(y_pred_flat - y_true_flat), axis=-1)

    return recon_loss # Returns loss per batch element
