import numpy as np
import tensorflow as tf

def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    x = x.reshape((256, 256, 3))
    # Undo InceptionV3 preprocessing
    x /= 2.0
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def random_shift(img, max_shift_pixels=20):
    h, w, c = img.shape
    h_shift = np.random.randint(-max_shift_pixels, max_shift_pixels + 1)
    w_shift = np.random.randint(-max_shift_pixels, max_shift_pixels + 1)

    img_shifted = np.roll(np.roll(img, h_shift, axis=0), w_shift, axis=1)
    return img_shifted, h_shift, w_shift

def undo_random_shift(img_shifted, h_shift, w_shift):
    img = np.roll(np.roll(img_shifted, -w_shift, axis=1), -h_shift, axis=0)
    return img


def apply_gaussian_blur(img, sigma=1.0):
    # Apply Gaussian blur for regularization
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=[sigma, sigma, 0]) # Blur each channel independently

def calculate_loss_and_gradients(img, model, target_layer_names):
  with tf.GradientTape() as tape:
    tape.watch(img)
    outputs = [model.get_layer(name).output for name in target_layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    activations = activation_model(img)
    loss = tf.zeros(shape=())
    if isinstance(activations, list):
      for act in activations:
        loss += tf.math.reduce_mean(act)
    else:
      loss += tf.math.reduce_mean(activations)
  grads = tape.gradient(loss, img)
  return loss, grads 

def deepdream_step(img, model, target_layer_names, learning_rate=0.01, max_shift_pixels=20, blur_sigma=1.0):
    # Random shift for translation invariance
    img_shifted, h_shift, w_shift = random_shift(img[0].numpy(), max_shift_pixels)
    shifted_img_tensor = tf.expand_dims(img_shifted, axis=0)

    loss, grads = calculate_loss_and_gradients(shifted_img_tensor, model, target_layer_names)

    # Normalize gradients
    grads /= tf.math.reduce_std(grads) + 1e-8

    # Apply gradient ascent
    img = img + grads * learning_rate

    # Undo the random shift
    img_np = deprocess_image(img[0].numpy())
    img_np = undo_random_shift(img_np, h_shift, w_shift)
    img = tf.expand_dims(preprocess_image_for_model(img_np), axis=0)

    # Apply Gaussian blur for regularization
    img_np = deprocess_image(img[0].numpy())
    img_np = apply_gaussian_blur(img_np, sigma=blur_sigma)
    img = tf.expand_dims(preprocess_image_for_model(img_np), axis=0)

    return img, loss

# Helper to preprocess image for the model (needed after deprocess and re-adding regularization)
def preprocess_image_for_model(img_np):
    img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)
    img_tensor = tf.keras.applications.inception_v3.preprocess_input(img_tensor)
    return img_tensor
