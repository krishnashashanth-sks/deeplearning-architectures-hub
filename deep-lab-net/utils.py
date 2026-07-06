import numpy as np
from PIL import Image

def create_dummy_file(filepath, size=(IMG_WIDTH, IMG_HEIGHT), channels=3, is_mask=False):
    if is_mask:
        dummy_data = np.random.randint(0, NUM_CLASSES, size, dtype=np.uint8)
        img = Image.fromarray(dummy_data, mode='L')
        img.save(filepath, format='PNG')
    else:
        dummy_data = np.random.randint(0, 256, (*size, channels), dtype=np.uint8)
        img = Image.fromarray(dummy_data)
        img.save(filepath, format='JPEG')  
        
# Redefine load_image_mask to handle PNGs for masks and clip values
def load_image_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_jpeg(image, channels=3)
    mask = tf.image.decode_png(mask, channels=1)

    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    mask = tf.image.resize(mask, (IMG_HEIGHT, IMG_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.int32)
    mask = tf.clip_by_value(mask, 0, NUM_CLASSES - 1)

    return image, mask

def augment_data(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask