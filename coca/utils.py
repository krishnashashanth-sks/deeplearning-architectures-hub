import tensorflow as tf

# Function to preprocess captions (tokenize, pad, add start/end tokens)
def preprocess_caption(caption_tensor,tokenizer,max_caption_length):
    # If the input is a TensorFlow string tensor, decode it
    if isinstance(caption_tensor, tf.Tensor):
        caption_string = caption_tensor.numpy().decode('utf-8')
    else: # Otherwise, assume it's already a Python string
        caption_string = caption_tensor

    # The captions already have <start> and <end> tokens, so we just tokenize and pad
    seq = tokenizer.texts_to_sequences([caption_string])[0]

    seq = tf.keras.preprocessing.sequence.pad_sequences(
        [seq], maxlen=max_caption_length, padding='post')[0] # pad to max_caption_length including start/end
    return seq

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    # Simulate image loading (create a dummy image tensor)
    # In a real scenario:
    # img = tf.io.read_file(image_path)
    # img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    # img = tf.keras.applications.inception_v3.preprocess_input(img)
    # return img

    # For dummy, just return a random tensor
    dummy_img = tf.random.uniform(shape=(IMG_SIZE, IMG_SIZE, 3), minval=0., maxval=1., dtype=tf.float32)
    return dummy_img

