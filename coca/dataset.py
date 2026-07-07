import tensorflow as tf
from utils import load_and_preprocess_image,preprocess_caption

# Create a tf.data.Dataset
def create_dataset(image_caption_pairs, tokenizer, max_caption_length,buffer_size,batch_size):
    image_paths = [pair[0] for pair in image_caption_pairs]
    captions = [pair[1] for pair in image_caption_pairs]

    image_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(image_paths))
    image_dataset = image_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    caption_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(captions))
    caption_dataset = caption_dataset.map(lambda x: tf.py_function(
        func=preprocess_caption, inp=[x,tokenizer,max_caption_length], Tout=tf.int32), # Tout=tf.int32 for sequence of token IDs
        num_parallel_calls=tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.zip((image_dataset, caption_dataset))
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
